import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import CheckinData
from model import Backbone
from utils import group_metrics, load_config, save_json, set_seed, show


def graph_scale(cfg, epoch):
    start, ramp = cfg["graph_start_epoch"], cfg["graph_ramp_epochs"]
    return 0.0 if epoch < start else min(1.0, (epoch - start + 1) / ramp)


def batch_of(split, idx, device):
    return split.user[idx].to(device), split.window[idx].to(device), split.pad[idx].to(device)


@torch.no_grad()
def core_logits(model, split, device, batch_size=512):
    for i in range(0, len(split), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(split)))
        yield idx, model(*batch_of(split, idx, device)).float()


@torch.no_grad()
def target_ranks(model, split, device):
    model.eval()
    ranks = []
    for idx, logits in core_logits(model, split, device):
        target = logits.gather(1, split.label[idx].to(device)[:, None])
        ranks.append((1 + (logits > target).sum(1)).cpu())
    return torch.cat(ranks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["backbone"]
    device = torch.device(args.device)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    data = CheckinData(cfg["data"]["dir"], cfg["data"]["window"])
    train, val, test = data.splits["train"], data.splits["val"], data.splits["test"]
    print(f"pois {data.num_pois} users {data.num_users} samples train {len(train)} val {len(val)} test {len(test)}", flush=True)
    model = Backbone(data, bcfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=bcfg["lr"], weight_decay=bcfg["weight_decay"])
    best = (-1.0, None, -1, 0.0)
    for epoch in range(bcfg["epochs"]):
        start = time.time()
        model.graph_scale = graph_scale(bcfg, epoch)
        model.train()
        total, n = 0.0, 0
        for idx in torch.randperm(len(train)).split(bcfg["batch_size"]):
            loss = F.cross_entropy(model(*batch_of(train, idx, device)), train.label[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            n += len(idx)
        val_hr1 = float((target_ranks(model, val, device) == 1).float().mean())
        print(f"epoch {epoch} loss {total / n:.4f} val HR@1 {val_hr1:.4f} time {time.time() - start:.0f}s", flush=True)
        if val_hr1 > best[0]:
            best = (val_hr1, copy.deepcopy(model.state_dict()), epoch, model.graph_scale)
    model.load_state_dict(best[1])
    model.graph_scale = best[3]
    torch.save({"model": best[1], "graph_scale": best[3], "epoch": best[2]}, run / "backbone.pt")
    metrics = group_metrics(target_ranks(model, test, device), test.freq)
    show(f"best epoch {best[2]} test", metrics)
    save_json(run / "backbone_metrics.json", {"epoch": best[2], "val_HR@1": best[0], "test": metrics})


if __name__ == "__main__":
    main()
