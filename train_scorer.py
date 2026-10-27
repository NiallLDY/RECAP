import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data import CheckinData
from model import Backbone, RevisitScorer, final_scores, revisit_bonus
from train_backbone import core_logits
from utils import group_metrics, load_config, save_json, set_seed, show

SCORER_STATS = [4, 3, 5, 1, 2]


def source_share(direct):
    count = np.clip(np.expm1(direct), 0.0, None)
    return count / max(1.0, float(count.sum()))


class RevisitSet:
    def __init__(self, split, mean, std):
        self.offsets = split.offsets
        self.cand = split.cand
        z = ((split.stats - mean) / std)[:, SCORER_STATS]
        share = np.zeros(split.stats.shape[0], dtype=np.float32)
        for i in range(len(split)):
            lo, hi = self.offsets[i], self.offsets[i + 1]
            if hi > lo:
                share[lo:hi] = (source_share(split.stats[lo:hi, 0]) - 0.25) * 4.0
        self.x = np.concatenate([z, share[:, None]], axis=1)
        self.width = max(1, int(np.diff(self.offsets).max()))
        self.label = split.label
        self.n = len(split)
        length = np.diff(self.offsets)
        rows = np.repeat(np.arange(self.n), length)
        hit = self.cand == split.label.numpy()[rows]
        hit_rows = torch.from_numpy(rows[hit])
        self.has_target = torch.zeros(self.n, dtype=torch.bool)
        self.has_target[hit_rows] = True
        self.target_pos = torch.zeros(self.n, dtype=torch.long)
        self.target_pos[hit_rows] = torch.from_numpy((np.arange(length.sum()) - self.offsets[:-1][rows])[hit])

    def gather(self, idx):
        start = self.offsets[idx]
        length = self.offsets[idx + 1] - start
        rows = np.repeat(np.arange(len(idx)), length)
        cols = np.arange(length.sum()) - np.repeat(np.cumsum(length) - length, length)
        flat = np.repeat(start, length) + cols
        ids = torch.zeros(len(idx), self.width, dtype=torch.long)
        x = torch.zeros(len(idx), self.width, self.x.shape[1])
        mask = torch.zeros(len(idx), self.width, dtype=torch.bool)
        ids[rows, cols] = torch.from_numpy(self.cand[flat])
        x[rows, cols] = torch.from_numpy(self.x[flat])
        mask[rows, cols] = True
        return ids, x, mask


@torch.no_grad()
def scorer_log_probs(scorer, rs, device, batch_size=4096):
    out = []
    for i in range(0, rs.n, batch_size):
        _, x, mask = rs.gather(np.arange(i, min(i + batch_size, rs.n)))
        s = scorer(x.to(device), mask.to(device))
        out.append((s - torch.logsumexp(s, 1, keepdim=True)).cpu())
    return torch.cat(out)


def scorer_hr1(log_prob, rs):
    target = log_prob.gather(1, rs.target_pos[:, None]).squeeze(1)
    hit = rs.has_target & torch.isfinite(target) & ((log_prob > target[:, None]).sum(1) == 0)
    return float(hit.float().mean())


@torch.no_grad()
def final_ranks(backbone, split, rs, log_prob, lam, device):
    ranks = []
    for idx, logits in core_logits(backbone, split, device):
        ids, _, mask = rs.gather(idx.numpy())
        mask = mask.to(device)
        scores = final_scores(logits, ids.to(device), revisit_bonus(log_prob[idx].to(device), mask), lam)
        target = scores.gather(1, split.label[idx].to(device)[:, None])
        ranks.append((1 + (scores > target).sum(1)).cpu())
    return torch.cat(ranks)


def load_backbone(data, cfg, path, device):
    backbone = Backbone(data, cfg["backbone"])
    state = torch.load(path, map_location="cpu")
    backbone.load_state_dict(state["model"])
    backbone.graph_scale = state["graph_scale"]
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def feature_stats(split):
    mean = split.stats.mean(axis=0).astype(np.float32)
    std = split.stats.std(axis=0).astype(np.float32)
    return mean, np.where(std < 1e-4, 1.0, std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["scorer"]
    device = torch.device(args.device)
    run = Path(args.run)
    set_seed(args.seed)
    data = CheckinData(cfg["data"]["dir"], cfg["data"]["window"], cfg["data"]["history_cap"])
    backbone = load_backbone(data, cfg, run / "backbone.pt", device)
    mean, std = feature_stats(data.splits["train"])
    sets = {name: RevisitSet(data.splits[name], mean, std) for name in ("train", "val", "test")}
    scorer = RevisitScorer(sets["train"].x.shape[1], scfg["hidden"]).to(device)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"])
    train = sets["train"]
    best = (-1.0, None, -1)
    for epoch in range(scfg["epochs"]):
        start = time.time()
        scorer.train()
        total, n = 0.0, 0
        for idx in torch.randperm(train.n).split(scfg["batch_size"]):
            _, x, mask = train.gather(idx.numpy())
            s = scorer(x.to(device), mask.to(device))
            target = s.gather(1, train.target_pos[idx].to(device)[:, None]).squeeze(1)
            ok = train.has_target[idx].to(device) & torch.isfinite(target)
            if not bool(ok.any()):
                continue
            loss = (torch.logsumexp(s, 1) - target)[ok].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        scorer.eval()
        val_hr1 = scorer_hr1(scorer_log_probs(scorer, sets["val"], device), sets["val"])
        print(f"epoch {epoch} loss {total / max(1, n):.4f} val scorer HR@1 {val_hr1:.4f} time {time.time() - start:.0f}s", flush=True)
        if val_hr1 > best[0]:
            best = (val_hr1, {k: v.detach().clone() for k, v in scorer.state_dict().items()}, epoch)
    scorer.load_state_dict(best[1])
    scorer.eval()
    torch.save({"model": best[1], "epoch": best[2], "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
               run / "scorer.pt")
    lam = float(scfg["lam"])
    result = {"epoch": best[2], "lambda": lam}
    for name in ("val", "test"):
        split, rs = data.splits[name], sets[name]
        ranks = final_ranks(backbone, split, rs, scorer_log_probs(scorer, rs, device), lam, device)
        result[name] = group_metrics(ranks, split.freq)
        show(name, result[name])
    save_json(run / "metrics.json", result)


if __name__ == "__main__":
    main()
