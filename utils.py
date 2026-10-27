import json
import random

import numpy as np
import torch
import yaml

HR_KS = (1, 5, 10, 20)
NDCG_KS = (5, 10, 20)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_metrics(rank):
    r = rank.float()
    out = {"count": int(r.numel())}
    for k in HR_KS:
        out[f"HR@{k}"] = float(((r >= 1) & (r <= k)).float().mean())
    for k in NDCG_KS:
        out[f"NDCG@{k}"] = float(torch.where(r <= k, 1.0 / torch.log2(r + 1.0), torch.zeros_like(r)).mean())
    return out


def group_metrics(rank, freq):
    groups = {"overall": torch.ones_like(freq, dtype=torch.bool), "tail": freq <= 1, "unseen": freq == 0}
    return {name: rank_metrics(rank[mask]) for name, mask in groups.items()}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def show(name, metrics):
    print(name + " | " + " | ".join(f"{g} HR@1 {m['HR@1']:.4f} HR@20 {m['HR@20']:.4f}" for g, m in metrics.items()), flush=True)
