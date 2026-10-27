from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ORIGIN = pd.Timestamp("2010-01-01", tz="UTC")
NUM_STATS = 6


def read_split(path):
    df = pd.read_csv(path)
    df["user"] = df["user_id"].astype(str)
    df["poi"] = df["POI_id"].astype(str)
    df["traj"] = df["trajectory_id"].astype(str)
    df["hour"] = (pd.to_datetime(df["local_time"], utc=True) - ORIGIN).dt.total_seconds() / 3600.0
    return df


class UserHistory:
    def __init__(self, num_pois, poi_cat):
        self.poi_cat = poi_cat
        self.visited = []
        self.count = np.zeros(num_pois, dtype=np.int32)
        self.last_step = np.zeros(num_pois, dtype=np.int64)
        self.last_hour = np.zeros(num_pois, dtype=np.float64)
        self.first_hour = np.zeros(num_pois, dtype=np.float64)
        self.step = 0
        self.moves = {}
        self.cat_moves = {}
        self.last_poi = None
        self.last_traj = None

    def observe(self, poi, traj, hour):
        if self.count[poi] == 0:
            self.visited.append(poi)
            self.first_hour[poi] = hour
        self.count[poi] += 1
        self.last_step[poi] = self.step
        self.last_hour[poi] = hour
        self.step += 1
        if self.last_poi is not None and traj == self.last_traj:
            row = self.moves.setdefault(self.last_poi, {})
            row[poi] = row.get(poi, 0) + 1
            row = self.cat_moves.setdefault(int(self.poi_cat[self.last_poi]), {})
            row[poi] = row.get(poi, 0) + 1
        self.last_poi = poi
        self.last_traj = traj

    def snapshot(self, src, hour, lat, lon, cap):
        if not self.visited:
            return np.zeros(0, dtype=np.int64), np.zeros((0, NUM_STATS), dtype=np.float32)
        cand = np.asarray(self.visited, dtype=np.int64)
        if cand.shape[0] > cap:
            keep = np.argsort(self.last_step[cand], kind="stable")[::-1][:cap]
            cand = cand[np.sort(keep)]
        n = cand.shape[0]
        recency = (float(hour) - self.last_hour[cand]).astype(np.float32)
        np.maximum(recency, 0.0, out=recency)
        moves = self.moves.get(src, {})
        cat_moves = self.cat_moves.get(int(self.poi_cat[src]), {})
        direct = np.fromiter((moves.get(int(c), 0) for c in cand), dtype=np.float32, count=n)
        same_cat = np.fromiter((cat_moves.get(int(c), 0) for c in cand), dtype=np.float32, count=n)
        dist = np.sqrt((lat[cand] - lat[src]) ** 2 + (lon[cand] - lon[src]) ** 2).astype(np.float32)
        span = (self.last_hour[cand] - self.first_hour[cand]).astype(np.float32) / 24.0
        stats = np.zeros((n, NUM_STATS), dtype=np.float32)
        stats[:, 0] = np.log1p(direct)
        stats[:, 1] = np.log1p(dist * 100.0)
        stats[:, 2] = np.log1p(same_cat)
        stats[:, 3] = np.log1p(recency)
        stats[:, 4] = self.count[cand].astype(np.float32) / max(float(self.step), 1.0)
        stats[:, 5] = np.log1p(np.maximum(span, 0.0))
        return cand, stats


class Split:
    def __init__(self, user, window, pad, src, label, offsets=None, cand=None, stats=None):
        self.user = torch.tensor(user, dtype=torch.long)
        self.window = torch.tensor(window, dtype=torch.long)
        self.pad = torch.tensor(pad, dtype=torch.bool)
        self.src = torch.tensor(src, dtype=torch.long)
        self.label = torch.tensor(label, dtype=torch.long)
        self.offsets = offsets
        self.cand = cand
        self.stats = stats
        self.freq = None

    def __len__(self):
        return int(self.label.shape[0])


class CheckinData:
    def __init__(self, root, window, history_cap=None):
        root = Path(root)
        frames = {name: read_split(root / f"{name}.csv") for name in ("train", "val", "test")}
        train = frames["train"]
        self.poi_index = {p: i for i, p in enumerate(sorted(train["poi"].unique()))}
        self.user_index = {u: i for i, u in enumerate(sorted(train["user"].unique()))}
        cat_index = {c: i for i, c in enumerate(sorted(int(c) for c in train["POI_catid_code"].unique()))}
        self.num_pois = len(self.poi_index)
        self.num_users = len(self.user_index)
        self.num_cats = len(cat_index)
        meta = train.sort_values(["user_id", "trajectory_id", "local_time"], kind="mergesort")
        meta = meta.assign(idx=meta["poi"].map(self.poi_index)).groupby("idx").agg(
            {"POI_catid_code": "first", "latitude": "first", "longitude": "first"}).sort_index()
        cats = meta["POI_catid_code"].map(lambda c: cat_index[int(c)]).to_numpy(dtype=np.int64)
        latlon = meta[["latitude", "longitude"]].to_numpy(dtype=np.float32)
        self.poi_cat = torch.from_numpy(cats)
        self.poi_latlon = torch.from_numpy(latlon)
        self.window = window
        self.history_cap = history_cap
        histories = {} if history_cap else None
        self.splits = {name: self._build(frames[name], name != "train", histories, cats, latlon)
                       for name in ("train", "val", "test")}
        train_split = self.splits["train"]
        self.transitions = Counter(zip(train_split.src.tolist(), train_split.label.tolist()))
        for split in self.splits.values():
            split.freq = torch.tensor([self.transitions.get(pair, 0) for pair in zip(split.src.tolist(), split.label.tolist())])
        self.adjacency = self._adjacency()

    def _build(self, df, known_only, histories, cats, latlon):
        if known_only:
            df = df[df["poi"].isin(self.poi_index)]
        df = df.sort_values(["user", "local_time"], kind="mergesort")
        k = self.window
        user, window, pad, src, label = [], [], [], [], []
        offsets, cand, stats = [0], [], []
        for name, group in df.groupby("user", sort=False):
            if name not in self.user_index:
                continue
            pois = group["poi"].map(self.poi_index).astype("int64").tolist()
            trajs = group["traj"].tolist()
            hours = group["hour"].astype("float64").tolist()
            history = None
            if histories is not None:
                history = histories.setdefault(name, UserHistory(self.num_pois, cats))
            for pos, poi in enumerate(pois):
                if pos > 0 and trajs[pos] == trajs[pos - 1]:
                    seq = pois[max(0, pos - k):pos]
                    user.append(self.user_index[name])
                    window.append([0] * (k - len(seq)) + seq)
                    pad.append([True] * (k - len(seq)) + [False] * len(seq))
                    src.append(pois[pos - 1])
                    label.append(poi)
                    if history is not None:
                        ids, values = history.snapshot(pois[pos - 1], hours[pos - 1], latlon[:, 0], latlon[:, 1], self.history_cap)
                        cand.append(ids)
                        stats.append(values)
                        offsets.append(offsets[-1] + ids.shape[0])
                if history is not None:
                    history.observe(poi, trajs[pos], hours[pos])
        if histories is None:
            return Split(user, window, pad, src, label)
        return Split(user, window, pad, src, label, np.asarray(offsets, dtype=np.int64),
                     np.concatenate(cand), np.concatenate(stats))

    def _adjacency(self):
        out = Counter()
        for (s, _), c in self.transitions.items():
            out[s] += c
        rows, cols, values = [], [], []
        for (s, d), c in self.transitions.items():
            rows.append(s)
            cols.append(d)
            values.append(float(c) / float(out[s]))
        return torch.sparse_coo_tensor(torch.tensor([rows, cols], dtype=torch.long), torch.tensor(values, dtype=torch.float32),
                                       (self.num_pois, self.num_pois)).coalesce()
