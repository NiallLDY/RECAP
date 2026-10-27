import argparse
from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "user_id",
    "POI_id",
    "POI_catid",
    "POI_catname",
    "latitude",
    "longitude",
    "timezone",
    "UTC_time",
]

OUTPUT_COLUMNS = [
    "user_id",
    "POI_id",
    "POI_catid",
    "POI_catid_code",
    "POI_catname",
    "latitude",
    "longitude",
    "timezone",
    "UTC_time",
    "local_time",
    "day_of_week",
    "norm_in_day_time",
    "trajectory_id",
]

SPLIT_RATIOS = (0.8, 0.1, 0.1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["NYC", "TKY", "CA"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--min-poi-checkins", type=int, default=0)
    parser.add_argument("--trajectory-gap-hours", type=float, default=48.0)
    parser.add_argument("--min-trajectory-length", type=int, default=2)
    return parser.parse_args()


def load_foursquare(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        dtype={
            "user_id": int,
            "POI_id": str,
            "POI_catid": str,
            "POI_catname": str,
            "latitude": float,
            "longitude": float,
            "timezone": int,
            "UTC_time": str,
        },
        encoding="latin1",
    )
    df["UTC_time"] = pd.to_datetime(df["UTC_time"], utc=True, format="%a %b %d %H:%M:%S %z %Y")
    df["local_time"] = df["UTC_time"] + pd.to_timedelta(df["timezone"], unit="m")
    df = df.sort_values(["user_id", "local_time"]).reset_index(drop=True)
    df["day_of_week"] = df["local_time"].dt.dayofweek
    seconds_in_day = (
        df["local_time"].dt.hour * 3600
        + df["local_time"].dt.minute * 60
        + df["local_time"].dt.second
    )
    df["norm_in_day_time"] = seconds_in_day / (24 * 3600)
    catid_codes = {catid: idx for idx, catid in enumerate(sorted(df["POI_catid"].unique()))}
    df["POI_catid_code"] = df["POI_catid"].map(catid_codes).astype(int)
    return df


def load_split_files(split_dir):
    parts = [pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "val", "test")]
    df = pd.concat(parts, ignore_index=True)
    df["UTC_time"] = pd.to_datetime(df["UTC_time"], utc=True)
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    df = df.drop(columns=["trajectory_id"])
    df = df.sort_values(["user_id", "local_time"], kind="mergesort").reset_index(drop=True)
    return df


def drop_rare_pois(df, min_poi_checkins):
    if min_poi_checkins > 0:
        poi_counts = df["POI_id"].value_counts()
        keep_pois = poi_counts[poi_counts >= min_poi_checkins].index
        dropped = int((~df["POI_id"].isin(keep_pois)).sum())
        print(f"POIs with fewer than {min_poi_checkins} check-ins: {int((poi_counts < min_poi_checkins).sum())} ({dropped} check-ins dropped)")
        df = df[df["POI_id"].isin(keep_pois)]
    return df.reset_index(drop=True)


def assign_trajectories(df, trajectory_gap_hours):
    df = df.sort_values(["user_id", "local_time"], kind="mergesort").reset_index(drop=True)
    gap_seconds = df.groupby("user_id")["local_time"].diff().dt.total_seconds()
    new_trajectory = gap_seconds.isna() | (gap_seconds > float(trajectory_gap_hours) * 3600.0)
    trajectory_index = new_trajectory.astype(int).groupby(df["user_id"]).cumsum()
    df["trajectory_id"] = df["user_id"].astype(str) + "_" + trajectory_index.astype(str)
    return df


def drop_short_trajectories(df, min_trajectory_length):
    if min_trajectory_length > 0:
        lengths = df["trajectory_id"].value_counts()
        keep = lengths[lengths >= min_trajectory_length].index
        dropped = int((~df["trajectory_id"].isin(keep)).sum())
        print(f"Trajectories shorter than {min_trajectory_length}: {int((lengths < min_trajectory_length).sum())} ({dropped} check-ins dropped)")
        df = df[df["trajectory_id"].isin(keep)]
    return df.reset_index(drop=True)


def split_by_time(df):
    df = df.sort_values("local_time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * SPLIT_RATIOS[0])
    val_end = train_end + int(n * SPLIT_RATIOS[1])
    return {"train": df.iloc[:train_end], "val": df.iloc[train_end:val_end], "test": df.iloc[val_end:]}


def drop_unseen_rows(splits):
    train_users = set(splits["train"]["user_id"])
    train_pois = set(splits["train"]["POI_id"])
    for name in ("val", "test"):
        df = splits[name]
        unseen_user = ~df["user_id"].isin(train_users)
        unseen_poi = ~df["POI_id"].isin(train_pois)
        drop = unseen_user | unseen_poi
        print(f"{name}: dropping {int(drop.sum())} of {len(df)} rows (unseen user: {int(unseen_user.sum())}, unseen POI: {int(unseen_poi.sum())}, both: {int((unseen_user & unseen_poi).sum())})")
        splits[name] = df[~drop].reset_index(drop=True)
    return splits


def save_splits(splits, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def describe(splits):
    full = pd.concat(list(splits.values()), ignore_index=True)
    print(f"Users: {full['user_id'].nunique()}")
    print(f"POIs: {full['POI_id'].nunique()}")
    print(f"Trajectories: {full['trajectory_id'].nunique()}")
    print(f"Check-ins: {len(full)}")
    print(f"Rows: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


def main():
    args = parse_args()
    if args.dataset == "CA":
        df = load_split_files(args.input)
    else:
        df = load_foursquare(args.input)
    print(f"Input check-ins: {len(df)}")
    df = drop_rare_pois(df, args.min_poi_checkins)
    df = assign_trajectories(df, args.trajectory_gap_hours)
    df = drop_short_trajectories(df, args.min_trajectory_length)
    df = df[OUTPUT_COLUMNS]
    splits = drop_unseen_rows(split_by_time(df))
    output_dir = args.output_root / args.dataset
    save_splits(splits, output_dir)
    describe(splits)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
