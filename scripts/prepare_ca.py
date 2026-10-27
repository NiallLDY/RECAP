import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

TIMEZONE = ZoneInfo("America/Los_Angeles")

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def convert_split(df):
    utc = pd.to_datetime(df["checkin_time"], utc=True, errors="coerce")
    if utc.isna().any():
        raise ValueError(f"{int(utc.isna().sum())} unparsable timestamps in checkin_time")
    local = utc.dt.tz_convert(TIMEZONE)
    offset_minutes = local.map(lambda t: int(round(t.utcoffset().total_seconds() / 60.0))).astype("int32")
    local_naive = local.dt.tz_localize(None)

    out = pd.DataFrame(index=df.index)
    out["user_id"] = df["user_id"]
    out["POI_id"] = df["POI_id"]
    out["POI_catid_code"] = df["POI_catid_code"].astype("int32")
    out["POI_catid"] = out["POI_catid_code"].astype(str)
    out["POI_catname"] = df["POI_catname"].astype(str)
    out["latitude"] = df["latitude"].astype("float32")
    out["longitude"] = df["longitude"].astype("float32")
    out["timezone"] = offset_minutes
    out["UTC_time"] = utc.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out["local_time"] = local_naive.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out["day_of_week"] = local_naive.dt.dayofweek.astype("int16")
    out["norm_in_day_time"] = (
        (local_naive.dt.hour * 3600 + local_naive.dt.minute * 60 + local_naive.dt.second).astype("float64") / 86400.0
    )
    out["trajectory_id"] = df["trajectory_id"]
    out = out.dropna(subset=["user_id", "POI_id", "POI_catid_code", "latitude", "longitude", "UTC_time", "trajectory_id"])
    return out[OUTPUT_COLUMNS]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        df = pd.read_csv(args.input_dir / f"gowalla_{name}.csv")
        out = convert_split(df)
        out.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(out)} rows")


if __name__ == "__main__":
    main()
