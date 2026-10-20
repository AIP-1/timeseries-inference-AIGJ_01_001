"""강사 InfluxDB(iot-bucket)의 과거 데이터를 월 단위 parquet 로 내려받는다.

  python -m analysis.export --days 400 --fields co2,temperature,humidity

결과: data/history/<field>/<YYYY-MM>.parquet  (columns: time, device_id, location, spot, value)
이미 받은 달은 건너뛴다(이번 달은 항상 다시 받음). 강사 서버는 읽기만 한다.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient

COLS = ["time", "device_id", "location", "spot", "value"]


def month_starts(start: datetime, end: datetime):
    m = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while m < end:
        nxt = (m + timedelta(days=32)).replace(day=1)
        yield max(m, start), min(nxt, end), m.strftime("%Y-%m")
        m = nxt


def fetch(api, bucket: str, org: str, field: str, start: datetime, stop: datetime) -> pd.DataFrame:
    q = f'''
from(bucket: "{bucket}")
  |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
  |> filter(fn: (r) => r._measurement == "{field}" and r._field == "value")
  |> keep(columns: ["_time", "_value", "device_id", "location", "spot"])
'''
    res = api.query_data_frame(q, org=org)
    frames = res if isinstance(res, list) else [res]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"_time": "time", "_value": "value"})
    for c in COLS:
        if c not in df:
            df[c] = None
    df = df[COLS]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--fields", default="co2,temperature,humidity")
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "history"))
    a = ap.parse_args()

    url, token = os.environ["INFLUX_URL"], os.environ["INFLUX_TOKEN"]
    org = os.environ.get("INFLUX_ORG_ID") or os.environ.get("INFLUX_ORG")
    bucket = os.environ.get("INFLUX_BUCKET", "iot-bucket")
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=a.days)
    this_month = end.strftime("%Y-%m")

    with InfluxDBClient(url=url, token=token, org=org, timeout=600_000) as c:
        api = c.query_api()
        for field in a.fields.split(","):
            outdir = Path(a.out) / field
            outdir.mkdir(parents=True, exist_ok=True)
            for s, e, tag in month_starts(start, end):
                path = outdir / f"{tag}.parquet"
                if path.exists() and tag != this_month:
                    continue
                df = fetch(api, bucket, org, field, s, e)
                df.to_parquet(path, index=False)
                print(f"{field} {tag}: {len(df):>8} rows, devices={df['device_id'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
