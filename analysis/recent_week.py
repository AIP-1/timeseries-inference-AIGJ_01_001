"""최근 1주 CO₂ 관찰값을 강사 InfluxDB(iot-bucket)에서 직접 다시 계산한다.

보고서 '최근 1주 관찰' 절의 수치는 Mac Grafana 패널에서 읽은 것이라, 같은 조건으로 보드에서 재계산해 확인하는 용도.

  docker compose run --rm lab python -m analysis.recent_week --start 2026-10-15 --stop 2026-10-22

계산(모두 KST):
  1. 센서별 밤 기준선: 02~05시 15분 평균의 중앙값 (공간 안 센서 간 보정 차이 확인)
  2. 1,000ppm 초과 비율: 15분 평균 → 공간 평균 → 1000 초과 칸 / 전체 칸
  3. 공간별 최고값과 그 시각(15분 평균 기준)
  4. 시간대별 평균(평일만, 공간 평균)
강사 서버는 읽기만 한다. 결과는 data/reports/recent_week/ 에 CSV 로 남긴다.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient

KST = "Asia/Seoul"
ROOMS = ["실습실", "사무실", "회의실"]


def fetch_15m(api, bucket: str, org: str, start: str, stop: str) -> pd.DataFrame:
    q = f'''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._measurement == "co2" and r._field == "value")
  |> filter(fn: (r) => r._value > 0.0 and r._value < 5000.0)
  |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
  |> keep(columns: ["_time", "_value", "device_id", "location"])
'''
    res = api.query_data_frame(q, org=org)
    frames = res if isinstance(res, list) else [res]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("데이터 없음: 기간·토큰 확인")
    df = pd.concat(frames, ignore_index=True).rename(columns={"_time": "time", "_value": "co2"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(KST)
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    return df[["time", "device_id", "location", "co2"]].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-15")
    ap.add_argument("--stop", default="2026-10-22")
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "reports", "recent_week"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def kst_iso(d: str) -> str:
        return pd.Timestamp(d, tz=KST).tz_convert("UTC").isoformat().replace("+00:00", "Z")

    url, token = os.environ["INFLUX_URL"], os.environ["INFLUX_TOKEN"]
    org = os.environ.get("INFLUX_ORG_ID") or os.environ.get("INFLUX_ORG")
    bucket = os.environ.get("INFLUX_BUCKET", "iot-bucket")
    with InfluxDBClient(url=url, token=token, org=org, timeout=600_000) as c:
        df = fetch_15m(c.query_api(), bucket, org, kst_iso(a.start), kst_iso(a.stop))
    df = df[df["location"].isin(ROOMS)]
    print(f"기간 {a.start} ~ {a.stop} KST, 15분 칸 {len(df)}개, 센서 {df['device_id'].nunique()}대")

    # 1. 센서별 밤 기준선
    night = df[df["time"].dt.hour.between(2, 4)]
    base = night.groupby(["location", "device_id"])["co2"].median().round(0).rename("night_median_ppm")
    base.to_csv(out / "night_baseline.csv")
    print("\n[1] 센서별 밤(02~05시) 기준선 중앙값(ppm)")
    print(base.to_string())
    spread = base.groupby("location").agg(["min", "max"])
    spread["diff"] = spread["max"] - spread["min"]
    print("\n공간 안 센서 간 차이(ppm):")
    print(spread.to_string())

    # 2. 공간 평균 15분 칸
    room = df.groupby(["location", "time"])["co2"].mean().reset_index()
    over = room.assign(over=room["co2"] > 1000).groupby("location")["over"].agg(["mean", "sum", "count"])
    over["share_pct"] = (over["mean"] * 100).round(1)
    over.to_csv(out / "over_1000.csv")
    print("\n[2] 1,000ppm 초과 비율(15분 평균, 공간 평균)")
    print(over[["share_pct", "sum", "count"]].to_string())

    # 3. 최고값
    idx = room.groupby("location")["co2"].idxmax()
    peak = room.loc[idx].set_index("location")
    peak["co2"] = peak["co2"].round(0)
    peak.to_csv(out / "peak.csv")
    print("\n[3] 공간별 최고 15분 평균과 시각")
    print(peak.to_string())

    # 4. 시간대별 평균(평일)
    wk = room[room["time"].dt.weekday < 5].copy()
    wk["hour"] = wk["time"].dt.hour
    hod = wk.pivot_table(index="hour", columns="location", values="co2", aggfunc="mean").round(0)
    hod.to_csv(out / "hour_of_day_weekday.csv")
    print("\n[4] 평일 시간대별 평균(ppm)")
    print(hod.to_string())
    print("\n시간대별 봉우리(상위 3시각):")
    for r in hod.columns:
        print(f"  {r}: {list(hod[r].nlargest(3).index)}")
    print(f"\nCSV: {out}/  (실행 {datetime.now(timezone.utc).isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
