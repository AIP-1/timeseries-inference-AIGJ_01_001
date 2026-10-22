"""수집 현황 보고: InfluxDB 에서 최근 N시간의 공간·항목별 커버율과 빠진 구간을 계산해 Markdown 으로 출력.

  python -m collector.report --hours 24 [--min-gap 5]
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from influxdb_client import InfluxDBClient

FIELDS = ("co2", "temperature", "humidity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--min-gap", type=int, default=5, help="이 분 수 이상 비면 결측 구간으로 표시")
    ap.add_argument("--measurement", default="env")
    a = ap.parse_args()

    bucket, org = os.environ["INFLUX_BUCKET"], os.environ["INFLUX_ORG"]
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=a.hours)
    fields = " or ".join(f'r._field == "{f}"' for f in FIELDS)
    q = f'''
from(bucket: "{bucket}")
  |> range(start: {start.isoformat()}, stop: {end.isoformat()})
  |> filter(fn: (r) => r._measurement == "{a.measurement}" and ({fields}))
  |> group(columns: ["space", "_field"])
  |> aggregateWindow(every: 1m, fn: count, createEmpty: false)
  |> keep(columns: ["space", "_field", "_time"])
'''
    minutes: dict[tuple[str, str], set[datetime]] = defaultdict(set)
    with InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"],
                        org=org, timeout=60_000) as c:
        for table in c.query_api().query(q):
            for r in table.records:
                # aggregateWindow 는 구간 끝 시각을 찍으므로 1분 당겨 구간 시작으로 맞춘다
                minutes[(r.values.get("space", "-"), r.get_field())].add(r.get_time() - timedelta(minutes=1))

    total = int(a.hours * 60)
    print(f"# 수집 현황 ({start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M} UTC, {a.hours:g}시간)\n")
    print("| 공간 | 항목 | 보고 주기 | 받은 횟수/기대 횟수 | 커버율 | 마지막 수신(UTC) |")
    print("|---|---|---:|---:|---:|---|")
    gaps_out = []
    for (space, field), mins in sorted(minutes.items()):
        ordered = sorted(mins)
        # 센서마다 보내는 주기가 다르다(1·2·10분). 커버율은 '그 주기로 왔어야 할 횟수' 대비로,
        # 빠진 구간은 '평소 간격 + min_gap' 이상 빈 곳만 센다(10분 센서의 평소 9분 빈칸은 결측이 아님).
        diffs = sorted(int((b - c).total_seconds() // 60) for c, b in zip(ordered, ordered[1:]))
        every = max(1, diffs[len(diffs) // 2]) if diffs else 1
        expected = max(1, total // every)
        pct = min(100.0, 100 * len(mins) / expected)
        print(f"| {space} | {field} | {every}분 | {len(mins)}/{expected} | {pct:.1f}% | {ordered[-1]:%m-%d %H:%M} |")
        prev = start - timedelta(minutes=1)
        for t in ordered + [end]:
            gap = int((t - prev).total_seconds() // 60) - 1
            if gap >= every - 1 + a.min_gap:
                gaps_out.append((space, field, prev + timedelta(minutes=1), t, gap))
            prev = t
    print(f"\n## 평소 간격보다 {a.min_gap}분 이상 더 빈 구간\n")
    if not gaps_out:
        print("없음")
    else:
        print("| 공간 | 항목 | 시작(UTC) | 끝(UTC) | 길이(분) | 추정 원인 |")
        print("|---|---|---|---|---:|---|")
        for space, field, s, e, g in gaps_out:
            print(f"| {space} | {field} | {s:%m-%d %H:%M} | {e:%m-%d %H:%M} | {g} | |")


if __name__ == "__main__":
    main()
