"""원본 기록(data/raw/*.jsonl.gz)을 지금 파서로 다시 읽어 InfluxDB 에 다시 쓴다.

재부팅 직후 보드 시계가 1970년이던 동안 받은 메시지는 1970 시각으로 저장됐다.
원본 기록에는 메시지 자체의 시각이 남아 있으므로, 고친 파서로 다시 읽어 올바른 시각으로 쓰고
1970~2000 사이의 잘못된 점을 지운다. 같은 시각·태그면 덮어쓰므로 여러 번 돌려도 같은 결과다.

  docker compose run --rm -v ./collector:/app/collector collector \
      python -m collector.backfill /data/raw/1970-01-01.jsonl.gz            # 먼저 --dry-run 으로 확인
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter

import yaml
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from .main import to_line
from .parse import CLOCK_MIN_NS, Parser

BAD_START, BAD_STOP = "1970-01-01T00:00:00Z", "2000-01-01T00:00:00Z"


def reparse(paths: list[str], parser: Parser) -> tuple[list[str], Counter]:
    lines, seen, n = [], set(), Counter()
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for raw in f:
                m = json.loads(raw)
                n["messages"] += 1
                try:
                    recs = parser.parse(m["topic"], m["payload"].encode("utf-8"), m["recv_ns"])
                except Exception:
                    n["parse_errors"] += 1
                    continue
                if not recs:
                    continue
                key = recs[0].dedup_key                 # 냉난방(xflow) 메시지는 키가 없다
                if key is not None:
                    if key in seen:
                        continue
                    seen.add(key)
                for r in recs:
                    if r.ts_ns < CLOCK_MIN_NS:          # 메시지에 시각이 없어 되살릴 수 없음
                        n["no_time"] += 1
                        continue
                    n[r.measurement] += 1
                    lines.append(to_line(r))
    return lines, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+", help="원본 기록 파일(.jsonl.gz)")
    ap.add_argument("--dry-run", action="store_true", help="세기만 하고 쓰거나 지우지 않음")
    ap.add_argument("--keep-bad", action="store_true", help="1970~2000 점을 지우지 않음")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.environ.get("COLLECTOR_CONFIG", "config/collector.yaml"), encoding="utf-8"))
    lines, n = reparse(args.raw, Parser(cfg))
    print("다시 읽음:", dict(n))
    if args.dry_run:
        return
    org, bucket = os.environ["INFLUX_ORG"], os.environ["INFLUX_BUCKET"]
    with InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], org=org, timeout=60_000) as c:
        w = c.write_api(write_options=SYNCHRONOUS)
        for i in range(0, len(lines), 5000):
            w.write(bucket=bucket, org=org, record=lines[i:i + 5000])
        print(f"썼음: {len(lines)}점")
        if not args.keep_bad:
            c.delete_api().delete(BAD_START, BAD_STOP, "", bucket=bucket, org=org)
            print(f"지움: {BAD_START} ~ {BAD_STOP}")


if __name__ == "__main__":
    main()
