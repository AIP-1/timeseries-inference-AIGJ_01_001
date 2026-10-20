"""과제 8: 상시 서비스 가동 기록 요약 (data/service/cycles-*.jsonl).

  python -m analysis.service_report [--since 2026-10-21T00:00Z] [--out data/reports/task8]

- 가동 시간, 주기 수, 빠진 주기(2분 넘게 기록이 없던 구간), 실패 주기
- 주기별 처리 시간(평균·p95·최대)
- 경보 발생 내역(raised/cleared), 데이터 끊김 경보의 시작·해제(수집기 정지 시험 확인용)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    d = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--logs", default=os.path.join(d, "service"))
    ap.add_argument("--since")
    ap.add_argument("--out", default=os.path.join(d, "reports", "task8"))
    a = ap.parse_args()
    rows = [json.loads(line) for f in sorted(Path(a.logs).glob("cycles-*.jsonl")) for line in open(f, encoding="utf-8") if line.strip()]
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if a.since:
        df = df[df["time"] >= pd.Timestamp(a.since)]
    df = df.sort_values("time")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gaps = df["time"].diff()
    holes = df.loc[gaps > pd.Timedelta("2min"), "time"]
    span = df["time"].iloc[-1] - df["time"].iloc[0]
    ok = df[df["ok"] == True]  # noqa: E712
    ev = pd.DataFrame([e for es in df.get("events", pd.Series(dtype=object)).dropna() for e in es])
    L = ["# 과제 8. 상시 서비스 가동 기록\n"]
    L.append(f"- 기간 {df['time'].iloc[0]:%Y-%m-%d %H:%M} ~ {df['time'].iloc[-1]:%Y-%m-%d %H:%M} UTC ({span.total_seconds() / 3600:.1f}시간)")
    L.append(f"- 주기 {len(df)}회, 성공 {len(ok)}회, 실패 {len(df) - len(ok)}회, 2분 넘게 빈 구간 {len(holes)}회"
             + (f" (최대 {gaps.max().total_seconds() / 60:.1f}분)" if len(holes) else ""))
    if len(ok):
        t = ok["total_s"]
        L.append(f"- 주기 처리 시간: 평균 {t.mean():.2f}초, p95 {t.quantile(0.95):.2f}초, 최대 {t.max():.2f}초 "
                 f"(읽기 평균 {ok['read_s'].mean():.2f}초, 계산 평균 {ok['compute_s'].mean():.2f}초)")
    if (df["ok"] == False).any():  # noqa: E712
        L.append("\n## 실패한 주기\n")
        for r in df[df["ok"] == False].itertuples():  # noqa: E712
            L.append(f"- {r.time:%m-%d %H:%M} {r.error}")
    L.append("\n## 경보 내역\n")
    if len(ev):
        ev["time"] = pd.to_datetime(ev["time"], utc=True)
        cnt = ev[ev.status == "raised"].groupby(["space", "alert"]).size().rename("raised").reset_index()
        L.append("| space | alert | raised |")
        L.append("|---|---|---|")
        L += [f"| {r.space} | {r.alert} | {r.raised} |" for r in cnt.itertuples()]
        gap = ev[ev.alert == "data_gap"].sort_values("time")
        if len(gap):
            L.append("\n### 데이터 끊김 경보 (수집기 정지 시험)\n")
            L += [f"- {r.time.tz_convert('Asia/Seoul'):%m-%d %H:%M} KST {r.space} {r.status}" for r in gap.itertuples()]
        ev.to_csv(out / "alerts.csv", index=False)
    else:
        L.append("경보 없음")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
