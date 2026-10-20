"""과제 2 실행: 정제 → 5분 격자 → 시간순 분할 → 진단 → 기준선 성능표.

  python -m analysis.task2 [--history data/history] [--out data/reports/task2]

산출물(out 폴더): report.md, pattern_<field>.png, grid.parquet(다음 과제 입력), split.json, norm.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import baselines, prep  # noqa: E402

plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["axes.unicode_minus"] = False


def md(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def co2_onsets(s: pd.Series) -> pd.DataFrame:
    """평일마다 '아침에 CO2가 오르기 시작한 시각'. 일정 자료가 없어 사용 시작의 대리 지표로 쓴다.

    기준: 그날 04~06시 중앙값보다 100ppm 이상 높아진 06~14시 사이 첫 5분 칸.
    """
    local = s.copy()
    local.index = s.index.tz_convert(prep.TZ)
    rows = []
    for day, g in local.groupby(local.index.date):
        if pd.Timestamp(day).dayofweek >= 5:
            continue
        night = g.between_time("04:00", "06:00").median()
        morning = g.between_time("06:00", "14:00")
        if pd.isna(night) or morning.notna().sum() < 60:
            continue
        hit = morning[morning > night + 100]
        onset = hit.index[0] if len(hit) else None
        rows.append({"date": str(day), "dow": pd.Timestamp(day).day_name()[:3],
                     "night_ppm": round(float(night)),
                     "onset": onset.strftime("%H:%M") if onset is not None else "없음"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    data = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--history", default=os.path.join(data, "history"))
    ap.add_argument("--out", default=os.path.join(data, "reports", "task2"))
    ap.add_argument("--test-days", type=int, default=28)
    ap.add_argument("--val-days", type=int, default=28)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 항목별로 따로 읽고 격자로 줄인 뒤 원자료는 버린다 (1년치 원자료를 한꺼번에 올리면 보드 메모리가 모자람)
    grids, grids_nofill, removed_all, dev_all, t_min, t_max = [], [], [], [], None, None
    for f in prep.FIELDS:
        raw = prep.load_history(a.history, f)
        t_min = min(filter(None, [t_min, raw["time"].min()]))
        t_max = max(filter(None, [t_max, raw["time"].max()]))
        dev_all.append(raw.groupby(["location", "field"], observed=True)
                       .agg(devices=("device_id", "nunique"), rows=("value", "size")).reset_index())
        kept, removed = prep.clean(raw)
        del raw
        removed_all.append(removed)
        grids.append(prep.to_grid(kept))
        grids_nofill.append(prep.to_grid(kept, prep.GridSpec(max_fill_bins=0)))
        del kept
    grid = pd.concat(grids, axis=1).sort_index(axis=1)
    grid_nofill = pd.concat(grids_nofill, axis=1).sort_index(axis=1)
    removed = pd.concat(removed_all, ignore_index=True)
    dev = pd.concat(dev_all, ignore_index=True)
    split = prep.make_split(grid.index, a.test_days, a.val_days)
    norm = prep.normalize_stats(grid, split)

    grid.to_parquet(out / "grid.parquet")
    norm.to_csv(out / "norm.csv")
    (out / "split.json").write_text(json.dumps({
        "train": [str(grid.index.min()), str(split.train_end)],
        "val": [str(split.train_end), str(split.val_end)],
        "test": [str(split.val_end), str(split.end)],
        "freq": "5min", "tz_for_patterns": prep.TZ,
    }, indent=1))

    L = []
    L.append("# 과제 2. 데이터 진단과 기준선\n")
    L.append(f"- 데이터: 실데이터, 강사 InfluxDB `iot-bucket`, {t_min:%Y-%m-%d} ~ {t_max:%Y-%m-%d %H:%M} UTC")
    L.append(f"- 분할(시간순, KST 자정 경계): 학습 ~{split.train_end:%Y-%m-%d}, 검증 ~{split.val_end:%Y-%m-%d}, 시험 ~{split.end:%Y-%m-%d}")
    L.append("- 분석 간격 5분: 과제 3이 5분 간격 예측을 요구하고, 1분 값의 측정 잡음(±수십 ppm)을 평균으로 줄인다.")
    L.append("- 정규화 통계는 학습 기간으로만 계산(norm.csv).\n")

    L.append("## 장비 수와 원자료 건수\n")
    L.append(md(dev))

    L.append("\n## 정제 규칙\n")
    L.append(md(pd.DataFrame([{"field": k, "min": v[0], "max": v[1]} for k, v in prep.RULES.items()])))
    L.append("\n- 범위 밖 값은 버린다(0ppm 등 깨진 값). 장비별 1분 평균 → 공간별 장비 평균 → 5분 평균.")
    L.append("- 5분 칸에 1분 값이 3개 미만이면 칸을 비운다(막 시작된 칸, 끊김 직후 칸).")
    L.append("- 15분 이하 결측은 선형 보간, 그보다 긴 결측은 비워 두고 그 시점은 평가에서 뺀다.\n")
    L.append("### 제거된 값\n")
    L.append(md(removed) if len(removed) else "없음")

    L.append("\n## 결측률 (5분 칸, 보간 전 / 후)\n")
    miss = pd.DataFrame({"before_fill_%": grid_nofill.isna().mean() * 100,
                         "after_fill_%": grid.isna().mean() * 100}).reset_index()
    L.append(md(miss))

    for field in ("co2", "temperature"):
        sub = grid.xs(field, axis=1, level="field")
        local = sub.copy()
        local.index = local.index.tz_convert(prep.TZ)
        wk = local[local.index.dayofweek < 5]
        pattern = wk.groupby(wk.index.hour + wk.index.minute / 60).mean()
        ax = pattern.plot(figsize=(9, 4), title=f"평일 시간대별 평균 {field} (KST, 학습+검증+시험 전체)")
        ax.set_xlabel("시각")
        ax.figure.tight_layout()
        ax.figure.savefig(out / f"pattern_{field}.png", dpi=110)
        plt.close(ax.figure)
        L.append(f"\n![{field}](pattern_{field}.png)")

    L.append("\n## 아침 CO₂ 상승 시각 (일정 자료 대신)\n")
    L.append("일정 자료가 아직 없어 '그날 새벽보다 100ppm 오른 첫 시각'으로 사용 시작을 대신 본다. 요일별 중앙값과, 평일인데 오르지 않은 날(휴일·결석 후보)을 적는다.\n")
    for loc in grid.columns.get_level_values("location").unique():
        if (loc, "co2") not in grid:
            continue
        on = co2_onsets(grid[(loc, "co2")])
        if on.empty:
            continue
        on.to_csv(out / f"onsets_{loc}.csv", index=False)
        has = on[on["onset"] != "없음"].copy()
        if len(has):
            mins = pd.to_datetime(has["onset"], format="%H:%M")
            has["m"] = mins.dt.hour * 60 + mins.dt.minute
            med = has.groupby("dow")["m"].median().map(lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}")
            L.append(f"**{loc}** 평일 {len(on)}일 중 상승 {len(has)}일, 요일별 중앙값: " +
                     ", ".join(f"{d} {t}" for d, t in med.items()))
        none = on[on["onset"] == "없음"]["date"].tolist()
        L.append(f"- 오르지 않은 평일 {len(none)}일: {', '.join(none[-15:])}{' …' if len(none) > 15 else ''}\n")

    L.append("## 기준선 성능 (MAE, 시험 기간, 모든 기준선·시점이 같은 평가 시점)\n")
    test_idx = grid.index[grid.index >= split.val_end]
    tables = []
    for field in ("co2", "temperature"):
        t = baselines.mae_table(grid, field, test_idx)
        if t.empty:
            continue
        tables.append(t)
        piv = t.pivot_table(index=["location", "baseline"], columns="horizon_min", values="mae").reset_index()
        piv.columns = [f"{c}분" if isinstance(c, int) else c for c in piv.columns]
        unit = "ppm" if field == "co2" else "℃"
        L.append(f"### {field} ({unit})\n")
        L.append(md(piv, ".2f" if field == "temperature" else ".1f"))
        L.append("")
    if tables:
        pd.concat(tables).to_csv(out / "baselines_test.csv", index=False)

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
