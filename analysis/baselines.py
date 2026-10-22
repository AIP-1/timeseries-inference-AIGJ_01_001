"""기준선 3종: 현재값 유지, 최근 1시간 평균, 어제 같은 시각.

모든 모델 평가(과제 3·6)는 eval_points() 가 고른 같은 시점에서 해야 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS_MIN = (5, 15, 30, 60)
STEP_MIN = 5


def predictions(series: pd.Series, horizon_min: int) -> pd.DataFrame:
    """시각 t 에서 t+h 를 예측한 값들. index 는 예측을 내는 시각 t."""
    h = horizon_min // STEP_MIN
    day = 24 * 60 // STEP_MIN
    return pd.DataFrame({
        "target": series.shift(-h),
        "persistence": series,                                    # 지금 값이 그대로
        "mean_1h": series.rolling(60 // STEP_MIN, min_periods=60 // STEP_MIN).mean(),  # 최근 1시간 평균
        "yesterday": series.shift(day - h),                       # 24시간 전의 t+h 값
    })


def eval_points(series: pd.Series, max_horizon_min: int = max(HORIZONS_MIN)) -> pd.Index:
    """모든 기준선과 모든 예측 시점의 정답이 존재하는 시각만 고른다(같은 시점, 같은 조건)."""
    ok = series.notna()
    ok &= series.rolling(12, min_periods=12).count().eq(12)            # 최근 1시간 꽉 참
    for h in HORIZONS_MIN:
        ok &= series.shift(-(h // STEP_MIN)).notna()
        ok &= series.shift(24 * 60 // STEP_MIN - h // STEP_MIN).notna()
    return series.index[ok.values]


def mae_table(grid: pd.DataFrame, field: str, mask_index: pd.Index | None = None) -> pd.DataFrame:
    """공간별 × 기준선 × 예측 시점 MAE. mask_index 로 평가 기간(예: 시험 기간)을 제한."""
    rows = []
    for loc in grid.columns.get_level_values("location").unique():
        if (loc, field) not in grid:
            continue
        s = grid[(loc, field)]
        pts = eval_points(s)
        if mask_index is not None:
            pts = pts.intersection(mask_index)
        if len(pts) == 0:
            continue
        for h in HORIZONS_MIN:
            p = predictions(s, h).loc[pts]
            for m in ("persistence", "mean_1h", "yesterday"):
                rows.append({"location": loc, "field": field, "horizon_min": h, "baseline": m,
                             "mae": float(np.abs(p[m] - p["target"]).mean()), "n": len(pts)})
    return pd.DataFrame(rows)
