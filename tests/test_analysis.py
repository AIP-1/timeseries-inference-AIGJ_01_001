import numpy as np
import pandas as pd

from analysis import baselines, prep


def test_fill_short_gaps_only_fills_short_runs():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0, np.nan])
    out = prep.fill_short_gaps(s, max_len=3)
    assert out[1] == 2.0                      # 1칸 결측은 보간
    assert out[3:7].isna().all()              # 4칸 결측은 한 칸도 채우지 않음
    assert np.isnan(out[8])                   # 끝의 결측은 바깥이라 채우지 않음


def test_clean_drops_impossible_values():
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC"),
                       "device_id": "d", "location": "L", "value": [0.0, 800.0, 9000.0], "field": "co2"})
    kept, removed = prep.clean(df)
    assert kept["value"].tolist() == [800.0]
    assert removed["removed"].sum() == 2


def test_eval_points_require_all_baselines_and_targets():
    idx = pd.date_range("2026-01-01", periods=24 * 12 * 3, freq="5min", tz="UTC")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    pts = baselines.eval_points(s)
    # 어제 같은 시각이 필요하므로 첫날은 빠지고, 60분 정답이 필요하므로 마지막 12칸도 빠진다
    assert pts.min() >= idx[24 * 12 - 12]
    assert pts.max() <= idx[-13]
    p = baselines.predictions(s, 60).loc[pts]
    assert (p["target"] - p["persistence"]).eq(12).all()


def test_split_is_chronological_and_norm_uses_train_only():
    idx = pd.date_range("2026-01-01", periods=24 * 12 * 90, freq="5min", tz="UTC")
    grid = pd.DataFrame({("L", "co2"): np.r_[np.zeros(len(idx) - 24 * 12 * 56), np.full(24 * 12 * 56, 1000.0)]},
                        index=idx)
    grid.columns = pd.MultiIndex.from_tuples(grid.columns, names=["location", "field"])
    sp = prep.make_split(idx, 28, 28)
    assert sp.train_end < sp.val_end < sp.end
    assert set(sp.label(idx)) == {"train", "val", "test"}
    assert prep.normalize_stats(grid, sp)["mean"].iloc[0] < 50   # 검증·시험의 1000 이 섞이지 않음


def test_to_grid_handles_mixed_sensor_intervals():
    t1 = pd.date_range("2026-01-01", periods=60, freq="1min", tz="UTC")
    t10 = pd.date_range("2026-01-01 00:02", periods=6, freq="10min", tz="UTC")
    df = pd.concat([
        pd.DataFrame({"time": t1, "device_id": "fast", "location": "L", "value": 20.0}),
        pd.DataFrame({"time": t10, "device_id": "slow", "location": "L", "value": 18.0}),   # 2℃ 낮은 10분 장비
        pd.DataFrame({"time": t10, "device_id": "only10", "location": "M", "value": 25.0}),
    ])
    df["field"] = pd.Categorical(["temperature"] * len(df), categories=list(prep.FIELDS))
    g = prep.to_grid(df, prep.GridSpec(max_fill_bins=0))
    assert (g[("L", "temperature")].iloc[:-1] == 19.0).all()   # 칸마다 흔들리지 않음
    assert g[("M", "temperature")].iloc[:-1].notna().all()     # 10분 장비만 있어도 칸이 채워짐
