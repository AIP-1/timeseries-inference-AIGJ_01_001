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


def _toy_grid(days=90):
    idx = pd.date_range("2026-01-01", periods=24 * 12 * days, freq="5min", tz="UTC")
    hour = idx.tz_convert(prep.TZ).hour
    co2 = 450 + 400 * ((hour >= 9) & (hour < 18)) + np.random.default_rng(0).normal(0, 10, len(idx))
    cols = {("A", "co2"): co2, ("A", "temperature"): 22 + 0.1 * np.sin(np.arange(len(idx)) / 50),
            ("A", "humidity"): np.full(len(idx), 40.0)}
    grid = pd.DataFrame(cols, index=idx)
    grid.columns = pd.MultiIndex.from_tuples(grid.columns, names=["location", "field"])
    return grid


def test_task3_samples_stay_inside_their_split():
    from analysis import task3
    grid = _toy_grid()
    sp = prep.make_split(grid.index, 28, 28)
    data = task3.Data(grid, prep.normalize_stats(grid, sp), sp)
    for part in ("train", "val"):
        _, t = data.candidates(part)
        lab = sp.label(grid.index)
        assert (lab[t] == part).all() and (lab[t + task3.HORIZON] == part).all()
    # 변화량 목표를 되돌리면 실제 미래값이 나온다
    li, ti = np.array([0]), np.array([500])
    back = data.to_values(li, ti, data.Y[li, ti])
    assert np.allclose(back[0, :, 0], grid[("A", "co2")].iloc[501:513].to_numpy(), atol=1e-2)


def test_grid_keeps_slow_sensors_and_stable_device_mix():
    t1 = pd.date_range("2026-01-01", periods=600, freq="1min", tz="UTC")
    t10 = t1[::10]
    fast = pd.DataFrame({"time": t1, "device_id": "fast", "value": 20.0})
    slow = pd.DataFrame({"time": t10, "device_id": "slow", "value": 24.0})
    only_slow = slow.assign(location="M")                     # 10분 센서 하나뿐인 공간 (회의실)
    mixed = pd.concat([fast, slow]).assign(location="L")     # 1분 + 10분 센서가 섞인 공간 (실습실)
    tiny = pd.DataFrame({"time": t1[:2], "device_id": "t", "value": 21.0, "location": "T"})   # 시험용 장비
    df = pd.concat([only_slow, mixed, tiny], ignore_index=True).assign(field="temperature")
    for c in ("device_id", "location"):
        df[c] = df[c].astype("category")
    g = prep.to_grid(df)
    assert ("M", "temperature") in g and g[("M", "temperature")].notna().mean() > 0.95
    assert ("T", "temperature") not in g
    assert np.allclose(g[("L", "temperature")].dropna(), 22.0)  # 칸마다 두 장비 평균, 튀지 않음
