"""과제 6: 사전학습 시계열 모델(Chronos-Bolt) 비교. 추가 학습 없이 CO2 60분 예측.

  python -m analysis.task6 [--model amazon/chronos-bolt-small] [--contexts 24,288,512] [--stride 3]

같은 시점에서 비교: 기준선(현재값 유지·1시간 평균·어제) / 직접 학습한 GRU(과제 3 gru_full.pt) / Chronos.
시점 = 과제 2 평가 시점(baselines.eval_points) ∩ 시험 기간 ∩ 가장 긴 과거 길이를 줄 수 있는 시각,
그중 stride 칸마다 하나(추론 시간을 줄이려고. 모든 방법이 같은 부분 집합을 쓴다).
처음 실행 때 모델을 Hugging Face 에서 내려받는다(보드에서 인터넷 필요, 이후는 캐시).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import baselines, prep, task3

QUANT = [0.1, 0.5, 0.9]
H = 12


def rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def load_pipeline(name: str, device: str):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(name, device_map=device, torch_dtype=torch.float32)


def chronos_forecast(pipe, series: np.ndarray, ti: np.ndarray, ctx: int, batch: int = 64) -> np.ndarray:
    """[N, H, 3] (10%, 50%, 90%). 과거 ctx 칸(결측은 NaN 그대로, 모델이 처리)."""
    out = []
    for i in range(0, len(ti), batch):
        ctxs = [torch.tensor(series[t - ctx + 1:t + 1], dtype=torch.float32) for t in ti[i:i + batch]]
        q, _ = pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=QUANT)
        out.append(q.cpu().numpy())
    return np.concatenate(out)


_GRU: dict = {}


def gru_forecast(task3_dir: Path, grid, norm, split, loc: str, ti: np.ndarray):
    """과제 3 GRU(full) 의 CO2 60분 예측. 체크포인트와 입력 구성이 안 맞으면 None."""
    if "model" not in _GRU:
        path = task3_dir / "gru_full.pt"
        if not path.exists():
            return None, "gru_full.pt 없음"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        data = task3.Data(grid, norm, split, ck["lookback"], use_calendar=True)
        if data.X.shape[2] != ck["n_in"] or data.n_static != ck["n_static"]:
            return None, f"입력 크기 불일치(체크포인트 {ck['n_in']}/{ck['n_static']}, 지금 {data.X.shape[2]}/{data.n_static})"
        model = task3.GRUForecaster(ck["n_in"], ck["n_static"], ck["hidden"], ck["layers"])
        model.load_state_dict(ck["state_dict"])
        _GRU.update(model=model, data=data)
    model, data = _GRU["model"], _GRU["data"]
    l = data.locations.index(loc)
    return task3.predict(model, data, l, ti, torch.device("cpu"))[:, :, 0], ""


def timing(pipe, series: np.ndarray, t: int, ctx: int, batch: int, reps: int = 30, warmup: int = 3) -> dict:
    ctxs = [torch.tensor(series[t - ctx + 1:t + 1], dtype=torch.float32)] * batch
    for _ in range(warmup):
        pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=QUANT)
    ts = []
    for _ in range(reps):
        a = time.perf_counter()
        pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=QUANT)
        ts.append((time.perf_counter() - a) * 1000)
    return {"batch": batch, "context": ctx, "mean_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95)), "reps": reps, "warmup": warmup}


def md(df: pd.DataFrame, fmt=".1f") -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    d = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--task2", default=os.path.join(d, "reports", "task2"))
    ap.add_argument("--task3", default=os.path.join(d, "reports", "task3"))
    ap.add_argument("--out", default=os.path.join(d, "reports", "task6"))
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--contexts", default="24,288,512", help="과거 길이(5분 칸): 2시간, 하루, 약 43시간")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 1)
    contexts = [int(c) for c in a.contexts.split(",")]

    grid, norm, split = task3.load_task2(a.task2)
    test = grid.index[grid.index >= split.val_end]
    pos = pd.Series(np.arange(len(grid.index)), index=grid.index)

    m0 = rss_mb()
    t0 = time.perf_counter()
    pipe = load_pipeline(a.model, device)
    load_s = time.perf_counter() - t0
    m_model = rss_mb() - m0

    rows, cover, notes, preds = [], [], [], []
    locs = sorted(l for l in grid.columns.get_level_values("location").unique() if (l, "co2") in grid)
    for loc in locs:
        s = grid[(loc, "co2")]
        pts = baselines.eval_points(s).intersection(test)
        ti = pos[pts].to_numpy()
        ti = ti[ti >= max(contexts) - 1][:: a.stride]
        if len(ti) == 0:
            continue
        pts = grid.index[ti]
        v = s.to_numpy(np.float64)
        target = np.column_stack([v[ti + k] for k in range(1, H + 1)])
        P = {}
        for h in baselines.HORIZONS_MIN:
            b = baselines.predictions(s, h).loc[pts]
            for m in ("persistence", "mean_1h", "yesterday"):
                P.setdefault(m, {})[h] = b[m].to_numpy()
        g, why = gru_forecast(Path(a.task3), grid, norm, split, loc, ti)
        if g is not None:
            P["gru"] = {h: g[:, h // 5 - 1] for h in baselines.HORIZONS_MIN}
        elif why not in notes:
            notes.append(why)
        for ctx in contexts:
            q = chronos_forecast(pipe, v.astype(np.float32), ti, ctx)
            P[f"chronos_{ctx}"] = {h: q[:, h // 5 - 1, 1] for h in baselines.HORIZONS_MIN}
            for h in baselines.HORIZONS_MIN:
                k = h // 5 - 1
                inside = (target[:, k] >= q[:, k, 0]) & (target[:, k] <= q[:, k, 2])
                cover.append({"location": loc, "context": ctx, "horizon_min": h, "coverage_%": inside.mean() * 100,
                              "width_ppm": float(np.mean(q[:, k, 2] - q[:, k, 0]))})
            if g is not None:
                P[f"gru+chronos_{ctx}"] = {h: (P["gru"][h] + P[f"chronos_{ctx}"][h]) / 2 for h in baselines.HORIZONS_MIN}
        for m, byh in P.items():
            for h, p in byh.items():
                rows.append({"location": loc, "model": m, "horizon_min": h, "mae": float(np.nanmean(np.abs(p - target[:, h // 5 - 1]))), "n": len(ti)})
        preds.append(pd.DataFrame({"location": loc, "time": pts, "target_60": target[:, -1],
                                   **{f"{m}_60": byh[60] for m, byh in P.items()}}))

    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions_60.csv", index=False)
    cov = pd.DataFrame(cover)
    cov.to_csv(out / "coverage.csv", index=False)

    best_ctx = max(contexts)
    s0 = grid[(locs[0], "co2")].to_numpy(np.float32)
    t_last = len(s0) - 1
    tim = [timing(pipe, s0, t_last, c, b) for c in contexts for b in (1, 4)]
    tim = pd.DataFrame(tim)
    tim.to_csv(out / "timing.csv", index=False)
    peak = rss_mb()
    gpu = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else float("nan")

    pooled = res.assign(w=res.mae * res.n).groupby(["model", "horizon_min"]).agg(w=("w", "sum"), n=("n", "sum"))
    pooled = (pooled.w / pooled.n).unstack("horizon_min").reset_index()
    pooled.columns = [f"{c}분" if isinstance(c, (int, np.integer)) else c for c in pooled.columns]
    L = ["# 과제 6. 사전학습 모델 비교 (CO₂)\n"]
    L.append(f"- 모델: `{a.model}`(추가 학습 없음), 장치 {device}. 과거 길이 {', '.join(f'{c}칸({c * 5 / 60:.0f}시간)' for c in contexts)}")
    L.append(f"- 평가: 시험 기간, 과제 2 평가 시점 중 {a.stride}칸(={a.stride * 5}분)마다 하나, 모든 방법 같은 시점")
    if notes:
        L.append(f"- GRU 비교 제외 사유: {'; '.join(notes)}")
    L.append("\n## 정확도 (MAE ppm, 공간 합산)\n")
    L += md(pooled)
    L.append("\n### 공간별 60분 MAE (ppm)\n")
    L += md(res[res.horizon_min == 60].pivot_table(index="location", columns="model", values="mae").reset_index())
    L.append("\n## 예측 구간(10~90%) 적중률 — 이상적이면 80%\n")
    L += md(cov.groupby(["context", "horizon_min"])[["coverage_%", "width_ppm"]].mean().reset_index())
    L.append("\n## 비용 (보드)\n")
    L.append(f"- 모델 불러오기 {load_s:.1f}초, 불러온 뒤 늘어난 메모리(RSS) {m_model:.0f}MB, 실행 중 최대 RSS {peak:.0f}MB"
             + (f", GPU 최대 {gpu:.0f}MB" if device == "cuda" else ""))
    L += md(tim)
    L.append("\n(지연은 predict_quantiles 1회. 워밍업 3회 뒤 30회 반복, 평균과 느린 쪽 5%)")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
