"""과제 7: 엣지 최적화. 예측(GRU)·이상(오토인코더)·분류(환기 MLP)를 ONNX 로 내보내고 실행 방식별 지연·오차를 잰다.

1단계 (lab 컨테이너, CPU):  python -m analysis.task7
    - ONNX 내보내기(배치 크기 가변) → data/reports/task7/onnx/*.onnx
    - PyTorch CPU / (있으면) PyTorch GPU / ONNX Runtime 지연(배치 1·4, 평균·p95) 과 결과 차이
    - TensorRT 비교용 입력·기준 출력(fp32 PyTorch)을 저장: bench/*.bin, bench/meta.json
2단계 (보드 호스트, TensorRT):  python3 scripts/trt_bench.py data/reports/task7
    - trtexec 로 FP16 엔진을 만들고 지연·결과 차이·전력·GPU 사용률을 재서 trt.json 저장
3단계:  python -m analysis.task7 --report   → report.md (두 결과 합침)

차이는 원래 단위로 잰다: GRU 는 ppm·℃, 분류는 판정이 바뀐 비율(환기 경보, 이상 판정).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import prep, task3, task4, task5

N_DIFF = 256           # 결과 차이를 잴 표본 수(시험 기간에서 고름)
WARMUP, REPS = int(os.environ.get("T7_WARMUP", 50)), int(os.environ.get("T7_REPS", 500))


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def load_models(rep: Path):
    grid, norm, split = task3.load_task2(rep / "task2")
    ck = torch.load(rep / "task3" / "gru_full.pt", map_location="cpu", weights_only=False)
    gru = task3.GRUForecaster(ck["n_in"], ck["n_static"], ck["hidden"], ck["layers"])
    gru.load_state_dict(ck["state_dict"])
    data = task3.Data(grid, norm, split, ck["lookback"], use_calendar=True)
    ae_ck = torch.load(rep / "task4" / "ae.pt", map_location="cpu", weights_only=False)
    aed = task4.AEData(grid, norm, ae_ck["locations"])
    x0, _, _ = aed.windows(0, np.array([task4.W - 1]))
    ae = task4.AE(x0.shape[1], task4.W * 3)
    ae.load_state_dict(ae_ck["state_dict"])
    thr = pd.read_csv(rep / "task4" / "thresholds.csv", index_col=[0, 1]).iloc[:, 0]
    v_ck = torch.load(rep / "task5" / "vent.pt", map_location="cpu", weights_only=False) if (rep / "task5" / "vent.pt").exists() else None
    vent = None
    if v_ck:
        vent = task5.MLP(v_ck["n_in"])
        vent.load_state_dict(v_ck["state_dict"])
    for m in (gru, ae, vent):
        if m is not None:
            m.eval()
    return grid, norm, split, data, gru, aed, ae, {k: float(v) for k, v in thr.items()}, vent, v_ck


def samples(grid, norm, split, data, aed, v_ck, rng):
    """시험 기간에서 무작위 N_DIFF 개. 모델별 입력 배열과, 원래 단위로 되돌리는 데 필요한 값."""
    test_i = np.flatnonzero(grid.index >= split.val_end)
    li, ti = data.candidates("train")[0][:0], np.array([], int)
    cand = [(l, t) for l in range(len(data.locations)) for t in test_i[::7]
            if t + task3.HORIZON < len(grid.index) and not np.isnan(data.raw[l, t]).any()]
    pick = rng.choice(len(cand), min(N_DIFF, len(cand)), replace=False)
    li, ti = np.array([cand[p][0] for p in pick]), np.array([cand[p][1] for p in pick])
    x, s, _, _ = data.batch(li, ti, "cpu")
    out = {"gru": {"inputs": {"x": x.numpy(), "s": s.numpy()}, "std": data.std[li]}}
    # 오토인코더: 같은 시각, 같은 공간(이상 탐지 모델 공간 목록 기준)
    ae_l = [aed.locations.index(data.locations[l]) if data.locations[l] in aed.locations else 0 for l in li]
    xs, vs, ms = zip(*(aed.windows(l, np.array([t])) for l, t in zip(ae_l, ti)))
    out["ae"] = {"inputs": {"x": np.concatenate(xs)}, "v": np.concatenate(vs), "m": np.concatenate(ms),
                 "loc": [aed.locations[l] for l in ae_l]}
    if v_ck:
        rows = []
        for l, t in zip(li, ti):
            loc = data.locations[l]
            if loc in v_ck["locations"] and (loc, "co2") in grid:
                X, _ = task5.features(grid, norm, loc, v_ck["locations"], lags=v_ck.get("lags", False))
                rows.append(X[t])
        if rows:
            out["vent"] = {"inputs": {"x": np.stack(rows).astype(np.float32)}}
    return out


def ae_flags(recon, v, m, locs, thr):
    """창별·항목별 복원 오차 → 임계값 넘으면 이상(task4 와 같은 계산)."""
    e = (np.abs(recon - v) * m).reshape(len(v), task4.W, 3)
    mm = m.reshape(len(v), task4.W, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = e.sum(1) / mm.sum(1)
    flags = np.zeros_like(err, bool)
    for i, loc in enumerate(locs):
        for j, f in enumerate(prep.FIELDS):
            if (loc, f) in thr and not np.isnan(err[i, j]):
                flags[i, j] = err[i, j] > thr[(loc, f)]
    return flags


def export(model, inputs: dict, path: Path):
    names = list(inputs)
    args = tuple(torch.from_numpy(inputs[n][:2]) for n in names)
    dyn = {n: {0: "batch"} for n in names}
    dyn["out"] = {0: "batch"}
    kw = dict(input_names=names, output_names=["out"], dynamic_axes=dyn, opset_version=17)
    try:
        torch.onnx.export(model, args, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, args, str(path), **kw)


def bench(fn, batch_inputs, warmup=WARMUP, reps=REPS) -> dict:
    for _ in range(warmup):
        fn(batch_inputs)
    ts = []
    for _ in range(reps):
        a = time.perf_counter()
        fn(batch_inputs)
        ts.append((time.perf_counter() - a) * 1000)
    return {"mean_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95)), "max_ms": float(np.max(ts))}


def run_stage1(rep: Path, out: Path):
    rng = np.random.default_rng(0)
    torch.set_num_threads(os.cpu_count() or 1)
    grid, norm, split, data, gru, aed, ae, thr, vent, v_ck = load_models(rep)
    S = samples(grid, norm, split, data, aed, v_ck, rng)
    models = {"gru": gru, "ae": ae, "vent": vent}
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    (out / "bench").mkdir(parents=True, exist_ok=True)
    rows, diffs, meta = [], [], {"models": {}}
    import onnxruntime as ort
    providers = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
    backends = ["pytorch_cpu"] + (["pytorch_gpu"] if torch.cuda.is_available() else []) + [f"onnxruntime:{p}" for p in providers]
    for name, model in models.items():
        if model is None or name not in S:
            continue
        inputs = S[name]["inputs"]
        path = out / "onnx" / f"{name}.onnx"
        export(model, inputs, path)
        with torch.no_grad():
            ref = model(*(torch.from_numpy(v) for v in inputs.values())).numpy()
        for n, v in inputs.items():
            v.astype("<f4").tofile(out / "bench" / f"{name}.{n}.bin")
        ref.astype("<f4").tofile(out / "bench" / f"{name}.ref.bin")
        info = {"onnx": str(path.relative_to(out)), "n": int(len(ref)),
                "inputs": {n: list(v.shape) for n, v in inputs.items()}, "out_shape": list(ref.shape)}
        if name == "gru":
            info["std"] = S["gru"]["std"].tolist()                         # [N, 2] co2·온도 표준편차 → 원래 단위
        if name == "vent":
            info["threshold"] = v_ck["threshold"]
        if name == "ae":
            S["ae"]["v"].astype("<f4").tofile(out / "bench" / "ae.v.bin")
            S["ae"]["m"].astype("<f4").tofile(out / "bench" / "ae.m.bin")
            info["locs"] = S["ae"]["loc"]
            info["thr"] = {f"{k[0]}|{k[1]}": v for k, v in thr.items()}
        meta["models"][name] = info
        params = sum(p.numel() for p in model.parameters())
        for be in backends:
            if be.startswith("pytorch"):
                dev = "cuda" if be == "pytorch_gpu" else "cpu"
                m = model.to(dev)

                def fn(bi, m=m, dev=dev):
                    with torch.no_grad():
                        r = m(*bi)
                        if dev == "cuda":
                            torch.cuda.synchronize()
                        return r
                mk = lambda b, dev=dev: tuple(torch.from_numpy(v[:b]).to(dev) for v in inputs.values())  # noqa: E731
                full_out = fn(mk(len(ref))).cpu().numpy()
                model.to("cpu")
            else:
                prov = be.split(":", 1)[1]
                try:
                    sess = ort.InferenceSession(str(path), providers=[prov])
                except Exception as e:  # noqa: BLE001
                    rows.append({"model": name, "backend": be, "batch": "-", "error": str(e)[:120]})
                    continue
                fn = lambda bi, sess=sess: sess.run(None, bi)  # noqa: E731
                mk = lambda b: {n: v[:b] for n, v in inputs.items()}  # noqa: E731
                full_out = fn(mk(len(ref)))[0]
            m0 = rss_mb()
            for b in (1, 4):
                rows.append({"model": name, "backend": be, "batch": b, "params": params, **bench(fn, mk(b)), "rss_mb": round(rss_mb(), 0)})
            diffs.append({"model": name, "backend": be, **compare(name, ref, full_out, S, thr, v_ck)})
    meta.update(platform=platform.platform(), torch=torch.__version__, onnxruntime=ort.__version__, providers=providers,
                warmup=WARMUP, reps=REPS, threads=torch.get_num_threads())
    (out / "bench" / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    pd.DataFrame(rows).to_csv(out / "latency_cpu.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diff_cpu.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(diffs).to_string())


def compare(name, ref, got, S, thr, v_ck) -> dict:
    """기준(PyTorch fp32)과의 차이. 원래 단위."""
    got = np.asarray(got, np.float64).reshape(ref.shape)
    if name == "gru":
        std = S["gru"]["std"][:, None, :]
        d = np.abs(got - ref) * std
        return {"co2_max_ppm": float(d[..., 0].max()), "co2_mean_ppm": float(d[..., 0].mean()),
                "temp_max_c": float(d[..., 1].max()), "flip_%": np.nan}
    if name == "ae":
        f0 = ae_flags(ref, S["ae"]["v"], S["ae"]["m"], S["ae"]["loc"], thr)
        f1 = ae_flags(got, S["ae"]["v"], S["ae"]["m"], S["ae"]["loc"], thr)
        return {"max_abs_norm": float(np.abs(got - ref).max()), "flip_%": float((f0 != f1).mean() * 100)}
    p0, p1 = 1 / (1 + np.exp(-ref)), 1 / (1 + np.exp(-got))
    t = v_ck["threshold"]
    return {"prob_max_diff": float(np.abs(p1 - p0).max()), "flip_%": float(((p0 >= t) != (p1 >= t)).mean() * 100)}


def report(out: Path):
    lat = pd.read_csv(out / "latency_cpu.csv")
    dif = pd.read_csv(out / "diff_cpu.csv")
    meta = json.loads((out / "bench" / "meta.json").read_text())
    trt = json.loads((out / "trt.json").read_text()) if (out / "trt.json").exists() else None
    if trt:
        lat = pd.concat([lat, pd.DataFrame(trt.get("latency", []))], ignore_index=True)
        dif = pd.concat([dif, pd.DataFrame(trt.get("diff", []))], ignore_index=True)

    def md(df):
        cols = list(df.columns)
        L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.itertuples(index=False):
            L.append("| " + " | ".join(("-" if pd.isna(v) else f"{v:.3f}") if isinstance(v, float) else str(v) for v in r) + " |")
        return L
    L = ["# 과제 7. 엣지 최적화\n", "## 측정 조건\n"]
    L.append(f"- 컨테이너: {meta['platform']}, torch {meta['torch']}, onnxruntime {meta['onnxruntime']} ({', '.join(meta['providers'])}), CPU 스레드 {meta['threads']}")
    L.append(f"- 반복: 워밍업 {meta['warmup']}회 뒤 {meta['reps']}회, 지연은 호출 1회(입력 준비 제외)")
    if trt:
        for k, v in trt.get("conditions", {}).items():
            L.append(f"- {k}: {v}")
    else:
        L.append("- TensorRT: 아직 측정 안 함(scripts/trt_bench.py 를 보드 호스트에서 실행)")
    L.append("\n## 지연 (ms)\n")
    cols = [c for c in ("model", "backend", "batch", "mean_ms", "p95_ms", "max_ms", "rss_mb", "gpu_load_%", "power_mw", "error") if c in lat]
    L += md(lat[cols])
    L.append("\n## 결과 차이 (기준: PyTorch CPU fp32, 시험 기간 표본)\n")
    L += md(dif)
    g = dif[(dif.model == "gru")]
    worst = g["co2_max_ppm"].max() if "co2_max_ppm" in g else np.nan
    flips = dif["flip_%"].max() if "flip_%" in dif else np.nan
    L.append(f"\n- 확인 기준: CO₂ 예측 최대 차이 {worst:.3f}ppm (1ppm 이하 {'통과' if worst <= 1 else '미달'}), "
             f"판정이 바뀐 비율 최대 {flips:.3f}% (0.1% 이하 {'통과' if flips <= 0.1 else '미달'})")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    d = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--reports", default=os.path.join(d, "reports"))
    ap.add_argument("--report", action="store_true", help="측정 없이 보고서만 다시 만든다(trt.json 합치기)")
    a = ap.parse_args()
    rep = Path(a.reports)
    out = rep / "task7"
    out.mkdir(parents=True, exist_ok=True)
    if not a.report:
        run_stage1(rep, out)
    report(out)


if __name__ == "__main__":
    main()
