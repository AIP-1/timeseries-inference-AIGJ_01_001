#!/usr/bin/env python3
"""과제 7 2단계: 보드 호스트에서 TensorRT(FP16) 측정. 표준 라이브러리만 쓴다(호스트에 pandas·torch 가 없어도 됨).

  python3 scripts/trt_bench.py data/reports/task7

필요: JetPack 의 trtexec (/usr/src/tensorrt/bin/trtexec). sudo 필요 없음.
입력: analysis.task7 1단계가 만든 onnx/*.onnx, bench/*.bin, bench/meta.json
출력: trt.json (지연, 결과 차이, 전력·GPU 사용률, 측정 조건) → python -m analysis.task7 --report 가 합친다.
"""
from __future__ import annotations

import array
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

WARMUP_MS, ITER = 2000, 1000


def trtexec() -> str:
    for p in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"):
        if p and os.path.exists(p):
            return p
    sys.exit("trtexec 를 찾을 수 없음 (JetPack TensorRT 설치 확인)")


def read_f32(path: Path) -> list[float]:
    a = array.array("f")
    a.frombytes(path.read_bytes())
    return a.tolist()


class Sampler(threading.Thread):
    """측정 중 전력(INA3221, mW)·GPU 사용률(%)·사용 가능 메모리(MB)를 0.1초마다 읽는다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rails = []
        for d in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
            for lab in glob.glob(f"{d}/in*_label"):
                n = lab.split("/")[-1][2:-6]
                v, c = f"{d}/in{n}_input", f"{d}/curr{n}_input"
                if os.path.exists(v) and os.path.exists(c):
                    self.rails.append((open(lab).read().strip(), v, c))
        loads = glob.glob("/sys/devices/platform/bus@0/*.gpu/load") + glob.glob("/sys/devices/platform/*.gpu/load") + glob.glob("/sys/devices/gpu.0/load")
        self.gpu = loads[0] if loads else None
        self.samples = []
        self.stop = threading.Event()

    def read(self):
        s = {"t": time.time()}
        for name, v, c in self.rails:
            try:
                s[name] = int(open(v).read()) * int(open(c).read()) / 1000
            except (OSError, ValueError):
                pass
        if self.gpu:
            try:
                s["gpu_load_%"] = int(open(self.gpu).read()) / 10
            except (OSError, ValueError):
                pass
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                s["mem_avail_mb"] = int(line.split()[1]) / 1024
        return s

    def run(self):
        while not self.stop.is_set():
            self.samples.append(self.read())
            time.sleep(0.1)

    def summary(self, since: float) -> dict:
        xs = [s for s in self.samples if s["t"] >= since]
        out = {}
        if not xs:
            return out
        keys = {k for s in xs for k in s if k != "t"}
        for k in sorted(keys):
            vals = [s[k] for s in xs if k in s]
            if k == "mem_avail_mb":
                out["mem_avail_min_mb"] = round(min(vals))
            else:
                out[f"{k}_mean" if k != "gpu_load_%" else "gpu_load_%"] = round(statistics.mean(vals), 1)
        return out


def shapes(info, b):
    return ",".join(f"{n}:{'x'.join(str(d) for d in [b] + s[1:])}" for n, s in info["inputs"].items())


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/reports/task7")
    meta = json.loads((out / "bench" / "meta.json").read_text())
    exe = trtexec()
    eng_dir = out / "trt"
    eng_dir.mkdir(exist_ok=True)
    sampler = Sampler()
    sampler.start()
    idle = sampler.read()
    time.sleep(2)
    idle_summary = sampler.summary(time.time() - 2)
    latency, diff = [], []
    for name, info in meta["models"].items():
        onnx = out / info["onnx"]
        n = info["n"]
        eng = eng_dir / f"{name}_fp16.plan"
        build = [exe, f"--onnx={onnx}", "--fp16", f"--saveEngine={eng}",
                 f"--minShapes={shapes(info, 1)}", f"--optShapes={shapes(info, 4)}", f"--maxShapes={shapes(info, n)}"]
        t0 = time.time()
        r = subprocess.run(build, capture_output=True, text=True)
        if r.returncode != 0:
            latency.append({"model": name, "backend": "tensorrt_fp16", "batch": "-", "error": r.stdout[-300:] + r.stderr[-300:]})
            continue
        build_s = time.time() - t0
        for b in (1, 4):
            times = eng_dir / f"{name}_b{b}_times.json"
            since = time.time()
            r = subprocess.run([exe, f"--loadEngine={eng}", f"--shapes={shapes(info, b)}", f"--warmUp={WARMUP_MS}",
                                f"--iterations={ITER}", "--duration=0", f"--exportTimes={times}", "--useSpinWait"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                latency.append({"model": name, "backend": "tensorrt_fp16", "batch": b, "error": r.stdout[-300:]})
                continue
            t = json.loads(times.read_text())
            lat = sorted(x["latencyMs"] for x in t)
            comp = sorted(x["computeMs"] for x in t)
            p95 = lambda xs: xs[min(len(xs) - 1, int(math.ceil(0.95 * len(xs))) - 1)]  # noqa: E731
            power = sampler.summary(since)
            latency.append({"model": name, "backend": "tensorrt_fp16", "batch": b, "mean_ms": statistics.mean(lat), "p95_ms": p95(lat),
                            "max_ms": lat[-1], "compute_mean_ms": statistics.mean(comp), "build_s": round(build_s, 1),
                            "power_mw": power.get("VDD_IN_mean"), "gpu_load_%": power.get("gpu_load_%"),
                            "mem_avail_min_mb": power.get("mem_avail_min_mb"), "engine_mb": round(eng.stat().st_size / 2 ** 20, 2)})
        # 결과 차이: 같은 입력 n 개를 넣고 fp32 PyTorch 기준 출력과 비교
        loads = ",".join(f"{k}:{out / 'bench' / f'{name}.{k}.bin'}" for k in info["inputs"])
        exp = eng_dir / f"{name}_out.json"
        r = subprocess.run([exe, f"--loadEngine={eng}", f"--shapes={shapes(info, n)}", f"--loadInputs={loads}",
                            "--iterations=1", "--warmUp=0", "--duration=0", f"--exportOutput={exp}"], capture_output=True, text=True)
        if r.returncode != 0:
            diff.append({"model": name, "backend": "tensorrt_fp16", "error": r.stdout[-300:]})
            continue
        got = json.loads(exp.read_text())[0]["values"]
        ref = read_f32(out / "bench" / f"{name}.ref.bin")
        diff.append({"model": name, "backend": "tensorrt_fp16", **compare(name, info, ref, got, out)})
    sampler.stop.set()
    cond = {"보드": open("/proc/device-tree/model").read().strip("\x00") if os.path.exists("/proc/device-tree/model") else "?",
            "L4T": open("/etc/nv_tegra_release").readline().strip() if os.path.exists("/etc/nv_tegra_release") else "?",
            "TensorRT": subprocess.run([exe, "--help"], capture_output=True, text=True).stdout.splitlines()[0][:80] if exe else "?",
            "전력 모드": _nvpmodel(), "클럭 고정": "안 함(jetson_clocks 는 sudo 필요). 기본 DVFS 상태에서 측정",
            "반복": f"trtexec 워밍업 {WARMUP_MS}ms, {ITER}회, spin wait. 지연=latencyMs(입출력 복사 포함), compute=GPU 계산만",
            "대기 전력(VDD_IN, mW)": idle_summary.get("VDD_IN_mean"), "전력 레일": [r[0] for r in sampler.rails]}
    (out / "trt.json").write_text(json.dumps({"latency": latency, "diff": diff, "conditions": cond}, ensure_ascii=False, indent=1))
    print(json.dumps({"latency": latency, "diff": diff, "conditions": cond}, ensure_ascii=False, indent=1))


def _nvpmodel():
    try:
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10)
        return " ".join(r.stdout.split())[:120] or r.stderr.strip()[:120]
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"확인 못 함({e})"


def compare(name, info, ref, got, out: Path) -> dict:
    n = info["n"]
    if name == "gru":
        std = info["std"]                       # [n][2]
        H = info["out_shape"][1]
        co2 = [abs(got[(i * H + k) * 2] - ref[(i * H + k) * 2]) * std[i][0] for i in range(n) for k in range(H)]
        tmp = [abs(got[(i * H + k) * 2 + 1] - ref[(i * H + k) * 2 + 1]) * std[i][1] for i in range(n) for k in range(H)]
        return {"co2_max_ppm": max(co2), "co2_mean_ppm": statistics.mean(co2), "temp_max_c": max(tmp)}
    if name == "vent":
        t = info["threshold"]
        sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
        p0, p1 = [sig(x) for x in ref], [sig(x) for x in got]
        return {"prob_max_diff": max(abs(a - b) for a, b in zip(p0, p1)),
                "flip_%": sum((a >= t) != (b >= t) for a, b in zip(p0, p1)) / n * 100}
    v, m = read_f32(out / "bench" / "ae.v.bin"), read_f32(out / "bench" / "ae.m.bin")
    W = info["out_shape"][1] // 3
    fields = ("co2", "temperature", "humidity")
    flips = total = 0
    for i in range(n):
        for j in range(3):
            idx = [i * W * 3 + w * 3 + j for w in range(W)]
            ms = sum(m[x] for x in idx)
            key = f"{info['locs'][i]}|{fields[j]}"
            if ms == 0 or key not in info["thr"]:
                continue
            e0 = sum(abs(ref[x] - v[x]) * m[x] for x in idx) / ms
            e1 = sum(abs(got[x] - v[x]) * m[x] for x in idx) / ms
            flips += (e0 > info["thr"][key]) != (e1 > info["thr"][key])
            total += 1
    return {"max_abs_norm": max(abs(a - b) for a, b in zip(ref, got)), "flip_%": flips / max(total, 1) * 100}


if __name__ == "__main__":
    main()
