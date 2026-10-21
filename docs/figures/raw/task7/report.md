# 과제 7. 엣지 최적화

## 측정 조건

- 컨테이너: Linux-6.8.12-1021-tegra-aarch64-with-glibc2.41, torch 2.14.1+cpu, onnxruntime 1.31.0 (CPUExecutionProvider), CPU 스레드 6
- 반복: 워밍업 50회 뒤 500회, 지연은 호출 1회(입력 준비 제외)
- 보드: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- L4T: # R39 (release), REVISION: 2.1, GCID: 46758480, BOARD: generic, EABI: aarch64, DATE: Fri Aug  7 05:54:22 AM UTC 2026
- TensorRT: &&&& RUNNING TensorRT.trtexec [TensorRT v101602] [b10] # /usr/bin/trtexec --help
- 전력 모드: NV Power Mode: MAXN_SUPER 2
- 클럭 고정: 안 함(jetson_clocks 는 sudo 필요). 기본 DVFS 상태에서 측정
- 반복: trtexec 워밍업 2000ms, 1000회, spin wait. 지연=latencyMs(입출력 복사 포함), compute=GPU 계산만
- 대기 전력(VDD_IN, mW): 6798.9
- 전력 레일: ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']

## 지연 (ms)

| model | backend | batch | mean_ms | p95_ms | max_ms | rss_mb | gpu_load_% | power_mw |
|---|---|---|---|---|---|---|---|---|
| gru | pytorch_cpu | 1 | 17.396 | 67.467 | 113.359 | 696.000 | - | - |
| gru | pytorch_cpu | 4 | 13.582 | 42.692 | 67.521 | 696.000 | - | - |
| gru | onnxruntime:CPUExecutionProvider | 1 | 0.214 | 0.231 | 0.305 | 715.000 | - | - |
| gru | onnxruntime:CPUExecutionProvider | 4 | 0.523 | 0.542 | 2.056 | 715.000 | - | - |
| ae | pytorch_cpu | 1 | 0.697 | 0.825 | 2.004 | 715.000 | - | - |
| ae | pytorch_cpu | 4 | 0.640 | 0.949 | 4.259 | 715.000 | - | - |
| ae | onnxruntime:CPUExecutionProvider | 1 | 0.025 | 0.026 | 0.167 | 716.000 | - | - |
| ae | onnxruntime:CPUExecutionProvider | 4 | 0.027 | 0.028 | 0.072 | 716.000 | - | - |
| vent | pytorch_cpu | 1 | 0.541 | 1.500 | 2.993 | 716.000 | - | - |
| vent | pytorch_cpu | 4 | 0.456 | 0.512 | 1.532 | 716.000 | - | - |
| vent | onnxruntime:CPUExecutionProvider | 1 | 0.041 | 0.025 | 4.064 | 716.000 | - | - |
| vent | onnxruntime:CPUExecutionProvider | 4 | 0.025 | 0.025 | 0.054 | 716.000 | - | - |
| gru | tensorrt_fp16 | 1 | 0.754 | 0.836 | 1.542 | - | 60.900 | 7827.800 |
| gru | tensorrt_fp16 | 4 | 0.733 | 0.761 | 0.791 | - | 64.200 | 7554.300 |
| ae | tensorrt_fp16 | 1 | 0.066 | 0.072 | 0.163 | - | 38.700 | 7161.800 |
| ae | tensorrt_fp16 | 4 | 0.067 | 0.079 | 0.123 | - | 38.400 | 7689.400 |
| vent | tensorrt_fp16 | 1 | 0.064 | 0.069 | 0.091 | - | 32.400 | 7734.800 |
| vent | tensorrt_fp16 | 4 | 0.065 | 0.072 | 0.106 | - | 32.200 | 7142.500 |

## 결과 차이 (기준: PyTorch CPU fp32, 시험 기간 표본)

| model | backend | co2_max_ppm | co2_mean_ppm | temp_max_c | flip_% | max_abs_norm | prob_max_diff |
|---|---|---|---|---|---|---|---|
| gru | pytorch_cpu | 0.000 | 0.000 | 0.000 | - | - | - |
| gru | onnxruntime:CPUExecutionProvider | 0.000 | 0.000 | 0.000 | - | - | - |
| ae | pytorch_cpu | - | - | - | 0.000 | 0.000 | - |
| ae | onnxruntime:CPUExecutionProvider | - | - | - | 0.000 | 0.000 | - |
| vent | pytorch_cpu | - | - | - | 0.000 | - | 0.000 |
| vent | onnxruntime:CPUExecutionProvider | - | - | - | 0.000 | - | 0.000 |
| gru | tensorrt_fp16 | 0.462 | 0.028 | 0.003 | - | - | - |
| ae | tensorrt_fp16 | - | - | - | 0.000 | 0.001 | - |
| vent | tensorrt_fp16 | - | - | - | 0.000 | - | 0.001 |

- 확인 기준: CO₂ 예측 최대 차이 0.462ppm (1ppm 이하 통과), 판정이 바뀐 비율 최대 0.000% (0.1% 이하 통과)