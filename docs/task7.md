# 과제 7. 엣지 최적화

| 단계 | 어디서 | 명령 | 측정 |
|---|---|---|---|
| 1 | lab 컨테이너(CPU) | `docker compose run --rm lab python -m analysis.task7` | ONNX 내보내기, PyTorch CPU·ONNX Runtime 지연과 차이, TensorRT 용 입력·기준 출력 저장 |
| 2 | 보드 호스트 | `python3 scripts/trt_bench.py data/reports/task7` | trtexec FP16 엔진, 지연(평균·p95), 결과 차이, 전력(INA3221 VDD_IN)·GPU 사용률·메모리 |
| 3 | lab 컨테이너 | `docker compose run --rm lab python -m analysis.task7 --report` | 두 결과를 합친 report.md |

- 모델: GRU(60분 예측), 오토인코더(이상), MLP(환기 경보). 배치 1(한 공간)·4(네 공간).
- 차이: 기준은 PyTorch CPU fp32. GRU 는 ppm·℃ 로 되돌려 최대·평균, 분류는 판정이 바뀐 비율(이상 판정: 항목별 임계값, 환기: 검증 기간 임계값).
- PyTorch GPU: lab 이미지는 CPU 판 torch 라 측정하지 않는다. 보드 GPU 용 torch 는 JetPack 전용 휠이 필요하다(측정 조건에 '없음'으로 기록).
- 클럭 고정(jetson_clocks)은 sudo 가 필요해 하지 않았다. 전력 모드는 `nvpmodel -q` 결과를 기록한다.
