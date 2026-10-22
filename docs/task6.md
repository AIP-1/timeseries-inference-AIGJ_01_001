# 과제 6. 사전학습 모델 비교

실행: `docker compose build lab && docker compose run --rm lab python -m analysis.task6`
(처음 한 번 Hugging Face 에서 `amazon/chronos-bolt-small` 을 내려받아 `data/hf` 에 캐시)

- 비교 대상: 기준선 3종, 과제 3 GRU(`data/reports/task3/gru_full.pt`), Chronos-Bolt(과거 2시간·하루·약 43시간), GRU+Chronos 평균(B)
- 같은 시점: 과제 2 평가 시점 ∩ 시험 기간 ∩ 가장 긴 과거를 줄 수 있는 시각, 15분마다 하나
- 예측 구간: Chronos 10%·90% 분위 사이에 실제 값이 들어온 비율(이상적 80%)과 구간 폭
- 비용: 모델 불러오기 시간, 늘어난 메모리(RSS), 배치 1·4 지연(워밍업 3회, 30회 반복, 평균·p95)
- 산출물 `data/reports/task6/`: report.md, metrics.csv, coverage.csv, timing.csv, predictions_60.csv
