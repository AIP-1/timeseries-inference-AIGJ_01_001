# 과제 3. 60분 예측

실행 (보드, 과제 2 결과가 `data/reports/task2` 에 있어야 함):

```
docker compose build lab
docker compose run --rm lab python -m analysis.task3
# 일정 자료가 생기면: data/schedule.csv (location,start,end, KST) 를 두고
docker compose run --rm lab python -m analysis.task3 --schedule /data/schedule.csv
```

산출물 `data/reports/task3/`: report.md(비교표·공간별·정보 제거 비교), metrics.csv, train_log.csv, co2_err_by_hour.csv, gru_<변형>.pt

## 설계 결정

| 항목 | 선택 | 이유 |
|---|---|---|
| 과거 길이 | 2시간(24칸) | 수업·회의 한 단위(1~2시간)의 상승 추세를 담는다. 하루치는 시각·요일 특징으로 대신하고 계산량을 줄인다. `--lookback` 으로 바꿔 비교 가능(B) |
| 입력 | CO₂·온도·습도 + 결측 표시, 시각·요일 sin/cos, 주말, 공간 one-hot | 결측을 0으로만 채우면 '평균값'과 구별이 안 되므로 표시를 같이 넣는다 |
| 일정 | 지금 칸은 시계열 입력, 앞으로 60분은 출력 직전에 붙임 | 일정은 미래에도 미리 아는 정보라 과거 창 안에만 두면 '곧 수업 시작'을 알 수 없다 |
| 출력 | 12칸 변화량을 한 번에 | 반복 예측은 오차가 누적된다. 변화량으로 두면 모델이 '지금 값 유지'에서 출발하므로 기준선보다 나빠지기 어렵다(`--variants absolute` 로 비교) |
| 모델 | GRU 2층 64, 모든 공간 한 모델 | 공간별 자료가 1년치라 공간마다 따로 두면 표본이 적다. 공간 차이는 one-hot 과 공간별 정규화로 흡수 |
| 손실 | 정규화 단위 MAE, 결측 정답은 제외 | 평가 지표와 같게. CO₂와 온도 단위 차이는 공간별 표준편차로 맞춘다 |
| 누수 방지 | 학습·검증 표본은 정답 60분까지 같은 구간 안. 정규화는 학습 기간 통계(과제 2 norm.csv) | 시험 기간은 평가에만 |
| 평가 | 과제 2 기준선과 같은 시점(baselines.eval_points ∩ 시험) | 같은 조건 비교 |

## 일정 정보

실데이터에는 일정이 없다. 지금은 `full`(시각·요일 포함)과 `no_calendar` 를 비교해 '달력 정보'의 가치만 잰다.
일정 CSV 를 받으면 `--schedule` 로 `full` 과 `no_schedule` 을 비교해 "일정 정보를 넣으면 60분 오차가 ○ppm 줄어든다"를 채운다.
