# 과제 3 보완. 검증 기간으로 고른 예측 (시험 기간 성적)

- 후보: GRU 변형 full, no_calendar, 현재값 유지, GRU·현재값 가중 평균(w=0.25, 0.5, 0.75). 공간·항목·시점마다 **검증 기간** MAE 가 가장 작은 것을 고르고, 시험 기간에는 한 번만 평가.
- 60분 CO₂: 현재값 유지 대비 22.9% 감소 → 미달 (30%)
- 60분 온도: 현재값 유지 대비 23.8% 감소 → 통과

## co2 공간 합산 MAE

| model | 5분 | 15분 | 30분 | 60분 |
|---|---|---|---|---|
| gru_full | 7.7 | 13.8 | 19.7 | 30.9 |
| gru_no_calendar | 7.6 | 13.2 | 19.4 | 32.1 |
| persistence | 7.8 | 15.6 | 23.8 | 39.1 |
| selected | 7.4 | 13.2 | 19.4 | 30.2 |

## temperature 공간 합산 MAE

| model | 5분 | 15분 | 30분 | 60분 |
|---|---|---|---|---|
| gru_full | 0.046 | 0.066 | 0.069 | 0.174 |
| gru_no_calendar | 0.031 | 0.042 | 0.073 | 0.123 |
| persistence | 0.022 | 0.047 | 0.084 | 0.155 |
| selected | 0.021 | 0.042 | 0.071 | 0.118 |

## 고른 결과 (60분)

| location | field | 고른 예측 | 시험 MAE | 현재값 유지 | 개선 % |
|---|---|---|---|---|---|
| 사무실 | co2 | gru_full | 27.692 | 37.583 | 26.3 |
| 사무실 | temperature | gru_full*0.75+persistence | 0.055 | 0.069 | 20.6 |
| 사무실 밖 | temperature | gru_no_calendar | 0.216 | 0.270 | 19.9 |
| 실습실 | co2 | gru_full*0.75+persistence | 19.305 | 24.378 | 20.8 |
| 실습실 | temperature | gru_full*0.75+persistence | 0.078 | 0.133 | 40.9 |
| 회의실 | co2 | gru_full*0.75+persistence | 43.597 | 55.569 | 21.5 |
| 회의실 | temperature | gru_full*0.5+persistence | 0.122 | 0.147 | 17.0 |