# 과제 5. 상태 추정

실행: `docker compose run --rm lab python -m analysis.task5`
일정·관찰 기록이 생기면: `... analysis.task5 --schedule /data/schedule.csv --observed /data/observed.csv`
(schedule.csv: location,start,end / observed.csv: location,start,end,state — 시각은 KST)

## 환기 선제 경보
- 양성 = 지금 CO₂ < 1000ppm 이고 30분 안에 1000ppm 이상. 이미 1000 이상인 시각은 평가에서 뺀다.
  넣으면 "지금 1000 이상 → 경보"가 쉽게 맞혀 점수가 부풀고, 이미 넘은 뒤의 경보는 '선제'가 아니다.
- 입력: 최근 1시간 CO₂(1000ppm 까지 거리), 5·15·30분 변화, 온도·습도와 30분 변화, 시각·쉬는 날, 공간.
- 불균형: 양성 가중치(BCE pos_weight). 임계값은 검증 기간에서 재현율 85% 이상 중 macro-F1 최대.
- 비교: 과제의 규칙(CO₂≥850), 같은 방식으로 검증 기간에서 고른 규칙, '항상 아님'(정확도가 무의미함을 보이는 용도).

## 사용 여부
- 정답은 일정에서 만들고 직접 관찰로 검증. 일정은 입력에 넣지 않는다(정답을 그대로 베끼게 됨).
- 센서만 쓴 모델과 시각까지 쓴 모델을 함께 낸다. 시간표가 일정하면 시각만으로 일정을 맞히므로 센서의 기여는 '센서만'으로 본다.
- 빈 방을 사용중으로 본 비율(empty_as_occupied_rate)을 따로 낸다.
- 일정이 아직 없어 지금은 규칙 결과(occupancy_rule.csv)만 남긴다.
