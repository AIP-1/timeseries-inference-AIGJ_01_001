# 과제 8. 상시 추론 서비스

- 시작: `docker compose up -d mqtt infer` (재부팅 뒤 자동 시작, restart: unless-stopped)
- 결과 형식: [service-api.md](service-api.md)
- 가동 기록 요약: `docker compose run --rm lab python -m analysis.service_report --since <시작 시각>` → data/reports/task8/report.md
- 필요한 학습 산출물: data/reports/task2(격자 구성·정규화), task3/gru_full.pt, task4/ae.pt·thresholds.csv, task5/vent.pt

## 장애 시험 방법
1. 12시간 무중단: 서비스 시작 시각을 적어 두고 12시간 뒤 service_report 로 실패·빈 구간 확인.
2. 데이터 끊김: `docker compose stop collector` → 10분 기다림 → `/status` 의 active_alerts 에 data_gap → `docker compose start collector` → 다음 몇 분 안에 cleared. service_report 의 '데이터 끊김 경보' 절에 시각이 남는다.
3. 계산 실패 회복: 주기 하나가 예외로 끝나도 다음 분에 다시 계산(cycles 기록의 ok=false 다음 줄이 ok=true).
4. 재부팅: 관리자 계정으로 재부팅 뒤 `docker compose ps` 와 `/healthz`.

## 가정
- 일정 자료가 없어 사용 여부는 규칙(15분 CO₂ +30ppm)으로 낸다. 일정이 생기면 과제 5 모델로 바꾼다.
- 공휴일 달력은 analysis/prep.py 의 KR_HOLIDAYS(2025-10 ~ 2026-12). 해가 바뀌면 추가해야 한다.
