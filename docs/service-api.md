# 추론 서비스 v1 결과 형식

팀 Jetson(`aip-1-3`, Tailscale 주소 `<Jetson IP>`)에서 매분 공간마다 **60분 예측·이상·상태**를 계산해 두 가지로 내보낸다.

| 방식 | 주소 | 용도 |
|---|---|---|
| MQTT (읽기 전용) | `<Jetson IP>:1884`, 토픽 `aigj/01-001/inference/<space>/state` (retained) | 대시보드·제어 시스템이 계속 받기. 새로 구독해도 최신값을 바로 받음 |
| MQTT 경보 | `aigj/01-001/inference/<space>/alert` (retained 아님) | 경보가 켜질 때·켜져 있는 동안 30분마다·꺼질 때 한 번씩 |
| MQTT 서비스 상태 | `aigj/01-001/inference/_service/status` (retained) | 서비스가 살아 있는지 |
| 조회 API | `http://<Jetson IP>:8091` | 필요할 때 묻기(LLM 에이전트 도구 등) |

`<space>` 키: `lab`(실습실), `office`(사무실), `office_hall`(사무실 밖), `meeting`(회의실), `storage`(창고). 수집기 설정(spaces.by_name)에 없는 공간은 한글 이름 그대로.
API 경로에는 키와 한글 이름 둘 다 쓸 수 있다.

## 조회 API

| 경로 | 답 |
|---|---|
| `GET /spaces` | 공간 목록 `[{space, key}]` |
| `GET /spaces/meeting` | 그 공간의 전체 결과(아래 state 와 같은 JSON) |
| `GET /spaces/meeting/forecast?minutes=30` | 30분 뒤(가장 가까운 5분 칸) 예측 하나 |
| `GET /spaces/meeting/anomaly` · `/state` | 이상 · 상태 부분만 |
| `GET /status` | 주기 수, 마지막 성공 시각, 마지막 오류, 켜진 경보 |
| `GET /healthz` | 마지막 성공이 5분 안이면 200, 아니면 503 |

예: "회의실 30분 뒤 CO₂는?" → `curl 'http://<Jetson IP>:8091/spaces/meeting/forecast?minutes=30'`
```json
{"space": "회의실", "asked_minutes": 30, "minutes": 30, "time": "2026-10-21T01:30:00+00:00",
 "co2_ppm": 812.4, "temperature_c": 24.1, "time_basis": "2026-10-21T01:00:00+00:00"}
```

## state JSON

```json
{
 "space": "회의실",
 "time": "2026-10-21T01:00:00+00:00",          // 판단 기준 5분 칸의 시작(UTC). 끝난 칸만 쓴다
 "computed_at": "2026-10-21T01:06:05+00:00",   // 계산한 시각
 "version": "v1",
 "current": {"co2": 702.5, "temperature": 23.8, "humidity": 51.2},   // 그 5분 칸 평균(ppm, ℃, %)
 "last_data_at": "2026-10-21T01:05:40+00:00",  // 이 공간 원자료의 마지막 시각
 "forecast": {                                  // 데이터가 끊겼으면 null
   "steps": [{"minutes": 5, "time": "...", "co2_ppm": 715.0, "temperature_c": 23.8}, "... 12칸(60분)"],
   "co2_over_1000_in_min": 45                   // 예측상 1000ppm 을 넘는 첫 시점(분). 안 넘으면 null, 이미 넘었으면 0
 },
 "anomaly": {
   "score": 0.42,                 // 항목별 (오토인코더 복원 오차 / 임계값) 중 최댓값. 1 넘으면 이상
   "is_anomaly": false,
   "cause_field": null,           // 이상이면 가장 큰 항목(co2|temperature|humidity)
   "field_scores": {"co2": 0.42, "temperature": 0.10, "humidity": 0.21},
   "paused": false,               // 데이터가 끊긴 동안과 다시 들어온 뒤 1시간은 true: 창에 빈 칸이 섞여 이상 판정을 멈춘다(is_anomaly=false)
   "rules": []                    // 규칙 경보: "data_gap", "stuck:co2", "spike:temperature", "night:co2"(새벽 기준선), "rise:co2"(평소보다 크게 오름: 환기 정지 의심), "out_of_range:co2"
 },
 "state": {                       // CO₂ 센서가 있는 공간만
   "occupied": true,              // 최근 15분 CO₂ 30ppm 이상 상승(일정 자료가 없어 규칙)
   "occupied_method": "rule_co2_rise_15min_30ppm",
   "vent_alert": true,            // 30분 안에 1000ppm 을 넘을 것으로 판단(과제 5 분류기). 이미 넘었으면 true
   "vent_prob": 0.91
 }
}
```

## 경보 JSON (`.../alert`)

```json
{"space": "회의실", "alert": "data_gap", "status": "raised", "time": "2026-10-21T01:06:05+00:00"}
```
- `alert`: `data_gap`(10분 넘게 새 값 없음), `stuck:<항목>`, `spike:<항목>`, `night:co2`, `rise:co2`, `out_of_range:<항목>`, `anomaly:<항목>`, `vent_soon`, `co2_over_1000_forecast`
- `status`: `raised`(켜짐) → `ongoing`(켜져 있음, 30분마다) → `cleared`(꺼짐). 매분 같은 경보를 보내지 않는다.

## 처리 방식(학습과 같음)

- 원자료: 우리 수집기가 쓰는 보드 안 InfluxDB. 로컬에 일주일치가 쌓이기 전에는 그 앞부분만 강사 `iot-bucket` 에서 읽는다(쓰지 않음).
- 전처리: 학습과 **같은 함수**(`analysis/prep.py` 의 `clean`, `to_grid`). 범위 밖 값 제거 → 장비별 5분 칸 → 공간 평균 → 15분 이하 결측 보간.
- 막 시작된 5분 칸은 버리고 **끝난 칸까지만** 쓴다(1~2분치 평균은 대표성이 없고, 학습 때도 그런 칸은 비웠다).
- 학습과 전처리가 다르면 모델이 본 적 없는 분포(예: 1분 평균 vs 5분 평균의 잡음 크기, 센서 섞임 톱니)를 받아 예측·이상 점수가 조용히 틀어진다. 그래서 별도 구현을 두지 않고 같은 모듈을 import 한다.

## 장애 대응

- 계산이 한 번 실패해도 기록만 남기고 다음 분에 처음부터 다시 한다(`/status` 의 `errors`, `last_error`).
- 일부 공간 데이터가 끊기면 그 공간만 `forecast: null` + `data_gap` 경보, 나머지는 정상 계산. 그 공간의 이상 판정은 끊긴 동안과 복구 뒤 1시간 멈춘다(`anomaly.paused`). 2026-10-21 장애 시험에서 끊김이 섞인 창 때문에 실습실 온도 헛경보가 난 것을 보고 넣었다.
- 보드 재부팅: Docker `restart: unless-stopped` 로 자동 시작. 헬스체크가 `/healthz` 를 1분마다 확인.
- 주기 기록: `data/service/cycles-YYYYMMDD.jsonl` → `python -m analysis.service_report` 로 요약.
