# 과제 1. 상시 수집기

## 구조

```
MQTT 브로커 ──> 수집기 ──┬─> data/raw/YYYY-MM-DD.jsonl.gz   (받은 메시지 원본, 다시 파싱용)
                         └─> data/spool.sqlite (디스크 큐) ──> InfluxDB
                 └─ :8090/status  수집 현황, :8090/healthz  상태 확인
```

- **설정만 바꿔 대응:** 토픽·시각·태그·필드 위치는 모두 `config/collector.yaml` 에 있다. 코드는 형식을 모른다.
- **끊김에 버팀:** 브로커가 재시작되면 1~10초 간격으로 다시 붙는다. 영구 세션(`clean_session=False`, QoS 1)이라 수집기가 잠시 꺼진 동안의 메시지도 브로커가 보관했다가 넘겨 준다(브로커가 세션 보관을 허용할 때). InfluxDB 가 죽으면 디스크 큐에 쌓아 두고 살아나면 보낸다.
- **재부팅 후 자동 시작:** `restart: unless-stopped`. sudo 없이 docker 그룹 권한만으로 된다.

## 저장 형식 (InfluxDB)

| measurement | 태그 | 필드 | 원천 |
|---|---|---|---|
| `env` | `space`, `device`(devEui), `model`, `spot` | `co2`, `temperature`, `humidity`, `pressure`, `tvoc`, `activity`, `LAeq`, `battery` … (장비마다 다름), `fcnt`, `rssi`, `snr`, `recv_lag_s` | ChirpStack 업링크 |
| `hvac` | `space`, `device`, `unit`(idu/odu), `name` | `power`, `current_temperature`, `target_temperature`, `mode`, `error_code` … | xflow 냉난방 |
| `hvac_controller` | `device`, `name` | `compressor_run`, `heat_demand` … | xflow 컨트롤러 |
| `gateway` | `gateway` | `seen` | 게이트웨이 접속 상태 |

- 숫자는 모두 float 로 저장한다. 같은 센서가 `59`와 `59.5`를 섞어 보내서 정수로 두면 InfluxDB 가 타입 충돌로 쓰기를 거부한다.
- 공간 ID: `lab`(실습실), `office`(사무실), `office_hall`(사무실 밖), `meeting`, `focus_meeting`, `ceo_room`, `dev_room`. 매핑에 없는 이름은 원문 그대로 저장되고 `/status` 의 `unmapped_spaces` 에 나온다.

## 시각: 무엇을 저장하나

**서버 수신 시각(`rxInfo[].nsTime` 중 가장 이른 값)을 저장한다.** 메시지 최상위 `time` 은 게이트웨이 시계를 따르는데, 게이트웨이 `24e124fffef5dccc` 는 약 4분 55초 빠르다(2026-10-20 확인). 그 값을 쓰면 해당 게이트웨이를 거친 측정이 5분 미래에 찍힌다. 센서 시각이 수신 시각과 10분 넘게 다르면 수신 시각을 쓰고 `/status` 의 `clock_skew_fallbacks` 를 올린다. `recv_lag_s`(수신 − 측정 시각)로 네트워크 지연을 나중에 볼 수 있다. 30초 밀렸다가 몰려 와도 측정 시각 기준으로 제자리에 들어간다.

## 중복(재전송)

ChirpStack 의 `deduplicationId` 를 최근 2만 건 기억해 같은 것이 다시 오면 버린다(`duplicates_dropped`). 수집기 재시작 직후에는 기억이 비지만, 같은 태그·같은 시각의 점은 InfluxDB 가 덮어쓰므로 저장 결과는 같다.

## 수집 현황 보는 법

```bash
curl -s localhost:8090/status            # 실시간(수집기 가동 후 기준)
docker compose run --rm collector python -m collector.report --hours 24   # InfluxDB 기준 24시간 커버율·결측 구간
```

## 메시지 형식이 바뀌었을 때 (산출물 예시)

예: 센서 업체가 `object.co2` 를 `object.co2_ppm` 으로 바꿨다. → `config/collector.yaml` 의 chirpstack 소스 `extra_fields` 에 `co2: object.co2_ppm` 한 줄을 추가하고 `docker compose restart collector` 한다. 코드와 이미지는 그대로다. 원본이 `data/raw/` 에 남아 있어 바뀐 시점 전후도 다시 파싱할 수 있다.

새 센서 종류(예: 다른 토픽의 JSON) → `sources:` 에 항목 하나를 추가한다.

## 확인 기준 시험 방법

| 기준 | 시험 |
|---|---|
| 브로커 재시작 후 1분 안에 재개 | 로컬 mosquitto 로 시험: 재시작 1초 뒤 재접속 확인. 실제 브로커는 끌 수 없으니 Jetson 네트워크를 잠시 끊고(`docker network disconnect`) `mqtt_reconnects`·`messages` 증가로 확인 |
| 재부팅 후 자동 재개 | `sudo reboot`(aip-1-3 계정) 후 `docker compose ps`, `/status` 의 `uptime_s` |
| 24시간 커버율 95% | 24시간 뒤 `collector.report --hours 24` |
