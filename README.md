# jetson_AIGJ_01_001 — 시계열 데이터 추론 실습

과제 안내는 [ASSIGNMENT.md](ASSIGNMENT.md). 공간 구성은 강사 확인에 따라 **실제 센서 데이터 기준**(실습실·사무실 등)으로 한다.

데이터 저장소는 두 곳이다.
- **강사 InfluxDB `iot-bucket`** (10.116.64.13, 읽기만): 다른 수집기가 2025-09 이전부터 쌓아 온 실데이터. 과제 2~7 학습·평가용. measurement = 항목 이름(`co2` 등), 필드 `value`, 태그 `device_id`·`location`·`spot`.
- **Jetson 로컬 InfluxDB `sensors`** (127.0.0.1:8086): 과제 1 우리 수집기가 쓰는 곳. 과제 8 서비스가 읽는 곳.

| 구성 | 위치 | 상태 |
|---|---|---|
| 과제 1 수집기 | `collector/`, `config/collector.yaml` | [docs/collector.md](docs/collector.md) |

## 빠른 시작 (Jetson)

```bash
git clone https://github.com/AIP-1-3/jetson_AIGJ_01_001 ~/jetson_AIGJ_01_001
cd ~/jetson_AIGJ_01_001
cp ~/ts/.env .env        # MQTT·강사 InfluxDB 값 (없으면 .env.example 을 복사해 채운다)
echo "LOCAL_INFLUX_TOKEN=$(openssl rand -hex 32)" >> .env
echo "LOCAL_INFLUX_PASSWORD=$(openssl rand -hex 16)" >> .env
echo "HOST_UID=$(id -u)" >> .env; echo "HOST_GID=$(id -g)" >> .env
mkdir -p data            # 먼저 만들어 둬야 컨테이너가 내 계정 권한으로 쓸 수 있다
docker compose up -d --build
curl -s localhost:8090/status | head -30
```
