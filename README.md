# jetson_AIGJ_01_001 — 시계열 데이터 추론 실습

과제 안내는 [ASSIGNMENT.md](ASSIGNMENT.md). 공간 구성은 강사 확인에 따라 **실제 센서 데이터 기준**(실습실·사무실 등)으로 한다.

| 구성 | 위치 | 상태 |
|---|---|---|
| 과제 1 수집기 | `collector/`, `config/collector.yaml` | [docs/collector.md](docs/collector.md) |

## 빠른 시작 (Jetson)

```bash
git clone https://github.com/AIP-1-3/jetson_AIGJ_01_001 ~/jetson_AIGJ_01_001
cd ~/jetson_AIGJ_01_001
cp ~/ts/.env .env        # 또는 .env.example 을 복사해 채운다
mkdir -p data            # 먼저 만들어 둬야 컨테이너가 내 계정 권한으로 쓸 수 있다
docker compose up -d --build
curl -s localhost:8090/status | head -30
```
