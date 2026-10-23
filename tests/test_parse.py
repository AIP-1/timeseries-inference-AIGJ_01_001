from pathlib import Path

import pytest
import yaml

from collector.parse import Parser, parse_time_ns, topic_matches
from collector.status import Status

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(__file__).parent / "samples"
CS_TOPIC = "application/96b4d719-f23f-40aa-9f94-a0f2d0354342/device/24e124126d152862/event/up"


@pytest.fixture
def parser():
    return Parser(yaml.safe_load((ROOT / "config/collector.yaml").read_text(encoding="utf-8")))


def sample(name):
    return (SAMPLES / name).read_bytes()


def ns(iso):
    return parse_time_ns(iso)


def test_topic_matches():
    assert topic_matches("application/+/device/+/event/up", CS_TOPIC)
    assert not topic_matches("application/+/device/+/event/up", CS_TOPIC + "/x")
    assert topic_matches("gateway/#", "gateway/a/state/conn")


def test_parse_time_nanoseconds_and_epoch_ms():
    assert ns("2026-10-20T01:37:32.638374020+00:00") % 1_000_000_000 == 638374020
    assert parse_time_ns(1792460252389, "ms") == ns("2026-10-20T01:37:32.389Z")


def test_chirpstack_uses_network_server_time_not_skewed_gateway_time(parser):
    recv = ns("2026-10-20T01:37:33Z")
    [r] = parser.parse(CS_TOPIC, sample("chirpstack_co2_skewed_gw.json"), recv)
    # 최상위 time(01:42:27)은 시계가 5분 빠른 게이트웨이 값. 가장 이른 nsTime 을 써야 한다.
    assert r.ts_ns == ns("2026-10-20T01:37:32.637934223+00:00")
    assert r.measurement == "env"
    assert r.tags == {"device": "24e124126d152862", "model": "EM500-CO2", "spot": "후방 우측", "space": "lab"}
    assert r.fields["co2"] == 887.0 and isinstance(r.fields["humidity"], float)
    assert r.fields["rssi"] == -53.0 and r.fields["fcnt"] == 104072.0
    assert r.dedup_key == "dfa609e9-5f77-4fcd-b8ed-603f5f035d73"


def test_clock_skew_falls_back_to_receive_time(parser):
    recv = ns("2026-10-20T03:00:00Z")  # 메시지 시각보다 80분 늦게 받음 → 허용치 10분 초과
    [r] = parser.parse(CS_TOPIC, sample("chirpstack_am107.json"), recv)
    assert r.ts_ns == recv and r.meta["ts_from"].startswith("skew:")


def test_xflow_hvac_explodes_unit_and_maps_space(parser):
    topic = "xflow/hvacr/idu/4a848cc0-8598-11f1-bcac-bdfb397bbe8a/status"
    [r] = parser.parse(topic, sample("xflow_idu.json"), parse_time_ns(1792460270000, "ms"))
    assert r.measurement == "hvac"
    assert r.tags["space"] == "ceo_room" and r.tags["unit"] == "idu" and r.tags["name"] == "대표실-실내기"
    assert r.fields["power"] is False and r.fields["target_temperature"] == 18.0
    assert r.ts_ns == parse_time_ns(1792460269207, "ms")


def test_string_fields_are_skipped(parser):
    [r] = parser.parse("xflow/controller/7c43/status", sample("xflow_controller.json"),
                       parse_time_ns(1792460253000, "ms"))
    assert "op_mode" not in r.fields and r.fields["compressor_cap"] == 4.0


def test_unmapped_space_is_reported(parser):
    body = sample("chirpstack_am107.json").replace("실습실".encode(), "새 공간".encode())
    [r] = parser.parse(CS_TOPIC, body, ns("2026-10-20T01:37:48Z"))
    assert r.tags["space"] == "새 공간" and "새 공간" in parser.unmapped


def test_gateway_raw_payload(parser):
    [r] = parser.parse("gateway/24e124fffef5dccc/state/conn", b"\x08\x01binary", 1)
    assert r.measurement == "gateway" and r.tags == {"gateway": "24e124fffef5dccc"}


def test_unknown_topic_ignored(parser):
    assert parser.parse("other/topic", b"{}", 1) == []


def test_broken_json_raises(parser):
    with pytest.raises(ValueError):
        parser.parse(CS_TOPIC, b"{not json", 1)


def test_duplicate_detection():
    st = Status()
    assert not st.is_duplicate("a")
    assert st.is_duplicate("a")
    assert not st.is_duplicate(None)
    assert st.duplicates == 1


def test_unset_board_clock_trusts_message_time(parser):
    recv = ns("1970-01-01T00:18:10Z")   # 재부팅 직후 NTP 전 보드 시계
    [r] = parser.parse(CS_TOPIC, sample("chirpstack_am107.json"), recv)
    assert r.ts_ns > ns("2024-01-01T00:00:00Z") and not r.meta["ts_from"].startswith("skew:")


def test_backfill_restores_message_time_from_raw_archive(parser, tmp_path):
    import gzip, json
    from collector.backfill import reparse
    p = tmp_path / "1970-01-01.jsonl.gz"
    msg = {"recv_ns": ns("1970-01-01T00:18:10Z"), "topic": CS_TOPIC,
           "payload": sample("chirpstack_am107.json").decode("utf-8")}
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n" + json.dumps(msg) + "\n")   # 같은 메시지 두 번 → 한 번만
    lines, n = reparse([str(p)], parser)
    assert n["messages"] == 2 and len(lines) == 1
    assert int(lines[0].rsplit(" ", 1)[1]) > ns("2024-01-01T00:00:00Z")


def test_backfill_keeps_messages_without_dedup_key(parser, tmp_path):
    import gzip, json
    from collector.backfill import reparse
    p = tmp_path / "1970-01-01.jsonl.gz"
    topic = "xflow/hvacr/idu/4a848cc0-8598-11f1-bcac-bdfb397bbe8a/status"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for _ in range(2):
            f.write(json.dumps({"recv_ns": ns("1970-01-01T00:18:10Z"), "topic": topic,
                                "payload": sample("xflow_idu.json").decode("utf-8")}) + "\n")
    _, n = reparse([str(p)], parser)
    assert n["hvac"] == 2
