"""상시 수집기: MQTT 구독 → 원본 보관 → 디스크 큐 → InfluxDB.

환경 변수
  MQTT_HOST, MQTT_PORT, MQTT_USER, MQTT_PASS
  INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET
  COLLECTOR_CONFIG (기본 /app/config/collector.yaml)
  DATA_DIR (기본 /data)  원본 메시지·디스크 큐 위치
  STATUS_PORT (기본 8090)
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import paho.mqtt.client as mqtt
import yaml
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from .parse import Parser, Record
from .spool import Spool
from .status import Status

log = logging.getLogger("collector")


def to_line(rec: Record) -> str:
    p = Point(rec.measurement)
    for k, v in rec.tags.items():
        p = p.tag(k, v)
    for k, v in rec.fields.items():
        p = p.field(k, v)
    return p.time(rec.ts_ns, WritePrecision.NS).to_line_protocol()


class RawArchive:
    """받은 메시지를 하루 단위 gzip JSONL 로 그대로 남긴다. 형식이 바뀌었을 때 다시 파싱할 수 있게."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()
        self._day = None
        self._fh = None

    def write(self, topic: str, payload: bytes, recv_ns: int) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(recv_ns / 1e9))
        line = json.dumps({"recv_ns": recv_ns, "topic": topic,
                           "payload": payload.decode("utf-8", "replace")}, ensure_ascii=False)
        with self._lock:
            if day != self._day:
                if self._fh:
                    self._fh.close()
                self._fh = gzip.open(os.path.join(self.root, f"{day}.jsonl.gz"), "at", encoding="utf-8")
                self._day = day
            self._fh.write(line + "\n")

    def flush(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.flush()


class Collector:
    def __init__(self, config: dict):
        self.cfg = config
        self.parser = Parser(config)
        data = os.environ.get("DATA_DIR", "/data")
        os.makedirs(data, exist_ok=True)
        self.spool = Spool(os.path.join(data, "spool.sqlite"))
        self.raw = RawArchive(os.path.join(data, "raw"))
        self.status = Status()
        self.stop = threading.Event()

    # ---- MQTT ----
    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            log.error("mqtt connect failed: %s", reason_code)
            return
        self.status.mqtt_connected = True
        qos = self.cfg.get("mqtt", {}).get("qos", 1)
        client.subscribe([(t, qos) for t in self.parser.topics()])
        log.info("mqtt connected, subscribed %s", self.parser.topics())

    def on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.status.mqtt_connected = False
        self.status.mqtt_reconnects += 1
        log.warning("mqtt disconnected: %s (재접속 시도)", reason_code)

    def on_message(self, client, userdata, msg):
        recv_ns = time.time_ns()
        st = self.status
        st.messages += 1
        st.last_message_at = recv_ns / 1e9
        try:
            self.raw.write(msg.topic, msg.payload, recv_ns)
            recs = self.parser.parse(msg.topic, msg.payload, recv_ns)
        except Exception as e:  # 한 메시지가 깨져도 수집은 계속
            st.errors += 1
            st.last_error = f"{msg.topic}: {e}"[:300]
            return
        if not recs:
            st.unmatched += 1
            return
        if st.is_duplicate(recs[0].dedup_key):
            return
        for r in recs:
            st.record(r)
        self.spool.put([to_line(r) for r in recs])

    # ---- InfluxDB 쓰기 ----
    def writer_loop(self):
        url, token = os.environ["INFLUX_URL"], os.environ["INFLUX_TOKEN"]
        org, bucket = os.environ["INFLUX_ORG"], os.environ["INFLUX_BUCKET"]
        backoff = 1.0
        client = InfluxDBClient(url=url, token=token, org=org, timeout=10_000)
        api = client.write_api(write_options=SYNCHRONOUS)
        while not self.stop.is_set():
            batch = self.spool.peek(5000)
            if not batch:
                self.stop.wait(1.0)
                continue
            try:
                api.write(bucket=bucket, org=org, record=[l for _, l in batch],
                          write_precision=WritePrecision.NS)
                self.spool.ack(batch[-1][0])
                self.status.last_write_ok_at = time.time()
                self.status.last_write_error = ""
                backoff = 1.0
            except Exception as e:
                self.status.last_write_error = str(e)[:300]
                log.warning("influx write failed (%d 대기): %s", len(batch), e)
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 60)
        client.close()

    # ---- 현황 HTTP ----
    def serve_status(self, port: int):
        col = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                snap = col.status.snapshot(col.parser.unmapped, col.spool.size())
                if self.path.startswith("/healthz"):
                    # 최근 5분 안에 메시지를 받았으면 정상
                    ok = snap["mqtt_connected"] and time.time() - col.status.last_message_at < 300
                    body, code = (b"ok\n", 200) if ok else (b"stale\n", 503)
                    self.send_response(code)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps(snap, ensure_ascii=False, indent=1).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def run(self):
        m = self.cfg.get("mqtt", {})
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id=m.get("client_id", "collector"), clean_session=False)
        client.username_pw_set(os.environ.get("MQTT_USER"), os.environ.get("MQTT_PASS"))
        client.on_connect, client.on_disconnect, client.on_message = (
            self.on_connect, self.on_disconnect, self.on_message)
        # 브로커가 재시작돼도 1~10초 간격으로 다시 붙는다 (확인 기준: 1분 안에 재개)
        client.reconnect_delay_set(min_delay=1, max_delay=10)

        threading.Thread(target=self.writer_loop, daemon=True).start()
        srv = self.serve_status(int(os.environ.get("STATUS_PORT", "8090")))

        def flusher():
            while not self.stop.wait(10):
                self.raw.flush()
        threading.Thread(target=flusher, daemon=True).start()

        def shutdown(*_):
            self.stop.set()
            client.disconnect()
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        host, port = os.environ["MQTT_HOST"], int(os.environ.get("MQTT_PORT", "1883"))
        client.connect_async(host, port, keepalive=m.get("keepalive", 30))
        client.loop_forever(retry_first_connection=True)
        srv.shutdown()
        self.raw.flush()


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    path = os.environ.get("COLLECTOR_CONFIG", "/app/config/collector.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    Collector(cfg).run()


if __name__ == "__main__":
    main()
