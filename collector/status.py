"""수집 현황: 공간·항목별 건수, 마지막 수신 시각, 1분 커버율(메모리 기준, 재시작 시 초기화)."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, defaultdict

from .parse import Record

WINDOW_MIN = 24 * 60


class Status:
    def __init__(self, dedup_size: int = 20000):
        self._lock = threading.Lock()
        self.started = time.time()
        self.counts: dict[tuple, int] = defaultdict(int)
        self.last_ts: dict[tuple, float] = {}
        self.minutes: dict[tuple, set[int]] = defaultdict(set)
        self.devices: dict[str, dict] = {}
        self.messages = 0
        self.unmatched = 0
        self.errors = 0
        self.last_error = ""
        self.duplicates = 0
        self.skew_fallbacks = 0
        self.mqtt_connected = False
        self.mqtt_reconnects = 0
        self.last_message_at = 0.0
        self.last_write_ok_at = 0.0
        self.last_write_error = ""
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._dedup_size = dedup_size

    def is_duplicate(self, key: str | None) -> bool:
        if not key:
            return False
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                self.duplicates += 1
                return True
            self._seen[key] = None
            if len(self._seen) > self._dedup_size:
                self._seen.popitem(last=False)
            return False

    def record(self, rec: Record) -> None:
        ts = rec.ts_ns / 1e9
        minute = int(ts // 60)
        space = rec.tags.get("space", "-")
        with self._lock:
            if rec.meta.get("ts_from", "").startswith("skew:"):
                self.skew_fallbacks += 1
            for f in rec.fields:
                if f in ("recv_lag_s", "fcnt", "rssi", "snr"):
                    continue
                k = (rec.measurement, space, f)
                self.counts[k] += 1
                self.last_ts[k] = max(ts, self.last_ts.get(k, 0))
                self.minutes[k].add(minute)
            dev = rec.tags.get("device") or rec.tags.get("gateway")
            if dev:
                self.devices[dev] = {
                    "measurement": rec.measurement, "space": space,
                    "model": rec.tags.get("model"), "last": ts,
                    "battery": rec.fields.get("battery"),
                }

    def snapshot(self, unmapped: set[str], spool_size: int) -> dict:
        now = time.time()
        now_min = int(now // 60)
        lo = now_min - WINDOW_MIN
        with self._lock:
            series = []
            for k, mins in self.minutes.items():
                mins.difference_update({m for m in mins if m <= lo})
                # 수집기 가동 시간이 24시간보다 짧으면 가동 시간 기준으로 나눈다
                span = min(WINDOW_MIN, max(1, now_min - int(self.started // 60)))
                series.append({
                    "measurement": k[0], "space": k[1], "field": k[2],
                    "count": self.counts[k],
                    "last": _iso(self.last_ts[k]),
                    "age_s": round(now - self.last_ts[k]),
                    "coverage_pct": round(100 * min(len(mins), span) / span, 1),
                    "window_min": span,
                })
            series.sort(key=lambda s: (s["measurement"], s["space"], s["field"]))
            devices = {d: {**v, "last": _iso(v["last"]), "age_s": round(now - v["last"])}
                       for d, v in sorted(self.devices.items())}
            return {
                "now": _iso(now), "uptime_s": round(now - self.started),
                "mqtt_connected": self.mqtt_connected, "mqtt_reconnects": self.mqtt_reconnects,
                "messages": self.messages, "unmatched_topics": self.unmatched,
                "duplicates_dropped": self.duplicates, "clock_skew_fallbacks": self.skew_fallbacks,
                "parse_errors": self.errors, "last_error": self.last_error,
                "spool_pending": spool_size,
                "last_write_ok": _iso(self.last_write_ok_at) if self.last_write_ok_at else None,
                "last_write_error": self.last_write_error,
                "unmapped_spaces": sorted(unmapped),
                "series": series, "devices": devices,
            }


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
