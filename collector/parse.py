"""MQTT 메시지를 설정 규칙에 따라 저장할 레코드로 바꾼다.

이 모듈은 네트워크·DB 를 모른다. 입력은 (토픽, 본문, 수신 시각), 출력은 Record 목록이다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_MISSING = object()


@dataclass
class Record:
    measurement: str
    tags: dict[str, str]
    fields: dict[str, float | bool | str]
    ts_ns: int
    source: str
    dedup_key: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---------- MQTT 토픽 매칭 ----------

def topic_matches(pattern: str, topic: str) -> bool:
    p, t = pattern.split("/"), topic.split("/")
    for i, part in enumerate(p):
        if part == "#":
            return True
        if i >= len(t):
            return False
        if part != "+" and part != t[i]:
            return False
    return len(p) == len(t)


# ---------- 시각 ----------

_FRAC = re.compile(r"\.(\d+)")


def parse_time_ns(value: Any, unit: str | None = None) -> int | None:
    """ISO 문자열(나노초 자릿수 허용) 또는 epoch 숫자를 epoch 나노초로."""
    if value is None or value is _MISSING:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        scale = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[unit or "s"]
        # 정수는 정수 연산으로 (float 로 곱하면 나노초 자리가 틀어진다)
        return value * scale if isinstance(value, int) else int(round(value * scale))
    if isinstance(value, str):
        s = value.strip()
        frac_ns = 0
        m = _FRAC.search(s)
        if m:
            digits = m.group(1)
            frac_ns = int((digits + "000000000")[:9])
            s = s[: m.start()] + s[m.end():]
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp()) * 1_000_000_000 + frac_ns
    return None


# ---------- 경로 ----------

def resolve(path: str, doc: Any, topic_parts: list[str], key: str | None = None) -> Any:
    """설정의 경로 문법으로 값을 꺼낸다. 없으면 _MISSING."""
    if path == "@key":
        return key if key is not None else _MISSING
    m = re.fullmatch(r"\$topic\[(\d+)\]", path)
    if m:
        i = int(m.group(1))
        return topic_parts[i] if i < len(topic_parts) else _MISSING
    if path == ".":
        return doc
    values = [doc]
    for part in path.split("."):
        star = part.endswith("[*]")
        name = part[:-3] if star else part
        nxt = []
        for v in values:
            if not isinstance(v, dict) or name not in v:
                continue
            child = v[name]
            if star:
                if isinstance(child, list):
                    nxt.extend(child)
            else:
                nxt.append(child)
        values = nxt
        if not values:
            return _MISSING
    if "[*]" in path:
        return values
    return values[0]


def _agg(value: Any, how: str | None) -> Any:
    if not isinstance(value, list):
        return value
    vals = [v for v in value if v is not None]
    if not vals:
        return _MISSING
    if how in (None, "first"):
        return vals[0]
    if how == "min":
        return min(vals)
    if how == "max":
        return max(vals)
    raise ValueError(f"unknown agg {how!r}")


def _spec(spec: Any) -> tuple[str, str | None, str | None]:
    if isinstance(spec, str):
        return spec, None, None
    return spec["path"], spec.get("agg"), spec.get("unit")


def _field_value(v: Any) -> float | bool | None:
    # 정수와 실수를 섞어 보내는 센서가 있어(습도 59 / 59.5) 숫자는 항상 float 로 저장한다.
    # 그러지 않으면 InfluxDB 가 필드 타입 충돌로 쓰기를 거부한다.
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return None


# ---------- 규칙 적용 ----------

CLOCK_MIN_NS = parse_time_ns("2024-01-01T00:00:00Z")   # 이보다 이른 시각은 시계가 아직 안 맞은 것


class Parser:
    def __init__(self, config: dict):
        self.sources = config["sources"]
        spaces = config.get("spaces") or {}
        self.space_by_name: dict[str, str] = spaces.get("by_name") or {}
        self.space_by_device: dict[str, str] = spaces.get("by_device") or {}
        self.max_skew_ns = int(config.get("max_clock_skew_s", 600) * 1e9)
        self.unmapped: set[str] = set()

    def topics(self) -> list[str]:
        return [s["topic"] for s in self.sources]

    def source_for(self, topic: str) -> dict | None:
        for s in self.sources:
            if topic_matches(s["topic"], topic):
                return s
        return None

    def _space(self, raw: Any, device: str | None) -> str:
        if device and device in self.space_by_device:
            return self.space_by_device[device]
        if raw is _MISSING or raw is None:
            return "unknown"
        raw = str(raw)
        if raw in self.space_by_name:
            return self.space_by_name[raw]
        self.unmapped.add(raw)
        return raw

    def _time(self, src: dict, doc: Any, parts: list[str], recv_ns: int) -> tuple[int, str]:
        for spec in src.get("time") or []:
            path, agg, unit = _spec(spec)
            v = resolve(path, doc, parts)
            if isinstance(v, list):
                parsed = [parse_time_ns(x, unit) for x in v]
                v = _agg([p for p in parsed if p is not None], agg)
                ts = v if isinstance(v, int) else None
            else:
                ts = parse_time_ns(v, unit)
            if ts is None:
                continue
            # 재부팅 직후 보드 시계가 아직 1970년이면 받은 시각을 믿을 수 없으니 메시지 시각을 쓴다
            if recv_ns >= CLOCK_MIN_NS and abs(ts - recv_ns) > self.max_skew_ns:
                return recv_ns, f"skew:{path}"
            return ts, path
        return recv_ns, "received"

    def parse(self, topic: str, payload: bytes, recv_ns: int) -> list[Record]:
        src = self.source_for(topic)
        if src is None:
            return []
        parts = topic.split("/")

        if src.get("payload") == "raw":
            tags = {k: str(v) for k, p in (src.get("tags") or {}).items()
                    if (v := resolve(p, None, parts)) is not _MISSING}
            return [Record(src["measurement"], tags, {"seen": 1.0}, recv_ns, src["name"])]

        try:
            doc = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"json: {e}") from e

        ts_ns, ts_from = self._time(src, doc, parts, recv_ns)
        dedup = None
        if src.get("dedup_key"):
            d = resolve(src["dedup_key"], doc, parts)
            dedup = None if d is _MISSING else str(d)

        items: list[tuple[str | None, Any]]
        if src.get("explode"):
            box = resolve(src["explode"], doc, parts)
            items = list(box.items()) if isinstance(box, dict) else []
        else:
            items = [(None, doc)]

        out = []
        for key, item in items:
            tags: dict[str, str] = {}
            for name, p in (src.get("tags") or {}).items():
                v = resolve(p, item, parts, key)
                if v is not _MISSING and v is not None and v != "":
                    tags[name] = str(v)
            space_raw = resolve(src["space_from"], item, parts, key) if src.get("space_from") else _MISSING
            if src.get("space_from"):
                tags["space"] = self._space(space_raw, tags.get("device"))

            fields: dict[str, float | bool | str] = {}
            box = resolve(src.get("fields_from", "."), item, parts, key)
            if isinstance(box, dict):
                for k, v in box.items():
                    fv = _field_value(v)
                    if fv is not None:
                        fields[k] = fv
            for name, spec in (src.get("extra_fields") or {}).items():
                path, agg, _ = _spec(spec)
                v = _agg(resolve(path, item, parts, key), agg)
                fv = _field_value(v)
                if fv is not None:
                    fields[name] = fv
            if not fields:
                continue
            # 수신 지연(초): 네트워크가 밀렸다가 몰려 오는 상황을 나중에 확인하기 위함
            fields["recv_lag_s"] = (recv_ns - ts_ns) / 1e9
            out.append(Record(src["measurement"], tags, fields, ts_ns, src["name"],
                              dedup_key=dedup, meta={"ts_from": ts_from}))
        return out
