"""과제 8: 상시 추론 서비스. 매분 공간마다 예측·이상·상태를 계산해 MQTT(retained)로 내보내고 조회 API 로 답한다.

  python -m analysis.service          (docker compose 의 infer 서비스)

흐름(매분):
  로컬 InfluxDB(우리 수집기)에서 원자료 → prep.clean / prep.to_grid (학습과 같은 함수) → 끝난 5분 칸까지만 사용
  → 과제 3 GRU(60분 예측) / 과제 4 오토인코더+규칙(이상) / 과제 5 MLP(환기 경보)+규칙(사용 여부)
  → MQTT 발행 + API 스냅샷 갱신 + 주기 기록(data/service/cycles-YYYYMMDD.jsonl)
결과 형식: docs/service-api.md
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch
import yaml

from . import prep, task3, task4, task5

log = logging.getLogger("infer")
STEP = pd.Timedelta("5min")
HISTORY = pd.Timedelta(days=8)          # 일주일 전 같은 시각 입력 + 2시간 창 + 여유
STALE = pd.Timedelta("10min")           # 이보다 오래 새 값이 없으면 '데이터 끊김'
REPEAT = pd.Timedelta("30min")          # 같은 경보는 켜져 있는 동안 30분마다 한 번만 다시 보낸다
VERSION = "v1"


def env(name, default=None):
    v = os.environ.get(name, default)
    if v is None:
        raise SystemExit(f"환경 변수 {name} 필요")
    return v


# ---------- 원자료 ----------

class Source:
    """로컬 InfluxDB(우리 수집기, measurement env) → 학습과 같은 긴 표(time, device_id, location, field, value).

    공간 이름은 수집기 설정(spaces.by_name)을 거꾸로 써서 학습 때 이름(실습실 등)으로 되돌린다.
    로컬에 일주일치가 아직 없으면 그 앞부분은 강사 iot-bucket 에서 읽기만 한다(INSTRUCTOR_INFLUX_* 가 있을 때).
    """

    def __init__(self):
        from influxdb_client import InfluxDBClient
        self.local = InfluxDBClient(url=env("INFLUX_URL"), token=env("INFLUX_TOKEN"), org=env("INFLUX_ORG"), timeout=60_000)
        self.bucket = env("INFLUX_BUCKET")
        self.org = env("INFLUX_ORG")
        cfg = yaml.safe_load(open(os.environ.get("COLLECTOR_CONFIG", "config/collector.yaml"), encoding="utf-8"))
        inv = {}
        for raw, key in (cfg.get("spaces", {}).get("by_name") or {}).items():
            inv.setdefault(key, raw)                # 여러 이름이 한 키면 첫 이름(학습 데이터 이름)
        self.to_name = inv
        self.instr = None
        if os.environ.get("INSTRUCTOR_INFLUX_URL"):
            self.instr = InfluxDBClient(url=os.environ["INSTRUCTOR_INFLUX_URL"], token=os.environ["INSTRUCTOR_INFLUX_TOKEN"],
                                        org=os.environ["INSTRUCTOR_INFLUX_ORG_ID"], timeout=120_000)
        self.cache = pd.DataFrame(columns=["time", "device_id", "location", "field", "value"])

    def _local(self, start: pd.Timestamp) -> pd.DataFrame:
        q = f'''
from(bucket: "{self.bucket}")
  |> range(start: {start.isoformat()})
  |> filter(fn: (r) => r._measurement == "env" and (r._field == "co2" or r._field == "temperature" or r._field == "humidity"))
  |> keep(columns: ["_time", "_value", "_field", "space", "device"])
'''
        res = self.local.query_api().query_data_frame(q, org=self.org)
        frames = [f for f in (res if isinstance(res, list) else [res]) if len(f)]
        if not frames:
            return self.cache.iloc[:0]
        df = pd.concat(frames, ignore_index=True).rename(columns={"_time": "time", "_value": "value", "_field": "field", "device": "device_id"})
        df["location"] = df["space"].map(lambda s: self.to_name.get(s, s))
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df[["time", "device_id", "location", "field", "value"]]

    def _instructor(self, start, stop) -> pd.DataFrame:
        from .export import fetch
        api = self.instr.query_api()
        org = os.environ["INSTRUCTOR_INFLUX_ORG_ID"]
        bucket = os.environ.get("INSTRUCTOR_INFLUX_BUCKET", "iot-bucket")
        parts = [fetch(api, bucket, org, f, start.to_pydatetime(), stop.to_pydatetime()).assign(field=f) for f in prep.FIELDS]
        return pd.concat(parts, ignore_index=True)[["time", "device_id", "location", "field", "value"]]

    def read(self, now: pd.Timestamp) -> pd.DataFrame:
        """처음엔 8일치, 그 뒤로는 마지막 받은 시각 15분 전부터만 다시 읽어 합친다."""
        start = now - HISTORY
        if len(self.cache):
            since = self.cache["time"].max() - pd.Timedelta("15min")
            new = self._local(since)
            df = pd.concat([self.cache[self.cache["time"] < since], new], ignore_index=True)
        else:
            df = self._local(start)
            first = df["time"].min() if len(df) else now
            if self.instr is not None and first > start + pd.Timedelta("1h"):
                try:
                    df = pd.concat([self._instructor(start, first), df], ignore_index=True)
                    log.info("과거 %s ~ %s 는 iot-bucket 에서 읽음", start, first)
                except Exception:
                    log.exception("iot-bucket 과거 읽기 실패(로컬만 사용)")
        df = df[df["time"] >= start].drop_duplicates(["time", "device_id", "field"])
        self.cache = df
        return df


# ---------- 계산 ----------

class Engine:
    """학습 산출물을 불러 한 시각의 판단을 만든다. 원자료(DataFrame)만 받으므로 InfluxDB 없이도 시험할 수 있다."""

    def __init__(self, reports: Path):
        g2, self.norm, self.split = task3.load_task2(reports / "task2")
        self.columns = g2.columns                               # 학습 때 (공간, 항목) 구성 그대로
        self.locations = sorted(self.columns.get_level_values("location").unique())
        self.night_ref = task4.night_reference(g2, self.split.train_end)
        del g2
        # 과제 3 보완(task3_select)이 검증 기간으로 고른 예측을 따른다. 없으면 gru_full 만.
        sel = reports / "task3" / "select_choice.csv"
        self.choice = pd.read_csv(sel) if sel.exists() else None
        need = {"full"}
        if self.choice is not None:
            need |= {c.split("*")[0][4:] for c in self.choice["candidate"] if c.startswith("gru_")}
        self.grus = {}
        for v in sorted(need):
            ck = torch.load(reports / "task3" / f"gru_{v}.pt", map_location="cpu", weights_only=False)
            m = task3.GRUForecaster(ck["n_in"], ck["n_static"], ck["hidden"], ck["layers"])
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.grus[v] = (m, ck)
        self.gru_ck = self.grus["full"][1]
        ae = torch.load(reports / "task4" / "ae.pt", map_location="cpu", weights_only=False)
        self.ae_locs = ae["locations"]
        self.ae = None
        self.ae_state = ae["state_dict"]
        thr = pd.read_csv(reports / "task4" / "thresholds.csv", index_col=[0, 1]).iloc[:, 0]
        self.ae_thr = {k: float(v) for k, v in thr.items()}
        vp = reports / "task5" / "vent.pt"
        self.vent = None
        if vp.exists():
            v = torch.load(vp, map_location="cpu", weights_only=False)
            self.vent = task5.MLP(v["n_in"])
            self.vent.load_state_dict(v["state_dict"])
            self.vent.eval()
            self.vent_thr, self.vent_locs = v["threshold"], v["locations"]

    def combine(self, loc: str, by_v: dict, cur: np.ndarray) -> np.ndarray:
        """[12, 2]. 칸마다 검증 기간에서 고른 후보(GRU 변형 / 현재값 유지 / 가중 평균)를 쓴다.
        고른 시점은 5·15·30·60분뿐이라, 그 사이 칸은 바로 위 시점의 선택을 따른다."""
        out = by_v["full"][loc].copy()
        if self.choice is None:
            return out
        c = self.choice[self.choice.location == loc]
        for j, f in enumerate(task3.TARGETS):
            for k in range(task3.HORIZON):
                minutes = 5 * (k + 1)
                h = min(x for x in (5, 15, 30, 60) if x >= minutes)
                row = c[(c.field == f) & (c.horizon_min == h)]
                if row.empty:
                    continue
                name = row.candidate.iloc[0]
                if name == "persistence":
                    out[k, j] = cur[j]
                    continue
                v, _, w = name.partition("*")
                v = v[4:]
                p = by_v.get(v, by_v["full"]).get(loc, out)[k, j]
                out[k, j] = p if not w else float(w.split("+")[0]) * p + (1 - float(w.split("+")[0])) * cur[j]
        return out

    def grid(self, raw: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        """학습과 같은 정제·격자. 막 시작된(끝나지 않은) 5분 칸은 버린다."""
        d = raw.copy()
        d["field"] = pd.Categorical(d["field"], categories=list(prep.FIELDS))
        for c in ("device_id", "location"):
            d[c] = d[c].astype("category")
        kept, _ = prep.clean(d)
        parts = []
        for f in prep.FIELDS:
            k = kept[kept["field"] == f]
            if len(k):
                parts.append(prep.to_grid(k))
        g = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        last_done = (now.floor("5min") - STEP)
        full = pd.date_range((now - HISTORY).floor("5min"), last_done, freq="5min", tz="UTC")
        g = g.reindex(index=full).reindex(columns=self.columns)
        g.columns = self.columns
        return g.astype("float32")

    def compute(self, raw: pd.DataFrame, now: pd.Timestamp) -> dict:
        g = self.grid(raw, now)
        t = len(g.index) - 1
        t_bin = g.index[-1]
        out = {}
        split = prep.Split(train_end=g.index[0], val_end=g.index[0], end=g.index[-1])
        by_v = {}
        for v, (m, ck) in self.grus.items():
            data = task3.Data(g, self.norm, split, ck["lookback"], use_calendar=(v != "no_calendar"),
                              absolute=(v == "absolute"), use_lags=(v != "no_lags"))
            by_v[v] = {loc: task3.predict(m, data, l, np.array([t]), torch.device("cpu"))[0]
                       for l, loc in enumerate(data.locations) if not np.isnan(data.raw[l, t]).all()}
        preds = {loc: self.combine(loc, by_v, data.raw[data.locations.index(loc), t]) for loc in by_v["full"]}
        aed = task4.AEData(g, self.norm, self.ae_locs)
        if self.ae is None:
            x0, _, _ = aed.windows(0, np.array([task4.W - 1]))
            self.ae = task4.AE(x0.shape[1], task4.W * 3)
            self.ae.load_state_dict(self.ae_state)
            self.ae.eval()
        ae_sc = task4.ae_scores(self.ae, aed, torch.device("cpu"))
        rules = task4.rule_scores(g, self.night_ref)
        last_seen = raw.groupby("location")["time"].max() if len(raw) else pd.Series(dtype="datetime64[ns, UTC]")
        recent = raw[raw["time"] >= now - STALE]
        for loc in self.locations:
            fields = [f for f in prep.FIELDS if (loc, f) in g]
            cur = {f: _f(g[(loc, f)].iloc[-1]) for f in fields}
            seen = last_seen.get(loc)
            stale = seen is None or now - seen > STALE
            r: dict = {"space": loc, "time": t_bin.isoformat(), "computed_at": now.isoformat(), "version": VERSION,
                       "current": cur, "last_data_at": seen.isoformat() if seen is not None else None}
            # 예측
            if loc in preds and not stale:
                p = preds[loc]
                steps = [{"minutes": 5 * (k + 1), "time": (t_bin + STEP * (k + 1)).isoformat(),
                          "co2_ppm": _f(p[k, 0]) if "co2" in fields else None,
                          "temperature_c": _f(p[k, 1]) if "temperature" in fields else None} for k in range(task3.HORIZON)]
                over = next((s["minutes"] for s in steps if s["co2_ppm"] is not None and s["co2_ppm"] >= 1000), None)
                r["forecast"] = {"steps": steps, "co2_over_1000_in_min": over if (cur.get("co2") or 0) < 1000 else 0}
            else:
                r["forecast"] = None
            # 이상
            scores = {f: _f(ae_sc[(loc, f)].iloc[-1] / self.ae_thr[(loc, f)]) for f in fields
                      if (loc, f) in ae_sc and (loc, f) in self.ae_thr and pd.notna(ae_sc[(loc, f)].iloc[-1])}
            rule_hits = sorted({f"{k[2]}:{k[1]}" for k, v in rules.items() if k[0] == loc and bool(v.iloc[-1])})
            rr = recent[recent["location"] == loc]
            oor = [f for f in fields if len(rr[rr["field"] == f]) and
                   ((rr.loc[rr["field"] == f, "value"] < prep.RULES[f][0]) | (rr.loc[rr["field"] == f, "value"] > prep.RULES[f][1])).any()]
            if stale:
                rule_hits.append("data_gap")
            rule_hits += [f"out_of_range:{f}" for f in oor]
            worst = max(scores, key=scores.get) if scores else None
            r["anomaly"] = {"score": scores.get(worst) if worst else None, "is_anomaly": bool(worst and scores[worst] > 1.0),
                            "cause_field": worst if worst and scores[worst] > 1.0 else None, "field_scores": scores,
                            "rules": rule_hits}
            # 상태
            st: dict = {}
            if "co2" in fields and not stale:
                s = g[(loc, "co2")]
                d15 = s.iloc[-1] - s.iloc[-4]
                st["occupied"] = bool(d15 >= 30) if pd.notna(d15) else None
                st["occupied_method"] = "rule_co2_rise_15min_30ppm"
                if self.vent is not None and loc in self.vent_locs:
                    X, ok = task5.features(g, self.norm, loc, self.vent_locs)
                    if ok[-1] and (cur.get("co2") or 0) < 1000:
                        pr = float(task5.prob(self.vent, X[-1:])[0])
                        st["vent_alert"] = pr >= self.vent_thr
                        st["vent_prob"] = round(pr, 3)
                    else:
                        st["vent_alert"] = bool((cur.get("co2") or 0) >= 1000)
                        st["vent_prob"] = None
            r["state"] = st or None
            out[loc] = r
        return out


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, 2)


# ---------- 경보 정리 ----------

class Alerts:
    """경보가 켜질 때 한 번, 켜져 있으면 30분마다 한 번, 꺼질 때 한 번 보낸다(매분 같은 경보를 보내지 않게)."""

    def __init__(self):
        self.active: dict[tuple, pd.Timestamp] = {}

    def update(self, snap: dict, now: pd.Timestamp) -> list[dict]:
        on = set()
        for loc, r in snap.items():
            for rule in (r["anomaly"] or {}).get("rules", []):
                on.add((loc, rule))
            if (r["anomaly"] or {}).get("is_anomaly"):
                on.add((loc, f"anomaly:{r['anomaly']['cause_field']}"))
            if (r.get("state") or {}).get("vent_alert"):
                on.add((loc, "vent_soon"))
            fc = r.get("forecast") or {}
            if fc.get("co2_over_1000_in_min"):
                on.add((loc, "co2_over_1000_forecast"))
        events = []
        for k in on:
            last = self.active.get(k)
            if last is None or now - last >= REPEAT:
                events.append({"space": k[0], "alert": k[1], "status": "raised" if last is None else "ongoing", "time": now.isoformat()})
                self.active[k] = now
        for k in list(self.active):
            if k not in on:
                events.append({"space": k[0], "alert": k[1], "status": "cleared", "time": now.isoformat()})
                del self.active[k]
        return events


# ---------- 내보내기 ----------

class Publisher:
    def __init__(self):
        self.prefix = os.environ.get("PUB_TOPIC_PREFIX", "aigj/01-001/inference")
        self.client = None
        host = os.environ.get("PUB_MQTT_HOST")
        if host:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="aigj-01-001-infer")
            self.client.reconnect_delay_set(1, 30)
            self.client.connect_async(host, int(os.environ.get("PUB_MQTT_PORT", "1883")))
            self.client.loop_start()

    def state(self, snap: dict, keys: dict):
        if not self.client:
            return
        for loc, r in snap.items():
            self.client.publish(f"{self.prefix}/{keys.get(loc, loc)}/state", json.dumps(r, ensure_ascii=False), qos=1, retain=True)

    def alerts(self, events: list[dict], keys: dict):
        if not self.client:
            return
        for e in events:
            self.client.publish(f"{self.prefix}/{keys.get(e['space'], e['space'])}/alert", json.dumps(e, ensure_ascii=False), qos=1)

    def status(self, st: dict):
        if self.client:
            self.client.publish(f"{self.prefix}/_service/status", json.dumps(st, ensure_ascii=False), qos=1, retain=True)


class Api(BaseHTTPRequestHandler):
    svc: "Service" = None

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        q = parse_qs(u.query)
        snap, st = self.svc.snapshot()
        if u.path == "/healthz":
            ok = st.get("last_ok_at") and pd.Timestamp.now(tz="UTC") - pd.Timestamp(st["last_ok_at"]) < pd.Timedelta("5min")
            return self._send(200 if ok else 503, {"ok": bool(ok), **st})
        if u.path in ("/", "/status"):
            return self._send(200, st)
        if parts == ["spaces"]:
            return self._send(200, {"spaces": [{"space": k, "key": self.svc.keys.get(k, k)} for k in snap]})
        if len(parts) >= 2 and parts[0] == "spaces":
            loc = self.svc.resolve(parts[1])
            if loc not in snap:
                return self._send(404, {"error": f"모르는 공간: {parts[1]}", "spaces": list(snap)})
            r = snap[loc]
            if len(parts) == 2:
                return self._send(200, r)
            if parts[2] in ("forecast", "anomaly", "state"):
                part = r.get(parts[2])
                if parts[2] == "forecast" and part and "minutes" in q:
                    m = int(q["minutes"][0])
                    step = min((s for s in part["steps"]), key=lambda s: abs(s["minutes"] - m))
                    return self._send(200, {"space": loc, "asked_minutes": m, **step, "time_basis": r["time"]})
                return self._send(200, {"space": loc, "time": r["time"], parts[2]: part})
        return self._send(404, {"error": "경로: /status, /healthz, /spaces, /spaces/<공간>, /spaces/<공간>/forecast?minutes=30, /spaces/<공간>/anomaly, /spaces/<공간>/state"})


class Service:
    def __init__(self):
        self.reports = Path(os.environ.get("DATA_DIR", "data")) / "reports"
        self.engine = Engine(self.reports)
        self.source = Source()
        self.alerts = Alerts()
        self.pub = Publisher()
        self.keys = {v: k for k, v in self.source.to_name.items()}      # 실습실 -> lab
        self.lock = threading.Lock()
        self._snap, self._st = {}, {"started_at": pd.Timestamp.now(tz="UTC").isoformat(), "cycles": 0, "errors": 0}
        self.logdir = Path(os.environ.get("DATA_DIR", "data")) / "service"
        self.logdir.mkdir(parents=True, exist_ok=True)

    def resolve(self, name: str) -> str:
        return self.source.to_name.get(name, name)

    def snapshot(self):
        with self.lock:
            return dict(self._snap), dict(self._st)

    def cycle(self):
        now = pd.Timestamp.now(tz="UTC").floor("s")
        t0 = time.perf_counter()
        rec = {"time": now.isoformat()}
        try:
            raw = self.source.read(now)
            t1 = time.perf_counter()
            snap = self.engine.compute(raw, now)
            t2 = time.perf_counter()
            events = self.alerts.update(snap, now)
            self.pub.state(snap, self.keys)
            self.pub.alerts(events, self.keys)
            rec.update(ok=True, read_s=round(t1 - t0, 3), compute_s=round(t2 - t1, 3), rows=len(raw),
                       spaces=len(snap), events=events)
            with self.lock:
                self._snap = snap
                self._st.update(cycles=self._st["cycles"] + 1, last_ok_at=now.isoformat(), last_cycle_s=round(time.perf_counter() - t0, 3),
                                active_alerts=[{"space": k[0], "alert": k[1]} for k in self.alerts.active])
        except Exception as e:                                   # 한 번 실패해도 다음 주기는 처음부터 다시
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            log.error("주기 실패: %s\n%s", e, traceback.format_exc())
            with self.lock:
                self._st.update(errors=self._st["errors"] + 1, last_error=rec["error"], last_error_at=now.isoformat())
        rec["total_s"] = round(time.perf_counter() - t0, 3)
        with open(self.logdir / f"cycles-{now:%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.pub.status(self.snapshot()[1])

    def run(self):
        Api.svc = self
        port = int(os.environ.get("API_PORT", "8091"))
        threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), Api).serve_forever, daemon=True).start()
        log.info("조회 API :%d", port)
        while True:
            self.cycle()
            time.sleep(60 - datetime.now(timezone.utc).second % 60 + 5)   # 매분 5초(수집기 기록이 들어온 뒤)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Service().run()


if __name__ == "__main__":
    main()
