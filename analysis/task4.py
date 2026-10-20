"""과제 4: 이상 탐지. 정상만 보고 배우는 오토인코더 + 규칙(튄 값, 값 고정, 새벽 기준선) 비교.

  python -m analysis.task4 [--task2 data/reports/task2] [--out data/reports/task4]

실데이터에는 고장 정답이 없다. 그래서
  - 학습: 학습 기간 데이터(대부분 정상이라고 가정)로 오토인코더를 배운다.
  - 임계값: 검증 기간(고장 주입 없음)에서 헛경보 예산(하루 횟수)을 넘지 않는 가장 낮은 값으로 정한다.
  - 평가: 시험 기간에 고장 5종을 직접 주입하고(inject), 이벤트 탐지율·지연·헛경보를 잰다.
    시험 기간에 원래 있던 진짜 이상도 헛경보로 세므로 헛경보 수는 보수적인 값이다.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import prep
from .task3 import calendar, load_task2

W = 12                       # 오토인코더 창: 1시간
STEP = pd.Timedelta("5min")
MERGE = 12                   # 1시간 안에 다시 울린 경보는 같은 경보로 본다
FAULTS = ("spike", "stuck", "drift", "cooling", "ventilation")
KO = {"spike": "튄 값", "stuck": "값 고정", "drift": "영점 틀어짐", "cooling": "냉방 정지", "ventilation": "환기 정지"}


# ---------- 고장 주입 ----------

@dataclass
class Event:
    location: str
    fault: str
    fields: tuple
    start: pd.Timestamp
    end: pd.Timestamp


def _used_day_hours(s: pd.Series) -> pd.DatetimeIndex:
    """CO2 가 새벽보다 100ppm 이상 높은 평일 10~15시 칸(수업·업무 중). 냉방·환기 고장을 넣을 자리."""
    local = s.index.tz_convert(prep.TZ)
    out = []
    for day, g in s.groupby(local.date):
        gl = g.copy()
        gl.index = gl.index.tz_convert(prep.TZ)
        night = gl.between_time("04:00", "06:00").median()
        mid = gl.between_time("10:00", "15:00")
        if pd.isna(night) or pd.Timestamp(day).dayofweek >= 5:
            continue
        hit = mid[mid > night + 100]
        out += list(hit.index.tz_convert("UTC"))
    return pd.DatetimeIndex(out)


def inject(grid: pd.DataFrame, start: pd.Timestamp, per_type: int, seed: int) -> tuple[pd.DataFrame, list[Event]]:
    rng = np.random.default_rng(seed)
    g = grid.astype("float64")
    idx = g.index[g.index >= start]
    events: list[Event] = []
    taken: dict[str, list[tuple]] = {}

    def free(loc, a, b):
        pad = pd.Timedelta("6h")
        return all(b + pad < x or a - pad > y for x, y in taken.get(loc, []))

    def pick(loc, cands, dur, tries=200):
        for _ in range(tries):
            a = cands[rng.integers(len(cands))]
            b = a + dur
            if b <= idx[-1] and free(loc, a, b):
                taken.setdefault(loc, []).append((a, b))
                return a, b
        return None

    locs = sorted(g.columns.get_level_values("location").unique())
    for loc in locs:
        has_co2 = (loc, "co2") in g
        fields = [f for f in prep.FIELDS if (loc, f) in g]
        main = "co2" if has_co2 else "temperature"
        if g.loc[idx, (loc, main)].notna().sum() == 0:      # 시험 기간에 값이 없는 공간(창고: 장기 결측)
            continue
        used = _used_day_hours(g[(loc, "co2")]).intersection(idx) if has_co2 else pd.DatetimeIndex([])
        for fault in FAULTS:
            if fault in ("drift", "ventilation") and not has_co2:
                continue
            if fault in ("cooling", "ventilation") and len(used) == 0:
                continue
            for _ in range(per_type):
                if fault == "spike":
                    ok = pick(loc, idx[g.loc[idx, (loc, main)].notna().to_numpy()], STEP * 0)
                    if not ok:
                        continue
                    a, _b = ok
                    size = rng.uniform(300, 800) if main == "co2" else rng.uniform(3, 8)
                    g.loc[a, (loc, main)] += size * rng.choice([-1, 1]) if main != "co2" else size
                    events.append(Event(loc, fault, (main,), a, a))
                elif fault == "stuck":
                    dur = STEP * int(rng.integers(12, 49))                 # 1~4시간
                    ok = pick(loc, idx, dur)
                    if not ok:
                        continue
                    a, b = ok
                    for f in fields:                                    # 장비가 멈추면 세 값이 함께 멈춘다
                        v = g.loc[:a, (loc, f)].dropna()
                        if len(v):
                            g.loc[a:b, (loc, f)] = v.iloc[-1]
                    events.append(Event(loc, fault, tuple(fields), a, b))
                elif fault == "drift":
                    ramp = STEP * int(rng.integers(36, 73))                # 3~6시간에 걸쳐
                    hold = pd.Timedelta("12h")                           # 틀어진 채로 유지(다음 점검 전까지)
                    ok = pick(loc, idx, ramp + hold)
                    if not ok:
                        continue
                    a, b = ok
                    size = rng.uniform(200, 400)
                    sl = g.loc[a:b, (loc, "co2")]
                    frac = np.clip((sl.index - a) / ramp, 0, 1)
                    g.loc[a:b, (loc, "co2")] = sl + size * frac
                    events.append(Event(loc, fault, ("co2",), a, b))
                elif fault == "cooling":
                    rise, hold = STEP * int(rng.integers(12, 25)), pd.Timedelta("2h")
                    ok = pick(loc, used, rise + hold)
                    if not ok:
                        continue
                    a, b = ok
                    size = rng.uniform(3, 5)
                    sl = g.loc[a:b, (loc, "temperature")]
                    frac = np.clip((sl.index - a) / rise, 0, 1)
                    g.loc[a:b, (loc, "temperature")] = sl + size * frac
                    events.append(Event(loc, fault, ("temperature",), a, b))
                elif fault == "ventilation":
                    dur = pd.Timedelta("3h")
                    ok = pick(loc, used, dur)
                    if not ok:
                        continue
                    a, b = ok
                    day = a.tz_convert(prep.TZ).normalize()
                    s = g[(loc, "co2")]
                    night = s[(s.index >= day + pd.Timedelta("4h")) & (s.index < day + pd.Timedelta("6h"))].median()
                    sl = g.loc[a:b, (loc, "co2")]
                    # 환기가 멈추면 사람이 내놓는 CO2 가 빠지지 않아 평소 상승 폭의 1.8배가 된다
                    g.loc[a:b, (loc, "co2")] = night + 1.8 * (sl - night).clip(lower=0) + (sl - night).clip(upper=0)
                    events.append(Event(loc, fault, ("co2",), a, b))
    return g, events


# ---------- 규칙 ----------

SPIKE = {"co2": 200.0, "temperature": 2.0, "humidity": 15.0}


def rule_scores(g: pd.DataFrame, night_ref: dict) -> dict[tuple, pd.Series]:
    """규칙별 경보(0/1). key = (location, field, rule). 모두 과거 값과 다음 한 칸까지만 본다(지연 5분)."""
    out = {}
    for loc in sorted(g.columns.get_level_values("location").unique()):
        for f in prep.FIELDS:
            if (loc, f) not in g:
                continue
            s = g[(loc, f)]
            prev, nxt = s.shift(1), s.shift(-1)
            # 튄 값: 앞뒤 두 칸 모두와 크게 다르고 앞뒤끼리는 비슷함. 다음 칸을 봐야 해서 한 칸 늦게 울린다.
            jump = ((s - prev).abs() > SPIKE[f]) & ((s - nxt).abs() > SPIKE[f]) & ((prev - nxt).abs() < SPIKE[f] / 2)
            out[(loc, f, "spike")] = jump.shift(1, fill_value=False)
        # 값 고정: CO2 는 측정 잡음이 있어 30분(6칸) 똑같으면 이상. 온도·습도는 분해능이 거칠어 둘 다 1시간 같아야 이상.
        if (loc, "co2") in g:
            d = g[(loc, "co2")].diff().eq(0)
            out[(loc, "co2", "stuck")] = d.rolling(6).sum().eq(6)
        th = [f for f in ("temperature", "humidity") if (loc, f) in g]
        if th:
            same = np.logical_and.reduce([g[(loc, f)].diff().eq(0) for f in th])
            out[(loc, th[0], "stuck")] = pd.Series(same, index=g.index).rolling(12).sum().eq(12)
        # 새벽 기준선(영점 틀어짐 전용): 사람이 없는 02~05시 CO2 는 외기 수준이어야 한다.
        if (loc, "co2") in g and loc in night_ref:
            s = g[(loc, "co2")]
            local = s.index.tz_convert(prep.TZ)
            night = ((local.hour >= 2) & (local.hour < 5))
            med = s.where(night).rolling(36, min_periods=24).median()
            out[(loc, "co2", "night")] = pd.Series(night, index=s.index) & (med > night_ref[loc])
    return out


def night_reference(grid: pd.DataFrame, end: pd.Timestamp) -> dict:
    """학습 기간 새벽 CO2 중앙값의 99% 분위 + 60ppm."""
    ref = {}
    for loc in sorted(grid.columns.get_level_values("location").unique()):
        if (loc, "co2") not in grid:
            continue
        s = grid.loc[grid.index < end, (loc, "co2")]
        local = s.index.tz_convert(prep.TZ)
        nm = s[(local.hour >= 2) & (local.hour < 5)].groupby(local[(local.hour >= 2) & (local.hour < 5)].date).median()
        if len(nm) > 10:
            ref[loc] = float(nm.quantile(0.99) + 60)
    return ref


# ---------- 오토인코더 ----------

class AE(nn.Module):
    def __init__(self, n_in, n_out, hidden=64, code=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, code))
        self.dec = nn.Sequential(nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.dec(self.enc(x))


class AEData:
    """공간별 [T, 3] 정규화 값 + 결측 표시. 창 하나 = 12칸 x 3항목(+표시) + 끝 시각 달력 + 공간 one-hot."""

    def __init__(self, g: pd.DataFrame, norm: pd.DataFrame, locations: list[str]):
        self.locations, self.index = locations, g.index
        T, nL = len(g.index), len(locations)
        self.V = np.zeros((nL, T, 3), np.float32)
        self.M = np.zeros((nL, T, 3), np.float32)
        for l, loc in enumerate(locations):
            for j, f in enumerate(prep.FIELDS):
                if (loc, f) in g and (loc, f) in norm.index:
                    v = ((g[(loc, f)] - norm.loc[(loc, f), "mean"]) / max(norm.loc[(loc, f), "std"], 1e-3)).to_numpy(np.float32)
                    # 정규화 범위 밖 값은 잘라서 넣는다(학습이 몇 개의 거대 값에 끌려가지 않게). 점수는 잘린 값으로도 충분히 크다.
                    self.M[l, :, j] = ~np.isnan(v)
                    self.V[l, :, j] = np.clip(np.nan_to_num(v), -8, 8)
        self.cal = calendar(g.index)

    def windows(self, l, ti):
        offs = np.arange(-W + 1, 1)
        v = self.V[l, ti[:, None] + offs[None, :]]                    # [N, W, 3]
        m = self.M[l, ti[:, None] + offs[None, :]]
        oh = np.zeros((len(ti), len(self.locations)), np.float32)
        oh[:, l] = 1
        x = np.concatenate([v.reshape(len(ti), -1), m.reshape(len(ti), -1), self.cal[ti], oh], 1)
        return x, v.reshape(len(ti), -1), m.reshape(len(ti), -1)

    def valid(self, l, lo, hi):
        t = np.arange(max(lo, W - 1), hi)
        cover = np.lib.stride_tricks.sliding_window_view(self.M[l], W, axis=0).mean(axis=2)  # [T-W+1, 3]
        has = self.M[l].any(axis=0)
        ok = (cover[t - W + 1][:, has] >= 0.8).all(axis=1) if has.any() else np.zeros(len(t), bool)
        return t[ok]


def train_ae(data: AEData, train_end_i: int, val_end_i: int, epochs: int, seed: int, device) -> AE:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    samples = [(l, t) for l in range(len(data.locations)) for t in data.valid(l, 0, train_end_i)]
    vals = [(l, t) for l in range(len(data.locations)) for t in data.valid(l, train_end_i, val_end_i)]
    li, ti = np.array([s[0] for s in samples]), np.array([s[1] for s in samples])
    vli, vti = np.array([s[0] for s in vals]), np.array([s[1] for s in vals])
    x0, _, _ = data.windows(0, np.array([W - 1]))
    model = AE(x0.shape[1], W * 3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss_on(lsel, tsel):
        xs, vs, ms = zip(*(data.windows(l, tsel[lsel == l]) for l in np.unique(lsel)))
        x, v, m = (torch.from_numpy(np.concatenate(a)).to(device) for a in (xs, vs, ms))
        return (torch.abs(model(x) - v) * m).sum() / m.sum().clamp(min=1)

    best, state, bad = np.inf, None, 0
    vpick = rng.choice(len(vti), min(len(vti), 30000), replace=False)
    for ep in range(epochs):
        model.train()
        for _ in range(200):
            p = rng.integers(0, len(ti), 512)
            loss = loss_on(li[p], ti[p])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_on(vli[vpick], vti[vpick]).item()
        print(f"[ae] epoch {ep + 1} val {v:.4f}", flush=True)
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(state)
    return model


def ae_scores(model: AE, data: AEData, device) -> dict[tuple, pd.Series]:
    """항목별 복원 오차(창 평균, 정규화 단위). 창 끝 시각에 붙는다 → 경보는 그 시각에 울린다."""
    out = {}
    model.eval()
    for l, loc in enumerate(data.locations):
        t = np.arange(W - 1, len(data.index))
        err = np.full((len(data.index), 3), np.nan, np.float32)
        with torch.no_grad():
            for i in range(0, len(t), 8192):
                tt = t[i:i + 8192]
                x, v, m = data.windows(l, tt)
                r = model(torch.from_numpy(x).to(device)).cpu().numpy()
                e = (np.abs(r - v) * m).reshape(len(tt), W, 3)
                mm = m.reshape(len(tt), W, 3)
                with np.errstate(invalid="ignore", divide="ignore"):
                    err[tt] = np.where(mm.sum(1) >= W * 0.8, e.sum(1) / mm.sum(1), np.nan)
        for j, f in enumerate(prep.FIELDS):
            if data.M[l, :, j].any():
                out[(loc, f)] = pd.Series(err[:, j], index=data.index)
    return out


# ---------- 경보·평가 ----------

def onsets(flag: pd.Series) -> pd.DatetimeIndex:
    """경보 시작 시각. 1시간 안에 다시 울린 것은 합친다."""
    f = flag.fillna(False).astype(bool).to_numpy()
    out, last = [], -10 ** 9
    for i in np.flatnonzero(f):
        if i - last > MERGE:
            out.append(i)
        last = i
    return flag.index[out]


def pick_threshold(score: pd.Series, budget: float) -> float:
    """검증 기간 경보 수가 budget(건) 이하가 되는 가장 낮은 임계값."""
    s = score.dropna()
    for q in (0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995, 1.0):
        thr = float(s.quantile(q)) * (1.0 if q < 1 else 1.05)
        if len(onsets(score > thr)) <= budget:
            return thr
    return float(s.max()) * 1.05


def alarms_from(flags: dict[tuple, pd.Series], cause_score: dict | None = None) -> pd.DataFrame:
    rows = []
    for key, fl in flags.items():
        for t in onsets(fl):
            rows.append({"location": key[0], "field": key[1], "detector": key[2] if len(key) > 2 else "ae", "time": t})
    return pd.DataFrame(rows, columns=["location", "field", "detector", "time"])


def active_from(flags: dict[tuple, pd.Series]) -> pd.DataFrame:
    """경보가 켜져 있는 모든 칸. 탐지 판정용: 고장 직전에 이미 켜져 있던 경보도 고장 중에 켜져 있으면 잡은 것으로 본다."""
    rows = [pd.DataFrame({"location": k[0], "field": k[1], "detector": k[2] if len(k) > 2 else "ae", "time": fl.index[fl.fillna(False).astype(bool).to_numpy()]})
            for k, fl in flags.items()]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["location", "field", "detector", "time"])


def merge_alarms(a: pd.DataFrame) -> pd.DataFrame:
    """같은 공간에서 1시간 안에 겹친 경보는 하나로(운영자에게 가는 알림 수 기준). 원인 항목은 첫 경보의 것."""
    out = []
    for loc, g in a.sort_values("time").groupby("location"):
        last = None
        for r in g.itertuples():
            if last is None or r.time - last > STEP * MERGE:
                out.append(r._asdict())
            last = r.time
    return pd.DataFrame(out, columns=["location", "field", "detector", "time"] + (["Index"] if out and "Index" in out[0] else []))


def evaluate(alarms: pd.DataFrame, events: list[Event], start, end, raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, float]:
    """이벤트별 탐지·지연(raw: 합치기 전 경보), 그리고 이벤트와 무관한 알림(헛경보, alarms: 합친 뒤) 하루당 수."""
    rows, matched = [], set()
    det = alarms if raw is None else raw
    for k, e in enumerate(events):
        a = det[(det.location == e.location) & (det.time >= e.start) & (det.time <= e.end + pd.Timedelta("30min"))]
        hit = len(a) > 0
        first = a.sort_values("time").iloc[0] if hit else None
        rows.append({"event": k, "location": e.location, "fault": e.fault, "start": e.start, "detected": hit,
                     "delay_min": (first.time - e.start).total_seconds() / 60 if hit else np.nan,
                     "cause_ok": bool(hit and first.field in e.fields),
                     "by": first.detector if hit else ""})
    # 이벤트 앞뒤 여유(고장 직전 직후 경보도 그 이벤트 탓으로 봄) 밖의 경보 = 헛경보
    fa = 0
    for _, r in alarms.iterrows():
        near = any(r.location == e.location and e.start - pd.Timedelta("1h") <= r.time <= e.end + pd.Timedelta("2h") for e in events)
        fa += 0 if near else 1
    days = (end - start).total_seconds() / 86400
    return pd.DataFrame(rows), fa / days


def scorecard(ev: pd.DataFrame) -> pd.DataFrame:
    t = ev.groupby("fault").agg(events=("detected", "size"), detected=("detected", "sum"),
                                delay_min=("delay_min", "median")).reindex(list(FAULTS)).dropna(how="all")
    t.loc["전체"] = [ev["detected"].size, ev["detected"].sum(), ev["delay_min"].median()]
    t["rate_%"] = t["detected"] / t["events"] * 100
    return t


def main():
    ap = argparse.ArgumentParser()
    d = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--task2", default=os.path.join(d, "reports", "task2"))
    ap.add_argument("--out", default=os.path.join(d, "reports", "task4"))
    ap.add_argument("--per-type", type=int, default=4, help="공간 x 고장 종류마다 주입할 건수")
    ap.add_argument("--ae-budget", type=float, default=0.4, help="오토인코더 헛경보 예산(전체 공간 합, 하루)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)

    grid, norm, split = load_task2(a.task2)
    locations = sorted(grid.columns.get_level_values("location").unique())
    idx = grid.index
    tr_i, va_i = int((idx < split.train_end).sum()), int((idx < split.val_end).sum())
    val_days = (split.val_end - split.train_end).total_seconds() / 86400
    test_start, test_end = split.val_end, idx[-1]

    # 1) 오토인코더: 학습 기간으로 배우고 검증 기간으로 조기 종료
    clean = AEData(grid, norm, locations)
    model = train_ae(clean, tr_i, va_i, a.epochs, a.seed, device)
    torch.save({"state_dict": model.state_dict(), "locations": locations, "W": W}, out / "ae.pt")

    # 2) 임계값: 검증 기간(주입 없음) 헛경보 예산. 항목 수로 나눠 공간·항목마다 같은 몫
    sc_clean = ae_scores(model, clean, device)
    val_mask = (idx >= split.train_end) & (idx < split.val_end)
    per = a.ae_budget * val_days / max(len(sc_clean), 1)
    thr = {k: pick_threshold(s[val_mask], per) for k, s in sc_clean.items()}
    pd.Series(thr).rename("threshold").to_csv(out / "thresholds.csv")
    night_ref = night_reference(grid, split.train_end)

    # 규칙의 검증 기간 헛경보(주입 없음): 규칙 기준이 실데이터에서 얼마나 조용한지
    val_rules = alarms_from({k: v[val_mask] for k, v in rule_scores(grid, night_ref).items()})
    val_rule_fa = len(merge_alarms(val_rules)) / val_days if len(val_rules) else 0.0

    # 3) 시험 기간에 고장 주입
    g_inj, events = inject(grid, test_start, a.per_type, a.seed)
    pd.DataFrame([e.__dict__ for e in events]).to_csv(out / "events.csv", index=False)
    inj = AEData(g_inj, norm, locations)
    sc = ae_scores(model, inj, device)
    test_mask = idx >= test_start
    ae_flags = {(k[0], k[1], "ae"): (s > thr[k])[test_mask] for k, s in sc.items()}
    rule_flags = {k: v[test_mask] for k, v in rule_scores(g_inj, night_ref).items()}

    ae_raw, ru_raw = active_from(ae_flags), active_from(rule_flags)      # 탐지 판정용(켜져 있는 칸 전부)
    ae_al = merge_alarms(alarms_from(ae_flags)).reset_index(drop=True)
    ru_al = merge_alarms(alarms_from(rule_flags)).reset_index(drop=True)
    both = merge_alarms(pd.concat([ae_al, ru_al], ignore_index=True)).reset_index(drop=True)
    pd.concat([ae_al.assign(set="ae"), ru_al.assign(set="rule"), both.assign(set="combined")]).to_csv(out / "alarms.csv", index=False)

    cards, summary, evs = {}, [], {}
    both_raw = pd.concat([ae_raw, ru_raw], ignore_index=True)
    for name, al, raw in (("모델(AE)", ae_al, ae_raw), ("규칙", ru_al, ru_raw), ("결합", both, both_raw)):
        ev, fa = evaluate(al, events, test_start, test_end, raw)
        evs[name] = ev
        cards[name] = scorecard(ev)
        summary.append({"detector": name, "events": len(ev), "detected": int(ev.detected.sum()),
                        "rate_%": ev.detected.mean() * 100, "median_delay_min": ev.delay_min.median(),
                        "false_alarms_per_day": fa, "cause_ok_%": ev[ev.detected].cause_ok.mean() * 100 if ev.detected.any() else np.nan})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    evs["결합"].to_csv(out / "event_results.csv", index=False)

    # 4) 임계값 배율을 바꿔 가며(오토인코더만, 규칙은 고정) 탐지율·헛경보
    sweep = []
    for mult in (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0):
        fl = {(k[0], k[1], "ae"): (s > thr[k] * mult)[test_mask] for k, s in sc.items()}
        al_raw = active_from(fl)
        al = merge_alarms(alarms_from(fl)).reset_index(drop=True)
        ev_ae, fa_ae = evaluate(al, events, test_start, test_end, al_raw)
        raw = pd.concat([al_raw, ru_raw], ignore_index=True)
        ev_c, fa_c = evaluate(merge_alarms(raw).reset_index(drop=True), events, test_start, test_end, raw)
        sweep.append({"multiplier": mult, "ae_rate_%": ev_ae.detected.mean() * 100, "ae_fa_per_day": fa_ae,
                      "combined_rate_%": ev_c.detected.mean() * 100, "combined_fa_per_day": fa_c})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "sweep.csv", index=False)

    # 5) 보고서
    fmt = lambda v: f"{v:.1f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("-" if isinstance(v, float) else str(v))  # noqa: E731

    def table(df: pd.DataFrame, index_name: str | None = None) -> list[str]:
        df = df.reset_index() if index_name else df
        cols = list(df.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
        return lines

    comb = summary[summary.detector == "결합"].iloc[0]
    L = ["# 과제 4. 이상 탐지\n"]
    L.append(f"- 시험 기간 {test_start:%Y-%m-%d} ~ {test_end:%Y-%m-%d} 에 고장 {len(events)}건을 주입(공간 x 종류마다 최대 {a.per_type}건). 주입 기록: events.csv")
    L.append(f"- 오토인코더: 1시간 창(12칸) x CO₂·온도·습도, 학습 기간으로 학습. 임계값은 검증 기간 헛경보가 하루 {a.ae_budget}회(전체 공간 합) 이하가 되게 공간·항목마다 정함(thresholds.csv)")
    L.append("- 규칙: 튄 값(앞뒤 칸과 CO₂ 200ppm·온도 2℃ 이상 차이), 값 고정(CO₂ 30분, 온도+습도 1시간 같은 값), 새벽 기준선(02~05시 CO₂ 가 학습 기간 새벽 99% 분위+60ppm 초과)")
    L.append(f"- 규칙의 검증 기간(주입 없음) 경보: 하루 {val_rule_fa:.2f}회\n")
    L.append("## 확인 기준\n")
    L.append(f"- 결합 이벤트 탐지율 {comb['rate_%']:.1f}% → {'통과' if comb['rate_%'] >= 80 else '미달'} (기준 80%)")
    L.append(f"- 결합 헛경보 하루 {comb['false_alarms_per_day']:.2f}회 → {'통과' if comb['false_alarms_per_day'] <= 1 else '미달'} (기준 1회, 시험 기간의 진짜 이상도 헛경보로 셈)\n")
    L.append("## 요약\n")
    L += table(summary)
    L.append("\n## 종류별 성적표 (탐지 / 건수, 지연 중앙값 분)\n")
    rows = []
    for f in list(FAULTS) + ["전체"]:
        r = {"고장": KO.get(f, f)}
        for name, c in cards.items():
            if f in c.index:
                r[name] = f"{int(c.loc[f, 'detected'])}/{int(c.loc[f, 'events'])} ({c.loc[f, 'delay_min']:.0f}분)" if c.loc[f, "detected"] else f"0/{int(c.loc[f, 'events'])}"
        rows.append(r)
    L += table(pd.DataFrame(rows))
    L.append("\n## 공간별 (결합)\n")
    ev = evs["결합"]
    L += table(ev.groupby(["location", "fault"]).agg(events=("detected", "size"), detected=("detected", "sum")).reset_index())
    L.append("\n## 임계값 배율에 따른 탐지율·헛경보 (오토인코더 임계값 x 배율, 규칙 고정)\n")
    L += table(sweep)
    L.append("\n## 원인 가리키기\n")
    L.append(f"- 결합 탐지에서 첫 경보가 가리킨 항목이 주입한 항목과 맞은 비율: {comb['cause_ok_%']:.1f}%")
    L.append("- 경보마다 공간·항목·탐지기를 alarms.csv 에 남긴다. 오토인코더는 복원 오차가 임계값을 넘은 항목을 원인으로 가리킨다.")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
