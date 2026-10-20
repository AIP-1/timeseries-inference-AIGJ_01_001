"""과제 3: 60분(5분 x 12칸) CO2·온도 예측. GRU 하나를 모든 공간에 같이 학습한다.

  python -m analysis.task3 [--task2 data/reports/task2] [--out data/reports/task3]
  python -m analysis.task3 --schedule data/schedule.csv     # 일정 자료가 생기면

입력 설계(근거는 docs/task3.md):
  - 과거 2시간(24칸)의 CO2·온도·습도(공간별 학습 기간 평균·표준편차로 정규화) + 결측 표시
  - 시각·요일(sin/cos), 주말 여부, 공간 one-hot
  - (선택) 일정: 지금과 앞으로 60분 동안 일정이 있는지. 미래에 미리 알 수 있는 정보라 출력 쪽에 넣는다
출력: 앞으로 12칸 각각의 '지금 값에서의 변화량'(CO2, 온도)을 한 번에 낸다(직접 다중 출력).
평가: 과제 2와 같은 시점(baselines.eval_points ∩ 시험 기간). 시험 기간은 학습·조기 종료에 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import baselines, prep

TARGETS = ("co2", "temperature")
HORIZON = 12                        # 5분 x 12 = 60분


# ---------- 특징 만들기 ----------

def load_task2(task2_dir: str | Path):
    d = Path(task2_dir)
    grid = pd.read_parquet(d / "grid.parquet")
    grid.columns = pd.MultiIndex.from_tuples(grid.columns, names=["location", "field"])
    norm = pd.read_csv(d / "norm.csv", index_col=[0, 1])
    sp = json.loads((d / "split.json").read_text())
    split = prep.Split(train_end=pd.Timestamp(sp["val"][0]), val_end=pd.Timestamp(sp["test"][0]),
                       end=pd.Timestamp(sp["test"][1]))
    return grid, norm, split


def load_schedule(path: str | Path, index: pd.DatetimeIndex, locations: list[str]) -> np.ndarray:
    """일정 CSV(location,start,end; KST 시각) → [공간, 시각] 0/1 배열."""
    s = pd.read_csv(path)
    out = np.zeros((len(locations), len(index)), dtype=np.float32)
    for r in s.itertuples():
        if r.location not in locations:
            continue
        a = pd.Timestamp(r.start).tz_localize(prep.TZ) if pd.Timestamp(r.start).tzinfo is None else pd.Timestamp(r.start)
        b = pd.Timestamp(r.end).tz_localize(prep.TZ) if pd.Timestamp(r.end).tzinfo is None else pd.Timestamp(r.end)
        out[locations.index(r.location), (index >= a) & (index < b)] = 1.0
    return out


def calendar(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert(prep.TZ)
    h = (local.hour + local.minute / 60).to_numpy() / 24 * 2 * np.pi
    d = local.dayofweek.to_numpy() / 7 * 2 * np.pi
    return np.stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d), (local.dayofweek >= 5).astype(float)], 1).astype(np.float32)


class Data:
    """공간 x 시각 x 특징 배열. 창(window)은 학습 때 인덱스로 잘라 쓴다(미리 다 만들면 보드 메모리가 모자람)."""

    def __init__(self, grid, norm, split, lookback=24, use_calendar=True, schedule=None, absolute=False):
        self.grid, self.split, self.lookback, self.absolute = grid, split, lookback, absolute
        self.index = grid.index
        self.locations = sorted(grid.columns.get_level_values("location").unique())
        nL, T = len(self.locations), len(self.index)
        cal = calendar(self.index)
        feats, raw, mean, std = [], np.full((nL, T, 2), np.nan, np.float32), np.zeros((nL, 2), np.float32), np.ones((nL, 2), np.float32)
        for li, loc in enumerate(self.locations):
            cols = []
            for f in prep.FIELDS:
                if (loc, f) in grid and (loc, f) in norm.index and norm.loc[(loc, f), "std"] > 0:
                    v = ((grid[(loc, f)] - norm.loc[(loc, f), "mean"]) / norm.loc[(loc, f), "std"]).to_numpy(np.float32)
                else:
                    v = np.full(T, np.nan, np.float32)
                cols += [np.nan_to_num(v, nan=0.0), (~np.isnan(v)).astype(np.float32)]
                if f in TARGETS and (loc, f) in grid:
                    j = TARGETS.index(f)
                    raw[li, :, j] = grid[(loc, f)].to_numpy(np.float32)
                    if (loc, f) in norm.index:
                        mean[li, j], std[li, j] = norm.loc[(loc, f), "mean"], max(norm.loc[(loc, f), "std"], 1e-3)
            onehot = np.zeros((T, nL), np.float32)
            onehot[:, li] = 1
            parts = [np.stack(cols, 1)]
            if use_calendar:
                parts.append(cal)
            parts.append(onehot)
            if schedule is not None:
                parts.append(schedule[li][:, None])
            feats.append(np.concatenate(parts, 1))
        self.X = np.stack(feats)                     # [L, T, F]
        self.raw, self.mean, self.std = raw, mean, std
        self.schedule = schedule
        self.n_static = HORIZON if schedule is not None else 0

        # 목표: 변화량(기본) 또는 절대값, 공간별 표준편차 단위
        fut = np.stack([np.roll(raw, -k, axis=1) for k in range(1, HORIZON + 1)], 2)   # [L, T, H, 2]
        fut[:, T - HORIZON:] = np.nan
        base = mean[:, None, None, :] if absolute else raw[:, :, None, :]
        self.Y = (fut - base) / std[:, None, None, :]
        self.Ymask = ~np.isnan(self.Y)
        self.Y = np.nan_to_num(self.Y, nan=0.0).astype(np.float32)

    def candidates(self, part: str) -> tuple[np.ndarray, np.ndarray]:
        """학습/검증 표본 (공간, 시각). 정답 60분이 모두 같은 구간 안에 있어야 한다(구간 경계 넘는 누수 방지)."""
        lab = self.split.label(self.index)
        t = np.arange(len(self.index))
        last = np.minimum(t + HORIZON, len(t) - 1)
        same = (lab == part) & (lab[last] == part) & (t >= self.lookback - 1)
        li, ti = [], []
        for l in range(len(self.locations)):
            ok = same & ~np.isnan(self.raw[l, :, 0]) & ~np.isnan(self.raw[l, :, 1]) & self.Ymask[l].any(axis=(1, 2))
            idx = np.flatnonzero(ok)
            li.append(np.full(len(idx), l))
            ti.append(idx)
        return np.concatenate(li), np.concatenate(ti)

    def batch(self, li, ti, device):
        offs = np.arange(-self.lookback + 1, 1)
        x = torch.from_numpy(self.X[li[:, None], ti[:, None] + offs[None, :]]).to(device)
        if self.schedule is not None:
            fidx = np.minimum(ti[:, None] + np.arange(1, HORIZON + 1)[None, :], len(self.index) - 1)
            s = torch.from_numpy(self.schedule[li[:, None], fidx]).to(device)
        else:
            s = torch.zeros((len(li), 0), device=device)
        y = torch.from_numpy(self.Y[li, ti]).to(device)
        m = torch.from_numpy(self.Ymask[li, ti]).to(device)
        return x, s, y, m

    def to_values(self, li, ti, out: np.ndarray) -> np.ndarray:
        """모델 출력(정규화 단위) → ppm, ℃. [N, H, 2]"""
        base = self.mean[li][:, None, :] if self.absolute else self.raw[li, ti][:, None, :]
        return base + out * self.std[li][:, None, :]


# ---------- 모델 ----------

class GRUForecaster(nn.Module):
    def __init__(self, n_in, n_static=0, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Linear(hidden, HORIZON * len(TARGETS)))

    def forward(self, x, s):
        _, h = self.gru(x)
        return self.head(torch.cat([h[-1], s], 1)).view(-1, HORIZON, len(TARGETS))


def masked_mae(pred, y, m):
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp(min=1)


def train(data: Data, args, device, log: list, name: str) -> GRUForecaster:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr_l, tr_t = data.candidates("train")
    va_l, va_t = data.candidates("val")
    if len(tr_t) == 0 or len(va_t) == 0:
        raise SystemExit("학습/검증 표본이 없음")
    if len(va_t) > args.max_val:
        pick = rng.choice(len(va_t), args.max_val, replace=False)
        va_l, va_t = va_l[pick], va_t[pick]
    model = GRUForecaster(data.X.shape[2], data.n_static, args.hidden, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    steps = min(args.steps, max(1, len(tr_t) // args.batch))
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for _ in range(steps):
            pick = rng.integers(0, len(tr_t), args.batch)
            x, s, y, m = data.batch(tr_l[pick], tr_t[pick], device)
            loss = masked_mae(model(x, s), y, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vs, vn = 0.0, 0.0
            for i in range(0, len(va_t), 4096):
                x, s, y, m = data.batch(va_l[i:i + 4096], va_t[i:i + 4096], device)
                vs += (torch.abs(model(x, s) - y) * m).sum().item()
                vn += m.sum().item()
        val = vs / max(vn, 1)
        log.append({"variant": name, "epoch": ep, "train_mae": tot / steps, "val_mae": val, "sec": round(time.time() - t0, 1)})
        print(f"[{name}] epoch {ep} train {tot / steps:.4f} val {val:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, data: Data, l: int, ti: np.ndarray, device) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(ti), 4096):
            t = ti[i:i + 4096]
            li = np.full(len(t), l)
            x, s, _, _ = data.batch(li, t, device)
            outs.append(data.to_values(li, t, model(x, s).cpu().numpy()))
    return np.concatenate(outs) if outs else np.zeros((0, HORIZON, 2))


def evaluate(model, data: Data, name: str, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시험 기간, 과제 2와 같은 평가 시점에서 기준선과 모델의 MAE. 60분 CO2 오차는 시각별로도 남긴다."""
    test = data.index[data.index >= data.split.val_end]
    pos = pd.Series(np.arange(len(data.index)), index=data.index)
    rows, hourly = [], []
    for l, loc in enumerate(data.locations):
        for j, field in enumerate(TARGETS):
            if (loc, field) not in data.grid:
                continue
            s = data.grid[(loc, field)]
            pts = baselines.eval_points(s).intersection(test)
            pts = pts[pos[pts].to_numpy() >= data.lookback - 1]
            if len(pts) == 0:
                continue
            ti = pos[pts].to_numpy()
            pred = predict(model, data, l, ti, device)[:, :, j]
            for h in baselines.HORIZONS_MIN:
                k = h // baselines.STEP_MIN
                p = baselines.predictions(s, h).loc[pts]
                err = np.abs(pred[:, k - 1] - p["target"].to_numpy())
                rows.append({"location": loc, "field": field, "horizon_min": h, "model": name,
                             "mae": float(err.mean()), "n": len(pts)})
                if h == 60:
                    hr = pts.tz_convert(prep.TZ).hour
                    pe = np.abs(p["persistence"] - p["target"]).to_numpy()
                    hourly.append(pd.DataFrame({"location": loc, "field": field, "model": name, "hour": hr,
                                                "weekday": pts.tz_convert(prep.TZ).dayofweek < 5,
                                                "err": err, "persistence_err": pe}))
    return pd.DataFrame(rows), (pd.concat(hourly) if hourly else pd.DataFrame())


def baseline_rows(data: Data) -> pd.DataFrame:
    test = data.index[data.index >= data.split.val_end]
    pos = pd.Series(np.arange(len(data.index)), index=data.index)
    rows = []
    for loc in data.locations:
        for field in TARGETS:
            if (loc, field) not in data.grid:
                continue
            s = data.grid[(loc, field)]
            pts = baselines.eval_points(s).intersection(test)
            pts = pts[pos[pts].to_numpy() >= data.lookback - 1]
            for h in baselines.HORIZONS_MIN:
                p = baselines.predictions(s, h).loc[pts]
                for m in ("persistence", "mean_1h", "yesterday"):
                    rows.append({"location": loc, "field": field, "horizon_min": h, "model": m,
                                 "mae": float(np.abs(p[m] - p["target"]).mean()), "n": len(pts)})
    return pd.DataFrame(rows)


# ---------- 보고서 ----------

def pooled(t: pd.DataFrame) -> pd.DataFrame:
    """공간을 합친 MAE(평가 시점 수로 가중)."""
    t = t.assign(w=t["mae"] * t["n"])
    g = t.groupby(["field", "model", "horizon_min"]).agg(w=("w", "sum"), n=("n", "sum")).reset_index()
    g["mae"] = g["w"] / g["n"]
    return g.drop(columns="w")


def md_pivot(t: pd.DataFrame, index, fmt: str) -> str:
    piv = t.pivot_table(index=index, columns="horizon_min", values="mae").reset_index()
    cols = [f"{c}분" if isinstance(c, (int, np.integer)) else c for c in piv.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in piv.iterrows():
        out.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def gain(t: pd.DataFrame, field: str, model: str, ref: str = "persistence", h: int = 60) -> float:
    a = t[(t.field == field) & (t.model == model) & (t.horizon_min == h)]["mae"]
    b = t[(t.field == field) & (t.model == ref) & (t.horizon_min == h)]["mae"]
    return float(1 - a.iloc[0] / b.iloc[0]) * 100 if len(a) and len(b) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    data_dir = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--task2", default=os.path.join(data_dir, "reports", "task2"))
    ap.add_argument("--out", default=os.path.join(data_dir, "reports", "task3"))
    ap.add_argument("--schedule", help="일정 CSV: location,start,end (KST)")
    ap.add_argument("--variants", default="", help="쉼표 구분. 기본: full,no_calendar (+일정 있으면 no_schedule), 추가로 absolute")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300, help="한 epoch 의 미니배치 수(무작위 추출)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-val", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)

    grid, norm, split = load_task2(a.task2)
    locations = sorted(grid.columns.get_level_values("location").unique())
    sched = load_schedule(a.schedule, grid.index, locations) if a.schedule else None
    variants = [v for v in a.variants.split(",") if v] or (["full", "no_schedule"] if sched is not None else ["full", "no_calendar"])
    print(f"device={device} locations={locations} variants={variants}", flush=True)

    log, results, hourly, base = [], [], [], None
    for v in variants:
        data = Data(grid, norm, split, a.lookback, use_calendar=(v != "no_calendar"),
                    schedule=None if v == "no_schedule" else sched, absolute=(v == "absolute"))
        if base is None:
            base = baseline_rows(data)
        model = train(data, a, device, log, v)
        torch.save({"state_dict": model.state_dict(), "variant": v, "locations": data.locations,
                    "lookback": a.lookback, "hidden": a.hidden, "layers": a.layers,
                    "n_in": data.X.shape[2], "n_static": data.n_static,
                    "norm": norm.reset_index().to_dict("records")}, out / f"gru_{v}.pt")
        r, hr = evaluate(model, data, f"gru_{v}", device)
        results.append(r)
        hourly.append(hr)
        del data

    allr = pd.concat([base] + results, ignore_index=True)
    allr.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    hourly = pd.concat(hourly, ignore_index=True)
    pool = pooled(allr)
    main_m = f"gru_{variants[0]}"

    L = ["# 과제 3. 60분 예측\n"]
    L.append(f"- 모델: GRU({a.layers}층, {a.hidden}), 과거 {a.lookback * 5}분, 12칸 변화량 직접 출력, 모든 공간 한 모델")
    L.append(f"- 학습 ~{split.train_end:%Y-%m-%d}, 조기 종료는 검증 기간 ~{split.val_end:%Y-%m-%d}, 평가는 시험 기간만(과제 2와 같은 시점)")
    L.append(f"- 장치 {device}, 변형 {', '.join(variants)}\n")
    L.append("## 확인 기준\n")
    g_co2, g_t = gain(pool, "co2", main_m), gain(pool, "temperature", main_m)
    L.append(f"- 60분 CO₂: 현재값 유지 대비 {g_co2:.1f}% 감소 → {'통과' if g_co2 >= 30 else '미달'} (기준 30%)")
    L.append(f"- 60분 온도: 현재값 유지 대비 {g_t:.1f}% 감소 → {'통과' if g_t > 0 else '미달'}")
    L.append("- 시험 기간: 학습 표본·조기 종료 모두 학습/검증 구간에서만 고름(정답 60분까지 구간 안)\n")
    for field, fmt, unit in (("co2", ".1f", "ppm"), ("temperature", ".3f", "℃")):
        L.append(f"## {field} 전체 MAE ({unit}, 공간 합산)\n")
        L.append(md_pivot(pool[pool.field == field], ["model"], fmt))
        L.append(f"\n### 공간별 60분 MAE ({unit})\n")
        sub = allr[(allr.field == field) & (allr.horizon_min == 60)].pivot_table(index="location", columns="model", values="mae")
        sub["gain_%"] = (1 - sub[main_m] / sub["persistence"]) * 100
        sub = sub.reset_index()
        cols = list(sub.columns)
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for _, r in sub.iterrows():
            L.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
        L.append("")
    L.append("## 60분 CO₂ 오차가 큰 시각 (평일, KST)\n")
    h = hourly[(hourly.field == "co2") & (hourly.model == main_m) & hourly.weekday]
    if len(h):
        top = h.groupby(["location", "hour"])["err"].mean().reset_index().sort_values("err", ascending=False)
        for loc, g in top.groupby("location"):
            L.append(f"- {loc}: " + ", ".join(f"{int(r.hour)}시 {r.err:.0f}ppm" for r in g.head(3).itertuples()))
        h.groupby(["location", "hour"])[["err", "persistence_err"]].mean().to_csv(out / "co2_err_by_hour.csv")
    if len(variants) > 1:
        L.append("\n## 정보 제거 비교 (60분, 공간 합산)\n")
        for v in variants[1:]:
            for field, unit in (("co2", "ppm"), ("temperature", "℃")):
                a_ = pool[(pool.field == field) & (pool.model == main_m) & (pool.horizon_min == 60)]["mae"].iloc[0]
                b_ = pool[(pool.field == field) & (pool.model == f"gru_{v}") & (pool.horizon_min == 60)]["mae"].iloc[0]
                L.append(f"- {v} 대비 {main_m}: {field} 60분 오차 {b_ - a_:.3f}{unit} 감소 ({b_:.3f} → {a_:.3f})")
        if sched is None:
            L.append("\n일정 자료가 없어 '일정 정보의 가치'는 아직 잴 수 없다. 대신 시각·요일 정보를 뺀 모델과 비교했다. "
                     "일정 CSV 가 생기면 `--schedule` 로 다시 돌리면 full/no_schedule 비교가 나온다.")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
