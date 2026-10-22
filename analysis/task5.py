"""과제 5: 상태 추정. (1) 환기 선제 경보 (2) 사용 여부 추정.

  python -m analysis.task5 [--task2 data/reports/task2] [--out data/reports/task5]
  python -m analysis.task5 --schedule data/schedule.csv --observed data/observed.csv   # 일정·관찰 기록이 생기면

(1) 환기 경보: 정답은 센서값만으로 만든다.
    양성 = 지금 CO2 < 1000ppm 이고 앞으로 30분(6칸) 안에 1000ppm 이상이 되는 시각.
    이미 1000 이상인 시각은 평가에서 뺀다: 넣으면 "지금 1000 이상이면 경보"라는 뻔한 규칙이 점수를 독차지해서
    '미리' 알려 주는 능력을 잴 수 없다(이미 넘은 뒤의 경보는 늦은 경보다).
(2) 사용 여부: 정답은 일정(수업·예약)에서 만들고 직접 관찰 기록으로 검증해야 한다. 둘 다 아직 없으면 이 부분은 건너뛴다.
    일정은 입력에 넣지 않는다: 정답을 일정에서 만들었는데 입력에도 넣으면 모델이 일정만 베껴 센서를 보지 않는다.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import prep
from .task3 import DAY, WEEK, calendar, load_schedule, load_task2

LIMIT, HORIZON, LOOK = 1000.0, 6, 12          # 1000ppm, 30분 앞, 과거 1시간


def features(grid: pd.DataFrame, norm: pd.DataFrame, loc: str, locations: list[str], use_calendar: bool = True,
             lags: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """[T, F]: 최근 1시간 CO2(1000ppm 기준 거리), 5·15·30분 변화, 온도·습도 현재와 30분 변화, 시각·쉬는 날, 공간.

    lags=True 면 하루 전·일주일 전 같은 시각의 CO2 와 그 뒤 30분 변화도 넣는다(이미 지난 값이라 누수 아님).
    수업·회의가 요일마다 되풀이되면 '지난주 이 시각엔 30분 뒤 1000을 넘었다'가 가장 직접적인 단서다.
    """
    T = len(grid.index)
    c = grid[(loc, "co2")].to_numpy(np.float64)
    cols = [(np.roll(c, k) - LIMIT) / 200 for k in range(LOOK)]
    cols += [(c - np.roll(c, k)) / 50 for k in (1, 3, 6)]
    for f in ("temperature", "humidity"):
        if (loc, f) in grid and (loc, f) in norm.index:
            v = ((grid[(loc, f)] - norm.loc[(loc, f), "mean"]) / max(norm.loc[(loc, f), "std"], 1e-3)).to_numpy(np.float64)
        else:
            v = np.zeros(T)
        cols += [v, v - np.roll(v, 6)]
    X = np.column_stack(cols)
    X[:LOOK] = np.nan
    extra = []
    if lags:
        for lag in (DAY, WEEK):
            past = np.full(T, np.nan)
            past[lag:] = c[:T - lag]
            fut = np.full((T, HORIZON), np.nan)
            for k in range(1, HORIZON + 1):
                fut[lag - k:, k - 1] = c[:T - lag + k]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # 지난 값이 모두 없는 칸
                peak = np.nanmax(fut, 1)
            Z = np.column_stack([(past - LIMIT) / 200, (peak - LIMIT) / 200, (fut[:, 2] - past) / 50, (fut[:, 5] - past) / 50])
            extra += [np.nan_to_num(Z, nan=0.0), np.isnan(Z).any(1, keepdims=True)]
    oh = np.zeros((T, len(locations)))
    oh[:, locations.index(loc)] = 1
    cal = calendar(grid.index) if use_calendar else np.zeros((T, 0))
    return np.column_stack([np.nan_to_num(X, nan=0.0), np.isnan(X).any(1), *extra, cal, oh]).astype(np.float32), \
        ~np.isnan(X[:, :LOOK + 3]).any(1)


def vent_labels(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(정답, 평가 가능). 평가 가능 = 지금 값이 있고 1000 미만, 앞으로 30분 값이 모두 있음."""
    c = s.to_numpy(np.float64)
    fut = np.column_stack([np.roll(c, -k) for k in range(1, HORIZON + 1)])
    fut[len(c) - HORIZON:] = np.nan
    ok = ~np.isnan(c) & (c < LIMIT) & ~np.isnan(fut).any(1)
    return (np.nan_to_num(fut, nan=0).max(1) >= LIMIT) & ok, ok


class MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit(Xtr, ytr, Xva, yva, seed=0, epochs=40):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1])
    pos = max(ytr.mean(), 1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))   # 불균형: 양성에 가중치
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva.astype(np.float32))
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for _ in range(200):
            p = torch.from_numpy(rng.integers(0, len(xt), 1024))
            loss = lossf(model(xt[p]), yt[p])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(xv), yv).item()
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(state)
    return model


def prob(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def metrics(y, p) -> dict:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn, tn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    f1p = 2 * pre * rec / max(pre + rec, 1e-9)
    rec_n, pre_n = tn / max(tn + fp, 1), tn / max(tn + fn, 1)
    f1n = 2 * pre_n * rec_n / max(pre_n + rec_n, 1e-9)
    return {"accuracy": (tp + tn) / len(y), "recall": rec, "precision": pre, "f1_pos": f1p, "macro_f1": (f1p + f1n) / 2,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def pick_threshold(y, pr, min_recall=0.85) -> float:
    """검증 기간: 재현율 85% 이상인 임계값 중 macro-F1 이 가장 큰 것."""
    best, thr = -1, 0.5
    for t in np.unique(np.quantile(pr, np.linspace(0.5, 0.999, 300))):
        m = metrics(y, pr >= t)
        if m["recall"] >= min_recall and m["macro_f1"] > best:
            best, thr = m["macro_f1"], float(t)
    return thr


def md(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return out


def ventilation(grid, norm, split, out: Path, L: list[str]):
    locs = sorted(l for l in grid.columns.get_level_values("location").unique() if (l, "co2") in grid)
    lab = split.label(grid.index)
    parts = {"train": [], "val": [], "test": []}
    for loc in locs:
        X, xok = features(grid, norm, loc, locs)
        XL, _ = features(grid, norm, loc, locs, lags=True)
        y, ok = vent_labels(grid[(loc, "co2")])
        ok &= xok
        # 정답 30분이 다른 구간으로 넘어가는 시각은 뺀다(누수 방지)
        nxt = np.roll(lab, -HORIZON)
        for part in parts:
            m = ok & (lab == part) & (nxt == part)
            parts[part].append(pd.DataFrame({"loc": loc, "t": grid.index[m], "y": y[m],
                                             "co2": grid[(loc, "co2")].to_numpy()[m]}).assign(_X=list(X[m]), _XL=list(XL[m])))
    D = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    stack = lambda k, c: np.stack(D[k][c].to_numpy()) if len(D[k]) else np.zeros((0, 1), np.float32)  # noqa: E731
    L.append("## 1. 환기 선제 경보 (30분 안에 1000ppm 초과)\n")
    L.append("| 구간 | 평가 시점 | 양성 | 양성 비율 |")
    L.append("|---|---|---|---|")
    for k in ("train", "val", "test"):
        L.append(f"| {k} | {len(D[k])} | {int(D[k].y.sum())} | {D[k].y.mean() * 100:.2f}% |")
    if D["train"].y.sum() < 20 or D["test"].y.sum() == 0:
        L.append("\n양성이 너무 적어 학습·평가하지 않음.")
        return
    yv = D["val"].y.to_numpy()
    # 입력 두 가지(최근 1시간만 / + 하루 전·일주일 전)를 각각 학습하고, 검증 기간 macro-F1 로 하나를 고른다(시험 기간은 보지 않음)
    cand = {}
    for lags, col in ((False, "_X"), (True, "_XL")):
        Xs = {k: stack(k, col) for k in D}
        m_ = fit(Xs["train"], D["train"].y.to_numpy(), Xs["val"], yv)
        pv = prob(m_, Xs["val"])
        t_ = pick_threshold(yv, pv)
        cand[lags] = (m_, t_, Xs, metrics(yv, pv >= t_)["macro_f1"])
    lags = max(cand, key=lambda k: cand[k][3])
    model, thr, Xs, _ = cand[lags]
    torch.save({"state_dict": model.state_dict(), "n_in": Xs["train"].shape[1], "threshold": thr, "locations": locs, "lags": lags},
               out / "vent.pt")                                     # 과제 8 서비스가 그대로 불러 쓴다
    # 규칙 임계값(850ppm)은 과제가 준 값. 같은 방식으로 검증 기간에서 고른 규칙도 함께 본다(공정 비교)
    rule_best, rbest = 850.0, -1
    for c in range(700, 1000, 10):
        m = metrics(D["val"].y, D["val"].co2 >= c)
        if m["recall"] >= 0.85 and m["macro_f1"] > rbest:
            rbest, rule_best = m["macro_f1"], float(c)
    pt = prob(model, Xs["test"])
    yt = D["test"].y.to_numpy()
    o = cand[not lags]
    rows = [{"방법": f"모델(MLP{', 하루·일주일 전 포함' if lags else ''}) ← 검증 기간에서 고름", **metrics(yt, pt >= thr)},
            {"방법": f"모델(MLP{', 하루·일주일 전 포함' if not lags else ''})", **metrics(yt, prob(o[0], o[2]["test"]) >= o[1])},
            {"방법": "규칙 CO₂≥850", **metrics(yt, D["test"].co2 >= 850)},
            {"방법": f"규칙 CO₂≥{rule_best:.0f}(검증 기간에서 고름)", **metrics(yt, D["test"].co2 >= rule_best)},
            {"방법": "항상 '아님'", **metrics(yt, np.zeros(len(yt), bool))}]
    res = pd.DataFrame(rows)
    res.to_csv(out / "vent_metrics.csv", index=False)
    L.append(f"\n모델 임계값 {thr:.3f}(검증 기간에서 재현율 85% 이상 중 macro-F1 최대). 입력 두 가지 중 검증 기간 macro-F1 "
             f"{cand[False][3]:.3f}(최근 1시간만) vs {cand[True][3]:.3f}(하루·일주일 전 포함) → {'포함' if lags else '최근 1시간만'} 사용. 시험 기간 결과:\n")
    L += md(res[["방법", "accuracy", "recall", "precision", "macro_f1", "tp", "fp", "fn", "tn"]])
    L.append("\n- '항상 아님'의 정확도가 가장 높게 나온다. 양성이 1~2%뿐이라 정확도는 의미가 없고, 재현율·macro-F1 로 본다.")
    m = res.iloc[0]
    r = res.iloc[2]
    L.append(f"- 확인 기준: 재현율 {m.recall * 100:.1f}% → {'통과' if m.recall >= 0.85 else '미달'} (85%), "
             f"macro-F1 {m.macro_f1:.3f} vs 규칙 {r.macro_f1:.3f} → {'통과' if m.macro_f1 > r.macro_f1 else '미달'}")
    # 혼동 사례: 놓친 양성·헛경보가 어느 시각·공간에 몰리는지
    T = D["test"].assign(p=pt >= thr)
    T["hour"] = T.t.dt.tz_convert(prep.TZ).dt.hour
    fn = T[T.y & ~T.p].groupby(["loc", "hour"]).size().sort_values(ascending=False).head(5)
    fp = T[~T.y & T.p].groupby(["loc", "hour"]).size().sort_values(ascending=False).head(5)
    L.append("\n### 가장 많이 틀린 경우 (시험 기간)\n")
    L.append("- 놓친 경보(공간, 시각: 건수): " + ", ".join(f"{a} {b}시: {n}" for (a, b), n in fn.items()))
    L.append("- 헛경보(공간, 시각: 건수): " + ", ".join(f"{a} {b}시: {n}" for (a, b), n in fp.items()))
    T.drop(columns=["_X", "_XL"]).to_csv(out / "vent_test_predictions.csv", index=False)


def occupancy(grid, norm, split, schedule_path, observed_path, out: Path, L: list[str]):
    L.append("\n## 2. 사용 여부 추정\n")
    if not schedule_path:
        L.append("일정(수업 시간표·회의실 예약) 자료가 없어 정답을 만들 수 없다. 받으면 `--schedule`(정답), "
                 "`--observed`(직접 관찰 기록, location,start,end,state) 로 다시 실행한다.\n")
        L.append("- 대신 규칙 '최근 15분 CO₂ 30ppm 이상 상승 = 사용중'의 결과만 남긴다(occupancy_rule.csv).")
        rows = []
        for loc in sorted(grid.columns.get_level_values("location").unique()):
            if (loc, "co2") in grid:
                s = grid[(loc, "co2")]
                rows.append(pd.DataFrame({"location": loc, "time": s.index, "rule_occupied": (s - s.shift(3)) >= 30}))
        if rows:
            pd.concat(rows).to_csv(out / "occupancy_rule.csv", index=False)
        return
    locs = sorted(l for l in grid.columns.get_level_values("location").unique() if (l, "co2") in grid)
    sched = load_schedule(schedule_path, grid.index, locs)
    lab = split.label(grid.index)
    res_all = []
    for use_cal, name in ((False, "모델(센서만)"), (True, "모델(센서+시각)")):
        res_all.append(_occupancy_once(grid, norm, lab, locs, sched, use_cal, name, observed_path))
    res = pd.concat(res_all).drop_duplicates(subset=["방법", "정답"])
    res["empty_as_occupied_rate"] = res["fp"] / (res["fp"] + res["tn"]).clip(lower=1)   # 빈 방을 사용중으로 본 비율
    res.to_csv(out / "occupancy_metrics.csv", index=False)
    L.append("- 모델(센서만): CO₂·온도·습도 변화만. 모델(센서+시각): 시각·요일도 넣음. 시간표가 매주 같으면 시각만으로도 일정을 거의 맞히므로, "
             "센서가 얼마나 보태는지는 '센서만' 으로 본다. empty_as_occupied_rate = 빈 방을 사용중으로 본 비율(냉난방 자동 제어에서 가장 중요).\n")
    L += md(res[["방법", "정답", "accuracy", "f1_pos", "macro_f1", "empty_as_occupied_rate", "tp", "fp", "fn", "tn"]])


def _occupancy_once(grid, norm, lab, locs, sched, use_cal, name, observed_path) -> pd.DataFrame:
    D = {"train": [], "val": [], "test": []}
    for li, loc in enumerate(locs):
        X, ok = features(grid, norm, loc, locs, use_cal)    # 일정은 입력에 넣지 않는다
        s = grid[(loc, "co2")]
        rule = ((s - s.shift(3)) >= 30).to_numpy()
        for part in D:
            m = ok & (lab == part)
            D[part].append((X[m], sched[li][m] > 0, rule[m], grid.index[m], loc))
    cat = {k: (np.concatenate([d[0] for d in v]), np.concatenate([d[1] for d in v]), np.concatenate([d[2] for d in v]),
               np.concatenate([np.array(d[3]) for d in v]), np.concatenate([[d[4]] * len(d[3]) for d in v])) for k, v in D.items()}
    model = fit(cat["train"][0], cat["train"][1], cat["val"][0], cat["val"][1])
    thr = pick_threshold(cat["val"][1], prob(model, cat["val"][0]), min_recall=0.0)
    Xt, yt, rt, tt, lt = cat["test"]
    pt = prob(model, Xt) >= thr
    rows = [{"방법": name, "정답": "일정", **metrics(yt, pt)}, {"방법": "규칙 15분 +30ppm", "정답": "일정", **metrics(yt, rt)}]
    if observed_path:
        obs = pd.read_csv(observed_path)
        ty = np.full(len(tt), -1)
        ti = pd.DatetimeIndex(tt)
        for r in obs.itertuples():
            a, b = pd.Timestamp(r.start).tz_localize(prep.TZ), pd.Timestamp(r.end).tz_localize(prep.TZ)
            m = (lt == r.location) & (ti >= a) & (ti < b)
            ty[m] = 1 if str(r.state) in ("1", "occupied", "사용중") else 0
        k = ty >= 0
        if k.any():
            rows += [{"방법": name, "정답": "관찰", **metrics(ty[k] == 1, pt[k])},
                     {"방법": "규칙 15분 +30ppm", "정답": "관찰", **metrics(ty[k] == 1, rt[k])},
                     {"방법": "일정 자체", "정답": "관찰", **metrics(ty[k] == 1, yt[k])}]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    d = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--task2", default=os.path.join(d, "reports", "task2"))
    ap.add_argument("--out", default=os.path.join(d, "reports", "task5"))
    ap.add_argument("--schedule")
    ap.add_argument("--observed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 1)
    grid, norm, split = load_task2(a.task2)
    L = ["# 과제 5. 상태 추정\n"]
    ventilation(grid, norm, split, out, L)
    occupancy(grid, norm, split, a.schedule, a.observed, out, L)
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
