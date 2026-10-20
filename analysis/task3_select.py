"""과제 3 보완: 검증 기간으로 공간·항목·시점마다 가장 나은 예측을 고르고, 시험 기간에 한 번 평가한다.

  python -m analysis.task3_select [--task3 data/reports/task3]

다시 학습하지 않는다. 과제 3 이 저장한 gru_<변형>.pt 를 불러 예측만 한다.
후보: 각 GRU 변형, 현재값 유지, GRU 와 현재값 유지의 가중 평균(w = 0.25 / 0.5 / 0.75).
고르는 기준은 **검증 기간 MAE** 뿐이다(시험 기간은 고른 뒤 성적 매기기에만 씀 → 누수 없음).
왜 필요한가: 한 모델이 모든 공간에 맞지 않는다. 예) 사무실 밖 온도는 시각 정보가 오히려 해가 되고,
쉬는 날이 많은 공간은 현재값 유지와 섞는 편이 안전하다. 이것을 사람이 시험 결과를 보고 고르면 누수다.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import baselines, task3

WEIGHTS = (0.25, 0.5, 0.75)


def period_points(data: task3.Data, s: pd.Series, period: str) -> np.ndarray:
    pos = pd.Series(np.arange(len(data.index)), index=data.index)
    if period == "val":
        # 정답 60분까지 검증 기간 안
        win = data.index[(data.index >= data.split.train_end) & (data.index < data.split.val_end - pd.Timedelta("60min"))]
    else:
        win = data.index[data.index >= data.split.val_end]
    pts = baselines.eval_points(s).intersection(win)
    ti = pos[pts].to_numpy()
    return ti[ti >= data.lookback - 1]


def main():
    ap = argparse.ArgumentParser()
    d = os.environ.get("DATA_DIR", "data")
    ap.add_argument("--task2", default=os.path.join(d, "reports", "task2"))
    ap.add_argument("--task3", default=os.path.join(d, "reports", "task3"))
    a = ap.parse_args()
    out = Path(a.task3)
    torch.set_num_threads(os.cpu_count() or 1)
    grid, norm, split = task3.load_task2(a.task2)
    variants = sorted(p.stem[4:] for p in out.glob("gru_*.pt"))
    preds: dict = {}                     # (variant, loc, field, period) -> [N, H]
    pts: dict = {}
    for v in variants:
        ck = torch.load(out / f"gru_{v}.pt", map_location="cpu", weights_only=False)
        data = task3.Data(grid, norm, split, ck["lookback"], use_calendar=(v != "no_calendar"),
                          absolute=(v == "absolute"), use_lags=(v != "no_lags"))
        if data.X.shape[2] != ck["n_in"] or data.n_static != ck["n_static"]:
            print(f"{v}: 입력 크기 불일치, 건너뜀")
            continue
        model = task3.GRUForecaster(ck["n_in"], ck["n_static"], ck["hidden"], ck["layers"])
        model.load_state_dict(ck["state_dict"])
        for l, loc in enumerate(data.locations):
            for j, f in enumerate(task3.TARGETS):
                if (loc, f) not in grid:
                    continue
                for per in ("val", "test"):
                    ti = period_points(data, grid[(loc, f)], per)
                    if len(ti) == 0:
                        continue
                    pts[(loc, f, per)] = ti
                    preds[(v, loc, f, per)] = task3.predict(model, data, l, ti, torch.device("cpu"))[:, :, j]
        del data
    gv = [v for v in variants if any(k[0] == v for k in preds)]
    rows = []
    for (loc, f, per), ti in pts.items():
        s = grid[(loc, f)].to_numpy(np.float64)
        cur = s[ti]
        for h in baselines.HORIZONS_MIN:
            k = h // 5
            tgt = s[ti + k]
            cands = {"persistence": cur}
            for v in gv:
                p = preds[(v, loc, f, per)][:, k - 1]
                cands[f"gru_{v}"] = p
                for w in WEIGHTS:
                    cands[f"gru_{v}*{w}+persistence"] = w * p + (1 - w) * cur
            for name, p in cands.items():
                rows.append({"location": loc, "field": f, "period": per, "horizon_min": h, "candidate": name,
                             "mae": float(np.mean(np.abs(p - tgt))), "n": len(ti)})
    R = pd.DataFrame(rows)
    R.to_csv(out / "select_all.csv", index=False)
    val = R[R.period == "val"]
    choice = val.loc[val.groupby(["location", "field", "horizon_min"])["mae"].idxmin(), ["location", "field", "horizon_min", "candidate"]]
    choice.to_csv(out / "select_choice.csv", index=False)
    test = R[R.period == "test"].merge(choice, on=["location", "field", "horizon_min", "candidate"])
    base = R[(R.period == "test") & R.candidate.isin(["persistence"] + [f"gru_{v}" for v in gv])]
    sel = test.assign(candidate="selected")
    T = pd.concat([base, sel])

    def pooled(t):
        t = t.assign(w=t.mae * t.n)
        g = t.groupby(["field", "candidate", "horizon_min"]).agg(w=("w", "sum"), n=("n", "sum")).reset_index()
        g["mae"] = g.w / g.n
        return g
    P = pooled(T)

    def gain(field):
        a = P[(P.field == field) & (P.candidate == "selected") & (P.horizon_min == 60)].mae.iloc[0]
        b = P[(P.field == field) & (P.candidate == "persistence") & (P.horizon_min == 60)].mae.iloc[0]
        return (1 - a / b) * 100

    L = ["# 과제 3 보완. 검증 기간으로 고른 예측 (시험 기간 성적)\n"]
    L.append(f"- 후보: GRU 변형 {', '.join(gv)}, 현재값 유지, GRU·현재값 가중 평균(w={', '.join(map(str, WEIGHTS))}). "
             "공간·항목·시점마다 **검증 기간** MAE 가 가장 작은 것을 고르고, 시험 기간에는 한 번만 평가.")
    g_c, g_t = gain("co2"), gain("temperature")
    L.append(f"- 60분 CO₂: 현재값 유지 대비 {g_c:.1f}% 감소 → {'통과' if g_c >= 30 else '미달'} (30%)")
    L.append(f"- 60분 온도: 현재값 유지 대비 {g_t:.1f}% 감소 → {'통과' if g_t > 0 else '미달'}\n")
    for f, fmt in (("co2", ".1f"), ("temperature", ".3f")):
        L.append(f"## {f} 공간 합산 MAE\n")
        piv = P[P.field == f].pivot_table(index="candidate", columns="horizon_min", values="mae").reset_index()
        L.append("| model | " + " | ".join(f"{c}분" for c in piv.columns[1:]) + " |")
        L.append("|---|" + "---|" * (len(piv.columns) - 1))
        L += ["| " + r[0] + " | " + " | ".join(f"{x:{fmt}}" for x in r[1:]) + " |" for r in piv.itertuples(index=False)]
        L.append("")
    L.append("## 고른 결과 (60분)\n")
    c60 = choice[choice.horizon_min == 60].merge(test[["location", "field", "horizon_min", "mae"]], on=["location", "field", "horizon_min"])
    pers = R[(R.period == "test") & (R.candidate == "persistence") & (R.horizon_min == 60)][["location", "field", "mae"]].rename(columns={"mae": "persistence"})
    c60 = c60.merge(pers, on=["location", "field"])
    L.append("| location | field | 고른 예측 | 시험 MAE | 현재값 유지 | 개선 % |")
    L.append("|---|---|---|---|---|---|")
    for r in c60.itertuples():
        L.append(f"| {r.location} | {r.field} | {r.candidate} | {r.mae:.3f} | {r.persistence:.3f} | {(1 - r.mae / r.persistence) * 100:.1f} |")
    (out / "select_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
