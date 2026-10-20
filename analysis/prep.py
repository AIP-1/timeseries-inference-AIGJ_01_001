"""정제와 5분 격자 만들기. 학습(과제 2~7)과 상시 서비스(과제 8)가 이 모듈 하나를 같이 쓴다.

정제 규칙과 그 이유는 RULES 와 docs/task2.md 에 있다. 규칙을 바꾸면 학습과 서비스가 함께 바뀐다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
FIELDS = ("co2", "temperature", "humidity")

# 물리적으로 가능한 범위. 밖이면 측정값이 아니라 고장·통신 오류로 보고 버린다.
#   co2: 외기가 약 420ppm 이라 실내가 그보다 크게 낮을 수 없다. 0ppm 은 센서 초기화·통신 깨짐.
#        5000ppm 은 사람이 쓰는 방에서 사실상 불가능(직업 노출 한도 수준).
#   temperature: 냉난방 되는 실내. 60℃ 같은 값은 센서 오류.
#   humidity: 0%, 100% 고정은 센서 고장이 흔한 모양이라 양 끝을 뺀다.
RULES = {
    "co2": (350.0, 5000.0),
    "temperature": (0.0, 45.0),
    "humidity": (1.0, 99.0),
}


@dataclass
class GridSpec:
    freq: str = "5min"        # 분석 간격
    min_minutes: int = 3      # 5분 칸 안에 1분 값이 이 개수 이상 있어야 칸을 채운다 (막 시작된 칸 배제)
    max_fill_bins: int = 3    # 짧은 결측(15분 이하)만 선형 보간, 그보다 긴 결측은 비워 둔다


def load_history(root: str | Path, field: str) -> pd.DataFrame:
    files = sorted(Path(root, field).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root}/{field}/*.parquet 없음. 먼저 python -m analysis.export")
    # 1년치 온습도는 수백만 행이라 문자열 열을 category 로 바꿔 메모리를 줄인다(보드 RAM 8GB 공유)
    df = pd.concat([pd.read_parquet(f, columns=["time", "device_id", "location", "value"]) for f in files],
                   ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["value"] = df["value"].astype("float32")
    for c in ("device_id", "location"):
        df[c] = df[c].astype("category")
    df["field"] = pd.Categorical([field] * len(df), categories=list(FIELDS))
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """범위 밖 값을 버린다. (남은 값, 공간별 제거 건수) 반환."""
    f = df["field"].astype(str)
    lo = f.map({k: v[0] for k, v in RULES.items()})
    hi = f.map({k: v[1] for k, v in RULES.items()})
    bad = df["value"].isna() | (df["value"] < lo) | (df["value"] > hi)
    removed = df[bad].groupby(["location", "field"], observed=True).size().rename("removed").reset_index()
    return df[~bad], removed


def fill_short_gaps(s: pd.Series, max_len: int) -> pd.Series:
    """길이가 max_len 칸 이하인 결측만 선형 보간한다.

    pandas interpolate(limit=n) 은 긴 결측의 앞 n 칸도 채워 버려서 쓰지 않는다.
    """
    na = s.isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("size")
    inside = s.ffill().notna() & s.bfill().notna()
    fillable = na & (run_len <= max_len) & inside
    filled = s.interpolate(limit_area="inside")
    return s.where(~fillable, filled)


def to_grid(df: pd.DataFrame, spec: GridSpec = GridSpec()) -> pd.DataFrame:
    """장비별 1분 평균 → 공간별 장비 평균 → 5분 평균(최소 개수 조건) → 짧은 결측 보간.

    반환: index=UTC 5분 시각(칸 시작), columns=MultiIndex(location, field)
    """
    d = df.copy()
    d["minute"] = d["time"].dt.floor("1min")
    # 센서마다 기록 주기가 달라도 1분 단위로 맞춘다. 같은 분에 여러 번 오면 평균.
    per_dev = d.groupby(["location", "field", "device_id", "minute"], observed=True)["value"].mean()
    # 한 공간에 같은 항목 센서가 여럿이면 평균 (위치 차이는 공간 대표값으로 흡수)
    per_space = per_dev.groupby(level=["location", "field", "minute"], observed=True).mean().unstack(["location", "field"])
    per_space = per_space.sort_index()
    full = pd.date_range(per_space.index.min().floor(spec.freq), per_space.index.max(), freq="1min", tz="UTC")
    per_space = per_space.reindex(full)
    agg = per_space.resample(spec.freq).agg(["mean", "count"])
    mean = agg.xs("mean", axis=1, level=-1)
    count = agg.xs("count", axis=1, level=-1)
    grid = mean.where(count >= spec.min_minutes)
    if spec.max_fill_bins > 0:
        grid = grid.apply(fill_short_gaps, max_len=spec.max_fill_bins)
    grid.columns = pd.MultiIndex.from_tuples(grid.columns, names=["location", "field"])
    return grid


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(TZ)
    return pd.DataFrame({
        "hour": local.hour + local.minute / 60,
        "dow": local.dayofweek,
        "weekend": (local.dayofweek >= 5).astype(int),
    }, index=index)


@dataclass
class Split:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    end: pd.Timestamp

    def label(self, t: pd.DatetimeIndex) -> np.ndarray:
        return np.where(t < self.train_end, "train", np.where(t < self.val_end, "val", "test"))


def make_split(index: pd.DatetimeIndex, test_days: int = 28, val_days: int = 28) -> Split:
    """시간순 분할: 마지막 test_days 는 시험, 그 앞 val_days 는 검증, 나머지는 학습.

    무작위로 섞으면 5분 간격 이웃 값(거의 같은 값)이 학습과 시험에 나뉘어 들어가 성능이 부풀려진다.
    경계는 KST 자정에 맞춰 하루가 쪼개지지 않게 한다.
    """
    end = index.max()
    local_end = end.tz_convert(TZ).normalize()
    test_start = (local_end - pd.Timedelta(days=test_days - 1)).tz_convert("UTC")
    val_start = test_start - pd.Timedelta(days=val_days)
    return Split(train_end=val_start, val_end=test_start, end=end)


def normalize_stats(grid: pd.DataFrame, split: Split) -> pd.DataFrame:
    """정규화용 평균·표준편차. 반드시 학습 기간 데이터로만 계산한다."""
    train = grid[grid.index < split.train_end]
    return pd.DataFrame({"mean": train.mean(), "std": train.std()})
