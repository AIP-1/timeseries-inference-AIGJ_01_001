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

# 공휴일·대체공휴일(미리 알 수 있는 정보). 과제 2 '오르지 않은 평일'과 대조해 확인함.
# 새 해가 오면 여기에 더한다.
KR_HOLIDAYS = pd.to_datetime([
    "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-02", "2026-05-01", "2026-05-05",
    "2026-05-25", "2026-06-03", "2026-07-17", "2026-08-17", "2026-09-24", "2026-09-25", "2026-10-05",
    "2026-10-09", "2026-12-25",
]).date


def offday(index: pd.DatetimeIndex) -> np.ndarray:
    """주말 또는 공휴일이면 1 (KST 날짜 기준)."""
    local = index.tz_convert(TZ)
    return ((local.dayofweek >= 5) | np.isin(local.date, KR_HOLIDAYS)).astype(np.float32)

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
    min_minutes: int = 3      # 1분 장비는 5분 칸에 값이 3개 이상이어야 칸을 채운다 (막 시작된 칸 배제). 느린 장비는 to_grid 참고
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


def report_intervals(df: pd.DataFrame) -> pd.Series:
    """장비별 기록 주기(분, 중앙값). 진단 표용. to_grid 도 장비마다 같은 방식으로 주기를 잰다."""
    m = df[["location", "field", "device_id"]].assign(minute=df["time"].dt.floor("1min")).drop_duplicates()
    m = m.sort_values("minute")
    gap = m.groupby(["location", "field", "device_id"], observed=True)["minute"].diff().dt.total_seconds() / 60
    iv = gap.groupby([m["location"], m["field"], m["device_id"]], observed=True).median()
    return iv.fillna(1.0).clip(lower=1.0)


def to_grid(df: pd.DataFrame, spec: GridSpec = GridSpec()) -> pd.DataFrame:
    """장비별 5분 평균(장비 주기에 맞춘 최소 개수) → 장비 주기만큼 앞 값 유지 → 공간별 장비 평균 → 짧은 결측 보간.

    장비마다 1·2·10분 등 주기가 달라서, 공간 평균을 1분 단위로 먼저 내면 10분 장비가 칸마다 들어왔다 빠지며
    평균이 톱니처럼 흔들린다(오프셋이 큰 장비일수록 심함). 그래서 장비별로 5분 칸을 먼저 만든다.
    반환: index=UTC 5분 시각(칸 시작), columns=MultiIndex(location, field)
    """
    d = df.copy()
    d["minute"] = d["time"].dt.floor("1min")
    # 같은 분에 여러 번 오면 평균
    per_min = d.groupby(["location", "field", "device_id", "minute"], observed=True)["value"].mean()
    mins = per_min.index.get_level_values("minute")
    bins = pd.date_range(mins.min().floor(spec.freq), mins.max(), freq=spec.freq)
    width = pd.Timedelta(spec.freq)
    per_space: dict[tuple, list[pd.Series]] = {}
    for (loc, field, _dev), s in per_min.groupby(level=["location", "field", "device_id"], observed=True):
        s = s.droplevel(["location", "field", "device_id"])
        step = s.index.to_series().diff().median()
        step = pd.Timedelta("1min") if pd.isna(step) else max(step, pd.Timedelta("1min"))
        # 칸 하나에 기대되는 개수의 60% 이상(최대 min_minutes, 최소 1). 1분 장비=3, 2분·10분 장비=1
        need = int(np.clip(width / step * 0.6, 1, spec.min_minutes))
        r = s.resample(spec.freq).agg(["mean", "count"])
        v = r["mean"].where(r["count"] >= need).reindex(bins)
        # 10분 장비는 다음 칸 하나까지 값 유지. 며칠에 한 번 오는 시험용 장비가 며칠을 채우지 않게 최대 2칸
        hold = min(int(np.ceil(step / width)) - 1, 2)
        if hold > 0:
            v = v.ffill(limit=hold)
        per_space.setdefault((loc, field), []).append(v)
    grid = pd.DataFrame({k: pd.concat(v, axis=1).mean(axis=1) for k, v in per_space.items()}, index=bins)
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
