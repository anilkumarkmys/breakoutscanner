"""Central Pivot Range (CPR) calculations and Virgin CPR detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config import (
    NARROW_CPR_PCT,
    NARROW_PERCENTILE,
    WIDE_CPR_PCT,
    WIDE_PERCENTILE,
    WIDTH_HISTORY_DAYS,
)

Position = Literal["above", "below", "inside"]
WidthType = Literal["narrow", "normal", "wide"]
VirginType = Literal["V+W", "V+N", "V", "WIDE", "NARROW", "TOUCHED", "—"]


@dataclass(frozen=True)
class CPRLevels:
    """CPR levels computed from one session's OHLC."""

    pivot: float
    tc: float
    bc: float
    r1: float
    s1: float
    width: float
    width_pct: float
    source_date: date

    @property
    def cpr_high(self) -> float:
        return self.tc

    @property
    def cpr_low(self) -> float:
        return self.bc


@dataclass
class VirginCPRResult:
    """Scanner output for one symbol."""

    symbol: str
    status: str
    cpr_type: VirginType
    distance_pct: float
    trend: Position
    ltp: float
    virgin_level: float
    pivot: float
    tc: float
    bc: float
    width_pct: float
    width_percentile: float
    narrow_threshold_pct: float
    days_virgin: int
    source_date: date
    session_date: date
    is_virgin: bool
    width_class: WidthType


def compute_cpr(high: float, low: float, close: float, source_date: date) -> CPRLevels:
    """Standard CPR from previous session OHLC."""
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = 2.0 * pivot - bc
    if tc < bc:
        tc, bc = bc, tc

    width = tc - bc
    width_pct = (width / pivot * 100.0) if pivot else 0.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high

    return CPRLevels(
        pivot=pivot,
        tc=tc,
        bc=bc,
        r1=r1,
        s1=s1,
        width=width,
        width_pct=width_pct,
        source_date=source_date,
    )


def _scalar(row: pd.Series, col: str) -> float:
    val = row[col]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val)


def build_cpr_width_history(daily: pd.DataFrame, max_sessions: int = WIDTH_HISTORY_DAYS) -> pd.Series:
    """CPR width % series over the last year (one value per session, from prior bar OHLC)."""
    if daily is None or len(daily) < 3:
        return pd.Series(dtype=float)

    df = daily.sort_index().copy()
    df.columns = [c.lower() for c in df.columns]
    if max_sessions > 0:
        df = df.tail(max_sessions + 1)

    widths: list[float] = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        src = df.index[i - 1]
        if hasattr(src, "date"):
            src = src.date()
        lv = compute_cpr(_scalar(prev, "high"), _scalar(prev, "low"), _scalar(prev, "close"), src)
        widths.append(lv.width_pct)

    return pd.Series(widths, dtype=float)


def classify_width_fixed(width_pct: float) -> WidthType:
    if width_pct <= NARROW_CPR_PCT:
        return "narrow"
    if width_pct >= WIDE_CPR_PCT:
        return "wide"
    return "normal"


def classify_width_relative(
    width_pct: float,
    history: pd.Series,
    narrow_percentile: float = NARROW_PERCENTILE,
    wide_percentile: float = WIDE_PERCENTILE,
) -> tuple[WidthType, float, float]:
    """Classify CPR width vs instrument's own 1-year history."""
    clean = history.dropna()
    if len(clean) < 30:
        return classify_width_fixed(width_pct), 50.0, NARROW_CPR_PCT

    narrow_thr = float(np.percentile(clean, narrow_percentile))
    wide_thr = float(np.percentile(clean, wide_percentile))
    rank = float((clean < width_pct).mean() * 100.0)

    if width_pct <= narrow_thr:
        return "narrow", rank, narrow_thr
    if width_pct >= wide_thr:
        return "wide", rank, narrow_thr
    return "normal", rank, narrow_thr


def cpr_zone_touched(day_high: float, day_low: float, tc: float, bc: float) -> bool:
    """True if the session range overlaps the CPR zone [BC, TC]."""
    return day_high >= bc and day_low <= tc


def _signed_distance_pct(price: float, tc: float, bc: float) -> tuple[float, float, Position]:
    """Distance % to CPR zone and trend position."""
    if price > tc:
        level = tc
        dist = (price - level) / level * 100.0
        return dist, level, "above"
    if price < bc:
        level = bc
        dist = (price - level) / level * 100.0
        return dist, level, "below"

    mid = (tc + bc) / 2.0
    dist = (price - mid) / mid * 100.0
    return dist, mid, "inside"


def _build_type(is_virgin: bool, width_class: WidthType) -> VirginType:
    if is_virgin:
        if width_class == "wide":
            return "V+W"
        if width_class == "narrow":
            return "V+N"
        return "V"
    if width_class == "wide":
        return "WIDE"
    if width_class == "narrow":
        return "NARROW"
    return "TOUCHED"


def _days_virgin_since_formed(df: pd.DataFrame, cpr_index: int, tc: float, bc: float) -> int:
    """Sessions since CPR formed that have not touched the zone (including today)."""
    days = 0
    for j in range(cpr_index, len(df)):
        bar = df.iloc[j]
        if cpr_zone_touched(_scalar(bar, "high"), _scalar(bar, "low"), tc, bc):
            break
        days += 1
    return days


def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLC to weekly bars ending Friday."""
    df = daily.copy()
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    resampled = df.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df.columns else "first",
        }
    ).dropna()
    return resampled


def scan_today_cpr(
    daily: pd.DataFrame,
    narrow_percentile: float = NARROW_PERCENTILE,
    wide_percentile: float = WIDE_PERCENTILE,
    timeframe: str = "Daily",
) -> Optional[VirginCPRResult]:
    """
    Scan today's CPR — levels from yesterday's OHLC applied to the latest session.

    Virgin: today's range has not touched yesterday's CPR zone [BC, TC].
    """
    if daily is None or len(daily) < 2:
        return None

    df_to_use = resample_to_weekly(daily) if timeframe == "Weekly" else daily
    if len(df_to_use) < 2:
        return None

    df = df_to_use.sort_index().copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    prev = df.iloc[-2]
    today = df.iloc[-1]
    src_date = df.index[-2]
    session_date = df.index[-1]
    if hasattr(src_date, "date"):
        src_date = src_date.date()
    if hasattr(session_date, "date"):
        session_date = session_date.date()

    levels = compute_cpr(
        _scalar(prev, "high"),
        _scalar(prev, "low"),
        _scalar(prev, "close"),
        src_date,
    )

    width_history = build_cpr_width_history(df)
    width_class, width_percentile, narrow_thr = classify_width_relative(
        levels.width_pct,
        width_history,
        narrow_percentile=narrow_percentile,
        wide_percentile=wide_percentile,
    )

    ltp = _scalar(today, "close")
    touched_today = cpr_zone_touched(
        _scalar(today, "high"), _scalar(today, "low"), levels.tc, levels.bc
    )
    is_virgin = not touched_today
    days_virgin = _days_virgin_since_formed(df, len(df) - 1, levels.tc, levels.bc) if is_virgin else 0

    dist, virgin_level, trend = _signed_distance_pct(ltp, levels.tc, levels.bc)

    return VirginCPRResult(
        symbol="",
        status=f"{virgin_level:.2f}",
        cpr_type=_build_type(is_virgin, width_class),
        distance_pct=round(dist, 2),
        trend=trend,
        ltp=round(ltp, 2),
        virgin_level=round(virgin_level, 2),
        pivot=round(levels.pivot, 2),
        tc=round(levels.tc, 2),
        bc=round(levels.bc, 2),
        width_pct=round(levels.width_pct, 3),
        width_percentile=round(width_percentile, 1),
        narrow_threshold_pct=round(narrow_thr, 3),
        days_virgin=days_virgin,
        source_date=levels.source_date,
        session_date=session_date,
        is_virgin=is_virgin,
        width_class=width_class,
    )


def levels_for_chart(daily: pd.DataFrame, timeframe: str = "Daily") -> dict[str, float]:
    """Today's CPR levels for chart overlay (from previous session)."""
    if daily is None or len(daily) < 2:
        return {}
    df_to_use = resample_to_weekly(daily) if timeframe == "Weekly" else daily
    if len(df_to_use) < 2:
        return {}

    df = df_to_use.sort_index()
    df.columns = [c.lower() for c in df.columns]
    prev = df.iloc[-2]
    src = df.index[-2]
    if hasattr(src, "date"):
        src = src.date()
    lv = compute_cpr(_scalar(prev, "high"), _scalar(prev, "low"), _scalar(prev, "close"), src)
    return {
        "CPR HIGH (TC)": lv.tc,
        "CPR TC": lv.tc,
        "CPR PIVOT": lv.pivot,
        "CPR BC": lv.bc,
        "CPR LOW (BC)": lv.bc,
        "R1": lv.r1,
        "S1": lv.s1,
    }
