"""Gap-up / gap-down scanner.

Compares each symbol's latest session open against the previous session's
close on daily bars. Pure observation of what happened at the open — no
prediction — consistent with the app's research-only stance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd

from data_loader import load_daily


def _gap_row(symbol: str, use_cache: bool) -> Optional[dict]:
    df = load_daily(symbol, days=60, use_cache=use_cache)
    if df is None or len(df) < 2:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    prev_close = float(prev["close"])
    if prev_close <= 0:
        return None
    open_ = float(last["open"])
    close = float(last["close"])
    gap_pct = (open_ - prev_close) / prev_close * 100
    since_open_pct = (close - open_) / open_ * 100 if open_ else 0.0
    vol = float(last.get("volume", 0) or 0)
    avg_vol = float(df["volume"].iloc[-21:-1].mean()) if "volume" in df.columns and len(df) >= 21 else 0.0
    return {
        "symbol": symbol.upper(),
        "session": df.index[-1].date().isoformat(),
        "direction": "gap-up" if gap_pct >= 0 else "gap-down",
        "prev_close": round(prev_close, 2),
        "open": round(open_, 2),
        "close": round(close, 2),
        "gap_pct": round(gap_pct, 2),
        "since_open_pct": round(since_open_pct, 2),
        "volume_ratio": round(vol / avg_vol, 2) if avg_vol > 0 else None,
    }


def scan_gaps(
    symbols: list[str],
    min_gap_pct: float = 2.0,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Scan the universe for opening gaps of at least `min_gap_pct` percent."""
    rows: list[dict] = []
    total = len(symbols)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_gap_row, s, use_cache): s for s in symbols}
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                row = fut.result()
            except Exception:
                row = None
            if row is not None and abs(row["gap_pct"]) >= min_gap_pct:
                rows.append(row)
            if progress_callback:
                progress_callback(done, total, futures[fut])
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session",
                "direction",
                "prev_close",
                "open",
                "close",
                "gap_pct",
                "since_open_pct",
                "volume_ratio",
            ]
        )
    df = pd.DataFrame(rows)
    return df.reindex(df["gap_pct"].abs().sort_values(ascending=False).index).reset_index(drop=True)
