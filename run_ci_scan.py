"""CI scan runner — breakout + CPR scans saved to data_cache with history.

Used by .github/workflows/scan.yml. Always saves results (even when zero
breakouts) so every scheduled run leaves a dated history entry.
"""

from __future__ import annotations

import argparse
import sys

from config import NARROW_PERCENTILE, UNIVERSE_NIFTY500, ensure_dirs
from cpr_scanner import scan_universe as scan_cpr_universe
from data_loader import load_universe_symbols, resolve_universe_symbols
from results_store import save_cpr_results, save_scan_results
from scanner import filter_results, scan_universe

TIMEFRAMES = ["1H", "1D", "1W"]
VOL_MULT = 1.25
LOOKBACK = 20


def _progress(done: int, total: int, label: str) -> None:
    if done % 50 == 0 or done == total:
        print(f"  [{done}/{total}] {label}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run breakout + CPR scans for CI")
    parser.add_argument("--max", type=int, default=500, help="Max NIFTY 500 symbols")
    parser.add_argument("--skip-cpr", action="store_true", help="Skip the CPR scan")
    args = parser.parse_args()

    ensure_dirs()
    nifty500 = load_universe_symbols()
    symbols, universe_sample, universe_total = resolve_universe_symbols(
        UNIVERSE_NIFTY500, nifty500, max_symbols=args.max
    )

    print(f"Breakout scan: {len(symbols)} symbols on {', '.join(TIMEFRAMES)} (standard)", flush=True)
    df = scan_universe(
        symbols,
        TIMEFRAMES,
        mode="standard",
        progress_callback=_progress,
        use_cache=True,
        vol_mult=VOL_MULT,
        lookback=LOOKBACK,
        atr_mult=None,
        direction_filter=None,
    )
    df = filter_results(df, min_vol_ratio=VOL_MULT, only_52w=False)
    save_scan_results(
        df,
        {
            "symbols": len(symbols),
            "universe_total": universe_total,
            "universe_sample": universe_sample,
            "universe_choice": UNIVERSE_NIFTY500,
            "timeframes": TIMEFRAMES,
            "mode": "Standard",
            "breakout_mode": "standard",
            "direction": "Both",
            "vol_mult": VOL_MULT,
            "lookback": LOOKBACK,
            "atr_mult": None,
            "only_52w": False,
            "max_symbols": args.max,
        },
    )
    print(f"Breakouts: {len(df)}", flush=True)

    if not args.skip_cpr:
        print(f"CPR scan: {len(symbols)} symbols (Daily)", flush=True)
        cpr = scan_cpr_universe(
            symbols,
            progress_callback=_progress,
            use_cache=True,
            narrow_percentile=NARROW_PERCENTILE,
            timeframe="Daily",
        )
        save_cpr_results(
            cpr,
            {
                "symbols": len(symbols),
                "universe_total": universe_total,
                "universe_sample": universe_sample,
                "universe_choice": UNIVERSE_NIFTY500,
                "timeframe": "Daily",
                "narrow_percentile": NARROW_PERCENTILE,
            },
        )
        print(f"CPR rows: {len(cpr)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
