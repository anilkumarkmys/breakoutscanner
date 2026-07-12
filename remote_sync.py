"""Pull newly committed scan artifacts from GitHub raw content.

The scheduled GitHub Action commits scan snapshots and history to main.
During market hours the app polls scan_info.csv (tiny) and, when the
remote scanned_at is newer than the local one, downloads the refreshed
files so an open browser session sees new scans without a redeploy.
"""

from __future__ import annotations

import io
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import DATA_DIR, HISTORY_DIR, SCAN_INFO_CSV

RAW_BASE = "https://raw.githubusercontent.com/anilkumarkmys/breakoutscanner/main/data_cache/"
_STATIC_FILES = (
    "scan_results.csv",
    "scan_info.csv",
    "scan_meta.json",
    "cpr_scan_results.csv",
    "cpr_scan_info.csv",
    "cpr_scan_meta.json",
)
_IST = ZoneInfo("Asia/Kolkata")


def _remote_text(name: str, timeout: float = 6.0) -> str | None:
    try:
        resp = requests.get(RAW_BASE + name, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException:
        pass
    return None


def _scanned_at_from_info(text: str) -> datetime | None:
    try:
        df = pd.read_csv(io.StringIO(text))
        return datetime.fromisoformat(str(df.iloc[0]["scanned_at"]))
    except Exception:
        return None


def local_scanned_at() -> datetime | None:
    if not SCAN_INFO_CSV.is_file():
        return None
    try:
        return _scanned_at_from_info(SCAN_INFO_CSV.read_text(encoding="utf-8"))
    except Exception:
        return None


def sync_from_github() -> bool:
    """Download newer committed scan results; True when local files changed.
    Never overwrites a local scan that is newer than the remote one."""
    try:
        remote_info = _remote_text("scan_info.csv")
        if remote_info is None:
            return False
        remote_ts = _scanned_at_from_info(remote_info)
        local_ts = local_scanned_at()
        if remote_ts is None or (local_ts is not None and remote_ts <= local_ts):
            return False

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "scan_info.csv").write_text(remote_info, encoding="utf-8", newline="\n")
        for name in _STATIC_FILES:
            if name == "scan_info.csv":
                continue
            text = _remote_text(name)
            if text is not None:
                (DATA_DIR / name).write_text(text, encoding="utf-8", newline="\n")
        today = datetime.now(_IST).date().isoformat()
        for kind in ("breakout", "cpr"):
            text = _remote_text(f"history/{kind}_{today}.csv")
            if text is not None:
                (HISTORY_DIR / f"{kind}_{today}.csv").write_text(text, encoding="utf-8", newline="\n")
        return True
    except Exception:
        return False
