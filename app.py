"""NIFTY 500 multi-timeframe breakout scanner — Streamlit UI."""

from __future__ import annotations

import base64
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    NARROW_PERCENTILE,
    NARROW_PERCENTILE_MAX,
    NARROW_PERCENTILE_MIN,
    NARROW_PERCENTILE_PRESETS,
    STRICT_ATR_MULT,
    STRICT_VOL_MULT,
    TIMEFRAMES,
    TIMEFRAME_ORDER,
    UNIVERSE_CHOICES,
    UNIVERSE_FNO,
    UNIVERSE_NIFTY500,
    ensure_dirs,
    sort_timeframes,
)
from cpr import levels_for_chart
from cpr_scanner import apply_narrow_percentile, filter_results as filter_cpr_results, scan_universe as scan_cpr_universe
from data_loader import load_bars, load_daily, load_universe_symbols, resolve_universe_symbols
from fno_loader import fno_symbol_set, load_fno_symbols
from results_store import (
    cached_cpr_scan_available,
    cached_scan_available,
    format_scanned_at,
    list_history_dates,
    load_cpr_results,
    load_history,
    load_scan_results,
    save_cpr_results,
    save_scan_results,
)
from options_plan import OptionPlan, build_plan, fetch_option_chain
from scanner import filter_results, scan_universe

ROOT_DIR = Path(__file__).resolve().parent
LOGO_PATH = ROOT_DIR / "assets" / "tag_trade_and_grow.png"
PUBLISHED_URL = "https://tag-breakoutscanner.streamlit.app/"
DISCLAIMER_URL = "https://github.com/anilkumarkmys/breakoutscanner/blob/main/DISCLAIMER.md"


def _logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


st.set_page_config(
    page_title="TAG Breakout Scanner",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "TAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 16% 0%, rgba(240,184,78,.18), transparent 34%),
        linear-gradient(180deg, #050505 0%, #0b0b0b 44%, #050505 100%);
    color: #f8f5ee;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090909 0%, #120f08 100%);
    border-right: 1px solid rgba(240,184,78,.34);
}
[data-testid="stSidebar"] * { color: #f8f5ee; }
h1, h2, h3 { color: #ffd36b; letter-spacing: 0; }
a { color: #ffd36b !important; }
.tag-hero {
    display: grid;
    grid-template-columns: 92px minmax(0, 1fr);
    gap: 18px;
    align-items: center;
    padding: 18px 20px;
    margin: 0 0 18px;
    border: 1px solid rgba(240,184,78,.38);
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(240,184,78,.18), rgba(10,10,10,.94));
    box-shadow: 0 18px 42px rgba(0,0,0,.30);
}
.tag-hero img {
    width: 92px;
    height: 92px;
    border-radius: 10px;
    object-fit: cover;
    border: 1px solid rgba(240,184,78,.60);
    box-shadow: 0 0 28px rgba(240,184,78,.20);
}
.tag-hero h1 {
    margin: 0;
    color: #ffd36b;
    font-size: 32px;
}
.tag-hero p {
    margin: 5px 0 0;
    color: #c5b38a;
    line-height: 1.4;
}
.tag-build {
    display: inline-block;
    margin-top: 7px;
    padding: 4px 8px;
    border: 1px solid rgba(240,184,78,.34);
    border-radius: 999px;
    background: rgba(240,184,78,.10);
    color: #f8f5ee;
    font-size: 12px;
    font-weight: 700;
}
.stButton > button, button[kind="primary"] {
    background: linear-gradient(135deg, #f0b84e, #7a4c0d);
    color: #080808;
    border: 1px solid rgba(255,211,107,.60);
    font-weight: 800;
}
[data-testid="stMetric"], div[data-testid="stDataFrame"], .stAlert {
    border: 1px solid rgba(240,184,78,.28);
    border-radius: 8px;
    background: rgba(16,16,16,.88);
}
.breakout-card {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(240,184,78,.30);
    box-shadow: 0 10px 26px rgba(0,0,0,.45), inset 0 1px 0 rgba(240,184,78,.12);
    min-height: 130px;
    background: linear-gradient(150deg, #17110a 0%, #0c0a07 55%, #060606 100%);
}
.breakout-card.bullish {
    border-left: 5px solid #5dbb7f;
    background: linear-gradient(150deg, #0f1a10 0%, #0b0f09 45%, #060606 100%);
}
.breakout-card.bearish {
    border-left: 5px solid #e06552;
    background: linear-gradient(150deg, #1c0f0b 0%, #110b08 45%, #060606 100%);
}
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
    margin-bottom: 0.55rem;
    flex-wrap: wrap;
}
.card-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    justify-content: flex-end;
}
.card-symbol {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffd36b;
    letter-spacing: 0.02em;
}
.card-badges span {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: rgba(240,184,78,.12);
    border: 1px solid rgba(240,184,78,.32);
    color: #ffd36b;
}
.card-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: rgba(240,184,78,.12);
    border: 1px solid rgba(240,184,78,.32);
    color: #ffd36b;
    margin-left: 0.25rem;
}
.card-badges .high52,
.card-pill.high52 {
    background: linear-gradient(135deg, rgba(240,184,78,.40), rgba(122,76,13,.40));
    border-color: rgba(255,211,107,.65);
    color: #ffe9b3;
}
.card-stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.75rem;
    margin-bottom: 0.45rem;
}
.card-stat {
    font-size: 0.82rem;
    color: #d9cfb6;
}
.card-stat b {
    color: #fff8e6;
    font-weight: 700;
}
.card-foot {
    font-size: 0.74rem;
    color: #b3a175;
    opacity: 0.92;
}
.summary-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.summary-metric {
    flex: 1 1 120px;
    min-width: 110px;
    padding: 0.5rem 0.65rem;
    border-radius: 10px;
    border: 1px solid rgba(240,184,78,.28);
    border-left: 3px solid #f0b84e;
    background: linear-gradient(150deg, #191307 0%, #0c0a07 60%, #060606 100%);
    box-shadow: 0 8px 20px rgba(0,0,0,.35), inset 0 1px 0 rgba(240,184,78,.10);
}
.summary-metric .sm-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.15rem;
    opacity: 0.9;
    color: #d9b878;
}
.summary-metric .sm-value {
    font-size: 0.95rem;
    font-weight: 700;
    line-height: 1.25;
    word-break: break-word;
    color: #f8f5ee;
}
.summary-metric.breakouts .sm-value { color: #ffd36b; }
.summary-metric.bullish { border-left-color: #5dbb7f; }
.summary-metric.bullish .sm-value { color: #9fe0b8; }
.summary-metric.bearish { border-left-color: #e06552; }
.summary-metric.bearish .sm-value { color: #f2a08e; }
.summary-metric.scanned .sm-value { font-size: 0.82rem; font-weight: 600; }
.cpr-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.cpr-legend-item {
    font-size: 0.78rem;
    padding: 0.35rem 0.65rem;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid rgba(240,184,78,.25);
    background: rgba(14,12,8,.92);
}
.cpr-legend-item.vw { color: #cbb8f0; border-color: rgba(167,139,250,.45); }
.cpr-legend-item.vn { color: #a9c6f2; border-color: rgba(96,165,250,.45); }
.cpr-legend-item.wide { color: #f2a08e; border-color: rgba(224,101,82,.45); }
.cpr-legend-item.virgin { color: #9fe0b8; border-color: rgba(93,187,127,.45); }
.cpr-legend-item.touched { color: #d9cfb6; border-color: rgba(240,184,78,.30); }
.cpr-meta-panel {
    background: rgba(240,184,78,.07);
    border: 1px solid rgba(240,184,78,.30);
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    margin: 0.35rem 0 0.85rem;
    font-size: 0.78rem;
    line-height: 1.5;
    color: #cbbfa2;
}
.cpr-meta-panel strong { color: #ffd36b; font-weight: 600; }
.cpr-meta-panel code {
    font-size: 0.74rem;
    color: #ffd36b;
    background: rgba(0,0,0,.55);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
}
.summary-metric.cpr-total .sm-value { color: #ffd36b; }
.summary-metric.cpr-virgin { border-left-color: #5dbb7f; }
.summary-metric.cpr-virgin .sm-value { color: #9fe0b8; }
.summary-metric.cpr-vw { border-left-color: #a78bfa; }
.summary-metric.cpr-vw .sm-value { color: #cbb8f0; }
.summary-metric.cpr-vn { border-left-color: #60a5fa; }
.summary-metric.cpr-vn .sm-value { color: #a9c6f2; }
.summary-metric.cpr-narrow .sm-value { color: #ffe9b3; }
.summary-metric.cpr-cutoff .sm-value { font-size: 0.88rem; }
.cpr-detail-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 0.85rem;
    margin: 0.35rem 0 0.75rem;
    padding: 0.55rem 0.75rem;
    border-radius: 10px;
    border: 1px solid rgba(240,184,78,.24);
    background: rgba(14,12,8,.85);
}
.cpr-detail-stat {
    font-size: 0.8rem;
    color: #d9cfb6;
}
.cpr-detail-stat b {
    color: #fff8e6;
    font-weight: 600;
}
.tag-disclaimer {
    background: linear-gradient(135deg, rgba(240,184,78,.14), rgba(10,10,10,.94));
    border: 1px solid rgba(240,184,78,.45);
    border-left: 5px solid #f0b84e;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 10px 26px rgba(0,0,0,.35);
}
.tag-disclaimer p {
    color: #f3e3bb;
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.55;
}
.tag-disclaimer strong { color: #ffd36b; }
.tag-disclaimer a { color: #ffd36b !important; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

_DIR_STYLE = {
    "bullish": ("🟢 Bullish", "#16a34a"),
    "bearish": ("🔴 Bearish", "#dc2626"),
}

_CPR_TYPE_COLORS = {
    "V+W": "#7c3aed",
    "V+N": "#2563eb",
    "V": "#16a34a",
    "WIDE": "#dc2626",
    "NARROW": "#f59e0b",
    "TOUCHED": "#6b7280",
}

_CPR_TREND_ICONS = {"above": "🟢 ↑", "below": "🔴 ↓", "inside": "🟡 •"}

_INDEX_SYMBOLS = frozenset({"VIX", "INDIAVIX", "NIFTY", "BANKNIFTY", "SENSEX"})

def _render_disclaimer_banner() -> None:
    """Single consolidated disclaimer: short banner + full text in an expander."""
    st.markdown(
        """
<div class="tag-disclaimer">
<p>
<strong>⚠️ Disclaimer:</strong> This app is for <strong>research and education only</strong> — it is
<strong>not investment advice</strong> and not a recommendation to buy or sell any security.
TAG (Trade And Grow) is <strong>not SEBI-registered</strong> (Investment Adviser or Research Analyst),
and using this app creates no adviser–client relationship. Signals are
<strong>automated algorithmic outputs</strong> — no human review — built from third-party data that may
be delayed, incomplete, or wrong; past breakouts do not guarantee future performance. Trading involves
<strong>substantial risk of loss</strong>: use only risk capital and consult a qualified,
SEBI-registered adviser before acting.
</p></div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("⚖️ Full legal disclaimer (tap to read in-app)"):
        disclaimer_path = ROOT_DIR / "DISCLAIMER.md"
        if disclaimer_path.exists():
            st.markdown(disclaimer_path.read_text(encoding="utf-8"))
        else:
            st.markdown(f"[Read the full disclaimer on GitHub]({DISCLAIMER_URL})")


_ROADMAP_SHIPPED = [
    "Breakout scanner — 1H / 1D / 1W",
    "Virgin & Narrow CPR scanner",
    "📅 Daily Review with scan history",
    "Scheduled scans via GitHub Actions",
    "📊 Signal audit vs real movement",
    "F&O options plan — CE/PE, strike, expiry, entry/TP/SL",
    "NIFTY 250 universe",
]

_ROADMAP_UPCOMING = [
    "Gap-up / gap-down scanner",
    "Opening-range breakout (ORB) scanner",
    "Volume-surge & delivery scanner",
    "Relative strength vs NIFTY (sector rotation)",
    "EMA crossover / RSI momentum scanner",
    "Per-stock detail view & signal history",
    "Watchlist / starred picks",
]


def _render_sidebar_roadmap() -> None:
    st.divider()
    st.markdown("#### 🚀 Scanner roadmap")
    st.markdown("\n".join(f"- ✅ {item}" for item in _ROADMAP_SHIPPED))
    st.markdown("**Upcoming**")
    st.markdown("\n".join(f"- 🔜 {item}" for item in _ROADMAP_UPCOMING))
    st.caption(
        "Suggest a scanner — [open an issue](https://github.com/anilkumarkmys/breakoutscanner/issues)."
    )


def _render_summary_metrics(
    *,
    breakouts: int,
    bullish: int,
    bearish: int,
    symbols_scanned: int | str,
    scanned_label: str,
) -> None:
    html = f"""
<div class="summary-metrics">
  <div class="summary-metric breakouts">
    <div class="sm-label">Breakouts</div>
    <div class="sm-value">{breakouts}</div>
  </div>
  <div class="summary-metric bullish">
    <div class="sm-label">Bullish</div>
    <div class="sm-value">{bullish}</div>
  </div>
  <div class="summary-metric bearish">
    <div class="sm-label">Bearish</div>
    <div class="sm-value">{bearish}</div>
  </div>
  <div class="summary-metric symbols">
    <div class="sm-label">Symbols Scanned</div>
    <div class="sm-value">{symbols_scanned}</div>
  </div>
  <div class="summary-metric scanned">
    <div class="sm-label">Last Scanned</div>
    <div class="sm-value">{scanned_label}</div>
  </div>
</div>
"""
    _render_card_html(html)


def _render_cpr_summary_metrics(
    *,
    total: int,
    virgin: int,
    v_w: int,
    v_n: int,
    narrow: int,
    narrow_pct: float,
    scanned_label: str,
) -> None:
    html = f"""
<div class="summary-metrics">
  <div class="summary-metric cpr-total">
    <div class="sm-label">Total</div>
    <div class="sm-value">{total}</div>
  </div>
  <div class="summary-metric cpr-virgin">
    <div class="sm-label">Virgin</div>
    <div class="sm-value">{virgin}</div>
  </div>
  <div class="summary-metric cpr-vw">
    <div class="sm-label">V + W</div>
    <div class="sm-value">{v_w}</div>
  </div>
  <div class="summary-metric cpr-vn">
    <div class="sm-label">V + N</div>
    <div class="sm-value">{v_n}</div>
  </div>
  <div class="summary-metric cpr-narrow">
    <div class="sm-label">Narrow</div>
    <div class="sm-value">{narrow}</div>
  </div>
  <div class="summary-metric cpr-cutoff">
    <div class="sm-label">Narrow cutoff</div>
    <div class="sm-value">≤{narrow_pct:g}%</div>
  </div>
  <div class="summary-metric scanned">
    <div class="sm-label">Last scan</div>
    <div class="sm-value">{scanned_label or "—"}</div>
  </div>
</div>
"""
    _render_card_html(html)


def _render_cpr_session_panel(
    *,
    session_date: object,
    scan_tf: str,
    universe_choice: str,
    narrow_pct: float,
) -> None:
    html = f"""
<div class="cpr-meta-panel">
  <strong>Session {session_date}</strong> · {scan_tf} CPR · {universe_choice} ·
  narrow = bottom {narrow_pct:g}% of 1Y width history ·
  cached in <code>data_cache/cpr_scan_results.csv</code>
</div>
"""
    _render_card_html(html)


def _render_last_scan_panel(meta: dict, results: pd.DataFrame | None = None) -> None:
    scanned_at = meta.get("scanned_at_display") or format_scanned_at(meta.get("scanned_at") or meta.get("saved_at"))
    if not scanned_at or scanned_at == "—":
        return

    n_breakouts = int(meta.get("breakout_count", len(results) if results is not None else 0))
    n_sym = meta.get("symbols_scanned", meta.get("symbols", "—"))
    timeframes = meta.get("timeframes", "")
    if isinstance(timeframes, list):
        timeframes = ", ".join(timeframes)
    mode = meta.get("mode", meta.get("breakout_mode", "—"))
    universe_label = meta.get("universe_choice", "")
    sample = meta.get("universe_sample", "")
    total = meta.get("universe_total")
    if sample == "even" and total:
        universe_txt = f"{n_sym} of {total} NIFTY 500 (even sample)"
    elif universe_label:
        universe_txt = f"{n_sym} symbols · {universe_label}"
    else:
        universe_txt = f"{n_sym} symbols"

    st.markdown(
        f"""
<div style="background:linear-gradient(135deg, rgba(240,184,78,.12), rgba(10,10,10,.92));
border:1px solid rgba(240,184,78,.38);border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:0.85rem;">
<p style="color:#ffd36b;margin:0 0 0.35rem;font-size:0.95rem;font-weight:700;">
🕒 Last scanned: {scanned_at}
</p>
<p style="color:#d9cfb6;margin:0;font-size:0.84rem;line-height:1.55;">
<strong>{n_breakouts}</strong> breakouts · <strong>{universe_txt}</strong> ·
<strong>{timeframes or "—"}</strong> · <strong>{mode}</strong>
</p>
<p style="color:#b3a175;margin:0.35rem 0 0;font-size:0.76rem;">
Cached locally in <code>data_cache/scan_results.csv</code> and <code>data_cache/scan_info.csv</code>
</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_disclaimer_footer() -> None:
    st.markdown(
        f"""
<div style="margin-top:2rem;padding:1rem 1.2rem;border-top:1px solid rgba(240,184,78,.40);
background:linear-gradient(180deg, rgba(240,184,78,.06), rgba(6,6,6,.95));border-radius:0 0 10px 10px;">
<p style="font-size:0.78rem;color:#c5b38a;line-height:1.6;margin:0;">
<strong style="color:#ffd36b;">⚖️</strong> Research and education only — not investment advice.
You are solely responsible for your trading decisions.
<a href="{DISCLAIMER_URL}" target="_blank" rel="noopener" style="color:#ffd36b;">Full disclaimer</a>
</p>
<p style="font-size:0.72rem;color:#8f8259;margin:0.5rem 0 0;">
© 2026 TAG · Trade And Grow · MIT License ·
<a href="{PUBLISHED_URL}" style="color:#f0b84e;">tag-breakoutscanner.streamlit.app</a>
</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _breakout_card_html(row: pd.Series) -> str:
    direction = str(row.get("direction", "")).lower()
    cls = "bullish" if direction == "bullish" else "bearish"
    dir_label = _DIR_STYLE.get(direction, ("—", ""))[0]
    tf = row.get("timeframe", "")
    tf_label = TIMEFRAMES[tf].label if tf in TIMEFRAMES else tf
    high52 = row.get("is_52w_high", False)
    if not isinstance(high52, bool):
        high52 = str(high52).strip().lower() in {"1", "true", "yes", "t"}
    break_pct = float(row.get("breakout_pct", 0))
    sign = "+" if break_pct >= 0 else ""
    vol = row.get("volume_ratio")
    vol_txt = f"{float(vol):.2f}×" if vol is not None and pd.notna(vol) else "—"
    tr_atr = row.get("tr_atr_ratio")
    bar_time = row.get("bar_time", "")
    is_strict = str(row.get("mode", "")).lower() == "strict"

    badges = [
        f'<span class="card-pill">{tf_label}</span>',
        f'<span class="card-pill">{dir_label}</span>',
    ]
    if high52:
        badges.append('<span class="card-pill high52">52W High</span>')
    if is_strict:
        badges.append('<span class="card-pill high52">Strict</span>')

    stats_row1 = [
        f'<span class="card-stat">Close <b>₹{float(row["close"]):,.2f}</b></span>',
        f'<span class="card-stat">Break <b>{sign}{break_pct:.2f}%</b></span>',
        f'<span class="card-stat">Vol <b>{vol_txt}</b></span>',
    ]
    if tr_atr is not None and pd.notna(tr_atr):
        stats_row1.append(f'<span class="card-stat">TR/ATR <b>{float(tr_atr):.2f}×</b></span>')

    stats_row2 = [
        f'<span class="card-stat">Level <b>₹{float(row["level"]):,.2f}</b></span>',
        (
            f'<span class="card-stat">Prior H/L '
            f'<b>₹{float(row["prior_high"]):,.0f}</b> / <b>₹{float(row["prior_low"]):,.0f}</b></span>'
        ),
    ]

    return (
        f'<div class="breakout-card {cls}">'
        f'<div class="card-top">'
        f'<span class="card-symbol">{row["symbol"]}</span>'
        f'<span class="card-badges">{"".join(badges)}</span>'
        f"</div>"
        f'<div class="card-stat-row">{"".join(stats_row1)}</div>'
        f'<div class="card-stat-row">{"".join(stats_row2)}</div>'
        f'<span class="card-foot">Bar {bar_time} · Lookback {int(row.get("lookback", 0))} bars</span>'
        f"</div>"
    )


def _render_card_html(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_breakout_cards(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No breakouts in this view.")
        return

    cols_per_row = 3
    for i in range(0, len(df), cols_per_row):
        chunk = df.iloc[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for c_idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[c_idx]:
                _render_card_html(_breakout_card_html(row))


def _style_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["direction"] = out["direction"].map(lambda d: _DIR_STYLE.get(d, ("—", ""))[0])
    out["timeframe"] = out["timeframe"].map(lambda t: TIMEFRAMES[t].label if t in TIMEFRAMES else t)
    out["is_52w_high"] = out["is_52w_high"].map(lambda x: "Yes" if x else "No")
    if "mode" in out.columns:
        out["mode"] = out["mode"].map(lambda m: "Strict (ATR)" if m == "strict" else "Standard")
    rename = {
        "symbol": "Symbol",
        "timeframe": "Timeframe",
        "direction": "Direction",
        "mode": "Mode",
        "close": "Close",
        "level": "Break Level",
        "breakout_pct": "Break %",
        "volume_ratio": "Vol Ratio",
        "tr_atr_ratio": "TR/ATR",
        "true_range": "True Range",
        "atr": "ATR(14)",
        "prior_high": "Prior High",
        "prior_low": "Prior Low",
        "bar_time": "Bar Date",
        "lookback": "Lookback",
        "is_52w_high": "52W High?",
    }
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


@st.cache_data(ttl=300, show_spinner=False)
def _cached_option_chain(symbol: str) -> dict | None:
    return fetch_option_chain(symbol)


def _fno_symbols() -> frozenset[str]:
    if "fno_symbols" not in st.session_state:
        st.session_state.fno_symbols = fno_symbol_set()
    return st.session_state.fno_symbols


def _render_fno_plan(df: pd.DataFrame, key: str) -> None:
    """F&O-only view: CE/PE, strike, expiry, option LTP and rule-based
    entry/TP/SL on the underlying — general stock columns omitted."""
    plan_df = df[df["symbol"].astype(str).str.upper().isin(_fno_symbols())]
    if plan_df.empty:
        return

    st.markdown(f"##### 🎯 F&O options plan — {len(plan_df)} signal(s)")
    live = st.toggle(
        "Fetch live strike / expiry / option LTP from NSE",
        value=False,
        key=f"fno_live_{key}",
        help="Needs direct NSE access (works locally in India; usually blocked on Streamlit Cloud). "
        "Off = estimated ATM strike and expiry, no option LTP.",
    )

    rows = []
    capped = plan_df.head(15)
    for _, r in capped.iterrows():
        chain = _cached_option_chain(str(r["symbol"])) if live else None
        p = build_plan(str(r["symbol"]), str(r["direction"]), float(r["close"]), float(r["level"]), chain)
        scanned = r.get("scanned_at")
        rows.append(
            {
                "Symbol": p.symbol,
                "TF": r.get("timeframe", ""),
                "Signal date": str(r.get("bar_time", "") or "—"),
                "CE/PE": p.opt_type,
                "Strike": f"{p.strike:g}",
                "Expiry": p.expiry,
                "Prem Entry ₹": f"{p.option_ltp:,.2f}" if p.option_ltp is not None else "—",
                "Prem TP1/2/3 ₹": (
                    " / ".join(f"{v:,.2f}" for v in p.prem_tps) if p.prem_tps else "—"
                ),
                "Prem SL ₹": f"{p.prem_sl:,.2f}" if p.prem_sl is not None else "—",
                "Spot Entry ₹": f"{p.entry:,.2f}",
                "Spot TP1/2/3 ₹": " / ".join(f"{v:,.2f}" for v in p.tps),
                "Spot SL ₹": f"{p.sl:,.2f}",
                "Scanned (IST)": format_scanned_at(scanned, short=True) if scanned and pd.notna(scanned) else "—",
                "Source": "NSE live" if p.live else "estimated",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, key=f"fno_plan_{key}")
    if len(plan_df) > len(capped):
        st.caption(f"Showing first {len(capped)} of {len(plan_df)} F&O signals.")
    if not live:
        st.info(
            "💡 **Premium columns need live NSE data** — turn on the toggle above (works when NSE is "
            "reachable, e.g. running locally in India). Premiums are never estimated offline."
        )
    st.caption(
        "⚖️ **Prem** = option premium: entry is the live NSE LTP; TP/SL premiums are Black–Scholes "
        "repricings of the same option at each spot target (chain IV, anchored to the LTP, no time decay). "
        "**Spot** = underlying levels: entry = scan close · SL = break level ∓0.5% · TP1/2/3 = 1R/2R/3R · "
        "ATM strike, nearest expiry · bullish → CE, bearish → PE. Algorithmic template for research — "
        "**not** trade advice; verify option liquidity and margins with your broker."
    )


def _row_plan(row: pd.Series) -> OptionPlan:
    return build_plan(
        str(row["symbol"]), str(row["direction"]), float(row["close"]), float(row["level"])
    )


def _chart(symbol: str, timeframe: str, level: float, plan: OptionPlan | None = None) -> go.Figure:
    df = load_bars(symbol, timeframe, use_cache=True)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{symbol} — no data")
        return fig

    tail_n = {"1H": 120, "1D": 120, "1W": 52}.get(timeframe.upper(), 100)
    tail = df.tail(tail_n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)

    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["open"],
            high=tail["high"],
            low=tail["low"],
            close=tail["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=level, line_dash="dash", line_color="#f59e0b", annotation_text="Break level", row=1, col=1)

    if plan is not None:
        fig.add_hline(
            y=plan.entry,
            line_color="#ffd36b",
            annotation_text=f"Entry {plan.entry:,.2f}",
            annotation_position="top left",
            row=1,
            col=1,
        )
        for i, tp in enumerate(plan.tps, start=1):
            fig.add_hline(
                y=tp,
                line_dash="dot",
                line_color="#5dbb7f",
                annotation_text=f"TP{i} {tp:,.2f}",
                row=1,
                col=1,
            )
        fig.add_hline(
            y=plan.sl,
            line_dash="dash",
            line_color="#e06552",
            annotation_text=f"SL {plan.sl:,.2f}",
            annotation_position="bottom left",
            row=1,
            col=1,
        )

    if "volume" in tail.columns:
        colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(tail["close"], tail["open"])]
        fig.add_trace(
            go.Bar(x=tail.index, y=tail["volume"], marker_color=colors, name="Volume", showlegend=False),
            row=2,
            col=1,
        )

    tf_label = TIMEFRAMES.get(timeframe.upper())
    title = f"{symbol} — {tf_label.label if tf_label else timeframe} breakout"
    fig.update_layout(
        title=title,
        height=480,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d24",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _cpr_status_rate(row: pd.Series) -> str:
    """Status/Rate column: index LTP or type badge (V + W, WIDE, etc.)."""
    sym = str(row.get("symbol", "")).upper()
    if sym in _INDEX_SYMBOLS:
        ltp = row.get("ltp")
        return f"{float(ltp):,.2f}" if ltp is not None and pd.notna(ltp) else "—"
    cpr_type = str(row.get("type", "—"))
    return cpr_type.replace("+", " + ")


def _normalize_cpr_results(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure CPR scan frames have expected columns after CSV round-trip."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ("is_virgin", "is_narrow"):
        if col in out.columns:
            out[col] = out[col].map(_parse_bool_value)
    for col in ("distance_pct", "width_pct", "width_percentile", "ltp", "tc", "bc", "pivot"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "is_narrow" not in out.columns:
        out["is_narrow"] = False
    if "type" not in out.columns:
        out["type"] = "—"
    return out


def _parse_bool_value(val: object) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "t"}


def _cpr_metric_count(df: pd.DataFrame, col: str, *, match: str | None = None) -> int:
    if df.empty or col not in df.columns:
        return 0
    if match is None:
        return int(df[col].sum())
    return int((df[col] == match).sum())


def _style_cpr_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    display = df.copy()
    display["status_rate"] = display.apply(_cpr_status_rate, axis=1)
    if "is_virgin" in display.columns:
        display["virgin_status"] = display["is_virgin"].map(lambda x: "VIRGIN" if x else "TOUCHED")
    else:
        display["virgin_status"] = "—"
    if "distance_pct" in display.columns:
        display["distance"] = display["distance_pct"].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
    else:
        display["distance"] = "—"
    if "trend" in display.columns:
        display["trend_icon"] = display["trend"].map(_CPR_TREND_ICONS).fillna("—")
    else:
        display["trend_icon"] = "—"
    display = display.drop(columns=["type"], errors="ignore")
    return display.rename(
        columns={
            "symbol": "Symbol",
            "status_rate": "Status / Rate",
            "virgin_status": "Type",
            "distance": "Distance",
            "trend_icon": "Trend",
            "ltp": "LTP",
            "tc": "TC",
            "bc": "BC",
            "pivot": "Pivot",
            "width_pct": "CPR Width %",
            "width_percentile": "Width %ile",
            "days_virgin": "Days Virgin",
            "source_date": "CPR From",
            "session_date": "Session",
        }
    )


def _cpr_chart(symbol: str, timeframe: str = "Daily") -> go.Figure:
    daily = load_daily(symbol, use_cache=True)
    fig = go.Figure()
    if daily.empty:
        fig.update_layout(title=f"{symbol} — no data")
        return fig

    tail = daily.tail(60)
    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["open"],
            high=tail["high"],
            low=tail["low"],
            close=tail["close"],
            name=symbol,
        )
    )

    levels = levels_for_chart(daily, timeframe=timeframe)
    colors = {
        "CPR HIGH (TC)": "#ef4444",
        "CPR TC": "#f97316",
        "CPR PIVOT": "#a855f7",
        "CPR BC": "#3b82f6",
        "CPR LOW (BC)": "#22c55e",
        "R1": "#eab308",
        "S1": "#14b8a6",
    }
    for label, price in levels.items():
        fig.add_hline(
            y=price,
            line_dash="dot" if "BC" in label or "TC" in label else "dash",
            line_color=colors.get(label, "#94a3b8"),
            annotation_text=f"{label} {price:.2f}",
        )

    fig.update_layout(
        title=f"{symbol} — {timeframe} CPR levels",
        xaxis_rangeslider_visible=False,
        height=480,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _render_cpr_legend() -> None:
    html = """
<div class="cpr-legend">
  <span class="cpr-legend-item vw">V + W — Virgin + Wide (high probability)</span>
  <span class="cpr-legend-item vn">V + N — Virgin + Narrow (trend continuation)</span>
  <span class="cpr-legend-item wide">WIDE — Wide CPR (move may expand)</span>
  <span class="cpr-legend-item virgin">VIRGIN — Untouched CPR</span>
  <span class="cpr-legend-item touched">TOUCHED — CPR already touched</span>
</div>
"""
    _render_card_html(html)


def _render_narrow_cpr_controls(cached_meta: dict) -> float:
    """Narrow CPR preset selector with optional custom percentile."""
    saved = cached_meta.get("narrow_percentile")
    default_pct = float(saved) if saved not in (None, "") else NARROW_PERCENTILE
    preset_labels = [f"{int(p)}%" for p in NARROW_PERCENTILE_PRESETS] + ["Custom"]
    default_preset = (
        f"{int(default_pct)}%"
        if default_pct in NARROW_PERCENTILE_PRESETS
        else "Custom"
    )

    st.markdown("##### Narrow CPR threshold")
    preset = st.selectbox(
        "Bottom percentile (narrowest widths)",
        preset_labels,
        index=preset_labels.index(default_preset) if default_preset in preset_labels else len(preset_labels) - 1,
        key="cpr_narrow_preset",
        help=(
            "Compare today's CPR width to each symbol's last 1-year history. "
            "**Narrow** when width ranks in the bottom X% (e.g. 5% = narrowest 5% of past widths)."
        ),
    )

    if preset == "Custom":
        st.session_state["cpr_narrow_custom_enabled"] = True
        custom_default = int(default_pct) if default_pct not in NARROW_PERCENTILE_PRESETS else int(NARROW_PERCENTILE)
        custom_pct = st.slider(
            "Custom bottom percentile (%)",
            int(NARROW_PERCENTILE_MIN),
            int(NARROW_PERCENTILE_MAX),
            min(max(custom_default, int(NARROW_PERCENTILE_MIN)), int(NARROW_PERCENTILE_MAX)),
            1,
            key="cpr_narrow_custom_pct",
        )
        narrow_pct = float(custom_pct)
    else:
        st.session_state["cpr_narrow_custom_enabled"] = False
        narrow_pct = float(preset.rstrip("%"))

    st.caption(
        f"Stocks with CPR width in the **bottom {narrow_pct:g}%** of their own 1-year history "
        f"are tagged **Narrow** (V + N when also Virgin)."
    )
    return narrow_pct


def _render_cpr_table(display: pd.DataFrame) -> None:
    table_cols = [
        c
        for c in [
            "Symbol",
            "Status / Rate",
            "Type",
            "Distance",
            "Trend",
            "LTP",
            "CPR Width %",
            "Width %ile",
            "TC",
            "BC",
            "Pivot",
            "Days Virgin",
            "CPR From",
            "Session",
        ]
        if c in display.columns
    ]
    if display.empty or not table_cols:
        st.info("No CPR rows match the current filters.")
        return

    def _color_type(val: str) -> str:
        if val == "VIRGIN":
            return "color: #4ade80; font-weight: 700"
        if val == "TOUCHED":
            return "color: #f87171; font-weight: 700"
        return ""

    def _color_status(val: str) -> str:
        key = str(val).replace(" ", "")
        if key == "V+W":
            return "background-color: rgba(124,58,237,0.35); color: #ddd6fe; font-weight: 700"
        if key == "V+N":
            return "background-color: rgba(37,99,235,0.35); color: #bfdbfe; font-weight: 700"
        if val == "WIDE":
            return "background-color: rgba(220,38,38,0.35); color: #fecaca; font-weight: 700"
        return ""

    subset = display[table_cols]
    try:
        styler = subset.style
        map_fn = getattr(styler, "map", styler.applymap)
        styled = map_fn(_color_status, subset=["Status / Rate"])
        map_fn2 = getattr(styled, "map", styled.applymap)
        styled = map_fn2(_color_type, subset=["Type"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(subset, use_container_width=True, hide_index=True)


def render_cpr_tab(
    scan_symbols: list[str],
    *,
    universe_choice: str,
    universe_total: int,
    universe_sample: str,
) -> None:
    st.markdown(
        """
<div style="background:linear-gradient(135deg,#1e1b4b 0%,#312e81 55%,#1e3a5f 100%);
padding:1rem 1.25rem;border-radius:12px;margin-bottom:1rem;">
<h3 style="color:white;margin:0;">📊 Virgin CPR Scanner</h3>
<p style="color:#c7d2fe;margin:0.35rem 0 0;font-size:0.9rem;">
Today's CPR from yesterday's OHLC. <strong>Virgin</strong> = price has not touched the CPR zone [BC, TC] today.
</p></div>
""",
        unsafe_allow_html=True,
    )

    _render_cpr_legend()

    cached_cpr_df, cached_cpr_meta = load_cpr_results()
    cpr_meta = cached_cpr_meta or {}

    c1, c2 = st.columns(2)
    with c1:
        cpr_timeframe = st.radio(
            "CPR timeframe",
            ["Daily", "Weekly"],
            horizontal=True,
            key="cpr_timeframe",
        )
    with c2:
        st.metric("Symbols in scan", len(scan_symbols))

    narrow_pct = _render_narrow_cpr_controls(cpr_meta)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        virgin_only = st.checkbox("Virgin only", value=False, key="cpr_virgin_only")
    with f2:
        narrow_only = st.checkbox(
            "Narrow only",
            value=False,
            key="cpr_narrow_only",
            help=f"Show only symbols in the bottom {narrow_pct:g}% narrow CPR bucket.",
        )
    with f3:
        type_filter = st.multiselect(
            "Type filter",
            ["V+W", "V+N", "V", "WIDE", "NARROW", "TOUCHED"],
            default=["V+W", "V+N", "V", "WIDE", "NARROW", "TOUCHED"],
            key="cpr_type_filter",
        )
    with f4:
        trend_filter = st.selectbox("Trend", ["All", "above", "below", "inside"], key="cpr_trend_filter")

    use_cache = st.checkbox("Use price cache", value=True, key="cpr_use_cache")
    force_cpr = st.button("🔍 Scan CPR", type="primary", key="cpr_force_scan")

    if not force_cpr and cached_cpr_df is not None and "cpr_results" not in st.session_state:
        st.session_state["cpr_results"] = _normalize_cpr_results(cached_cpr_df)
        st.session_state["cpr_scan_meta"] = cached_cpr_meta

    if force_cpr:
        progress = st.progress(0.0, text="Scanning CPR…")
        status = st.empty()

        def on_cpr_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / max(total, 1), text=f"CPR scan {label} ({done}/{total})…")

        with st.spinner("Computing Virgin CPR…"):
            raw = scan_cpr_universe(
                scan_symbols,
                progress_callback=on_cpr_progress,
                use_cache=use_cache,
                narrow_percentile=float(narrow_pct),
                timeframe=cpr_timeframe,
            )
        progress.progress(1.0, text="Done")
        status.empty()

        scan_meta = {
            "symbols": len(scan_symbols),
            "universe_total": universe_total,
            "universe_sample": universe_sample,
            "universe_choice": universe_choice,
            "timeframe": cpr_timeframe,
            "narrow_percentile": narrow_pct,
        }
        save_cpr_results(raw, scan_meta)
        _, saved_meta = load_cpr_results()
        st.session_state["cpr_results"] = _normalize_cpr_results(raw)
        st.session_state["cpr_scan_meta"] = saved_meta or scan_meta
        st.session_state["cpr_scan_timeframe"] = cpr_timeframe
        if chart_symbols := raw["symbol"].astype(str).tolist() if "symbol" in raw.columns else []:
            prev_pick = st.session_state.get("cpr_chart_pick")
            if prev_pick not in chart_symbols:
                st.session_state["cpr_chart_pick"] = chart_symbols[0]
        st.success(f"CPR scan complete — {len(raw)} symbols")

    results = st.session_state.get("cpr_results")
    if results is None or (isinstance(results, pd.DataFrame) and results.empty):
        if cached_cpr_scan_available():
            st.info("Cached CPR results available. Click **Scan CPR** to refresh, or load from cache on next visit.")
        else:
            st.info("Click **Scan CPR** to build the Virgin CPR screener.")
        return

    scan_tf = st.session_state.get("cpr_scan_timeframe", cached_cpr_meta.get("timeframe", "Daily"))
    if scan_tf != cpr_timeframe:
        st.warning(
            f"Selected timeframe ({cpr_timeframe}) differs from cached scan ({scan_tf}). "
            "Click **Scan CPR** to refresh."
        )

    meta = st.session_state.get("cpr_scan_meta", cached_cpr_meta or {})
    results = _normalize_cpr_results(results)

    results = apply_narrow_percentile(results, float(narrow_pct))
    scan_narrow_raw = meta.get("narrow_percentile")
    try:
        scan_narrow = float(scan_narrow_raw) if scan_narrow_raw not in (None, "") else float(narrow_pct)
    except (TypeError, ValueError):
        scan_narrow = float(narrow_pct)
    if abs(scan_narrow - narrow_pct) > 0.01:
        st.info(
            f"Narrow threshold changed to **bottom {narrow_pct:g}%** — "
            "types updated instantly from cached width percentiles (no rescan needed)."
        )
    filtered = filter_cpr_results(
        results,
        virgin_only=virgin_only,
        narrow_only=narrow_only,
        types=type_filter or None,
        trend=trend_filter,
    )

    scanned_label = meta.get("scanned_at_display") or format_scanned_at(
        meta.get("scanned_at"), short=True
    )
    session_date = (
        filtered["session_date"].iloc[0]
        if not filtered.empty and "session_date" in filtered.columns
        else results["session_date"].iloc[0] if "session_date" in results.columns else "—"
    )

    _render_cpr_summary_metrics(
        total=len(filtered),
        virgin=_cpr_metric_count(filtered, "is_virgin"),
        v_w=_cpr_metric_count(filtered, "type", match="V+W"),
        v_n=_cpr_metric_count(filtered, "type", match="V+N"),
        narrow=_cpr_metric_count(filtered, "is_narrow"),
        narrow_pct=narrow_pct,
        scanned_label=scanned_label or "—",
    )

    _render_cpr_session_panel(
        session_date=session_date,
        scan_tf=scan_tf,
        universe_choice=universe_choice,
        narrow_pct=narrow_pct,
    )

    display = _style_cpr_results(filtered)
    _render_cpr_table(display)

    chart_symbols = (
        filtered["symbol"].astype(str).tolist()
        if not filtered.empty and "symbol" in filtered.columns
        else results["symbol"].astype(str).tolist() if "symbol" in results.columns else []
    )
    if chart_symbols:
        prev_pick = st.session_state.get("cpr_chart_pick")
        if prev_pick not in chart_symbols:
            st.session_state["cpr_chart_pick"] = chart_symbols[0]
        pick = st.selectbox("Chart symbol", chart_symbols, key="cpr_chart_pick")
        st.plotly_chart(_cpr_chart(pick, timeframe=scan_tf), use_container_width=True, key="cpr_plotly_chart")
        row = filtered[filtered["symbol"].astype(str) == pick] if "symbol" in filtered.columns else pd.DataFrame()
        if row.empty and "symbol" in results.columns:
            row = results[results["symbol"].astype(str) == pick]
        if not row.empty:
            r = row.iloc[0]
            width_pctile = r.get("width_percentile")
            dist = r.get("distance_pct")
            width_txt = f"{float(width_pctile):.1f}" if pd.notna(width_pctile) else "—"
            dist_txt = f"{float(dist):+.2f}%" if pd.notna(dist) else "—"
            _render_card_html(
                f"""
<div class="cpr-detail-stats">
  <span class="cpr-detail-stat">Type <b>{r.get("type", "—")}</b></span>
  <span class="cpr-detail-stat">Virgin <b>{"Yes" if r.get("is_virgin") else "Touched"}</b></span>
  <span class="cpr-detail-stat">Width %ile <b>{width_txt}</b></span>
  <span class="cpr-detail-stat">TC / BC <b>{r.get("tc", "—")} / {r.get("bc", "—")}</b></span>
  <span class="cpr-detail-stat">Distance <b>{dist_txt}</b></span>
</div>
"""
            )

        st.download_button(
            "Download CPR CSV",
            filtered.to_csv(index=False),
            file_name="cpr_scan.csv",
            mime="text/csv",
            key="cpr_download",
        )
        st.caption("⚖️ Exported data is for research use only — not a recommendation to trade.")


def _review_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if kind == "breakout":
        out["Symbol"] = df["symbol"]
        out["TF"] = df["timeframe"]
        out["Direction"] = df["direction"].astype(str).str.title()
        out["Close ₹"] = pd.to_numeric(df["close"], errors="coerce").map(lambda v: f"{v:,.2f}")
        out["Break %"] = pd.to_numeric(df["breakout_pct"], errors="coerce").map(lambda v: f"{v:+.2f}%")
        if "volume_ratio" in df.columns:
            out["Vol ×"] = pd.to_numeric(df["volume_ratio"], errors="coerce").map(
                lambda v: f"{v:.2f}×" if pd.notna(v) else "—"
            )
        out["Level ₹"] = pd.to_numeric(df["level"], errors="coerce").map(lambda v: f"{v:,.2f}")
        if "bar_time" in df.columns:
            out["Bar date"] = df["bar_time"].astype(str)
    else:
        out["Symbol"] = df["symbol"]
        if "type" in df.columns:
            out["Type"] = df["type"]
        if "ltp" in df.columns:
            out["LTP ₹"] = pd.to_numeric(df["ltp"], errors="coerce").map(
                lambda v: f"{v:,.2f}" if pd.notna(v) else "—"
            )
        if "width_pct" in df.columns:
            out["Width %"] = pd.to_numeric(df["width_pct"], errors="coerce").map(
                lambda v: f"{v:.2f}%" if pd.notna(v) else "—"
            )
        if "width_percentile" in df.columns:
            out["Width %ile"] = pd.to_numeric(df["width_percentile"], errors="coerce").map(
                lambda v: f"{v:.0f}" if pd.notna(v) else "—"
            )
        if "distance_pct" in df.columns:
            out["Dist %"] = pd.to_numeric(df["distance_pct"], errors="coerce").map(
                lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
            )
    if "scanned_at" in df.columns:
        out["Scanned (IST)"] = df["scanned_at"].map(lambda v: format_scanned_at(v, short=True))
    return out


def render_fno_tab() -> None:
    st.markdown("#### 🎯 F&O options plan")
    st.caption(
        "Options view of the current breakout signals for F&O-listed stocks — "
        "CE/PE, strike, expiry, premium and spot entry/TP/SL."
    )

    results = st.session_state.get("breakout_results")
    if results is None or (isinstance(results, pd.DataFrame) and results.empty):
        results, _ = load_scan_results()
    if results is None or results.empty:
        st.info("No breakout scan results yet — run **Force Refresh Scan** in the Breakout Scanner tab.")
        return

    fno_df = results[results["symbol"].astype(str).str.upper().isin(_fno_symbols())]
    if fno_df.empty:
        st.info("No F&O-listed stocks in the current breakout results.")
        return

    tf_options = ["All"] + [t for t in TIMEFRAME_ORDER if t in set(fno_df["timeframe"])]
    tf_pick = st.radio("Timeframe", tf_options, horizontal=True, key="fno_tab_tf")
    view = fno_df if tf_pick == "All" else fno_df[fno_df["timeframe"] == tf_pick]

    _render_fno_plan(view, "fno_tab")

    if not view.empty:
        pick = st.selectbox("Chart symbol", view["symbol"].tolist(), key="fno_tab_chart_pick")
        row = view[view["symbol"] == pick].iloc[0]
        st.plotly_chart(
            _chart(pick, row["timeframe"], float(row["level"]), plan=_row_plan(row)),
            use_container_width=True,
            key=f"fno_tab_chart_{pick}",
        )
        st.caption(
            "Chart levels — 🟡 Entry · 🟢 TP1/TP2/TP3 (dotted) · 🔴 SL (dashed): "
            "rule-based template, not advice."
        )


def _audit_signal(row: pd.Series) -> dict | None:
    """Compare one logged signal against real subsequent price movement."""
    symbol = str(row["symbol"])
    tf = str(row["timeframe"])
    direction = str(row["direction"]).lower()
    bullish = direction == "bullish"
    entry = float(row["close"])
    plan = _row_plan(row)

    bars = load_bars(symbol, tf, use_cache=True)
    if bars is None or bars.empty:
        return None

    # anchor = when the signal was logged (scan time, IST); fall back to bar date EOD
    anchor = pd.to_datetime(row.get("scanned_at"), errors="coerce")
    if pd.isna(anchor):
        anchor = pd.to_datetime(str(row.get("bar_time")), errors="coerce")
        if pd.isna(anchor):
            return None
        anchor = anchor + pd.Timedelta(hours=15, minutes=30)
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert("Asia/Kolkata").tz_localize(None)

    idx = bars.index
    try:
        idx_cmp = idx.tz_convert("Asia/Kolkata").tz_localize(None) if idx.tz is not None else idx
    except (TypeError, AttributeError):
        idx_cmp = idx
    post = bars.loc[idx_cmp > anchor]

    latest = float(bars["close"].iloc[-1])
    move_pct = (latest - entry) / entry * 100
    pnl_pct = move_pct if bullish else -move_pct

    tp_hit, sl_hit, ambiguous = 0, False, False
    peak = entry
    for _, b in post.iterrows():
        hi, lo = float(b["high"]), float(b["low"])
        peak = max(peak, hi) if bullish else min(peak, lo)
        if bullish:
            bar_tp = sum(hi >= tp for tp in plan.tps)
            bar_sl = lo <= plan.sl
        else:
            bar_tp = sum(lo <= tp for tp in plan.tps)
            bar_sl = hi >= plan.sl
        if bar_sl:
            sl_hit = True
            if bar_tp > tp_hit:
                ambiguous = True
                tp_hit = bar_tp
            break
        tp_hit = max(tp_hit, bar_tp)
        if tp_hit == 3:
            break
    peak_pct = (peak - entry) / entry * 100
    peak_pct = peak_pct if bullish else -peak_pct

    if post.empty:
        status = "⏳ awaiting bars"
    elif sl_hit and tp_hit == 0:
        status = "🔴 SL hit"
    elif sl_hit:
        status = f"🎯 TP{tp_hit} ✓ → SL" + (" ⚠️ same bar" if ambiguous else "")
    elif tp_hit > 0:
        status = f"🎯 TP{tp_hit} hit"
    else:
        status = "⏳ running"

    return {
        "Symbol": symbol,
        "TF": tf,
        "Dir": "🟢" if bullish else "🔴",
        "Signal date": str(row.get("bar_time", "") or "—"),
        "Entry ₹": f"{entry:,.2f}",
        "Latest ₹": f"{latest:,.2f}",
        "P&L %": f"{pnl_pct:+.2f}%",
        "Peak %": f"{peak_pct:+.2f}%",
        "Status": status,
        "_pnl": pnl_pct,
        "_tp": tp_hit,
        "_sl": sl_hit,
        "_open": not sl_hit and tp_hit < 3 and not post.empty,
    }


def _render_signal_audit(bdf: pd.DataFrame) -> None:
    st.divider()
    st.markdown("##### 📊 Signal audit — suggested signals vs real movement")
    run = st.toggle(
        "Run audit against live prices",
        value=False,
        key="review_audit",
        help="Fetches current bars for each logged signal and checks whether the rule-based "
        "TP/SL levels were actually reached after the signal was logged.",
    )
    if not run:
        return

    b = bdf.copy()
    if "scanned_at" in b.columns:
        keys = [k for k in ("symbol", "timeframe", "direction", "bar_time") if k in b.columns]
        b = b.sort_values("scanned_at").drop_duplicates(subset=keys, keep="last")
    b = b.head(30)

    results = []
    progress = st.progress(0.0, text="Auditing signals…")
    for i, (_, row) in enumerate(b.iterrows(), start=1):
        try:
            audited = _audit_signal(row)
        except Exception:
            audited = None
        if audited:
            results.append(audited)
        progress.progress(i / len(b), text=f"Auditing {row['symbol']} ({i}/{len(b)})…")
    progress.empty()

    if not results:
        st.warning("Could not audit any signals (no price data).")
        return

    adf = pd.DataFrame(results)
    n_tp = int((adf["_tp"] > 0).sum())
    n_sl = int(adf["_sl"].sum())
    n_open = int(adf["_open"].sum())
    avg_pnl = adf["_pnl"].mean()
    st.markdown(
        f"**{len(adf)}** audited · 🎯 **{n_tp}** reached ≥ TP1 · 🔴 **{n_sl}** hit SL · "
        f"⏳ **{n_open}** running · avg underlying P&L **{avg_pnl:+.2f}%**"
    )
    st.dataframe(
        adf.drop(columns=["_pnl", "_tp", "_sl", "_open"]),
        hide_index=True,
        use_container_width=True,
        key="review_audit_table",
    )
    st.caption(
        "⚖️ Audit of **logged signals only**, on the underlying: P&L assumes entry at the scan close, "
        "direction-adjusted; TP/SL checks walk the real bars recorded **after** each signal's scan time. "
        "Same-bar TP+SL is flagged ⚠️ (order unknowable from bar data). Historical observation — not a "
        "performance promise."
    )


def render_daily_review_tab() -> None:
    st.markdown("#### 📅 Daily Review — top picks by category")
    st.caption(
        "⚖️ The day's algorithmic scan output grouped for review, with scan date & time (IST). "
        "Research only — not trade recommendations."
    )

    hist_dates = sorted(set(list_history_dates("breakout")) | set(list_history_dates("cpr")), reverse=True)

    if hist_dates:
        mode = st.radio(
            "Review period",
            ["Single day", "Custom date range"],
            horizontal=True,
            key="review_mode",
            label_visibility="collapsed",
        )
        if mode == "Single day":
            day = st.selectbox(
                "Review date",
                hist_dates,
                format_func=lambda d: d.strftime("%A, %d %b %Y"),
                key="review_date",
            )
            bdf = load_history("breakout", day)
            cdf = load_history("cpr", day)
        else:
            oldest, newest = min(hist_dates), max(hist_dates)
            default_start = max(oldest, newest - timedelta(days=6))
            picked = st.date_input(
                "Date range",
                value=(default_start, newest),
                min_value=oldest,
                max_value=newest,
                key="review_range",
                help=f"History recorded since {oldest.strftime('%d %b %Y')} — every scan day is kept, "
                "so the range can span months or years as the archive grows.",
            )
            if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
                st.info("Pick both a start and an end date.")
                return
            start, end = picked
            days_in_range = [d for d in hist_dates if start <= d <= end]
            if not days_in_range:
                st.warning("No recorded scan days in that range.")
                return
            st.caption(
                f"**{len(days_in_range)}** recorded day(s): "
                f"{min(days_in_range).strftime('%d %b %Y')} → {max(days_in_range).strftime('%d %b %Y')}"
            )
            b_parts = [x for x in (load_history("breakout", d) for d in days_in_range) if x is not None]
            c_parts = [x for x in (load_history("cpr", d) for d in days_in_range) if x is not None]
            bdf = pd.concat(b_parts, ignore_index=True) if b_parts else None
            cdf = pd.concat(c_parts, ignore_index=True) if c_parts else None
    else:
        st.info(
            "No scan history recorded yet — showing the latest cached scan. "
            "A dated history entry is written automatically on every **Force Refresh Scan**."
        )
        bdf, _ = load_scan_results()
        cdf, _ = load_cpr_results()

    if (bdf is None or bdf.empty) and (cdf is None or cdf.empty):
        st.warning("Nothing to review for this date.")
        return

    stamps: list[str] = []
    for df in (bdf, cdf):
        if df is not None and "scanned_at" in df.columns:
            stamps.extend(str(v) for v in df["scanned_at"].dropna().unique())
    if stamps:
        stamps = sorted(set(stamps))
        st.markdown(
            f"🕒 **{len(stamps)}** scan run(s) · first **{format_scanned_at(stamps[0], short=True)}** · "
            f"last **{format_scanned_at(stamps[-1], short=True)}** IST"
        )

    top_n = st.slider("Top picks per category", 3, 15, 5, key="review_top_n")

    sections: list[tuple[str, pd.DataFrame, str]] = []

    if bdf is not None and not bdf.empty:
        b = bdf.copy()
        b["breakout_pct"] = pd.to_numeric(b["breakout_pct"], errors="coerce")
        vol = pd.to_numeric(b["volume_ratio"], errors="coerce") if "volume_ratio" in b.columns else 1.0
        b["_score"] = b["breakout_pct"].abs() * pd.Series(vol, index=b.index).fillna(1.0)
        if "scanned_at" in b.columns:
            # latest run per signal; bar_time in the key keeps each day's
            # distinct signals when reviewing a multi-day range
            keys = [k for k in ("symbol", "timeframe", "direction", "bar_time") if k in b.columns]
            b = b.sort_values("scanned_at").drop_duplicates(subset=keys, keep="last")
        for tf in [t for t in TIMEFRAME_ORDER if t in set(b["timeframe"])]:
            for direction, icon in (("bullish", "🟢"), ("bearish", "🔴")):
                sub = b[(b["timeframe"] == tf) & (b["direction"].astype(str).str.lower() == direction)]
                if not sub.empty:
                    label = TIMEFRAMES[tf].label if tf in TIMEFRAMES else tf
                    sections.append((f"{icon} {direction.title()} · {label}", sub, "breakout"))
        if "is_52w_high" in b.columns and b["is_52w_high"].any():
            sections.append(("🏆 52-Week High breaks", b[b["is_52w_high"]], "breakout"))
        if "mode" in b.columns:
            strict = b[b["mode"].astype(str).str.lower() == "strict"]
            if not strict.empty:
                sections.append(("💪 Strict-mode passes", strict, "breakout"))

    if cdf is not None and not cdf.empty:
        c = cdf.copy()
        if "scanned_at" in c.columns:
            c_keys = [k for k in ("symbol", "session_date") if k in c.columns]
            c = c.sort_values("scanned_at").drop_duplicates(subset=c_keys, keep="last")
        width_pctile = (
            pd.to_numeric(c["width_percentile"], errors="coerce") if "width_percentile" in c.columns else None
        )
        # narrower CPR ranks higher for review priority
        c["_score"] = -(width_pctile.fillna(100.0)) if width_pctile is not None else 0.0
        if "is_virgin" in c.columns and c["is_virgin"].any():
            sections.append(("✨ Virgin CPR", c[c["is_virgin"]], "cpr"))
        if "type" in c.columns:
            for cpr_type, label in (("V+W", "🟣 Virgin + Wide CPR"), ("V+N", "🔵 Virgin + Narrow CPR")):
                sub = c[c["type"].astype(str) == cpr_type]
                if not sub.empty:
                    sections.append((label, sub, "cpr"))
        if "is_narrow" in c.columns and c["is_narrow"].any():
            sections.append(("📏 Narrow CPR", c[c["is_narrow"]], "cpr"))

    if not sections:
        st.warning("No categorised signals for this date.")
        return

    cols = st.columns(2)
    for i, (title, sub, kind) in enumerate(sections):
        with cols[i % 2]:
            ranked = sub.sort_values("_score", ascending=False)
            st.markdown(f"##### {title} — {len(ranked)}")
            st.dataframe(
                _review_table(ranked.head(top_n), kind),
                hide_index=True,
                use_container_width=True,
                key=f"review_{i}",
            )
            if len(ranked) > top_n:
                with st.expander(f"All {len(ranked)} in this category"):
                    st.dataframe(
                        _review_table(ranked, kind),
                        hide_index=True,
                        use_container_width=True,
                        key=f"review_all_{i}",
                    )

    st.caption("⚖️ Ranked by |break %| × volume ratio (breakouts) and narrowest width percentile (CPR).")

    if bdf is not None and not bdf.empty:
        _render_signal_audit(bdf)


def render_breakout_tab(
    scan_symbols: list[str],
    *,
    universe_choice: str,
    universe_total: int,
    universe_sample: str,
) -> None:
    symbols = scan_symbols
    dir_filter = None if direction == "Both" else direction.lower()

    cached_df, cached_meta = load_scan_results()

    force_refresh = st.button(
        "Force Refresh Scan",
        type="primary",
        help="Re-run scan and overwrite local CSV cache.",
        key="breakout_force_refresh",
    )

    if not force_refresh and cached_df is not None and "breakout_results" not in st.session_state:
        st.session_state["breakout_results"] = cached_df
        st.session_state["breakout_scan_meta"] = cached_meta

    if force_refresh:
        progress = st.progress(0.0, text="Loading prices…")
        status = st.empty()

        def on_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / max(total, 1), text=f"Loading {label} ({done}/{total})…")
            status.caption(label)

        with st.spinner("Scanning breakouts…"):
            raw = scan_universe(
                symbols,
                selected_tfs or ["1D"],
                mode=breakout_mode,
                progress_callback=on_progress,
                use_cache=use_cache,
                vol_mult=vol_mult,
                lookback=lookback,
                atr_mult=atr_mult if breakout_mode == "strict" else None,
                direction_filter=dir_filter,
            )
            filtered = filter_results(
                raw,
                timeframes=selected_tfs,
                directions=[dir_filter] if dir_filter else None,
                min_vol_ratio=vol_mult,
                only_52w=only_52w,
            )

        progress.progress(1.0, text="Done")
        status.empty()

        scan_meta = {
            "symbols": len(symbols),
            "universe_total": universe_total,
            "universe_sample": universe_sample,
            "universe_choice": universe_choice,
            "timeframes": sort_timeframes(selected_tfs or ["1D"]),
            "mode": breakout_mode_label,
            "breakout_mode": breakout_mode,
            "direction": direction,
            "vol_mult": vol_mult,
            "lookback": lookback,
            "atr_mult": atr_mult if breakout_mode == "strict" else None,
            "only_52w": only_52w,
            "max_symbols": max_symbols,
        }
        save_scan_results(filtered, scan_meta)
        _, saved_meta = load_scan_results()
        st.session_state["breakout_results"] = filtered
        st.session_state["breakout_scan_meta"] = saved_meta or scan_meta
        scanned_display = format_scanned_at(
            st.session_state.get("breakout_scan_meta", {}).get("scanned_at")
        )
        st.success(
            f"Scan complete — {len(filtered)} breakouts saved at **{scanned_display}** "
            f"to `data_cache/scan_results.csv`."
        )

    panel_meta = st.session_state.get("breakout_scan_meta") or cached_meta
    panel_results = st.session_state.get("breakout_results", cached_df)
    if panel_meta and panel_results is not None:
        _render_last_scan_panel(panel_meta, panel_results)

    if "breakout_results" in st.session_state:
        results = st.session_state["breakout_results"]
        meta = st.session_state.get("breakout_scan_meta", {})
        n_sym = meta.get("symbols_scanned", meta.get("symbols", len(symbols)))

        cached_tfs_raw = meta.get("timeframes") or results["timeframe"].unique().tolist()
        if isinstance(cached_tfs_raw, str):
            cached_tfs = sort_timeframes([t.strip() for t in cached_tfs_raw.split(",") if t.strip()])
        else:
            cached_tfs = sort_timeframes(cached_tfs_raw)
        display_tfs = [t for t in cached_tfs if t in TIMEFRAMES]
        if selected_tfs and set(selected_tfs) != set(cached_tfs):
            st.info("Sidebar filters changed. Results below are from the last saved scan until you force refresh.")

        if results.empty:
            st.warning(f"No breakouts found across {n_sym} symbols on selected filters.")
        else:
            bull = int((results["direction"] == "bullish").sum())
            bear = int((results["direction"] == "bearish").sum())
            scanned_label = meta.get("scanned_at_display") or format_scanned_at(
                meta.get("scanned_at") or meta.get("saved_at"),
                short=True,
            )
            _render_summary_metrics(
                breakouts=len(results),
                bullish=bull,
                bearish=bear,
                symbols_scanned=n_sym,
                scanned_label=scanned_label,
            )
            if meta.get("mode"):
                st.caption(f"Mode: **{meta['mode']}** · cached results — use **Force Refresh Scan** to update")

            tf_tabs = st.tabs(["All"] + [TIMEFRAMES[t].label for t in display_tfs])

            def _show(df: pd.DataFrame, key: str) -> None:
                view = st.radio(
                    "View",
                    ["Cards", "Table"],
                    horizontal=True,
                    key=f"view_{key}",
                    label_visibility="collapsed",
                )

                if view == "Cards":
                    render_breakout_cards(df)
                else:
                    styled = _style_results(df)
                    st.dataframe(styled, use_container_width=True, hide_index=True, key=f"df_{key}")

                if not df.empty:
                    pick = st.selectbox(
                        "Chart symbol",
                        df["symbol"].tolist(),
                        key=f"chart_pick_{key}",
                    )
                    row = df[df["symbol"] == pick].iloc[0]
                    st.plotly_chart(
                        _chart(pick, row["timeframe"], float(row["level"]), plan=_row_plan(row)),
                        use_container_width=True,
                        key=f"plotly_{key}_{pick}",
                    )
                    st.caption(
                        "Chart levels — 🟡 Entry · 🟢 TP1/TP2/TP3 (dotted) · 🔴 SL (dashed): "
                        "rule-based template, not advice."
                    )
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False),
                        file_name=f"breakouts_{key}.csv",
                        mime="text/csv",
                        key=f"dl_{key}",
                    )
                    st.caption("⚖️ Exported data is for research use only — not a recommendation to trade.")

            with tf_tabs[0]:
                _show(results, "all")
            for i, tf in enumerate(display_tfs, start=1):
                with tf_tabs[i]:
                    _show(results[results["timeframe"] == tf], tf.lower())

    elif not cached_scan_available():
        st.info("No cached scan yet. Configure settings and click **Force Refresh Scan**.")


ensure_dirs()

logo_src = _logo_data_uri()
logo_img = f'<img src="{logo_src}" alt="TAG Trade And Grow logo">' if logo_src else ""
st.markdown(
    f"""
<div class="tag-hero">
  {logo_img}
  <div>
    <h1>TAG Breakout Scanner</h1>
    <p>Trade And Grow scanner for Donchian breakouts, volume confirmation and Virgin CPR context across 1H, 1D and 1W timeframes.</p>
    <p><a href="{PUBLISHED_URL}" target="_blank" rel="noopener">tag-breakoutscanner.streamlit.app</a></p>
    <span class="tag-build">TAG black/gold build - 2026-07-06</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

_render_disclaimer_banner()

universe = load_universe_symbols()

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=96)
    st.markdown("### TAG")
    st.caption("Trade And Grow")
    st.divider()
    st.header("Universe")
    universe_choice = st.selectbox(
        "Symbol universe",
        list(UNIVERSE_CHOICES),
        index=list(UNIVERSE_CHOICES).index(UNIVERSE_NIFTY500),
        help="Applies to both Breakout and CPR scanners.",
    )
    max_symbols = st.slider(
        "Max symbols (NIFTY 500 only)",
        10,
        len(universe),
        len(universe),
        10,
        disabled=universe_choice != UNIVERSE_NIFTY500,
        help="Evenly sample across NIFTY 500 when less than full index.",
    )
    if universe_choice == UNIVERSE_FNO:
        if "fno_symbols" not in st.session_state:
            st.session_state.fno_symbols = fno_symbol_set()
        fno_count = len(st.session_state.fno_symbols)
        if st.button("Refresh F&O list from NSE", use_container_width=True):
            st.session_state.fno_symbols = fno_symbol_set(refresh=True)
            st.rerun()
        st.caption(f"**{fno_count}** F&O equity symbols (indices excluded).")
    elif universe_choice == UNIVERSE_NIFTY500 and max_symbols < len(universe):
        st.caption(
            f"Scanning **{max_symbols}** of **{len(universe)}** NIFTY 500 symbols "
            "(evenly spaced across the index)."
        )

    scan_symbols, universe_sample, universe_total = resolve_universe_symbols(
        universe_choice,
        universe,
        max_symbols=max_symbols if universe_choice == UNIVERSE_NIFTY500 else None,
    )
    st.metric("Symbols to scan", len(scan_symbols))

    st.divider()
    st.header("Breakout Scan Settings")
    breakout_mode_label = st.selectbox(
        "Breakout mode",
        ["Standard", "Strict (ATR)"],
        index=0,
        help=(
            "Standard: Donchian + volume + strong close. "
            "Strict: adds true range > ATR multiplier × ATR(14); default 1.5× volume on 1D."
        ),
    )
    breakout_mode = "strict" if breakout_mode_label == "Strict (ATR)" else "standard"
    selected_tfs = st.multiselect(
        "Timeframes",
        options=list(TIMEFRAME_ORDER),
        default=["1H", "1D", "1W"],
        format_func=lambda k: TIMEFRAMES[k].label,
    )
    selected_tfs = sort_timeframes(selected_tfs)
    direction = st.selectbox("Direction", ["Both", "Bullish", "Bearish"], index=0)
    vol_default = STRICT_VOL_MULT if breakout_mode == "strict" else 1.25
    vol_mult = st.slider("Min volume ratio", 1.0, 3.0, vol_default, 0.05)
    lookback = st.slider("Donchian lookback (bars)", 5, 60, 20, 1)
    atr_mult = st.slider(
        "Min TR / ATR(14) ratio",
        0.8,
        2.0,
        STRICT_ATR_MULT,
        0.05,
        disabled=breakout_mode != "strict",
        help="Breakout bar true range must exceed this multiple of 14-bar ATR.",
    )
    only_52w = st.checkbox("52-week high breakouts only (1D/1W)", value=False)
    use_cache = st.checkbox("Use price cache", value=True)
    if breakout_mode == "strict":
        st.caption(
            "Strict: close > prior N-bar high/low · volume > threshold × 20-bar avg · "
            "true range > ATR mult × ATR(14) · strong close."
        )
    else:
        st.caption(
            "Standard: close > prior N-bar high/low + volume surge + strong close. "
            "Weekly bars from daily data (Fri close)."
        )

    _render_sidebar_roadmap()

tab_breakout, tab_cpr, tab_fno, tab_review = st.tabs(
    ["Breakout Scanner", "CPR Scanner", "🎯 F&O Plan", "📅 Daily Review"]
)

with tab_breakout:
    render_breakout_tab(
        scan_symbols,
        universe_choice=universe_choice,
        universe_total=universe_total,
        universe_sample=universe_sample,
    )

with tab_cpr:
    render_cpr_tab(
        scan_symbols,
        universe_choice=universe_choice,
        universe_total=universe_total,
        universe_sample=universe_sample,
    )

with tab_fno:
    render_fno_tab()

with tab_review:
    render_daily_review_tab()

_render_disclaimer_footer()


