"""Indicative F&O options plan for breakout signals.

Trade levels are computed on the UNDERLYING from real scan data (scan close
and breakout level). Strike, expiry, and option LTP come live from the NSE
option-chain API when reachable; nothing is synthesised — when NSE is not
reachable the strike/expiry fall back to clearly-labelled estimates and the
option LTP is left blank.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import exp, log, sqrt
from statistics import NormalDist
from typing import Any, Optional

import requests

NSE_CHAIN_URL = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
_NSE_HOME = "https://www.nseindia.com/option-chain"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _NSE_HOME,
}

# SL sits 0.5% beyond the broken level; targets are 1R/2R/3R from entry.
SL_BUFFER = 0.005
TP_MULTIPLES = (1.0, 2.0, 3.0)
RISK_FREE_RATE = 0.065  # approx. RBI repo; short-dated options are insensitive to it
MIN_PREMIUM = 0.05  # NSE option tick floor

_NORM = NormalDist()


def _bs_price(spot: float, strike: float, t_years: float, iv: float, opt_type: str) -> float:
    """Black-Scholes European price; intrinsic value at/after expiry."""
    call = opt_type.upper() == "CE"
    if t_years <= 0 or iv <= 0:
        return max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
    d1 = (log(spot / strike) + (RISK_FREE_RATE + iv * iv / 2) * t_years) / (iv * sqrt(t_years))
    d2 = d1 - iv * sqrt(t_years)
    disc = exp(-RISK_FREE_RATE * t_years)
    if call:
        return spot * _NORM.cdf(d1) - strike * disc * _NORM.cdf(d2)
    return strike * disc * _NORM.cdf(-d2) - spot * _NORM.cdf(-d1)


def _implied_vol(price: float, spot: float, strike: float, t_years: float, opt_type: str) -> Optional[float]:
    """Back out IV from a market premium by bisection; None if unpriceable."""
    if price <= 0 or t_years <= 0:
        return None
    lo, hi = 0.01, 3.0
    if not (_bs_price(spot, strike, t_years, lo, opt_type) <= price <= _bs_price(spot, strike, t_years, hi, opt_type)):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if _bs_price(spot, strike, t_years, mid, opt_type) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _years_to_expiry(expiry_label: str, today: Optional[date] = None) -> Optional[float]:
    try:
        exp_date = datetime.strptime(expiry_label.strip(), "%d-%b-%Y").date()
    except ValueError:
        return None
    days = ((exp_date - (today or date.today())).days) or 1
    return max(days, 1) / 365.0


@dataclass
class OptionPlan:
    symbol: str
    direction: str  # bullish / bearish
    opt_type: str  # CE / PE
    strike: float
    expiry: str
    option_ltp: Optional[float]  # premium entry (market LTP)
    entry: float  # spot levels
    sl: float
    tps: tuple[float, ...]
    live: bool  # strike/expiry/LTP from NSE (True) or estimated (False)
    prem_tps: Optional[tuple[float, ...]] = None  # premium at spot TPs
    prem_sl: Optional[float] = None  # premium at spot SL
    iv: Optional[float] = None  # IV used for the premium mapping


def fetch_option_chain(symbol: str, timeout: float = 6.0) -> Optional[dict[str, Any]]:
    """Fetch the NSE equity option chain; None when unreachable/blocked."""
    try:
        with requests.Session() as s:
            s.headers.update(_HEADERS)
            s.get(_NSE_HOME, timeout=timeout)  # cookie warm-up
            resp = s.get(NSE_CHAIN_URL.format(symbol=symbol.upper()), timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict) and data.get("records", {}).get("data"):
            return data
    except Exception:
        pass
    return None


def estimate_strike_step(price: float) -> float:
    """Indicative NSE strike interval by price band (fallback only)."""
    if price < 50:
        return 1.0
    if price < 100:
        return 2.5
    if price < 250:
        return 5.0
    if price < 500:
        return 10.0
    if price < 1000:
        return 20.0
    if price < 2500:
        return 50.0
    return 100.0


def estimate_next_expiry(today: Optional[date] = None) -> date:
    """Last Tuesday of the current month, else of next month (NSE monthly
    stock-option expiry moved to Tuesdays from Sep 2025). Estimate only."""
    today = today or date.today()

    def last_tuesday(year: int, month: int) -> date:
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        return last_day - timedelta(days=(last_day.weekday() - 1) % 7)

    exp = last_tuesday(today.year, today.month)
    if exp < today:
        year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        exp = last_tuesday(year, month)
    return exp


def build_plan(
    symbol: str,
    direction: str,
    close: float,
    level: float,
    chain: Optional[dict[str, Any]] = None,
) -> OptionPlan:
    """Rule-based plan: entry = scan close, SL = break level -/+ 0.5%,
    TPs = entry +/- 1R/2R/3R. Bullish -> CE, bearish -> PE (ATM strike)."""
    bullish = str(direction).lower() == "bullish"
    opt_type = "CE" if bullish else "PE"

    entry = float(close)
    if bullish:
        sl = float(level) * (1 - SL_BUFFER)
        risk = max(entry - sl, entry * 0.001)
        tps = tuple(round(entry + m * risk, 2) for m in TP_MULTIPLES)
    else:
        sl = float(level) * (1 + SL_BUFFER)
        risk = max(sl - entry, entry * 0.001)
        tps = tuple(round(entry - m * risk, 2) for m in TP_MULTIPLES)

    strike: Optional[float] = None
    expiry: Optional[str] = None
    option_ltp: Optional[float] = None
    live = False
    chain_iv: Optional[float] = None

    if chain:
        try:
            records = chain["records"]
            expiries = records.get("expiryDates") or []
            expiry = expiries[0] if expiries else None
            strikes = sorted({float(r["strikePrice"]) for r in records["data"] if r.get("strikePrice")})
            if strikes:
                strike = min(strikes, key=lambda s: abs(s - entry))
            if expiry and strike is not None:
                for r in records["data"]:
                    if r.get("expiryDate") == expiry and float(r.get("strikePrice", -1)) == strike:
                        leg = r.get(opt_type)
                        if leg and leg.get("lastPrice") is not None:
                            option_ltp = float(leg["lastPrice"])
                            iv_raw = leg.get("impliedVolatility")
                            if iv_raw:
                                chain_iv = float(iv_raw) / 100.0
                        break
                live = True
        except Exception:
            strike = expiry = option_ltp = None
            live = False

    # Premium ladder: reprice the option at each spot target with Black-Scholes,
    # anchored to the live market LTP (no time decay assumed). Only when live.
    prem_tps: Optional[tuple[float, ...]] = None
    prem_sl: Optional[float] = None
    used_iv: Optional[float] = None
    if live and option_ltp and option_ltp > 0 and strike and expiry:
        t_years = _years_to_expiry(expiry)
        if t_years:
            used_iv = chain_iv or _implied_vol(option_ltp, entry, float(strike), t_years, opt_type)
            if used_iv:
                base = _bs_price(entry, float(strike), t_years, used_iv, opt_type)

                def _prem_at(spot: float) -> float:
                    model = _bs_price(spot, float(strike), t_years, used_iv, opt_type)
                    return round(max(option_ltp + (model - base), MIN_PREMIUM), 2)

                prem_tps = tuple(_prem_at(tp) for tp in tps)
                prem_sl = _prem_at(sl)

    if strike is None:
        step = estimate_strike_step(entry)
        strike = round(entry / step) * step
    if expiry is None:
        expiry = estimate_next_expiry().strftime("%d-%b-%Y") + " (est.)"

    return OptionPlan(
        symbol=symbol.upper(),
        direction=direction,
        opt_type=opt_type,
        strike=float(strike),
        expiry=str(expiry),
        option_ltp=option_ltp,
        entry=round(entry, 2),
        sl=round(sl, 2),
        tps=tps,
        live=live,
        prem_tps=prem_tps,
        prem_sl=prem_sl,
        iv=used_iv,
    )
