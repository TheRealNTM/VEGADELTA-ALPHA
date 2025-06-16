"""
term_structure.py  ·  Live USD risk-free curve (1 d → 2.5 y) from Yahoo Finance
"""
from typing import Callable, Tuple, Dict, List
import yfinance as yf, numpy as np, math, time

# ── 1.  Map each tenor to one or more Yahoo tickers that carry a usable number
_TENORS: List[Tuple[str, int, List[str]]] = [
    ("ON",    1,   ["ZQ=F"]),                     # Fed-funds futures (30-day)
    ("1W",    7,   ["ZQ=F"]),
    ("1M",   30,   ["ZQ=F"]),
    ("3M",   90,   ["^IRX"]),                     # 13-week bill yield %
    ("6M",  180,   ["^IRX"]),                     # proxy with 3-month bill
    ("1Y",  365,   ["2YY=F", "^IRX"]),            # 2-year yield futures %
    ("2Y",  730,   ["2YY=F"]),
    ("2.5Y",912,   ["2YY=F"]),
]

# ── 2.  Convert various quote conventions → pure decimal yield
# term_structure.py  ── replace the old _to_decimal function
def _to_decimal(x: float) -> float:
    """
    Convert raw Yahoo numbers into a pure decimal yield:
      • x  > 100   → basis-points×100  (e.g. 399.90  → 0.03999)
      • 70 < x≤100 → futures *price*  (e.g. 95.71   → 0.0429)
      • 1  < x≤70  → percent          (e.g. 4.2320  → 0.04232)
      • else       → already decimal
    """
    if x > 100:                     # bp×100
        return x / 10_000.0
    elif x > 70:                    # Fed-funds or SOFR futures price
        return (100.0 - x) / 100.0
    elif x > 1:                     # quoted in percent
        return x / 100.0
    return x                        # already decimal


# ── 3.  Pull latest close, trying each fallback ticker
def _pull_latest(tickers: List[str], retries=3) -> float:
    for tkr in tickers:
        for _ in range(retries):
            try:
                hist = yf.Ticker(tkr).history(period="7d")["Close"].dropna()
                if not hist.empty:
                    return _to_decimal(float(hist.iloc[-1]))
            except Exception:
                time.sleep(0.3)
    return math.nan             # will print “n/a” in the sidebar

def get_live_rates() -> Dict[str, float]:
    return {lbl: _pull_latest(tkrs) for lbl, _, tkrs in _TENORS}

# ── 4.  Build linear (T, r) interpolator from the finite quotes
def _interp(days: np.ndarray, quotes: np.ndarray) -> Callable[[float], float]:
    yrs = days / 365.0
    if 0 not in days:
        days   = np.insert(days,   0, 0)
        quotes = np.insert(quotes, 0, quotes[0])
        yrs    = days / 365.0
    return lambda T: float(np.interp(T, yrs, quotes))

def default_rate_curve():
    live = get_live_rates()
    days, quotes = zip(*[(d, live[lbl]) for lbl, d, _ in _TENORS if math.isfinite(live[lbl])])
    return _interp(np.array(days), np.array(quotes)), live

# ── 5.  Dividend curve (still flat) and handy one-liner
def flat_dividend_curve(level: float):
    return lambda T, _=level: _

def default_curves(dividend_yield=0.0):
    r_curve, live = default_rate_curve()
    return r_curve, flat_dividend_curve(dividend_yield), live
