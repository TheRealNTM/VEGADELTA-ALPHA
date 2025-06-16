import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Black‑Scholes‑Merton helper
# ---------------------------------------------------------------------------

def _d1_d2(S, K, r, q, sigma, T):
    """
    Vectorised Black–Scholes d1 and d2.

    Accepts scalars or NumPy arrays for each argument and broadcasts them.
    """
    S      = np.asarray(S, dtype=float)
    K      = np.asarray(K, dtype=float)
    sigma  = np.asarray(sigma, dtype=float)
    T      = np.asarray(T, dtype=float)

    # Avoid divide-by-zero / log(0) / sqrt(0)
    eps = 1e-12
    sigma = np.where(sigma <= 0.0, eps, sigma)
    T     = np.where(T     <= 0.0, eps, T)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

# ---------------------------------------------------------------------------
# First‑order greeks
# ---------------------------------------------------------------------------

def delta(S, K, r, q, sigma, T, option="call"):
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    if option.lower().startswith("c"):
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)

def gamma(S, K, r, q, sigma, T):
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, r, q, sigma, T):
    """Black–Scholes vega (vectorised, same shape as inputs)."""
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    return np.exp(-q * T) * S * norm.pdf(d1) * np.sqrt(T)

def theta(S, K, r, q, sigma, T, option="call"):
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    common = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option.lower().startswith("c"):
        return common - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    return common + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)

def rho(S, K, r, q, sigma, T, option="call"):
    _, d2 = _d1_d2(S, K, r, q, sigma, T)
    if option.lower().startswith("c"):
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# ---------------------------------------------------------------------------
# Second‑order / cross greeks
# ---------------------------------------------------------------------------

def vanna(S, K, r, q, sigma, T):
    """dVega/dS (or dDelta/dVol)."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    return np.exp(-q * T) * norm.pdf(d1) * d2 / sigma

def vomma(S, K, r, q, sigma, T):
    """Volga: dVega/dVol."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    return vega(S, K, r, q, sigma, T) * d1 * d2 / sigma

def charm(S, K, r, q, sigma, T, option="call"):
    """dDelta/dT (in calendar time)."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    part = np.exp(-q * T) * norm.pdf(d1) * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    return -part if option.lower().startswith("c") else part

def speed(S, K, r, q, sigma, T):
    """dGamma/dS."""
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    g = gamma(S, K, r, q, sigma, T)
    return -g / S * (d1 / (sigma * np.sqrt(T)) + 1)

# Mapping for easy dispatch --------------------------------------------------

GreekFuncs = {
    "Delta": delta,
    "Gamma": gamma,
    "Theta": theta,
    "Vega": vega,
    "Rho": rho,
    "Vanna": vanna,
    "Vomma": vomma,
    "Charm": charm,
    "Speed": speed,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

import pandas as pd
import inspect

def compute_greek(df, greek, spot_price, r, q):
    import pandas as pd
    import inspect

    if greek not in GreekFuncs:
        raise ValueError(f"Greek '{greek}' not recognized.")

    fn = GreekFuncs[greek]
    sig = inspect.signature(fn)
    needs_option = 'option' in sig.parameters

    results = []
    for _, row in df.iterrows():
        K        = row["StrikePrice"]
        T        = row["TimeToExpiry"]
        sigma    = row["ImpliedVolatility"]
        opt_type = row.get("OptionType","Call").lower()
        r_local = row.get("RiskFreeRate", r)
        q_local = row.get("DividendYield", q)

        # compute raw Greek
        if needs_option:
            raw = fn(spot_price, K, r, q, sigma, T, option=opt_type)
        else:
            raw = fn(spot_price, K, r, q, sigma, T)

        # scale for display:
        if greek in ("Delta","Gamma"):
            display = raw
        elif greek == "Speed":
            # thousandth‐of‐a‐dollar units
            display = raw * 0.001
        else:
            # hundredth‐of‐a‐dollar units
            display = raw * 0.01

        results.append(display)

    out = df.copy()
    out["GreekValue"] = results
    return out

