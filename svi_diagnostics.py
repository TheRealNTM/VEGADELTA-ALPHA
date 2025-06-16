import numpy as np
import pandas as pd

# small increment for finite differences
_DK = 1e-4          # log-moneyness step
_DT = 1e-4          # year-fraction step

def _total_variance(iv, T):
    return (iv ** 2) * T

def check_svi_slice(df_slice, atol_slope=1e-6, atol_convex=1e-6):
    """
    Returns slope/convexity tests for one maturity slice.
    A *negative* dw/dk is perfectly acceptable (gives σ′≤0).
    We only fail the slope test if dw/dk flips **positive** anywhere.
    """
    k  = df_slice["LogMoneyness"].values
    iv = df_slice["ImpliedVolatility"].values
    T  = float(df_slice["TimeToExpiry"].iloc[0])

    w = (iv ** 2) * T
    dw_dk   = np.gradient(w, k)
    slope_ok = np.all(dw_dk <= atol_slope)        #  now allow ≤0

    d2w_dk2  = np.gradient(dw_dk, k)
    convex_ok = np.all(d2w_dk2 >= -atol_convex)

    return {
        "slope_OK":       slope_ok,
        "convex_OK":      convex_ok,
        "max_pos_slope":  dw_dk.max(),
        "min_convexity":  d2w_dk2.min()
    }

def enforce_calendar_consistency(svi_df, k_grid=None):
    """
    Shifts each maturity's total variance up minimally so that
    w(k,T2) ≥ w(k,T1) ∀k when T2>T1. Works in-place and returns df.
    """
    if k_grid is None:
        k_grid = np.linspace(svi_df["LogMoneyness"].min(),
                             svi_df["LogMoneyness"].max(), 301)
    maturities = sorted(svi_df["TimeToExpiry"].unique())
    prev_w = None
    bumps  = {}
    for T in maturities:
        sl = svi_df[svi_df["TimeToExpiry"] == T]
        w   = np.interp(k_grid, sl["LogMoneyness"], (sl["ImpliedVolatility"]**2)*T)
        if prev_w is not None:
            diff = prev_w - w
            bump = max(0.0, diff.max())           # minimal upward shift
            bumps[T] = bump
            if bump > 0:
                # Apply bump
                mask = svi_df["TimeToExpiry"] == T
                svi_df.loc[mask, "ImpliedVolatility"] = np.sqrt(
                    ((svi_df.loc[mask, "ImpliedVolatility"]**2) * T + bump) / T
                )
                w += bump
        prev_w = w
    return svi_df, bumps

def check_calendar(df_all, atol=1e-6):
    """
    Checks w(k,T1) ≤ w(k,T2) for T2>T1 on interpolated k grid.
    """
    maturities = sorted(df_all["TimeToExpiry"].unique())
    # common k grid
    k_union = np.linspace(df_all["LogMoneyness"].min(),
                          df_all["LogMoneyness"].max(), 201)
    prev_w = None
    for T in maturities:
        slice_df = df_all[df_all["TimeToExpiry"] == T]
        iv_interp = np.interp(k_union, slice_df["LogMoneyness"], slice_df["ImpliedVolatility"])
        w = _total_variance(iv_interp, T)
        if prev_w is not None and np.any(w < prev_w - atol):
            return False
        prev_w = w
    return True

def diagnostics_report(svi_df, market_df):
    """
    Compares calibrated vs market vols & arbitrage tests.
    """
    res = {}
    # RMSE per slice
    errs = []
    for T in svi_df["TimeToExpiry"].unique():
        mkt = market_df[market_df["TimeToExpiry"] == T]
        mod = svi_df[svi_df["TimeToExpiry"] == T]
        iv_mod = np.interp(mkt["LogMoneyness"], mod["LogMoneyness"], mod["ImpliedVolatility"])
        errs.extend((iv_mod - mkt["ImpliedVolatility"]).values**2)
        res[f"T={T:.3f}"] = check_svi_slice(mod)
    res["RMSE_bps"] = (np.sqrt(np.mean(errs))*10000).round(1)  # bp
    res["calendar_OK"] = check_calendar(svi_df)
    return res

def pretty_print(report):
    """
    Neat console table for diagnostics_report().
    """
    lines = []
    hdr = f"{'Maturity':>8} | {'Slope':^5} | {'Convex':^6} | {'max w′':>8} | {'min w″':>8}"
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for k, v in sorted(report.items()):
        if not k.startswith('T='):
            continue
        slope = 'OK' if v['slope_OK'] else 'BAD'
        conv  = 'OK' if v['convex_OK'] else 'BAD'
        lines.append(f"{k[2:]:>8} | {slope:^5} | {conv:^6} | {v['max_pos_slope']:8.4f} | {v['min_convexity']:8.4f}")
    lines.append('-' * len(hdr))
    lines.append(f" Calendar: {'OK' if report['calendar_OK'] else 'BAD'}"
                 f"   RMSE: {report['RMSE_bps']:.1f} bp")
    print('\n'.join(lines))
