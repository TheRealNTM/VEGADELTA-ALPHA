# svi_module.py

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import greeks_module as gm

def svi_model(k, a, b, rho, m_param, sigma):
    """Stochastic Volatility Inspired (SVI) model formula.
    
    Args:
        k: Log moneyness values
        a: Vertical shift parameter
        b: Slope/curvature parameter 
        rho: Correlation/asymmetry parameter
        m_param: Horizontal shift parameter
        sigma: Smoothness parameter
    
    Returns:
        Implied volatility values according to SVI model
    """
    return a + b * (rho * (k - m_param) + np.sqrt((k - m_param)**2 + sigma**2))

def svi_objective(params, k, v_pct, w=None,
                  lambda_reg=0.0, slope_penalty=1e4):
    """
    Weighted least-squares + slope-penalty objective.

    v_pct      : implied vols in PERCENT (as before)
    w          : weights (None ⇒ equal)
    slope_penalty : penalty coeff applied to any positive w′(k)
    """
    a, b, rho, m, sigma = params
    # parameter domain guard
    if b < 0 or sigma <= 0 or not (-1 < rho < 0):
        return 1e10

    # model vols (still %)
    model_v = svi_model(k, a, b, rho, m, sigma)

    if w is None:
        w = np.ones_like(k)

    # --- core WLS error -------------------------------------------------
    err = np.sum(w * (model_v - v_pct) ** 2)

    # --- positive-slope penalty ----------------------------------------
    # total variance  w(k) = model_v^2  × 1  (T cancels out in sign test)
    # analytic derivative: w′(k) = b * ( rho + (k-m)/sqrt((k-m)^2+sigma^2) )
    z = (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2)
    w_prime = b * (rho + z)
    pos_slope = np.maximum(w_prime, 0.0)
    slope_cost = slope_penalty * np.sum(pos_slope ** 2)

    # --- L2 regularisation (keeps σ & b moderate) -----------------------
    reg = lambda_reg * (b ** 2 + sigma ** 2)

    return err + slope_cost + reg


def calibrate_svi(k, v, w=None, lambda_reg=0.0,
                  multi_start_count=7,               # a few more starts
                  slope_penalty_start=1e6,           # start higher
                  tol=1e-10,                        # numerical tolerance
                  verbose=True):
    """
    Adaptive SVI calibration that **guarantees** w'(k) ≤ 0 (within tol).

    Returns best parameter vector [a, b, rho, m, sigma].
    """
    def max_pos_slope(p):
        a, b, rho, m, sigma = p
        z = (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2)
        return np.max(b * (rho + z))                 # max of w'(k)

    penalty = slope_penalty_start
    round_no = 0
    best_params = None
    best_fun    = np.inf

    while True:                                      # loop until slice is clean
        round_no += 1
        base_a = np.min(v)
        current_best, current_fun = None, np.inf

        for _ in range(multi_start_count):
            x0 = [
                base_a + np.random.uniform(-0.5, 0.5),
                0.1 + np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.99, -0.01),      # ρ < 0
                np.random.uniform(-0.1, 0.1),
                0.1 + np.random.uniform(0, 0.1),
            ]
            bounds = [
                (-np.inf, np.inf), (0.0, np.inf),
                (-0.99, -1e-3), (-np.inf, np.inf), (1e-3, np.inf)
            ]

            try:
                res = minimize(
                    lambda p: svi_objective(p, k, v, w,
                                            lambda_reg,
                                            slope_penalty=penalty),
                    x0, bounds=bounds, method="L-BFGS-B"
                )
                if res.fun < current_fun:
                    current_best, current_fun = res.x, res.fun
            except Exception:
                continue

        # keep global best for diagnostics / fallback
        if current_fun < best_fun:
            best_params, best_fun = current_best, current_fun

        upslope = max_pos_slope(current_best)
        if verbose:
            print(f"   round {round_no:2d}  penalty={penalty:.1e} "
                  f"max w′={upslope:+.3e}")

        if upslope <= tol:            # ✅ slope non-positive – we’re done 
            return current_best

        # else escalate and try again
        penalty *= 1000               # MUCH steeper escalation





def generate_svi_calibrated_data(
        unique_maturities,
        imp_vol_data,
        spot_price,
        risk_free_rate,
        dividend_yield,
        reg_weight=0.0,
        multi_start_count=5,
        option_data_type="Both"):
    """
    Calibrate an SVI slice for each maturity and return a smooth grid.

    Args:
        unique_maturities (iterable): distinct time-to-expiry values
        imp_vol_data (DataFrame): must contain TimeToExpiry, LogMoneyness,
                                  StrikePrice, ImpliedVolatility columns
        spot_price (float): current underlying price
        risk_free_rate (float): continuously-compounded risk-free rate
        dividend_yield (float): continuously-compounded dividend yield
        reg_weight (float): L2 regularisation weight for SVI objective
        multi_start_count (int): random starts for optimiser
        option_data_type (str): label for downstream plotting (“Calls Only” …)

    Returns:
        pd.DataFrame with calibrated (k,σ) points for each maturity
    """
    svi_rows = []

    for tau in unique_maturities:
        slice_df = imp_vol_data[np.isclose(imp_vol_data["TimeToExpiry"], tau)]
        if len(slice_df) < 5:
            continue

        k = slice_df["LogMoneyness"].values
        v_pct = slice_df["ImpliedVolatility"].values * 100.0  # % IV expected
        strikes = slice_df["StrikePrice"].values

        # -------- Black-Scholes vegas → weights --------------------------
        vegas = gm.vega(spot_price, strikes, risk_free_rate,
                        dividend_yield, slice_df["ImpliedVolatility"].values, tau)
        # avoid divide-by-zero / NaN
        vegas = np.where(vegas <= 0.0, 1e-12, vegas)
        w = vegas / vegas.max()       # normalise for numeric stability

        params = calibrate_svi(k, v_pct, w=w,
                               lambda_reg=reg_weight,
                               multi_start_count=multi_start_count)
        if params is None:
            continue

        a, b, rho, m, sigma = params
        k_smooth = np.linspace(k.min() - 0.1, k.max() + 0.1, 100)
        v_smooth = svi_model(k_smooth, a, b, rho, m, sigma)   # still in %
        strikes_smooth = spot_price * np.exp(k_smooth)

        for k_i, v_i, K_i in zip(k_smooth, v_smooth, strikes_smooth):
            svi_rows.append({
                "TimeToExpiry": tau,
                "LogMoneyness": k_i,
                "StrikePrice": K_i,
                "Moneyness": np.exp(k_i),
                "ImpliedVolatility": v_i / 100.0,     # back to decimals
                "OptionType": option_data_type.replace(" Only", "")
            })

    if svi_rows:
        return pd.DataFrame(svi_rows)
    # empty fallback with complete column list
    return pd.DataFrame(columns=[
        "TimeToExpiry", "LogMoneyness", "StrikePrice",
        "Moneyness", "ImpliedVolatility", "OptionType"
    ])

def filter_outliers(imp_vol_data):
    """Filter out outliers from the implied volatility data.
    
    Args:
        imp_vol_data: DataFrame with implied volatility data
    
    Returns:
        DataFrame with outliers removed
    """
    # Filter by volume if available
    if "volume" in imp_vol_data.columns:
        imp_vol_data = imp_vol_data[imp_vol_data["volume"] > 10]
        
    # Remove outliers based on IQR method
    Q1 = imp_vol_data["ImpliedVolatility"].quantile(0.25)
    Q3 = imp_vol_data["ImpliedVolatility"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR
    upper_bound = Q3 + 2.0 * IQR
    
    return imp_vol_data[(imp_vol_data["ImpliedVolatility"] >= lower_bound) &
                        (imp_vol_data["ImpliedVolatility"] <= upper_bound)]
