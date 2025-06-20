
from scipy.optimize import newton
from datetime import datetime, timezone
import numpy as np
from scipy.stats import norm
import scipy as sq


def Call_BS_Value(S, X, r, T, v, q):
    # Calculates the value of a call option (Black-Scholes formula for call options with dividends)
    # S is the share price at time T
    # X is the strike price
    # r is the risk-free interest rate
    # T is the time to maturity in years (days/365)
    # v is the volatility
    # q is the dividend yield
    d_1 = (np.log(S / X) + (r - q + v ** 2 * 0.5) * T) / (v * np.sqrt(T))
    d_2 = d_1 - v * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d_1) - X * np.exp(-r * T) * norm.cdf(d_2)


def Call_IV_Obj_Function(S, X, r, T, v, q, Call_Price):
    # Objective function which sets market and model prices equal to zero (Function needed for Call_IV)
    # The parameters are explained in the Call_BS_Value function
    return Call_Price - Call_BS_Value(S, X, r, T, v, q)


def Call_IV(S, X, r, T, Call_Price, q, a=-0.000001, b=5, xtol=0.000001):
    # Calculates the implied volatility for a call option with Brent's method
    # The first four parameters are explained in the Call_BS_Value function
    # Call_Price is the price of the call option
    # q is the dividend yield
    # Last three variables are needed for Brent's method
    _S, _X, _r, _t, _Call_Price, _q = S, X, r, T, Call_Price, q

    def fcn(v):
        return Call_IV_Obj_Function(_S, _X, _r, _t, v, _q, _Call_Price)

    try:
        result = sq.optimize.brentq(fcn, a=a, b=b, xtol=xtol)
        return np.nan if result <= xtol else result
    except ValueError:
        return np.nan


def Put_BS_Value(S, X, r, T, v, q):
    # Calculates the value of a put option (Black-Scholes formula for put options with dividends)
    # The parameters are explained in the Call_BS_Value function
    d_1 = (np.log(S / X) + (r - q + v ** 2 * 0.5) * T) / (v * np.sqrt(T))
    d_2 = d_1 - v * np.sqrt(T)
    return X * np.exp(-r * T) * norm.cdf(-d_2) - S * np.exp(-q * T) * norm.cdf(-d_1)


def Put_IV_Obj_Function(S, X, r, T, v, q, Put_Price):
    # Objective function which sets market and model prices equal to zero (Function needed for Put_IV)
    # The parameters are explained in the Call_BS_Value function
    return Put_Price - Put_BS_Value(S, X, r, T, v, q)


def Put_IV(S, X, r, T, Put_Price, q, a=0.000001, b=5, xtol=0.000001):
    # Calculates the implied volatility for a put option with Brent's method
    # The first four parameters are explained in the Call_BS_Value function
    # Put_Price is the price of the put option
    # q is the dividend yield
    # Last three variables are needed for Brent's method
    _S, _X, _r, _t, _Put_Price, _q = S, X, r, T, Put_Price, q

    def fcn(v):
        return Put_IV_Obj_Function(_S, _X, _r, _t, v, _q, _Put_Price)

    try:
        result = sq.optimize.brentq(fcn, a=a, b=b, xtol=xtol)
        return np.nan if result <= xtol else result
    except ValueError:
        return np.nan


def Calculate_IV_Call_Put(S, X, r, T, Option_Price, Put_or_Call, q):
    # This is a general function witch summarizes Call_IV and Put_IV (delivers the same results)
    # Can be used for a Lambda function within Pandas
    # The first four parameters are explained in the Call_BS_Value function
    # Put_or_Call:
    # 'C' returns the implied volatility of a call
    # 'P' returns the implied volatility of a put
    # Option_Price is the price of the option.
    # q is the dividend yield

    if Put_or_Call == 'C':
        return Call_IV(S, X, r, T, Option_Price, q)
    if Put_or_Call == 'P':
        return Put_IV(S, X, r, T, Option_Price, q)
    else:
        return 'Neither call or put'


# functions.py
from datetime import datetime, timezone   # <- add timezone to the import list

def calculate_time_to_expiration(expiration_date_str: str) -> float:
    """
    Time to expiry in ACT/365F years, using UTC-aware timestamps.
    """
    # Expiry date at 00:00 UTC (midnight). Adjust if you prefer 23:59 UTC.
    expiration_date = datetime.strptime(expiration_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Current time in UTC
    now_utc = datetime.now(timezone.utc)

    # Actual seconds between now and expiry
    seconds_to_expiry = (expiration_date - now_utc).total_seconds()

    # Convert to ACT/365F years
    return seconds_to_expiry / (365.0 * 24 * 60 * 60)


