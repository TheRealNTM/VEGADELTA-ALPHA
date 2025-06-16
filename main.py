# main.py  — TOP OF FILE
import yfinance as yf
import pandas as pd
import numpy as np
import functions as f
import streamlit as st
from scipy.interpolate import griddata
import plotly.graph_objects as go

session = None          # exported so app.py can use m.session

# ----------------------------------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd

# ── 1. stock history & spot price ───────────────────────────────────
@st.cache_resource(                   # ← CHANGED from cache_data
    show_spinner="🔄  Loading stock & price history…",
    ttl=300                            # refresh after 5 min
)
def get_stock_data(ticker_symbol, period="max"):
    global session
    stock       = yf.Ticker(ticker_symbol, session=session)
    spot_prices = stock.history(period=period)["Close"].to_frame()

    spot_data  = stock.history(period="1d")["Close"]
    spot_price = spot_data.iloc[-1] if not spot_data.empty else spot_prices.iloc[-1, 0]

    if pd.isna(spot_price):
        raise ValueError(
            f"No data available for ticker {ticker_symbol}. "
            "Please check the symbol or try again later."
        )

    return stock, spot_prices, spot_price


# ── 2. option chains for all expirations (unchanged) ───────────────
@st.cache_data(
    show_spinner="🔄  Fetching option chains…",
    ttl=900,
    hash_funcs={yf.Ticker: lambda x: x.ticker}
)
def get_options_data(stock):
    """Return concatenated option chains for all available expirations."""
    expiration_dates = stock.options
    calls_dict = {date: stock.option_chain(date).calls for date in expiration_dates}
    puts_dict = {date: stock.option_chain(date).puts  for date in expiration_dates}

    for date, df in calls_dict.items():
        df["expiration"] = date
    for date, df in puts_dict.items():
        df["expiration"] = date

    calls_all = pd.concat(calls_dict.values())
    puts_all  = pd.concat(puts_dict.values())
    return calls_all, puts_all, expiration_dates

# ----------------------------------------------------------------------


def filter_calls_data(calls_data, spot_price, min_strike_price, max_strike_price):
    filtered_calls_data = calls_data[(calls_data['strike'] >= min_strike_price) & (calls_data['strike'] <= max_strike_price)]
    filtered_calls_data = filtered_calls_data[filtered_calls_data['expiration'].apply(f.calculate_time_to_expiration) >= 0.07]

    return filtered_calls_data.reset_index(drop=True)


@st.cache_data(
    show_spinner="🔄  Computing implied volatilities…",
    ttl=900,                                        # 15-minute cache
    hash_funcs={pd.DataFrame: lambda df: hash(tuple(df.index))}
)
def calculate_implied_volatility(filtered_calls_data: pd.DataFrame,
                                 spot_price: float,
                                 _r_curve, _q_curve) -> pd.DataFrame:
    """
    Fast IV builder – r/q curves are prefixed with “_” so Streamlit ignores
    them when hashing and doesn’t raise UnhashableParamError.
    """
    records = []

    for exp, df in filtered_calls_data.groupby("expiration"):
        T = f.calculate_time_to_expiration(exp)
        if T <= 0:
            continue

        r_local = _r_curve(T)
        q_local = _q_curve(T)

        strikes = df["strike"].to_numpy()
        prices  = df["lastPrice"].to_numpy()
        symbols = df["contractSymbol"].to_numpy()

        ivs = np.vectorize(
            lambda X, P: f.Call_IV(S=spot_price, X=X, r=r_local,
                                   T=T, Call_Price=P, q=q_local),
            otypes=[float]
        )(strikes, prices)

        records.extend(zip(symbols, strikes,
                           np.full_like(strikes, T, dtype=float),
                           ivs,
                           np.full_like(strikes, r_local, dtype=float),
                           np.full_like(strikes, q_local, dtype=float)))

    return pd.DataFrame.from_records(
        records,
        columns=["ContractSymbol", "StrikePrice", "TimeToExpiry",
                 "ImpliedVolatility", "RiskFreeRate", "DividendYield"]
    )



def get_plot_data(filtered_df):
    X = filtered_df['TimeToExpiry'].values  
    Y = filtered_df['StrikePrice'].values
    Z = filtered_df['ImpliedVolatility'].values * 100

    return X, Y, Z


# Optional: a function to create the plot if needed.
def plot_implied_volatility(X, Y, Z):
    # Define grid for interpolation
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate Z values over the grid
    zi = griddata((X, Y), Z, (xi, yi), method='linear')

    # Create the 3D plot using Plotly
    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale='Viridis')])
    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Time to Expiration (years)',
            yaxis_title='Strike Price ($)',
            zaxis_title='Implied Volatility (%)'
        )
    )

    return fig
