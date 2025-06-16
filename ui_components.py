# ui_components.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline
from scipy.ndimage     import gaussian_filter


def setup_page_config():
    st.set_page_config(page_title="VegaDelta Options Modelling", layout="wide")
    
    st.markdown(
        """
        <style>
        /* 1) Modern font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        * {
            font-family: 'Poppins', sans-serif !important;
        }
        h1,h2,h3,h4,h5,h6 {
            font-weight: 600 !important;
        }

        /* 2) Page background (light grey) */
        body, .stApp, .block-container {
            background-color: #f0f2f6 !important;
        }

        /* 3) Light blue‑grey sidebar */
        section[data-testid="stSidebar"] {
            background-color: #e3eef7 !important;
        }
        section[data-testid="stSidebar"] div:first-child {
            background-color: #e3eef7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )




def display_stock_metrics(ticker, stock_info, options_quality):
    """Display key stock metrics in columns."""
    current_price    = stock_info.get('regularMarketPrice', 0)
    previous_close   = stock_info.get('previousClose',    0)
    price_change     = current_price - previous_close
    price_change_pct = (price_change / previous_close) * 100 if previous_close else 0
    market_cap       = stock_info.get('marketCap', 0) / 1e9

    # Four columns: Price | Change | Data Quality | Market Cap
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=f"{ticker} Price",
                  value=f"${current_price:.2f}")
    with col2:
        st.metric(label="Change",
                  value=f"${price_change:.2f}",
                  delta=f"{price_change_pct:.2f}%")
    with col3:
        st.metric(label="Options Data Quality",
                  value=options_quality)
    with col4:
        st.markdown(f"**Market Cap:** ${market_cap:.2f} B")


# ui_components.py
# --- replace the old create_sidebar_inputs(...) completely -------------
def create_sidebar_inputs(live_rates: dict, dividend_yield: float):
    """
    Sidebar: mini yield-curve chart + model controls.
    """
    import streamlit as st, pandas as pd, numpy as np, altair as alt

    # ------------ Yield-curve chart ------------------------------------
    st.sidebar.header("Live USD Yield Curve")

    order = ["ON", "1W", "1M", "3M", "6M", "1Y", "2Y", "2.5Y"]
    rows = [
        {"Tenor": lbl, "Yield": live_rates.get(lbl, np.nan) * 100}
        for lbl in order
        if np.isfinite(live_rates.get(lbl, np.nan))
    ]

    if rows:
        df = pd.DataFrame(rows)
        min_y, max_y = df["Yield"].min(), df["Yield"].max()
        pad = max(0.05 * (max_y - min_y), 0.05)  # ≥ 0.05 %

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Tenor:N", sort=order, title=""),
                y=alt.Y(
                    "Yield:Q",
                    title="%",
                    scale=alt.Scale(domain=[min_y - pad, max_y + pad], nice=False)
                ),
                tooltip=["Tenor:N", alt.Tooltip("Yield:Q", format=".2f")]
            )
            .properties(height=140)
        )
        st.sidebar.altair_chart(chart, use_container_width=True)
    else:
        st.sidebar.info("No rate data available.")

    # ------------ Dividend + user controls ----------------------------
    st.sidebar.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")

    option_type = st.sidebar.selectbox(
        "Y-axis basis",
        ["Strike Price", "Moneyness"],
        key="sidebar_yaxis_basis",
    )
    option_data_type = st.sidebar.radio(
        "Option Data",
        ["Calls Only", "Puts Only", "Both Calls & Puts"],
        index=0,
        key="sidebar_option_data",
    )
    return option_type, option_data_type






def create_advanced_settings(imp_vol_data, spot_price):
    """Create and return advanced settings for SVI calibration,
       but only when the user clicks “Apply”."""
    form = st.sidebar.form(key="svi_settings_form")
    form.header("Advanced Settings for SVI Calibration")
    
    # Strike price range inputs
    min_pct = form.number_input("Min Strike (% of Spot)", 20, 200, 75, key="min_pct")
    max_pct = form.number_input("Max Strike (% of Spot)", 20, 200, 125, key="max_pct")
    
    # Time to expiration range
    t_min, t_max = (0, 365) if imp_vol_data.empty else (
        int(imp_vol_data["TimeToExpiry"].min()*365),
        int(imp_vol_data["TimeToExpiry"].max()*365)
    )
    min_days = form.number_input("Min Days to Expiry", t_min, t_max, t_min, key="min_days")
    max_days = form.number_input("Max Days to Expiry", t_min, t_max, t_max, key="max_days")
    
    form.slider("Regularization Weight (λ)", 0.0, 0.1, 0.0, step=0.005, key="reg_weight")
    form.slider("Multi-Start Count", 1, 10, 5, key="multi_start")
    form.checkbox("Filter Low Volume & Outliers", True, key="filter_opts")
    
    # The one-and-only Submit button:
    submit = form.form_submit_button(label="Apply")
    
    if not submit:
        # If the user hasn't hit “Apply” yet, we return None (or previous settings)
        st.sidebar.info("Hit Apply to update surface parameters")
        return None  

    # Once they click Apply, compute the derived values:
    min_strike = spot_price * (min_pct/100)
    max_strike = spot_price * (max_pct/100)
    time_range = (min_days/365.0, max_days/365.0)
    
    return min_strike, max_strike, time_range, st.session_state.filter_opts, st.session_state.reg_weight, st.session_state.multi_start


def create_iv_lookup_sidebar(imp_vol_data, min_strike_price, max_strike_price, spot_price, svi_df, option_type):
    """Create IV lookup UI in sidebar and process lookup request."""
    st.sidebar.markdown("---")
    st.sidebar.header("IV Lookup")
    
    specific_strike = st.sidebar.number_input(
        "Enter Specific Strike Price ($)",
        min_value=float(min_strike_price),
        max_value=float(max_strike_price),
        value=float(spot_price),
        step=1.0
    )
    
    available_days = sorted(imp_vol_data["TimeToExpiry"].unique() * 365)
    if available_days:
        days_to_expiry = st.sidebar.selectbox(
            "Days to Expiry", 
            options=available_days, 
            format_func=lambda x: f"{int(x)} days"
        )
        time_to_maturity_lookup = days_to_expiry / 365.0
    else:
        st.sidebar.warning("No expiration dates available")
        days_to_expiry = 30
        time_to_maturity_lookup = days_to_expiry / 365.0

    if st.sidebar.button("Look Up IV"):
        try:
            log_moneyness = np.log(specific_strike / spot_price)
            mask = np.isclose(svi_df["TimeToExpiry"], time_to_maturity_lookup, atol=0.01)
            filtered_by_time = svi_df[mask]
            
            if len(filtered_by_time) > 0:
                if option_type == "Moneyness":
                    specific_moneyness = specific_strike / spot_price
                    closest_row = filtered_by_time.iloc[(filtered_by_time["Moneyness"] - specific_moneyness).abs().argsort()[0]]
                    closest_strike_desc = f"moneyness {closest_row['Moneyness']:.2f}"
                else:
                    closest_row = filtered_by_time.iloc[(filtered_by_time["StrikePrice"] - specific_strike).abs().argsort()[0]]
                    closest_strike_desc = f"strike ${closest_row['StrikePrice']:.2f}"
                    
                iv_value = closest_row["ImpliedVolatility"]
                option_type_info = f" ({closest_row['OptionType']})" if "OptionType" in closest_row else ""
                
                st.sidebar.success(f"Implied Volatility: {iv_value:.2%}")
                st.sidebar.info(f"At {closest_strike_desc} with {int(closest_row['TimeToExpiry']*365)} days to expiry{option_type_info}")
            else:
                st.sidebar.warning("No data found for the selected expiry. Try another date.")
        except Exception as e:
            st.sidebar.error(f"Error looking up IV: {str(e)}")

def create_surface_plot(X, Y, Z, option_type, ticker, option_data_type):
    """Create the 3D volatility surface plot with blue→muted green→yellow→red scale."""
    import numpy as np
    from scipy.interpolate import griddata
    import plotly.graph_objects as go
    import streamlit as st

    if len(X) == 0 or len(Y) == 0 or len(Z) == 0:
        st.error("No data available for the selected parameters or time range. Please adjust your inputs.")
        return None

    # Build grid
    xi = np.linspace(min(X), max(X), 50)
    yi = np.linspace(min(Y), max(Y), 50)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((X, Y), Z, (xi, yi), method="linear", fill_value=np.nan)

    # Blue → muted green → yellow → red colorscale
    colorscale = [
        [0.0, "rgb(0,0,255)"],     # blue
        [0.33, "rgb(0,180,0)"],    # muted green
        [0.66, "rgb(255,255,0)"],  # yellow
        [1.0, "rgb(255,0,0)"]      # red
    ]

    fig = go.Figure(data=[go.Surface(
        x=xi,
        y=yi,
        z=zi,
        colorscale=colorscale,
        opacity=0.95,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True,
                size=(np.nanmax(zi) - np.nanmin(zi)) / 20
            )
        ),
        lighting=dict(
            ambient=0.8,
            diffuse=0.5,
            specular=0.5,
            roughness=0.9
        )
    )])

    if option_type == "Moneyness":
        hovertemplate = (
            "Time to Expiry: %{x:.2f} years<br>"
            "Moneyness: %{y:.2f}<br>"
            "Implied Volatility: %{z:.2f}%<extra></extra>"
        )
    else:
        hovertemplate = (
            "Time to Expiry: %{x:.2f} years<br>"
            "Strike Price: $%{y:.2f}<br>"
            "Implied Volatility: %{z:.2f}%<extra></extra>"
        )
    fig.update_traces(hovertemplate=hovertemplate)

    option_data_label = option_data_type.replace(" Only", "s").replace("Both Calls & Puts", "Calls & Puts")
    fig.update_layout(
        title=f"SVI Calibrated Volatility Surface of {ticker} Using {option_data_label}",
        scene=dict(
            xaxis_title="Time to Expiration (years)",
            yaxis_title="Moneyness" if option_type == "Moneyness" else "Strike Price ($)",
            zaxis_title="Implied Volatility (%)"
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#e5ecf6",
        width=1000,
        height=800
    )

    return fig

    if len(X) == 0 or len(Y) == 0 or len(Z) == 0:
        st.error("No data available for the selected parameters or time range. Please adjust your inputs.")
        return None
        
    xi = np.linspace(min(X), max(X), 50)
    yi = np.linspace(min(Y), max(Y), 50)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((X, Y), Z, (xi, yi), method="linear", fill_value=np.nan)
    
    fig = go.Figure(data=[go.Surface(
        x=xi,
        y=yi,
        z=zi,
        colorscale="Plasma",
        opacity=0.95,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True,
                size=(np.nanmax(zi)-np.nanmin(zi))/20
            )
        ),
        lighting=dict(
            ambient=0.8,
            diffuse=0.5,
            specular=0.5,
            roughness=0.9
        )
    )])
    
    if option_type == "Moneyness":
        hovertemplate = (
            "Time to Expiry: %{x:.2f} years<br>" +
            "Moneyness: %{y:.2f}<br>" +
            "Implied Volatility: %{z:.2f}%<extra></extra>"
        )
    else:
        hovertemplate = (
            "Time to Expiry: %{x:.2f} years<br>" +
            "Strike Price: $%{y:.2f}<br>" +
            "Implied Volatility: %{z:.2f}%<extra></extra>"
        )
    fig.update_traces(hovertemplate=hovertemplate)
    
    option_data_label = option_data_type.replace(" Only", "s").replace("Both Calls & Puts", "Calls & Puts")
    
    fig.update_layout(
        title=f"SVI Calibrated Volatility Surface of {ticker} Using {option_data_label}",
        scene=dict(
            xaxis_title="Time to Expiration (years)",
            yaxis_title="Moneyness" if option_type == "Moneyness" else "Strike Price ($)",
            zaxis_title="Implied Volatility (%)"
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#e5ecf6",
        width=1000, 
        height=800
    )
    
    return fig

def create_smile_plot(svi_df, imp_vol_data, selected_maturity, option_type, spot_price, option_data_type):
    """Create the volatility smile plot for a given maturity."""
    smile_fig = go.Figure()
    selected_cal_data = svi_df[np.isclose(svi_df["TimeToExpiry"], selected_maturity)]
    
    if len(selected_cal_data) > 0:
        if option_type == "Moneyness":
            smile_fig.add_trace(go.Scatter(
                x=selected_cal_data["Moneyness"],
                y=selected_cal_data["ImpliedVolatility"] * 100,
                mode="lines",
                name="SVI Model",
                line=dict(width=3)
            ))
        else:
            smile_fig.add_trace(go.Scatter(
                x=selected_cal_data["StrikePrice"],
                y=selected_cal_data["ImpliedVolatility"] * 100,
                mode="lines",
                name="SVI Model",
                line=dict(width=3)
            ))
        
        # Add spot price line
        spot_line_x = 1.0 if option_type == "Moneyness" else spot_price
        smile_fig.add_vline(
            x=spot_line_x,
            line_dash="dash",
            line_width=1,
            line_color="red",
            annotation_text="Spot Price",
            annotation_position="top right"
        )
        
        # Add market data points
        selected_market_data = imp_vol_data[np.isclose(imp_vol_data["TimeToExpiry"], selected_maturity)]
        if len(selected_market_data) > 0:
            if option_type == "Moneyness":
                smile_fig.add_trace(go.Scatter(
                    x=selected_market_data["Moneyness"],
                    y=selected_market_data["ImpliedVolatility"] * 100,
                    mode="markers",
                    name="Market Data",
                    marker=dict(size=10, symbol="circle")
                ))
            else:
                smile_fig.add_trace(go.Scatter(
                    x=selected_market_data["StrikePrice"],
                    y=selected_market_data["ImpliedVolatility"] * 100,
                    mode="markers",
                    name="Market Data",
                    marker=dict(size=10, symbol="circle")
                ))
        
        option_data_label = option_data_type.replace(" Only", "s").replace("Both Calls & Puts", "Calls & Puts")
        selected_days = int(selected_maturity * 365)
        
        smile_fig.update_layout(
            title=f"Volatility Smile for {selected_days} Days to Expiry ({option_data_label})",
            xaxis_title="Moneyness" if option_type == "Moneyness" else "Strike Price ($)",
            yaxis_title="Implied Volatility (%)",
            legend_title="Data Source",
            height=500,
            plot_bgcolor="rgba(240,240,240,0.2)",
            hovermode="closest"
        )
    
    return smile_fig

def create_comparison_plot(svi_df, unique_maturities, option_type, spot_price, option_data_type):
    """Create comparison plot of volatility smiles across all maturities."""
    comparison_fig = go.Figure()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, maturity in enumerate(unique_maturities):
        maturity_data = svi_df[np.isclose(svi_df["TimeToExpiry"], maturity)]
        if len(maturity_data) > 0:
            color = colors[i % len(colors)]
            days = int(maturity * 365)
            
            if option_type == "Moneyness":
                comparison_fig.add_trace(go.Scatter(
                    x=maturity_data["Moneyness"],
                    y=maturity_data["ImpliedVolatility"] * 100,
                    mode="lines",
                    name=f"{days} days",
                    line=dict(color=color)
                ))
            else:
                comparison_fig.add_trace(go.Scatter(
                    x=maturity_data["StrikePrice"],
                    y=maturity_data["ImpliedVolatility"] * 100,
                    mode="lines",
                    name=f"{days} days",
                    line=dict(color=color)
                ))
    
    # Add spot price line
    spot_line_x = 1.0 if option_type == "Moneyness" else spot_price
    comparison_fig.add_vline(
        x=spot_line_x,
        line_dash="dash",
        line_width=1,
        line_color="black",
        annotation_text="Spot Price",
        annotation_position="top right"
    )
    
    option_data_label = option_data_type.replace(" Only", "s").replace("Both Calls & Puts", "Calls & Puts")
    
    comparison_fig.update_layout(
        title=f"Comparison of Volatility Smiles Across All Maturities ({option_data_label})",
        xaxis_title="Moneyness" if option_type == "Moneyness" else "Strike Price ($)",
        yaxis_title="Implied Volatility (%)",
        legend_title="Days to Expiry",
        height=600,
        plot_bgcolor="rgba(240,240,240,0.2)",
        hovermode="closest"
    )
    
    return comparison_fig
def create_greek_2d_plot(greek_df, option_type, ticker, greek_name):
    """2‑D line+marker plot: Greeks vs Strike (or Moneyness) with labeled hover."""
    import plotly.graph_objects as go

    x = greek_df['Moneyness'] if option_type == 'Moneyness' else greek_df['StrikePrice']
    y = greek_df['GreekValue']

    if option_type == 'Moneyness':
        hover = (
            f"Moneyness: %{{x:.2f}}<br>"
            f"{greek_name}: %{{y:.4f}}<extra></extra>"
        )
    else:
        hover = (
            f"Strike Price: $%{{x:.2f}}<br>"
            f"{greek_name}: %{{y:.4f}}<extra></extra>"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name=greek_name,
        hovertemplate=hover
    ))
    fig.update_layout(
        title=f"{greek_name} vs {'Moneyness' if option_type=='Moneyness' else 'Strike Price ($)'} for {ticker}",
        xaxis_title="Moneyness" if option_type == 'Moneyness' else "Strike Price ($)",
        yaxis_title=greek_name,
        height=500,
        plot_bgcolor="rgba(240,240,240,0.2)"
    )
    return fig


def create_greek_surface_plot(X, Y, Z, option_type, ticker, greek_name):
    """3‑D surface: Greek vs Expiration & Strike/Moneyness with labeled hover."""
    from scipy.interpolate import griddata
    import plotly.graph_objects as go
    import numpy as np

    xi = np.linspace(min(X), max(X), 50)
    yi = np.linspace(min(Y), max(Y), 50)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((X, Y), Z, (xi, yi), method="linear", fill_value=np.nan)

    if option_type == "Moneyness":
        hover = (
            f"Expiration: %{{x:.2f}} yrs<br>"
            f"Moneyness: %{{y:.2f}}<br>"
            f"{greek_name}: %{{z:.4f}}<extra></extra>"
        )
    else:
        hover = (
            f"Expiration: %{{x:.2f}} yrs<br>"
            f"Strike Price: $%{{y:.2f}}<br>"
            f"{greek_name}: %{{z:.4f}}<extra></extra>"
        )

    surface = go.Surface(
        x=xi,
        y=yi,
        z=zi,
        colorscale="Viridis",
        hovertemplate=hover
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=f"{greek_name} Surface for {ticker}",
        scene=dict(
            xaxis_title="Time to Expiration (years)",
            yaxis_title="Moneyness" if option_type == 'Moneyness' else "Strike Price ($)",
            zaxis_title=greek_name
        ),
        width=1000, height=800
    )
    return fig
