import streamlit as st
import main as m
import svi_module as svi
import ui_components as ui
import numpy as np
import yfinance as yf
import pandas as pd
import warnings
import greeks_module as gm
import time   # add at top of your app.py
import term_structure as ts   # NEW
import svi_diagnostics as diag
import sys, os
def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)


warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
ui.setup_page_config()

st.markdown("""
  <style>
    /* force light-ish backgrounds even if dark theme is active */
    .css-18e3th9 { background-color: #f0f2f6 !important; }
    .css-1d391kg { background-color: #ffffff !important; }
    /* override sidebar to light too */
    [data-testid="stSidebar"] { background-color: #ffffff !important; }
  </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header & ticker input
# -----------------------------------------------------------------------------
# ——— Header with logo + title ———
col1, col2 = st.columns([1, 9])
with col1:
    st.image(resource_path("vegadelta_logo.png"), width=200)
with col2:
    st.header("VegaDelta Options Analytics")
st.markdown("---")


with st.container():
    st.markdown('### Enter Stock Ticker')
    ticker = st.text_input('', value='AAPL', key='ticker_input').upper()

    try:
        # ---------------- Stock data & live rates -----------------------
        stock, spot_prices, spot_price = m.get_stock_data(ticker)

        # --- Options-data quality flag ---------------------------------
        num_expirations = len(stock.options or [])
        if num_expirations >= 15:
            options_quality = "High"
        elif num_expirations >= 7:
            options_quality = "Medium"
        else:
            options_quality = "Low"

        stock_info = yf.Ticker(ticker, session=m.session).info

        # Dividend yield (forward ➜ trailing fallback) ------------------
        dividend_yield = stock_info.get('dividendYield')
        if dividend_yield is None:
            dividend_yield = stock_info.get('trailingAnnualDividendYield') or 0.0
        if dividend_yield > 0.20:          # Yahoo sometimes returns 0.51 = 51 %
            dividend_yield /= 100.0

        # ----------- 🔥 NEW: live term-structure & sidebar -------------
        r_curve, q_curve, live_rates = ts.default_curves(dividend_yield)
        # For any legacy scalar use:
        risk_free_rate = live_rates.get("2Y", 0.03)

        # Show key metrics in the main area
        ui.display_stock_metrics(ticker, stock_info, options_quality)
        st.markdown('---')

        # Sidebar now shows ON-to-2.5Y curve
        option_type, option_data_type = ui.create_sidebar_inputs(live_rates,
                                                                 dividend_yield)

    except Exception as e:
        st.error(f"Error retrieving data for {ticker}: {e}")
        st.stop()




# -----------------------------------------------------------------------------
# Option chain
# -----------------------------------------------------------------------------
calls_data, puts_data, _ = m.get_options_data(stock)

if option_data_type == 'Calls Only':
    options_data = calls_data.copy()
    st.info('Using call options data only for volatility calculations.')
elif option_data_type == 'Puts Only':
    options_data = puts_data.copy()
    st.info('Using put options data only for volatility calculations.')
else:
    options_data = pd.concat([calls_data, puts_data], ignore_index=True)
    st.info('Using both call and put options data for volatility calculations.')

# -----------------------------------------------------------------------------
# Advanced‑settings placeholder & default ranges
# -----------------------------------------------------------------------------
adv_placeholder = st.sidebar.empty()

default_strike_range = (75, 125)  # % of spot
min_strike_price = spot_price * default_strike_range[0] / 100
max_strike_price = spot_price * default_strike_range[1] / 100

# Initial IV calc (needed so advanced‑settings slider bounds make sense)
filtered_options_data = m.filter_calls_data(options_data,
                                            spot_price,
                                            min_strike_price,
                                            max_strike_price)
imp_vol_data = m.calculate_implied_volatility(filtered_options_data, spot_price,
                                               r_curve, q_curve)

# -----------------------------------------------------------------------------
# Advanced settings UI (inside sidebar placeholder)
# -----------------------------------------------------------------------------
with adv_placeholder.container():
    adv_settings = ui.create_advanced_settings(imp_vol_data, spot_price)

# Decide on settings – either user‑supplied or sensible defaults
if adv_settings is None:
    surface_time_range = (imp_vol_data['TimeToExpiry'].min(),
                          imp_vol_data['TimeToExpiry'].max())
    filter_options     = True
    reg_weight         = 0.0
    multi_start_count  = 5
    # keep min/max_strike_price defaults
else:
    (min_strike_price, max_strike_price,
     surface_time_range, filter_options,
     reg_weight, multi_start_count) = adv_settings

# -----------------------------------------------------------------------------
# Re‑filter options with updated strike bounds
# -----------------------------------------------------------------------------
filtered_options_data = m.filter_calls_data(options_data, spot_price,
                                           min_strike_price, max_strike_price)
imp_vol_data = m.calculate_implied_volatility(filtered_options_data, spot_price,
                                               r_curve, q_curve)

# Tag option types -------------------------------------------------------------
if option_data_type == 'Calls Only':
    imp_vol_data['OptionType'] = 'Call'
elif option_data_type == 'Puts Only':
    imp_vol_data['OptionType'] = 'Put'
else:
    if 'contractSymbol' in filtered_options_data.columns:
        imp_vol_data['OptionType'] = filtered_options_data['contractSymbol'].apply(
            lambda x: 'Call' if 'C' in x.upper() else 'Put')
    else:
        imp_vol_data['OptionType'] = 'Both'

# Outlier filter ---------------------------------------------------------------
if filter_options:
    imp_vol_data = svi.filter_outliers(imp_vol_data)

# Add derived columns ----------------------------------------------------------
imp_vol_data['LogMoneyness'] = np.log(imp_vol_data['StrikePrice'] / spot_price)
if option_type == 'Moneyness':
    imp_vol_data['Moneyness'] = imp_vol_data['StrikePrice'] / spot_price

# -----------------------------------------------------------------------------
# SVI calibration
# -----------------------------------------------------------------------------
unique_maturities = sorted(imp_vol_data['TimeToExpiry'].unique())

with st.spinner('Working hard or hardly working...'):
    svi_df = svi.generate_svi_calibrated_data(
            unique_maturities,
            imp_vol_data,
            spot_price,
            risk_free_rate,      # NEW
            dividend_yield,      # NEW
            reg_weight,
            multi_start_count,
            option_data_type)

# -----------------------------------------------------------------------------
# Sidebar IV lookup
# -----------------------------------------------------------------------------
ui.create_iv_lookup_sidebar(imp_vol_data, min_strike_price, max_strike_price,
                            spot_price, svi_df, option_type)

# -----------------------------------------------------------------------------
# Surface plot variables (for SVI tab)
# -----------------------------------------------------------------------------
svi_filtered = svi_df[(svi_df['TimeToExpiry'] >= surface_time_range[0]) &
                      (svi_df['TimeToExpiry'] <= surface_time_range[1])]
X_surface = svi_filtered['TimeToExpiry'].values
Y_surface = (svi_filtered['Moneyness'].values if option_type == 'Moneyness'
             else svi_filtered['StrikePrice'].values)
Z_surface = svi_filtered['ImpliedVolatility'].values * 100

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(['SVI Surface', 'Volatility Smile',
                                  'Comparison', 'Greeks'])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – SVI Surface
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader('SVI Calibrated Volatility Surface')
    fig = ui.create_surface_plot(X_surface, Y_surface, Z_surface,
                                 option_type, ticker, option_data_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        msg = 'with low volume options and outliers filtered out.' if filter_options else '.'
        st.info(f'Displaying SVI calibrated volatility surface using {option_data_type.lower()} {msg}')

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – Smile (single maturity)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader('Calibrated Volatility Smiles')
    col1, col2 = st.columns([3, 1])

    with col1:
        day_choices = [int(m * 365) for m in unique_maturities]
        if not day_choices:
            st.warning('No maturity options available.')
            selected_maturity = None
        else:
            days = st.selectbox('Select Days to Expiry', options=day_choices,
                                format_func=lambda x: f'{x} days', key='smile_days')
            selected_maturity = days / 365.0

    with col2:
        show_raw = st.button('Show Raw Data', key='raw_data_button')

    if show_raw:
        st.subheader('Raw Volatility Data')
        st.write('Market Data:')
        st.dataframe(imp_vol_data.sort_values(['TimeToExpiry', 'StrikePrice']))
        st.write('Calibrated SVI Data:')
        st.dataframe(svi_df.sort_values(['TimeToExpiry', 'StrikePrice']))

    if selected_maturity is not None:
        smile_fig = ui.create_smile_plot(svi_df, imp_vol_data, selected_maturity,
                                         option_type, spot_price, option_data_type)
        st.plotly_chart(smile_fig, use_container_width=True)
        st.info(f'Volatility smile for options expiring in {int(selected_maturity*365)} days.')

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Comparison (all maturities)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader('Comparison of Volatility Smiles')
    comp_fig = ui.create_comparison_plot(svi_df, unique_maturities,
                                         option_type, spot_price, option_data_type)
    st.plotly_chart(comp_fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 – Greeks
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader('Greek Visualizations')

    # Controls
    colA, colB, colC, colD = st.columns([2,2,1,1])
    with colA:
        greek_choice = st.selectbox('Select Greek', list(gm.GreekFuncs.keys()), key='greek_select')
    with colB:
        data_source = st.radio('Data Source', ['SVI Model', 'Market (Raw)'], horizontal=True, key='greek_source')
    with colC:
        plot_type = st.radio('Plot Type', ['3D Surface', '2D (Strike vs Greek)'], horizontal=True, key='greek_plot_type')
    with colD:
        generate = st.button('Generate', key='greek_generate')

    if generate:
        # Step 1: Compute Greeks
        with st.spinner('Step 1/3 — Computing Greek values…'):
            time.sleep(0.7)
            try:
                df_src   = svi_df if data_source=='SVI Model' else imp_vol_data
                greek_df = gm.compute_greek(df_src, greek_choice,
                                            spot_price, risk_free_rate, dividend_yield)
            except Exception as e:
                st.error(f'Error computing {greek_choice}: {e}')
                greek_df = pd.DataFrame()

        # Step 2: Filter/subset
        with st.spinner('Step 2/3 — Filtering & cleaning data…'):
            time.sleep(0.7)
            if not greek_df.empty and 'GreekValue' in greek_df.columns:
                if plot_type.startswith('2D'):
                    days_list = sorted(greek_df['TimeToExpiry'].unique()*365)
                    days      = st.selectbox('Days to Expiry', days_list,
                                              format_func=lambda x: f"{int(x)} days",
                                              key='greek2d_days')
                    ttm       = days/365.0
                    greek_df  = greek_df[np.isclose(greek_df['TimeToExpiry'], ttm, atol=0.01)]

        # Step 3: Render plot
        with st.spinner('Step 3/3 — Rendering chart…'):
            time.sleep(0.7)
            if greek_df.empty:
                st.warning('Nothing to plot after filtering.')
            else:
                if plot_type=='3D Surface':
                    Xg = greek_df['TimeToExpiry'].values
                    Yg = greek_df['Moneyness'].values if option_type=='Moneyness' else greek_df['StrikePrice'].values
                    Zg = greek_df['GreekValue'].values

                    fig = ui.create_greek_surface_plot(Xg, Yg, Zg,
                                                       option_type, ticker, greek_choice)
                    st.plotly_chart(fig, use_container_width=True)

                else:  # 2D plot
                    fig = ui.create_greek_2d_plot(greek_df,
                                                 option_type, ticker, greek_choice)
                    st.plotly_chart(fig, use_container_width=True)

# enforce calendar monotonicity
svi_df, bumps = diag.enforce_calendar_consistency(svi_df)

# --------------------------------------------------------------
#  AFTER you finish building svi_df
# --------------------------------------------------------------
import svi_diagnostics as diag

report = diag.diagnostics_report(svi_df, imp_vol_data)
diag.pretty_print(report)          # <<< neat table to terminal

# force flush even in buffered environments
import sys, os
sys.stdout.flush()

print(f"\nDiagnostics printed from: {os.path.abspath(__file__)}")
sys.stdout.flush()
# --------------------------------------------------------------
