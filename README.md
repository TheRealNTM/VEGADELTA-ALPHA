**VEGADELTA VOLATILITY MODELLING** - PRE RELEASE VERSION
vegadelta.com

This Streamlit‑based application lets you model, visualise and interrogate the full volatility surface for listed equity or index options. Enter a ticker and explore:


**Volatility surface** (all modelling hyper‑parameters—such as smoothing window, fit method and extrapolation anchors—are user‑adjustable)

  ** - Stochastic‑volatility**‑inspired surface (SVIS, a parametric extrapolation technique often used when market strikes are sparse)

  **Automatic dividend and market rate estimation** (values are pulled automatically from the latest market data for dividends and for the yield structure)

  **Clean UI for rapid experimentation**

  **Implied‑volatility** (IV) lookup for any strike/expiry

  **Miscellaneous analytics for additional exploratory work**


**Smile analytics**

  **Volatility smile** (parameters including strike range, moneyness buckets and smoothing splines are fully configurable)

  **Comparison of volatility smiles across dates**

  **Experimental utilities for advanced users**


**Greeks analyser**

  **First‑, second‑ and third‑order Greeks** (including vanna – sensitivity of delta to volatility, speed – third‑order sensitivity of price to the underlying, charm – rate of change of delta over time)

  **2‑D plots** of each Greek versus strike

  **3‑D surfaces of Greeks versus strike** and expiration

  **Huge thanks to GD for the initial mock‑up!!!**


Initial loading can take a couple seconds because the code has not yet been optimised; this will improve in future releases.

Notes on Data Requests

(To be completed)

