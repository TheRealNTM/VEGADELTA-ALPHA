# VEGADELTA VOLATILITY MODELLING - DEMO VERSION


vegadelta.com by TheRealNTM

Contact: ntm@vegadelta.com

This **Streamlit**‑based application lets you model, visualise and interrogate an options *volatility surface* for any listed equity or index. Simply enter a ticker and explore the tools below.

## Key Features

### Volatility Surface

* Interactive 3‑D surface with user‑tunable modelling hyper‑parameters (smoothing window, fit method, extrapolation anchors, etc.).

### Stochastic‑Volatility‑Inspired Surface (SVIS)

* Parametric extrapolation technique that extends the market surface where strikes are sparse.

### Market Data Auto‑Fill

* Dividend yields and risk‑free term structure are fetched automatically from the latest data.

### Implied‑Volatility (IV) Lookup

* Query IV for any strike/expiry pair.

### Miscellaneous Analytics

* Extra utilities for exploratory analysis.

## Smile Analytics

* **Volatility smile**: fully configurable strike range, moneyness buckets and spline smoothing.
* **Smile comparison**: overlay smiles from different dates or underlyings.
* Experimental utilities for advanced users.

## Greeks Analyser

* First‑, second‑ and third‑order Greeks, including

  * **Vanna** – sensitivity of delta to volatility
  * **Speed** – third‑order sensitivity of price to the underlying
  * **Charm** – rate of change of delta over time
* 2‑D plots of each Greek versus strike.
* 3‑D Greek surfaces versus strike and expiration.

*Huge thanks to **GD** for the initial mock‑up!*

## Performance

> Initial load can exceed **10 seconds** because the code is not yet optimised — improvements are planned.

## Notes on Data Requests

*To be completed*
