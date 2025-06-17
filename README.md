VEGADELTA VOLATILITY MODELLING - PRE RELEASE  - If you want to model the vol surface of options this is the place to do it! The UI is based in streamlit which makes it great for simple use.
Just input a ticker and get a bunch of info about the options. The output is 
1. **Volatility surface** (you can change a heap of parameters to make it better)Add commentMore actions
   - Stochastic volatility inspired surface
   - Automatic dividend and risk free rate calculation
   - Nice UI
   - IV lookup
   - More random stuff
2. **Volatility smile**(you can also change a bunch of parameters to make it better here too)
   - Comparison of volatility smiles
   - More random shit
3. **Greeks Analyser**
   - First, second and third derivative Greeks (Incl vanna, speed, charm, etc, etc)
   - 2D plotting of greeks compared to strike
   - 3D greek surface compared to strike and expiration
Huge thanks to GD for the initial mock up!

vegadelta.com


# Options Volatility Surface Toolkit

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
