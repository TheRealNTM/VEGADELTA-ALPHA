# VEGADELTA VOLATILITY MODELLING - DEMO VERSION

![image](https://github.com/user-attachments/assets/030b2c70-50be-43b8-9276-61ff84e152a2)

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
![Skjermbilde 2025-06-17 021922](https://github.com/user-attachments/assets/0f24a347-300e-4ba5-96e2-734574fc826a)
![Skjermbilde 2025-06-17 021914](https://github.com/user-attachments/assets/6db66542-9890-421d-b9e2-f20de65ca422)

### Implied‑Volatility (IV) Lookup

* Query IV for any strike/expiry pair.![Skjermbilde 2025-06-17 021958](https://github.com/user-attachments/assets/ef43aede-087f-4114-8182-39297bf5c6e3)


### Miscellaneous Analytics

* Extra utilities for exploratory analysis.

## Smile Analytics

* **Volatility smile**: fully configurable strike range, moneyness buckets and spline smoothing.
* **Smile comparison**: overlay smiles from different dates or underlyings.
  ![Skjermbilde 2025-06-17 022023](https://github.com/user-attachments/assets/4b5d3142-f902-4f0d-9331-f544f56f138b)

* Experimental utilities for advanced users.

## Greeks Analyser

* First‑, second‑ and third‑order Greeks, including

  * **Vanna** – sensitivity of delta to volatility
  * **Speed** – third‑order sensitivity of price to the underlying
  * **Charm** – rate of change of delta over time
* 2‑D plots of each Greek versus strike.
* ![image](https://github.com/user-attachments/assets/5324cf20-350b-4066-9d50-99e190b1e106)

* 3‑D Greek surfaces versus strike and expiration.
![Uploading image.png…]()

*Huge thanks to **GD** for the initial mock‑up!*

## Performance

> Initial load can exceed **10 seconds** because the code is not yet optimised — improvements are planned.

## Notes on Data Requests

*To be completed*
