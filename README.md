# Market Risk & Instrument Valuation Toolkit

Python project covering market risk (VaR, Expected Shortfall, backtesting) and valuation of financial instruments (options, bonds).

## What it does
1. **VaR (1-day, 95% and 99%)** using three methods: Historical, Parametric (variance-covariance), Monte Carlo (Cholesky).
2. **Expected Shortfall** (average loss beyond VaR) and 10-day scaling (square-root-of-time).
3. **Backtesting**: rolling 250-day VaR, exception count, Kupiec POF test.
4. **Black-Scholes** call/put pricing, Greeks (delta, gamma, vega, theta), put-call parity check.
5. **Bond valuation**: price, Macaulay/modified duration, convexity, +100bp shock comparison.

## Run
```
pip install numpy pandas scipy matplotlib
python market_risk_toolkit.py
```
Uses simulated data by default. To use real data, add `data/prices.csv` (Date + one price column per asset).

## Interview cheat sheet (learn this, don't memorise it)
- **VaR**: the loss that won't be exceeded with 99% confidence over 1 day. In my run, the 99% 1-day VaR is about 2.5% of the portfolio.
- **Why 3 methods?** Historical: no distribution assumption, but depends on the window. Parametric: fast, but assumes normality and underestimates fat tails. Monte Carlo: flexible, but model-dependent and computationally heavy.
- **Why Expected Shortfall?** VaR ignores how bad the tail is; ES measures it, and Basel FRTB replaced VaR with ES for this reason.
- **Cholesky**: turns independent random numbers into correlated ones that match the covariance matrix.
- **Kupiec test**: checks if exceptions (7) are statistically consistent with the expected rate (1% -> ~5). p-value 0.41 means we can't reject the model.
- **Square-root-of-time**: 10-day VaR = 1-day VaR x sqrt(10), valid only if returns are i.i.d.
- **Limitations (say these, it shows maturity)**: simulated data, equal weights, linear portfolio (no option positions in VaR), square-root rule, constant volatility in Black-Scholes.
- **Duration/convexity**: duration is first-order price sensitivity to yield; convexity is the curvature correction. My estimate (-6.849%) vs full repricing (-6.860%) shows convexity captures most of the gap.
- **Black-Scholes assumptions**: lognormal prices, constant vol and rate, no jumps, European exercise.

## Possible extensions
Real NSE/Yahoo data, GARCH volatility, EWMA VaR, delta-normal VaR for options, Streamlit dashboard.
