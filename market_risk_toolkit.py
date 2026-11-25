"""
Market Risk & Instrument Valuation Toolkit
==========================================
Part 1: Portfolio Value-at-Risk (Historical, Parametric, Monte Carlo) + Expected Shortfall
Part 2: VaR backtesting (rolling window, Kupiec POF test)
Part 3: Valuation of financial instruments
        - Black-Scholes option pricing + Greeks + put-call parity check
        - Fixed-rate bond pricing, duration, convexity

Data: by default uses SIMULATED correlated daily prices (so the project runs anywhere).
To use real data, put a CSV with a Date column and one price column per asset at
data/prices.csv (e.g. downloaded from Yahoo Finance / NSE) and re-run.

Run:  python market_risk_toolkit.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

PORTFOLIO_VALUE = 1_000_000  # portfolio size in currency units


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def load_prices(path="data/prices.csv"):
    """Load real prices if available, otherwise simulate correlated GBM prices."""
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
        print(f"Loaded real price data: {df.shape[0]} days, assets = {list(df.columns)}")
        return df

    print("No data/prices.csv found -> using simulated correlated prices (3 years).")
    assets = ["Equity_A", "Equity_B", "Equity_C", "Bond_Fund"]
    mu = np.array([0.12, 0.10, 0.14, 0.05]) / 252          # annual drift -> daily
    vol = np.array([0.22, 0.18, 0.28, 0.06]) / np.sqrt(252)  # annual vol -> daily
    corr = np.array([[1.0, 0.6, 0.5, -0.1],
                     [0.6, 1.0, 0.4, -0.1],
                     [0.5, 0.4, 1.0, -0.2],
                     [-0.1, -0.1, -0.2, 1.0]])
    cov = np.outer(vol, vol) * corr
    n = 756
    # fat tails: scale normal shocks by a random chi-square factor (Student-t like)
    shocks = np.random.multivariate_normal(mu, cov, size=n)
    shocks *= np.sqrt(5 / np.random.chisquare(5, size=(n, 1)))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.bdate_range(end="2026-11-20", periods=n)
    return pd.DataFrame(prices, index=idx, columns=assets)


# ----------------------------------------------------------------------------
# 2. VALUE-AT-RISK
# ----------------------------------------------------------------------------
def historical_var(port_returns, cl=0.99):
    """Loss exceeded only (1-cl) of the time in the past data."""
    return -np.percentile(port_returns, (1 - cl) * 100)


def historical_es(port_returns, cl=0.99):
    """Expected Shortfall: average loss on days worse than VaR."""
    var = historical_var(port_returns, cl)
    tail = port_returns[port_returns <= -var]
    return -tail.mean()


def parametric_var(returns, weights, cl=0.99):
    """Variance-covariance VaR, assumes normally distributed returns."""
    mu_p = returns.mean().values @ weights
    sigma_p = np.sqrt(weights @ returns.cov().values @ weights)
    return -(mu_p + norm.ppf(1 - cl) * sigma_p)


def monte_carlo_var(returns, weights, cl=0.99, n_sims=100_000):
    """Simulate correlated returns using the Cholesky decomposition of the covariance."""
    mu = returns.mean().values
    L = np.linalg.cholesky(returns.cov().values)
    z = np.random.standard_normal((n_sims, len(weights)))
    sims = mu + z @ L.T
    port_sims = sims @ weights
    return -np.percentile(port_sims, (1 - cl) * 100), port_sims


# ----------------------------------------------------------------------------
# 3. BACKTESTING
# ----------------------------------------------------------------------------
def backtest_var(port_returns, window=250, cl=0.99):
    """Rolling historical VaR; count days when actual loss > VaR (exceptions)."""
    var_series = port_returns.rolling(window).apply(
        lambda x: -np.percentile(x, (1 - cl) * 100), raw=True).shift(1)
    data = pd.DataFrame({"ret": port_returns, "var": var_series}).dropna()
    data["exception"] = data["ret"] < -data["var"]
    return data


def kupiec_pof(n_obs, n_exc, cl=0.99):
    """Kupiec proportion-of-failures likelihood-ratio test.
    H0: the true exception rate equals 1-cl. Reject if LR > chi2(1) critical value."""
    p = 1 - cl
    x, n = n_exc, n_obs
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    else:
        phat = x / n
        lr = -2 * (((n - x) * np.log(1 - p) + x * np.log(p))
                   - ((n - x) * np.log(1 - phat) + x * np.log(phat)))
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value


# ----------------------------------------------------------------------------
# 4. INSTRUMENT VALUATION
# ----------------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option="call", q=0.0):
    """European option price and Greeks (continuous dividend yield q)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1))
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma,
            "vega_per_1pct": vega / 100, "theta_per_day": theta / 365}


def bond_metrics(face, coupon_rate, ytm, years, freq=2):
    """Price, Macaulay & modified duration, convexity of a fixed-rate bond."""
    n = int(years * freq)
    t = np.arange(1, n + 1)
    cf = np.full(n, face * coupon_rate / freq)
    cf[-1] += face
    y = ytm / freq
    disc = (1 + y) ** t
    pv = cf / disc
    price = pv.sum()
    mac = (t / freq * pv).sum() / price
    mod = mac / (1 + y)
    conv = (cf * t * (t + 1) / disc / (1 + y) ** 2).sum() / (price * freq ** 2)
    return {"price": price, "macaulay_duration": mac, "modified_duration": mod, "convexity": conv}


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    prices = load_prices()
    returns = prices.pct_change().dropna()
    w = np.full(returns.shape[1], 1 / returns.shape[1])  # equal weights
    port_ret = pd.Series(returns.values @ w, index=returns.index)

    print("\n=== PART 1: 1-DAY VALUE-AT-RISK (portfolio = {:,.0f}) ===".format(PORTFOLIO_VALUE))
    rows = []
    for cl in (0.95, 0.99):
        mc_var, mc_sims = monte_carlo_var(returns, w, cl)
        rows.append({
            "Confidence": f"{cl:.0%}",
            "Historical VaR": historical_var(port_ret, cl) * PORTFOLIO_VALUE,
            "Parametric VaR": parametric_var(returns, w, cl) * PORTFOLIO_VALUE,
            "Monte Carlo VaR": mc_var * PORTFOLIO_VALUE,
            "Expected Shortfall (hist)": historical_es(port_ret, cl) * PORTFOLIO_VALUE,
        })
    var_table = pd.DataFrame(rows).set_index("Confidence").round(0)
    print(var_table.to_string())
    var_table.to_csv(f"{OUT}/var_results.csv")

    # 10-day regulatory-style scaling (square-root-of-time rule)
    v99 = historical_var(port_ret, 0.99) * PORTFOLIO_VALUE
    print(f"\n10-day 99% VaR (sqrt-of-time scaling, as in Basel market risk): {v99 * np.sqrt(10):,.0f}")

    # Chart: return distribution with VaR lines
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(port_ret * 100, bins=50, color="#4C72B0", alpha=0.8)
    ax.axvline(-historical_var(port_ret, 0.95) * 100, color="orange", ls="--", label="95% VaR")
    ax.axvline(-historical_var(port_ret, 0.99) * 100, color="red", ls="--", label="99% VaR")
    ax.set_title("Daily Portfolio Returns with Historical VaR")
    ax.set_xlabel("Daily return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/var_distribution.png", dpi=150)
    plt.close(fig)

    print("\n=== PART 2: BACKTESTING (rolling 250-day historical VaR, 99%) ===")
    bt = backtest_var(port_ret, 250, 0.99)
    n_obs, n_exc = len(bt), int(bt["exception"].sum())
    lr, pval = kupiec_pof(n_obs, n_exc, 0.99)
    expected = n_obs * 0.01
    # Basel traffic light (250 days): green 0-4, yellow 5-9, red 10+
    zone = "GREEN" if n_exc <= 4 else "YELLOW" if n_exc <= 9 else "RED"
    print(f"Observations: {n_obs} | Exceptions: {n_exc} | Expected: {expected:.1f}")
    print(f"Kupiec LR = {lr:.3f}, p-value = {pval:.3f} -> "
          f"{'model ACCEPTED' if pval > 0.05 else 'model REJECTED'} at 5% level")
    print(f"Basel-style traffic light zone (scaled to 250 days): "
          f"{zone if n_obs <= 300 else 'see Kupiec (sample > 250 days)'}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt.index, bt["ret"] * 100, lw=0.7, label="Daily return")
    ax.plot(bt.index, -bt["var"] * 100, color="red", lw=1, label="99% VaR (negative)")
    ax.scatter(bt.index[bt["exception"]], bt["ret"][bt["exception"]] * 100,
               color="black", zorder=5, label="Exceptions")
    ax.set_title("VaR Backtest")
    ax.set_ylabel("Return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/var_backtest.png", dpi=150)
    plt.close(fig)

    print("\n=== PART 3A: BLACK-SCHOLES OPTION VALUATION ===")
    S, K, T, r, sigma = 100, 105, 0.5, 0.065, 0.25
    call = black_scholes(S, K, T, r, sigma, "call")
    put = black_scholes(S, K, T, r, sigma, "put")
    print(f"Inputs: S={S}, K={K}, T={T}y, r={r:.1%}, vol={sigma:.0%}")
    print("CALL:", {k: round(v, 4) for k, v in call.items()})
    print("PUT :", {k: round(v, 4) for k, v in put.items()})
    parity_lhs = call["price"] - put["price"]
    parity_rhs = S - K * np.exp(-r * T)
    print(f"Put-call parity check: C-P = {parity_lhs:.4f}, S-K*e^(-rT) = {parity_rhs:.4f} "
          f"-> {'OK' if abs(parity_lhs - parity_rhs) < 1e-8 else 'FAIL'}")

    print("\n=== PART 3B: FIXED-RATE BOND VALUATION ===")
    b = bond_metrics(face=1000, coupon_rate=0.07, ytm=0.065, years=10, freq=2)
    print({k: round(v, 4) for k, v in b.items()})
    dy = 0.01  # +100bp
    approx = -b["modified_duration"] * dy + 0.5 * b["convexity"] * dy ** 2
    new_price = bond_metrics(1000, 0.07, 0.065 + dy, 10, 2)["price"]
    actual = new_price / b["price"] - 1
    print(f"+100bp yield shock: duration+convexity estimate = {approx:.4%}, "
          f"full repricing = {actual:.4%}")

    print(f"\nDone. Charts and CSV saved in ./{OUT}/")


if __name__ == "__main__":
    main()
