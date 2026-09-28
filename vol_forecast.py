"""Volatility forecasting: rolling std vs EWMA vs GARCH vs GJR-GARCH.
Out-of-sample one-day-ahead forecasts, expanding window, refit every 21 days.
Usage: python vol_forecast.py [--synthetic]"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model

TICKERS = {"TLT": "20y+ US Treasuries", "IEF": "7-10y US Treasuries", "LQD": "IG Credit", "SPY": "US Equities"}
MIN_TRAIN, REFIT = 1000, 21

def load_returns(synthetic=False):
    if synthetic:
        rng = np.random.default_rng(0); n = 3000
        out = {}
        for t in TICKERS:
            r, s2 = [], 1.0
            for _ in range(n):
                e = np.sqrt(s2) * rng.standard_normal()
                s2 = 0.05 + 0.08 * e**2 + 0.9 * s2; r.append(e)
            out[t] = r
        return pd.DataFrame(out, index=pd.bdate_range("2012-01-02", periods=n))
    import yfinance as yf
    px = yf.download(list(TICKERS), start="2008-01-01", auto_adjust=True, progress=False)["Close"].dropna()
    return 100 * np.log(px).diff().dropna()      # percent log returns

def garch_forecasts(r, kind):
    """Refit every REFIT days; between refits, filter with fixed params. Forecast for t+1 made at t."""
    out = pd.Series(index=r.index, dtype=float)
    kw = dict(vol="GARCH", p=1, q=1) if kind == "garch" else dict(vol="GARCH", p=1, o=1, q=1)
    for i0 in range(MIN_TRAIN, len(r) - 1, REFIT):
        res = arch_model(r.iloc[:i0], mean="Constant", dist="t", **kw).fit(disp="off")
        full = arch_model(r, mean="Constant", dist="t", **kw).fix(res.params)
        k = min(REFIT, len(r) - i0)
        f = full.forecast(horizon=1, start=i0 - 1, reindex=False).variance.iloc[:k, 0]
        out.iloc[i0: i0 + k] = f.values           # forecast for day t made with data through t-1
    return out

def forecasts(r):
    df = pd.DataFrame(index=r.index)
    df["RollingStd63"] = (r.rolling(63).var()).shift(1)
    df["EWMA_0.94"] = (r**2).ewm(alpha=0.06, adjust=False).mean().shift(1)
    df["GARCH"] = garch_forecasts(r, "garch")
    df["GJR-GARCH"] = garch_forecasts(r, "gjr")
    return df

def evaluate(r, f):
    rv = r**2                                          # squared-return proxy for realized variance
    rows = {}
    for c in f:
        m = f[c].notna() & (f[c] > 0)
        rows[c] = {"MSE": ((rv[m] - f[c][m])**2).mean(), "QLIKE": (np.log(f[c][m]) + rv[m] / f[c][m]).mean(), "N": int(m.sum())}
    return pd.DataFrame(rows).T.sort_values("QLIKE")

if __name__ == "__main__":
    ret = load_returns("--synthetic" in sys.argv)
    allres = []
    for t in TICKERS:
        r = ret[t].dropna(); f = forecasts(r)
        common = f.dropna().index
        res = evaluate(r.loc[common], f.loc[common]); res.insert(0, "Asset", t); allres.append(res)
        ax = np.sqrt(f.loc[common, ["RollingStd63", "GARCH"]]).plot(figsize=(10, 4), title=f"{t}: 1-day vol forecasts (%)")
        r.loc[common].abs().plot(ax=ax, alpha=0.2, color="gray", label="|return|"); ax.legend(); plt.tight_layout()
        plt.savefig(f"vol_{t}.png", dpi=120); plt.close()
    table = pd.concat(allres); table.to_csv("results.csv"); print(table.round(4))
