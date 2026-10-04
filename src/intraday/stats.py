"""Descriptive and inferential statistics used throughout (A&B Table 1 etc.)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations 1..nlags via FFT (O(n log n), fine for 10^6 obs)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 ** int(np.ceil(np.log2(2 * n))))
    c = np.fft.irfft(f * np.conj(f))[: nlags + 1]
    return c[1:] / c[0]


def ljung_box(x: np.ndarray, lags: int = 10) -> float:
    n = len(x)
    r = acf(x, lags)
    return float(n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1))))


def table1_row(R: pd.DataFrame, k: int) -> dict:
    """A&B Table 1 columns for k-interval returns.

    VR   = K Var(R^k) / Var(sum_n R^k)                      (eq. 6)
    VR^A = K Var(|R^k|) / Var(sum_n |R^k|)                   (footnote 26)
    """
    T, N = R.shape
    K = N // k
    Rk = R.to_numpy().reshape(T, K, k).sum(axis=2)
    x, ax = Rk.ravel(), np.abs(Rk).ravel()
    return {
        "k": k, "T/k": x.size, "mean": x.mean(), "sd": x.std(ddof=1),
        "skew": st.skew(x), "kurtosis": st.kurtosis(x, fisher=False),
        "rho1": acf(x, 1)[0], "Q10": ljung_box(x),
        "VR": K * Rk.var(ddof=1) / Rk.sum(axis=1).var(ddof=1) if K > 1 else np.nan,
        "rho1_abs": acf(ax, 1)[0], "Q10_abs": ljung_box(ax),
        "VR_abs": K * np.abs(Rk).var(ddof=1) / np.abs(Rk).sum(axis=1).var(ddof=1) if K > 1 else np.nan,
    }


def interval_mean_ci(R: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Per-interval mean with (i) A&B's constant iid band and (ii) an
    interval-specific band using that interval's own variance."""
    z = st.norm.ppf(0.5 + level / 2)
    m, s, n = R.mean(), R.std(ddof=1), R.count()
    pooled = R.to_numpy().std(ddof=1) / np.sqrt(n)
    return pd.DataFrame({"mean": m, "se": s / np.sqrt(n),
                         "lo": m - z * s / np.sqrt(n), "hi": m + z * s / np.sqrt(n),
                         "iid_lo": -z * pooled, "iid_hi": z * pooled})


def block_bootstrap_ci(R: pd.DataFrame, func, n_boot: int = 500, block: int = 20,
                       level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Moving-block bootstrap over *days* for any per-interval statistic.

    Resampling whole days keeps the intraday shape intact; blocks of `block`
    consecutive days preserve the day-to-day volatility clustering that an
    iid bootstrap would wrongly destroy (which would make bands too narrow).
    """
    rng = np.random.default_rng(seed)
    T = len(R)
    nb = int(np.ceil(T / block))
    X = R.to_numpy()
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, T - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T]
        draws.append(np.asarray(func(pd.DataFrame(X[idx], columns=R.columns))))
    d = np.array(draws)
    a = (1 - level) / 2
    return pd.DataFrame({"est": np.asarray(func(R)), "lo": np.quantile(d, a, axis=0),
                         "hi": np.quantile(d, 1 - a, axis=0)}, index=R.columns)


def fdr_bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries at false-discovery rate q."""
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0
    mask = np.zeros(m, bool)
    mask[order[:k]] = True
    return mask


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """t-stat of the mean of x with Newey-West (Bartlett) HAC standard error."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = x - x.mean()
    lrv = u @ u / n
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return float(x.mean() / np.sqrt(lrv / n))
