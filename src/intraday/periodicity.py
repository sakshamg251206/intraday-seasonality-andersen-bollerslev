"""Intraday periodic volatility component s_{t,n}.

Model (A&B eq. 7):   R_{t,n} = E(R) + sigma_t * s_{t,n} * Z_{t,n} / sqrt(N)

Taking logs of squared demeaned returns isolates s (A&B eq. A.1):

    x_{t,n} = 2 log|R_{t,n} - Rbar| - log sigma_t^2 + log N = log s_{t,n}^2 + log Z_{t,n}^2

so a regression of x on smooth functions of n recovers log s^2 up to the
constant E[log Z^2], which the final normalisation (eq. A.4) removes anyway.

Two estimators:
* fit_fff      - A&B's Flexible Fourier Form (eq. A.3), estimated by OLS.
* mean_abs_profile - the model-free average |R| profile (A&B Fig. 2), used as
                 the benchmark and as a robustness check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _basis(n: np.ndarray, N: int, P: int, dummies: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    """The bracketed term of eq. A.3 for interval indices n (1..N)."""
    N1, N2 = (N + 1) / 2, (N + 1) * (N + 2) / 6
    cols = [np.ones_like(n, float), n / N1, n ** 2 / N2]
    names = ["const", "n/N1", "n2/N2"]
    for d in dummies:
        cols.append((n == d).astype(float))
        names.append(f"I(n={d})")
    for p in range(1, P + 1):
        cols += [np.cos(2 * np.pi * p * n / N), np.sin(2 * np.pi * p * n / N)]
        names += [f"cos{p}", f"sin{p}"]
    return np.column_stack(cols), names


@dataclass
class FFF:
    N: int
    P: int
    J: int
    dummies: tuple[int, ...]
    beta: np.ndarray
    names: list[str]
    tstat: np.ndarray
    norm: float = 1.0           # constant making mean(s_hat)=1 on the fitting sample
    n_zero_dropped: int = 0
    r2: float = np.nan
    extra: dict = field(default_factory=dict)

    def _design(self, sigma: np.ndarray, n: np.ndarray) -> np.ndarray:
        B, _ = _basis(n, self.N, self.P, self.dummies)
        return np.hstack([B * (sigma[:, None] ** j) for j in range(self.J + 1)])

    def f_hat(self, sigma: pd.Series) -> pd.DataFrame:
        """Fitted log s^2 (+const) for each day in `sigma` and each interval."""
        n = np.arange(1, self.N + 1)
        T = len(sigma)
        X = self._design(np.repeat(sigma.to_numpy(float), self.N), np.tile(n, T))
        return pd.DataFrame((X @ self.beta).reshape(T, self.N), index=sigma.index,
                            columns=pd.RangeIndex(1, self.N + 1, name="n"))

    def s_hat(self, sigma: pd.Series) -> pd.DataFrame:
        """Periodic component, eq. A.4, using the *in-sample* normaliser."""
        return np.exp(self.f_hat(sigma) / 2) / self.norm


def fit_fff(R: pd.DataFrame, sigma: pd.Series, P: int, J: int = 0,
            dummies: tuple[int, ...] = (), method: str = "ppml") -> FFF:
    """Estimate the Flexible Fourier Form (eq. A.2-A.3).

    R       T x N percent returns
    sigma   daily volatility factor sigma_t (percent), aligned with R.index; it
            must be known at the start of day t (e.g. a GARCH one-step forecast).
    method  "ols"  - A&B's estimator: OLS of x = 2 log|R - Rbar| - log sigma^2 + log N
                     on the regressors. Exact-zero returns (log 0 = -inf) are
                     dropped. With discrete prices ~7% of 5-min returns are 0, and
                     dropping them biases s upward where zeros cluster.
            "ppml" - Poisson pseudo-ML for E[N R^2 / sigma^2] = exp(f): the same
                     multiplicative model, consistent whenever the conditional
                     mean is right, and zeros are ordinary observations
                     (Santos Silva & Tenreyro, 2006). Default.
    t-stats are heteroskedasticity-robust (sandwich) in both cases.
    """
    T, N = R.shape
    sig = sigma.reindex(R.index).to_numpy(float)
    dev = (R - np.nanmean(R.to_numpy())).to_numpy()
    n = np.tile(np.arange(1, N + 1), T)
    model = FFF(N, P, J, tuple(dummies), np.array([]), [], np.array([]))
    Xall = model._design(np.repeat(sig, N), n)
    _, base = _basis(np.arange(1, 3), N, P, tuple(dummies))
    model.names = [f"{b}*sig^{j}" if j else b for j in range(J + 1) for b in base]
    y2 = (N * dev ** 2 / sig[:, None] ** 2).ravel()        # E[y2] = s^2 under eq. 7

    if method == "ols":
        ok = (R.to_numpy().ravel() != 0) & np.isfinite(y2)
        X, y = Xall[ok], np.log(y2[ok])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        u = y - X @ beta
        bread = np.linalg.pinv(X.T @ X)
        cov = bread @ (X.T * u ** 2) @ X @ bread
        model.n_zero_dropped = int((~ok).sum())
    elif method == "ppml":
        ok = np.isfinite(y2)
        X, y = Xall[ok], y2[ok]
        beta = np.linalg.lstsq(X, np.log(y + y.mean() * 0.1), rcond=None)[0]   # starting value
        for _ in range(100):                                 # IRLS / Newton for Poisson QMLE
            mu = np.exp(X @ beta)
            step = np.linalg.solve((X.T * mu) @ X, X.T @ (y - mu))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-9:
                break
        mu = np.exp(X @ beta)
        bread = np.linalg.inv((X.T * mu) @ X)
        cov = bread @ (X.T * (y - mu) ** 2) @ X @ bread
    else:
        raise ValueError(method)
    model.beta = beta
    model.tstat = beta / np.sqrt(np.diag(cov))
    model.extra["method"] = method
    # fit criterion common to both methods: Gaussian quasi-likelihood of the
    # variance model, mean(y2/s2 + log s2) with s2 = exp(f) rescaled to the
    # data. Defined with zeros, comparable across methods and (P, J).
    f = Xall[np.isfinite(y2)] @ beta
    yy = y2[np.isfinite(y2)]
    s2 = np.exp(f) * yy.mean() / np.exp(f).mean()
    model.extra["qlik"] = float(np.mean(yy / s2 + np.log(s2)))
    model.extra["nobs"] = int(np.isfinite(y2).sum())
    model.r2 = float(np.corrcoef(np.exp(f / 2), np.sqrt(yy))[0, 1] ** 2)
    model.norm = float(np.exp(model.f_hat(sigma.reindex(R.index)) / 2).to_numpy().mean())
    return model


def mean_abs_profile(R: pd.DataFrame) -> pd.Series:
    """s_n estimated as average |R_n| relative to the overall average |R|."""
    a = R.abs().mean()
    return a / a.mean()


def bic_select(R: pd.DataFrame, sigma: pd.Series, P_grid=range(1, 9), J_grid=(0, 1),
               dummies: tuple[int, ...] = (), method: str = "ppml") -> pd.DataFrame:
    """BIC = n * quasi-loglik criterion + k log n for each (P, J).

    Used only on TRAINING data, to choose the FFF order without peeking at the
    test period.
    """
    rows = []
    for J in J_grid:
        for P in P_grid:
            m = fit_fff(R, sigma, P, J, dummies, method)
            n_obs, k = m.extra["nobs"], len(m.beta)
            rows.append({"P": P, "J": J, "k": k, "qlik": m.extra["qlik"],
                         "bic": n_obs * m.extra["qlik"] + k * np.log(n_obs)})
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
