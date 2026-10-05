"""MA(1)-GARCH(1,1) by Gaussian quasi-maximum likelihood (A&B section 4.2).

    R_t      = mu + theta * e_{t-1} + e_t
    h_t      = omega + alpha * e_{t-1}^2 + beta * h_{t-1},      E_{t-1}[e_t^2] = h_t

Both recursions are linear filters, so they are evaluated with
scipy.signal.lfilter instead of a Python loop - this is what makes fitting
1.6 million 5-minute returns feasible. Standard errors are the
Bollerslev-Wooldridge (1992) robust sandwich, as in the paper.

The `arch` package is not used because it offers AR but not MA mean dynamics.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

LOG2PI = np.log(2 * np.pi)


def _filter(params: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, theta, omega, alpha, beta = params
    # e_t = (y_t - mu) - theta * e_{t-1}
    e = lfilter([1.0], [1.0, theta], y - mu)
    e2 = e ** 2
    h0 = e2.mean()  # backcast initial variance
    # h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}
    drive = np.empty_like(e2)
    drive[0] = omega + alpha * h0
    drive[1:] = omega + alpha * e2[:-1]
    h = lfilter([1.0], [1.0, -beta], drive, zi=[beta * h0])[0]
    return e, h


def _loglik_obs(params: np.ndarray, y: np.ndarray) -> np.ndarray:
    e, h = _filter(params, y)
    h = np.maximum(h, 1e-12)
    return -0.5 * (LOG2PI + np.log(h) + e ** 2 / h)


# unconstrained <-> constrained: omega>0, alpha>=0, beta>=0, |theta|<1
def _to_params(z: np.ndarray, scale: float) -> np.ndarray:
    mu, th, w, a, b = z
    return np.array([mu, np.tanh(th), np.exp(w) * scale, np.exp(a), np.exp(b)])


def _from_params(p: np.ndarray, scale: float) -> np.ndarray:
    mu, th, w, a, b = p
    return np.array([mu, np.arctanh(th), np.log(w / scale), np.log(a), np.log(b)])


@dataclass
class GarchFit:
    params: pd.Series     # mu, theta, omega, alpha, beta
    se: pd.Series         # robust standard errors
    loglik: float
    nobs: int
    converged: bool

    @property
    def persistence(self) -> float:
        return float(self.params["alpha"] + self.params["beta"])

    def persistence_measures(self, minutes_per_obs: float) -> dict:
        """Half-life, mean lag, median lag (A&B Table 2 notes), in minutes."""
        a, b = self.params["alpha"], self.params["beta"]
        p = a + b
        out = {"half_life": np.inf, "mean_lag": np.inf, "median_lag": np.inf}
        if p < 1:
            out["half_life"] = -np.log(2) / np.log(p) * minutes_per_obs
            out["mean_lag"] = a / (1 - a - 2 * b + a * b + b ** 2) * minutes_per_obs
            if 2 * a + b < 1:
                out["median_lag"] = 0.5 * minutes_per_obs  # "less than 1/2"
            else:
                out["median_lag"] = (0.5 + (np.log(1 - b) - np.log(a) - np.log(2)) / np.log(p)) * minutes_per_obs
        return out


def fit_ma1_garch11(y: np.ndarray | pd.Series, starts: list[tuple[float, float]] | None = None) -> GarchFit:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    scale = y.var()
    best = None
    # a few starting values guard against local optima, a known issue at
    # frequencies where persistence is weak
    for a0, b0 in starts or [(0.05, 0.90), (0.15, 0.75), (0.30, 0.30)]:
        p0 = np.array([y.mean(), 0.0, scale * (1 - a0 - b0), a0, b0])
        z0 = _from_params(p0, scale)
        nll = lambda z: -_loglik_obs(_to_params(z, scale), y).sum() / len(y)
        # Finite-difference gradients can step to the edge of the parameter
        # space (tanh -> +-1, h -> 0), where the loglik is inf/nan; L-BFGS-B
        # rejects such steps. The resulting RuntimeWarning is benign noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = minimize(nll, z0, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    p = _to_params(best.x, scale)
    se = _robust_se(p, y)
    names = ["mu", "theta", "omega", "alpha", "beta"]
    return GarchFit(pd.Series(p, names), pd.Series(se, names),
                    float(-best.fun * len(y)), len(y), bool(best.success))


def _robust_se(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sandwich covariance A^-1 B A^-1 from numerical derivatives."""
    k = len(p)
    step = 1e-4 * np.maximum(np.abs(p), 1e-3)
    # per-observation scores by central differences
    S = np.empty((len(y), k))
    for i in range(k):
        d = np.zeros(k); d[i] = step[i]
        S[:, i] = (_loglik_obs(p + d, y) - _loglik_obs(p - d, y)) / (2 * step[i])
    B = S.T @ S
    # Hessian of total log-likelihood
    L = lambda q: _loglik_obs(q, y).sum()
    A = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            di = np.zeros(k); di[i] = step[i]
            dj = np.zeros(k); dj[j] = step[j]
            A[i, j] = A[j, i] = (L(p + di + dj) - L(p + di - dj) - L(p - di + dj) + L(p - di - dj)) / (4 * step[i] * step[j])
    try:
        Ainv = np.linalg.inv(A)
        cov = Ainv @ B @ Ainv
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def conditional_sd(fit: GarchFit, y: pd.Series) -> pd.Series:
    """One-step-ahead conditional s.d. h_t^{1/2}: uses information up to t-1 only."""
    if y.isna().any():
        raise ValueError("NaN in input: the GARCH recursion would propagate it to every later day")
    _, h = _filter(fit.params.to_numpy(), y.to_numpy(float))
    return pd.Series(np.sqrt(h), index=y.index)


def aggregate(R: pd.DataFrame, k: int) -> np.ndarray:
    """Non-overlapping k-interval returns R^k_{t,n} within each day, chronologically flattened."""
    T, N = R.shape
    if N % k:
        raise ValueError(f"k={k} does not divide N={N}")
    return R.to_numpy().reshape(T, N // k, k).sum(axis=2).ravel()


def divisors(N: int) -> list[int]:
    return [k for k in range(1, N + 1) if N % k == 0 and k < N]


def _fit_one(args):
    y, k, bar_minutes = args
    fit = fit_ma1_garch11(y)
    row = {"k": k, "T/k": fit.nobs, **fit.params.to_dict(),
           **{f"se_{c}": v for c, v in fit.se.to_dict().items()},
           "persistence": fit.persistence, "converged": fit.converged}
    row.update(fit.persistence_measures(minutes_per_obs=k * bar_minutes))
    return row


def garch_by_frequency(R: pd.DataFrame, bar_minutes: int = 5, ks: list[int] | None = None,
                       workers: int = 8) -> pd.DataFrame:
    """MA(1)-GARCH(1,1) on non-overlapping k-interval returns for each k (A&B Tables 2, 4, 5).

    Returns are concatenated across days in calendar order, as in the paper
    (the overnight/weekend gap is simply skipped, not modelled).
    """
    from concurrent.futures import ProcessPoolExecutor

    ks = ks or divisors(R.shape[1])
    jobs = [(aggregate(R, k), k, bar_minutes) for k in ks]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_fit_one, jobs))
    return pd.DataFrame(rows).set_index("k")


def ex_ante_daily_sigma(daily: pd.Series, train_end_year: int) -> tuple[pd.Series, "GarchFit"]:
    """sigma_t for every day using parameters estimated on years <= train_end_year.

    The recursion only uses returns up to t-1, and the parameters only use the
    training period, so sigma_t is a genuine ex-ante forecast for test days.
    """
    fit = fit_ma1_garch11(daily[daily.index.year <= train_end_year])
    return conditional_sd(fit, daily), fit
