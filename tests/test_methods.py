"""Recovery tests on simulated data with known truth."""
import numpy as np
import pandas as pd
import pytest

from intraday.garch import aggregate, divisors, fit_ma1_garch11, GarchFit
from intraday.periodicity import fit_fff, mean_abs_profile

RNG = np.random.default_rng(0)


def simulate_ab(T=400, N=80, J_effect=0.0):
    """A&B eq. 7 with a U-shaped s_n and a GARCH-like daily sigma_t."""
    n = np.arange(1, N + 1)
    log_s2 = 0.8 * np.cos(2 * np.pi * n / N) + 0.3 * np.cos(4 * np.pi * n / N)
    sigma = np.exp(RNG.normal(0, 0.3, T).cumsum() * 0.1)  # persistent daily vol
    sigma = pd.Series(sigma / sigma.mean(), index=pd.RangeIndex(T))
    tilt = J_effect * (sigma.to_numpy()[:, None] - 1) * n / N  # shape depends on sigma
    s = np.exp((log_s2[None, :] + tilt) / 2)
    s /= s.mean()
    Z = RNG.standard_normal((T, N))
    R = pd.DataFrame(sigma.to_numpy()[:, None] * s * Z / np.sqrt(N), columns=pd.RangeIndex(1, N + 1))
    return R, sigma, s


def test_fff_recovers_periodic_component():
    R, sigma, s = simulate_ab()
    m = fit_fff(R, sigma, P=2, J=0)
    s_hat = m.s_hat(sigma).to_numpy()
    assert abs(s_hat.mean() - 1) < 1e-10                    # eq. A.4 normalisation
    assert np.corrcoef(s_hat[0], s[0])[0, 1] > 0.99
    assert np.max(np.abs(s_hat[0] / s[0] - 1)) < 0.08


def test_fff_interaction_term_detects_shape_change():
    R, sigma, s = simulate_ab(T=1500, J_effect=3.0)
    m = fit_fff(R, sigma, P=2, J=1)
    s_hat = m.s_hat(sigma).to_numpy()
    assert np.corrcoef(s_hat.ravel(), s.ravel())[0, 1] > 0.97


def test_mean_abs_profile_matches_truth():
    R, sigma, s = simulate_ab(T=2000)
    prof = mean_abs_profile(R).to_numpy()
    assert np.corrcoef(prof, s[0])[0, 1] > 0.99


def simulate_garch(n, mu=0.0, theta=-0.05, omega=0.05, alpha=0.08, beta=0.9):
    e = np.empty(n); h = np.empty(n); y = np.empty(n)
    h_prev, e_prev = omega / (1 - alpha - beta), 0.0
    for t in range(n):
        h[t] = omega + alpha * e_prev ** 2 + beta * h_prev
        e[t] = np.sqrt(h[t]) * RNG.standard_normal()
        y[t] = mu + theta * e_prev + e[t]
        h_prev, e_prev = h[t], e[t]
    return y


def test_garch_recovers_parameters():
    y = simulate_garch(30_000)
    fit = fit_ma1_garch11(y)
    p, se = fit.params, fit.se
    for name, true in [("theta", -0.05), ("alpha", 0.08), ("beta", 0.9)]:
        assert abs(p[name] - true) < 4 * se[name] + 0.01, (name, p[name], se[name])
    assert (se > 0).all()


def test_persistence_measures_formulas():
    fit = GarchFit(pd.Series({"mu": 0, "theta": 0, "omega": 1, "alpha": 0.1, "beta": 0.85}),
                   pd.Series(dtype=float), 0.0, 1, True)
    m = fit.persistence_measures(minutes_per_obs=1)
    assert m["half_life"] == pytest.approx(np.log(0.5) / np.log(0.95))
    # mean lag from the MA(inf) weights theta_i = alpha*(alpha+beta)^(i-1), theta_0=1
    i = np.arange(1, 5000)
    w = np.r_[1, 0.1 * 0.95 ** (i - 1)]
    assert m["mean_lag"] == pytest.approx((np.r_[0, i] * w).sum() / w.sum(), rel=1e-6)


def test_aggregation_and_divisors():
    R = pd.DataFrame(np.arange(12, dtype=float).reshape(2, 6))
    assert aggregate(R, 3).tolist() == [3, 12, 21, 30]
    assert len(divisors(288)) == 17 and len(divisors(80)) == 9   # as in A&B Table 1


def test_ppml_handles_zero_returns_and_ols_is_biased():
    """Discretise returns so many become exactly 0 (as with tick-size prices).

    PPML must still recover s; A&B's log-OLS (zeros dropped) is distorted.
    """
    R, sigma, s = simulate_ab(T=1500)
    tick = 0.6 * R.abs().to_numpy().mean()
    Rd = (R / tick).round() * tick
    assert (Rd == 0).to_numpy().mean() > 0.2
    ppml = fit_fff(Rd, sigma, P=2, J=0, method="ppml").s_hat(sigma).to_numpy()[0]
    ols = fit_fff(Rd, sigma, P=2, J=0, method="ols").s_hat(sigma).to_numpy()[0]
    err_ppml = np.max(np.abs(ppml / s[0] - 1))
    err_ols = np.max(np.abs(ols / s[0] - 1))
    assert err_ppml < 0.08
    assert err_ols > 2 * err_ppml
