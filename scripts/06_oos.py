"""Step 6: strict chronological out-of-sample tests.

Splits (config.py):  train <= train_end  <  validation <= valid_end  <  test
  EUR/USD, S&P 500 : train ..2016 | validation 2017-2019 | test 2020-2025
  BTC/USDT         : train ..2021 | validation 2022      | test 2023-2025

H6  Volatility forecasting (walk-forward, re-estimated every January using only
    data from earlier years). The FFF order (P, J) is chosen ONCE, on the
    validation years, from models fitted on the training years; the test years
    are then touched exactly once.
H5  Mean-return seasonality as a trading signal, designed on training data,
    evaluated on the test years after transaction costs.

Look-ahead audit (every forecast for day t / interval n uses):
  sigma_t      daily GARCH params from years < Y, recursion through day t-1
  profile/FFF  days from years < Y
  intraday h   GARCH params from the 3 years before Y, recursion through n-1
  trading sign training-period statistics, or returns on days < t
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intraday.bars import get_panel
from intraday.config import ASSETS, TABLES
from intraday.garch import _filter, conditional_sd, fit_ma1_garch11
from intraday.periodicity import fit_fff
from intraday.plots import ASSET_COLOR, INK2, MUTED, SERIES, save, setup
from intraday.stats import fdr_bh, newey_west_tstat
from scipy import stats as st

setup()
NAME = {"eurusd": "EUR/USD", "spx": "S&P 500", "btc": "BTC/USDT"}
# Fixed-interval dummies chosen from institutional knowledge, not from the data:
# S&P - A&B's three post-cash-close intervals; EUR/USD - the 08:30 NY US data
# release interval (ends 08:35 NY = interval 187 of the 17:00-start day).
DUMMIES = {"eurusd": (187,), "spx": (78, 79, 80), "btc": ()}
MODELS = ["M0 flat", "M1 hist. profile", "M2 FFF", "M3 FFF+intraday GARCH", "M4 raw intraday GARCH"]
# Round-trip transaction cost per trade, % of notional (low / base / high)
COSTS = {"eurusd": (0.003, 0.006, 0.012), "spx": (0.004, 0.0075, 0.015), "btc": (0.02, 0.10, 0.20)}
GARCH_YEARS = 3


def qlike(r2, v):
    """Gaussian quasi-likelihood loss r2/v + log v: QLIKE up to terms free of v,
    robust to noise in the r2 proxy (Patton 2011) and defined when r2 = 0."""
    return r2 / v + np.log(v)


# ------------------------------------------------------------------ forecasts
def daily_sigma(P, year):
    fit = fit_ma1_garch11(P.daily[P.daily.index.year < year])
    return conditional_sd(fit, P.daily)


def choose_fff_order(P):
    a = P.asset
    sig = daily_sigma(P, a.train_end + 1)
    tr = P.R[P.R.index.year <= a.train_end]
    va = P.R[(P.R.index.year > a.train_end) & (P.R.index.year <= a.valid_end)]
    rows = []
    for J in (0, 1):
        for Pn in (1, 2, 3, 4, 6, 8):
            m = fit_fff(tr, sig.reindex(tr.index), P=Pn, J=J, dummies=DUMMIES[a.key], method="ppml")
            sv = sig.reindex(va.index)
            v = (sv.to_numpy()[:, None] ** 2) * np.exp(m.f_hat(sv).to_numpy()) / va.shape[1]
            rows.append({"P": Pn, "J": J, "valid_qlike": float(np.mean(qlike(va.to_numpy() ** 2, v)))})
    t = pd.DataFrame(rows).sort_values("valid_qlike")
    t.to_csv(TABLES / f"oos_fff_order_selection_{a.key}.csv", index=False, float_format="%.6f")
    return int(t.iloc[0].P), int(t.iloc[0].J)


def intraday_garch_h(series_train: np.ndarray, series_all: np.ndarray, n_test: int) -> np.ndarray:
    fit = fit_ma1_garch11(series_train)
    _, h = _filter(fit.params.to_numpy(), series_all)
    return h[-n_test:]


def forecasts_for_year(P, year, order):
    """All model forecasts for the days of `year`, using only earlier years."""
    R, N = P.R, P.N
    sig = daily_sigma(P, year)
    past = R[R.index.year < year]
    cur = R[R.index.year == year]
    sp, sc = sig.reindex(past.index).to_numpy(), sig.reindex(cur.index).to_numpy()
    ratio = past.to_numpy() ** 2 / sp[:, None] ** 2
    out = {}
    out["M0 flat"] = (sc[:, None] ** 2) * ratio.mean() * np.ones((1, N))
    out["M1 hist. profile"] = (sc[:, None] ** 2) * ratio.mean(axis=0)[None, :]
    m = fit_fff(past, sig.reindex(past.index), P=order[0], J=order[1], dummies=DUMMIES[P.asset.key], method="ppml")
    f_cur = m.f_hat(sig.reindex(cur.index)).to_numpy()
    out["M2 FFF"] = (sc[:, None] ** 2) * np.exp(f_cur) / N
    # intraday GARCH, estimated on the last GARCH_YEARS years, run through the test year
    recent = R[(R.index.year < year) & (R.index.year >= year - GARCH_YEARS)]
    s_recent = m.s_hat(sig.reindex(recent.index)).to_numpy()
    s_cur = m.s_hat(sig.reindex(cur.index)).to_numpy()
    filt_tr = (recent.to_numpy() / s_recent).ravel()
    filt_all = np.r_[filt_tr, (cur.to_numpy() / s_cur).ravel()]
    h = intraday_garch_h(filt_tr, filt_all, cur.size)
    out["M3 FFF+intraday GARCH"] = s_cur ** 2 * h.reshape(cur.shape)
    raw_tr = recent.to_numpy().ravel()
    h_raw = intraday_garch_h(raw_tr, np.r_[raw_tr, cur.to_numpy().ravel()], cur.size)
    out["M4 raw intraday GARCH"] = h_raw.reshape(cur.shape)
    return cur, out


def evaluate_forecasts(key):
    P = get_panel(key)
    a = P.asset
    order = choose_fff_order(P)
    years = sorted(y for y in set(P.R.index.year) if y > a.valid_end)
    daily_losses, risk_z = [], {mname: [] for mname in MODELS}
    for y in years:
        cur, fc = forecasts_for_year(P, y, order)
        r2 = cur.to_numpy() ** 2
        for mname, v in fc.items():
            v = np.maximum(v, 1e-12)
            daily_losses.append(pd.DataFrame({"day": cur.index, "model": mname,
                                              "qlike": qlike(r2, v).mean(axis=1),
                                              "mse": ((r2 - v) ** 2).mean(axis=1)}))
            risk_z[mname].append(pd.DataFrame(cur.to_numpy() / np.sqrt(v), index=cur.index, columns=cur.columns))
        print(key, y, flush=True)
    L = pd.concat(daily_losses)
    L.to_csv(TABLES / f"oos_daily_losses_{key}.csv", index=False, float_format="%.6g")
    Z = {mname: pd.concat(z) for mname, z in risk_z.items()}
    return P, order, L, Z


def summarise(key, L, Z, order):
    piv = {c: L.pivot(index="day", columns="model", values=c)[MODELS] for c in ("qlike", "mse")}
    rows = []
    for mname in MODELS:
        row = {"asset": key, "model": mname, "P": order[0], "J": order[1], "test_days": len(piv["qlike"])}
        for c in ("qlike", "mse"):
            row[f"mean_{c}"] = piv[c][mname].mean()
            for bench in ("M0 flat", "M1 hist. profile"):
                d = piv[c][mname] - piv[c][bench]
                row[f"{c}_diff_vs_{bench.split()[0]}"] = d.mean()
                row[f"{c}_DM_t_vs_{bench.split()[0]}"] = newey_west_tstat(d.to_numpy()) if mname != bench else np.nan
        if mname == "M3 FFF+intraday GARCH":   # key pair: periodicity-filtered vs raw intraday GARCH
            for c in ("qlike", "mse"):
                d = piv[c][mname] - piv[c]["M4 raw intraday GARCH"]
                row[f"{c}_diff_vs_M4"] = d.mean()
                row[f"{c}_DM_t_vs_M4"] = newey_west_tstat(d.to_numpy())
        z = Z[mname]
        sd_n = z.std()                                  # realised risk per unit of forecast risk, by interval
        row["risk_sd_overall"] = float(np.nanstd(z.to_numpy()))
        row["risk_mean_abs_log_sd_by_interval"] = float(np.mean(np.abs(np.log(sd_n / np.nanstd(z.to_numpy())))))
        row["risk_share_intervals_off_by_25pct"] = float(np.mean(np.abs(sd_n / np.nanstd(z.to_numpy()) - 1) > 0.25))
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / f"oos_forecast_summary_{key}.csv", index=False, float_format="%.5f")
    return t


# ------------------------------------------------------------------ H5 trading
def trading(key):
    P = get_panel(key)
    a, R = P.asset, P.R
    tr = R[R.index.year <= a.train_end]
    te = R[R.index.year > a.valid_end]
    va = R[(R.index.year > a.train_end) & (R.index.year <= a.valid_end)]
    t = np.array([newey_west_tstat(tr[c].to_numpy()) for c in R.columns])
    disc = fdr_bh(2 * st.norm.sf(np.abs(t)), 0.05)
    sign = pd.Series(np.where(disc, np.sign(t), 0.0), index=R.columns)
    res, curves = [], {}

    def pnl_fdr(X):
        return (X * sign).sum(axis=1), int((sign != 0).sum())   # daily gross P&L (%), trades/day

    def pnl_hks(X_all, X):
        # same-interval momentum: sign of the mean return at interval n over the previous 20 days
        sig = np.sign(X_all.rolling(20).mean().shift(1)).reindex(X.index)
        return (X * sig).sum(axis=1), X.shape[1]

    for strat in ["FDR intervals", "HKS same-interval momentum"]:
        for lab, X in [("train", tr), ("validation", va), ("test", te)]:
            if strat == "FDR intervals":
                g, n_tr = pnl_fdr(X)
            else:
                g, n_tr = pnl_hks(R, X)
                g = g.dropna()
            for cost_lab, c in zip(["low", "base", "high"], COSTS[key]):
                net = g - n_tr * c
                ann = np.sqrt(252 if key != "btc" else 365)
                res.append({"asset": key, "strategy": strat, "sample": lab, "cost": cost_lab, "cost_pct_rt": c,
                            "trades_per_day": n_tr, "days": len(g),
                            "gross_mean_daily_pct": g.mean(), "gross_sharpe": g.mean() / g.std() * ann if g.std() > 0 else np.nan,
                            "gross_t_NW": newey_west_tstat(g.to_numpy()) if g.std() > 0 else np.nan,
                            "net_mean_daily_pct": net.mean(), "net_sharpe": net.mean() / net.std() * ann if net.std() > 0 else np.nan,
                            "breakeven_cost_pct_rt": g.mean() / n_tr if n_tr else np.nan})
            curves[(strat, lab)] = g.cumsum()
            if lab == "test":
                curves[(strat, "breakeven")] = g.mean() / n_tr if n_tr else np.nan
    # where does the same-interval momentum P&L come from? (test, gross, by local hour)
    sig_all = np.sign(R.rolling(20).mean().shift(1))
    g = (te * sig_all.reindex(te.index)).sum()
    start = int(a.day_start[:2]) * 60 + int(a.day_start[3:])
    hour = ((start + 5 * (g.index.to_numpy() - 1)) // 60) % 24
    g.groupby(hour).sum().rename("test_gross_pnl_pct").to_csv(TABLES / f"oos_hks_pnl_by_hour_{key}.csv",
                                                               float_format="%.4f")
    return pd.DataFrame(res), curves, int(disc.sum())


# ------------------------------------------------------------------ figures
def fig_forecasts(summ, all_L):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, (key, t) in zip(axes, summ.items()):
        t = t.set_index("model").loc[MODELS]
        rel = 100 * (t["mean_qlike"] - t.loc["M0 flat", "mean_qlike"]) / abs(t.loc["M0 flat", "mean_qlike"])
        L = all_L[key].pivot(index="day", columns="model", values="qlike")
        se = [100 * (L[m] - L["M0 flat"]).std() / np.sqrt(len(L)) / abs(t.loc["M0 flat", "mean_qlike"]) * 1.96
              for m in MODELS]
        colors = [MUTED, SERIES[3], SERIES[0], SERIES[1], SERIES[7]]
        ax.barh(range(len(MODELS)), rel, xerr=se, color=colors, height=0.6, error_kw={"lw": 0.8, "ecolor": INK2})
        ax.set_yticks(range(len(MODELS)), MODELS if key == "eurusd" else [""] * len(MODELS))
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(f"{NAME[key]} (test {int(t['test_days'].iloc[0])} days)")
        ax.set_xlabel("QLIKE vs M0 flat (%), lower = better")
    fig.suptitle("Out-of-sample 5-minute volatility forecasts, walk-forward, 95% CI", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "oos_forecast_qlike")


def fig_risk(Zs):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, Z) in zip(axes, Zs.items()):
        for mname, c in [("M0 flat", MUTED), ("M1 hist. profile", SERIES[3]), ("M2 FFF", SERIES[0])]:
            sd = Z[mname].std()
            ax.plot(sd.index, sd / np.nanstd(Z[mname].to_numpy()), color=c, lw=1.1, label=mname)
        ax.axhline(1, color=INK2, lw=0.6, ls="--")
        ax.set_title(f"{NAME[key]}: realised risk / target, test")
        ax.set_xlabel("Interval n")
    axes[0].set_ylabel("sd of R / forecast sd (1 = on target)")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "oos_risk_targeting")


def fig_trading(curves_all):
    """GROSS cumulative returns (net curves are straight lines: costs dominate).
    Titles compare the test-period break-even cost with the assumed base cost."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8))
    for j, (key, curves) in enumerate(curves_all.items()):
        for i, strat in enumerate(["FDR intervals", "HKS same-interval momentum"]):
            ax = axes[i, j]
            offset, drawn = 0.0, False
            for lab, c in [("train", MUTED), ("validation", SERIES[3]), ("test", ASSET_COLOR[key])]:
                cv = curves.get((strat, lab))
                if cv is None or len(cv) == 0 or not np.isfinite(cv.iloc[-1]) or cv.abs().max() == 0:
                    continue
                ax.plot(cv.index, cv + offset, color=c, lw=1.1, label=lab)
                offset += cv.iloc[-1]
                drawn = True
            if not drawn:
                ax.text(0.5, 0.5, "no intervals survive FDR control\nin training: nothing to trade",
                        ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            be = curves.get((strat, "breakeven"), np.nan)
            sub = (f"break-even {be * 1e4:.2f} bp vs cost {COSTS[key][1] * 1e4:.1f} bp per round trip"
                   if np.isfinite(be) else "")
            ax.set_title(f"{NAME[key]}: {strat}\n{sub}", fontsize=8.5)
            ax.axhline(0, color=INK2, lw=0.5)
        axes[i, 0].set_ylabel("Cumulative GROSS return (%)")
    axes[0, 0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "oos_trading_is_vs_oos")


def main():
    summ, Ls, Zs, trade_tabs, curves_all = {}, {}, {}, [], {}
    for key in ASSETS:
        P, order, L, Z = evaluate_forecasts(key)
        summ[key] = summarise(key, L, Z, order)
        Ls[key], Zs[key] = L, Z
        print(summ[key][["model", "mean_qlike", "qlike_diff_vs_M0", "qlike_DM_t_vs_M0", "qlike_DM_t_vs_M1",
                         "risk_mean_abs_log_sd_by_interval"]].to_string(), flush=True)
        t, curves, ndisc = trading(key)
        trade_tabs.append(t.assign(fdr_intervals=ndisc))
        curves_all[key] = curves
    pd.concat(summ.values()).to_csv(TABLES / "oos_forecast_summary_all.csv", index=False, float_format="%.5f")
    tt = pd.concat(trade_tabs)
    tt.to_csv(TABLES / "oos_trading.csv", index=False, float_format="%.5f")
    print(tt[(tt.cost == "base")][["asset", "strategy", "sample", "trades_per_day", "gross_sharpe", "gross_t_NW",
                                   "net_sharpe", "breakeven_cost_pct_rt"]].to_string())
    fig_forecasts(summ, Ls)
    fig_risk(Zs)
    fig_trading(curves_all)
    print("done")


if __name__ == "__main__":
    main()
