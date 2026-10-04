"""Step 3: replicate A&B (1997) sections 2-5 on modern data.

Headline windows are fixed ex ante to mirror the paper's sample lengths
(chosen before any result was seen, to avoid picking a flattering window):
  EUR/USD  2024-10-01 .. 2025-09-30   (A&B: one year, Oct 1992 - Sep 1993)
  S&P 500  2022-01-01 .. 2025-12-31   (A&B: four years, 1986 - 1989)
Every rolling window is analysed in 05_robustness.py.

Outputs: tables/rep_*.csv, figures/rep_*.{png,pdf}, results/replication_fff.pkl
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intraday.bars import get_panel
from intraday.config import RESULTS, ROOT, TABLES
from intraday.garch import conditional_sd, divisors, fit_ma1_garch11, garch_by_frequency
from intraday.periodicity import fit_fff, mean_abs_profile
from intraday.plots import ASSET_COLOR, INK2, MUTED, SERIES, note, save, setup
from intraday.stats import acf, interval_mean_ci, table1_row, block_bootstrap_ci

setup()
TABLES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SPEC = {
    # A&B Appendix B choices, kept identical for the replication
    "eurusd": dict(window=("2024-10-01", "2025-09-30"), P=6, J=0, dummies=(), market="fx"),
    "spx": dict(window=("2022-01-01", "2025-12-31"), P=2, J=1, dummies=(78, 79, 80), market="equity"),
}
REF = pd.read_csv(ROOT / "reference" / "ab1997_values.csv")


def tick_labels(key, N):
    """(interval, clock label) pairs; labels give the END time of the interval."""
    start = {"eurusd": 17 * 60, "spx": 9 * 60 + 35, "btc": 0}[key]
    every = 36 if N > 100 else 12
    return [(n, f"{(start + 5 * n) // 60 % 24:02d}:{(start + 5 * n) % 60:02d}") for n in range(every, N + 1, 2 * every)]


def label_axis(ax, key, N):
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xlabel("Five-minute interval n  (top axis: " + ("UTC" if key == "btc" else "New York time") + ")")
    sec = ax.secondary_xaxis("top")
    t = tick_labels(key, N)
    sec.set_xticks([a for a, _ in t], [b for _, b in t], fontsize=7.5, color=MUTED)
    sec.tick_params(length=0)
    sec.spines["top"].set_visible(False)


def name(key):
    return {"eurusd": "EUR/USD", "spx": "S&P 500", "btc": "BTC/USDT"}[key]


def daily_garch_table(daily: pd.Series, max_k: int = 10) -> pd.DataFrame:
    """A&B Table 3: MA(1)-GARCH(1,1) on non-overlapping k-day returns; measures in days."""
    rows = []
    for k in range(1, max_k + 1):
        x = daily.to_numpy()
        f = fit_ma1_garch11(x[: len(x) // k * k].reshape(-1, k).sum(axis=1))
        rows.append({"k_days": k, "T/k": f.nobs, "alpha": f.params.alpha, "beta": f.params.beta,
                     "persistence": f.persistence, **f.persistence_measures(minutes_per_obs=k)})
    return pd.DataFrame(rows).set_index("k_days")


def main():
    out = {}
    for key, spec in SPEC.items():
        P = get_panel(key)
        R = P.R.loc[spec["window"][0]:spec["window"][1]]
        N = R.shape[1]
        lab = P.asset.label

        # ---- daily volatility factor sigma_t: one-step-ahead daily MA(1)-GARCH(1,1)
        daily_hist = P.daily.loc[:spec["window"][1]]
        dfit = fit_ma1_garch11(daily_hist)
        sigma = conditional_sd(dfit, daily_hist).reindex(R.index)

        # ---- periodic component (Appendix B)
        fff = fit_fff(R, sigma, P=spec["P"], J=spec["J"], dummies=spec["dummies"], method="ppml")
        fff_ols = fit_fff(R, sigma, P=spec["P"], J=spec["J"], dummies=spec["dummies"], method="ols")
        s_hat = fff.s_hat(sigma)
        Rf = R / s_hat                                   # filtered (Table 4)
        Rs = R / s_hat.mul(sigma, axis=0)                # standardized (Table 5)
        pd.DataFrame({"ppml_coef": fff.beta, "ppml_robust_t": fff.tstat,
                      "ols_coef": fff_ols.beta, "ols_robust_t": fff_ols.tstat}, index=fff.names).to_csv(
            TABLES / f"rep_fff_coefficients_{key}.csv", float_format="%.4f")
        Rf_ols = R / fff_ols.s_hat(sigma)

        # ---- Table 1
        t1 = pd.DataFrame([table1_row(R, k) for k in divisors(N)]).set_index("k")
        t1.to_csv(TABLES / f"rep_table1_{key}.csv", float_format="%.4f")

        # ---- Tables 2, 4, 5: intraday GARCH at all frequencies; Table 3: daily
        t2 = garch_by_frequency(R)
        t4 = garch_by_frequency(Rf)
        t5 = garch_by_frequency(Rs)
        t4_ols = garch_by_frequency(Rf_ols)
        t4_ols.to_csv(TABLES / f"rep_table4_filtered_ols_{key}.csv", float_format="%.4f")
        for fname, t in [("table2_raw", t2), ("table4_filtered", t4), ("table5_standardized", t5)]:
            t.to_csv(TABLES / f"rep_{fname}_{key}.csv", float_format="%.4f")
        t3 = daily_garch_table(daily_hist)
        t3.to_csv(TABLES / f"rep_table3_daily_{key}.csv", float_format="%.4f")
        out[key] = dict(R=R, sigma=sigma, fff=fff, fff_ols=fff_ols, t4_ols=t4_ols, s_hat=s_hat, Rf=Rf, Rs=Rs, t1=t1, t2=t2, t3=t3, t4=t4, t5=t5,
                        daily_fit=dfit)
        print(lab, "zero-return share", round(float((R == 0).to_numpy().mean()), 4),
              "| OLS drops", fff_ols.n_zero_dropped)
        print(pd.DataFrame({"raw": t2.persistence, "filtered": t4.persistence, "filtered_ols": t4_ols.persistence,
                            "standardized": t5.persistence}).round(3))

    with open(RESULTS / "replication.pkl", "wb") as f:
        pickle.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()}, f)

    # =============================================================== figures
    keys = list(SPEC)
    win = {k: f"{SPEC[k]['window'][0][:7]} to {SPEC[k]['window'][1][:7]}" for k in keys}

    # Fig 1: average returns with A&B's constant iid band and interval-specific band
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    for ax, key in zip(axes, keys):
        ci = interval_mean_ci(out[key]["R"])
        n = ci.index
        ax.fill_between(n, ci.lo, ci.hi, color=ASSET_COLOR[key], alpha=0.15, lw=0, label="95% band, interval-specific variance")
        ax.plot(n, ci["mean"], color=ASSET_COLOR[key], lw=1.2, label="Mean return")
        ax.plot(n, ci.iid_lo, color=MUTED, lw=0.8, ls="--", label="95% band, iid null (A&B)")
        ax.plot(n, ci.iid_hi, color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=INK2, lw=0.6)
        out_i = ((ci["mean"] < ci.iid_lo) | (ci["mean"] > ci.iid_hi)).sum()
        out_s = ((ci.lo > 0) | (ci.hi < 0)).sum()
        ax.set_title(f"{'ab'[keys.index(key)]})  {name(key)}: "
                     f"{out_i} iid-band / {out_s} specific-band violations of {len(ci)} (5% expected: {0.05*len(ci):.0f})")
        ax.set_ylabel("Average return (%)")
        label_axis(ax, key, len(ci))
    axes[0].legend(loc="upper left", ncol=3, fontsize=7.5)
    fig.tight_layout()
    save(fig, "rep_fig1_mean_returns")

    # Fig 2 + 6: average absolute returns with block-bootstrap CI and the FFF fit
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    for ax, key in zip(axes, keys):
        R = out[key]["R"]
        ci = block_bootstrap_ci(R, lambda d: d.abs().mean().to_numpy(), n_boot=300)
        fitted = out[key]["s_hat"].mean() * R.abs().to_numpy().mean()
        n = ci.index
        ax.fill_between(n, ci.lo, ci.hi, color=ASSET_COLOR[key], alpha=0.2, lw=0, label="95% CI (20-day block bootstrap)")
        ax.plot(n, ci.est, color=ASSET_COLOR[key], lw=1.3, label="Average |R|")
        ax.plot(n, fitted, color=INK2, lw=1.1, ls="--", label="Flexible Fourier Form fit")
        ax.set_ylabel("Average |return| (%)")
        ax.set_ylim(0, None)
        ax.set_title(f"{'ab'[keys.index(key)]})  {name(key)}, {win[key]}")
        label_axis(ax, key, len(n))
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "rep_fig2_volatility_profile")

    # Fig 3 + 4: five-day correlograms of returns and absolute returns
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex="col")
    for j, key in enumerate(keys):
        R = out[key]["R"]
        N = R.shape[1]
        x = R.to_numpy().ravel()
        lags = np.arange(1, 5 * N + 1) / N
        band = 1.96 / np.sqrt(len(x))
        for i, (series, ttl) in enumerate([(x, "returns"), (np.abs(x), "absolute returns")]):
            ax = axes[i, j]
            ax.plot(lags, acf(series, 5 * N), color=ASSET_COLOR[key], lw=0.8)
            ax.axhspan(-band, band, color=MUTED, alpha=0.15, lw=0)
            ax.axhline(0, color=INK2, lw=0.5)
            ax.set_title(f"{name(key)}: {ttl}")
            if i == 1:
                ax.set_xlabel("Lag (trading days)")
            if j == 0:
                ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    save(fig, "rep_fig3_4_correlograms")

    # Fig 5: 40-day correlogram of intraday |R| vs the daily |R| autocorrelation
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key in zip(axes, keys):
        R = out[key]["R"]
        N = R.shape[1]
        a = acf(np.abs(R.to_numpy().ravel()), 40 * N)
        ax.plot(np.arange(1, 40 * N + 1) / N, a, color=ASSET_COLOR[key], lw=0.5, label=f"Five-minute |R|")
        P = get_panel(key)
        d = acf(np.abs(P.daily.to_numpy()), 40)
        ax.plot(np.arange(1, 41), d, "o-", color=INK2, ms=3, lw=1, label="Daily |R| (full history)")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(f"{name(key)}: 40-day correlogram")
        ax.set_xlabel("Lag (trading days)")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Autocorrelation")
    fig.tight_layout()
    save(fig, "rep_fig5_daily_vs_intraday")

    # Persistence across sampling frequencies: this study vs A&B (Tables 2, 4, 5)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, key in zip(axes, keys):
        ref = REF[REF.market == SPEC[key]["market"]].set_index("k")
        for (col, tab, ls, lbl) in [("persist_raw", "t2", "-", "Raw"), ("persist_filtered", "t4", "-", "Filtered R/s"),
                                    ("persist_standardized", "t5", "-", "Standardized R/(sigma s)")]:
            c = {"t2": SERIES[0], "t4": SERIES[1], "t5": SERIES[2]}[tab]
            t = out[key][tab]
            ax.plot(t.index, t.persistence.clip(upper=1.05), "o-", color=c, ms=4, lw=1.4, label=f"{lbl}: this study")
            ax.plot(ref.index, ref[col].clip(upper=1.05), "s--", color=c, ms=3.5, lw=0.9, mfc="none", alpha=0.85,
                    label=f"{lbl}: A&B 1997")
        ax.set_xscale("log")
        ticks = list(out[key]["t2"].index)
        ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=7)
        ax.set_xlabel("Return interval k (x 5 minutes)")
        ax.set_title(f"{('EUR/USD vs DM-$' if key=='eurusd' else 'S&P 500 (2022-25 vs 1986-89)')}")
        ax.axhline(1, color=MUTED, lw=0.6, ls=":")
    axes[0].set_ylabel("GARCH persistence  alpha + beta")
    axes[1].legend(fontsize=6.8, loc="lower left", ncol=1)
    fig.tight_layout()
    save(fig, "rep_fig_persistence_by_frequency")

    # Fig 7: correlograms of raw / filtered / standardized |R|
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6))
    for j, key in enumerate(keys):
        N = out[key]["R"].shape[1]
        for i, days in enumerate([5, 40]):
            ax = axes[i, j]
            for nm, c in [("R", SERIES[0]), ("Rf", SERIES[1]), ("Rs", SERIES[2])]:
                a = acf(np.abs(out[key][nm].to_numpy().ravel()), days * N)
                ax.plot(np.arange(1, days * N + 1) / N, a, color=c, lw=0.6 if days == 40 else 0.9,
                        label={"R": "Raw |R|", "Rf": "Filtered |R/s|", "Rs": "Standardized |R/(sigma s)|"}[nm])
            ax.axhline(0, color=INK2, lw=0.5)
            ax.set_title(f"{name(key)}: {days}-day correlogram")
            ax.set_xlabel("Lag (trading days)")
    axes[0, 0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "rep_fig7_filtered_correlograms")

    # Table-1 comparison chart: abs-return first-order autocorrelation and VR^A vs A&B
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, col, ttl in [(axes[0], "rho1_abs", "First-order autocorrelation of |R^k|"), (axes[1], "VR_abs", "Variance ratio VR^A")]:
        for key in keys:
            ref = REF[REF.market == SPEC[key]["market"]].set_index("k")
            t1 = out[key]["t1"]
            ax.plot(t1.index, t1[col], "o-", color=ASSET_COLOR[key], ms=4, label=f"{name(key)}: this study")
            ax.plot(ref.index, ref[col], "s--", color=ASSET_COLOR[key], ms=3.5, mfc="none", lw=0.9,
                    label=f"{'DM-$' if key=='eurusd' else 'S&P 500'}: A&B 1997")
        ax.set_xscale("log")
        ax.set_xlabel("Return interval k (x 5 minutes)")
        ax.set_title(ttl)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "rep_table1_comparison")
    # Estimator comparison: model-free profile vs PPML vs A&B log-OLS
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    for ax, key in zip(axes, keys):
        R, sg = out[key]["R"], out[key]["sigma"]
        ax.plot(mean_abs_profile(R).index, mean_abs_profile(R), color=ASSET_COLOR[key], lw=1.2,
                label="Model-free: average |R_n| / average |R|")
        ax.plot(R.columns, out[key]["fff"].s_hat(sg).mean(), color=INK2, lw=1.3, label="FFF, Poisson PML (this study)")
        ax.plot(R.columns, out[key]["fff_ols"].s_hat(sg).mean(), color=SERIES[7], lw=1.1, ls="--",
                label="FFF, log-OLS (A&B), zero returns dropped")
        ax.set_ylabel("Periodic component s_n")
        ax.set_title(f"{'ab'[keys.index(key)]})  {name(key)}: {100 * float((R == 0).to_numpy().mean()):.1f}% of returns are exactly zero")
        label_axis(ax, key, R.shape[1])
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "rep_fig_estimator_comparison")
    print("done")


if __name__ == "__main__":
    main()
