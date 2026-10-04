"""Step 5: robustness of the replication result (H2) and of the FFF specification.

R1  Every rolling one-year window (EUR/USD Oct-Sep, S&P and BTC calendar years):
    MA(1)-GARCH(1,1) at all frequencies on raw and FFF-filtered returns.
    Aggregation consistency = dispersion across k of log10(half-life in
    minutes); aggregation theory (Drost-Nijman) says the half-life in calendar
    time should not depend on k, so smaller dispersion = more consistent.
    H2: dispersion(filtered) < dispersion(raw), tested across windows with a
    one-sided Wilcoxon signed-rank test.
R2  FFF specification sensitivity on the headline windows: P, J, OLS vs PPML.

sigma_t for the FFF uses one-step-ahead daily GARCH with parameters estimated
on data up to the end of each window (descriptive, like A&B).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from intraday.bars import get_panel
from intraday.config import TABLES
from intraday.garch import conditional_sd, fit_ma1_garch11, garch_by_frequency
from intraday.periodicity import fit_fff
from intraday.plots import ASSET_COLOR, INK2, MUTED, SERIES, save, setup

setup()
NAME = {"eurusd": "EUR/USD", "spx": "S&P 500", "btc": "BTC/USDT"}
SPEC = {"eurusd": dict(P=6, J=0, dummies=()), "spx": dict(P=2, J=1, dummies=(78, 79, 80)),
        "btc": dict(P=6, J=0, dummies=())}


def windows(key, P):
    years = sorted(set(P.R.index.year))
    if key == "eurusd":
        return [(f"{y}-10-01", f"{y + 1}-09-30") for y in years[:-1]]
    return [(f"{y}-01-01", f"{y}-12-31") for y in years if (P.R.index.year == y).sum() > 150]


def sigma_for(P, end):
    daily = P.daily.loc[:end]
    return conditional_sd(fit_ma1_garch11(daily), daily)


def dispersion(t: pd.DataFrame) -> tuple[float, int]:
    hl = t.half_life.replace(np.inf, np.nan)
    return float(np.log10(hl.dropna()).std()), int(hl.isna().sum())


def r1_rolling():
    rows, per_k = [], []
    for key in ["eurusd", "spx", "btc"]:
        P = get_panel(key)
        for a, b in windows(key, P):
            R = P.R.loc[a:b]
            if len(R) < 150:
                continue
            sig = sigma_for(P, b).reindex(R.index)
            fff = fit_fff(R, sig, method="ppml", **SPEC[key])
            raw, filt = garch_by_frequency(R), garch_by_frequency(R / fff.s_hat(sig))
            for lab, t in [("raw", raw), ("filtered", filt)]:
                disp, n_inf = dispersion(t)
                rows.append({"asset": key, "window": a[:7], "days": len(R), "series": lab,
                             "logHL_dispersion": disp, "n_nonstationary": n_inf,
                             "min_persistence": t.persistence.min(),
                             "k_at_min": int(t.persistence.idxmin()),
                             "mean_alpha_k<=4": t.alpha[t.index <= 4].mean()})
                per_k.append(t[["persistence", "alpha", "half_life"]].assign(asset=key, window=a[:7], series=lab))
            print(key, a, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "rob_r1_rolling_windows.csv", index=False, float_format="%.4f")
    pd.concat(per_k).to_csv(TABLES / "rob_r1_rolling_by_k.csv", float_format="%.4f")
    return res, pd.concat(per_k)


def r1_tests(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, g in res.groupby("asset"):
        w = g.pivot(index="window", columns="series", values="logHL_dispersion").dropna()
        m = g.pivot(index="window", columns="series", values="min_persistence")
        test = st.wilcoxon(w["filtered"], w["raw"], alternative="less")
        out.append({"asset": key, "windows": len(w),
                    "median_disp_raw": w.raw.median(), "median_disp_filtered": w.filtered.median(),
                    "share_windows_filtered_better": float((w.filtered < w.raw).mean()),
                    "wilcoxon_p_one_sided": test.pvalue,
                    "median_min_persistence_raw": m.raw.median(),
                    "median_min_persistence_filtered": m.filtered.median()})
    t = pd.DataFrame(out)
    t.to_csv(TABLES / "rob_r1_h2_tests.csv", index=False, float_format="%.4f")
    print(t.to_string())
    return t


def r1_figures(res, per_k):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key in zip(axes, ["eurusd", "spx", "btc"]):
        g = res[res.asset == key].pivot(index="window", columns="series", values="logHL_dispersion")
        ax.scatter(g.raw, g.filtered, color=ASSET_COLOR[key], s=22, zorder=3)
        lim = [0, max(g.max().max(), 0.1) * 1.1]
        ax.plot(lim, lim, color=MUTED, ls="--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Raw returns")
        ax.set_title(f"{NAME[key]}: {int((g.filtered < g.raw).sum())}/{len(g)} windows below diagonal")
    axes[0].set_ylabel("Filtered returns")
    fig.suptitle("Dispersion of log half-life across sampling frequencies, one point per one-year window "
                 "(below diagonal = filtering restores aggregation consistency)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "rob_r1_dispersion_scatter")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, key in zip(axes, ["eurusd", "spx", "btc"]):
        d = per_k[per_k.asset == key].reset_index()
        for lab, c in [("raw", SERIES[0]), ("filtered", SERIES[1])]:
            q = d[d.series == lab].groupby("k").persistence.quantile([0.1, 0.5, 0.9]).unstack()
            ax.fill_between(q.index, q[0.1].clip(upper=1.05), q[0.9].clip(upper=1.05), color=c, alpha=0.18, lw=0)
            ax.plot(q.index, q[0.5].clip(upper=1.05), "o-", color=c, ms=3.5, label=f"{lab}: median, 10-90% band")
        ax.set_xscale("log")
        ks = sorted(d.k.unique())
        ax.set_xticks(ks[::2], [str(k) for k in ks[::2]], fontsize=7)
        ax.axhline(1, color=MUTED, lw=0.6, ls=":")
        ax.set_title(f"{NAME[key]}: alpha+beta across windows")
        ax.set_xlabel("k (x 5 minutes)")
    axes[0].set_ylabel("alpha + beta")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "rob_r1_persistence_bands")


def r2_spec():
    rows = []
    heads = {"eurusd": ("2024-10-01", "2025-09-30"), "spx": ("2022-01-01", "2025-12-31")}
    for key, (a, b) in heads.items():
        P = get_panel(key)
        R = P.R.loc[a:b]
        sig = sigma_for(P, b).reindex(R.index)
        dummies = SPEC[key]["dummies"]
        for method in ["ppml", "ols"]:
            for J in [0, 1]:
                for Pn in [1, 2, 4, 6, 8]:
                    m = fit_fff(R, sig, P=Pn, J=J, dummies=dummies, method=method)
                    t = garch_by_frequency(R / m.s_hat(sig))
                    disp, n_inf = dispersion(t)
                    rows.append({"asset": key, "method": method, "J": J, "P": Pn, "qlik": m.extra["qlik"],
                                 "logHL_dispersion": disp, "n_nonstationary": n_inf,
                                 "min_persistence": t.persistence.min()})
            print(key, method, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(TABLES / "rob_r2_fff_spec_sensitivity.csv", index=False, float_format="%.4f")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, key in zip(axes, heads):
        for (method, J), c, ls in [(("ppml", 0), SERIES[0], "-"), (("ppml", 1), SERIES[1], "-"),
                                   (("ols", 0), SERIES[0], "--"), (("ols", 1), SERIES[1], "--")]:
            g = r[(r.asset == key) & (r.method == method) & (r.J == J)]
            ax.plot(g.P, g.logHL_dispersion, ls, marker="o", color=c, ms=4, label=f"{method.upper()}, J={J}")
        ax.set_title(f"{NAME[key]}: aggregation consistency vs FFF order")
        ax.set_xlabel("Number of Fourier pairs P")
    axes[0].set_ylabel("Dispersion of log10 half-life")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "rob_r2_fff_spec")


def main():
    res, per_k = r1_rolling()
    r1_tests(res)
    r1_figures(res, per_k)
    r2_spec()
    print("done")


if __name__ == "__main__":
    main()
