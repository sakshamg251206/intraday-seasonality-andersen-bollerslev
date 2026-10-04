"""Step 4: extensions beyond A&B (1997) - full samples, all three assets.

E1 cross-asset volatility profiles on a common UTC clock
E2 volume vs volatility (H3)                     - BTC (true exchange volume)
E3 mean-return seasonality (H5, in-sample part)  - HAC t-stats + Benjamini-Hochberg
E4 time-of-day x day-of-week heatmaps
E5 regimes: ex-ante volatility terciles (H4), year-by-year stability, DST
E6 return distributions before/after standardisation
E7 the 08:30 New York announcement effect (EUR/USD)
E8 BTC: weekend vs weekday, and the US spot-ETF break (H7, original research)

Look-ahead: sigma_t (used for regimes and standardisation) comes from a daily
MA(1)-GARCH(1,1) whose parameters are estimated on TRAINING years only and whose
recursion uses returns up to t-1 (garch.ex_ante_daily_sigma).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from intraday.bars import get_panel, interval_end_utc
from intraday.config import ASSETS, TABLES
from intraday.garch import ex_ante_daily_sigma
from intraday.periodicity import mean_abs_profile
from intraday.plots import ASSET_COLOR, DIV, INK2, MUTED, SEQ, SERIES, save, setup
from intraday.stats import block_bootstrap_ci, fdr_bh, newey_west_tstat

setup()
NAME = {"eurusd": "EUR/USD", "spx": "S&P 500", "btc": "BTC/USDT"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ETF_DATE = pd.Timestamp("2024-01-11")   # first trading day of US spot bitcoin ETFs


def load():
    data = {}
    for key in ASSETS:
        P = get_panel(key)
        sigma, _ = ex_ante_daily_sigma(P.daily, P.asset.train_end)
        data[key] = dict(P=P, R=P.R, sigma=sigma.reindex(P.R.index), t_utc=interval_end_utc(P))
    return data


# ----------------------------------------------------------------- E1
def e1_cross_asset(data):
    """Normalised volatility profile s_n per 5-minute UTC slot, one axis for all assets."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    rows = {}
    for key, d in data.items():
        t = pd.DatetimeIndex(d["t_utc"].to_numpy().ravel())
        a = pd.Series(np.abs(d["R"].to_numpy().ravel()), index=t.hour * 60 + t.minute)
        prof = a.groupby(level=0).mean()
        prof = prof / prof.mean()
        prof = prof.reindex(range(5, 1441, 5) if key != "btc" else prof.index)
        rows[key] = prof
        ax.plot(prof.index / 60, prof, color=ASSET_COLOR[key], lw=1.2, label=NAME[key])
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("UTC hour (end of 5-minute interval)")
    ax.set_ylabel("Volatility relative to asset's daily mean")
    ax.set_title("Intraday volatility profiles on a common clock (full samples)")
    for h, lab in [(0, "Tokyo open"), (7, "London open"), (13.5, "NY data / cash open (summer)")]:
        ax.axvline(h, color=MUTED, lw=0.7, ls=":")
        ax.annotate(lab, (h + 0.1, ax.get_ylim()[1] * 0.95), fontsize=7, color=MUTED)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "ext_e1_cross_asset_profiles")
    pd.DataFrame(rows).to_csv(TABLES / "ext_e1_profiles_utc.csv", float_format="%.4f")


# ----------------------------------------------------------------- E2
def e2_volume(data):
    d = data["btc"]
    R, V = d["R"], d["P"].volume
    vol_share = V.div(V.sum(axis=1), axis=0).mean() * len(R.columns)   # mean share, 1 = uniform
    absr = mean_abs_profile(R)
    ci = block_bootstrap_ci(V.div(V.sum(axis=1), axis=0) * len(R.columns), lambda x: x.mean().to_numpy(), n_boot=200)
    r_log = np.corrcoef(np.log(vol_share), np.log(absr))[0, 1]
    # day-level: correlation between |R| and volume within each day, averaged
    within = pd.Series([np.corrcoef(np.abs(R.iloc[i]), V.iloc[i])[0, 1] for i in range(len(R))], index=R.index)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    hrs = R.columns / 12
    axes[0].fill_between(hrs, ci.lo, ci.hi, color=SERIES[2], alpha=0.2, lw=0)
    axes[0].plot(hrs, vol_share, color=SERIES[2], lw=1.3, label="Volume (share x N)")
    axes[0].plot(hrs, absr, color=INK2, lw=1.1, ls="--", label="Volatility s_n (avg |R| / mean)")
    axes[0].set_xlabel("UTC hour")
    axes[0].set_xticks(range(0, 25, 3))
    axes[0].set_ylabel("Relative to daily mean")
    axes[0].set_title("a)  BTC/USDT: volume and volatility profiles, 2018-2025")
    axes[0].legend(loc="upper left")
    axes[1].scatter(np.log(vol_share), np.log(absr), s=10, color=SERIES[2], alpha=0.7, edgecolor="none")
    b = np.polyfit(np.log(vol_share), np.log(absr), 1)
    xs = np.linspace(np.log(vol_share).min(), np.log(vol_share).max(), 10)
    axes[1].plot(xs, np.polyval(b, xs), color=INK2, lw=1)
    axes[1].set_xlabel("log volume profile")
    axes[1].set_ylabel("log volatility profile")
    axes[1].set_title(f"b)  288 intervals: corr {r_log:.2f}, slope {b[0]:.2f}")
    fig.tight_layout()
    save(fig, "ext_e2_volume_volatility")
    pd.DataFrame({"stat": ["corr_log_profiles", "elasticity", "mean_within_day_corr", "share_days_within_corr_pos"],
                  "value": [r_log, b[0], within.mean(), (within > 0).mean()]}).to_csv(
        TABLES / "ext_e2_volume_volatility.csv", index=False, float_format="%.4f")


# ----------------------------------------------------------------- E3
def e3_mean_returns(data):
    """Per-interval mean return on TRAINING days: Newey-West t-stats, BH-FDR at 5%."""
    rows, fig, axes = [], *plt.subplots(3, 1, figsize=(8, 7.5))
    for ax, (key, d) in zip(axes, data.items()):
        a = d["P"].asset
        R = d["R"][d["R"].index.year <= a.train_end]
        t = np.array([newey_west_tstat(R[c].to_numpy()) for c in R.columns])
        p = 2 * st.norm.sf(np.abs(t))
        disc = fdr_bh(p, 0.05)
        rows.append({"asset": key, "train_days": len(R), "N": len(t),
                     "naive_p<0.05": int((p < 0.05).sum()), "expected_false_at_5%": 0.05 * len(t),
                     "bh_fdr_discoveries": int(disc.sum()),
                     "discovered_intervals": ",".join(map(str, R.columns[disc]))})
        ax.bar(R.columns, t, color=np.where(disc, ASSET_COLOR[key], "#c9c8c3"), width=0.9)
        for z in (-1.96, 1.96):
            ax.axhline(z, color=MUTED, lw=0.7, ls="--")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(f"{NAME[key]}: {disc.sum()} of {len(t)} intervals significant after FDR control "
                     f"({(p < 0.05).sum()} at naive 5%)")
        ax.set_ylabel("NW t-stat")
    axes[-1].set_xlabel("Five-minute interval n (training sample only)")
    fig.tight_layout()
    save(fig, "ext_e3_mean_return_tstats")
    pd.DataFrame(rows).to_csv(TABLES / "ext_e3_mean_return_tests.csv", index=False)

    # cumulative average intraday return path with 95% CI (all days)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (key, d) in zip(axes, data.items()):
        cum = d["R"].cumsum(axis=1)
        m, se = cum.mean(), cum.std() / np.sqrt(len(cum))
        ax.fill_between(cum.columns, m - 1.96 * se, m + 1.96 * se, color=ASSET_COLOR[key], alpha=0.2, lw=0)
        ax.plot(cum.columns, m, color=ASSET_COLOR[key], lw=1.3)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(NAME[key])
        ax.set_xlabel("Interval n")
    axes[0].set_ylabel("Mean cumulative return (%)")
    fig.tight_layout()
    save(fig, "ext_e3_cumulative_return_path")


# ----------------------------------------------------------------- E4
def e4_heatmaps(data):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))
    for row, (key, d) in enumerate(data.items()):
        t = pd.DatetimeIndex(d["t_utc"].to_numpy().ravel())
        local = t.tz_localize("UTC").tz_convert(d["P"].asset.tz) if t.tz is None else t.tz_convert(d["P"].asset.tz)
        lt = local - pd.Timedelta(minutes=5)       # interval START, so hour h = [h, h+1)
        df = pd.DataFrame({"r": d["R"].to_numpy().ravel(), "dow": np.repeat(d["R"].index.dayofweek, d["R"].shape[1]),
                           "hour": lt.hour})
        df["a"] = df.r.abs()
        vol = df.pivot_table(index="dow", columns="hour", values="a", aggfunc="mean")
        vol = vol / vol.stack().mean()
        tst = df.groupby(["dow", "hour"]).r.agg(lambda x: x.mean() / x.std() * np.sqrt(len(x))).unstack()
        days = [DOW[i] for i in vol.index]
        im = axes[row, 0].imshow(vol, aspect="auto", cmap=SEQ, vmin=0)
        axes[row, 0].set_title(f"{NAME[key]}: volatility (1 = average)")
        lim = 4
        im2 = axes[row, 1].imshow(tst.clip(-lim, lim), aspect="auto", cmap=DIV, vmin=-lim, vmax=lim)
        axes[row, 1].set_title(f"{NAME[key]}: mean-return t-stat (clipped at +-{lim})")
        for ax, h in [(axes[row, 0], im), (axes[row, 1], im2)]:
            ax.set_yticks(range(len(days)), days)
            ax.set_xticks(range(len(vol.columns))[::2], [f"{c:02d}" for c in vol.columns[::2]], fontsize=7)
            ax.set_xlabel("Hour of day, " + ("UTC" if key == "btc" else "New York time"))
            ax.grid(False)
            fig.colorbar(h, ax=ax, fraction=0.025)
        # day label: for EUR/USD the session ending on that day starts 17:00 NY the previous evening
    fig.tight_layout()
    save(fig, "ext_e4_heatmaps_dow_hour")


# ----------------------------------------------------------------- E5
def e5_regimes(data):
    # (a) ex-ante volatility terciles
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    rows = []
    for ax, (key, d) in zip(axes, data.items()):
        q = pd.qcut(d["sigma"], 3, labels=["Low", "Mid", "High"])
        for lab, c in zip(["Low", "Mid", "High"], [SERIES[0], SERIES[3], SERIES[7]]):
            prof = mean_abs_profile(d["R"][q == lab])
            ax.plot(prof.index, prof, color=c, lw=1.0, label=f"{lab} sigma_t")
        ax.set_title(f"{NAME[key]}: profile by ex-ante volatility tercile")
        ax.set_xlabel("Interval n")
        # J-shape statistic: last-hour vs first-hour volatility, per tercile, block-bootstrap CI
        N, h = d["R"].shape[1], 12
        for lab in ["Low", "Mid", "High"]:
            Rq = d["R"][q == lab]
            stat = lambda x: np.array([x.iloc[:, -h:].abs().to_numpy().mean() / x.iloc[:, :h].abs().to_numpy().mean()])
            ci = block_bootstrap_ci(Rq.reset_index(drop=True), stat, n_boot=300)
            rows.append({"asset": key, "tercile": lab, "days": len(Rq), "last_over_first_hour": ci.est.iloc[0],
                         "ci_lo": ci.lo.iloc[0], "ci_hi": ci.hi.iloc[0]})
    axes[0].set_ylabel("s_n")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "ext_e5a_profile_by_vol_regime")
    pd.DataFrame(rows).to_csv(TABLES / "ext_e5a_jshape_by_regime.csv", index=False, float_format="%.4f")

    # (b) stability: year x interval heatmap of normalised profiles + corr with full-sample profile
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    stab = []
    for ax, (key, d) in zip(axes, data.items()):
        full = mean_abs_profile(d["R"])
        yp = d["R"].groupby(d["R"].index.year).apply(lambda x: mean_abs_profile(x))
        yp = yp.unstack() if isinstance(yp, pd.Series) else yp
        for y, row in yp.iterrows():
            stab.append({"asset": key, "year": y, "corr_with_full": np.corrcoef(row, full)[0, 1],
                         "days": int((d["R"].index.year == y).sum())})
        im = ax.imshow(yp.to_numpy(), aspect="auto", cmap=SEQ, vmin=0, vmax=np.quantile(yp.to_numpy(), 0.99),
                       extent=[0.5, yp.shape[1] + 0.5, yp.index.max() + 0.5, yp.index.min() - 0.5])
        ax.set_title(f"{NAME[key]}: volatility profile s_n by year")
        ax.set_ylabel("Year")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.02)
    axes[-1].set_xlabel("Interval n")
    fig.tight_layout()
    save(fig, "ext_e5b_profile_stability_by_year")
    stab = pd.DataFrame(stab)
    stab.to_csv(TABLES / "ext_e5b_profile_stability.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(7, 3))
    for key in data:
        s = stab[stab.asset == key]
        ax.plot(s.year, s.corr_with_full, "o-", color=ASSET_COLOR[key], ms=4, label=NAME[key])
    ax.set_ylabel("Corr(year profile, full-sample profile)")
    ax.set_title("Rolling stability of the intraday volatility shape")
    ax.legend()
    fig.tight_layout()
    save(fig, "ext_e5c_profile_stability_corr")

    # (c) DST: EUR/USD in the New-York clock, US summer vs winter
    d = data["eurusd"]
    ny_off = (pd.DatetimeIndex(d["t_utc"].iloc[:, 0]).tz_localize("UTC") if pd.DatetimeIndex(d["t_utc"].iloc[:, 0]).tz is None
              else pd.DatetimeIndex(d["t_utc"].iloc[:, 0]))
    summer = np.array([bool(t.tz_convert("America/New_York").dst()) for t in ny_off])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for mask, lab, c in [(~summer, "US standard time (winter)", SERIES[0]), (summer, "US daylight time (summer)", SERIES[1])]:
        prof = mean_abs_profile(d["R"][mask])
        ax.plot(prof.index, prof, color=c, lw=1.1, label=lab)
    ticks = [(n, f"{(17 * 60 + 5 * n) // 60 % 24:02d}:00") for n in range(12, 289, 36)]
    ax.set_xticks([a for a, _ in ticks], [b for _, b in ticks])
    ax.set_xlabel("New York time (end of interval)")
    ax.set_ylabel("s_n")
    ax.set_title("EUR/USD: Asian-session activity shifts by one hour in the NY clock; US events do not")
    ax.legend()
    fig.tight_layout()
    save(fig, "ext_e5d_eurusd_dst")


# ----------------------------------------------------------------- E6
def e6_distributions(data):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    rows = []
    for j, (key, d) in enumerate(data.items()):
        R, N = d["R"], d["R"].shape[1]
        s = mean_abs_profile(R[R.index.year <= d["P"].asset.train_end])     # training-sample profile
        raw = (R / R.to_numpy().std()).to_numpy().ravel()
        stdz = (R / s / d["sigma"].to_numpy()[:, None] * np.sqrt(N)).to_numpy().ravel()
        stdz = stdz / np.nanstd(stdz)
        x = np.linspace(-6, 6, 400)
        for z, lab, c in [(raw, "R / sd(R)", MUTED), (stdz, "R / (sigma_t s_n)", ASSET_COLOR[key])]:
            z = z[np.isfinite(z)]
            axes[0, j].hist(np.clip(z, -6, 6), bins=200, density=True, histtype="step", color=c, lw=1.1, label=lab)
            rows.append({"asset": key, "series": lab, "skew": st.skew(z), "kurtosis": st.kurtosis(z, fisher=False),
                         "share_beyond_4sd": float((np.abs(z) > 4).mean())})
        axes[0, j].plot(x, st.norm.pdf(x), color=INK2, lw=0.8, ls="--", label="N(0,1)")
        axes[0, j].set_yscale("log")
        axes[0, j].set_ylim(1e-5, 1)
        axes[0, j].set_title(NAME[key])
        q = np.linspace(0.0005, 0.9995, 400)
        zz = stdz[np.isfinite(stdz)]
        axes[1, j].plot(st.norm.ppf(q), np.quantile(zz, q), color=ASSET_COLOR[key], lw=1.2, label="standardized")
        axes[1, j].plot(st.norm.ppf(q), np.quantile(raw[np.isfinite(raw)], q), color=MUTED, lw=1, label="raw")
        axes[1, j].plot([-4, 4], [-4, 4], color=INK2, lw=0.6, ls="--")
        axes[1, j].set_xlabel("Normal quantile")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[0, 0].set_ylabel("Density (log scale)")
    axes[1, 0].set_ylabel("Sample quantile")
    fig.tight_layout()
    save(fig, "ext_e6_distributions")
    pd.DataFrame(rows).to_csv(TABLES / "ext_e6_distribution_moments.csv", index=False, float_format="%.4f")


# ----------------------------------------------------------------- E7
def e7_announcements(data):
    d = data["eurusd"]
    R = d["R"]
    n830 = (8 * 60 + 35 - 17 * 60) % (24 * 60) // 5          # interval ending 08:35 NY
    a = R[n830].abs() / R.abs().mean(axis=1)                  # relative to that day's average |R|
    dow = R.index.dayofweek
    first_fri = (dow == 4) & (R.index.day <= 7)               # US employment report days
    groups = {"Mon": dow == 0, "Tue": dow == 1, "Wed": dow == 2, "Thu": dow == 3,
              "Fri (not 1st)": (dow == 4) & ~first_fri, "1st Fri (payrolls)": first_fri}
    rows = []
    for lab, m in groups.items():
        x = a[m].to_numpy()
        bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(500)]
        rows.append({"group": lab, "days": int(m.sum()), "mean": x.mean(), "lo": np.quantile(bs, 0.025),
                     "hi": np.quantile(bs, 0.975)})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "ext_e7_0830_effect.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(t.group, t["mean"], color=SERIES[0], width=0.6)
    ax.errorbar(t.group, t["mean"], yerr=[t["mean"] - t.lo, t.hi - t["mean"]], fmt="none", color=INK2, lw=1)
    ax.axhline(1, color=MUTED, ls="--", lw=0.8)
    ax.set_ylabel("|R| 08:30-08:35 NY / day's mean |R|")
    ax.set_title("EUR/USD: the 08:30 New York data-release interval, by weekday (95% CI)")
    fig.tight_layout()
    save(fig, "ext_e7_0830_effect")


# ----------------------------------------------------------------- E8
def e8_btc(data):
    d = data["btc"]
    R, V = d["R"], d["P"].volume
    dow = R.index.dayofweek
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for m, lab, c in [(dow < 5, "Weekdays", SERIES[2]), (dow >= 5, "Weekends", SERIES[6])]:
        axes[0].plot(R.columns / 12, R[m].abs().mean(), color=c, lw=1.1, label=f"{lab}: avg |R|")
    axes[0].set_xlabel("UTC hour")
    axes[0].set_ylabel("Average |R| (%)")
    axes[0].set_xticks(range(0, 25, 3))
    axes[0].set_title("a)  BTC: weekday vs weekend volatility profile")
    axes[0].legend()

    # US cash session 09:30-16:00 NY expressed in UTC, per day (DST-aware)
    t = d["t_utc"]
    tny = pd.DatetimeIndex(t.to_numpy().ravel())
    tny = (tny.tz_localize("UTC") if tny.tz is None else tny).tz_convert("America/New_York")
    mins = (tny.hour * 60 + tny.minute).to_numpy().reshape(t.shape)
    us = (mins > 9 * 60 + 30) & (mins <= 16 * 60)              # interval END in (09:30, 16:00]
    wk = (dow < 5)
    share_var = pd.Series(((R ** 2).to_numpy() * us).sum(1) / (R ** 2).to_numpy().sum(1), index=R.index)[wk]
    share_vol = pd.Series((V.to_numpy() * us).sum(1) / V.to_numpy().sum(1), index=R.index)[wk]
    uniform = us[wk].mean()
    yearly = pd.DataFrame({"variance_share": share_var.groupby(share_var.index.year).mean(),
                           "volume_share": share_vol.groupby(share_vol.index.year).mean()})
    yearly["uniform_benchmark"] = uniform
    yearly.to_csv(TABLES / "ext_e8_btc_us_hours_share_by_year.csv", float_format="%.4f")
    for col, c in [("variance_share", SERIES[2]), ("volume_share", SERIES[0])]:
        axes[1].plot(yearly.index, yearly[col], "o-", color=c, ms=4, label=col.replace("_", " "))
    axes[1].axhline(uniform, color=MUTED, ls="--", lw=0.8, label="share of clock time")
    axes[1].axvline(2024 - 0.5 + 10 / 365, color=INK2, ls=":", lw=0.9)
    axes[1].annotate("US spot ETFs", (2024 - 0.45, axes[1].get_ylim()[0] + 0.01), fontsize=7.5, color=INK2)
    axes[1].set_title("b)  Share of weekday BTC activity in US cash hours")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "ext_e8_btc_weekend_etf")

    # pre/post test: mean share in the two years either side of the ETF launch,
    # 20-day block bootstrap of the difference (shares are autocorrelated)
    rows = []
    for name, s in [("variance_share", share_var), ("volume_share", share_vol)]:
        pre = s[(s.index >= ETF_DATE - pd.DateOffset(years=2)) & (s.index < ETF_DATE)].to_numpy()
        post = s[(s.index >= ETF_DATE) & (s.index < ETF_DATE + pd.DateOffset(years=2))].to_numpy()
        rng = np.random.default_rng(0)

        def boot(x):
            b = 20
            idx = (rng.integers(0, len(x) - b, len(x) // b + 1)[:, None] + np.arange(b)).ravel()[: len(x)]
            return x[idx].mean()
        diffs = np.array([boot(post) - boot(pre) for _ in range(2000)])
        rows.append({"measure": name, "pre_2y_mean": pre.mean(), "post_2y_mean": post.mean(),
                     "diff": post.mean() - pre.mean(), "ci_lo": np.quantile(diffs, 0.025),
                     "ci_hi": np.quantile(diffs, 0.975), "pre_days": len(pre), "post_days": len(post)})
    pd.DataFrame(rows).to_csv(TABLES / "ext_e8_btc_etf_test.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows))


def main():
    data = load()
    for f in [e1_cross_asset, e2_volume, e3_mean_returns, e4_heatmaps, e5_regimes, e6_distributions,
              e7_announcements, e8_btc]:
        print("running", f.__name__, flush=True)
        f(data)
    print("done")


if __name__ == "__main__":
    main()
