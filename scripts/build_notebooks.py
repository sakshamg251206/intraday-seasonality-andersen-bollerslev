"""Create and execute the two notebooks (outputs are real, produced by running them)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from intraday.config import ROOT

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

walk = nbf.v4.new_notebook()
walk.cells = [
    md("# 1. Method walkthrough: A&B (1997) step by step\n\n"
       "This notebook runs the core method on **one year of EUR/USD** so each step can be inspected. "
       "Nothing here is new relative to `src/intraday`; it only calls the library functions in order.\n\n"
       "Prerequisite: `make data` (or `python scripts/01_download.py eurusd`)."),
    code("import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from intraday.bars import get_panel\n"
         "from intraday.garch import fit_ma1_garch11, conditional_sd, garch_by_frequency\n"
         "from intraday.periodicity import fit_fff, mean_abs_profile\n"
         "P = get_panel('eurusd')\n"
         "R = P.R.loc['2024-10-01':'2025-09-30']\n"
         "print(R.shape, 'days x intervals;', f'{(R == 0).to_numpy().mean():.1%} exact-zero returns')"),
    md("## Step 1 - the daily volatility factor $\\sigma_t$\n"
       "A daily MA(1)-GARCH(1,1). `conditional_sd` returns the one-step-ahead forecast, so $\\sigma_t$ uses returns up to day $t-1$ only."),
    code("daily = P.daily.loc[:'2025-09-30']\n"
         "dfit = fit_ma1_garch11(daily)\n"
         "print(dfit.params.round(4).to_dict())\n"
         "sigma = conditional_sd(dfit, daily).reindex(R.index)\n"
         "sigma.plot(figsize=(8, 2.5), title='sigma_t (daily %)');"),
    md("## Step 2 - the periodic component $s_n$\n"
       "Model-free profile (average $|R_n|$ / average $|R|$) versus the Flexible Fourier Form estimated by Poisson PML. "
       "The FFF uses 2P+3 parameters instead of N=288."),
    code("fff = fit_fff(R, sigma, P=6, J=0, method='ppml')\n"
         "s_hat = fff.s_hat(sigma)\n"
         "ax = mean_abs_profile(R).plot(figsize=(8, 3), label='model-free')\n"
         "s_hat.mean().plot(ax=ax, label='FFF (PPML)'); ax.legend(); ax.set_xlabel('interval n (n=1 starts 17:00 New York)');"),
    md("## Step 3 - GARCH at every sampling frequency, raw vs filtered\n"
       "Aggregation theory says the half-life in *minutes* should not depend on k. Raw returns violate this badly; filtered returns much less."),
    code("raw = garch_by_frequency(R)\n"
         "filt = garch_by_frequency(R / s_hat)\n"
         "tab = pd.DataFrame({'raw a+b': raw.persistence, 'filtered a+b': filt.persistence,\n"
         "                    'raw half-life (min)': raw.half_life, 'filtered half-life (min)': filt.half_life})\n"
         "tab.round(3)"),
    code("ax = tab[['raw a+b', 'filtered a+b']].clip(upper=1.05).plot(marker='o', logx=True, figsize=(7, 3))\n"
         "ax.set_xlabel('k (x 5 minutes)'); ax.axhline(1, ls=':', c='grey');"),
    md("**Reading the result.** Raw persistence is >1 at 5-10 minutes (nonstationary) and collapses at 1-1.5 hours; "
       "after dividing by $\\hat s$ it is stable. This is A&B's Table 2 vs Table 4."),
]

tour = nbf.v4.new_notebook()
figs = [("rep_fig2_volatility_profile", "Volatility profiles (A&B Fig. 2/6)"),
        ("rep_fig_persistence_by_frequency", "Persistence by frequency vs A&B (Tables 2/4/5)"),
        ("rob_r1_persistence_bands", "All one-year windows (H2)"),
        ("ext_e1_cross_asset_profiles", "Cross-asset profiles"),
        ("ext_e4_heatmaps_dow_hour", "Time-of-day x day-of-week"),
        ("ext_e8_btc_weekend_etf", "BTC: weekend and US-hours share (H7)"),
        ("oos_forecast_qlike", "Out-of-sample volatility forecasts (H6)"),
        ("oos_risk_targeting", "Risk targeting"),
        ("oos_trading_is_vs_oos", "Mean-return strategies, in- vs out-of-sample (H5)")]
tour.cells = [md("# 2. Results tour\n\nLoads the saved tables and figures produced by `make all`. No estimation happens here."),
              code("import pandas as pd\nfrom IPython.display import Image, display\nfrom intraday.config import TABLES, FIGURES")]
for name, title in figs:
    tour.cells += [md(f"## {title}"), code(f"display(Image(filename=str(FIGURES / '{name}.png'), width=900))")]
tour.cells += [md("## Key tables"),
               code("pd.read_csv(TABLES / 'rob_r1_h2_tests.csv').round(4)"),
               code("pd.read_csv(TABLES / 'oos_forecast_summary_all.csv')[['asset','model','mean_qlike','qlike_DM_t_vs_M1','qlike_DM_t_vs_M4','risk_mean_abs_log_sd_by_interval']].round(3)"),
               code("t = pd.read_csv(TABLES / 'oos_trading.csv'); t[t.cost == 'base'][['asset','strategy','sample','gross_sharpe','gross_t_NW','net_sharpe','breakeven_cost_pct_rt']].round(4)")]

for nb, name in [(walk, "01_method_walkthrough"), (tour, "02_results_tour")]:
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")
    print("wrote", name)
