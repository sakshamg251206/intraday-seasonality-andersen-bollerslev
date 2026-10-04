"""Step 7: build the paper's LaTeX tables from results CSVs (no hand-typed numbers)."""
import numpy as np
import pandas as pd

from intraday.config import ROOT, TABLES

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
REF = pd.read_csv(ROOT / "reference" / "ab1997_values.csv")
NAME = {"eurusd": "EUR/USD", "spx": "S\\&P 500", "btc": "BTC/USDT"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)


# Data summary
q = pd.read_csv(TABLES / "data_quality.csv")
rows = "\n".join(f"{r.asset.replace('&', chr(92) + '&')} & {r.first_day} & {r.last_day} & {r.days:,} & {r.N} & {r.obs:,} & "
                 f"{r.days_dropped} & {100 * r.fresh_share:.2f} & {100 * r.zero_return_share:.2f} \\\\" for r in q.itertuples())
write("data_summary", r"""\begin{tabular}{lllrrrrrr}
\toprule
Asset & First day & Last day & Days & $N$ & Returns & Dropped & Fresh (\%) & Zero (\%) \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}""")

# Replication of Table 1 (selected columns) and Tables 2/4/5 persistence
for key, market in [("eurusd", "fx"), ("spx", "equity")]:
    t1 = pd.read_csv(TABLES / f"rep_table1_{key}.csv", index_col=0)
    ref = REF[REF.market == market].set_index("k")
    t2 = pd.read_csv(TABLES / f"rep_table2_raw_{key}.csv", index_col=0)
    t4 = pd.read_csv(TABLES / f"rep_table4_filtered_{key}.csv", index_col=0)
    t5 = pd.read_csv(TABLES / f"rep_table5_standardized_{key}.csv", index_col=0)
    rows = []
    for k in t1.index:
        rows.append(f"{k} & {f(t1.loc[k, 'sd'])} & {f(ref.loc[k, 'sd'])} & {f(t1.loc[k, 'kurtosis'], 1)} & {f(ref.loc[k, 'kurtosis'], 1)} & "
                    f"{f(t1.loc[k, 'rho1_abs'])} & {f(ref.loc[k, 'rho1_abs'])} & {f(t1.loc[k, 'VR_abs'])} & {f(ref.loc[k, 'VR_abs'])} \\\\")
    write(f"rep_table1_{key}", r"""\begin{tabular}{rrrrrrrrr}
\toprule
 & \multicolumn{2}{c}{St.\ dev.\ (\%)} & \multicolumn{2}{c}{Kurtosis} & \multicolumn{2}{c}{$\rho_1^A$} & \multicolumn{2}{c}{$VR^A$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
$k$ & here & A\&B & here & A\&B & here & A\&B & here & A\&B \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    rows = []
    for k in t2.index:
        rows.append(f"{k} & {f(t2.loc[k, 'alpha'])} & {f(t2.loc[k, 'persistence'])} & {f(ref.loc[k, 'persist_raw'])} & "
                    f"{f(t4.loc[k, 'alpha'])} & {f(t4.loc[k, 'persistence'])} & {f(ref.loc[k, 'persist_filtered'])} & "
                    f"{f(t5.loc[k, 'persistence'])} & {f(ref.loc[k, 'persist_standardized'])} \\\\")
    write(f"rep_persistence_{key}", r"""\begin{tabular}{rrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{Raw $R$ (Table 2)} & \multicolumn{3}{c}{Filtered $R/\hat s$ (Table 4)} & \multicolumn{2}{c}{Standardised (Table 5)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
$k$ & $\hat\alpha$ & $\hat\alpha+\hat\beta$ & A\&B & $\hat\alpha$ & $\hat\alpha+\hat\beta$ & A\&B & $\hat\alpha+\hat\beta$ & A\&B \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# H2 rolling-window tests
h = pd.read_csv(TABLES / "rob_r1_h2_tests.csv")
rows = "\n".join(f"{NAME[r.asset]} & {r.windows} & {f(r.median_disp_raw)} & {f(r.median_disp_filtered)} & "
                 f"{100 * r.share_windows_filtered_better:.0f}\\% & {f(r.wilcoxon_p_one_sided, 4)} & "
                 f"{f(r.median_min_persistence_raw)} & {f(r.median_min_persistence_filtered)} \\\\" for r in h.itertuples())
write("rob_h2", r"""\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{Median dispersion} & Filtered & Wilcoxon & \multicolumn{2}{c}{Median $\min_k(\hat\alpha+\hat\beta)$} \\
\cmidrule(lr){3-4}\cmidrule(lr){7-8}
Asset & Windows & raw & filtered & better & $p$ (1-sided) & raw & filtered \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}""")

# OOS forecasts
o = pd.read_csv(TABLES / "oos_forecast_summary_all.csv")
rows = []
for key in ["eurusd", "spx", "btc"]:
    g = o[o.asset == key]
    first = True
    for r in g.itertuples():
        lab = f"\\multirow{{5}}{{*}}{{{NAME[key]}}}" if first else ""
        first = False
        dm4 = f(getattr(r, "qlike_DM_t_vs_M4"), 1) if r.model.startswith("M3") else ""
        rows.append(f"{lab} & {r.model} & {f(r.mean_qlike, 4)} & {f(100 * r.qlike_diff_vs_M0 / abs(g.mean_qlike.iloc[0]), 2)} & "
                    f"{f(r.qlike_DM_t_vs_M0, 1)} & {f(r.qlike_DM_t_vs_M1, 1)} & {dm4} & {f(r.risk_mean_abs_log_sd_by_interval)} \\\\")
    rows.append("\\midrule")
write("oos_forecasts", r"""\begin{tabular}{llrrrrrr}
\toprule
Asset & Model & QLIKE & $\Delta$ vs M0 (\%) & DM $t$ vs M0 & DM $t$ vs M1 & DM $t$ vs M4 & Risk error \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}""")

# Trading
tr = pd.read_csv(TABLES / "oos_trading.csv")
tr = tr[tr.cost == "base"]
rows = []
for r in tr.itertuples():
    if r.trades_per_day == 0:
        continue
    rows.append(f"{NAME[r.asset]} & {r.strategy.replace('HKS same-interval momentum', 'HKS momentum')} & {r.sample} & "
                f"{r.trades_per_day} & {f(r.gross_sharpe, 2)} & {f(r.gross_t_NW, 2)} & {f(r.net_sharpe, 1)} & "
                f"{f(100 * r.breakeven_cost_pct_rt, 3)} & {f(100 * r.cost_pct_rt, 2)} \\\\")
write("oos_trading", r"""\begin{tabular}{lllrrrrrr}
\toprule
Asset & Strategy & Sample & Trades/day & Gross SR & NW $t$ & Net SR & Break-even (bp) & Cost (bp) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# Extensions summary
e3 = pd.read_csv(TABLES / "ext_e3_mean_return_tests.csv")
rows = "\n".join(f"{NAME[r['asset']]} & {r['train_days']} & {r['N']} & {r['naive_p<0.05']} & "
                 f"{f(r['expected_false_at_5%'], 1)} & {r['bh_fdr_discoveries']} \\\\" for r in e3.to_dict("records"))
write("ext_mean_returns", r"""\begin{tabular}{lrrrrr}
\toprule
Asset & Training days & Intervals & Naive $p<0.05$ & Expected false & BH-FDR discoveries \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}""")
j = pd.read_csv(TABLES / "ext_e5a_jshape_by_regime.csv")
rows = "\n".join(f"{NAME[r.asset]} & {r.tercile} & {r.days} & {f(r.last_over_first_hour)} & [{f(r.ci_lo)}, {f(r.ci_hi)}] \\\\" for r in j.itertuples())
write("ext_jshape", r"""\begin{tabular}{llrrr}
\toprule
Asset & $\sigma_t$ tercile & Days & Last/first hour vol. & 95\% CI \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}""")
print("tables written to", OUT)

# Machine-readable headline results (generated, never hand-edited)
import json

from intraday.config import RESULTS

h2 = pd.read_csv(TABLES / "rob_r1_h2_tests.csv").set_index("asset")
oos = pd.read_csv(TABLES / "oos_forecast_summary_all.csv")
m3 = oos[oos.model.str.startswith("M3")].set_index("asset")
m0 = oos[oos.model.str.startswith("M0")].set_index("asset")
m1 = oos[oos.model.str.startswith("M1")].set_index("asset")
trd = pd.read_csv(TABLES / "oos_trading.csv")
trd = trd[(trd.cost == "base") & (trd["sample"] == "test")]
etf = pd.read_csv(TABLES / "ext_e8_btc_etf_test.csv").set_index("measure")
key = {
    "H2_rolling_windows": {a: {"windows": int(r.windows), "share_filtered_better": round(r.share_windows_filtered_better, 3),
                               "wilcoxon_p": round(r.wilcoxon_p_one_sided, 5),
                               "median_min_persistence_raw": round(r.median_min_persistence_raw, 3),
                               "median_min_persistence_filtered": round(r.median_min_persistence_filtered, 3)}
                           for a, r in h2.iterrows()},
    "H6_oos_qlike": {a: {"M3_vs_M4_DM_t": round(m3.loc[a, "qlike_DM_t_vs_M4"], 2),
                         "M3_gain_vs_flat_pct": round(-100 * m3.loc[a, "qlike_diff_vs_M0"] / abs(m0.loc[a, "mean_qlike"]), 2),
                         "risk_error_flat": round(m0.loc[a, "risk_mean_abs_log_sd_by_interval"], 3),
                         "risk_error_hist_profile": round(m1.loc[a, "risk_mean_abs_log_sd_by_interval"], 3)}
                     for a in m3.index},
    "H5_test_trading_base_cost": [{"asset": r.asset, "strategy": r.strategy, "gross_sharpe": round(r.gross_sharpe, 2),
                                   "net_sharpe": round(r.net_sharpe, 2),
                                   "breakeven_bp": round(100 * r.breakeven_cost_pct_rt, 3), "cost_bp": 100 * r.cost_pct_rt}
                                  for r in trd.itertuples() if r.trades_per_day > 0],
    "H7_btc_us_hours_variance_share": {"pre_2y": round(etf.loc["variance_share", "pre_2y_mean"], 4),
                                       "post_2y": round(etf.loc["variance_share", "post_2y_mean"], 4),
                                       "diff_ci95": [round(etf.loc["variance_share", "ci_lo"], 4), round(etf.loc["variance_share", "ci_hi"], 4)]},
}
(RESULTS / "key_results.json").write_text(json.dumps(key, indent=2))
print("wrote results/key_results.json")
