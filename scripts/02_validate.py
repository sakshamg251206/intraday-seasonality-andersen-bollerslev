"""Step 2: build panels and validate the data. Writes tables/data_*.csv."""
import random
from datetime import date

import numpy as np
import pandas as pd

from intraday.bars import get_panel
from intraday.config import ASSETS, TABLES
from intraday.data import dukascopy_day, load_prices
from intraday.validate import compare_sources, panel_report

TABLES.mkdir(exist_ok=True)

# 1. Panels + quality report ---------------------------------------------------
reports, dropped = [], []
for key in ASSETS:
    P = get_panel(key, rebuild=True)
    reports.append(panel_report(P))
    dropped.append(P.dropped.rename("reason").to_frame().assign(asset=key))
pd.DataFrame(reports).to_csv(TABLES / "data_quality.csv", index=False)
pd.concat(dropped).to_csv(TABLES / "data_dropped_days.csv", index_label="day")
print(pd.DataFrame(reports).T)

# 2. Clock check, every year x DST state. Each group's 24h profile of mean
#    |1-min return| by New-York minute is correlated with the pooled profile
#    shifted by -60, 0, +60 minutes. A correct clock maximises at shift 0; a
#    one-hour error would maximise at +-60. Using the whole profile avoids
#    relying on one event minute (08:30 data, 09:30 open and 10:00 data all
#    compete at the 1-minute level).
rows = []
for key in ["eurusd", "spx"]:
    p = load_prices(key).set_index("ts")["close"]
    r = np.log(p).diff().abs()
    r = r[(p.index.to_series().diff() == pd.Timedelta(minutes=1)).to_numpy()]
    ny = r.index.tz_convert("America/New_York")
    df = pd.DataFrame({"r": r.to_numpy(), "year": ny.year,
                       "dst": (ny.tz_localize(None) - r.index.tz_localize(None)) == pd.Timedelta(hours=-4),
                       "m": ny.hour * 60 + ny.minute})
    prof = df.groupby(["year", "dst", "m"]).r.mean().unstack("m").reindex(columns=range(1440))
    pooled = df.groupby("m").r.mean().reindex(range(1440))
    for (y, d), row in prof.iterrows():
        ok = row.notna() & pooled.notna()
        corr = {L: np.corrcoef(row[ok], pooled.shift(L)[ok].fillna(pooled.mean()))[0, 1] for L in (-60, 0, 60)}
        best = max(corr, key=corr.get)
        rows.append({"asset": key, "year": y, "dst": d, "corr_shift_-60": corr[-60], "corr_shift_0": corr[0],
                     "corr_shift_+60": corr[60], "best_shift_min": best, "ok": best == 0})
tz = pd.DataFrame(rows)
tz.to_csv(TABLES / "data_timezone_check.csv", index=False, float_format="%.3f")
print(tz.groupby("asset").agg(ok=("ok", "sum"), n=("ok", "size"), min_corr0=("corr_shift_0", "min")))
print(tz[~tz.ok])

# 3. Cross-source check against Dukascopy on random days (rate-limited source)
random.seed(1)
cmp = []
for key, sym in [("eurusd", "EURUSD"), ("spx", "USA500IDXUSD")]:
    P = get_panel(key)
    days = random.sample(list(P.R.index[P.R.index.year >= 2015]), 6)
    hd = load_prices(key).set_index("ts")["close"]
    parts = []
    for d in days:
        for dd in (d - pd.Timedelta(days=1), d):
            x = dukascopy_day(sym, dd.date())
            if len(x):
                parts.append(x[x.volume > 0])
    if not parts:
        continue
    du = pd.concat(parts).drop_duplicates("ts").set_index("ts")["close"].sort_index()
    grid = du.resample("5min", label="right", closed="right").last().dropna().index
    ra = 100 * np.log(hd.reindex(grid, method="ffill")).diff()
    rb = 100 * np.log(du.reindex(grid, method="ffill")).diff()
    ok = (grid.to_series().diff() == pd.Timedelta(minutes=5)).to_numpy()
    cmp.append({"asset": key, **compare_sources(ra[ok], rb[ok])})
pd.DataFrame(cmp).to_csv(TABLES / "data_source_crosscheck.csv", index=False)
print(pd.DataFrame(cmp))
