"""Data validation: checks that would silently corrupt every result if wrong."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bars import Panel


def spike_minute_utc(prices: pd.DataFrame, local_tz: str, window_utc=("12:00", "16:00")) -> pd.DataFrame:
    """UTC minute with the largest mean |1-min return| inside a window, split by
    whether the local zone is on daylight-saving time.

    A known local-time event (cash open 09:30 NY; US data at 08:30 NY) must move
    by exactly one hour in UTC between winter and summer. If the source's clock
    were mis-specified the spike would *not* move, or move the wrong way.
    """
    p = prices.set_index("ts")["close"]
    r = np.log(p).diff().abs()
    gap = p.index.to_series().diff() == pd.Timedelta(minutes=1)
    r = r[gap.to_numpy()]
    local = r.index.tz_convert(local_tz)
    # UTC offset of the local zone; anything other than standard time = DST
    offset = local.tz_localize(None) - r.index.tz_localize(None)
    is_dst = offset != offset.min()
    minute = r.index.strftime("%H:%M")
    df = pd.DataFrame({"r": r.to_numpy(), "dst": is_dst, "minute": minute})
    df = df[(df.minute >= window_utc[0]) & (df.minute < window_utc[1])]
    prof = df.groupby(["dst", "minute"])["r"].mean()
    out = prof.groupby(level="dst").idxmax().map(lambda x: x[1]).rename("spike_utc").to_frame()
    out["ratio_to_window_median"] = prof.groupby(level="dst").max() / prof.groupby(level="dst").median()
    return out


def panel_report(P: Panel) -> dict:
    R = P.R
    return {
        "asset": P.asset.label,
        "first_day": R.index.min().date().isoformat(),
        "last_day": R.index.max().date().isoformat(),
        "days": len(R), "N": P.N, "obs": R.size,
        "days_dropped": len(P.dropped),
        "fresh_share": float(P.fresh.to_numpy().mean()),
        "zero_return_share": float((R == 0).to_numpy().mean()),
        "max_abs_return_pct": float(R.abs().to_numpy().max()),
        "intraday_sum_vs_daily_corr": float(np.corrcoef(R.sum(axis=1), P.daily)[0, 1]),
    }


def compare_sources(a: pd.Series, b: pd.Series) -> dict:
    """Agreement of two 5-min return series on their common timestamps."""
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    return {"n": len(j), "corr": float(j.corr().iloc[0, 1]),
            "abs_corr": float(j.abs().corr().iloc[0, 1]),
            "sd_ratio": float(j.iloc[:, 0].std() / j.iloc[:, 1].std())}
