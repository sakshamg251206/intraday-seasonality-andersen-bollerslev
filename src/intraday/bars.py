"""Turn a price stream into A&B's R_{t,n} panel: T trading days x N intervals.

Conventions
-----------
* Price at grid time g = the last observed close with timestamp <= g
  ("previous-tick"). A&B linearly interpolated between the quotes either side
  of g, which uses a quote *after* g; previous-tick is the look-ahead-free
  modern standard and differs only by a fraction of a quote interval.
* R_{t,n} = 100 * (log p_{t,n} - log p_{t,n-1})   (percent, as in A&B).
* An interval is "fresh" if at least one price update fell inside it. Days with
  too few fresh intervals (holidays, outages) are dropped and logged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BAR_MINUTES, Asset

STALE_LIMIT = pd.Timedelta(minutes=30)  # older than this -> price treated as missing
MIN_FRESH = 0.80                        # min fraction of fresh intervals per day


@dataclass
class Panel:
    """Intraday returns for one asset. All frames share the same day index."""
    asset: Asset
    R: pd.DataFrame        # T x N percent log returns
    fresh: pd.DataFrame    # T x N bool, price updated inside the interval
    volume: pd.DataFrame   # T x N traded volume (NaN if source has none)
    activity: pd.DataFrame # T x N number of source bars in the interval (HistData:
                           # active 1-min bars, 0-5 = quote-activity proxy)
    daily: pd.Series       # daily percent log return incl. overnight / weekend gap
    dropped: pd.Series     # dates dropped by the quality filter -> reason

    @property
    def N(self) -> int:
        return self.R.shape[1]


def _grid(asset: Asset, days: pd.DatetimeIndex) -> pd.DataFrame:
    """UTC timestamps of the N+1 grid points (n = 0..N) for each trading day.

    The day label is the local calendar date on which the day *ends*.
    """
    step = pd.Timedelta(minutes=BAR_MINUTES)
    h, m = map(int, asset.day_start.split(":"))
    span = step * asset.n_intervals
    # local start of the day; if the session crosses midnight it starts on the previous date
    start_offset = pd.Timedelta(hours=h, minutes=m)
    if start_offset + span > pd.Timedelta(days=1):
        start_offset -= pd.Timedelta(days=1)
    starts = (days + start_offset).tz_localize(asset.tz, nonexistent="shift_forward", ambiguous=False)
    offsets = np.arange(asset.n_intervals + 1) * np.timedelta64(BAR_MINUTES, "m")
    g = starts.tz_convert("UTC").values[:, None] + offsets[None, :]
    return pd.DataFrame(g, index=days)


def candidate_days(asset: Asset, prices: pd.DataFrame) -> pd.DatetimeIndex:
    local = prices["ts"].dt.tz_convert(asset.tz)
    days = pd.DatetimeIndex(sorted(local.dt.normalize().dt.tz_localize(None).unique()))
    if asset.key == "eurusd":
        # A&B: drop Fri 21:00 - Sun 21:00 GMT, i.e. days ending on Sat or Sun.
        days = days[days.dayofweek < 5]
    elif asset.key == "spx":
        days = days[days.dayofweek < 5]
    return days


def build_panel(asset: Asset, prices: pd.DataFrame) -> Panel:
    prices = prices.dropna(subset=["close"]).sort_values("ts")
    days = candidate_days(asset, prices)
    days = days[(days.year >= asset.first_year) & (days.year <= asset.last_year)]
    grid = _grid(asset, days)
    m = _asof(grid, prices[["ts", "close"]].assign(obs_ts=prices["ts"]))
    age = m["g"] - m["obs_ts"]
    logp = np.log(m["close"].where(age <= STALE_LIMIT)).to_numpy().reshape(grid.shape).copy()
    last_obs = m["obs_ts"].values.reshape(grid.shape)
    # Session open: if the price at g_0 is stale (e.g. the Sunday FX open,
    # where the last quote is Friday's), use the first quote inside interval 1
    # as the base. This keeps the weekend gap out of R_1, as A&B did.
    first = _first_after(grid.iloc[:, 0], prices[["ts", "close"]])
    fill = np.isnan(logp[:, 0])
    logp[fill, 0] = np.log(first.to_numpy())[fill]

    R = pd.DataFrame(100 * np.diff(logp, axis=1), index=days,
                     columns=pd.RangeIndex(1, asset.n_intervals + 1, name="n"))
    # fresh: the latest observation at g_n arrived after g_{n-1}
    fresh = pd.DataFrame(last_obs[:, 1:] > grid.values[:, :-1], index=days, columns=R.columns)

    volume = _interval_volume(prices, grid, R)
    activity = _interval_volume(prices.assign(volume=1.0), grid, R)

    reasons = {}
    frac = fresh.mean(axis=1)
    for d in days[frac < MIN_FRESH]:
        reasons[d] = f"fresh={frac[d]:.2f}"
    for d in days[R.isna().any(axis=1)]:
        reasons.setdefault(d, "missing price")
    if asset.key == "spx":
        for d in days[_is_early_close(days)]:
            reasons.setdefault(d, "early close")
    if asset.key != "btc":  # holidays are ordinary days in a 24/7 market
        for d in days[(days.month == 12) & (days.day == 25) | (days.month == 1) & (days.day == 1)]:
            reasons.setdefault(d, "Christmas/New Year")
    # Daily close-to-close return (previous valid day end -> end of day t),
    # including the overnight/weekend move that the intraday panel excludes.
    # Computed before the quality filter so it always spans one trading day.
    close = pd.Series(logp[:, -1], index=days).dropna()   # skip holidays: last VALID close
    daily_all = (100 * close.diff()).reindex(days)
    for d in days[daily_all.isna().to_numpy()]:
        reasons.setdefault(d, "no previous close")
    dropped = pd.Series(reasons, dtype=object).sort_index()
    keep = ~days.isin(dropped.index)

    daily = daily_all[keep]
    R, fresh, volume, activity = R[keep], fresh[keep], volume[keep], activity[keep]
    return Panel(asset, R, fresh, volume, activity, daily, dropped)


def _asof(grid: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Last observation at or before every grid point (grid rows are chronological)."""
    flat = pd.DataFrame({"g": pd.to_datetime(grid.values.ravel(), utc=True)})
    return pd.merge_asof(flat, obs, left_on="g", right_on="ts", direction="backward")


def _first_after(g0: pd.Series, obs: pd.DataFrame) -> pd.Series:
    """First observed price in (g0, g0 + one bar] for each day, NaN if none."""
    left = pd.DataFrame({"g": pd.to_datetime(g0.to_numpy(), utc=True)})
    m = pd.merge_asof(left, obs, left_on="g", right_on="ts", direction="forward",
                      allow_exact_matches=False, tolerance=pd.Timedelta(minutes=BAR_MINUTES))
    return pd.Series(m["close"].to_numpy(), index=g0.index)


def _interval_volume(prices: pd.DataFrame, grid: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    if prices["volume"].isna().all():
        return pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    # cumulative volume at each grid point -> differences give volume inside each interval
    m = _asof(grid, prices[["ts"]].assign(cv=prices["volume"].fillna(0).cumsum()))
    cv = m["cv"].fillna(0).to_numpy().reshape(grid.shape)
    return pd.DataFrame(np.diff(cv, axis=1), index=R.index, columns=R.columns)


def _is_early_close(days: pd.DatetimeIndex) -> np.ndarray:
    """NYSE 13:00 early closes: Jul 3, day after Thanksgiving, Dec 24 (weekdays)."""
    thanksgiving_fri = (days.month == 11) & (days.dayofweek == 4) & (days.day >= 23) & (days.day <= 29)
    jul3 = (days.month == 7) & (days.day == 3)
    dec24 = (days.month == 12) & (days.day == 24)
    return np.asarray(thanksgiving_fri | jul3 | dec24)


def get_panel(key: str, rebuild: bool = False) -> Panel:
    """Build (or load the cached) panel for an asset."""
    import pickle

    from .config import ASSETS, PROCESSED
    from .data import load_prices

    path = PROCESSED / f"{key}_panel.pkl"
    if path.exists() and not rebuild:
        return pickle.loads(path.read_bytes())
    P = build_panel(ASSETS[key], load_prices(key))
    path.write_bytes(pickle.dumps(P))
    return P


def interval_end_utc(P: Panel) -> pd.DataFrame:
    """UTC end time of every (day, interval) cell of a panel."""
    g = _grid(P.asset, pd.DatetimeIndex(P.R.index))
    return pd.DataFrame(g.values[:, 1:], index=P.R.index, columns=P.R.columns)
