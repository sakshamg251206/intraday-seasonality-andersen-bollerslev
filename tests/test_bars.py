"""Panel construction: no look-ahead, correct session-open handling."""
import numpy as np
import pandas as pd

from intraday.bars import build_panel
from intraday.config import Asset

ASSET = Asset("eurusd", "TEST", "binance", "X", 2024, 2024, "America/New_York", "17:00", 288, 2024, 2024)


def minute_prices(start, end, rng):
    ts = pd.date_range(start, end, freq="1min", tz="America/New_York").tz_convert("UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(ts))))
    return pd.DataFrame({"ts": ts, "close": close, "volume": np.nan})


def test_returns_use_only_past_prices_and_skip_weekend_gap():
    rng = np.random.default_rng(1)
    # Friday session ends 17:00, market reopens Sunday 17:01 with a +5% weekend gap
    fri = minute_prices("2024-03-07 17:01", "2024-03-08 17:00", rng)
    sun = minute_prices("2024-03-10 17:01", "2024-03-11 17:00", rng)
    sun["close"] *= 1.05
    P = build_panel(ASSET, pd.concat([fri, sun], ignore_index=True))
    mon = pd.Timestamp("2024-03-11")
    assert mon in P.R.index
    # the 5% weekend jump must not appear in the first interval of Monday's session
    assert abs(P.R.loc[mon, 1]) < 0.5
    # previous-tick: the price at a grid point is the last observation at or before it
    prices = sun.set_index("ts")["close"]
    g = pd.Timestamp("2024-03-11 09:00", tz="America/New_York").tz_convert("UTC")
    n = int((g - pd.Timestamp("2024-03-10 17:00", tz="America/New_York").tz_convert("UTC")) / pd.Timedelta("5min"))
    expected = 100 * (np.log(prices[:g].iloc[-1]) - np.log(prices[:g - pd.Timedelta("5min")].iloc[-1]))
    assert np.isclose(P.R.loc[mon, n], expected)
