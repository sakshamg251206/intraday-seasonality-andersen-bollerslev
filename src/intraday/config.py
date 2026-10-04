"""Project-wide constants: paths, asset universe, sample splits.

Everything that defines *what* is studied lives here so that the analysis
scripts never hard-code a date or a symbol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"

BAR_MINUTES = 5  # A&B's sampling interval


@dataclass(frozen=True)
class Asset:
    key: str            # short name used in file names
    label: str          # name used in figures
    source: str         # histdata | binance | dukascopy
    symbol: str         # symbol at the source
    first_year: int
    last_year: int
    # Session definition, in the asset's *local* clock (see bars.py).
    tz: str             # time zone that defines the trading day
    day_start: str      # "HH:MM" local time at which interval n=1 begins
    n_intervals: int    # N, number of 5-min intervals per day
    # Chronological split: train <= train_end < valid <= valid_end < test
    train_end: int
    valid_end: int


ASSETS: dict[str, Asset] = {
    # A&B's DM-$ analogue. A&B's day ran 21:00-21:00 GMT and dropped Fri 21:00 -
    # Sun 21:00 GMT. The modern FX week runs Sun 17:00 - Fri 17:00 New York
    # (verified in the data: every year opens 17:00 NY Sunday, closes 17:00 NY
    # Friday), which is 21:00 GMT in summer but 22:00 GMT in winter. The day is
    # therefore defined 17:00-17:00 New York, the market's own rollover.
    "eurusd": Asset("eurusd", "EUR/USD", "histdata", "EURUSD", 2004, 2025,
                    "America/New_York", "17:00", 288, 2016, 2019),
    # A&B's S&P 500 futures analogue. The paper keeps 08:35-15:15 Chicago time
    # (= 09:35-16:15 New York): the cash session minus the overnight-contaminated
    # first 5 minutes, plus the 15 minutes of futures trading after the cash
    # close. SPXUSD trades ~23h/day, so the identical 80-interval window is used.
    "spx": Asset("spx", "S&P 500", "histdata", "SPXUSD", 2010, 2025,
                 "America/New_York", "09:35", 80, 2016, 2019),
    # Original-research asset: a 24/7 market with genuine exchange volume.
    "btc": Asset("btc", "BTC/USDT", "binance", "BTCUSDT", 2018, 2025,
                 "UTC", "00:00", 288, 2021, 2022),
}
