"""Fetch a weekly-rotating sample of Dukascopy 1-min candles (with Dukascopy's
own volume) for 2024: one weekday per week, Monday..Friday in rotation.
Dukascopy rate-limits heavily, so this runs slowly and is fully resumable."""
from datetime import date, timedelta

from intraday.data import dukascopy_day

day, i = date(2024, 1, 1), 0
while day.year == 2024:
    d = day + timedelta(days=i % 5)            # rotate weekday
    for sym in ("EURUSD", "USA500IDXUSD"):
        for dd in (d - timedelta(days=1), d):  # previous UTC day covers the NY session start
            dukascopy_day(sym, dd)
    print(d, flush=True)
    day += timedelta(days=7)
    i += 1
