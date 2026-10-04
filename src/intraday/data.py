"""Data acquisition: free, key-less public sources, cached on disk.

Canonical output of this module (one parquet per asset in data/processed):

    ts      UTC timestamp of the END of the bar, i.e. the time at which `close`
            is the latest observed price. Using the bar end avoids an easy
            look-ahead bug: the close of the bar labelled 10:00 is only known at
            10:00, never before.
    close   last price in the bar
    volume  traded volume (NaN when the source has none)

Sources
-------
HistData.com   1-minute bid bars. Documented as EST without DST, but verified
               empirically to be New York local time with DST (see
               histdata_year and scripts/02_validate.py).
Binance        5-minute spot klines from data.binance.vision, UTC.
Dukascopy      1-minute bid/ask candles with Dukascopy's own volume, UTC.
               Heavily rate limited, so only used for a validation subset.
"""
from __future__ import annotations

import io
import lzma
import re
import struct
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import ASSETS, PROCESSED, RAW, Asset

_UA = {"User-Agent": "Mozilla/5.0 (research; intraday-seasonality)"}


def _get(url: str, *, retries: int = 6, method: str = "GET", **kw) -> requests.Response:
    """HTTP with exponential back-off on 429/5xx and network errors."""
    for attempt in range(retries):
        try:
            r = requests.request(method, url, timeout=120, headers={**_UA, **kw.pop("headers", {})}, **kw)
            if r.status_code not in (429, 500, 502, 503, 504):
                return r
        except requests.RequestException:
            pass
        time.sleep(min(2 ** attempt * 5, 300))
    raise RuntimeError(f"giving up on {url}")


# --------------------------------------------------------------------------- HistData
def _histdata_zip(pair: str, year: int) -> Path:
    out = RAW / "histdata" / f"{pair}_{year}.zip"
    if out.exists():
        return out
    page = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"
    s = requests.Session()
    s.headers.update(_UA)
    html = s.get(page, timeout=120).text
    form = dict(re.findall(r'<input type="hidden" name="(\w+)" id="\w+" value="([^"]*)"', html))
    if "tk" not in form:
        raise RuntimeError(f"HistData form not found for {pair} {year}")
    r = s.post("https://www.histdata.com/get.php", data=form, headers={"Referer": page}, timeout=300)
    if not r.content.startswith(b"PK"):
        raise RuntimeError(f"HistData returned no zip for {pair} {year}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


def histdata_year(pair: str, year: int) -> pd.DataFrame:
    with zipfile.ZipFile(_histdata_zip(pair, year)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name), sep=";", header=None,
                         names=["t", "open", "high", "low", "close", "volume"])
    raw = pd.to_datetime(df["t"], format="%Y%m%d %H%M%S")   # bar START, source clock
    ny = raw + _clock_offset(raw, _WEEKLY_OPEN[pair])
    # The repeated 01:00-01:59 hour at the autumn change is ambiguous -> NaT,
    # dropped; it is a quiet hour.
    start = ny.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    out = pd.DataFrame({"ts": (start + pd.Timedelta(minutes=1)).dt.tz_convert("UTC"),
                        "close": df["close"].astype(float),
                        "volume": np.nan})
    return out.dropna(subset=["ts"])


# Known New-York time of the first bar of each trading week (FX: 17:00 Sunday;
# CME Globex equity futures: 18:00 Sunday).
_WEEKLY_OPEN = {"EURUSD": 17, "SPXUSD": 18}


def _clock_offset(raw: pd.Series, open_hour: int) -> pd.Series:
    """Offset to add to HistData's clock to obtain New York local time.

    HistData documents "EST without DST", but the data show New York local
    time in most weeks, and - from 2019 on - a clock one hour BEHIND New York
    in the weeks when the US is on daylight time but Europe is not (the source
    evidently follows European DST dates). The offset is therefore estimated
    week by week from the observed weekly open, whose true New York time is
    known; weeks without an observed open fall back to the rule above.
    Verified independently by the event-time check in scripts/02_validate.py.
    """
    week = (raw - pd.Timedelta(hours=12)).dt.to_period("W-SAT")   # Sun 12:00 -> Sun 12:00
    sunday_pm = (raw.dt.dayofweek == 6) & (raw.dt.hour >= 12)
    first = raw[sunday_pm].groupby(week[sunday_pm]).min()
    est = ((open_hour * 60 - (first.dt.hour * 60 + first.dt.minute)) / 60).round()
    est = est[est.isin([0, 1])]
    # fallback rule
    mid = (raw.dt.normalize() + pd.Timedelta(hours=12))
    us = mid.dt.tz_localize("America/New_York").map(lambda t: bool(t.dst()))
    eu = mid.dt.tz_localize("Europe/London").map(lambda t: bool(t.dst()))
    rule = ((us & ~eu) & (raw.dt.year >= 2019)).astype(float)
    hours = week.map(est).astype(float).fillna(rule)
    return pd.to_timedelta(hours, unit="h")


# --------------------------------------------------------------------------- Binance
def binance_month(symbol: str, year: int, month: int, interval: str = "5m") -> pd.DataFrame:
    name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    out = RAW / "binance" / name
    if not out.exists():
        r = _get(f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{name}")
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
    with zipfile.ZipFile(out) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]  # some files carry a header row
    t = df[0].astype("int64")
    # Binance switched spot kline timestamps from ms to us on 2025-01-01.
    unit = np.where(t > 10 ** 14, "us", "ms")
    start = pd.to_datetime(np.where(unit == "us", t // 1000, t), unit="ms", utc=True)
    return pd.DataFrame({"ts": start + pd.Timedelta(interval.replace("m", "min")),
                         "close": df[4].astype(float).to_numpy(),
                         "volume": df[5].astype(float).to_numpy()})


# --------------------------------------------------------------------------- Dukascopy
_DUKA_POINT = {"EURUSD": 1e-5, "USA500IDXUSD": 1e-3}


def dukascopy_day(symbol: str, day: date, side: str = "BID") -> pd.DataFrame:
    """1-minute candles for one UTC day. Columns: ts, close, volume."""
    out = RAW / "dukascopy" / symbol / f"{day:%Y%m%d}_{side}.bi5"
    if not out.exists():
        url = (f"https://datafeed.dukascopy.com/datafeed/{symbol}/{day.year}/"
               f"{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5")
        r = _get(url)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content if r.status_code == 200 else b"")
        time.sleep(2.0)  # be polite: Dukascopy rate-limits aggressively
    raw = out.read_bytes()
    if not raw:
        return pd.DataFrame(columns=["ts", "close", "volume"])
    raw = lzma.decompress(raw)
    rec = np.array(struct.unpack(f">{len(raw) // 24 * 6}i", raw), dtype=np.int64).reshape(-1, 6)
    # last field is a big-endian float32, re-read it properly
    vol = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5].astype(float)
    start = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(rec[:, 0], unit="s")
    return pd.DataFrame({"ts": start + pd.Timedelta(minutes=1),
                         "close": rec[:, 2] * _DUKA_POINT[symbol],
                         "volume": vol})


# --------------------------------------------------------------------------- assemble
def build_minute_file(asset: Asset) -> Path:
    """Download (if needed) and write the canonical price file for an asset."""
    out = PROCESSED / f"{asset.key}_prices.parquet"
    if asset.source == "histdata":
        parts = [histdata_year(asset.symbol, y) for y in range(asset.first_year, asset.last_year + 1)]
    elif asset.source == "binance":
        parts = [binance_month(asset.symbol, y, m)
                 for y in range(asset.first_year, asset.last_year + 1) for m in range(1, 13)]
    else:
        raise ValueError(asset.source)
    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates("ts").sort_values("ts").reset_index(drop=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_prices(key: str) -> pd.DataFrame:
    path = PROCESSED / f"{key}_prices.parquet"
    if not path.exists():
        build_minute_file(ASSETS[key])
    return pd.read_parquet(path)
