"""Step 1: download raw data and build canonical price files (idempotent, resumable)."""
import sys

from intraday.config import ASSETS
from intraday.data import build_minute_file

for key in sys.argv[1:] or ASSETS:
    print(key, "->", build_minute_file(ASSETS[key]), flush=True)
