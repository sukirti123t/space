"""
utils/tle_fetch.py
-------------------
Standalone helper script to bulk-fetch TLEs for every satellite/debris
object listed in data/satellites.json and data/debris_catalog.csv, and
cache them locally as tle_cache.json. Useful for offline demos or to
refresh the fallback TLEs in orbit.py.

Run:
    python utils/tle_fetch.py
"""

import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CELESTRAK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"


def fetch_one(norad_id: int):
    url = CELESTRAK_TLE_URL.format(norad_id=norad_id)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"Malformed TLE for {norad_id}")
    return lines[0], lines[1], lines[2]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data", "satellites.json")) as f:
        sats = json.load(f)["satellites"]
    debris_df = pd.read_csv(os.path.join(root, "data", "debris_catalog.csv"))

    cache = {}
    targets = [(s["name"], s["norad_id"]) for s in sats] + \
              [(row["name"], int(row["norad_id"])) for _, row in debris_df.iterrows()]

    for name, norad_id in targets:
        try:
            tle_name, l1, l2 = fetch_one(norad_id)
            cache[str(norad_id)] = {"name": tle_name, "line1": l1, "line2": l2}
            print(f"[OK]   {name} ({norad_id})")
        except Exception as e:
            print(f"[FAIL] {name} ({norad_id}): {e}")
        time.sleep(0.3)  # be polite to the API

    out_path = os.path.join(root, "data", "tle_cache.json")
    with open(out_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nSaved {len(cache)} TLEs to {out_path}")


if __name__ == "__main__":
    main()
