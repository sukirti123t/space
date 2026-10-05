"""
nasa_api.py
-----------
NASA Near Earth Object (NEO) API integration module.

Fetches asteroid / NEO data from NASA's public NeoWs API
(api.nasa.gov) and classifies threat levels based on:
  - Estimated diameter
  - Close approach distance (in Lunar Distances / km)
  - Relative velocity
  - Whether the object is classified as Potentially Hazardous (PHA)

Uses disk + in-memory caching to avoid repeated API calls within a
session and across restarts.  Degrades gracefully with demo data
when no API key is available.

Author: BSERC Internship Project
"""

import os
import json
import time
import hashlib
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NASA_NEO_FEED_URL = "https://api.nasa.gov/neo/rest/v1/feed"
NASA_NEO_BROWSE_URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"

# Cache settings
CACHE_DIR = os.path.join("data", "cache")
NEO_CACHE_TTL_SECONDS = 3600  # 1 hour

# In-memory cache (session-level)
_mem_cache: Dict[str, Any] = {}


def _get_nasa_api_key() -> str:
    """
    Resolve NASA API key from multiple sources (priority order):
      1. Environment variable NASA_API_KEY
      2. Streamlit secrets (if running inside Streamlit)
      3. DEMO_KEY (rate-limited: 30 req/hour, 50 req/day)
    """
    key = os.environ.get("NASA_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("NASA_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return "DEMO_KEY"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NEOObject:
    """Represents a single Near Earth Object from NASA NeoWs."""
    id: str
    name: str
    nasa_jpl_url: str
    absolute_magnitude_h: float
    estimated_diameter_min_m: float
    estimated_diameter_max_m: float
    is_potentially_hazardous: bool
    close_approach_date: str
    close_approach_date_full: str
    relative_velocity_kms: float
    miss_distance_km: float
    miss_distance_lunar: float
    miss_distance_au: float
    orbiting_body: str
    risk_level: str = ""
    risk_score: float = 0.0

    def __post_init__(self):
        if not self.risk_level:
            self.risk_level, self.risk_score = classify_neo_risk(self)


@dataclass
class NEOFeedResult:
    """Aggregated result from a NEO feed query."""
    start_date: str
    end_date: str
    element_count: int
    objects: List[NEOObject] = field(default_factory=list)
    fetched_at: str = ""
    api_key_type: str = ""
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def classify_neo_risk(neo: NEOObject) -> tuple:
    """
    Multi-factor risk classification for a Near Earth Object.

    Factors:
      - Estimated size (avg diameter in meters)
      - Closest approach distance in Lunar Distances (LD)
      - Relative velocity
      - PHA designation from NASA

    Returns (risk_level: str, risk_score: float 0-100)
    """
    score = 0.0

    # --- Size factor (0-30 points) ---
    avg_diameter_m = (neo.estimated_diameter_min_m + neo.estimated_diameter_max_m) / 2
    if avg_diameter_m > 1000:
        score += 30
    elif avg_diameter_m > 500:
        score += 25
    elif avg_diameter_m > 140:
        score += 20
    elif avg_diameter_m > 50:
        score += 12
    elif avg_diameter_m > 10:
        score += 6
    else:
        score += 2

    # --- Distance factor (0-35 points) ---
    ld = neo.miss_distance_lunar
    if ld < 1.0:
        score += 35
    elif ld < 5.0:
        score += 28
    elif ld < 10.0:
        score += 20
    elif ld < 20.0:
        score += 12
    elif ld < 50.0:
        score += 5
    else:
        score += 1

    # --- Velocity factor (0-20 points) ---
    v = neo.relative_velocity_kms
    if v > 30:
        score += 20
    elif v > 20:
        score += 15
    elif v > 10:
        score += 10
    elif v > 5:
        score += 5
    else:
        score += 2

    # --- PHA designation (0-15 points) ---
    if neo.is_potentially_hazardous:
        score += 15

    # Classify
    if score >= 70:
        level = "CRITICAL"
    elif score >= 50:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    elif score >= 15:
        level = "LOW"
    else:
        level = "NEGLIGIBLE"

    return level, round(score, 1)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_key(start_date: str, end_date: str) -> str:
    raw = f"neo_feed_{start_date}_{end_date}"
    return hashlib.md5(raw.encode()).hexdigest()


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _load_disk_cache(key: str) -> Optional[dict]:
    _ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Check TTL
        cached_at = data.get("_cached_at", 0)
        if time.time() - cached_at > NEO_CACHE_TTL_SECONDS:
            return None  # expired
        return data
    except Exception:
        return None


def _save_disk_cache(key: str, data: dict):
    _ensure_cache_dir()
    data["_cached_at"] = time.time()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # Non-critical; proceed without disk cache


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------

def fetch_neo_feed(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
    timeout: int = 15,
) -> NEOFeedResult:
    """
    Fetch Near Earth Objects for a date range from NASA NeoWs API.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format. Max 7 days from start_date.
    force_refresh : bool
        Bypass cache if True.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    NEOFeedResult with parsed NEO objects.
    """
    cache_k = _cache_key(start_date, end_date)

    # Check memory cache first
    if not force_refresh and cache_k in _mem_cache:
        result = _mem_cache[cache_k]
        result.from_cache = True
        return result

    # Check disk cache
    if not force_refresh:
        disk_data = _load_disk_cache(cache_k)
        if disk_data:
            result = _parse_feed_response(disk_data, start_date, end_date)
            result.from_cache = True
            _mem_cache[cache_k] = result
            return result

    # Live API call
    api_key = _get_nasa_api_key()
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": api_key,
    }

    try:
        resp = requests.get(NASA_NEO_FEED_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            raise NASAAPIError(
                "NASA API key is invalid or rate-limited. "
                "Get a free key at https://api.nasa.gov"
            ) from e
        raise NASAAPIError(f"NASA API HTTP error: {e}") from e
    except requests.exceptions.Timeout:
        raise NASAAPIError("NASA API request timed out. Try again later.")
    except requests.exceptions.ConnectionError:
        raise NASAAPIError(
            "Cannot reach NASA API. Check your internet connection."
        )
    except Exception as e:
        raise NASAAPIError(f"Unexpected error fetching NEO data: {e}") from e

    # Cache the raw response
    _save_disk_cache(cache_k, raw)

    result = _parse_feed_response(raw, start_date, end_date)
    result.api_key_type = "DEMO_KEY" if api_key == "DEMO_KEY" else "USER_KEY"
    result.fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    _mem_cache[cache_k] = result
    return result


def _parse_feed_response(
    raw: dict, start_date: str, end_date: str
) -> NEOFeedResult:
    """Parse raw NASA API JSON into structured NEOFeedResult."""
    objects = []
    neo_data = raw.get("near_earth_objects", {})

    for date_str, neo_list in neo_data.items():
        for neo in neo_list:
            try:
                diameter = neo.get("estimated_diameter", {})
                meters = diameter.get("meters", {})
                close_approach = neo.get("close_approach_data", [])

                if not close_approach:
                    continue

                ca = close_approach[0]  # Closest approach for this date
                rel_vel = ca.get("relative_velocity", {})
                miss_dist = ca.get("miss_distance", {})

                obj = NEOObject(
                    id=str(neo.get("id", "")),
                    name=neo.get("name", "Unknown"),
                    nasa_jpl_url=neo.get("nasa_jpl_url", ""),
                    absolute_magnitude_h=float(
                        neo.get("absolute_magnitude_h", 0)
                    ),
                    estimated_diameter_min_m=float(
                        meters.get("estimated_diameter_min", 0)
                    ),
                    estimated_diameter_max_m=float(
                        meters.get("estimated_diameter_max", 0)
                    ),
                    is_potentially_hazardous=bool(
                        neo.get("is_potentially_hazardous_asteroid", False)
                    ),
                    close_approach_date=ca.get(
                        "close_approach_date", date_str
                    ),
                    close_approach_date_full=ca.get(
                        "close_approach_date_full", ""
                    ),
                    relative_velocity_kms=float(
                        rel_vel.get("kilometers_per_second", 0)
                    ),
                    miss_distance_km=float(
                        miss_dist.get("kilometers", 0)
                    ),
                    miss_distance_lunar=float(
                        miss_dist.get("lunar", 0)
                    ),
                    miss_distance_au=float(
                        miss_dist.get("astronomical", 0)
                    ),
                    orbiting_body=ca.get("orbiting_body", "Earth"),
                )
                objects.append(obj)
            except (KeyError, ValueError, TypeError):
                continue  # Skip malformed entries

    # Sort by risk score descending
    objects.sort(key=lambda o: o.risk_score, reverse=True)

    return NEOFeedResult(
        start_date=start_date,
        end_date=end_date,
        element_count=len(objects),
        objects=objects,
    )


# ---------------------------------------------------------------------------
# Demo / fallback data
# ---------------------------------------------------------------------------

DEMO_NEO_OBJECTS = [
    NEOObject(
        id="demo_1", name="(2024 AA1) Demo Asteroid",
        nasa_jpl_url="https://ssd.jpl.nasa.gov/",
        absolute_magnitude_h=22.1,
        estimated_diameter_min_m=90, estimated_diameter_max_m=200,
        is_potentially_hazardous=True,
        close_approach_date="2026-07-17",
        close_approach_date_full="2026-Jul-17 09:30",
        relative_velocity_kms=18.5,
        miss_distance_km=1_850_000, miss_distance_lunar=4.81,
        miss_distance_au=0.0124, orbiting_body="Earth",
    ),
    NEOObject(
        id="demo_2", name="(2023 BZ3) Demo NEO",
        nasa_jpl_url="https://ssd.jpl.nasa.gov/",
        absolute_magnitude_h=25.3,
        estimated_diameter_min_m=25, estimated_diameter_max_m=56,
        is_potentially_hazardous=False,
        close_approach_date="2026-07-18",
        close_approach_date_full="2026-Jul-18 14:12",
        relative_velocity_kms=9.2,
        miss_distance_km=7_420_000, miss_distance_lunar=19.3,
        miss_distance_au=0.0496, orbiting_body="Earth",
    ),
    NEOObject(
        id="demo_3", name="(2025 CD7) Demo Rock",
        nasa_jpl_url="https://ssd.jpl.nasa.gov/",
        absolute_magnitude_h=28.0,
        estimated_diameter_min_m=5, estimated_diameter_max_m=12,
        is_potentially_hazardous=False,
        close_approach_date="2026-07-19",
        close_approach_date_full="2026-Jul-19 21:45",
        relative_velocity_kms=4.1,
        miss_distance_km=32_000_000, miss_distance_lunar=83.2,
        miss_distance_au=0.214, orbiting_body="Earth",
    ),
]


def get_demo_feed(start_date: str, end_date: str) -> NEOFeedResult:
    """Return realistic-looking demo data when the API is unavailable."""
    return NEOFeedResult(
        start_date=start_date,
        end_date=end_date,
        element_count=len(DEMO_NEO_OBJECTS),
        objects=DEMO_NEO_OBJECTS,
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        api_key_type="DEMO_DATA",
        from_cache=False,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def fetch_neo_threat_data(
    days_ahead: int = 7,
    force_refresh: bool = False,
) -> NEOFeedResult:
    """
    High-level convenience: fetch NEO data from today through `days_ahead`.
    Falls back to demo data on any failure.
    """
    today = dt.date.today()
    end = today + dt.timedelta(days=min(days_ahead, 7))  # API max 7 days
    start_str = today.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    try:
        return fetch_neo_feed(start_str, end_str, force_refresh=force_refresh)
    except NASAAPIError:
        return get_demo_feed(start_str, end_str)
    except Exception:
        return get_demo_feed(start_str, end_str)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class NASAAPIError(Exception):
    """Raised when the NASA API call fails."""
