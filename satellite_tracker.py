"""
satellite_tracker.py
--------------------
Real-time satellite position tracking module using Skyfield.

Computes live geodetic coordinates (latitude, longitude, altitude) and
ECI position for selected satellites. Designed for continuous or
on-demand tracking of ISRO satellites.

Features:
  - Live position computation from TLE data
  - Ground track generation for visualization
  - Satellite pass prediction over a ground station
  - Multi-source TLE fetching (Celestrak primary, Space-Track optional)

Author: BSERC Internship Project
"""

import os
import json
import time
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import requests
from skyfield.api import EarthSatellite, load, wgs84, utc

from orbit import fetch_tle, build_satellite, TLEFetchError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
SPACETRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_TLE_URL = (
    "https://www.space-track.org/basicspacedata/query/class/tle_latest/"
    "NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/limit/1/format/tle"
)

# TLE disk cache
TLE_CACHE_DIR = os.path.join("data", "tle_cache")
TLE_CACHE_TTL = 7200  # 2 hours

# In-memory TLE cache for the session
_tle_mem_cache: Dict[int, Tuple[str, str, str, float]] = {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SatellitePosition:
    """Instantaneous satellite position and metadata."""
    name: str
    norad_id: int
    timestamp_utc: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    velocity_kms: float
    # ECI coordinates (km)
    eci_x: float
    eci_y: float
    eci_z: float
    # Metadata
    satellite_type: str = ""
    agency: str = "ISRO"


@dataclass
class GroundTrack:
    """Ground track data for map visualization."""
    name: str
    norad_id: int
    latitudes: List[float] = field(default_factory=list)
    longitudes: List[float] = field(default_factory=list)
    altitudes: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TLE fetching with multi-source fallback and caching
# ---------------------------------------------------------------------------

def _ensure_tle_cache_dir():
    os.makedirs(TLE_CACHE_DIR, exist_ok=True)


def _tle_cache_path(norad_id: int) -> str:
    return os.path.join(TLE_CACHE_DIR, f"tle_{norad_id}.json")


def _load_cached_tle(norad_id: int) -> Optional[Tuple[str, str, str]]:
    """Load TLE from memory or disk cache if still fresh."""
    # Memory cache
    if norad_id in _tle_mem_cache:
        name, l1, l2, cached_at = _tle_mem_cache[norad_id]
        if time.time() - cached_at < TLE_CACHE_TTL:
            return name, l1, l2

    # Disk cache
    _ensure_tle_cache_dir()
    path = _tle_cache_path(norad_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if time.time() - data.get("cached_at", 0) < TLE_CACHE_TTL:
                name, l1, l2 = data["name"], data["line1"], data["line2"]
                _tle_mem_cache[norad_id] = (name, l1, l2, data["cached_at"])
                return name, l1, l2
        except Exception:
            pass
    return None


def _save_tle_cache(norad_id: int, name: str, l1: str, l2: str):
    """Save TLE to both memory and disk cache."""
    now = time.time()
    _tle_mem_cache[norad_id] = (name, l1, l2, now)
    _ensure_tle_cache_dir()
    path = _tle_cache_path(norad_id)
    try:
        with open(path, "w") as f:
            json.dump({
                "name": name, "line1": l1, "line2": l2,
                "cached_at": now, "norad_id": norad_id,
            }, f)
    except Exception:
        pass


def fetch_tle_multi_source(
    norad_id: int,
    use_spacetrack: bool = False,
    spacetrack_identity: str = "",
    spacetrack_password: str = "",
    timeout: int = 10,
) -> Tuple[str, str, str]:
    """
    Fetch TLE with multi-source fallback:
      1. Cache (memory → disk)
      2. Celestrak (primary, no auth required)
      3. Space-Track.org (optional, requires credentials)
      4. Hardcoded fallbacks (from orbit.py)

    Returns (name, line1, line2)
    """
    # 1. Check cache
    cached = _load_cached_tle(norad_id)
    if cached:
        return cached

    # 2. Try Celestrak
    try:
        name, l1, l2 = fetch_tle(norad_id, timeout=timeout)
        _save_tle_cache(norad_id, name, l1, l2)
        return name, l1, l2
    except TLEFetchError:
        pass

    # 3. Try Space-Track.org if credentials provided
    if use_spacetrack and spacetrack_identity and spacetrack_password:
        try:
            tle = _fetch_from_spacetrack(
                norad_id, spacetrack_identity, spacetrack_password, timeout
            )
            if tle:
                _save_tle_cache(norad_id, *tle)
                return tle
        except Exception:
            pass

    # 4. Use orbit.py fallbacks (will raise TLEFetchError if not found)
    name, l1, l2 = fetch_tle(norad_id, timeout=1)
    return name, l1, l2


def _fetch_from_spacetrack(
    norad_id: int, identity: str, password: str, timeout: int
) -> Optional[Tuple[str, str, str]]:
    """Fetch TLE from Space-Track.org REST API."""
    session = requests.Session()
    try:
        # Login
        login_resp = session.post(
            SPACETRACK_LOGIN_URL,
            data={"identity": identity, "password": password},
            timeout=timeout,
        )
        login_resp.raise_for_status()

        # Fetch TLE
        url = SPACETRACK_TLE_URL.format(norad_id=norad_id)
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()

        lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        elif len(lines) == 2:
            return f"NORAD-{norad_id}", lines[0], lines[1]
    except Exception:
        pass
    finally:
        session.close()
    return None


# ---------------------------------------------------------------------------
# Position computation
# ---------------------------------------------------------------------------

def compute_live_position(
    name: str,
    norad_id: int,
    line1: str,
    line2: str,
    satellite_type: str = "",
    at_time: Optional[dt.datetime] = None,
) -> SatellitePosition:
    """
    Compute the instantaneous geodetic position of a satellite.

    Parameters
    ----------
    name : str
        Satellite display name.
    norad_id : int
        NORAD catalog number.
    line1, line2 : str
        TLE lines.
    satellite_type : str
        Type tag (Earth Observation, Communication, etc.)
    at_time : datetime, optional
        If None, uses current UTC time.

    Returns
    -------
    SatellitePosition with lat/lon/alt and ECI coordinates.
    """
    ts = load.timescale()
    sat = EarthSatellite(line1, line2, name, ts)

    if at_time is None:
        at_time = dt.datetime.now(dt.timezone.utc)
    elif at_time.tzinfo is None:
        at_time = at_time.replace(tzinfo=utc)

    t = ts.from_datetime(at_time)

    geocentric = sat.at(t)
    subpoint = wgs84.subpoint(geocentric)

    pos_km = geocentric.position.km
    vel_kms = geocentric.velocity.km_per_s
    speed = float(np.linalg.norm(vel_kms))

    return SatellitePosition(
        name=name,
        norad_id=norad_id,
        timestamp_utc=at_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        latitude_deg=float(subpoint.latitude.degrees),
        longitude_deg=float(subpoint.longitude.degrees),
        altitude_km=float(subpoint.elevation.km),
        velocity_kms=speed,
        eci_x=float(pos_km[0]),
        eci_y=float(pos_km[1]),
        eci_z=float(pos_km[2]),
        satellite_type=satellite_type,
        agency="ISRO",
    )


def compute_ground_track(
    name: str,
    norad_id: int,
    line1: str,
    line2: str,
    duration_minutes: int = 90,
    step_seconds: int = 30,
) -> GroundTrack:
    """
    Compute the ground track (lat/lon over time) for map visualization.

    Parameters
    ----------
    duration_minutes : int
        How far ahead to project, starting from now.
    step_seconds : int
        Time resolution.

    Returns
    -------
    GroundTrack with lists of lat, lon, alt, and timestamps.
    """
    ts = load.timescale()
    sat = EarthSatellite(line1, line2, name, ts)

    start = dt.datetime.now(dt.timezone.utc)
    n_steps = max(2, int(duration_minutes * 60 / step_seconds))

    track = GroundTrack(name=name, norad_id=norad_id)

    for i in range(n_steps):
        at_time = start + dt.timedelta(seconds=i * step_seconds)
        t = ts.from_datetime(at_time)
        geocentric = sat.at(t)
        subpoint = wgs84.subpoint(geocentric)

        track.latitudes.append(float(subpoint.latitude.degrees))
        track.longitudes.append(float(subpoint.longitude.degrees))
        track.altitudes.append(float(subpoint.elevation.km))
        track.timestamps.append(at_time.strftime("%H:%M:%S"))

    return track


def get_all_satellite_positions(
    catalog_path: str = "data/satellites.json",
) -> List[SatellitePosition]:
    """
    Compute live positions for all satellites in the catalog.
    Skips satellites whose TLE cannot be fetched.
    """
    try:
        with open(catalog_path, "r") as f:
            data = json.load(f)
        satellites = data.get("satellites", [])
    except Exception:
        return []

    positions = []
    for sat_info in satellites:
        try:
            name, l1, l2 = fetch_tle_multi_source(sat_info["norad_id"])
            pos = compute_live_position(
                name=sat_info["name"],
                norad_id=sat_info["norad_id"],
                line1=l1,
                line2=l2,
                satellite_type=sat_info.get("type", ""),
            )
            positions.append(pos)
        except Exception:
            continue

    return positions
