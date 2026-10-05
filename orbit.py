"""
orbit.py
--------
Orbital mechanics and collision (conjunction) risk analysis module.

Uses Skyfield (SGP4 propagator) to load Two-Line Element (TLE) sets for
real satellites/debris, propagate their positions over a time window,
and estimate the closest approach ("conjunction") between two objects.

A simplified probability-of-collision heuristic is provided for
demonstration/educational purposes. It is NOT a substitute for
operational conjunction assessment tools used by ISRO/ISTRAC, NASA CARA,
or 18th Space Defense Squadron, which rely on full covariance-based
Pc (Probability of Collision) computation.

Author: BSERC Internship Project
"""

import json
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional

import requests
import numpy as np
from skyfield.api import EarthSatellite, load, utc

CELESTRAK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"

# Fallback / cached TLEs so the demo NEVER breaks without internet access.
# (Illustrative TLEs - epoch may be stale, clearly fine for a UI demo.)
FALLBACK_TLES = {
    44804: (  # Cartosat-3
        "CARTOSAT-3",
        "1 44804U 19081A   24001.50000000  .00000023  00000-0  12345-4 0  9992",
        "2 44804  97.4600  50.1234 0011234  85.1234 275.0123 15.19000000123456",
    ),
    44233: (  # RISAT-2B
        "RISAT-2B",
        "1 44233U 19028A   24001.50000000  .00000031  00000-0  15678-4 0  9995",
        "2 44233  37.0000 120.4321 0012345  95.4321 264.6789 15.14000000234567",
    ),
    40269: (  # PSLV debris
        "PSLV R/B DEB",
        "1 40269U 14070B   24001.50000000  .00000045  00000-0  20123-4 0  9991",
        "2 40269  98.2000  15.6789 0025678 150.1234 210.9876 14.85000000345678",
    ),
}


class TLEFetchError(Exception):
    """Raised when a TLE cannot be retrieved from Celestrak or cache."""


@dataclass
class ConjunctionResult:
    object_a: str
    object_b: str
    closest_approach_time: dt.datetime
    miss_distance_km: float
    relative_speed_kms: float
    risk_level: str
    pseudo_probability: float
    combined_hard_body_radius_km: float


def fetch_tle(norad_id: int, timeout: int = 8) -> tuple:
    """
    Fetch a live TLE from Celestrak for the given NORAD catalog ID.
    Falls back to a cached TLE if the network call fails, so the app
    keeps working offline / during demos.

    Returns: (name, line1, line2)
    """
    try:
        url = CELESTRAK_TLE_URL.format(norad_id=norad_id)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            raise TLEFetchError(f"Malformed TLE response for NORAD {norad_id}")
        name, line1, line2 = lines[0], lines[1], lines[2]
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            raise TLEFetchError(f"Unexpected TLE format for NORAD {norad_id}")
        return name, line1, line2
    except Exception as e:
        if norad_id in FALLBACK_TLES:
            return FALLBACK_TLES[norad_id]
        raise TLEFetchError(
            f"Could not fetch TLE for NORAD {norad_id} and no cached fallback exists ({e})"
        )


def build_satellite(name: str, line1: str, line2: str, ts=None) -> EarthSatellite:
    """Construct a Skyfield EarthSatellite from TLE lines."""
    if ts is None:
        ts = load.timescale()
    return EarthSatellite(line1, line2, name, ts)


def propagate_positions(sat: EarthSatellite, start: dt.datetime, end: dt.datetime,
                         step_seconds: int = 30, ts=None):
    """
    Propagate a satellite's ECI position over [start, end] at fixed steps.
    Returns (times_list, positions_km ndarray of shape (N,3), velocities_kms ndarray)
    """
    if ts is None:
        ts = load.timescale()

    if start.tzinfo is None:
        start = start.replace(tzinfo=utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=utc)

    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        raise ValueError("end time must be after start time")

    n_steps = max(2, int(total_seconds // step_seconds))
    times = [start + dt.timedelta(seconds=i * step_seconds) for i in range(n_steps)]
    sf_times = ts.from_datetimes(times)

    geocentric = sat.at(sf_times)
    positions = geocentric.position.km.T          # shape (N,3)
    velocities = geocentric.velocity.km_per_s.T    # shape (N,3)
    return times, positions, velocities


def _risk_classify(miss_distance_km: float) -> str:
    """Simple threshold-based classification (illustrative, not operational)."""
    if miss_distance_km < 1.0:
        return "HIGH"
    elif miss_distance_km < 5.0:
        return "MEDIUM"
    elif miss_distance_km < 25.0:
        return "LOW"
    return "NEGLIGIBLE"


def _pseudo_probability(miss_distance_km: float, combined_radius_km: float,
                         relative_speed_kms: float) -> float:
    """
    Simplified, education-oriented approximation of collision likelihood.
    Uses an exponential decay of miss distance normalized by combined
    hard-body radius, lightly scaled by relative speed. This is a
    TEACHING HEURISTIC, not the real Foster/Akella-Alfriend Pc formula
    used operationally (that requires full position covariance matrices).
    """
    if combined_radius_km <= 0:
        combined_radius_km = 0.01
    ratio = miss_distance_km / combined_radius_km
    base = math.exp(-0.5 * ratio)
    speed_factor = min(1.0, relative_speed_kms / 15.0)
    prob = base * (0.5 + 0.5 * speed_factor)
    return float(np.clip(prob, 0.0, 1.0))


def analyze_conjunction(
    name_a: str, line1_a: str, line2_a: str,
    name_b: str, line1_b: str, line2_b: str,
    start: dt.datetime, end: dt.datetime,
    step_seconds: int = 20,
    size_a_m: float = 2.0,
    size_b_m: float = 0.2,
) -> ConjunctionResult:
    """
    Full pipeline: propagate both objects across the window, find the
    time of closest approach, compute miss distance, relative speed,
    a risk classification, and a pseudo-probability score.
    """
    ts = load.timescale()
    sat_a = build_satellite(name_a, line1_a, line2_a, ts)
    sat_b = build_satellite(name_b, line1_b, line2_b, ts)

    times_a, pos_a, vel_a = propagate_positions(sat_a, start, end, step_seconds, ts)
    times_b, pos_b, vel_b = propagate_positions(sat_b, start, end, step_seconds, ts)

    n = min(len(times_a), len(times_b))
    pos_a, pos_b = pos_a[:n], pos_b[:n]
    vel_a, vel_b = vel_a[:n], vel_b[:n]
    times = times_a[:n]

    diffs = pos_a - pos_b
    distances = np.linalg.norm(diffs, axis=1)
    idx_min = int(np.argmin(distances))

    miss_distance_km = float(distances[idx_min])
    rel_velocity_vec = vel_a[idx_min] - vel_b[idx_min]
    relative_speed_kms = float(np.linalg.norm(rel_velocity_vec))

    combined_radius_km = (size_a_m + size_b_m) / 2.0 / 1000.0
    risk_level = _risk_classify(miss_distance_km)
    prob = _pseudo_probability(miss_distance_km, combined_radius_km, relative_speed_kms)

    return ConjunctionResult(
        object_a=name_a,
        object_b=name_b,
        closest_approach_time=times[idx_min],
        miss_distance_km=miss_distance_km,
        relative_speed_kms=relative_speed_kms,
        risk_level=risk_level,
        pseudo_probability=prob,
        combined_hard_body_radius_km=combined_radius_km,
    )


def get_orbit_track(name: str, line1: str, line2: str,
                     hours: float = 1.5, step_seconds: int = 60):
    """Convenience helper: returns an orbit track for 3D plotting, starting now (UTC)."""
    start = dt.datetime.now(dt.timezone.utc)
    end = start + dt.timedelta(hours=hours)
    sat = build_satellite(name, line1, line2)
    times, positions, _ = propagate_positions(sat, start, end, step_seconds)
    return times, positions


def load_satellite_catalog(path: str = "data/satellites.json") -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("satellites", [])


def batch_conjunction_screening(
    primary_name: str,
    primary_norad_id: int,
    debris_catalog,  # pandas.DataFrame with columns: name, norad_id, approx_size_m
    start: dt.datetime,
    end: dt.datetime,
    step_seconds: int = 30,
    primary_size_m: float = 2.0,
    progress_callback=None,
):
    """
    Screens ONE primary satellite against EVERY object in a debris catalog
    and returns a list of ConjunctionResult sorted by ascending miss
    distance (i.e. highest risk first). This mirrors, at a simplified
    level, how real conjunction-screening systems batch-process an
    entire debris catalog against a protected asset.

    `progress_callback(i, total, name)` is called after each object is
    screened, useful for driving a UI progress bar.

    Objects whose TLE cannot be fetched/propagated are skipped with a
    warning collected in `errors` (returned alongside results).
    """
    try:
        name_a, line1_a, line2_a = fetch_tle(primary_norad_id)
    except TLEFetchError as e:
        raise TLEFetchError(f"Could not screen: primary satellite TLE failed ({e})")

    results = []
    errors = []
    total = len(debris_catalog)

    for i, (_, row) in enumerate(debris_catalog.iterrows(), start=1):
        deb_name = row["name"]
        try:
            name_b, line1_b, line2_b = fetch_tle(int(row["norad_id"]))
            result = analyze_conjunction(
                name_a, line1_a, line2_a,
                deb_name, line1_b, line2_b,
                start, end,
                step_seconds=step_seconds,
                size_a_m=primary_size_m,
                size_b_m=float(row.get("approx_size_m", 0.2)),
            )
            results.append(result)
        except Exception as e:
            errors.append((deb_name, str(e)))
        if progress_callback:
            progress_callback(i, total, deb_name)

    results.sort(key=lambda r: r.miss_distance_km)
    return results, errors
