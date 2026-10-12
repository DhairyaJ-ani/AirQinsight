"""CPCB / NAMP Yearly PDF Parser.

Reads the ``data/all_pollutants/Location_data_<year>.pdf`` files and extracts
the annual-average ambient-air-quality table: State, City, SO2, NO2, PM10, PM2.5
(µg/m³).  Values reported as ``NM`` (Not Measured) or ``-`` are converted to NaN.

The older ``NAMP_Data_for_2011_to_2015.pdf`` uses a slightly different column
layout and is handled separately.

Usage::

    uv run python -m ml_engine.ingestion.namp_pdf \\
        --pdf-dir data/all_pollutants \\
        -o data/namp_all_pollutants.csv

This produces a single CSV with columns:
    year, state, city, so2, no2, pm10, pm25
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Columns we look for (upper-case, stripped)
_POLLUTANT_ALIASES: dict[str, str] = {
    "SO2": "so2",
    "SO 2": "so2",
    "NO2": "no2",
    "NO 2": "no2",
    "PM10": "pm10",
    "PM 10": "pm10",
    "PM2.5": "pm25",
    "PM 2.5": "pm25",
    "PM2.5 ": "pm25",
}

# --- Coordinate lookup (approximate centroids for Indian cities in CPCB reports)
# Generated from public sources; extended on demand.
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "mumbai": (19.076, 72.877),
    "delhi": (28.679, 77.213),
    "pune": (18.520, 73.856),
    "chennai": (13.083, 80.270),
    "kolkata": (22.572, 88.363),
    "bangalore": (12.972, 77.595),
    "bengaluru": (12.972, 77.595),
    "hyderabad": (17.387, 78.491),
    "ahmedabad": (23.033, 72.585),
    "surat": (21.170, 72.831),
    "visakhapatnam": (17.686, 83.218),
    "patna": (25.594, 85.137),
    "lucknow": (26.847, 80.947),
    "kanpur": (26.449, 80.331),
    "nagpur": (21.145, 79.082),
    "guwahati": (26.144, 91.736),
    "bhopal": (23.259, 77.412),
    "indore": (22.718, 75.857),
    "jaipur": (26.912, 75.787),
    "chandigarh": (30.733, 76.779),
    "amritsar": (31.634, 74.872),
    "agra": (27.177, 78.008),
    "varanasi": (25.317, 82.973),
    "allahabad": (25.435, 81.847),
    "prayagraj": (25.435, 81.847),
    "meerut": (28.984, 77.706),
    "ghaziabad": (28.668, 77.441),
    "faridabad": (28.408, 77.313),
    "gurgaon": (28.459, 77.027),
    "gurugram": (28.459, 77.027),
    "noida": (28.535, 77.391),
    "thane": (19.218, 72.978),
    "raipur": (21.251, 81.629),
    "ranchi": (23.344, 85.309),
    "bhubaneswar": (20.296, 85.825),
    "cuttack": (20.463, 85.883),
    "vadodara": (22.307, 73.181),
    "rajkot": (22.303, 70.802),
    "srinagar": (34.086, 74.797),
    "jammu": (32.726, 74.857),
    "dehradun": (30.316, 78.032),
    "shimla": (31.104, 77.173),
    "gangtok": (27.329, 88.612),
    "panaji": (15.499, 73.824),
    "thiruvananthapuram": (8.524, 76.936),
    "kochi": (9.931, 76.267),
    "coimbatore": (11.017, 76.956),
    "madurai": (9.925, 78.120),
    "trichy": (10.790, 78.705),
    "tiruchirappalli": (10.790, 78.705),
    "vijayawada": (16.506, 80.648),
    "guntur": (16.309, 80.436),
    "tirupati": (13.629, 79.419),
    "mangalore": (12.914, 74.856),
    "mysore": (12.296, 76.639),
    "mysuru": (12.296, 76.639),
    "hubli": (15.364, 75.124),
    "belgaum": (15.849, 74.497),
    "kozhikode": (11.258, 75.780),
    "thrissur": (10.527, 76.214),
    "imphal": (24.817, 93.936),
    "shillong": (25.578, 91.893),
    "aizawl": (23.727, 92.717),
    "itanagar": (27.084, 93.606),
    "kohima": (25.674, 94.110),
    "agartala": (23.831, 91.279),
    "port blair": (11.623, 92.726),
    "daman": (20.397, 72.832),
    "silvassa": (20.274, 73.015),
}


def _to_float(val) -> float:
    """Convert a raw cell value to float, returning NaN for NM/- etc."""
    if val is None:
        return np.nan
    s = str(val).strip().upper()
    if s in ("", "NM", "-", "NA", "N/A", "ND", "N.D.", "NOT MEASURED"):
        return np.nan
    # Sometimes values like "22*" or "22#" appear – strip trailing non-digits
    s = re.sub(r"[^0-9.]", "", s)
    try:
        return float(s) if s else np.nan
    except ValueError:
        return np.nan


def _clean_name(name: str) -> str:
    """Normalise city / state names: strip extra whitespace and unicode chars."""
    return re.sub(r"\s+", " ", str(name)).strip()


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _city_coords(city: str) -> tuple[float | None, float | None]:
    key = _norm_key(city)
    # Exact match
    if key in _CITY_COORDS:
        return _CITY_COORDS[key]
    # Prefix match
    for k, v in _CITY_COORDS.items():
        if key.startswith(k) or k.startswith(key[:6]):
            return v
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Core parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_one_pdf(pdf_path: Path, year: int) -> pd.DataFrame:
    """Return a DataFrame[state, city, so2, no2, pm10, pm25] for one year PDF."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required: uv add pdfplumber") from exc

    rows: list[dict] = []
    current_state: str = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                row = _parse_line(line, current_state, year)
                if row is None:
                    # Try to pick up a carried-over state name from a partial line
                    possible_state = _maybe_state_line(line)
                    if possible_state:
                        current_state = possible_state
                    continue
                if row.get("_new_state"):
                    current_state = row["_new_state"]
                    # Line may also carry data; re-try without state extraction
                    row.pop("_new_state")
                if row.get("city"):
                    rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No rows parsed from %s", pdf_path)
        return df
    # Drop helper column if it slipped through
    df = df.drop(columns=["_new_state"], errors="ignore")
    df["year"] = year
    df["state"] = df["state"].replace("", np.nan).ffill()
    return df[["year", "state", "city", "so2", "no2", "pm10", "pm25"]]


def _maybe_state_line(line: str) -> str:
    """If a line looks like purely a state/UT name (no numeric content), return it."""
    stripped = line.strip()
    if not stripped:
        return ""
    # Skip header lines
    if any(kw in stripped.lower() for kw in ("annual", "average", "state", "city", "town",
                                               "microgram", "ambient", "integrated")):
        return ""
    # Looks like a data line if it contains digits
    if re.search(r"\d", stripped):
        return ""
    # Short enough to be a state name
    if 4 < len(stripped) < 60:
        return _clean_name(stripped)
    return ""


def _parse_line(line: str, current_state: str, year: int) -> dict | None:
    """Try to extract a data row from one text line.

    CPCB PDF lines look like:
        ``Andhra Pradesh Visakhapatnam 11 34 129 50``
    or a continuation of a long state/city name across lines.
    """
    line = line.strip()
    if not line:
        return None

    # Skip obvious header/title lines
    skip_keywords = (
        "annual", "average", "state /", "city /", "town /", "microgram",
        "ambient", "integrated", "monitoring", "pollution", "pollutants",
        "so2", "no2", "pm10", "pm2", "cpcb", "spcb", "union territory",
        "page", "©", "table",
    )
    if any(kw in line.lower() for kw in skip_keywords):
        return None

    # Split out trailing numeric tokens
    tokens = line.split()
    if len(tokens) < 3:
        return None

    # Identify the numeric tail (up to 4 values: SO2 NO2 PM10 PM2.5)
    numeric_vals: list[str] = []
    text_tokens: list[str] = []
    for tok in reversed(tokens):
        if re.match(r"^[\d.]+$|^NM$|^-$", tok, re.IGNORECASE) and len(numeric_vals) < 4:
            numeric_vals.insert(0, tok)
        else:
            text_tokens.insert(0, tok)
            if numeric_vals:
                # Stop consuming text tokens once we've hit the text part
                break

    # Continue consuming text tokens before the numeric block
    # (all remaining tokens before numeric_vals should be city/state)
    remaining_text = " ".join(text_tokens).strip()

    if not numeric_vals:
        return None

    # Try to assign values based on known column order: SO2, NO2, PM10, PM2.5
    vals = [_to_float(v) for v in numeric_vals]
    so2 = vals[0] if len(vals) > 0 else np.nan
    no2 = vals[1] if len(vals) > 1 else np.nan
    pm10 = vals[2] if len(vals) > 2 else np.nan
    pm25 = vals[3] if len(vals) > 3 else np.nan

    # If only 3 values, for 2023 format it's still SO2 NO2 PM10 (PM2.5 missing)
    # If only 2 values, likely an incomplete row – skip
    if len(vals) < 2:
        return None

    # Parse city / state from remaining_text
    state, city = _split_state_city(remaining_text, current_state)

    if not city:
        return None

    return {
        "state": state or current_state,
        "city": city,
        "so2": so2,
        "no2": no2,
        "pm10": pm10,
        "pm25": pm25,
    }


# ─── Known multi-word state names to help split ────────────────────────────
_STATE_NAMES = [
    "Andaman & Nicobar", "Andaman and Nicobar",
    "Andhra Pradesh", "Arunachal Pradesh", "Dadra & Nagar Haveli",
    "Dadara & Nagar Haveli", "Daman & Diu", "Himachal Pradesh",
    "Jammu & Kashmir", "Jammu and Kashmir", "Madhya Pradesh",
    "Tamil Nadu", "Uttar Pradesh", "Uttarakhand", "Uttaranchal",
    "West Bengal", "Andaman", "Nicobar", "Chandigarh",
    "Maharashtra", "Karnataka", "Rajasthan", "Gujarat", "Bihar",
    "Jharkhand", "Odisha", "Chhattisgarh", "Chattisgarh", "Assam",
    "Telangana", "Haryana", "Punjab", "Tripura", "Meghalaya",
    "Manipur", "Mizoram", "Nagaland", "Sikkim", "Goa", "Kerala",
    "Delhi", "Puducherry", "Lakshadweep", "Dadra",
]
_STATE_NAMES_SORTED = sorted(_STATE_NAMES, key=len, reverse=True)


def _split_state_city(text: str, current_state: str) -> tuple[str, str]:
    """Heuristically split ``text`` into (state, city).

    E.g. ``"Andhra Pradesh Visakhapatnam"`` → ``("Andhra Pradesh", "Visakhapatnam")``.
    """
    text = _clean_name(text)
    # Try each known state name as a prefix
    for sname in _STATE_NAMES_SORTED:
        if text.lower().startswith(sname.lower()):
            city = _clean_name(text[len(sname):])
            if city:
                return sname, city
            else:
                return sname, ""

    # No known state prefix – use current_state as fallback, whole text is city
    if current_state:
        return current_state, text
    # If text is short it's probably a city only
    words = text.split()
    if len(words) <= 3:
        return current_state, text
    # Last word might be city, rest is state
    return " ".join(words[:-1]), words[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_all_pdfs(pdf_dir: str | Path, include_coords: bool = True) -> pd.DataFrame:
    """Parse all yearly CPCB PDFs in ``pdf_dir`` and return a unified DataFrame.

    Returns columns: ``year, state, city, so2, no2, pm10, pm25[, lat, lon]``
    """
    pdf_dir = Path(pdf_dir)
    frames: list[pd.DataFrame] = []

    # Location_data_<year>.pdf  (2015–2023)
    for pdf_path in sorted(pdf_dir.glob("Location_data_*.pdf")):
        m = re.search(r"(\d{4})", pdf_path.name)
        if not m:
            continue
        year = int(m.group(1))
        log.info("Parsing %s (year %d)", pdf_path.name, year)
        try:
            df = _parse_one_pdf(pdf_path, year)
            if not df.empty:
                frames.append(df)
                log.info("  → %d rows", len(df))
        except Exception as exc:
            log.warning("Failed to parse %s: %s", pdf_path.name, exc)

    # NAMP_Data_for_2011_to_2015.pdf – may overlap with Location_data, keep both
    namp_path = pdf_dir / "NAMP_Data_for_2011_to_2015.pdf"
    if namp_path.exists():
        log.info("Parsing NAMP historical (2011–2015)")
        for year in range(2011, 2016):
            try:
                df = _parse_one_pdf(namp_path, year)
                if not df.empty:
                    df = df[df["year"] == year]
                    frames.append(df)
            except Exception as exc:
                log.warning("NAMP %d: %s", year, exc)

    if not frames:
        raise RuntimeError(f"No usable data parsed from {pdf_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # Normalise
    for col in ("so2", "no2", "pm10", "pm25"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Remove obvious junk rows (header fragments)
    combined = combined[combined["city"].str.len() > 1]
    combined = combined[~combined["city"].str.lower().str.contains(
        r"state|territory|city|town|annual|average|monitoring", regex=True, na=False)]

    # Clip to physically plausible ranges
    combined.loc[combined["so2"] > 500, "so2"] = np.nan
    combined.loc[combined["no2"] > 500, "no2"] = np.nan
    combined.loc[combined["pm10"] > 1000, "pm10"] = np.nan
    combined.loc[combined["pm25"] > 500, "pm25"] = np.nan
    combined.loc[combined["so2"] < 0, "so2"] = np.nan
    combined.loc[combined["no2"] < 0, "no2"] = np.nan

    # Deduplicate (same city/year – keep first)
    combined["city_key"] = combined["city"].str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    combined = combined.drop_duplicates(subset=["year", "city_key"], keep="first")
    combined = combined.drop(columns=["city_key"])

    combined = combined.sort_values(["year", "state", "city"]).reset_index(drop=True)

    if include_coords:
        combined[["lat", "lon"]] = combined["city"].apply(
            lambda c: pd.Series(_city_coords(c))
        )

    log.info("Total: %d city-year rows from %d files", len(combined), len(frames))
    return combined


def parse_and_save(pdf_dir: str | Path, out_csv: str | Path, include_coords: bool = True) -> pd.DataFrame:
    """Parse PDFs and write the result to ``out_csv``."""
    df = parse_all_pdfs(pdf_dir, include_coords=include_coords)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %d rows to %s", len(df), out_csv)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Parse CPCB/NAMP yearly PDFs into a multi-pollutant CSV")
    p.add_argument("--pdf-dir", default="data/all_pollutants", help="Folder containing Location_data_*.pdf")
    p.add_argument("-o", "--out", default="data/namp_all_pollutants.csv")
    p.add_argument("--no-coords", action="store_true", help="Skip adding lat/lon columns")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    parse_and_save(args.pdf_dir, args.out, include_coords=not args.no_coords)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
