"""data/sentinel_test_samples: 8 real Sentinel-5P images that load through the main model's file loader."""

from __future__ import annotations

import csv
import filecmp
from pathlib import Path

import numpy as np
import pytest

from ml_engine.ingestion.files import load_no2_geotiffs

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLES = DATA / "sentinel_test_samples"


def _manifest():
    with open(SAMPLES / "manifest.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_eight_images_load_as_a_daily_calendar_on_the_expected_grid():
    """The loader returns one layer per calendar day; days without a file come back fully NaN."""
    dates, no2, grid = load_no2_geotiffs(SAMPLES)
    assert str(dates[0].date()) == "2025-11-05" and str(dates[-1].date()) == "2025-12-24" and len(dates) == 50
    assert no2.shape == (50, 13, 10)
    west, south, east, north = grid.bbox
    assert round(west, 3) == 72.77 and round(east, 3) == 73.12
    assert round(south, 3) == 18.865 and round(north, 3) == 19.32
    have = {r["date"] for r in _manifest()}
    assert len(have) == 8
    for day, layer in zip(dates, no2):
        assert np.isfinite(layer).any() == (str(day.date()) in have), day


def test_values_are_physical_and_clouds_are_nan():
    _, no2, _ = load_no2_geotiffs(SAMPLES)
    valid = no2[np.isfinite(no2)]
    assert valid.size > 0 and valid.min() >= 0.0 and valid.max() < 1000.0     # umol/m2, Mumbai winter


def test_manifest_matches_the_files():
    dates, no2, _ = load_no2_geotiffs(SAMPLES)
    index = {str(d.date()): i for i, d in enumerate(dates)}
    for row in _manifest():
        layer = no2[index[row["date"]]]
        assert np.isnan(layer).mean() == pytest.approx(float(row["cloud_masked_fraction"]), abs=1e-3)
        assert int(np.isfinite(layer).sum()) == int(row["valid_pixels"])


def test_mix_of_clear_and_cloudy_days():
    fractions = [float(r["cloud_masked_fraction"]) for r in _manifest()]
    assert sum(f <= 0.02 for f in fractions) >= 3 and max(fractions) > 0.3


def test_files_are_untouched_copies_of_the_full_set():
    for row in _manifest():
        assert filecmp.cmp(SAMPLES / row["file"], DATA / "test_data" / row["file"], shallow=False)
