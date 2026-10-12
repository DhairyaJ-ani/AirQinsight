"""service.forecast_point: the improved solver is opt-in and agrees with the legacy one where they should."""

from __future__ import annotations

import numpy as np
import pytest

from ml_engine import service
from ml_engine.grid import GridSpec

RES = 0.0025  # ~275 m


@pytest.fixture
def scene():
    grid = GridSpec(west=72.80, north=19.40, res=RES, width=120, height=120)
    rr, cc = np.mgrid[0:120, 0:120].astype(float)
    surface = 20.0 + 80.0 * np.exp(-(((cc - 30) ** 2 + (rr - 60) ** 2) / (2 * 4.0 ** 2)))
    u = np.full(surface.shape, 3.0)
    v = np.zeros(surface.shape)
    return grid, surface, u, v


def _centroid_col(field):
    bump = field - np.median(field)
    cols = np.arange(field.shape[1])[None, :]
    return float((bump * cols).sum() / bump.sum())


def test_both_solvers_return_every_requested_horizon(scene):
    grid, surface, u, v = scene
    for solver in service.PHYSICS_SOLVERS:
        out = service._advect_surface(surface, u, v, grid, [1, 2], 1, solver)
        assert sorted(out) == [1, 2]
        assert out[1].shape == surface.shape and np.isfinite(out[1]).all()


def test_solvers_move_the_plume_alike(scene):
    """Both run in persistent-emission mode (the source keeps re-emitting), so the centroid moves less than pure
    advection would; what must hold is that both move it downwind by about the same amount."""
    grid, surface, u, v = scene
    start = _centroid_col(surface)
    moved = {s_: _centroid_col(service._advect_surface(surface, u, v, grid, [2], 1, s_)[2]) - start
             for s_ in service.PHYSICS_SOLVERS}
    assert all(m > 40.0 for m in moved.values()), moved
    assert moved["legacy"] == pytest.approx(moved["improved"], abs=4.0), moved


def test_unknown_solver_is_rejected(scene):
    grid, surface, u, v = scene
    with pytest.raises(ValueError, match="solver"):
        service._advect_surface(surface, u, v, grid, [1], 1, "magic")
    with pytest.raises(ValueError, match="solver"):
        service.forecast_point(19.0, 72.9, 3, solver="magic")   # rejected before any Earth Engine call


def test_legacy_stays_the_default():
    import inspect

    assert inspect.signature(service.forecast_point).parameters["solver"].default == "legacy"
