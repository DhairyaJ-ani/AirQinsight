"""Pollutant-specific behaviour of the advection-diffusion-loss solver.

A uniform field with zero wind and zero diffusion isolates the loss term, so the
expected answer is exactly C0 * exp(-k t) with k = 1/tau_chem + v_d/H.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

H = W = 16
DX = 277.0
C0 = 50.0


@pytest.fixture
def grid():
    y = np.linspace(19.0, 19.04, H, dtype=np.float32)
    x = np.linspace(72.8, 72.84, W, dtype=np.float32)
    return y, x


def _run(cfg, field=None, wind=(0.0, 0.0), grid_xy=None, **kw):
    from ml_engine.forecasting.physics import ImprovedPhysicsSolver

    c = np.full((H, W), C0, dtype=np.float32) if field is None else field
    u = np.full((H, W), wind[0], dtype=np.float32)
    v = np.full((H, W), wind[1], dtype=np.float32)
    y, x = grid_xy
    solver = ImprovedPhysicsSolver(H, W, DX, DX, cfg)
    return solver.forecast(c, [(u, v)], y, x, **kw)


def _free_loss_cfg(**kw):
    from ml_engine.forecasting.config import PhysicsConfig

    base = dict(emission_mode="none", diffusivity_m2_s=0.0, mass_policy="none")
    base.update(kw)
    return PhysicsConfig(**base)


class TestLossTerms:
    def test_no2_default_is_four_hour_decay(self, grid):
        ds = _run(_free_loss_cfg(), grid_xy=grid)
        got = ds["no2_forecast"].values[:, 0, 0] / C0
        want = np.exp(-np.array([30, 60, 90, 120]) / 60.0 / 4.0)
        np.testing.assert_allclose(got, want, rtol=1e-5)

    def test_dry_deposition_matches_analytic(self, grid):
        cfg = _free_loss_cfg(lifetime_h=float("inf"), deposition_velocity_ms=0.01, mixing_height_m=500.0)
        ds = _run(cfg, grid_xy=grid)
        got = ds["no2_forecast"].values[-1, 0, 0] / C0
        np.testing.assert_allclose(got, math.exp(-0.01 / 500.0 * 7200.0), rtol=1e-5)

    def test_chemistry_and_deposition_add(self, grid):
        cfg = _free_loss_cfg(lifetime_h=24.0, deposition_velocity_ms=0.01, mixing_height_m=500.0)
        got = _run(cfg, grid_xy=grid)["no2_forecast"].values[-1, 0, 0] / C0
        k = 1.0 / (24 * 3600) + 0.01 / 500.0
        np.testing.assert_allclose(got, math.exp(-k * 7200.0), rtol=1e-5)

    def test_shallow_mixing_layer_deposits_faster(self, grid):
        cfg = _free_loss_cfg(lifetime_h=float("inf"), deposition_velocity_ms=0.01)
        deep = _run(cfg, grid_xy=grid, mixing_height_m=1500.0)["no2_forecast"].values[-1]
        shallow = _run(cfg, grid_xy=grid, mixing_height_m=200.0)["no2_forecast"].values[-1]
        assert shallow.mean() < deep.mean()

    def test_mixing_height_accepts_spatial_array(self, grid):
        cfg = _free_loss_cfg(lifetime_h=float("inf"), deposition_velocity_ms=0.01)
        hmix = np.full((H, W), 500.0)
        hmix[:, W // 2:] = 100.0
        out = _run(cfg, grid_xy=grid, mixing_height_m=hmix)["no2_forecast"].values[-1]
        assert out[:, : W // 2].mean() > out[:, W // 2:].mean()

    def test_conservative_pollutant_has_no_loss(self, grid):
        cfg = _free_loss_cfg(lifetime_h=float("inf"), deposition_velocity_ms=None)
        out = _run(cfg, grid_xy=grid)["no2_forecast"].values
        np.testing.assert_allclose(out, C0, rtol=1e-6)

    def test_persistent_source_holds_steady_state_under_loss(self, grid):
        """S = C0*k balances the loss, so a uniform field stays uniform."""
        cfg = _free_loss_cfg(lifetime_h=24.0, deposition_velocity_ms=0.01, emission_mode="persistent")
        out = _run(cfg, grid_xy=grid)["no2_forecast"].values
        np.testing.assert_allclose(out, C0, rtol=1e-4)


class TestForPollutant:
    def test_no2_keeps_reference_behaviour(self):
        from ml_engine.forecasting.config import PhysicsConfig

        a, b = PhysicsConfig.for_pollutant("no2"), PhysicsConfig()
        assert (a.lifetime_h, a.diffusivity_m2_s, a.mass_policy, a.pollutant) == (
            b.lifetime_h, b.diffusivity_m2_s, b.mass_policy, b.pollutant)

    @pytest.mark.parametrize("key", ["pm25", "pm10", "so2", "co", "o3"])
    def test_non_no2_does_not_reuse_no2_constants(self, key):
        from ml_engine.forecasting.config import PhysicsConfig
        from ml_engine.pollutant_config import get_pollutant

        cfg, p = PhysicsConfig.for_pollutant(key), get_pollutant(key)
        assert cfg.pollutant == key
        assert cfg.diffusivity_m2_s == p.diffusivity_m2_s
        assert cfg.deposition_velocity_ms == p.deposition_velocity_ms
        assert cfg.mass_policy == "none"
        if p.chemical_lifetime_h is None:
            assert math.isinf(cfg.lifetime_h)
        else:
            assert cfg.lifetime_h == p.chemical_lifetime_h

    def test_co_uses_mg_units(self):
        from ml_engine.forecasting.config import PhysicsConfig

        assert PhysicsConfig.for_pollutant("co").units == "mg m-3"

    def test_overrides_win(self):
        from ml_engine.forecasting.config import PhysicsConfig

        assert PhysicsConfig.for_pollutant("so2", lifetime_h=12.0).lifetime_h == 12.0

    def test_unknown_pollutant_raises(self):
        from ml_engine.forecasting.config import PhysicsConfig

        with pytest.raises(KeyError):
            PhysicsConfig.for_pollutant("benzene")

    def test_output_variable_named_after_pollutant(self, grid):
        from ml_engine.forecasting.config import PhysicsConfig

        ds = _run(PhysicsConfig.for_pollutant("so2"), grid_xy=grid)
        assert "so2_forecast" in ds and "no2_forecast" not in ds
        assert ds["so2_forecast"].attrs["pollutant"] == "so2"

    def test_pollutants_diverge_on_same_input(self, grid):
        from ml_engine.forecasting.config import PhysicsConfig

        field = np.full((H, W), C0, dtype=np.float32)
        so2 = _run(PhysicsConfig.for_pollutant("so2", emission_mode="none"), field, grid_xy=grid)
        pm25 = _run(PhysicsConfig.for_pollutant("pm25", emission_mode="none"), field, grid_xy=grid)
        assert not np.allclose(so2["so2_forecast"].values, pm25["pm25_forecast"].values)


class TestMassPolicy:
    def test_none_policy_exposes_loss(self, grid):
        cfg = _free_loss_cfg(lifetime_h=4.0, mass_policy="none")
        err = _run(cfg, grid_xy=grid)["mass_error"].values
        assert err[-1] == pytest.approx(1 - math.exp(-0.5), rel=1e-3)

    def test_rescale_policy_hides_loss(self, grid):
        cfg = _free_loss_cfg(lifetime_h=4.0, mass_policy="rescale")
        err = _run(cfg, grid_xy=grid)["mass_error"].values
        assert err.max() < 1e-4


class TestInferenceWiring:
    """run_inference must use the configured pollutant and the supplied mixing height."""

    def _infer(self, grid, cfg, **kw):
        from ml_engine.forecasting.inference import run_inference

        y, x = grid
        c0 = np.full((H, W), C0, dtype=np.float32)
        wind = [(np.zeros((H, W), np.float32), np.zeros((H, W), np.float32))]
        return run_inference(c0, wind, None, None, y, x, cfg=cfg, **kw)

    def test_so2_forecast_uses_so2_physics(self, grid):
        from ml_engine.forecasting.config import ForecastConfig

        cfg = ForecastConfig.for_pollutant("so2", emission_mode="none", diffusivity_m2_s=0.0)
        res = self._infer(grid, cfg)
        assert res.meta["pollutant"] == "so2"
        assert res.meta["physics_mass_policy"] == "none"
        # SO2 loses mass (decay + deposition); the NO2 rescale would have hidden it.
        assert res.mass_error[-1] > 0.01

    def test_mixing_height_reaches_solver(self, grid):
        from ml_engine.forecasting.config import ForecastConfig

        cfg = ForecastConfig.for_pollutant("so2", emission_mode="none", diffusivity_m2_s=0.0)
        deep = self._infer(grid, cfg, mixing_height_m=2000.0).no2_physics[-1].mean()
        shallow = self._infer(grid, cfg, mixing_height_m=100.0).no2_physics[-1].mean()
        assert shallow < deep

    def test_default_config_is_still_no2(self, grid):
        from ml_engine.forecasting.config import ForecastConfig

        res = self._infer(grid, ForecastConfig())
        assert res.meta["pollutant"] == "no2"


class TestAnalyticPuff:
    """A Gaussian puff has a closed-form solution: variance grows by 2*K*t per axis
    and the centroid moves with the wind. This checks the scheme itself, with no data."""

    N, DXP, S0 = 120, 250.0, 750.0

    def _puff(self, order, u, K, minutes=60):
        from ml_engine.forecasting.config import PhysicsConfig
        from ml_engine.forecasting.physics import ImprovedPhysicsSolver

        n, dx = self.N, self.DXP
        rr, cc = np.mgrid[0:n, 0:n].astype(float)
        c0 = np.exp(-(((cc - 30) * dx) ** 2 + ((rr - 60) * dx) ** 2) / (2 * self.S0 ** 2))
        cfg = PhysicsConfig(emission_mode="none", mass_policy="none", diffusivity_m2_s=K,
                            lifetime_h=float("inf"), advection_order=order, enable_terrain_effect=False,
                            export_interval_min=minutes, max_horizon_min=minutes)
        out = ImprovedPhysicsSolver(n, n, dx, dx, cfg).forecast(
            c0, [(np.full((n, n), u), np.zeros((n, n)))], np.arange(n), np.arange(n))
        c = out["no2_forecast"].values[-1].astype(float)
        m = c.sum()
        mx = (c * cc * dx).sum() / m
        my = (c * rr * dx).sum() / m
        return {
            "var_x": (c * (cc * dx - mx) ** 2).sum() / m - self.S0 ** 2,
            "var_y": (c * (rr * dx - my) ** 2).sum() / m - self.S0 ** 2,
            "shift_x": mx - 30 * dx,
            "mass": m / c0.sum(),
        }

    def test_zero_wind_diffusion_is_exact(self):
        r = self._puff(1, 0.0, 50.0)
        assert r["var_x"] == pytest.approx(2 * 50.0 * 3600, rel=0.01)
        assert r["var_y"] == pytest.approx(2 * 50.0 * 3600, rel=0.01)

    @pytest.mark.parametrize("u", [1.0, 3.0])
    def test_centroid_moves_with_wind_and_mass_is_kept(self, u):
        for order in (1, 3):
            r = self._puff(order, u, 50.0)
            assert r["shift_x"] == pytest.approx(u * 3600, abs=50.0)
            assert r["mass"] == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("u", [1.0, 3.0])
    def test_cubic_advection_matches_analytic_spread(self, u):
        r = self._puff(3, u, 50.0)
        assert r["var_x"] == pytest.approx(2 * 50.0 * 3600, rel=0.03)

    @pytest.mark.parametrize("u", [1.0, 3.0])
    def test_bilinear_adds_numerical_diffusion_along_wind(self, u):
        """Documents the known artefact that motivates advection_order=3."""
        lin, cub = self._puff(1, u, 50.0), self._puff(3, u, 50.0)
        assert lin["var_x"] > 1.1 * cub["var_x"]
        assert lin["var_y"] == pytest.approx(cub["var_y"], rel=0.03)   # cross-wind is unaffected

    def test_for_pollutant_uses_cubic_except_no2(self):
        from ml_engine.forecasting.config import PhysicsConfig

        assert PhysicsConfig.for_pollutant("no2").advection_order == 1
        assert PhysicsConfig.for_pollutant("so2").advection_order == 3
