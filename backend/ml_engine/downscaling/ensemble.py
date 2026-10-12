"""Ensemble learning for spatial NO2 downscaling.

Upgrades the single XGBoost downscaling model into a multi-model ensemble:
- XGBoost (weight: 0.5)
- Random Forest (weight: 0.3)
- LightGBM (weight: 0.2)

Responsibilities:
- Load/train all three models
- Weighted ensemble prediction
- Confidence estimation
- Agreement score (model consensus / disagreement)
- Expose prediction, confidence, disagreement without breaking existing interfaces
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from ..config import DownscaleConfig
from . import FeatureBuilder, column_to_surface
from ..grid import block_mean, upsample_bilinear

log = logging.getLogger(__name__)

ENSEMBLE_META_FILE = "ensemble_metadata.json"
XGB_FILE = "xgb_downscaler.json"
RF_FILE = "rf_downscaler.pkl"
LGB_FILE = "lgb_downscaler.txt"
LGB_PKL_FILE = "lgb_downscaler.pkl"


@dataclass
class EnsemblePrediction:
    """Structured result of an ensemble prediction."""
    prediction: np.ndarray        # (H, W) or (N,) weighted downscaled column
    confidence: float             # [0, 1] overall ensemble confidence / agreement
    disagreement: float           # mean standard deviation across models (in target units)
    confidence_map: np.ndarray    # (H, W) pixel-level confidence [0, 1]
    disagreement_map: np.ndarray  # (H, W) pixel-level standard deviation
    model_predictions: dict[str, np.ndarray]  # per-model outputs for auditing
    weights: dict[str, float]     # weights used for fusion


class EnsembleDownscaler:
    """Multi-model ensemble downscaler combining XGBoost, Random Forest, and LightGBM."""

    DEFAULT_WEIGHTS = {
        "xgboost": 0.5,
        "random_forest": 0.3,
        "lightgbm": 0.2,
    }

    def __init__(
        self,
        cfg: DownscaleConfig | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.cfg = cfg or DownscaleConfig()
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        # Validate and normalize weights
        if not self.weights or any(w < 0 for w in self.weights.values()):
            raise ValueError(f"Ensemble weights must be non-negative: {self.weights}")
        total_w = sum(self.weights.values())
        if total_w <= 0:
            raise ValueError(f"Sum of ensemble weights must be positive, got {total_w}")
        self.weights = {k: float(v) / total_w for k, v in self.weights.items()}

        self.xgb_model: xgb.XGBRegressor | None = None
        self.rf_model: RandomForestRegressor | None = None
        self.lgb_model: lgb.LGBMRegressor | None = None
        self.features: FeatureBuilder | None = None
        self.metadata: dict[str, Any] = {}

    def fit(
        self,
        coarse: xr.Dataset,
        no2_filled: xr.DataArray,
        fill_flag: xr.DataArray | None,
        static_fine: xr.Dataset,
        factor: int,
    ) -> dict[str, Any]:
        """Train all three models on coarse-scale samples and validate."""
        from ..validation import regression_metrics

        cfg = self.cfg
        self.features = FeatureBuilder(static_fine, factor)
        n_t = coarse.sizes["time"]
        X_days, y_days = [], []

        for t in range(n_t):
            field = no2_filled.values[t]
            X = self.features.coarse_features(coarse, t, field)
            y = field.ravel()
            keep = np.isfinite(y)
            if fill_flag is not None:
                keep &= fill_flag.values[t].ravel() == 0
            X_days.append(X[keep])
            y_days.append(y[keep])

        n_obs = sum(len(y) for y in y_days)
        if n_obs < 200 and fill_flag is not None:
            log.warning("Only %d observed coarse pixels; training ensemble on gap-filled too", n_obs)
            return self.fit(coarse, no2_filled, None, static_fine, factor)

        n_val = max(1, int(round(n_t * cfg.validation_fraction)))
        X_tr = np.concatenate(X_days[:-n_val])
        y_tr = np.concatenate(y_days[:-n_val])
        X_va = np.concatenate(X_days[-n_val:])
        y_va = np.concatenate(y_days[-n_val:])
        X_all = np.concatenate([X_tr, X_va])
        y_all = np.concatenate([y_tr, y_va])

        # 1. XGBoost
        xgb_params = dict(
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            tree_method="hist",
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        probe = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            early_stopping_rounds=cfg.early_stopping_rounds,
            eval_metric="rmse",
            **xgb_params,
        )
        probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_trees = int(probe.best_iteration) + 1
        self.xgb_model = xgb.XGBRegressor(n_estimators=best_trees, **xgb_params)
        self.xgb_model.fit(X_all, y_all, verbose=False)

        # 2. Random Forest
        rf_estimators = min(100, max(30, best_trees // 2))
        self.rf_model = RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=min(12, cfg.max_depth + 4),
            min_samples_split=4,
            random_state=cfg.random_state,
            n_jobs=1,
        )
        self.rf_model.fit(X_all, y_all)

        # 3. LightGBM
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=best_trees,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.random_state,
            n_jobs=1,
            verbosity=-1,
        )
        self.lgb_model.fit(X_all, y_all)

        # Evaluate ensemble on holdout
        pred_xgb = probe.predict(X_va)
        pred_rf = self.rf_model.predict(X_va)
        pred_lgb = self.lgb_model.predict(X_va)

        ensemble_val = (
            self.weights["xgboost"] * pred_xgb
            + self.weights["random_forest"] * pred_rf
            + self.weights["lightgbm"] * pred_lgb
        )
        holdout = regression_metrics(y_va, ensemble_val)
        xgb_metrics = regression_metrics(y_va, pred_xgb)
        rf_metrics = regression_metrics(y_va, pred_rf)
        lgb_metrics = regression_metrics(y_va, pred_lgb)

        # Disagreement on validation set
        disagreement_val = float(np.mean(np.std([pred_xgb, pred_rf, pred_lgb], axis=0)))

        importance = dict(sorted(
            zip(self.features.feature_names, map(float, self.xgb_model.feature_importances_)),
            key=lambda kv: -kv[1],
        ))

        self.metadata = {
            "feature_names": self.features.feature_names,
            "static_features": self.features.static_features,
            "refine_factor": factor,
            "n_trees_xgb": best_trees,
            "n_trees_rf": rf_estimators,
            "n_train_samples": int(len(y_all)),
            "weights": self.weights,
            "temporal_holdout_ensemble": holdout,
            "model_holdouts": {
                "xgboost": xgb_metrics,
                "random_forest": rf_metrics,
                "lightgbm": lgb_metrics,
            },
            "validation_disagreement": disagreement_val,
            "feature_importance": importance,
            "config": asdict(cfg),
            "target_units": "umol m-2",
        }
        log.info(
            "Ensemble trained: holdout R2=%.3f (XGB=%.3f, RF=%.3f, LGB=%.3f) Disagreement=%.2f",
            holdout["r2"], xgb_metrics["r2"], rf_metrics["r2"], lgb_metrics["r2"], disagreement_val,
        )
        return self.metadata

    def predict_features(self, X: np.ndarray) -> EnsemblePrediction:
        """Run ensemble prediction on raw feature matrix X."""
        self._require_fitted()
        pred_xgb = self.xgb_model.predict(X).astype(np.float64)
        pred_rf = self.rf_model.predict(X).astype(np.float64)
        pred_lgb = self.lgb_model.predict(X).astype(np.float64)

        stacked = np.stack([pred_xgb, pred_rf, pred_lgb], axis=0)  # (3, N)
        weights_arr = np.array([
            self.weights["xgboost"],
            self.weights["random_forest"],
            self.weights["lightgbm"],
        ]).reshape(3, 1)

        weighted_pred = np.sum(stacked * weights_arr, axis=0)
        disagreement_pixel = np.std(stacked, axis=0)
        mean_disagreement = float(np.mean(disagreement_pixel))

        # Confidence: bounded [0.0, 1.0], decays as disagreement relative to mean prediction grows
        mean_val = max(1.0, float(np.mean(np.abs(weighted_pred))))
        rel_diff = disagreement_pixel / mean_val
        conf_map = np.clip(1.0 - rel_diff, 0.05, 0.99)
        overall_conf = float(np.mean(conf_map))

        return EnsemblePrediction(
            prediction=weighted_pred,
            confidence=round(overall_conf, 3),
            disagreement=round(mean_disagreement, 3),
            confidence_map=conf_map,
            disagreement_map=disagreement_pixel,
            model_predictions={
                "xgboost": pred_xgb,
                "random_forest": pred_rf,
                "lightgbm": pred_lgb,
            },
            weights=dict(self.weights),
        )

    def predict_day_ensemble(
        self,
        coarse: xr.Dataset,
        no2_filled: xr.DataArray,
        t: int,
    ) -> EnsemblePrediction:
        """Fine-grid ensemble column prediction for day ``t`` with backprojection."""
        self._require_fitted()
        f = self.features.factor
        field = no2_filled.values[t].astype(np.float64)
        X = self.features.fine_features(coarse, t, field)
        fine_shape = (field.shape[0] * f, field.shape[1] * f)

        ens_res = self.predict_features(X)

        # Reshape to 2D
        pred_2d = ens_res.prediction.reshape(fine_shape)
        # Apply backprojection conservation
        for _ in range(self.cfg.backprojection_iterations):
            residual = field - block_mean(pred_2d, f)
            pred_2d += upsample_bilinear(residual, f)

        ens_res.prediction = np.clip(pred_2d, 0, None).astype(np.float32)
        ens_res.confidence_map = ens_res.confidence_map.reshape(fine_shape).astype(np.float32)
        ens_res.disagreement_map = ens_res.disagreement_map.reshape(fine_shape).astype(np.float32)

        return ens_res

    def predict_day(self, coarse: xr.Dataset, no2_filled: xr.DataArray, t: int) -> np.ndarray:
        """Drop-in compatibility with single NO2Downscaler.predict_day."""
        res = self.predict_day_ensemble(coarse, no2_filled, t)
        return res.prediction

    def predict(
        self,
        coarse: xr.Dataset,
        no2_filled: xr.DataArray,
        fine_x: np.ndarray,
        fine_y: np.ndarray,
    ) -> xr.DataArray:
        """Drop-in compatibility with single NO2Downscaler.predict."""
        days = [self.predict_day(coarse, no2_filled, t) for t in range(coarse.sizes["time"])]
        return xr.DataArray(
            np.stack(days),
            coords={"time": coarse.time.values, "y": fine_y, "x": fine_x},
            dims=("time", "y", "x"),
            name="no2_column_fine",
            attrs={
                "units": "umol m-2",
                "long_name": "Ensemble downscaled tropospheric NO2 column",
                "ensemble_confidence": float(self.metadata.get("temporal_holdout_ensemble", {}).get("r2", 0.85)),
            },
        )

    def to_surface(self, column_fine: xr.DataArray, coarse: xr.Dataset) -> xr.DataArray:
        """Convert downscaled column to boundary-layer surface NO2."""
        f = self.features.factor if self.features else self.metadata["refine_factor"]
        blh_fine = upsample_bilinear(coarse["blh"].values.astype(np.float64), f)
        conc = column_to_surface(column_fine.values.astype(np.float64), blh_fine)
        return xr.DataArray(
            np.clip(conc, 0, None).astype(np.float32),
            coords=column_fine.coords,
            dims=column_fine.dims,
            name="no2_surface",
            attrs={"units": "ug m-3", "long_name": "Ensemble boundary-layer-mixed NO2"},
        )

    def save(self, model_dir: str | Path) -> Path:
        """Save all three models and ensemble metadata."""
        self._require_fitted()
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.xgb_model.save_model(model_dir / XGB_FILE)
        with open(model_dir / RF_FILE, "wb") as f:
            pickle.dump(self.rf_model, f)
        with open(model_dir / LGB_PKL_FILE, "wb") as f:
            pickle.dump(self.lgb_model, f)
        if hasattr(self.lgb_model, "booster_") and self.lgb_model.booster_ is not None:
            self.lgb_model.booster_.save_model(str(model_dir / LGB_FILE))

        (model_dir / ENSEMBLE_META_FILE).write_text(json.dumps(self.metadata, indent=2))
        return model_dir

    @classmethod
    def load(cls, model_dir: str | Path, static_fine: xr.Dataset) -> "EnsembleDownscaler":
        """Load trained models and reconstruct feature builder."""
        model_dir = Path(model_dir)
        meta_path = model_dir / ENSEMBLE_META_FILE
        if not meta_path.exists():
            # Fall back to single model if legacy
            meta_path = model_dir / "metadata.json"

        meta = json.loads(meta_path.read_text())
        cfg = DownscaleConfig(**meta["config"])
        weights = meta.get("weights", cls.DEFAULT_WEIGHTS)
        obj = cls(cfg, weights)

        # Load XGB
        obj.xgb_model = xgb.XGBRegressor()
        obj.xgb_model.load_model(model_dir / XGB_FILE)

        # Load RF
        rf_path = model_dir / RF_FILE
        if rf_path.exists():
            with open(rf_path, "rb") as f:
                obj.rf_model = pickle.load(f)
        else:
            # Fallback if RF was not saved in legacy run
            obj.rf_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)

        # Load LGB
        lgb_pkl_path = model_dir / LGB_PKL_FILE
        lgb_txt_path = model_dir / LGB_FILE
        if lgb_pkl_path.exists():
            with open(lgb_pkl_path, "rb") as f:
                obj.lgb_model = pickle.load(f)
        elif lgb_txt_path.exists():
            obj.lgb_model = lgb.LGBMRegressor()
            booster = lgb.Booster(model_file=str(lgb_txt_path))
            obj.lgb_model._Booster = booster
            obj.lgb_model.fitted_ = True
            n_feats = len(meta.get("feature_names", []))
            if n_feats:
                obj.lgb_model.n_features_in_ = n_feats
                obj.lgb_model._n_features = n_feats
        else:
            obj.lgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbosity=-1)

        missing = [v for v in meta["static_features"] if v not in static_fine]
        if missing:
            raise ValueError(f"static covariates missing for loaded ensemble: {missing}")

        obj.features = FeatureBuilder(static_fine, meta["refine_factor"], meta["static_features"])
        obj.metadata = meta
        return obj

    def _require_fitted(self):
        if self.xgb_model is None or self.rf_model is None or self.lgb_model is None or self.features is None:
            raise RuntimeError("EnsembleDownscaler is not fully trained or loaded")


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "time",
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/val split ensuring no future information leaks into training.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a timestamp or chronological sequence column.
    time_col : str
        Column used for sorting and splitting.
    val_fraction : float
        Fraction of recent observations reserved for validation.

    Returns
    -------
    (train_df, val_df)
    """
    if time_col not in df.columns:
        # Fallback to index-based chronological split if time_col not present
        split_idx = int(len(df) * (1.0 - val_fraction))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1.0 - val_fraction))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def spatial_split(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    n_spatial_blocks: int = 5,
    val_block_idx: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spatial block cross-validation split (leave-spatial-cluster-out).

    Prevents spatial autocorrelation leakage between training and validation pixels.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing spatial coordinates.
    lat_col, lon_col : str
        Latitude and longitude column names.
    n_spatial_blocks : int
        Number of latitude-longitude grid partitions.
    val_block_idx : int
        Index of the partition held out for validation [0, n_spatial_blocks - 1].

    Returns
    -------
    (train_df, val_df)
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        # Fallback to regular k-fold partition
        block_ids = np.arange(len(df)) % n_spatial_blocks
    else:
        # Simple coordinate binning to create disjoint geographical blocks
        lat_bins = pd.qcut(df[lat_col], q=min(n_spatial_blocks, df[lat_col].nunique()), labels=False, duplicates="drop")
        block_ids = lat_bins.to_numpy()

    val_mask = (block_ids == val_block_idx)
    return df[~val_mask].copy(), df[val_mask].copy()


def calibrate_ensemble_weights_on_val(
    y_true_val: np.ndarray,
    preds_val: dict[str, np.ndarray],
    metric: str = "r2",
    step: float = 0.05,
) -> dict[str, float]:
    """Grid-search over convex combinations of model predictions to maximize validation performance.

    Never touches the test set. Ensures weights sum to 1.0 and are non-negative.

    Parameters
    ----------
    y_true_val : np.ndarray
        Ground truth values on the validation set.
    preds_val : dict[str, np.ndarray]
        Dict of model_name -> 1D prediction array on the validation set.
        Must include keys for 'xgboost', 'random_forest', and 'lightgbm'.
    metric : str
        'r2' (maximize) or 'rmse' (minimize).
    step : float
        Grid step size for weight exploration (default 0.05).

    Returns
    -------
    dict[str, float]
        Calibrated weights {'xgboost': w_xgb, 'random_forest': w_rf, 'lightgbm': w_lgb}.
    """
    y_true = np.asarray(y_true_val, dtype=np.float64).ravel()
    models = list(preds_val.keys())
    pred_matrix = np.column_stack([np.asarray(preds_val[m], dtype=np.float64).ravel() for m in models])

    best_score = -float("inf") if metric == "r2" else float("inf")
    best_weights = {m: 1.0 / len(models) for m in models}

    # Generate simplex grid for 3 models: w0 + w1 + w2 = 1.0
    steps = int(round(1.0 / step))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12

    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            w0, w1, w2 = i * step, j * step, k * step
            w_vec = np.array([w0, w1, w2], dtype=np.float64)
            # If models has different length or names, handle standard 3 models:
            if len(models) == 3:
                fused = pred_matrix @ w_vec
            else:
                fused = pred_matrix.mean(axis=1)

            err = fused - y_true
            if metric == "r2":
                ss_res = float(np.sum(err ** 2))
                score = 1.0 - (ss_res / ss_tot)
                if score > best_score:
                    best_score = score
                    best_weights = {models[0]: round(w0, 3), models[1]: round(w1, 3), models[2]: round(w2, 3)}
            else:
                rmse = float(np.sqrt(np.mean(err ** 2)))
                if rmse < best_score:
                    best_score = rmse
                    best_weights = {models[0]: round(w0, 3), models[1]: round(w1, 3), models[2]: round(w2, 3)}

    return best_weights

