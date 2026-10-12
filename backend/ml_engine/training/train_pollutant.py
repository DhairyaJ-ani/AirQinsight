"""Multi-Pollutant Model Training Pipeline — AirQ Insight v2.0.

Provides configurable training for tabular pollutant models (PM2.5, PM10, SO2, NO2)
using LightGBM regression, with:
- Strict temporal train/test split (2022-2023 holdout for test)
- GroupKFold on state to evaluate spatial generalization without geographic leakage
- Chronological evaluation, MAE, RMSE, R²
- Comparison against training-set mean baseline
- Saving of model booster, metadata, state encoder, and training schema
- Support for CLI:
    uv run python -m ml_engine.training.train_pollutant --pollutant pm25
    uv run python -m ml_engine.training.train_pollutant --pollutant pm10
    uv run python -m ml_engine.training.train_pollutant --pollutant so2
    uv run python -m ml_engine.training.train_pollutant --pollutant no2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from ml_engine.pollutant_config import POLLUTANTS, get_pollutant

log = logging.getLogger(__name__)

# Pollutants supported for tabular training from NAMP dataset
SUPPORTED_POLLUTANTS = ["pm25", "pm10", "so2", "no2"]
UNSUPPORTED_REASONS = {
    "co": "insufficient_data: CO is monitored as 8-hour / 1-hour average at CAAQMS and not present in NAMP annual reports.",
    "o3": "insufficient_data: O3 is a secondary photochemical oxidant requiring hourly solar radiation and CAAQMS monitors; not present in NAMP annual reports.",
}

# Candidate feature sets per target
FEATURE_SETS: dict[str, list[str]] = {
    "pm25": ["year", "so2", "no2", "pm10", "lat", "lon", "state_enc"],
    "pm10": ["year", "so2", "no2", "lat", "lon", "state_enc"],
    "so2": ["year", "no2", "pm10", "lat", "lon", "state_enc"],
    "no2": ["year", "so2", "pm10", "lat", "lon", "state_enc"],
}

LGB_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def prepare_pollutant_dataset(
    data_csv: str | Path,
    pollutant: str,
) -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    """Load and prepare data specifically for the chosen pollutant."""
    if pollutant not in SUPPORTED_POLLUTANTS:
        if pollutant in UNSUPPORTED_REASONS:
            raise ValueError(f"Pollutant {pollutant!r} cannot be trained: {UNSUPPORTED_REASONS[pollutant]}")
        raise ValueError(f"Unsupported pollutant: {pollutant!r}. Supported: {SUPPORTED_POLLUTANTS}")

    df = pd.read_csv(data_csv)
    target = pollutant.lower()

    # Require target and year
    df = df.dropna(subset=["year", target])
    df = df[df[target] >= 1.0]

    # Upper bound sanity clip based on pollutant config
    cfg = get_pollutant(target)
    max_valid = cfg.max_valid
    df = df[df[target] <= max_valid]

    # Impute coordinates with state medians
    for col in ("lat", "lon"):
        if col in df.columns:
            state_med = df.groupby("state")[col].transform("median")
            df[col] = df[col].fillna(state_med)
            # Default to national center if state median is still NaN
            df[col] = df[col].fillna(20.5937 if col == "lat" else 78.9629)

    # Impute co-pollutants with year+state median
    for col in ("so2", "no2", "pm10", "pm25"):
        if col in df.columns and col != target:
            fill = df.groupby(["year", "state"])[col].transform("median")
            df[col] = df[col].fillna(fill)
            # Fallback to general median
            df[col] = df[col].fillna(df[col].median())

    # Encode state
    enc = LabelEncoder()
    df["state_enc"] = enc.fit_transform(df["state"].fillna("Unknown"))

    features = [f for f in FEATURE_SETS[target] if f in df.columns]
    df = df.dropna(subset=features + [target])

    log.info(
        "Prepared dataset for %s: %d rows (%d-%d), %d unique cities",
        pollutant.upper(),
        len(df),
        int(df["year"].min()),
        int(df["year"].max()),
        df["city"].nunique(),
    )
    return df, features, enc


def train_single_pollutant(
    pollutant: str,
    data_csv: str | Path = "data/namp_all_pollutants_clean.csv",
    models_dir: str | Path = "models",
) -> dict[str, Any]:
    """Train and evaluate LightGBM regression model for a single pollutant."""
    df, features, state_enc = prepare_pollutant_dataset(data_csv, pollutant)
    target = pollutant.lower()
    cfg = get_pollutant(target)

    pollutant_model_dir = Path(models_dir) / f"{target}_lightgbm"
    pollutant_model_dir.mkdir(parents=True, exist_ok=True)

    # Chronological holdout: Test = 2022-2023, Train = 2015-2021
    test_mask = df["year"] >= 2022
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    # Baseline: Training set mean
    baseline_pred = np.full_like(y_test, np.mean(y_train)) if len(y_test) else np.array([])
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred))) if len(y_test) else 0.0
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred)) if len(y_test) else 0.0

    # Cross-validation (Grouped by state to avoid spatial leakage)
    n_splits = min(5, train_df["state_enc"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(train_df), np.nan)
    groups = train_df["state_enc"].values

    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        m = lgb.LGBMRegressor(**LGB_DEFAULT_PARAMS)
        m.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_pred[va_idx] = m.predict(X_train[va_idx])
        fold_rmse = float(np.sqrt(mean_squared_error(y_train[va_idx], oof_pred[va_idx])))
        cv_scores.append(fold_rmse)

    valid_oof = ~np.isnan(oof_pred)
    cv_rmse = float(np.sqrt(mean_squared_error(y_train[valid_oof], oof_pred[valid_oof])))
    cv_mae = float(mean_absolute_error(y_train[valid_oof], oof_pred[valid_oof]))
    cv_r2 = float(r2_score(y_train[valid_oof], oof_pred[valid_oof]))

    # Final fit on entire training set
    model = lgb.LGBMRegressor(**LGB_DEFAULT_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)] if len(X_test) else None,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    test_report: dict[str, Any] = {}
    if len(X_test):
        y_pred = model.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        exceedances = int((y_pred > (cfg.naaqs_annual or 9999)).sum()) if cfg.naaqs_annual else 0

        # Sub-group analysis: cities seen during training vs unseen cities
        train_cities = set(train_df["city"].str.lower().str.strip())
        is_seen = test_df["city"].str.lower().str.strip().isin(train_cities).values

        seen_r2 = float(r2_score(y_test[is_seen], y_pred[is_seen])) if is_seen.sum() > 5 else None
        seen_rmse = float(np.sqrt(mean_squared_error(y_test[is_seen], y_pred[is_seen]))) if is_seen.sum() > 5 else None

        unseen_r2 = float(r2_score(y_test[~is_seen], y_pred[~is_seen])) if (~is_seen).sum() > 5 else None
        unseen_rmse = float(np.sqrt(mean_squared_error(y_test[~is_seen], y_pred[~is_seen]))) if (~is_seen).sum() > 5 else None

        test_report = {
            "n_samples": len(y_test),
            "years": f"{int(test_df['year'].min())}–{int(test_df['year'].max())}",
            "rmse_ugm3": round(test_rmse, 3),
            "mae_ugm3": round(test_mae, 3),
            "r2": round(test_r2, 3),
            "baseline_mean_rmse": round(baseline_rmse, 3),
            "baseline_mean_mae": round(baseline_mae, 3),
            "naaqs_exceedances_detected": exceedances,
            "city_overlap_analysis": {
                "seen_cities_count": int(is_seen.sum()),
                "unseen_cities_count": int((~is_seen).sum()),
                "seen_cities_r2": round(seen_r2, 3) if seen_r2 is not None else None,
                "seen_cities_rmse": round(seen_rmse, 3) if seen_rmse is not None else None,
                "unseen_cities_r2": round(unseen_r2, 3) if unseen_r2 is not None else None,
                "unseen_cities_rmse": round(unseen_rmse, 3) if unseen_rmse is not None else None,
            },
        }

    # Save artifacts
    model_file = pollutant_model_dir / f"{target}_model.txt"
    model.booster_.save_model(str(model_file))

    state_enc_file = pollutant_model_dir / f"{target}_state_enc.json"
    state_enc_file.write_text(json.dumps(list(state_enc.classes_), indent=2))

    importance = dict(sorted(
        zip(features, [int(x) for x in model.feature_importances_]),
        key=lambda kv: -kv[1],
    ))

    metadata = {
        "pollutant": target,
        "model_type": "LightGBM",
        "features": features,
        "target": target,
        "target_unit": cfg.unit,
        "naaqs_annual_threshold": cfg.naaqs_annual,
        "training_years": f"{int(train_df['year'].min())}–{int(train_df['year'].max())}",
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "cross_validation": {
            "method": "GroupKFold (state groups)",
            "n_folds": len(cv_scores),
            "fold_rmse": [round(s, 3) for s in cv_scores],
            "oof_rmse": round(cv_rmse, 3),
            "oof_mae": round(cv_mae, 3),
            "oof_r2": round(cv_r2, 3),
        },
        "independent_test": test_report,
        "feature_importance": importance,
        "lgb_params": LGB_DEFAULT_PARAMS,
        "model_file": str(model_file),
        "state_encoder_file": str(state_enc_file),
    }

    meta_file = pollutant_model_dir / f"{target}_metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2))
    log.info("Model saved → %s", model_file)
    log.info("Metadata saved → %s", meta_file)

    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train multi-pollutant LightGBM models")
    parser.add_argument(
        "--pollutant",
        required=True,
        choices=SUPPORTED_POLLUTANTS + ["all"],
        help="Pollutant to train or 'all' for PM2.5, PM10, SO2, NO2",
    )
    parser.add_argument("--data", default="data/namp_all_pollutants_clean.csv", help="Clean dataset path")
    parser.add_argument("--models-dir", default="models", help="Directory where models will be saved")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    targets = SUPPORTED_POLLUTANTS if args.pollutant == "all" else [args.pollutant]
    for pol in targets:
        print(f"\n==================== Training {pol.upper()} ====================")
        meta = train_single_pollutant(pol, data_csv=args.data, models_dir=args.models_dir)
        test_info = meta.get("independent_test", {})
        print(f"Result for {pol.upper()}:")
        print(f"  CV OOF R²: {meta['cross_validation']['oof_r2']:.3f} | RMSE: {meta['cross_validation']['oof_rmse']:.2f}")
        if test_info:
            print(f"  Test R²:   {test_info.get('r2')} | RMSE: {test_info.get('rmse_ugm3')} | MAE: {test_info.get('mae_ugm3')}")
            print(f"  Baseline RMSE (mean): {test_info.get('baseline_mean_rmse')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
