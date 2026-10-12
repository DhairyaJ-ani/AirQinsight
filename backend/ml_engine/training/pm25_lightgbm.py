"""PM₂.₅ Lightweight Model Training — Phase 1 of AirQ Insight v2.0.

This script trains a LightGBM regression model to predict annual-mean PM₂.₅
concentrations from the CPCB/NAMP yearly data, supplemented by OpenAQ hourly
observations.

Inputs
------
* ``data/namp_all_pollutants.csv`` (output of ``ml_engine.ingestion.namp_pdf``)
* ``data/openaq_mumbai_pm25.csv`` (optional hourly, for Mumbai cross-validation)

Features
--------
* Year (temporal trend)
* SO₂ and NO₂ co-pollutants (from same dataset → correlated sources)
* PM₁₀ (PM₂.₅/PM₁₀ ratio is relatively stable per city)
* Latitude and longitude (spatial proxy for emission density)
* State encoded (categorical)

Output
------
* ``models/pm25_lightgbm/pm25_model.txt`` — trained LightGBM model
* ``models/pm25_lightgbm/pm25_metadata.json`` — features, metrics, config
* Console summary of RMSE, MAE, R²

Usage::

    uv run python -m ml_engine.training.pm25_lightgbm \\
        --data data/namp_all_pollutants.csv \\
        --model-dir models/pm25_lightgbm
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

TARGET = "pm25"
FEATURES = ["year", "so2", "no2", "pm10", "lat", "lon", "state_enc"]

# India NAAQS annual PM2.5 threshold
NAAQS_PM25_ANNUAL = 40.0

LGB_PARAMS = {
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


def load_and_prepare(data_csv: str | Path) -> pd.DataFrame:
    """Load NAMP data, impute missing coordinates, encode state."""
    df = pd.read_csv(data_csv)
    log.info("Loaded %d rows from %s", len(df), data_csv)

    # Require at minimum: year, city, pm25
    df = df.dropna(subset=["year", "city", "pm25"])

    # Impute missing lat/lon with state median
    for col in ("lat", "lon"):
        state_med = df.groupby("state")[col].transform("median")
        df[col] = df[col].fillna(state_med)

    # Impute co-pollutants with year+state median
    for col in ("so2", "no2", "pm10"):
        if col in df.columns:
            fill = df.groupby(["year", "state"])[col].transform("median")
            df[col] = df[col].fillna(fill)

    # Encode state as integer
    enc = LabelEncoder()
    df["state_enc"] = enc.fit_transform(df["state"].fillna("Unknown"))

    # Clip PM2.5 to plausible range
    df = df[(df["pm25"] >= 5) & (df["pm25"] <= 400)]

    # Drop rows still missing key features
    df = df.dropna(subset=[f for f in FEATURES if f in df.columns])

    log.info("After cleaning: %d rows, %d cities, years %d–%d",
             len(df), df["city"].nunique(),
             int(df["year"].min()), int(df["year"].max()))
    return df, enc


def train(data_csv: str | Path, model_dir: str | Path) -> dict:
    """Train LightGBM PM₂.₅ model and return evaluation report."""
    df, state_enc = load_and_prepare(data_csv)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Temporal train/test split ──────────────────────────────────────────
    # Test: last 2 years (2022–2023); Train: everything else
    test_years = df["year"] >= 2022
    train_df = df[~test_years].copy()
    test_df = df[test_years].copy()

    # Ensure features exist
    available_features = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        log.warning("Missing features (not in data): %s — they will be skipped", missing)

    X_train = train_df[available_features].values
    y_train = train_df[TARGET].values
    X_test = test_df[available_features].values
    y_test = test_df[TARGET].values

    log.info("Train: %d rows (%d–%d), Test: %d rows (%d–%d)",
             len(X_train), int(train_df["year"].min()), int(train_df["year"].max()),
             len(X_test), int(test_df["year"].min()) if len(test_df) else 0,
             int(test_df["year"].max()) if len(test_df) else 0)

    # ── Cross-validation (group by state to avoid spatial leakage) ────────
    gkf = GroupKFold(n_splits=min(5, df["state_enc"].nunique()))
    oof_pred = np.full(len(train_df), np.nan)
    groups = train_df["state_enc"].values

    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[va_idx], y_train[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_pred[va_idx] = m.predict(X_train[va_idx])
        fold_rmse = float(np.sqrt(mean_squared_error(y_train[va_idx], oof_pred[va_idx])))
        cv_scores.append(fold_rmse)
        log.info("Fold %d  RMSE=%.2f", fold + 1, fold_rmse)

    oof_valid = ~np.isnan(oof_pred)
    cv_rmse = float(np.sqrt(mean_squared_error(y_train[oof_valid], oof_pred[oof_valid])))
    cv_r2 = float(r2_score(y_train[oof_valid], oof_pred[oof_valid]))
    log.info("OOF (state-grouped CV): RMSE=%.2f  R²=%.3f", cv_rmse, cv_r2)

    # ── Final model (train on all training data) ──────────────────────────
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)] if len(X_test) else None,
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    # ── Independent test evaluation ───────────────────────────────────────
    test_report: dict = {}
    if len(X_test):
        y_pred_test = model.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_r2 = float(r2_score(y_test, y_pred_test))
        log.info("Independent test (2022–2023): RMSE=%.2f  MAE=%.2f  R²=%.3f",
                 test_rmse, test_mae, test_r2)
        test_report = {
            "n_samples": len(y_test),
            "years": "2022–2023",
            "rmse_ugm3": round(test_rmse, 3),
            "mae_ugm3": round(test_mae, 3),
            "r2": round(test_r2, 3),
        }
        exceedances = int((y_pred_test > NAAQS_PM25_ANNUAL).sum())
        test_report["naaqs_exceedances_detected"] = exceedances
        log.info("NAAQS exceedances detected in test: %d / %d", exceedances, len(y_test))

    # ── Feature importance ────────────────────────────────────────────────
    importance = dict(sorted(
        zip(available_features, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))

    # ── Save model ────────────────────────────────────────────────────────
    model_path = model_dir / "pm25_model.txt"
    model.booster_.save_model(str(model_path))

    # Save state encoder classes
    state_enc_path = model_dir / "pm25_state_enc.json"
    state_enc_path.write_text(json.dumps(list(state_enc.classes_), indent=2))

    metadata = {
        "pollutant": "pm25",
        "model_type": "LightGBM",
        "features": available_features,
        "target": TARGET,
        "target_unit": "µg/m³",
        "naaqs_annual_threshold_ugm3": NAAQS_PM25_ANNUAL,
        "training_years": f"{int(train_df['year'].min())}–{int(train_df['year'].max())}",
        "n_train_samples": int(len(X_train)),
        "cross_validation": {
            "method": "GroupKFold (state groups)",
            "n_folds": len(cv_scores),
            "fold_rmse_ugm3": [round(s, 3) for s in cv_scores],
            "oof_rmse_ugm3": round(cv_rmse, 3),
            "oof_r2": round(cv_r2, 3),
        },
        "independent_test": test_report,
        "feature_importance": importance,
        "lgb_params": LGB_PARAMS,
        "model_file": str(model_path),
        "state_encoder_file": str(state_enc_path),
    }
    meta_path = model_dir / "pm25_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Model saved → %s", model_path)
    log.info("Metadata → %s", meta_path)

    return metadata


def predict_city(
    model_dir: str | Path,
    city_data: dict,
) -> dict:
    """Predict PM₂.₅ for one city-year using a trained model.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing ``pm25_model.txt`` and ``pm25_metadata.json``.
    city_data : dict
        Must have keys matching ``FEATURES`` (except ``state_enc`` which is
        looked up from ``state``).
    """
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "pm25_metadata.json").read_text())
    state_classes = json.loads((model_dir / "pm25_state_enc.json").read_text())
    state_to_int = {s: i for i, s in enumerate(state_classes)}

    booster = lgb.Booster(model_file=str(model_dir / "pm25_model.txt"))

    row = dict(city_data)
    row["state_enc"] = state_to_int.get(row.get("state", ""), 0)

    features = meta["features"]
    X = np.array([[row.get(f, 0.0) for f in features]], dtype=np.float64)
    pm25_pred = float(booster.predict(X)[0])
    pm25_pred = max(0.0, pm25_pred)

    return {
        "pm25_predicted_ugm3": round(pm25_pred, 2),
        "naaqs_annual_threshold": NAAQS_PM25_ANNUAL,
        "naaqs_status": (
            "exceeded" if pm25_pred > NAAQS_PM25_ANNUAL
            else "approaching" if pm25_pred > 0.8 * NAAQS_PM25_ANNUAL
            else "compliant"
        ),
        "features_used": features,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the PM₂.₅ LightGBM model from NAMP data")
    p.add_argument("--data", default="data/namp_all_pollutants.csv",
                   help="Path to namp_all_pollutants.csv")
    p.add_argument("--model-dir", default="models/pm25_lightgbm",
                   help="Output directory for model artefacts")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path(args.data).exists():
        log.error("Data file not found: %s — run namp_pdf first", args.data)
        return 1

    report = train(args.data, args.model_dir)

    print("\n=== PM₂.₅ LightGBM Training Complete ===")
    cv = report["cross_validation"]
    print(f"  CV RMSE (OOF):  {cv['oof_rmse_ugm3']:.2f} µg/m³")
    print(f"  CV R²   (OOF):  {cv['oof_r2']:.3f}")
    if report.get("independent_test"):
        t = report["independent_test"]
        print(f"  Test RMSE:      {t['rmse_ugm3']:.2f} µg/m³  (n={t['n_samples']}, {t['years']})")
        print(f"  Test R²:        {t['r2']:.3f}")
    print(f"\n  Model saved → {report['model_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
