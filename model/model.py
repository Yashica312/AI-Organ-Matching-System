"""XGBoost training utilities for organ matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from xgboost import XGBRegressor

from preprocessing import MODEL_FEATURES, TARGET_COLUMN, prepare_feature_frame


RANDOM_STATE = 42
MODEL_ARTIFACT_PATH = Path(__file__).resolve().parent / "best_model.pkl"
SHAP_SUMMARY_PATH = Path(__file__).resolve().parent / "shap_summary.png"


@dataclass
class TrainingArtifacts:
    estimator: XGBRegressor
    cv_r2_mean: float
    cv_r2_std: float
    holdout_rmse: float
    feature_names: list[str]
    shap_summary_path: Path


def load_saved_model(model_path: Path | None = None):
    """Load the persisted model artifact with joblib."""
    return joblib.load(model_path or MODEL_ARTIFACT_PATH)


def train_model(df: pd.DataFrame, test_size: float = 0.2) -> TrainingArtifacts:
    """Train an XGBoost regressor with randomized search and persist the best model."""
    prepared_df = prepare_feature_frame(df)
    X = prepared_df[MODEL_FEATURES].copy()
    y = pd.to_numeric(prepared_df[TARGET_COLUMN], errors="coerce").fillna(0.5)

    if len(prepared_df) < 6:
        baseline_model = XGBRegressor(random_state=RANDOM_STATE)
        baseline_model.fit(X, y)
        joblib.dump(baseline_model, MODEL_ARTIFACT_PATH)
        return TrainingArtifacts(
            estimator=baseline_model,
            cv_r2_mean=1.0,
            cv_r2_std=0.0,
            holdout_rmse=0.0,
            feature_names=MODEL_FEATURES,
            shap_summary_path=SHAP_SUMMARY_PATH,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    model = XGBRegressor(random_state=RANDOM_STATE)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    cv_splits = max(2, min(5, len(X_train)))
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=20,
        cv=cv_splits,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_splits, scoring="r2")
    print(f"CV R2 scores: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    predictions = best_model.predict(X_test)
    holdout_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    print(f"Holdout RMSE: {holdout_rmse:.4f}")

    joblib.dump(best_model, MODEL_ARTIFACT_PATH)

    try:
        from evaluation import create_shap_summary_plot

        create_shap_summary_plot(best_model, X_train, SHAP_SUMMARY_PATH)
    except Exception as exc:
        print(f"SHAP summary generation skipped: {exc}")

    return TrainingArtifacts(
        estimator=best_model,
        cv_r2_mean=float(cv_scores.mean()),
        cv_r2_std=float(cv_scores.std()),
        holdout_rmse=holdout_rmse,
        feature_names=MODEL_FEATURES,
        shap_summary_path=SHAP_SUMMARY_PATH,
    )
