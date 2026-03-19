"""Model training utilities for organ matching."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import MODEL_FEATURES, TARGET_COLUMN


RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Create a shared preprocessing + RandomForest pipeline."""
    categorical_features = ["donor_bg", "recipient_bg", "organ_type", "dataset_source"]
    numeric_features = ["donor_age", "recipient_age", "health_score", "urgency_score", "distance", "compatibility_score"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(kidney_df: pd.DataFrame, test_size: float = 0.2):
    X = kidney_df[MODEL_FEATURES].copy()
    y = kidney_df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline