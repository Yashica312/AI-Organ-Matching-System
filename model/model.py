"""XGBoost training utilities for organ donor-recipient matching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "Kidney_Organ_SupplyChain_RawDataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "xgb_model.pkl"

FEATURE_COLUMNS = [
    "blood_compatibility",
    "age_difference",
    "hla_score",
    "urgency_weight",
    "distance_score",
]
TARGET_COLUMN = "target_score"


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def normalize_blood_group(value: object) -> str:
    cleaned = str(value).strip().upper().replace("+", "").replace("-", "")
    return cleaned if cleaned in {"O", "A", "B", "AB"} else "O"


def blood_compatibility(donor_bg: str, recipient_bg: str) -> int:
    return int(normalize_blood_group(donor_bg) == normalize_blood_group(recipient_bg))


def simulate_hla_score(identifier: object) -> int:
    seed = _stable_seed("hla", identifier)
    rng = np.random.default_rng(seed)
    return int(rng.integers(60, 101))


def simulate_distance_score(identifier: object) -> float:
    seed = _stable_seed("distance", identifier)
    rng = np.random.default_rng(seed)
    return float(np.round(rng.uniform(0.05, 0.95), 4))


def map_urgency_weight(raw_value: object) -> int:
    value = str(raw_value).strip().lower()
    if any(token in value for token in ["critical", "stage 5", "esrd", "high"]):
        return 3
    if any(token in value for token in ["pending", "matched", "stage 4", "medium"]):
        return 2
    return 1


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the upgraded feature matrix from the kidney dataset."""
    engineered = pd.DataFrame()

    donor_ids = df.get("Donor_ID", pd.Series([f"DON-{idx:04d}" for idx in range(len(df))], index=df.index))
    donor_age = pd.to_numeric(df.get("Donor_Age"), errors="coerce").fillna(40).clip(18, 75)
    recipient_age = pd.to_numeric(df.get("Patient_Age"), errors="coerce").fillna(45).clip(1, 100)
    donor_bg = df.get("Donor_BloodType", "O").astype(str).map(normalize_blood_group)
    recipient_bg = df.get("Patient_BloodType", "O").astype(str).map(normalize_blood_group)

    engineered["blood_compatibility"] = [
        blood_compatibility(donor, recipient) for donor, recipient in zip(donor_bg, recipient_bg, strict=True)
    ]
    engineered["age_difference"] = (donor_age - recipient_age).abs().round(2)
    engineered["hla_score"] = [simulate_hla_score(identifier) for identifier in donor_ids]

    urgency_source = df.get("Organ_Condition_Alert", df.get("Organ_Status", df.get("Diagnosis_Result", "medium")))
    engineered["urgency_weight"] = urgency_source.map(map_urgency_weight)

    engineered["distance_score"] = [simulate_distance_score(identifier) for identifier in donor_ids]

    survival = pd.to_numeric(df.get("Predicted_Survival_Chance"), errors="coerce")
    if survival.notna().any():
        target = survival.clip(0, 100)
    else:
        target = (
            (35 * engineered["blood_compatibility"])
            + (0.45 * engineered["hla_score"])
            + (12 * engineered["urgency_weight"])
            + (25 * (1 - engineered["distance_score"]))
            + (20 * (1 - (engineered["age_difference"] / 60).clip(0, 1)))
        ).clip(0, 100)

    engineered[TARGET_COLUMN] = target.round(2)
    return engineered


def load_training_data(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    raw_df = pd.read_csv(dataset_path)
    return engineer_features(raw_df)


def build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    )


def save_model(model: XGBRegressor, model_path: Path = MODEL_PATH) -> None:
    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(payload, model_path)


def train_model(df: pd.DataFrame | None = None, test_size: float = 0.2, save_artifact: bool = True):
    """Train the upgraded XGBoost model and persist it to /model/xgb_model.pkl."""
    training_df = load_training_data() if df is None else df.copy()
    if not set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(training_df.columns):
        training_df = engineer_features(training_df)

    X = training_df[FEATURE_COLUMNS].copy()
    y = training_df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    print(f"RMSE: {rmse:.4f}")

    if save_artifact:
        save_model(model, MODEL_PATH)

    return model


def load_saved_model(model_path: Path = MODEL_PATH):
    payload = joblib.load(model_path)
    return payload["model"], payload["feature_columns"]
