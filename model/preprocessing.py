"""Dataset standardization and ML feature engineering for organ matching."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rules import build_success_probability, compatibility_score, normalize_series


BLOOD_GROUPS = ["A", "B", "AB", "O"]
TARGET_COLUMN = "success_probability"

FULL_BLOOD_COMPATIBILITY: dict[str, set[str]] = {
    "O-": {"O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"},
    "O+": {"O+", "A+", "B+", "AB+"},
    "A-": {"A-", "A+", "AB-", "AB+"},
    "A+": {"A+", "AB+"},
    "B-": {"B-", "B+", "AB-", "AB+"},
    "B+": {"B+", "AB+"},
    "AB-": {"AB-", "AB+"},
    "AB+": {"AB+"},
}

ORGAN_CODE_MAP = {"kidney": 0, "heart": 1, "liver": 2, "lung": 3, "pancreas": 4}
DATASET_SOURCE_MAP = {"kidney": 0, "heart": 1, "live": 2, "synthetic": 3}

FINAL_SCHEMA = [
    "pair_id",
    "donor_age",
    "recipient_age",
    "donor_bg",
    "recipient_bg",
    "health_score",
    "urgency_score",
    "distance",
    "compatibility_score",
    "organ_type",
    "dataset_source",
    "donor_health_score",
    "recipient_health_score",
    "wait_time_days",
    "distance_km",
    "age_diff",
    "urgency_distance_ratio",
    "health_gap",
    "urgency_x_wait",
    "blood_compat_score",
    "organ_code",
    "dataset_source_code",
    TARGET_COLUMN,
]

MODEL_FEATURES = [
    "donor_age",
    "recipient_age",
    "health_score",
    "urgency_score",
    "distance",
    "compatibility_score",
    "donor_health_score",
    "recipient_health_score",
    "wait_time_days",
    "distance_km",
    "age_diff",
    "urgency_distance_ratio",
    "health_gap",
    "urgency_x_wait",
    "blood_compat_score",
    "organ_code",
    "dataset_source_code",
]

SCHEMA_HINTS: dict[str, list[str]] = {
    "pair_id": ["pair_id", "patient_id", "donor_id", "id", "case_id", "record_id"],
    "donor_age": ["donor_age", "donor age", "age_donor"],
    "recipient_age": ["recipient_age", "patient_age", "recipient age", "age"],
    "donor_bg": ["donor_bloodtype", "donor_blood_group", "donor_bg", "blood_group_donor"],
    "recipient_bg": ["patient_bloodtype", "recipient_blood_group", "recipient_bg", "blood_group_recipient", "blood_group"],
    "organ_type": ["organ_required", "organ_donated", "organ_type", "organ", "transplant_type"],
    "health_score": [
        "realtime_organ_healthscore",
        "health_score",
        "predicted_survival_chance",
        "biological_markers",
        "survival_score",
        "stop",
        "event",
    ],
    "urgency_score": ["urgency_score", "urgency", "priority_score", "match_status", "organ_status"],
    "distance": ["distance", "distance_km", "travel_distance"],
    "wait_time_days": ["wait_time_days", "wait_time", "waiting_time", "days_waited"],
}


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _normalize_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def _similarity_score(source_name: str, hint_name: str) -> int:
    source = _normalize_name(source_name)
    hint = _normalize_name(hint_name)
    if source == hint:
        return 100
    if source in hint or hint in source:
        return 80
    source_tokens = set(source.replace("type", " type ").replace("score", " score ").split())
    hint_tokens = set(hint.replace("type", " type ").replace("score", " score ").split())
    return len(source_tokens & hint_tokens) * 10


def _find_best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    best_column = None
    best_score = -1
    for column in df.columns:
        for candidate in candidates:
            score = _similarity_score(column, candidate)
            if score > best_score:
                best_column = column
                best_score = score
    return best_column if best_score >= 20 else None


def inspect_columns(csv_path: str | Path, dataset_name: str) -> list[str]:
    """Print raw columns to show the original dataset schema before mapping."""
    df = pd.read_csv(csv_path)
    print(f"{dataset_name.title()} dataset columns:")
    print(list(df.columns))
    return list(df.columns)


def build_column_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    """Automatically map raw columns to the shared canonical schema."""
    lowered_df = df.rename(columns={column: str(column).strip().lower() for column in df.columns})
    return {canonical_name: _find_best_column(lowered_df, hints) for canonical_name, hints in SCHEMA_HINTS.items()}


def _synthetic_blood_group(values: pd.Series, dataset_name: str, role: str) -> pd.Series:
    generated = []
    for value in values.astype(str):
        seed = _stable_seed(dataset_name, role, value)
        generated.append(BLOOD_GROUPS[seed % len(BLOOD_GROUPS)])
    return pd.Series(generated, index=values.index)


def _synthetic_urgency(values: pd.Series, dataset_name: str) -> pd.Series:
    generated = []
    for value in values.astype(str):
        seed = _stable_seed(dataset_name, "urgency", value)
        rng = np.random.default_rng(seed)
        generated.append(int(rng.integers(1, 11)))
    return pd.Series(generated, index=values.index)


def _synthetic_distance(values: pd.Series, dataset_name: str) -> pd.Series:
    generated = []
    for value in values.astype(str):
        seed = _stable_seed(dataset_name, "distance", value)
        rng = np.random.default_rng(seed)
        generated.append(rng.uniform(10.0, 500.0))
    return pd.Series(generated, index=values.index).round(2)


def _extract_series(df: pd.DataFrame, mapping: dict[str, str | None], canonical_name: str) -> pd.Series:
    column_name = mapping.get(canonical_name)
    if column_name and column_name in df.columns:
        return df[column_name]
    return pd.Series(index=df.index, dtype="object")


def _as_series(df: pd.DataFrame, column_name: str, default_value: Any) -> pd.Series:
    if column_name in df.columns:
        return pd.Series(df[column_name], index=df.index)
    return pd.Series([default_value] * len(df), index=df.index)


def _build_pair_id(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    mapped = _extract_series(df, mapping, "pair_id")
    if mapped.notna().any():
        return mapped.astype(str)
    return pd.Series([f"{dataset_name}-{index}" for index in range(len(df))], index=df.index)


def _create_health_score(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    health_source = pd.to_numeric(_extract_series(df, mapping, "health_score"), errors="coerce")
    if health_source.notna().any():
        if health_source.max(skipna=True) > 1.0:
            health_source = normalize_series(health_source.fillna(health_source.median()))
        return normalize_series(health_source.fillna(health_source.median())).round(4)

    synthetic = []
    for value in _build_pair_id(df, mapping, dataset_name).astype(str):
        seed = _stable_seed(dataset_name, "health", value)
        rng = np.random.default_rng(seed)
        synthetic.append(rng.uniform(0.4, 1.0))
    return pd.Series(synthetic, index=df.index).round(4)


def _create_urgency_score(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    urgency_source = _extract_series(df, mapping, "urgency_score")
    if urgency_source.notna().any():
        numeric_urgency = pd.to_numeric(urgency_source, errors="coerce")
        if numeric_urgency.notna().any():
            return numeric_urgency.fillna(numeric_urgency.median()).clip(1, 10).round().astype(int)

        mapped_urgency = urgency_source.astype(str).str.lower().map(
            {"critical": 10, "pending": 8, "matched": 6, "transplanted": 5, "yes": 8, "no": 4}
        )
        if mapped_urgency.notna().any():
            return mapped_urgency.fillna(5).astype(int)

    return _synthetic_urgency(_build_pair_id(df, mapping, dataset_name), dataset_name)


def _create_distance(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    distance_source = pd.to_numeric(_extract_series(df, mapping, "distance"), errors="coerce")
    if distance_source.notna().any():
        return distance_source.fillna(distance_source.median()).clip(lower=0).round(2)
    return _synthetic_distance(_build_pair_id(df, mapping, dataset_name), dataset_name)


def _create_wait_time(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    wait_source = pd.to_numeric(_extract_series(df, mapping, "wait_time_days"), errors="coerce")
    if wait_source.notna().any():
        return wait_source.fillna(wait_source.median()).clip(lower=1).round().astype(int)

    generated = []
    for value in _build_pair_id(df, mapping, dataset_name).astype(str):
        seed = _stable_seed(dataset_name, "wait_time", value)
        rng = np.random.default_rng(seed)
        generated.append(int(rng.integers(1, 180)))
    return pd.Series(generated, index=df.index)


def _clean_blood_group(value: Any) -> str:
    cleaned = str(value).upper().replace("POSITIVE", "+").replace("NEGATIVE", "-").replace(" ", "").strip()
    if cleaned in FULL_BLOOD_COMPATIBILITY:
        return cleaned
    if cleaned in {"A", "B", "AB", "O"}:
        return f"{cleaned}+"
    return ""


def compute_blood_compatibility_score(donor_bg: Any, recipient_bg: Any) -> int:
    donor = _clean_blood_group(donor_bg)
    recipient = _clean_blood_group(recipient_bg)
    if donor and recipient and recipient in FULL_BLOOD_COMPATIBILITY.get(donor, set()):
        return 1
    return 0


def _normalize_organ(value: Any) -> str:
    raw = str(value).strip().lower()
    return raw if raw in ORGAN_CODE_MAP else "kidney"


def _normalize_dataset_source(value: Any) -> str:
    raw = str(value).strip().lower()
    return raw if raw in DATASET_SOURCE_MAP else "synthetic"


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add robust engineered features used by training, ranking, and evaluation."""
    feature_df = df.copy()

    for column, fallback in {
        "donor_age": 45,
        "recipient_age": 45,
        "health_score": 0.7,
        "urgency_score": 5,
        "distance": 50.0,
    }.items():
        feature_df[column] = pd.to_numeric(_as_series(feature_df, column, fallback), errors="coerce").fillna(fallback)

    feature_df["donor_health_score"] = pd.to_numeric(
        _as_series(feature_df, "donor_health_score", feature_df["health_score"]),
        errors="coerce",
    ).fillna(feature_df["health_score"])
    feature_df["recipient_health_score"] = pd.to_numeric(
        _as_series(feature_df, "recipient_health_score", feature_df["health_score"]),
        errors="coerce",
    ).fillna(feature_df["health_score"])
    feature_df["wait_time_days"] = pd.to_numeric(_as_series(feature_df, "wait_time_days", 1), errors="coerce").fillna(1).clip(lower=1)

    raw_distance = pd.to_numeric(_as_series(feature_df, "distance_km", feature_df["distance"]), errors="coerce").fillna(
        feature_df["distance"]
    )
    feature_df["distance_km"] = np.where(raw_distance <= 1.0, raw_distance * 100.0, raw_distance)
    feature_df["distance"] = normalize_series(feature_df["distance_km"]).clip(0, 1).round(4)

    donor_bg = _as_series(feature_df, "donor_bg", "O+")
    recipient_bg = _as_series(feature_df, "recipient_bg", "A+")
    feature_df["donor_bg"] = donor_bg.fillna("O+").map(_clean_blood_group).replace("", "O+")
    feature_df["recipient_bg"] = recipient_bg.fillna("A+").map(_clean_blood_group).replace("", "A+")

    feature_df["blood_compat_score"] = feature_df.apply(
        lambda row: compute_blood_compatibility_score(row["donor_bg"], row["recipient_bg"]),
        axis=1,
    )
    feature_df["compatibility_score"] = feature_df.get("compatibility_score", feature_df["blood_compat_score"]).fillna(
        feature_df["blood_compat_score"]
    )
    feature_df["compatibility_score"] = feature_df["blood_compat_score"]

    feature_df["age_diff"] = (feature_df["donor_age"] - feature_df["recipient_age"]).abs()
    feature_df["urgency_distance_ratio"] = feature_df["urgency_score"] / (feature_df["distance_km"] + 1.0)
    feature_df["health_gap"] = feature_df["donor_health_score"] - feature_df["recipient_health_score"]
    feature_df["urgency_x_wait"] = feature_df["urgency_score"] * feature_df["wait_time_days"]

    feature_df["organ_type"] = _as_series(feature_df, "organ_type", "kidney").map(_normalize_organ)
    feature_df["dataset_source"] = _as_series(feature_df, "dataset_source", "synthetic").map(
        _normalize_dataset_source
    )
    feature_df["organ_code"] = feature_df["organ_type"].map(ORGAN_CODE_MAP).fillna(0).astype(int)
    feature_df["dataset_source_code"] = feature_df["dataset_source"].map(DATASET_SOURCE_MAP).fillna(3).astype(int)

    if TARGET_COLUMN in feature_df.columns:
        feature_df[TARGET_COLUMN] = pd.to_numeric(feature_df[TARGET_COLUMN], errors="coerce").fillna(0.5)

    return feature_df


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DataFrame contains every ML feature required by the model."""
    prepared = compute_engineered_features(df)
    for feature in MODEL_FEATURES:
        prepared[feature] = pd.to_numeric(prepared.get(feature, 0), errors="coerce").fillna(0.0)
    return prepared


def standardize_dataset(raw_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Map a raw dataset into the shared schema required by training and testing."""
    df = raw_df.rename(columns={column: str(column).strip().lower() for column in raw_df.columns})
    mapping = build_column_mapping(df)

    standardized = pd.DataFrame(index=df.index)
    standardized["pair_id"] = _build_pair_id(df, mapping, dataset_name)

    donor_age = pd.to_numeric(_extract_series(df, mapping, "donor_age"), errors="coerce")
    recipient_age = pd.to_numeric(_extract_series(df, mapping, "recipient_age"), errors="coerce")
    standardized["donor_age"] = donor_age.abs() if donor_age.notna().any() else recipient_age.abs().fillna(40) + 5
    standardized["recipient_age"] = recipient_age.abs() if recipient_age.notna().any() else standardized["donor_age"] + 2

    donor_bg = _extract_series(df, mapping, "donor_bg").map(_clean_blood_group)
    recipient_bg = _extract_series(df, mapping, "recipient_bg").map(_clean_blood_group)
    standardized["donor_bg"] = donor_bg.replace("", np.nan).fillna(
        _synthetic_blood_group(standardized["pair_id"], dataset_name, "donor") + "+"
    )
    standardized["recipient_bg"] = recipient_bg.replace("", np.nan).fillna(
        _synthetic_blood_group(standardized["pair_id"], dataset_name, "recipient") + "+"
    )

    standardized["organ_type"] = dataset_name.title()
    standardized["health_score"] = _create_health_score(df, mapping, dataset_name)
    standardized["urgency_score"] = _create_urgency_score(df, mapping, dataset_name)
    standardized["distance_km"] = _create_distance(df, mapping, dataset_name)
    standardized["distance"] = normalize_series(standardized["distance_km"]).clip(0, 1).round(4)
    standardized["dataset_source"] = dataset_name.lower()
    standardized["wait_time_days"] = _create_wait_time(df, mapping, dataset_name)
    standardized["donor_health_score"] = standardized["health_score"]
    standardized["recipient_health_score"] = (standardized["health_score"] - 0.05).clip(lower=0.1, upper=1.0)

    standardized["compatibility_score"] = standardized.apply(
        lambda row: compatibility_score(
            str(row["donor_bg"]).replace("+", "").replace("-", ""),
            str(row["recipient_bg"]).replace("+", "").replace("-", ""),
        ),
        axis=1,
    )
    standardized["blood_compat_score"] = standardized.apply(
        lambda row: compute_blood_compatibility_score(row["donor_bg"], row["recipient_bg"]),
        axis=1,
    )
    standardized[TARGET_COLUMN] = build_success_probability(
        standardized["blood_compat_score"],
        standardized["urgency_score"],
        standardized["health_score"],
        standardized["distance"],
    )

    standardized = compute_engineered_features(standardized)
    return standardized[FINAL_SCHEMA]


def load_and_standardize_dataset(csv_path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Load a dataset and pass it through the automatic mapping layer."""
    raw_df = pd.read_csv(csv_path)
    return standardize_dataset(raw_df, dataset_name)
