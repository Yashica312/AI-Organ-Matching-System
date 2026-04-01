"""Load, map, clean, and standardize organ datasets into one shared schema."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rules import build_success_probability, compatibility_score, normalize_series


BLOOD_GROUPS = ["A", "B", "AB", "O"]
TARGET_COLUMN = "success_probability"
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
    TARGET_COLUMN,
]
MODEL_FEATURES = [
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
]

# This mapping layer describes the canonical features we want and the likely
# source columns that may represent them in different datasets.
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
    overlap = len(source_tokens & hint_tokens)
    return overlap * 10


def _find_best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    best_column = None
    best_score = -1

    for column in df.columns:
        for candidate in candidates:
            score = _similarity_score(column, candidate)
            if score > best_score:
                best_column = column
                best_score = score

    # Small threshold so unrelated columns are not accidentally mapped.
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
    mapping: dict[str, str | None] = {}
    for canonical_name, hints in SCHEMA_HINTS.items():
        mapping[canonical_name] = _find_best_column(lowered_df, hints)
    return mapping


def _synthetic_blood_group(values: pd.Series, dataset_name: str, role: str) -> pd.Series:
    """Generate deterministic blood groups when no equivalent column exists."""
    generated = []
    for value in values.astype(str):
        seed = _stable_seed(dataset_name, role, value)
        generated.append(BLOOD_GROUPS[seed % len(BLOOD_GROUPS)])
    return pd.Series(generated, index=values.index)


def _synthetic_urgency(values: pd.Series, dataset_name: str) -> pd.Series:
    """Generate deterministic urgency values in the required 1-10 range."""
    generated = []
    for value in values.astype(str):
        seed = _stable_seed(dataset_name, "urgency", value)
        rng = np.random.default_rng(seed)
        generated.append(int(rng.integers(1, 11)))
    return pd.Series(generated, index=values.index)


def _synthetic_distance(values: pd.Series, dataset_name: str) -> pd.Series:
    """Generate deterministic normalized distance values when missing."""
    generated = []
    for value in values.astype(str):
        seed = _stable_seed(dataset_name, "distance", value)
        rng = np.random.default_rng(seed)
        generated.append(rng.uniform(0.0, 1.0))
    return pd.Series(generated, index=values.index).round(4)


def _extract_series(df: pd.DataFrame, mapping: dict[str, str | None], canonical_name: str) -> pd.Series:
    """Safely extract a mapped series or an empty series if no match was found."""
    column_name = mapping.get(canonical_name)
    if column_name and column_name in df.columns:
        return df[column_name]
    return pd.Series(index=df.index, dtype="object")


def _build_pair_id(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    """Create a stable row identifier from mapped IDs or the row index."""
    mapped = _extract_series(df, mapping, "pair_id")
    if mapped.notna().any():
        return mapped.astype(str)
    return pd.Series([f"{dataset_name}-{index}" for index in range(len(df))], index=df.index)


def _create_health_score(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    """Create health_score from the closest real feature or a synthetic fallback."""
    health_source = pd.to_numeric(_extract_series(df, mapping, "health_score"), errors="coerce")

    if health_source.notna().any():
        # Survival-like features may be on a 0-100 scale, so compress when needed.
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
    """Create urgency_score from a mapped feature or generate one synthetically."""
    urgency_source = _extract_series(df, mapping, "urgency_score")

    if urgency_source.notna().any():
        numeric_urgency = pd.to_numeric(urgency_source, errors="coerce")
        if numeric_urgency.notna().any():
            return numeric_urgency.fillna(numeric_urgency.median()).clip(1, 10).round().astype(int)

        # Handle categorical urgency-like fields from real datasets.
        mapped_urgency = urgency_source.astype(str).str.lower().map(
            {
                "critical": 10,
                "pending": 8,
                "matched": 6,
                "transplanted": 5,
                "yes": 8,
                "no": 4,
            }
        )
        if mapped_urgency.notna().any():
            return mapped_urgency.fillna(5).astype(int)

    return _synthetic_urgency(_build_pair_id(df, mapping, dataset_name), dataset_name)


def _create_distance(df: pd.DataFrame, mapping: dict[str, str | None], dataset_name: str) -> pd.Series:
    """Use a mapped distance column if present, else create a synthetic one."""
    distance_source = pd.to_numeric(_extract_series(df, mapping, "distance"), errors="coerce")
    if distance_source.notna().any():
        return normalize_series(distance_source.fillna(distance_source.median())).round(4)
    return _synthetic_distance(_build_pair_id(df, mapping, dataset_name), dataset_name)


def _clean_blood_group(value: Any) -> str:
    cleaned = str(value).upper().replace("+", "").replace("-", "").strip()
    return cleaned if cleaned in {"A", "B", "AB", "O"} else ""


def standardize_dataset(raw_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Map a raw dataset into the identical schema required by training and testing."""
    df = raw_df.rename(columns={column: str(column).strip().lower() for column in raw_df.columns})
    mapping = build_column_mapping(df)

    standardized = pd.DataFrame(index=df.index)
    standardized["pair_id"] = _build_pair_id(df, mapping, dataset_name)

    donor_age = pd.to_numeric(_extract_series(df, mapping, "donor_age"), errors="coerce")
    recipient_age = pd.to_numeric(_extract_series(df, mapping, "recipient_age"), errors="coerce")

    # If donor age is missing but recipient age exists, infer a plausible donor age.
    if donor_age.notna().any():
        standardized["donor_age"] = donor_age.abs()
    else:
        inferred_donor_age = recipient_age.abs().fillna(recipient_age.median() if recipient_age.notna().any() else 40) + 5
        standardized["donor_age"] = inferred_donor_age

    if recipient_age.notna().any():
        standardized["recipient_age"] = recipient_age.abs()
    else:
        standardized["recipient_age"] = standardized["donor_age"] + 2

    donor_bg = _extract_series(df, mapping, "donor_bg").map(_clean_blood_group)
    recipient_bg = _extract_series(df, mapping, "recipient_bg").map(_clean_blood_group)

    if donor_bg.replace("", np.nan).notna().any():
        standardized["donor_bg"] = donor_bg.replace("", np.nan)
    else:
        standardized["donor_bg"] = _synthetic_blood_group(standardized["pair_id"], dataset_name, "donor")

    if recipient_bg.replace("", np.nan).notna().any():
        standardized["recipient_bg"] = recipient_bg.replace("", np.nan)
    else:
        standardized["recipient_bg"] = _synthetic_blood_group(standardized["pair_id"], dataset_name, "recipient")

    organ_type = _extract_series(df, mapping, "organ_type")
    if organ_type.notna().any():
        standardized["organ_type"] = dataset_name.title() 
    else:
        standardized["organ_type"] = dataset_name.title()

    standardized["health_score"] = _create_health_score(df, mapping, dataset_name)
    standardized["urgency_score"] = _create_urgency_score(df, mapping, dataset_name)
    standardized["distance"] = _create_distance(df, mapping, dataset_name)
    standardized["dataset_source"] = dataset_name.lower()

    # Final shared cleanup ensures both datasets leave preprocessing with exactly the same schema.
    for numeric_column in ["donor_age", "recipient_age", "health_score", "urgency_score", "distance"]:
        standardized[numeric_column] = pd.to_numeric(standardized[numeric_column], errors="coerce")
        fallback = 5 if numeric_column == "urgency_score" else 0.5
        standardized[numeric_column] = standardized[numeric_column].fillna(standardized[numeric_column].median())
        standardized[numeric_column] = standardized[numeric_column].fillna(fallback)

    standardized["urgency_score"] = standardized["urgency_score"].clip(1, 10).round().astype(int)
    standardized["health_score"] = normalize_series(standardized["health_score"]).clip(0, 1).round(4)
    standardized["distance"] = normalize_series(standardized["distance"]).clip(0, 1).round(4)
    standardized["donor_bg"] = standardized["donor_bg"].fillna(
        _synthetic_blood_group(standardized["pair_id"], dataset_name, "donor")
    )
    standardized["recipient_bg"] = standardized["recipient_bg"].fillna(
        _synthetic_blood_group(standardized["pair_id"], dataset_name, "recipient")
    )

    standardized["compatibility_score"] = standardized.apply(
        lambda row: compatibility_score(row["donor_bg"], row["recipient_bg"]),
        axis=1,
    )
    standardized[TARGET_COLUMN] = build_success_probability(
        standardized["compatibility_score"],
        standardized["urgency_score"],
        standardized["health_score"],
        standardized["distance"],
    )

    return standardized[FINAL_SCHEMA]


def load_and_standardize_dataset(csv_path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Load a dataset and pass it through the automatic mapping layer."""
    raw_df = pd.read_csv(csv_path)
    return standardize_dataset(raw_df, dataset_name)
