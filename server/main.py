from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
KIDNEY_DATASET = DATA_DIR / "Kidney_Organ_SupplyChain_RawDataset.csv"


class MatchRequest(BaseModel):
    blood_group: str = Field(..., examples=["A+"])
    age: int = Field(..., ge=0, le=120, examples=[45])
    organ: str = Field(..., examples=["kidney"])
    urgency: str = Field(..., examples=["high"])


app = FastAPI(title="AI Organ Matching API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_model_path() -> None:
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))


def _load_joblib_model(model_path: Path):
    import joblib

    return joblib.load(model_path)


def _load_keras_model(model_path: Path):
    from tensorflow import keras

    return keras.models.load_model(model_path)


def _load_torch_model(model_path: Path):
    import torch

    loaded = torch.load(model_path, map_location="cpu")
    if hasattr(loaded, "eval"):
        loaded.eval()
    return loaded


def encode_payload(payload: MatchRequest) -> dict[str, float]:
    """Encode categorical input into numeric values for artifact models and heuristics."""
    blood_map = {
        "O-": 0,
        "O+": 1,
        "A-": 2,
        "A+": 3,
        "B-": 4,
        "B+": 5,
        "AB-": 6,
        "AB+": 7,
    }
    organ_map = {
        "kidney": 0,
        "heart": 1,
        "liver": 2,
        "lung": 3,
        "pancreas": 4,
    }
    urgency_map = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "critical": 3,
    }

    return {
        "blood_group": float(blood_map.get(payload.blood_group.strip().upper(), 0)),
        "age": max(0.0, min(payload.age / 100.0, 1.0)),
        "organ": float(organ_map.get(payload.organ.strip().lower(), 0)),
        "urgency": float(urgency_map.get(payload.urgency.strip().lower(), 1)),
    }


def _artifact_feature_vector(encoded: dict[str, float]) -> np.ndarray:
    return np.array(
        [[encoded["blood_group"], encoded["age"], encoded["organ"], encoded["urgency"]]],
        dtype=float,
    )


def _normalized_bg(blood_group: str) -> str:
    cleaned = blood_group.strip().upper().replace("+", "").replace("-", "")
    return cleaned if cleaned in {"O", "A", "B", "AB"} else "O"


def _runtime_candidate_frame(payload: MatchRequest) -> pd.DataFrame:
    """Build simple donor candidates that match the legacy model schema."""
    urgency_map = {
        "low": 3,
        "medium": 5,
        "high": 8,
        "critical": 10,
    }
    urgency_score = urgency_map.get(payload.urgency.strip().lower(), 5)
    recipient_bg = _normalized_bg(payload.blood_group)
    organ_value = payload.organ.strip().title()

    recipient_age = payload.age
    donor_options = [
        {
            "donor": f"Donor_{organ_value[:3].upper()}_101",
            "donor_age": max(18, payload.age - 4),
            "donor_bg": recipient_bg,
            "health_score": 0.94,
            "distance": 0.18,
            "compatibility_score": 1.0,
        },
        {
            "donor": f"Donor_{organ_value[:3].upper()}_204",
            "donor_age": max(18, payload.age + 2),
            "donor_bg": "O",
            "health_score": 0.87,
            "distance": 0.28,
            "compatibility_score": 0.9,
        },
        {
            "donor": f"Donor_{organ_value[:3].upper()}_318",
            "donor_age": max(18, payload.age - 7),
            "donor_bg": recipient_bg if recipient_bg != "AB" else "A",
            "health_score": 0.79,
            "distance": 0.39,
            "compatibility_score": 1.0 if recipient_bg != "AB" else 0.4,
        },
    ]

    frame = pd.DataFrame(donor_options)
    frame["recipient_age"] = recipient_age
    frame["recipient_bg"] = recipient_bg
    frame["urgency_score"] = urgency_score
    frame["organ_type"] = organ_value
    frame["dataset_source"] = "api"
    return frame


def _predict_with_loaded_artifact(model: Any, model_type: str, encoded: dict[str, float]) -> float:
    features = _artifact_feature_vector(encoded)

    if model_type == ".pkl":
        prediction = model.predict(features)
        return float(np.ravel(prediction)[0])

    if model_type == ".h5":
        prediction = model.predict(features, verbose=0)
        return float(np.ravel(prediction)[0])

    if model_type == ".pt":
        import torch

        tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(tensor) if callable(model) else model
        return float(np.ravel(prediction)[0])

    raise ValueError("Unsupported model type")


def _compatibility_label(match_score: int) -> str:
    if match_score >= 85:
        return "High"
    if match_score >= 70:
        return "Medium"
    return "Low"


def _build_response(donor_id: str, match_score: int, urgency: str, blood_group: str) -> dict[str, Any]:
    compatibility = _compatibility_label(match_score)
    hla_match = int(max(58, min(99, match_score - 3)))
    wait_time = {
        "critical": 2,
        "high": 5,
        "medium": 11,
        "low": 18,
    }.get(urgency.strip().lower(), 9)

    return {
        "donor": donor_id,
        "recipient": "Matched Patient",
        "match_score": match_score,
        "compatibility": compatibility,
        "hlaMatch": hla_match,
        "waitTime": wait_time,
    }


def _fallback_response(payload: MatchRequest) -> dict[str, Any]:
    encoded = encode_payload(payload)
    fallback_score = int(
        max(
            60,
            min(
                96,
                round(62 + (encoded["urgency"] * 8) + (encoded["organ"] * 2) + (18 - abs(payload.age - 42) / 3)),
            ),
        )
    )
    donor_id = f"Donor_{payload.organ.strip().upper()[:3]}_FALLBACK"
    return _build_response(donor_id, fallback_score, payload.urgency, payload.blood_group)


def _load_supported_artifact() -> tuple[Any | None, str | None]:
    loaders = {
        ".pkl": _load_joblib_model,
        ".h5": _load_keras_model,
        ".pt": _load_torch_model,
    }

    for extension, loader in loaders.items():
        artifacts = sorted(MODEL_DIR.glob(f"*{extension}"))
        if not artifacts:
            continue

        model_path = artifacts[0]
        try:
            return loader(model_path), extension
        except Exception:
            return None, None

    return None, None


def _load_runtime_model():
    """Train the existing legacy ML pipeline on the kidney dataset when no artifact exists."""
    if not KIDNEY_DATASET.exists():
        raise FileNotFoundError("Kidney training dataset not found for runtime training.")

    _ensure_model_path()
    legacy_preprocessing = importlib.import_module("preprocessing")
    legacy_model_module = importlib.import_module("model")

    kidney_df = legacy_preprocessing.load_and_standardize_dataset(KIDNEY_DATASET, "kidney")
    trained_model = legacy_model_module.train_model(kidney_df)
    return trained_model, legacy_preprocessing.MODEL_FEATURES


@lru_cache(maxsize=1)
def get_model_bundle() -> dict[str, Any]:
    artifact_model, artifact_type = _load_supported_artifact()
    if artifact_model is not None and artifact_type is not None:
        return {
            "mode": "artifact",
            "model": artifact_model,
            "model_type": artifact_type,
            "features": None,
        }

    runtime_model, model_features = _load_runtime_model()
    return {
        "mode": "runtime",
        "model": runtime_model,
        "model_type": "runtime",
        "features": model_features,
    }


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/match")
def match(payload: MatchRequest):
    try:
        model_bundle = get_model_bundle()

        if model_bundle["mode"] == "artifact":
            encoded = encode_payload(payload)
            raw_prediction = _predict_with_loaded_artifact(
                model_bundle["model"],
                model_bundle["model_type"],
                encoded,
            )
            match_score = int(max(0, min(round(raw_prediction * 100) if raw_prediction <= 1 else round(raw_prediction), 100)))
            donor_id = f"Donor_{payload.organ.strip().upper()[:3]}_MODEL"
            return _build_response(donor_id, match_score, payload.urgency, payload.blood_group)

        candidate_frame = _runtime_candidate_frame(payload)
        feature_columns = model_bundle["features"]
        predictions = model_bundle["model"].predict(candidate_frame[feature_columns])
        candidate_frame["predicted_score"] = np.ravel(predictions)
        top_candidate = candidate_frame.sort_values("predicted_score", ascending=False).iloc[0]

        raw_score = float(top_candidate["predicted_score"])
        match_score = int(max(0, min(round(raw_score * 100) if raw_score <= 1 else round(raw_score), 100)))
        return _build_response(str(top_candidate["donor"]), match_score, payload.urgency, payload.blood_group)
    except Exception:
        return _fallback_response(payload)
