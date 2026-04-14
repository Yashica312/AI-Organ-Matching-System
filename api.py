import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from database import (
    init_db, add_user, validate_user,
    add_donor, add_recipient,
    get_donors, get_recipients
)
from model import train_model
from ranking import rank_recipients
from rules import compatibility_score
from preprocessing import prepare_training_data

app = FastAPI(title="AI Organ Matching API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


class AuthRequest(BaseModel):
    username: str
    password: str


class DonorRequest(BaseModel):
    name: str
    age: int
    blood_group: str
    organ: str
    health_score: float
    distance: float


class RecipientRequest(BaseModel):
    name: str
    age: int
    blood_group: str
    organ: str
    urgency_score: int


class MatchRequest(BaseModel):
    recipient_id: int


@app.post("/api/register")
def register(req: AuthRequest):
    try:
        add_user(req.username, req.password)
        return {"success": True, "message": "Account created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/login")
def login(req: AuthRequest):
    if validate_user(req.username, req.password):
        return {"success": True, "message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/api/donors")
def donors():
    df = get_donors()
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.post("/api/donors")
def create_donor(req: DonorRequest):
    add_donor(req.name, req.age, req.blood_group, req.organ, req.health_score, req.distance)
    return {"success": True, "message": "Donor added"}


@app.get("/api/recipients")
def recipients():
    df = get_recipients()
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.post("/api/recipients")
def create_recipient(req: RecipientRequest):
    add_recipient(req.name, req.age, req.blood_group, req.organ, req.urgency_score)
    return {"success": True, "message": "Recipient added"}


@app.post("/api/match")
def match(req: MatchRequest):
    donors_df = get_donors()
    recipients_df = get_recipients()

    if donors_df.empty or recipients_df.empty:
        raise HTTPException(status_code=400, detail="No donors or recipients found")

    if req.recipient_id not in recipients_df["id"].values:
        raise HTTPException(status_code=404, detail="Recipient not found")

    selected = recipients_df[recipients_df["id"] == req.recipient_id].iloc[0]

    train_df = prepare_training_data(donors_df)
    artifacts = train_model(train_df)

    pairs = donors_df.copy()
    pairs["recipient_age"] = selected["recipient_age"]
    pairs["recipient_bg"] = selected["recipient_bg"]
    pairs["urgency_score"] = selected["urgency_score"]
    pairs["organ_type"] = str(selected["required_organ"]).lower()
    pairs["dataset_source"] = "live"
    pairs["donor_health_score"] = pairs["health_score"]
    pairs["recipient_health_score"] = max(0.3, 1 - (selected["urgency_score"] / 12))
    pairs["wait_time_days"] = 45
    pairs["distance_km"] = pairs["distance"].fillna(0.5).apply(
        lambda v: v * 100 if v <= 1 else v
    )
    pairs["compatibility_score"] = pairs.apply(
        lambda row: compatibility_score(
            str(row["donor_bg"]).replace("+", "").replace("-", ""),
            str(selected["recipient_bg"]).replace("+", "").replace("-", ""),
        ),
        axis=1,
    )
    pairs = pairs[
        pairs["donor_organ"].str.lower() == str(selected["required_organ"]).lower()
    ].copy()

    if pairs.empty:
        return {"matches": [], "message": "No donors found for this organ type"}

    ranked = rank_recipients(pairs, top_n=10, model=artifacts.estimator)

    if ranked.empty:
        return {"matches": [], "message": "No blood-compatible donors found"}

    result_cols = [
        "donor_name", "donor_organ", "donor_age", "donor_bg",
        "health_score", "distance", "blood_compat_score",
        "age_diff", "predicted_score"
    ]
    available = [c for c in result_cols if c in ranked.columns]
    return {
        "matches": ranked[available].to_dict(orient="records"),
        "best_match": ranked.iloc[0][available].to_dict()
    }


@app.get("/api/model/insights")
def model_insights():
    donors_df = get_donors()
    if donors_df.empty:
        raise HTTPException(status_code=400, detail="No donor data to train on")
    train_df = prepare_training_data(donors_df)
    artifacts = train_model(train_df)
    importances = {}
    if hasattr(artifacts.estimator, "feature_importances_"):
        importances = dict(zip(artifacts.feature_names,
                               artifacts.estimator.feature_importances_.tolist()))
    return {
        "cv_r2_mean": round(artifacts.cv_r2_mean, 3),
        "holdout_rmse": round(artifacts.holdout_rmse, 3),
        "feature_importances": importances
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
