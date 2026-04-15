import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import (
    add_donor,
    add_recipient,
    add_user,
    get_donors,
    get_recipients,
    init_db,
    validate_user,
)
from model import train_model
from preprocessing import prepare_training_data
from ranking import rank_recipients
from rules import compatibility_score


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

init_db()


def json_error(message: str, status_code: int):
    return jsonify({"success": False, "error": {"message": message}}), status_code


def json_ok(data=None, message: str | None = None, status_code: int = 200):
    payload = {"success": True, "data": data}
    if message:
        payload["message"] = message
    return jsonify(payload), status_code


@app.post("/api/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    if not username or not password:
        return json_error("Username and password are required", 400)
    try:
        add_user(username, password, password)
        return json_ok(data=None, message="Account created", status_code=201)
    except Exception as exc:
        return json_error(str(exc), 400)


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    if validate_user(username, password):
        return json_ok(data={"username": username}, message="Login successful")
    return json_error("Invalid credentials", 401)


@app.get("/api/donors")
def donors():
    df = get_donors()
    data = [] if df.empty else df.to_dict(orient="records")
    return json_ok(data=data)


@app.post("/api/donors")
def create_donor():
    payload = request.get_json(silent=True) or {}
    try:
        add_donor(
            payload["name"],
            int(payload["age"]),
            payload["blood_group"],
            payload["organ"],
            float(payload["health_score"]),
            float(payload["distance"]),
        )
        return json_ok(data=None, message="Donor added", status_code=201)
    except Exception as exc:
        return json_error(str(exc), 400)


@app.get("/api/recipients")
def recipients():
    df = get_recipients()
    data = [] if df.empty else df.to_dict(orient="records")
    return json_ok(data=data)


@app.post("/api/recipients")
def create_recipient():
    payload = request.get_json(silent=True) or {}
    try:
        add_recipient(
            payload["name"],
            int(payload["age"]),
            payload["blood_group"],
            payload["organ"],
            int(payload["urgency_score"]),
        )
        return json_ok(data=None, message="Recipient added", status_code=201)
    except Exception as exc:
        return json_error(str(exc), 400)


@app.post("/api/match")
def match():
    payload = request.get_json(silent=True) or {}
    recipient_id = payload.get("recipient_id")

    donors_df = get_donors()
    recipients_df = get_recipients()

    if donors_df.empty or recipients_df.empty:
        return json_error("No donors or recipients found", 400)

    if recipient_id not in recipients_df["id"].values:
        return json_error("Recipient not found", 404)

    selected = recipients_df[recipients_df["id"] == recipient_id].iloc[0]

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
    pairs["distance_km"] = pairs["distance"].fillna(0.5).apply(lambda v: v * 100 if v <= 1 else v)
    pairs["compatibility_score"] = pairs.apply(
        lambda row: compatibility_score(
            str(row["donor_bg"]).replace("+", "").replace("-", ""),
            str(selected["recipient_bg"]).replace("+", "").replace("-", ""),
        ),
        axis=1,
    )
    pairs = pairs[pairs["donor_organ"].str.lower() == str(selected["required_organ"]).lower()].copy()

    if pairs.empty:
        return json_ok(data={"matches": []}, message="No donors found for this organ type")

    ranked = rank_recipients(pairs, top_n=10, model=artifacts.estimator)

    if ranked.empty:
        return json_ok(data={"matches": []}, message="No blood-compatible donors found")

    result_cols = [
        "donor_name",
        "donor_organ",
        "donor_age",
        "donor_bg",
        "health_score",
        "distance",
        "blood_compat_score",
        "age_diff",
        "predicted_score",
    ]
    available = [column for column in result_cols if column in ranked.columns]
    return json_ok(
        data={
            "matches": ranked[available].to_dict(orient="records"),
            "best_match": ranked.iloc[0][available].to_dict(),
        }
    )


@app.get("/api/model/insights")
def model_insights():
    return json_error("Model insights endpoint disabled", 404)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
