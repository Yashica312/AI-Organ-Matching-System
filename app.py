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
    return jsonify({"success": False, "detail": message}), status_code


@app.post("/api/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    if not username or not password:
        return json_error("Username and password are required", 400)
    try:
        add_user(username, password)
        return jsonify({"success": True, "message": "Account created"})
    except Exception as exc:
        return json_error(str(exc), 400)


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    if validate_user(username, password):
        return jsonify({"success": True, "message": "Login successful"})
    return json_error("Invalid credentials", 401)


@app.get("/api/donors")
def donors():
    df = get_donors()
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


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
        return jsonify({"success": True, "message": "Donor added"})
    except Exception as exc:
        return json_error(str(exc), 400)


@app.get("/api/recipients")
def recipients():
    df = get_recipients()
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


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
        return jsonify({"success": True, "message": "Recipient added"})
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
        return jsonify({"matches": [], "message": "No donors found for this organ type"})

    ranked = rank_recipients(pairs, top_n=10, model=artifacts.estimator)

    if ranked.empty:
        return jsonify({"matches": [], "message": "No blood-compatible donors found"})

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
    return jsonify(
        {
            "matches": ranked[available].to_dict(orient="records"),
            "best_match": ranked.iloc[0][available].to_dict(),
        }
    )


@app.get("/api/model/insights")
def model_insights():
    donors_df = get_donors()
    if donors_df.empty:
        return json_error("No donor data to train on", 400)

    train_df = prepare_training_data(donors_df)
    artifacts = train_model(train_df)
    importances = {}
    if hasattr(artifacts.estimator, "feature_importances_"):
        importances = dict(
            zip(artifacts.feature_names, artifacts.estimator.feature_importances_.tolist())
        )

    return jsonify(
        {
            "cv_r2_mean": round(artifacts.cv_r2_mean, 3),
            "holdout_rmse": round(artifacts.holdout_rmse, 3),
            "feature_importances": importances,
        }
    )


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
