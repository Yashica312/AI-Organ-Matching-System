from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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
from ranking import rank_recipients
from rules import compatibility_score


def auth_ui():
    st.markdown("## Secure Match Login")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        user = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if validate_user(user, pwd):
                st.session_state["logged_in"] = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with register_tab:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")
        confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        if st.button("Register"):
            if not new_user.strip():
                st.error("Username cannot be empty")
            elif new_pass != confirm:
                st.error("Passwords do not match")
            else:
                add_user(new_user.strip(), new_pass)
                st.success("Account created. Please log in.")


def prepare_training_data(donors_df: pd.DataFrame) -> pd.DataFrame:
    train_df = donors_df.copy()
    train_df["recipient_age"] = 40
    train_df["recipient_bg"] = "A+"
    train_df["urgency_score"] = 5
    train_df["organ_type"] = train_df["donor_organ"].fillna("Kidney").str.lower()
    train_df["dataset_source"] = "live"
    train_df["donor_health_score"] = train_df["health_score"]
    train_df["recipient_health_score"] = 0.72
    train_df["wait_time_days"] = 30
    train_df["distance_km"] = train_df["distance"].fillna(0.5).apply(lambda value: value * 100 if value <= 1 else value)
    train_df["compatibility_score"] = train_df.apply(
        lambda row: compatibility_score(str(row["donor_bg"]).replace("+", "").replace("-", ""), "A"),
        axis=1,
    )
    train_df["success_probability"] = (
        0.5 * train_df["compatibility_score"]
        + 0.3 * (train_df["urgency_score"] / 10)
        + 0.2 * train_df["health_score"]
        - 0.1 * train_df["distance"].fillna(0.5)
    )
    return train_df


def render_registry_tables(donors_df: pd.DataFrame, recipients_df: pd.DataFrame):
    st.subheader("Live Donor and Recipient Registry")
    donor_col, recipient_col = st.columns(2)

    with donor_col:
        st.markdown("#### Donors")
        st.dataframe(
            donors_df[["id", "donor_name", "donor_organ", "donor_bg", "donor_age", "health_score", "distance"]],
            use_container_width=True,
        )

    with recipient_col:
        st.markdown("#### Recipients")
        st.dataframe(
            recipients_df[["id", "recipient_name", "required_organ", "recipient_bg", "recipient_age", "urgency_score"]],
            use_container_width=True,
        )


def render_model_insights(artifacts):
    st.subheader("Model Insights")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("CV R2 Score", f"{artifacts.cv_r2_mean:.3f}")
    metric_col2.metric("Holdout RMSE", f"{artifacts.holdout_rmse:.3f}")

    importances = getattr(artifacts.estimator, "feature_importances_", None)
    if importances is not None:
        importance_df = pd.DataFrame(
            {"feature": artifacts.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        st.markdown("#### XGBoost Feature Importances")
        st.bar_chart(importance_df.set_index("feature"))

    shap_path = Path(artifacts.shap_summary_path)
    st.markdown("#### SHAP Summary")
    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info("SHAP summary will appear after the model is trained successfully.")


st.set_page_config(page_title="AI Organ Matching", layout="wide")
init_db()

if "logged_in" not in st.session_state:
    auth_ui()
    st.stop()

st.title("AI-Based Organ Matching System")

st.sidebar.header("Add Donor")
d_name = st.sidebar.text_input("Donor Name")
d_age = st.sidebar.number_input("Donor Age", 18, 80)
d_bg = st.sidebar.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
d_organ = st.sidebar.selectbox("Donor Organ", ["Kidney", "Heart", "Liver", "Lung", "Pancreas"])
d_health = st.sidebar.slider("Health Score", 0.0, 1.0)
d_dist = st.sidebar.slider("Distance", 0.0, 1.0)

if st.sidebar.button("Add Donor"):
    if d_name.strip():
        add_donor(d_name.strip(), d_age, d_bg, d_organ, d_health, d_dist)
        st.sidebar.success("Donor added")
    else:
        st.sidebar.error("Please enter the donor name")

st.sidebar.header("Add Recipient")
r_name = st.sidebar.text_input("Recipient Name")
r_age = st.sidebar.number_input("Recipient Age", 18, 80)
r_bg = st.sidebar.selectbox("Recipient BG", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
r_organ = st.sidebar.selectbox("Required Organ", ["Kidney", "Heart", "Liver", "Lung", "Pancreas"])
r_urg = st.sidebar.slider("Urgency", 1, 10)

if st.sidebar.button("Add Recipient"):
    if r_name.strip():
        add_recipient(r_name.strip(), r_age, r_bg, r_organ, r_urg)
        st.sidebar.success("Recipient added")
    else:
        st.sidebar.error("Please enter the recipient name")

donors_df = get_donors()
recipients_df = get_recipients()

if donors_df.empty or recipients_df.empty:
    st.warning("Add donors and recipients first")
    st.stop()

donors_df["donor_name"] = donors_df["donor_name"].fillna("Unknown Donor")
donors_df["donor_organ"] = donors_df["donor_organ"].fillna("Kidney")
recipients_df["recipient_name"] = recipients_df["recipient_name"].fillna("Unknown Recipient")
recipients_df["required_organ"] = recipients_df["required_organ"].fillna("Kidney")

render_registry_tables(donors_df, recipients_df)

train_df = prepare_training_data(donors_df)
artifacts = train_model(train_df)

st.subheader("Select Recipient")
recipient_options = recipients_df.index.tolist()
selected_index = st.selectbox(
    "Choose recipient",
    recipient_options,
    format_func=lambda idx: f"{recipients_df.loc[idx, 'recipient_name']} - {recipients_df.loc[idx, 'required_organ']}",
)
selected = recipients_df.loc[selected_index]

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Name", selected["recipient_name"])
metric_col2.metric("Organ", selected["required_organ"])
metric_col3.metric("Blood Group", selected["recipient_bg"])

metric_col4, metric_col5 = st.columns(2)
metric_col4.metric("Age", int(selected["recipient_age"]))
metric_col5.metric("Urgency", int(selected["urgency_score"]))

pairs = donors_df.copy()
pairs["recipient_age"] = selected["recipient_age"]
pairs["recipient_bg"] = selected["recipient_bg"]
pairs["urgency_score"] = selected["urgency_score"]
pairs["organ_type"] = str(selected["required_organ"]).lower()
pairs["dataset_source"] = "live"
pairs["donor_health_score"] = pairs["health_score"]
pairs["recipient_health_score"] = max(0.3, 1 - (selected["urgency_score"] / 12))
pairs["wait_time_days"] = 45
pairs["distance_km"] = pairs["distance"].fillna(0.5).apply(lambda value: value * 100 if value <= 1 else value)
pairs["compatibility_score"] = pairs.apply(
    lambda row: compatibility_score(
        str(row["donor_bg"]).replace("+", "").replace("-", ""),
        str(selected["recipient_bg"]).replace("+", "").replace("-", ""),
    ),
    axis=1,
)
pairs = pairs[pairs["donor_organ"].str.lower() == str(selected["required_organ"]).lower()].copy()

if pairs.empty:
    st.warning("No donors found for the selected organ yet.")
    st.stop()

ranked = rank_recipients(pairs, top_n=10, model=artifacts.estimator)

if ranked.empty:
    st.warning("No blood-compatible donors found after the hard filter.")
    st.stop()

top = ranked.iloc[0]

st.subheader("Best Donor Match")
best_col1, best_col2, best_col3 = st.columns(3)
best_col1.metric("Donor Name", top["donor_name"])
best_col2.metric("Donor Organ", top["donor_organ"])
best_col3.metric("Predicted Score", round(top["predicted_score"], 3))

best_col4, best_col5 = st.columns(2)
best_col4.metric("Donor Age", int(top["donor_age"]))
best_col5.metric("Blood Compatibility", int(top["blood_compat_score"]))

st.markdown("### Why this donor?")
reasons = []
if top["blood_compat_score"] == 1:
    reasons.append("Blood type is compatible through the full ABO + Rh matrix")
if top["age_diff"] < 10:
    reasons.append("Low age difference between donor and recipient")
if top["health_gap"] >= 0:
    reasons.append("Donor health score is favorable for the recipient")
if top["urgency_score"] >= 8:
    reasons.append("Recipient urgency is high, increasing priority")

for reason in reasons:
    st.write("- ", reason)

st.subheader("Top Donor Matches")
st.dataframe(
    ranked[
        [
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
    ],
    use_container_width=True,
)

st.subheader("Top 5 Donor Scores")
top5 = ranked.head(5)
fig, ax = plt.subplots()
ax.barh(range(len(top5)), top5["predicted_score"])
ax.set_yticks(range(len(top5)))
ax.set_yticklabels(top5["donor_name"])
ax.set_xlabel("Predicted Match Score")
st.pyplot(fig)

render_model_insights(artifacts)
