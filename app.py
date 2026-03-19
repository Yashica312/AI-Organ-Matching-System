import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import train_model
from rules import compatibility_score
from database import (
    init_db, add_user, validate_user,
    add_donor, add_recipient,
    get_donors, get_recipients
)

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Organ Matching", layout="wide")

# =========================
# INIT DB
# =========================
init_db()

# =========================
# AUTH UI
# =========================
def auth_ui():
    st.markdown("## 🔐 Secure Match Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # LOGIN
    with tab1:
        user = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if validate_user(user, pwd):
                st.session_state["logged_in"] = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # REGISTER
    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if new_pass != confirm:
                st.error("Passwords do not match")
            elif new_user == "":
                st.error("Username cannot be empty")
            else:
                add_user(new_user, new_pass)
                st.success("Account created! Please login.")

# =========================
# LOGIN CHECK
# =========================
if "logged_in" not in st.session_state:
    auth_ui()
    st.stop()

# =========================
# MAIN APP
# =========================
st.title("🧠 AI-Based Organ Matching System")

# =========================
# ADD DATA
# =========================
st.sidebar.header("➕ Add Donor")
d_age = st.sidebar.number_input("Donor Age", 18, 80)
d_bg = st.sidebar.selectbox("Blood Group", ["A", "B", "AB", "O"])
d_health = st.sidebar.slider("Health Score", 0.0, 1.0)
d_dist = st.sidebar.slider("Distance", 0.0, 1.0)

if st.sidebar.button("Add Donor"):
    add_donor(d_age, d_bg, d_health, d_dist)
    st.sidebar.success("Donor added")

st.sidebar.header("➕ Add Recipient")
r_age = st.sidebar.number_input("Recipient Age", 18, 80)
r_bg = st.sidebar.selectbox("Recipient BG", ["A", "B", "AB", "O"])
r_urg = st.sidebar.slider("Urgency", 1, 10)

if st.sidebar.button("Add Recipient"):
    add_recipient(r_age, r_bg, r_urg)
    st.sidebar.success("Recipient added")

# =========================
# LOAD DATA
# =========================
donors_df = get_donors()
recipients_df = get_recipients()

if donors_df.empty or recipients_df.empty:
    st.warning("Add donors and recipients first")
    st.stop()

# =========================
# PREPARE TRAIN DATA
# =========================
train_df = donors_df.copy()

# Add required fields for model
train_df["recipient_age"] = 40
train_df["recipient_bg"] = "A"
train_df["urgency_score"] = 5
train_df["organ_type"] = "Kidney"
train_df["dataset_source"] = "live"

train_df["compatibility_score"] = train_df.apply(
    lambda x: compatibility_score(x["donor_bg"], "A"), axis=1
)

# Create target
train_df["success_probability"] = (
    0.5 * train_df["compatibility_score"] +
    0.3 * (train_df["urgency_score"] / 10) +
    0.2 * train_df["health_score"] -
    0.1 * train_df["distance"]
)

# =========================
# TRAIN MODEL
# =========================
model = train_model(train_df)

# =========================
# SELECT RECIPIENT
# =========================
st.subheader("👤 Select Recipient")

rec_index = st.selectbox("Choose recipient", recipients_df.index)
selected = recipients_df.loc[rec_index]

c1, c2, c3 = st.columns(3)
c1.metric("Age", int(selected["recipient_age"]))
c2.metric("Blood Group", selected["recipient_bg"])
c3.metric("Urgency", int(selected["urgency_score"]))

# =========================
# MATCHING
# =========================
pairs = donors_df.copy()

pairs["recipient_age"] = selected["recipient_age"]
pairs["recipient_bg"] = selected["recipient_bg"]
pairs["urgency_score"] = selected["urgency_score"]
pairs["organ_type"] = "Kidney"
pairs["dataset_source"] = "live"

pairs["compatibility_score"] = pairs.apply(
    lambda x: compatibility_score(x["donor_bg"], selected["recipient_bg"]),
    axis=1
)

# IMPORTANT: keep same columns as training
FEATURES = [
    "donor_age", "recipient_age", "donor_bg", "recipient_bg",
    "health_score", "urgency_score", "distance",
    "compatibility_score", "organ_type", "dataset_source"
]

pairs["ml_score"] = model.predict(pairs[FEATURES])

# Ranking
pairs["final_score"] = (
    0.4 * pairs["ml_score"] +
    0.3 * pairs["urgency_score"] +
    0.2 * pairs["compatibility_score"] -
    0.1 * pairs["distance"]
)

ranked = pairs.sort_values(by="final_score", ascending=False)

# =========================
# BEST MATCH
# =========================
top = ranked.iloc[0]

st.subheader("🏆 Best Donor Match")

col1, col2, col3 = st.columns(3)
col1.metric("Donor Age", int(top["donor_age"]))
col2.metric("Compatibility", round(top["compatibility_score"], 2))
col3.metric("Final Score", round(top["final_score"], 3))

# =========================
# EXPLANATION
# =========================
st.markdown("### 💡 Why this donor?")

reasons = []
if top["compatibility_score"] == 1:
    reasons.append("Perfect blood group match")
if top["urgency_score"] >= 8:
    reasons.append("High urgency")
if top["distance"] < 0.2:
    reasons.append("Close proximity")
if top["health_score"] > 0.8:
    reasons.append("Healthy donor")

for r in reasons:
    st.write("✔️", r)

# =========================
# GRAPH (FIXED)
# =========================
st.subheader("📊 Top 5 Donor Scores")

top5 = ranked.head(5)

fig, ax = plt.subplots()
ax.barh(range(len(top5)), top5["final_score"])
ax.set_yticks(range(len(top5)))
ax.set_yticklabels([f"Donor {i+1}" for i in range(len(top5))])
ax.set_xlabel("Final Score")

st.pyplot(fig)