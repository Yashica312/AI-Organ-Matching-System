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

st.set_page_config(page_title="AI Organ Matching", layout="wide")

# =========================
# INIT DATABASE
# =========================
init_db()

# =========================
# LOGIN
# =========================
st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if validate_user(username, password):
        st.session_state["logged_in"] = True
    else:
        st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.stop()

st.title("🧠 AI-Based Organ Matching System")

# =========================
# ADD DATA (REAL-TIME)
# =========================
st.sidebar.subheader("➕ Add Donor")

d_age = st.sidebar.number_input("Donor Age", 18, 80)
d_bg = st.sidebar.selectbox("Blood Group", ["A", "B", "AB", "O"])
d_health = st.sidebar.slider("Health Score", 0.0, 1.0)
d_dist = st.sidebar.slider("Distance", 0.0, 1.0)

if st.sidebar.button("Add Donor"):
    add_donor(d_age, d_bg, d_health, d_dist)
    st.success("Donor added")

st.sidebar.subheader("➕ Add Recipient")

r_age = st.sidebar.number_input("Recipient Age", 18, 80)
r_bg = st.sidebar.selectbox("Recipient BG", ["A", "B", "AB", "O"])
r_urg = st.sidebar.slider("Urgency", 1, 10)

if st.sidebar.button("Add Recipient"):
    add_recipient(r_age, r_bg, r_urg)
    st.success("Recipient added")

# =========================
# LOAD DB DATA
# =========================
donors_df = get_donors()
recipients_df = get_recipients()

if donors_df.empty or recipients_df.empty:
    st.warning("Add donors and recipients first")
    st.stop()

# =========================
# TRAIN MODEL (using donors as base)
# =========================
# Create dummy dataset for training
train_df = donors_df.copy()
train_df["recipient_age"] = 40
train_df["recipient_bg"] = "A"
train_df["urgency_score"] = 5
train_df["organ_type"] = "Kidney"
train_df["dataset_source"] = "live"

train_df["compatibility_score"] = train_df.apply(
    lambda x: compatibility_score(x["donor_bg"], "A"), axis=1
)

train_df["success_probability"] = (
    0.5 * train_df["compatibility_score"] +
    0.3 * (train_df["urgency_score"] / 10) +
    0.2 * train_df["health_score"] -
    0.1 * train_df["distance"]
)

model = train_model(train_df)

# =========================
# SELECT RECIPIENT
# =========================
st.subheader("👤 Select Recipient")

recipient_index = st.selectbox("Choose recipient", recipients_df.index)
selected = recipients_df.loc[recipient_index]

col1, col2, col3 = st.columns(3)
col1.metric("Age", int(selected["recipient_age"]))
col2.metric("Blood Group", selected["recipient_bg"])
col3.metric("Urgency", int(selected["urgency_score"]))

# =========================
# BUILD MATCHES
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

FEATURES = [
    "donor_age","recipient_age","donor_bg","recipient_bg",
    "health_score","urgency_score","distance",
    "compatibility_score","organ_type","dataset_source"
]

X = pairs[FEATURES]
pairs["ml_score"] = model.predict(X)

pairs["final_score"] = (
    0.4 * pairs["ml_score"] +
    0.3 * pairs["urgency_score"] +
    0.2 * pairs["compatibility_score"] -
    0.1 * pairs["distance"]
)

ranked = pairs.sort_values(by="final_score", ascending=False)

# =========================
# TOP DONOR
# =========================
top = ranked.iloc[0]

st.subheader("🏆 Best Donor Match")

c1, c2, c3 = st.columns(3)
c1.metric("Donor Age", int(top["donor_age"]))
c2.metric("Compatibility", round(top["compatibility_score"], 2))
c3.metric("Final Score", round(top["final_score"], 3))

# =========================
# EXPLANATION
# =========================
st.markdown("### 💡 Why this donor?")

reasons = []

if top["compatibility_score"] == 1:
    reasons.append("Perfect blood match")
if top["urgency_score"] >= 8:
    reasons.append("High urgency")
if top["distance"] < 0.2:
    reasons.append("Close distance")
if top["health_score"] > 0.8:
    reasons.append("Healthy donor")

for r in reasons:
    st.write("✔️", r)

# =========================
# COLOR TABLE
# =========================
st.subheader("📋 Top Donors")

def highlight(row):
    if row.name == ranked.index[0]:
        return ['background-color: lightgreen'] * len(row)
    elif row.name == ranked.index[1]:
        return ['background-color: lightyellow'] * len(row)
    return [''] * len(row)

display_cols = [
    "donor_age","donor_bg","health_score",
    "distance","compatibility_score","final_score"
]

st.dataframe(ranked[display_cols].head(10).style.apply(highlight, axis=1))

# =========================
# BETTER GRAPH (HORIZONTAL)
# =========================
st.subheader("📊 Top 5 Donors (Score Comparison)")

top5 = ranked.head(5)

fig, ax = plt.subplots()

ax.barh(
    y=[f"Donor {i+1}" for i in range(len(top5))],
    width=top5["final_score"]
)

ax.set_xlabel("Final Score")
ax.set_title("Top Donor Ranking")

st.pyplot(fig)