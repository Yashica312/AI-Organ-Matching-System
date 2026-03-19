import sqlite3
import pandas as pd

DB_NAME = "organ_matching.db"

# =========================
# INITIALIZE DATABASE
# =========================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    # Donors table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS donors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        donor_age REAL,
        donor_bg TEXT,
        health_score REAL,
        distance REAL
    )
    """)

    # Recipients table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS recipients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient_age REAL,
        recipient_bg TEXT,
        urgency_score REAL
    )
    """)

    conn.commit()
    conn.close()


# =========================
# USER FUNCTIONS
# =========================
def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
        (username, password)
    )

    conn.commit()
    conn.close()


def validate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cur.fetchone()
    conn.close()

    return user is not None


# =========================
# DONOR FUNCTIONS
# =========================
def add_donor(age, bg, health, distance):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO donors (donor_age, donor_bg, health_score, distance) VALUES (?, ?, ?, ?)",
        (age, bg, health, distance)
    )

    conn.commit()
    conn.close()


def get_donors():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM donors", conn)
    conn.close()
    return df


# =========================
# RECIPIENT FUNCTIONS
# =========================
def add_recipient(age, bg, urgency):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO recipients (recipient_age, recipient_bg, urgency_score) VALUES (?, ?, ?)",
        (age, bg, urgency)
    )

    conn.commit()
    conn.close()


def get_recipients():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM recipients", conn)
    conn.close()
    return df