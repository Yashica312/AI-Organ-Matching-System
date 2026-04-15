import sqlite3

import pandas as pd


DB_NAME = "organ_matching.db"


def _ensure_column(cur, table_name, column_name, column_definition):
    """Add a column to an existing table if it does not exist yet."""
    cur.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cur.fetchall()}
    if column_name not in existing_columns:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


# =========================
# INITIALIZE DATABASE
# =========================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
        """
    )

    # Donors table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS donors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            donor_name TEXT,
            donor_age REAL,
            donor_bg TEXT,
            donor_organ TEXT,
            health_score REAL,
            distance REAL
        )
        """
    )

    # Recipients table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recipients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipient_name TEXT,
            recipient_age REAL,
            recipient_bg TEXT,
            required_organ TEXT,
            urgency_score REAL
        )
        """
    )

    # Keep existing databases compatible after schema changes.
    _ensure_column(cur, "donors", "donor_name", "TEXT")
    _ensure_column(cur, "donors", "donor_organ", "TEXT")
    _ensure_column(cur, "recipients", "recipient_name", "TEXT")
    _ensure_column(cur, "recipients", "required_organ", "TEXT")

    conn.commit()
    conn.close()


# =========================
# USER FUNCTIONS
# =========================
def add_user(username, password, confirm_password=None):
    if confirm_password is not None and password != confirm_password:
        raise ValueError("Passwords do not match")
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
        (username, password),
    )

    conn.commit()
    conn.close()


def validate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password),
    )

    user = cur.fetchone()
    conn.close()

    return user is not None


# =========================
# DONOR FUNCTIONS
# =========================
def add_donor(name, age, bg, organ, health, distance):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO donors (donor_name, donor_age, donor_bg, donor_organ, health_score, distance)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, age, bg, organ, health, distance),
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
def add_recipient(name, age, bg, organ, urgency):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO recipients (recipient_name, recipient_age, recipient_bg, required_organ, urgency_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, age, bg, organ, urgency),
    )

    conn.commit()
    conn.close()


def get_recipients():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM recipients", conn)
    conn.close()
    return df
