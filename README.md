# AI Organ Matching System

## Project Overview

This project is a Python-based AI organ donor-recipient matching system built around a Streamlit interface. It combines rule-based compatibility scoring, machine learning ranking, and a lightweight SQLite-backed donor and recipient registry so users can add real-time entries and identify the best donor match for a selected recipient.

The current implementation focuses on organ matching features such as donor age, recipient age, blood group compatibility, health score, urgency, and distance. A `RandomForestRegressor` is used to estimate match quality and support final donor ranking.

## Features

- Streamlit dashboard for interactive donor-recipient matching
- Secure login and registration flow backed by SQLite
- Real-time donor and recipient entry forms
- Rule-based blood group compatibility scoring
- Machine learning-assisted donor ranking
- Visual ranking display and score comparison chart
- Modular project structure for preprocessing, modeling, evaluation, ranking, and rules

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- SQLite
- Matplotlib

## Project Structure

- `app.py`: Streamlit user interface
- `database.py`: SQLite setup and CRUD helpers
- `preprocessing.py`: dataset schema mapping and shared feature preparation
- `rules.py`: compatibility and target scoring logic
- `model.py`: model training pipeline
- `evaluation.py`: evaluation metrics for kidney and heart datasets
- `ranking.py`: candidate ranking logic
- `data_generation.py`: synthetic dataset generator

## How to Run the App

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Start the Streamlit app:

```powershell
streamlit run app.py
```

3. Open the local Streamlit URL shown in the terminal.
4. Register a new account from the app or log in with an existing SQLite-backed user account.

## Notes

- The kidney dataset is used as the training dataset.
- The heart dataset is used for cross-organ generalization testing.
- If GitHub authentication prompts for a password during push, use a GitHub Personal Access Token instead of your account password.
