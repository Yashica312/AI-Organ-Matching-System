# AI Organ Matching System

## Project Description

AI Organ Matching System is a full-stack healthcare decision-support demo built for donor-recipient ranking. It combines a polished React dashboard, a FastAPI backend, and an upgraded XGBoost-based matching engine to rank multiple donors, explain why the top donors were selected, and visualize the match results in a presentation-ready interface.

The system is structured for Lovable preview and local development:

- `client/` → React + Vite frontend
- `server/` → FastAPI backend
- `model/` → ML training code and saved model artifacts
- `data/` → source datasets

## Features

- ML-based donor-recipient ranking using engineered transplant features
- Multi-donor matching with Top 5 ranked donor results
- Explainable AI output for each donor match
- Demo Mode for stable presentations with preloaded realistic rankings
- Live API mode for backend-driven ranking
- Compatibility filtering and organ filtering
- Ranked donor cards with visual highlighting for the top match
- Match-score bar chart for quick comparison
- Responsive dashboard layout for desktop and mobile

## Tech Stack

- React
- TypeScript
- Vite
- Tailwind CSS
- FastAPI
- Python
- XGBoost
- scikit-learn
- Pandas
- NumPy
- Joblib

## How to Run

### Backend

```powershell
cd server
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```powershell
cd client
npm install
npm run dev
```

### Environment

Create or update:

`client/.env`

```env
VITE_API_URL=http://localhost:8000
```

## Notes

- The backend endpoint used by the polished dashboard is `POST /match-multiple`.
- The frontend supports both live API calls and demo mode.
- The XGBoost model artifact is saved to `model/xgb_model.pkl` when training runs.
- Existing ML code and datasets are preserved under `model/` and `data/`.
