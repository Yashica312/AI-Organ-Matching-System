# AI Organ Matching System

## Project Overview

This repository has been restructured into a Lovable-compatible full-stack app:

- `client/` contains a Vite + React frontend
- `server/` contains a FastAPI backend
- `model/` contains the original ML and legacy Python app code
- `data/` contains the original datasets

The web app now includes a professional React dashboard, a FastAPI backend, and ML-aware matching logic. The frontend calls `POST /match`, the backend attempts to load a supported model artifact from `model/`, and it falls back to realistic generated match results if no runnable model file is available.

## Features

- React frontend built with Vite
- FastAPI backend with CORS enabled
- Frontend-to-backend fetch flow via `VITE_API_URL`
- Safe error handling in the frontend
- Dynamic backend model loading for `.pkl`, `.h5`, and `.pt` files
- Realistic fallback predictions when model inference is unavailable
- Existing ML code preserved under `model/`
- Existing datasets preserved under `data/`

## Tech Stack

- React
- Vite
- FastAPI
- Uvicorn
- Python

## Project Structure

- `client/`: Lovable frontend entry
- `server/`: API backend
- `model/`: legacy ML code and scripts
- `data/`: dataset files

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

Environment configuration:

`client/.env`

```env
VITE_API_URL=http://localhost:8000
```

The frontend uses this variable when calling the backend.

## Notes

- The ML code was preserved and moved into `model/` without being folded into the new web app.
- The datasets were preserved and moved into `data/`.
- If a supported saved model file is later added to `model/`, the backend will try to use it automatically.
