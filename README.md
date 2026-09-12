# Zimswitch Fraud Detection System

An unofficial portfolio project: a real-time fraud detection system split
into two independently deployable pieces, the way a production system
would actually be built.

```
zimswitch-fraud-system/
├── api/                    ← the model-serving backend (FastAPI)
│   ├── main.py             ← the API itself
│   ├── features.py         ← shared feature engineering + explanation logic
│   ├── models/              ← the trained Random Forest pipeline
│   ├── requirements.txt
│   └── render.yaml         ← one-file Render deployment config
│
└── app/                    ← the user-facing frontend (Streamlit)
    ├── streamlit_frontend.py  ← calls the API over HTTP, has no model in it
    └── requirements.txt
```

**Why split it this way?** In a real payments company, the fraud model
doesn't live inside the app someone clicks around in — it runs as its
own service that any number of clients (a web dashboard, a mobile app,
a batch job) can call. This project mirrors that: the API can be
redeployed, scaled, or swapped for a better model without touching the
frontend at all, and the frontend could be a mobile app instead of
Streamlit tomorrow without touching the model.

## What it does

**Check one transaction** — type in amount, city, time, how it compares
to that card's recent activity — and click Analyze. The frontend sends
it to the API, which scores it and returns a plain verdict plus a
checklist of what it looked at and why.

**Upload a file of transactions** — upload a CSV of raw transactions
(a sample template is provided in the app) and every row gets scored
in one request. The API computes each card's behavioral features —
transaction velocity, travel speed between locations, deviation from
that card's typical spend — directly from the transaction history in
the file, the same way the model was trained. Results come back as a
table with a fraud probability and verdict per row, plus a downloadable
CSV.

Both paths call the same trained Random Forest (99.5% ROC-AUC on
held-out data) through the same API.

## Running it locally (both pieces)

**Terminal 1 — start the API:**
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` to see and test the API directly —
FastAPI generates interactive docs automatically. This is worth showing
in an interview: a working, documented API, independent of any UI.

**Terminal 2 — start the frontend:**
```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_frontend.py
```
The frontend defaults to `http://127.0.0.1:8000`, so it'll find the
locally-running API automatically.

## Deploying it for real (both free)

**1. Deploy the API to Render:**
- Push the `api/` folder to a GitHub repo (its own repo, or a subfolder — Render lets you set a root directory)
- On [render.com](https://render.com): New → Web Service → connect the repo
- Render will detect `render.yaml` and configure itself, or set manually:
  - Build command: `pip install -r requirements.txt`
  - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Deploy. You'll get a URL like `https://zimswitch-fraud-api.onrender.com`
- Note: Render's free tier spins down after inactivity, so the first request after a while takes 30-60 seconds to wake up — worth mentioning if you demo this live.

**2. Deploy the frontend to Streamlit Community Cloud:**
- Push the `app/` folder to a repo (or a subfolder of the same repo)
- On [share.streamlit.io](https://share.streamlit.io): New app → point at `streamlit_frontend.py`
- In the app's sidebar once it's live, paste your Render API URL into the "Fraud detection API URL" field — or set it permanently via an `API_URL` entry in the app's Secrets

You'll end up with two public URLs: one for the API (with live `/docs`),
one for the interactive frontend that talks to it.

## Files

| File | What it does |
|---|---|
| `api/main.py` | FastAPI app: loads the model once at startup, exposes `/predict` (single transaction), `/predict_batch` (CSV upload), `/health`, `/metadata`, `/cities` |
| `api/features.py` | Feature engineering (both single-transaction and batch/historical) and the plain-language checklist logic |
| `api/models/rf_pipeline.joblib` | The trained Random Forest pipeline |
| `api/models/metadata.json` | Evaluation metrics (ROC-AUC, confusion matrix, feature importance) served via `/metadata` |
| `app/streamlit_frontend.py` | The UI — two tabs, single transaction and bulk CSV upload, both pure frontend, no model, everything goes through the API |
| `app/sample_template.csv` | Example file showing the exact columns the batch upload expects |

Full training pipeline, dataset generator, and the standalone (non-split)
Streamlit apps from earlier iterations of this project are available
separately if useful for context on how the model was built.

---
⚠️ Unofficial portfolio project. Not affiliated with, endorsed by, or
built for Zimswitch. All data is synthetic — no real cardholder,
account, or transaction data is used anywhere in this project.
