# Payment Switch Fraud Detection — Synthetic Data Project

A portfolio project simulating fraud detection on a national payment
switch (structured like Zimswitch): multi-bank POS, ATM, ZIPIT-style
transfers, and mobile-money interoperability.

**All data is synthetic** — generated programmatically, no real
cardholder or transaction data of any kind. This is a safe way to build
and demonstrate fraud-detection skills without needing access to a real
institution's data.

## Files

| File | Description |
|---|---|
| `generate_data.py` | Generates `transactions.csv` (~61k transactions, 4,000 synthetic cards) with 5 injected fraud patterns |
| `features.py` | Shared feature engineering, used by both training and the app |
| `train_model.py` | Trains/evaluates Logistic Regression and Random Forest models (standalone report) |
| `save_models.py` | Trains and pickles both pipelines into `models/` for the Streamlit app |
| `streamlit_app.py` | **The technical app** — sliders, full metrics, per-transaction coefficient breakdown |
| `explainer_app.py` | **The plain-language app** — same real model, no jargon, story-driven, three-button verdicts (best for non-technical reviewers) |
| `requirements.txt` | Python dependencies for the Streamlit app |
| `models/rf_pipeline.joblib` | Trained Random Forest pipeline (the one scoring transactions live in the app) |
| `models/lr_pipeline.joblib` | Trained Logistic Regression pipeline (used for the per-transaction explanation) |
| `models/metadata.json` | Evaluation metrics, feature importances, and sample flagged transactions |
| `transactions.csv` | The generated dataset |
| `cards.csv` | Synthetic card/cardholder profiles (home city, average spend) |

## Running the app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py       # technical version
streamlit run explainer_app.py       # plain-language version
```

This opens the console at `http://localhost:8501`. The `models/` folder
already contains trained pipelines, so it starts instantly — no need to
regenerate data or retrain unless you want to.

To regenerate everything from scratch:
```bash
python generate_data.py   # rebuilds transactions.csv
python save_models.py     # retrains and re-pickles both models
```

## Deploying it publicly (free)

1. Push this whole folder to a GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app".
3. Point it at your repo, branch, and `streamlit_app.py` as the main file.
4. Deploy. You'll get a public URL like
   `https://your-username-beacon-fraud.streamlit.app` you can put on a CV,
   LinkedIn, or send directly to Zimswitch.

The repo is small enough (~9 MB, mostly the pickled Random Forest) to
deploy comfortably on Streamlit Community Cloud's free tier.

## Dataset design

Transactions span 6 banks-worth of issuers/acquirers (CBZ, Stanbic, FBC,
Steward, ZB, NMB, CABS, Ecobank, BancABC, POSB), 5 channels (POS, ATM,
ZIPIT, Mobile Money, Internet Banking), and two currencies (ZWG, USD),
mirroring how a real switch aggregates interbank and mobile-money
interoperability traffic.

**Fraud rate: ~2%**, in line with real-world card fraud prevalence —
this matters because it forces you to handle severe class imbalance
rather than a toy 50/50 dataset.

### Injected fraud patterns
1. **Card testing** — many small (<$3), rapid transactions on one card, typical of criminals validating stolen card numbers
2. **Velocity abuse** — unusually frequent transactions in a short window
3. **Amount anomaly** — a transaction far above the card's historical spending profile
4. **Geographic jump ("impossible travel")** — two transactions too far apart geographically to be physically possible in the elapsed time
5. **Odd-hour high value** — large transactions at 1–4 AM

## Modeling approach

Feature engineering is done **causally** — every behavioral feature
(running average spend, transaction velocity, implied travel speed) is
computed using only transactions *prior* to the current one, avoiding
label/target leakage that's a common mistake in fraud-detection
projects.

Key engineered features:
- `implied_speed_kmh` — distance-over-time between consecutive
  transactions on the same card ("impossible travel" signal)
- `txn_count_1h` / `txn_count_24h` — trailing transaction velocity
- `amount_zscore` — deviation from the card's own running average spend
- `seconds_since_prev` — time gap since the card's last transaction

Two models are trained and compared:
- **Logistic Regression** (interpretable baseline, `class_weight="balanced"`)
- **Random Forest** (captures non-linear interactions between features)

### Results

| Model | ROC-AUC | PR-AUC | Fraud Recall | Fraud Precision |
|---|---|---|---|---|
| Logistic Regression | 0.986 | 0.919 | 0.92 | 0.48 |
| Random Forest | 0.995 | 0.948 | 0.88 | 0.93 |

PR-AUC (precision-recall AUC) is reported alongside ROC-AUC because
with ~2% fraud prevalence, ROC-AUC alone can look deceptively good —
PR-AUC is the more honest metric for imbalanced fraud problems.

## How to extend this for an interview

- Swap in **XGBoost/LightGBM** and compare against the Random Forest
- Add a **cost-sensitive evaluation**: assign a real monetary cost to
  false negatives (missed fraud) vs. false positives (blocked
  legitimate customers) and pick the decision threshold that
  minimizes expected cost, not just F1
- Try **graph-based features** — shared devices/IPs across cards is a
  strong real-world fraud signal this synthetic dataset doesn't model
- Discuss **concept drift** — fraud patterns change over time, so a
  production system needs monitoring and periodic retraining, not a
  one-off model
- Talk through the **precision/recall tradeoff** in a live interview:
  in payments, a false positive means blocking a real customer's
  transaction (bad for user experience and revenue), while a false
  negative means fraud losses — the "right" threshold depends on the
  business's cost structure, which is a great thing to discuss with
  interviewers rather than just optimizing accuracy
