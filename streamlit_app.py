"""
Zimswitch Fraud Console — unofficial portfolio demo
=================================================================
Streamlit app. Scores what-if transactions live using the actual trained
Random Forest pipeline (same model evaluated in the metrics below), and
shows a per-instance explanation using the Logistic Regression pipeline's
coefficients.

Run locally:      streamlit run streamlit_app.py
Deploy for free:   push this repo to GitHub, then deploy at
                    https://share.streamlit.io (Streamlit Community Cloud)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import build_single_row, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODELS_DIR = Path(__file__).parent / "models"

MCC_LABELS = {
    "5411": "Grocery store", "5541": "Fuel station", "5812": "Restaurant",
    "5912": "Pharmacy", "6011": "ATM / cash withdrawal", "5651": "Clothing",
    "4900": "Utilities", "5999": "Retail — misc", "6051": "Money transfer",
    "7011": "Hotel", "5311": "Department store", "4814": "Telecom / airtime",
}
PATTERN_LABELS = {
    "odd_hour_high_value": "Odd-hour high value", "velocity_abuse": "Velocity abuse",
    "card_testing": "Card testing", "amount_anomaly": "Amount anomaly",
    "geo_jump": "Impossible travel",
}
FRIENDLY_FEATURE = {
    "amount": "Amount", "hour": "Hour of day", "is_odd_hour": "Odd-hour transaction (1-4am)",
    "day_of_week": "Day of week", "seconds_since_prev": "Time since last transaction",
    "km_from_prev_txn": "Distance from last transaction", "implied_speed_kmh": "Implied travel speed",
    "amount_zscore": "Deviation from usual spend", "txn_count_1h": "Transactions in the last hour",
    "txn_count_24h": "Transactions in the last 24 hours",
}
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SCENARIOS = {
    "Typical purchase": dict(channel="POS", mcc="5411", amount=38, currency="USD",
                              hour=14, dow=2, gap=240, dist=1.5, t1h=0, t24h=1, avg=42, std=17),
    "Card testing burst": dict(channel="Internet Banking", mcc="5999", amount=2, currency="USD",
                                hour=9, dow=3, gap=1, dist=0, t1h=9, t24h=11, avg=48, std=18),
    "Velocity abuse": dict(channel="ATM", mcc="6011", amount=130, currency="USD",
                            hour=11, dow=4, gap=6, dist=3, t1h=6, t24h=9, avg=52, std=19),
    "Amount anomaly": dict(channel="Internet Banking", mcc="5999", amount=900, currency="USD",
                            hour=15, dow=1, gap=300, dist=2, t1h=0, t24h=1, avg=45, std=12),
    "Impossible travel": dict(channel="POS", mcc="5812", amount=64, currency="USD",
                               hour=19, dow=5, gap=22, dist=480, t1h=1, t24h=2, avg=51, std=20),
    "Odd-hour cash-out": dict(channel="ATM", mcc="6011", amount=410, currency="USD",
                               hour=3, dow=6, gap=200, dist=5, t1h=0, t24h=1, avg=49, std=18),
}

DEFAULTS = SCENARIOS["Typical purchase"]
FIELD_KEYS = ["channel", "mcc", "amount", "currency", "hour", "dow", "gap", "dist", "t1h", "t24h", "avg", "std"]

st.set_page_config(page_title="Zimswitch Fraud Console", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_models():
    rf = joblib.load(MODELS_DIR / "rf_pipeline.joblib")
    lr = joblib.load(MODELS_DIR / "lr_pipeline.joblib")
    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)
    return rf, lr, meta


rf_pipe, lr_pipe, meta = load_models()

for k in FIELD_KEYS:
    if k not in st.session_state:
        st.session_state[k] = DEFAULTS[k]


def apply_scenario(name):
    for k, v in SCENARIOS[name].items():
        st.session_state[k] = v


# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.mono { font-family: 'IBM Plex Mono', monospace; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
.factor-row { display:flex; justify-content:space-between; padding:8px 12px; margin-bottom:6px;
              background:rgba(120,120,120,0.08); border-radius:8px; font-size:0.88rem; }
.badge { display:inline-block; padding:3px 9px; border-radius:6px; font-size:0.78rem;
         background:rgba(214,87,44,0.15); color:#D6572C; border:1px solid rgba(214,87,44,0.3); }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("## 🛡️ Zimswitch Fraud Console")
    st.caption("Fraud detection for interbank switch traffic — unofficial portfolio demo, synthetic data")
with col_status:
    st.markdown("<div style='text-align:right; padding-top:18px; color:#2FA77E;'>● scoring live</div>",
                unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions scored", f"{meta['n_transactions']:,}")
m2.metric("Flagged as fraud", f"{meta['fraud_rate']*100:.2f}%")
m3.metric("ROC-AUC (Random Forest)", f"{meta['metrics']['random_forest']['roc_auc']:.3f}")
tp = meta["metrics"]["random_forest"]["confusion_matrix"]["tp"]
fp = meta["metrics"]["random_forest"]["confusion_matrix"]["fp"]
m4.metric("Precision at threshold", f"{tp/(tp+fp)*100:.0f}%")

st.divider()

# ---------- Simulator ----------
st.markdown("### Score a transaction")
st.caption("Pick a scenario or adjust the inputs directly. The score comes from the actual "
           "trained Random Forest pipeline evaluated below — not a mock number.")

btn_cols = st.columns(len(SCENARIOS))
for i, name in enumerate(SCENARIOS):
    if btn_cols[i].button(name, use_container_width=True):
        apply_scenario(name)
        st.rerun()

left, right = st.columns([1.2, 0.8])

with left:
    c1, c2 = st.columns(2)
    with c1:
        channel = st.selectbox("Channel", ["POS", "ATM", "ZIPIT", "Mobile Money", "Internet Banking"],
                                key="channel")
        amount = st.slider("Amount", 1, 1500, key="amount")
        hour = st.slider("Hour of day", 0, 23, key="hour")
        gap = st.slider("Minutes since this card's last transaction", 1, 1440, key="gap")
        t1h = st.slider("Transactions in the last hour", 0, 15, key="t1h")
        avg = st.slider("Card's typical average spend ($)", 5, 200, key="avg")
    with c2:
        mcc = st.selectbox("Merchant category", list(MCC_LABELS.keys()),
                            format_func=lambda x: MCC_LABELS[x], key="mcc")
        currency = st.selectbox("Currency", ["USD", "ZWG"], key="currency")
        dow = st.slider("Day of week", 0, 6, format="", key="dow",
                         help="0 = Monday, 6 = Sunday")
        st.caption(DOW[st.session_state["dow"]])
        dist = st.slider("Distance from usual location (km)", 0, 900, key="dist")
        t24h = st.slider("Transactions in the last 24 hours", 0, 30, key="t24h")
        std = st.slider("Card's typical spend swing (std, $)", 2, 100, key="std")

with right:
    row = build_single_row(
        amount=st.session_state["amount"], hour=st.session_state["hour"],
        day_of_week=st.session_state["dow"], minutes_since_prev=st.session_state["gap"],
        km_from_prev=st.session_state["dist"], txn_count_1h=st.session_state["t1h"],
        txn_count_24h=st.session_state["t24h"], channel=st.session_state["channel"],
        mcc=st.session_state["mcc"], currency=st.session_state["currency"],
        avg_amount=st.session_state["avg"], std_amount=st.session_state["std"],
    )

    prob = float(rf_pipe.predict_proba(row)[0, 1])
    pct = round(prob * 100, 1)

    if prob < 0.15:
        color, verdict = "#2FA77E", "Approve — low risk"
    elif prob < 0.5:
        color, verdict = "#D4A13D", "Review — elevated risk"
    elif prob < 0.8:
        color, verdict = "#D6572C", "Step-up auth — high risk"
    else:
        color, verdict = "#C6414B", "Block — critical risk"

    st.markdown(
        f"<div style='text-align:center; padding:12px 0 4px;'>"
        f"<div style='font-family:Space Grotesk; font-size:44px; font-weight:700; color:{color};'>{pct}%</div>"
        f"<div style='color:#888; font-size:0.85rem;'>probability of fraud</div>"
        f"</div>", unsafe_allow_html=True
    )
    st.progress(min(prob, 1.0))
    st.markdown(
        f"<div style='text-align:center; padding:10px; border-radius:8px; "
        f"background:{color}22; color:{color}; font-weight:600; margin-top:8px;'>{verdict}</div>",
        unsafe_allow_html=True
    )

    # per-instance explanation via the logistic regression pipeline
    prep = lr_pipe.named_steps["prep"]
    coefs = lr_pipe.named_steps["clf"].coef_[0]
    feature_names = prep.get_feature_names_out()
    transformed = prep.transform(row)
    transformed = np.asarray(transformed.todense()) if hasattr(transformed, "todense") else transformed
    contributions = transformed[0] * coefs
    order = np.argsort(-np.abs(contributions))[:5]

    st.markdown("<div style='margin-top:18px; font-size:0.85rem; color:#888;'>Top contributing factors</div>",
                unsafe_allow_html=True)
    for idx in order:
        raw_name = feature_names[idx]
        contrib = contributions[idx]
        if raw_name.startswith("num__"):
            label = FRIENDLY_FEATURE.get(raw_name.replace("num__", ""), raw_name)
        elif raw_name.startswith("cat__channel_"):
            label = "Channel: " + raw_name.replace("cat__channel_", "")
        elif raw_name.startswith("cat__mcc_"):
            code = raw_name.replace("cat__mcc_", "")
            label = "Category: " + MCC_LABELS.get(code, code)
        elif raw_name.startswith("cat__currency_"):
            label = "Currency: " + raw_name.replace("cat__currency_", "")
        else:
            label = raw_name
        arrow = "▲ raises risk" if contrib >= 0 else "▼ lowers risk"
        arrow_color = "#E8703F" if contrib >= 0 else "#2FA77E"
        st.markdown(
            f"<div class='factor-row'><span>{label}</span>"
            f"<span class='mono' style='color:{arrow_color};'>{arrow}</span></div>",
            unsafe_allow_html=True
        )

st.divider()

# ---------- Model performance ----------
st.markdown("### Model performance")
st.caption(f"Evaluated on a held-out 25% test split ({meta['test_size']:,} transactions, stratified).")

perf1, perf2 = st.columns(2)
with perf1:
    rf_m = meta["metrics"]["random_forest"]
    lr_m = meta["metrics"]["logistic_regression"]
    st.markdown(f"""
| Metric | Random Forest | Logistic Regression |
|---|---|---|
| ROC-AUC | {rf_m['roc_auc']:.3f} | {lr_m['roc_auc']:.3f} |
| PR-AUC | {rf_m['pr_auc']:.3f} | {lr_m['pr_auc']:.3f} |
""")
    cm = rf_m["confusion_matrix"]
    st.markdown("**Confusion matrix — Random Forest**")
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual legit", "Actual fraud"],
        columns=["Predicted legit", "Predicted fraud"],
    )
    st.dataframe(cm_df, use_container_width=True)

with perf2:
    st.markdown("**Feature importance (Random Forest)**")
    imp_df = pd.DataFrame(meta["feature_importance"], columns=["feature", "importance"])
    imp_df["feature"] = imp_df["feature"].replace(FRIENDLY_FEATURE)
    st.bar_chart(imp_df.set_index("feature"), horizontal=True)

st.divider()

# ---------- Flagged transactions ----------
st.markdown("### Recently flagged")
st.caption("A sample of transactions the model flagged in the test set.")
flagged_df = pd.DataFrame(meta["flagged_samples"])
flagged_df["fraud_pattern"] = flagged_df["fraud_pattern"].map(PATTERN_LABELS).fillna(flagged_df["fraud_pattern"])
flagged_df = flagged_df.rename(columns={
    "transaction_id": "Transaction", "card_id": "Card", "channel": "Channel",
    "amount": "Amount", "currency": "Currency", "city": "City", "hour": "Hour",
    "fraud_pattern": "Pattern",
})
st.dataframe(flagged_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "⚠️ Unofficial portfolio project, not affiliated with, endorsed by, or built for Zimswitch. "
    "Made independently to demonstrate fraud-detection skills for a job application. All data is "
    "synthetic — no real cardholder, account, or transaction data is used anywhere in this project."
)
