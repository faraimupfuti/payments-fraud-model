"""
Zimswitch Transaction Analyzer — real-time fraud check (Streamlit)
=================================================================
Redesigned around a single idea: type in a real transaction, watch the
actual trained model check it step by step, get a clear verdict.

Unlike the scenario-button apps in this project, every input here is
free-form — there are no preset "scenarios." The checklist shown during
analysis is generated live from whatever numbers you enter, using the
same trained Random Forest pipeline that's evaluated in the metrics
section at the bottom.

Run:      streamlit run fraud_detector_app.py
Deploy:   push this repo to GitHub, deploy free at share.streamlit.io
"""

import json
import time
from pathlib import Path

import joblib
import streamlit as st

from features import build_single_row, haversine_km

MODELS_DIR = Path(__file__).parent / "models"

CITIES = {
    "Harare": (-17.8292, 31.0522), "Bulawayo": (-20.1500, 28.5833),
    "Mutare": (-18.9707, 32.6709), "Gweru": (-19.4500, 29.8167),
    "Kwekwe": (-18.9281, 29.8149), "Masvingo": (-20.0637, 30.8277),
    "Chinhoyi": (-17.3667, 30.2000), "Victoria Falls": (-17.9243, 25.8567),
}
MCC_LABELS = {
    "5411": "Groceries", "5541": "Fuel", "5812": "Restaurant", "5912": "Pharmacy",
    "6011": "ATM / cash withdrawal", "5651": "Clothing", "4900": "Utilities",
    "5999": "Retail — misc", "6051": "Money transfer", "7011": "Hotel",
    "5311": "Department store", "4814": "Telecom / airtime",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Zimswitch Transaction Analyzer", page_icon="🔍", layout="centered")


@st.cache_resource
def load_model():
    rf = joblib.load(MODELS_DIR / "rf_pipeline.joblib")
    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)
    return rf, meta


rf_pipe, meta = load_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.mono { font-family: 'IBM Plex Mono', monospace; }
.check-row {
    display:flex; align-items:flex-start; gap:12px; padding:12px 14px; border-radius:10px;
    margin-bottom:8px; background:rgba(120,120,120,0.06); border-left:3px solid transparent;
    animation: fadein 0.3s ease-in;
}
.check-row.flagged { border-left-color:#D6572C; background:rgba(214,87,44,0.08); }
.check-row.clear { border-left-color:#2FA77E; background:rgba(47,167,126,0.06); }
.check-icon { font-size:1.15rem; line-height:1.3; }
.check-label { font-weight:600; font-size:0.92rem; }
.check-value { font-family:'IBM Plex Mono', monospace; font-size:0.82rem; color:#999; margin-top:1px;}
.check-detail { font-size:0.85rem; color:#aaa; margin-top:4px; }
@keyframes fadein { from {opacity:0; transform:translateY(4px);} to {opacity:1; transform:translateY(0);} }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔍 Zimswitch Transaction Analyzer")
st.caption("Type in a transaction. Watch the model check it, step by step, in real time — "
           "no preset examples, this runs on whatever you enter below.")
st.divider()

# ---------------- Input form ----------------
st.markdown("### This transaction")
c1, c2 = st.columns(2)
with c1:
    amount = st.number_input("Amount", min_value=0.01, value=50.0, step=1.0)
    currency = st.selectbox("Currency", ["USD", "ZWG"])
    channel = st.selectbox("Channel", ["POS", "ATM", "ZIPIT", "Mobile Money", "Internet Banking"])
with c2:
    mcc = st.selectbox("Type of purchase", list(MCC_LABELS.keys()), format_func=lambda x: MCC_LABELS[x])
    city_now = st.selectbox("City where this is happening", list(CITIES.keys()))
    hour = st.slider("Hour of day", 0, 23, 14)

day_name = st.select_slider("Day of week", options=DAYS, value="Wednesday")
day_of_week = DAYS.index(day_name)

st.markdown("### This card's recent activity")
c3, c4 = st.columns(2)
with c3:
    city_prev = st.selectbox("City of this card's previous transaction", list(CITIES.keys()),
                              index=list(CITIES.keys()).index(city_now))
    minutes_since_prev = st.number_input("Minutes since that previous transaction", min_value=1, value=120)
    typical_amount = st.number_input("What does this card usually spend per transaction?",
                                      min_value=1.0, value=50.0, step=1.0)
with c4:
    t1h = st.number_input("Transactions on this card in the last hour", min_value=0, value=0)
    t24h = st.number_input("Transactions on this card in the last 24 hours", min_value=0, value=1)

analyze = st.button("Analyze this transaction", type="primary", use_container_width=True)

st.divider()

if analyze:
    lat1, lon1 = CITIES[city_now]
    lat2, lon2 = CITIES[city_prev]
    km_apart = float(haversine_km(lat1, lon1, lat2, lon2))

    row = build_single_row(
        amount=amount, hour=hour, day_of_week=day_of_week,
        minutes_since_prev=minutes_since_prev, km_from_prev=km_apart,
        txn_count_1h=t1h, txn_count_24h=t24h, channel=channel, mcc=mcc,
        currency=currency, avg_amount=typical_amount, std_amount=typical_amount * 0.4,
    )

    hours_gap = max(minutes_since_prev / 60, 1 / 60)
    implied_speed = min(km_apart / hours_gap, 2000)
    zscore = (amount - typical_amount) / max(typical_amount * 0.4, 1)
    is_odd_hour = 1 <= hour <= 4

    checks = [
        {
            "label": "Travel speed between transactions",
            "value": f"{km_apart:,.0f} km apart, {minutes_since_prev} min apart → {implied_speed:,.0f} km/h implied",
            "flag": implied_speed > 150,
            "detail": ("No real form of transport reaches this speed — this card can't genuinely be "
                       "in both places.") if implied_speed > 150 else
                      "Consistent with normal travel time, or no travel at all.",
        },
        {
            "label": "Recent transaction frequency",
            "value": f"{t1h} transaction(s) in the last hour, {t24h} in the last 24 hours",
            "flag": t1h >= 4,
            "detail": ("Rapid repeated use in a short window — a common pattern when someone is "
                       "testing whether a stolen card still works.") if t1h >= 4 else
                      "Normal transaction frequency for this card.",
        },
        {
            "label": "Amount vs. this card's typical spend",
            "value": f"${amount:,.2f} vs. a usual ${typical_amount:,.2f} ({zscore:+.1f}σ)",
            "flag": abs(zscore) > 3,
            "detail": ("Far outside how this card is normally used — worth a second look.")
                      if abs(zscore) > 3 else "Within this card's normal spending range.",
        },
        {
            "label": "Time of transaction",
            "value": f"{hour:02d}:00 on {day_name}" + (f", ${amount:,.2f}" if is_odd_hour else ""),
            "flag": is_odd_hour and amount > typical_amount * 1.5,
            "detail": ("A larger-than-usual transaction during overnight hours (1am-4am) is an "
                       "uncommon pattern.") if (is_odd_hour and amount > typical_amount * 1.5) else
                      "Normal timing for a transaction.",
        },
    ]

    st.markdown("### Analyzing…")
    placeholder = st.empty()
    revealed = []
    for chk in checks:
        revealed.append(chk)
        html = ""
        for r in revealed:
            cls = "flagged" if r["flag"] else "clear"
            icon = "🚩" if r["flag"] else "✅"
            html += (
                f'<div class="check-row {cls}"><div class="check-icon">{icon}</div>'
                f'<div><div class="check-label">{r["label"]}</div>'
                f'<div class="check-value">{r["value"]}</div>'
                f'<div class="check-detail">{r["detail"]}</div></div></div>'
            )
        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(0.45)

    prob = float(rf_pipe.predict_proba(row)[0, 1])
    is_fraud = prob >= 0.5
    n_flags = sum(1 for c in checks if c["flag"])

    st.markdown("<br>", unsafe_allow_html=True)
    if is_fraud:
        st.markdown(f"""
        <div style="text-align:center; padding:36px 20px; border-radius:16px; background:rgba(198,65,75,0.12); border:1px solid rgba(198,65,75,0.3);">
            <div style="font-size:52px;">🚫</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:#E06670; margin:10px 0 6px;">FLAGGED AS FRAUDULENT</div>
            <div style="color:#ccc; font-size:0.95rem;">{n_flags} of {len(checks)} checks raised a concern · model confidence: {prob*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center; padding:36px 20px; border-radius:16px; background:rgba(47,167,126,0.10); border:1px solid rgba(47,167,126,0.3);">
            <div style="font-size:52px;">✅</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:#4FCB9A; margin:10px 0 6px;">CLEAN — NO FRAUD DETECTED</div>
            <div style="color:#ccc; font-size:0.95rem;">{n_flags} of {len(checks)} checks raised a concern · model confidence: {(1-prob)*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Adjust decision sensitivity"):
        st.caption("The model outputs a fraud probability from 0-100%. By default, anything at or "
                   "above 50% is flagged. A payments team can move this line depending on how they "
                   "want to trade off catching more fraud against inconveniencing genuine customers.")
        threshold = st.slider("Flag as fraud when probability is at least:", 0, 100, 50, format="%d%%")
        adj_verdict = "FRAUDULENT" if prob * 100 >= threshold else "CLEAN"
        st.markdown(f"At a **{threshold}%** threshold, this transaction would be: **{adj_verdict}** "
                    f"(its score is {prob*100:.1f}%)")

st.divider()

# ---------------- Model performance (kept brief) ----------------
with st.expander("How accurate is this, and how does it work under the hood?"):
    rf_m = meta["metrics"]["random_forest"]
    cm = rf_m["confusion_matrix"]
    st.markdown(f"""
Tested on {meta['test_size']:,} transactions it had never seen before, where the true answer was
already known:

- Caught **{cm['tp']} of {cm['tp']+cm['fn']}** real fraud cases ({cm['tp']/(cm['tp']+cm['fn'])*100:.1f}% recall)
- Wrongly flagged **{cm['fp']} of {cm['fp']+cm['tn']:,}** genuine transactions ({cm['tp']/(cm['tp']+cm['fp'])*100:.1f}% precision)
- ROC-AUC {rf_m['roc_auc']:.3f}, PR-AUC {rf_m['pr_auc']:.3f}

**Model:** a Random Forest classifier trained on {meta['n_transactions']:,} synthetic transactions
({meta['fraud_rate']*100:.2f}% fraud rate). It looks at behavioral patterns computed from each
card's own transaction history — not fixed rules — including transaction velocity, time since the
card's last transaction, implied travel speed between consecutive transaction locations, and how
far an amount deviates from that card's own typical spend. The four checks shown during analysis
above are simplified, human-readable versions of the strongest signals the model actually relies on.

Full training code and the dataset are available alongside this app.
    """)

st.caption(
    "⚠️ Unofficial portfolio project — not affiliated with, endorsed by, or built for Zimswitch. "
    "Made independently to demonstrate fraud-detection skills for a job application. All data is "
    "synthetic — no real cardholder, account, or transaction data is used anywhere in this project."
)
