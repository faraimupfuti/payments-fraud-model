"""
Zimswitch Fraud Console — plain-language explainer (Streamlit)
=================================================================
The non-technical companion to streamlit_app.py. No sliders, no
statistics up front — three short stories, then a "try it yourself"
section with plain-English buttons. The verdicts are genuinely computed
by the same trained Random Forest pipeline used elsewhere in this
project; only the plain-language description of each scenario is
hand-written for clarity.

Run:      streamlit run explainer_app.py
Deploy:   same as streamlit_app.py — push to GitHub, deploy free at
          https://share.streamlit.io
"""

import json
from pathlib import Path

import joblib
import streamlit as st

from features import build_single_row

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Zimswitch Fraud Console — how it works", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_model():
    rf = joblib.load(MODELS_DIR / "rf_pipeline.joblib")
    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)
    return rf, meta


rf_pipe, meta = load_model()

# ---------- scenarios: inputs feed the real model; text is hand-written for clarity ----------
SCENARIOS = {
    "A normal purchase": dict(
        icon="✅",
        reason="Same city the cardholder always shops in, a typical amount, no unusual activity nearby. "
               "This is what happens to 98% of transactions — nothing to flag.",
        inputs=dict(amount=38, hour=14, dow=2, gap=240, dist=1.5, t1h=0, t24h=1,
                    channel="POS", mcc="5411", currency="USD", avg=42, std=17),
    ),
    "Someone testing a stolen card": dict(
        icon="🚨",
        reason="9 tiny transactions on the same card within a minute. Real shoppers don't behave this way — "
               "this is the signature of someone probing whether a stolen card number still works.",
        inputs=dict(amount=2, hour=9, dow=3, gap=1, dist=0, t1h=9, t24h=11,
                    channel="Internet Banking", mcc="5999", currency="USD", avg=48, std=18),
    ),
    "Used in two cities at once": dict(
        icon="🚨",
        reason="The same card was used in two cities 480km apart, 22 minutes apart. No one can travel that "
               "fast — one of these transactions isn't genuine.",
        inputs=dict(amount=64, hour=19, dow=5, gap=22, dist=480, t1h=1, t24h=2,
                    channel="POS", mcc="5812", currency="USD", avg=51, std=20),
    ),
    "A sudden spending spree": dict(
        icon="⚠️",
        reason="6 transactions in under an hour, each bigger than this card's usual spending. Not impossible "
               "on its own, but unusual enough to have a human take a second look.",
        inputs=dict(amount=130, hour=11, dow=4, gap=6, dist=3, t1h=6, t24h=9,
                    channel="ATM", mcc="6011", currency="USD", avg=52, std=19),
    ),
    "One unusually large transfer": dict(
        icon="⚠️",
        reason="A single transaction much larger than this card typically spends. Could be completely "
               "legitimate — a big purchase does happen — so it's held for a quick check rather than "
               "blocked outright.",
        inputs=dict(amount=900, hour=15, dow=1, gap=300, dist=2, t1h=0, t24h=1,
                    channel="Internet Banking", mcc="5999", currency="USD", avg=45, std=12),
    ),
    "A large cash withdrawal at 3am": dict(
        icon="⚠️",
        reason="A large ATM withdrawal in the middle of the night is uncommon for this cardholder's usual "
               "pattern — worth a second look before it goes through.",
        inputs=dict(amount=410, hour=3, dow=6, gap=200, dist=5, t1h=0, t24h=1,
                    channel="ATM", mcc="6011", currency="USD", avg=49, std=18),
    ),
}

VERDICT_STYLE = {
    "approved": dict(label="Approved instantly", color="#2FA77E", bg="#173A2E22"),
    "review":   dict(label="Sent for review",    color="#D6572C", bg="#3A231722"),
    "blocked":  dict(label="Blocked",            color="#C6414B", bg="#3A1A1D22"),
}


def classify(prob):
    if prob < 0.15:
        return "approved"
    elif prob < 0.75:
        return "review"
    else:
        return "blocked"


def score(inputs):
    row = build_single_row(
        amount=inputs["amount"], hour=inputs["hour"], day_of_week=inputs["dow"],
        minutes_since_prev=inputs["gap"], km_from_prev=inputs["dist"],
        txn_count_1h=inputs["t1h"], txn_count_24h=inputs["t24h"],
        channel=inputs["channel"], mcc=inputs["mcc"], currency=inputs["currency"],
        avg_amount=inputs["avg"], std_amount=inputs["std"],
    )
    return float(rf_pipe.predict_proba(row)[0, 1])


# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&display=swap');
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.story-card { background: rgba(120,120,120,0.06); border-radius: 14px; padding: 20px 24px; margin-bottom: 16px; }
.story-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("# 🛡️ How this system catches fraud")
st.caption("No dashboards or jargon — just real situations it handles automatically, every day, "
           "in a fraction of a second.")

st.divider()

# ---------- Three fixed stories ----------
stories = [
    ("Situation 1", "Someone tries a stolen card number",
     "A criminal doesn't know if a stolen card is still active, so they run a string of tiny "
     "purchases — a few cents each — in under a minute to test it before making a real charge. "
     "The system notices this card has been used 9 times in the last 60 seconds, for amounts no "
     "real shopper would bother with.", "🚫 Flagged and blocked"),
    ("Situation 2", "A card is used in two cities at once",
     "A card is swiped in Harare. Twenty-two minutes later, the same card is swiped in Bulawayo — "
     "440 km away. No form of travel gets a person there that fast, so the system flags one of "
     "these transactions as impossible.", "🚫 Flagged and blocked"),
    ("Situation 3", "You buy your usual groceries",
     "Same city, normal time of day, an amount close to what this card always spends, and no other "
     "unusual activity around it. Nothing here looks out of place.", "✅ Approved instantly"),
]
for eyebrow, title, body, verdict in stories:
    st.markdown(f"""
    <div class="story-card">
        <div style="font-size:0.8rem; color:#888;">{eyebrow}</div>
        <div class="story-title">{title}</div>
        <div style="color:#aaa; font-size:0.92rem;">{body}</div>
        <div style="margin-top:10px; font-weight:600;">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------- Try it yourself ----------
st.markdown("## Try it yourself")
st.caption("Pick a situation and see what the system decides — and why, in plain terms. "
           "This uses the real trained model, live.")

if "chosen" not in st.session_state:
    st.session_state.chosen = "A normal purchase"

cols = st.columns(3)
names = list(SCENARIOS.keys())
for i, name in enumerate(names):
    with cols[i % 3]:
        if st.button(name, use_container_width=True):
            st.session_state.chosen = name

chosen = SCENARIOS[st.session_state.chosen]
prob = score(chosen["inputs"])
bucket = classify(prob)
style = VERDICT_STYLE[bucket]

st.markdown(f"""
<div style="text-align:center; padding:34px 20px; border-radius:14px; background:{style['bg']}; margin-top:10px;">
    <div style="font-size:46px;">{chosen['icon']}</div>
    <div style="font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; color:{style['color']}; margin:8px 0 14px;">
        {style['label']}
    </div>
    <div style="max-width:480px; margin:0 auto; color:#ccc; font-size:0.95rem;">{chosen['reason']}</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------- Does it work? ----------
st.markdown("## Does it actually work?")
st.caption("Tested against 15,313 transactions it had never seen before, where the true answer was already known.")

rf_m = meta["metrics"]["random_forest"]
cm = rf_m["confusion_matrix"]
c1, c2, c3 = st.columns(3)
c1.metric("Real fraud cases caught", f"{cm['tp']} of {cm['tp']+cm['fn']}")
c2.metric("Genuine purchases wrongly flagged", f"{cm['fp']} of {cm['fp']+cm['tn']:,}")
c3.metric("Time to score a transaction", "< 1 sec")

with st.expander("For the technically curious — model details"):
    lr_m = meta["metrics"]["logistic_regression"]
    st.markdown(f"""
A **Random Forest classifier** trained on {meta['n_transactions']:,} synthetic transactions
({meta['fraud_rate']*100:.2f}% fraud rate), using behavioral features computed causally per
card — transaction velocity, time-since-last-transaction, implied travel speed between
consecutive locations, and deviation from the card's own historical spend. Evaluated on a
stratified 25% held-out test split.

- **ROC-AUC:** {rf_m['roc_auc']:.3f} · **PR-AUC:** {rf_m['pr_auc']:.3f}
- **Recall:** {cm['tp']/(cm['tp']+cm['fn'])*100:.1f}% · **Precision:** {cm['tp']/(cm['tp']+cm['fp'])*100:.1f}%
- Compared against a Logistic Regression baseline (ROC-AUC {lr_m['roc_auc']:.3f}), used elsewhere
  in this project for transparent, per-transaction explanations

Full code, the training pipeline, and the dataset are available alongside this app.
    """)

st.divider()
st.caption(
    "⚠️ Unofficial portfolio project — not affiliated with, endorsed by, or built for Zimswitch. "
    "Made independently to demonstrate fraud-detection skills for a job application. All data is "
    "synthetic — no real cardholder, account, or transaction data is used anywhere in this project."
)
