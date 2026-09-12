"""
Zimswitch Transaction Analyzer — Streamlit frontend
=================================================================
This app has no model in it. Every prediction is a live HTTP call to
the FastAPI service in ../api (deployed separately, e.g. on Render).
That split — a model-serving API plus a separate UI that consumes it —
mirrors how a real fraud detection system is built: the model can be
scaled, monitored, versioned, and reused by other clients independently
of any one interface.

Run locally (with the API also running locally on port 8000):
    streamlit run streamlit_frontend.py

Point at a deployed API by setting the API_URL in the sidebar, or via
the API_URL environment variable / Streamlit secrets before deploying.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MCC_LABELS = {
    "5411": "Groceries", "5541": "Fuel", "5812": "Restaurant", "5912": "Pharmacy",
    "6011": "ATM / cash withdrawal", "5651": "Clothing", "4900": "Utilities",
    "5999": "Retail — misc", "6051": "Money transfer", "7011": "Hotel",
    "5311": "Department store", "4814": "Telecom / airtime",
}
CITIES = ["Harare", "Bulawayo", "Mutare", "Gweru", "Kwekwe", "Masvingo", "Chinhoyi", "Victoria Falls"]

DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Zimswitch Transaction Analyzer", page_icon="🔍", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.check-row {
    display:flex; align-items:flex-start; gap:12px; padding:12px 14px; border-radius:10px;
    margin-bottom:8px; background:rgba(120,120,120,0.06); border-left:3px solid transparent;
}
.check-row.flagged { border-left-color:#D6572C; background:rgba(214,87,44,0.08); }
.check-row.clear { border-left-color:#2FA77E; background:rgba(47,167,126,0.06); }
.check-label { font-weight:600; font-size:0.92rem; }
.check-value { font-family:'IBM Plex Mono', monospace; font-size:0.82rem; color:#999; margin-top:1px;}
.check-detail { font-size:0.85rem; color:#aaa; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### API connection")
    api_url = st.text_input("Fraud detection API URL", value=DEFAULT_API_URL,
                             help="Deployed FastAPI service. Defaults to localhost for local testing.")
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        if health.ok and health.json().get("status") == "ok":
            st.success("Connected — model loaded")
        else:
            st.warning("API reachable but model still loading")
    except Exception:
        st.error("Can't reach the API. Is it running / deployed?")

st.markdown("# 🔍 Zimswitch Transaction Analyzer")
st.caption("Check one transaction by hand, or upload a file of transactions and let the API "
           "score all of them — every prediction is a live call to the fraud-detection model "
           "running as its own backend service.")

tab_single, tab_bulk = st.tabs(["Check one transaction", "Upload a file of transactions"])

# ============================================================
# TAB 1 — single transaction (unchanged behavior)
# ============================================================
with tab_single:
    st.markdown("### This transaction")
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount", min_value=0.01, value=50.0, step=1.0, key="s_amount")
        currency = st.selectbox("Currency", ["USD", "ZWG"], key="s_currency")
        channel = st.selectbox("Channel", ["POS", "ATM", "ZIPIT", "Mobile Money", "Internet Banking"], key="s_channel")
    with c2:
        mcc = st.selectbox("Type of purchase", list(MCC_LABELS.keys()), format_func=lambda x: MCC_LABELS[x], key="s_mcc")
        city_now = st.selectbox("City where this is happening", CITIES, key="s_city_now")
        hour = st.slider("Hour of day", 0, 23, 14, key="s_hour")

    day_name = st.select_slider("Day of week", options=DAYS, value="Wednesday", key="s_day")

    st.markdown("### This card's recent activity")
    c3, c4 = st.columns(2)
    with c3:
        city_prev = st.selectbox("City of this card's previous transaction", CITIES,
                                  index=CITIES.index(city_now), key="s_city_prev")
        minutes_since_prev = st.number_input("Minutes since that previous transaction", min_value=1, value=120, key="s_gap")
        typical_amount = st.number_input("What does this card usually spend per transaction?",
                                          min_value=1.0, value=50.0, step=1.0, key="s_typical")
    with c4:
        t1h = st.number_input("Transactions on this card in the last hour", min_value=0, value=0, key="s_t1h")
        t24h = st.number_input("Transactions on this card in the last 24 hours", min_value=0, value=1, key="s_t24h")

    analyze = st.button("Analyze this transaction", type="primary", use_container_width=True, key="s_analyze")
    st.divider()

    if analyze:
        payload = {
            "amount": amount, "currency": currency, "channel": channel, "mcc": mcc,
            "city_now": city_now, "city_prev": city_prev, "hour": hour,
            "day_of_week": DAYS.index(day_name), "minutes_since_prev": minutes_since_prev,
            "txn_count_1h": t1h, "txn_count_24h": t24h, "typical_amount": typical_amount,
        }

        try:
            with st.spinner("Sending transaction to the fraud detection API…"):
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=15)
            if not resp.ok:
                st.error(f"API returned an error ({resp.status_code}): {resp.text}")
                st.stop()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the fraud detection API at `{api_url}`. "
                     f"Is it running and deployed? ({e})")
            st.stop()

        st.markdown("### Analyzing…")
        placeholder = st.empty()
        revealed = []
        for chk in result["checks"]:
            revealed.append(chk)
            html = ""
            for r in revealed:
                cls = "flagged" if r["flag"] else "clear"
                icon = "🚩" if r["flag"] else "✅"
                html += (
                    f'<div class="check-row {cls}"><div>{icon}</div>'
                    f'<div><div class="check-label">{r["label"]}</div>'
                    f'<div class="check-value">{r["value"]}</div>'
                    f'<div class="check-detail">{r["detail"]}</div></div></div>'
                )
            placeholder.markdown(html, unsafe_allow_html=True)
            time.sleep(0.4)

        prob = result["probability"]
        n_flags = sum(1 for c in result["checks"] if c["flag"])

        st.markdown("<br>", unsafe_allow_html=True)
        if result["is_fraud"]:
            st.markdown(f"""
            <div style="text-align:center; padding:36px 20px; border-radius:16px; background:rgba(198,65,75,0.12); border:1px solid rgba(198,65,75,0.3);">
                <div style="font-size:52px;">🚫</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:#E06670; margin:10px 0 6px;">FLAGGED AS FRAUDULENT</div>
                <div style="color:#ccc; font-size:0.95rem;">{n_flags} of {len(result['checks'])} checks raised a concern · model confidence: {prob*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center; padding:36px 20px; border-radius:16px; background:rgba(47,167,126,0.10); border:1px solid rgba(47,167,126,0.3);">
                <div style="font-size:52px;">✅</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:#4FCB9A; margin:10px 0 6px;">CLEAN — NO FRAUD DETECTED</div>
                <div style="color:#ccc; font-size:0.95rem;">{n_flags} of {len(result['checks'])} checks raised a concern · model confidence: {(1-prob)*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"Response from the fraud detection API at `{api_url}` — "
                   f"this transaction scored {prob*100:.1f}% probability of fraud.")

# ============================================================
# TAB 2 — bulk upload
# ============================================================
with tab_bulk:
    st.markdown("### Upload a file of transactions")
    st.caption(
        "Upload a CSV and every row gets scored. Cards with multiple transactions in the file "
        "get richer results — the model computes each card's spending pattern and transaction "
        "velocity directly from the history in your file, the same way it was trained."
    )

    required_cols = ["card_id", "timestamp", "amount", "currency", "channel", "mcc", "latitude", "longitude"]
    st.markdown("**Required columns:** `" + "`, `".join(required_cols) + "`  \n"
                "*(an optional `transaction_id` column is carried through to the results)*")

    try:
        with open(Path(__file__).parent / "sample_template.csv", "rb") as f:
            st.download_button("Download a sample template", f, file_name="sample_transactions.csv",
                                mime="text/csv", use_container_width=False)
    except FileNotFoundError:
        pass

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    run_batch = st.button("Analyze uploaded transactions", type="primary",
                           use_container_width=True, disabled=uploaded is None)

    if run_batch and uploaded is not None:
        try:
            with st.spinner("Uploading and scoring every transaction…"):
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                resp = requests.post(f"{api_url}/predict_batch", files=files, timeout=60)
            if not resp.ok:
                st.error(f"API returned an error ({resp.status_code}): {resp.json().get('detail', resp.text)}")
                st.stop()
            batch = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the fraud detection API at `{api_url}`. Is it running and deployed? ({e})")
            st.stop()

        results_df = pd.DataFrame(batch["results"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Transactions scored", batch["n_transactions"])
        m2.metric("Flagged as fraud", batch["n_flagged"])
        m3.metric("Flagged rate", f"{batch['flagged_rate']*100:.1f}%")

        st.markdown("#### Results")

        def highlight_verdict(val):
            if val == "FRAUDULENT":
                return "background-color: rgba(198,65,75,0.18); color:#E06670; font-weight:600;"
            return "background-color: rgba(47,167,126,0.10); color:#4FCB9A;"

        styled = results_df.style.applymap(highlight_verdict, subset=["verdict"]) \
            .format({"fraud_probability": "{:.1%}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv_bytes, file_name="fraud_results.csv",
                            mime="text/csv", use_container_width=True)

st.divider()
st.caption(
    "⚠️ Unofficial portfolio project — not affiliated with, endorsed by, or built for Zimswitch. "
    "Made independently to demonstrate fraud-detection skills for a job application. All data is "
    "synthetic — no real cardholder, account, or transaction data is used anywhere in this project."
)
