"""
Synthetic Payment Switch Transaction Data Generator
====================================================
Generates realistic transaction data modeled on how a national payment
switch (interbank/interoperability hub) operates - similar in structure
to Zimswitch: POS, ATM, ZIPIT-style transfers, and mobile money
interoperability, across multiple issuing/acquiring banks.

This is 100% synthetic - no real cardholder, account, or transaction
data of any kind is used or referenced.

Fraud patterns injected (labelled y=1):
  1. Card testing        - many small, rapid transactions on one card
  2. Velocity abuse       - unusually high transaction frequency in a
                            short window
  3. Amount anomaly       - transaction far above the card's normal
                            spending profile
  4. Geographic jump      - transactions in different locations too
                            close together in time to be physically
                            possible
  5. Odd-hour high-value  - large transactions at unusual hours
                            (01:00-04:00) relative to the cardholder's
                            normal pattern
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_CARDS = 4000
N_TRANSACTIONS = 60000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)

ISSUERS = ["CBZ Bank", "Stanbic Bank", "FBC Bank", "Steward Bank",
           "ZB Bank", "NMB Bank", "CABS", "Ecobank", "BancABC", "POSB"]

CHANNELS = ["POS", "ATM", "ZIPIT", "Mobile Money", "Internet Banking"]
CHANNEL_WEIGHTS = [0.42, 0.18, 0.20, 0.15, 0.05]

MCC_CODES = {
    "5411": "Grocery Stores", "5541": "Fuel Stations", "5812": "Restaurants",
    "5912": "Pharmacies", "6011": "ATM/Cash Withdrawal", "5651": "Clothing",
    "4900": "Utilities", "5999": "Retail - Misc", "6051": "Money Transfer",
    "7011": "Hotels", "5311": "Department Stores", "4814": "Telecom/Airtime",
}
MCC_LIST = list(MCC_CODES.keys())

CITIES = [
    ("Harare", -17.8292, 31.0522), ("Bulawayo", -20.1500, 28.5833),
    ("Mutare", -18.9707, 32.6709), ("Gweru", -19.4500, 29.8167),
    ("Kwekwe", -18.9281, 29.8149), ("Masvingo", -20.0637, 30.8277),
    ("Chinhoyi", -17.3667, 30.2000), ("Victoria Falls", -17.9243, 25.8567),
]

CURRENCIES = ["ZWG", "USD"]
CURRENCY_WEIGHTS = [0.55, 0.45]


def random_timestamp():
    delta = END_DATE - START_DATE
    seconds = RNG.integers(0, int(delta.total_seconds()))
    return START_DATE + timedelta(seconds=int(seconds))


def make_cards(n):
    cards = []
    for i in range(n):
        home_city = CITIES[RNG.integers(0, len(CITIES))]
        avg_amount = RNG.gamma(shape=2.0, scale=35) + 5   # typical spend profile
        cards.append({
            "card_id": f"CARD{i:06d}",
            "issuer": ISSUERS[RNG.integers(0, len(ISSUERS))],
            "home_city": home_city[0],
            "home_lat": home_city[1],
            "home_lon": home_city[2],
            "avg_amount": avg_amount,
            "preferred_currency": RNG.choice(CURRENCIES, p=CURRENCY_WEIGHTS),
        })
    return pd.DataFrame(cards)


def base_transactions(cards_df, n):
    rows = []
    card_idx = RNG.integers(0, len(cards_df), size=n)
    for i in range(n):
        card = cards_df.iloc[card_idx[i]]
        ts = random_timestamp()
        channel = RNG.choice(CHANNELS, p=CHANNEL_WEIGHTS)
        mcc = RNG.choice(MCC_LIST) if channel == "POS" else (
            "6011" if channel == "ATM" else "6051")
        amount = max(1.0, RNG.normal(loc=card["avg_amount"], scale=card["avg_amount"] * 0.4))
        acquirer = ISSUERS[RNG.integers(0, len(ISSUERS))]
        rows.append({
            "transaction_id": f"TXN{i:07d}",
            "card_id": card["card_id"],
            "issuer_bank": card["issuer"],
            "acquirer_bank": acquirer,
            "channel": channel,
            "mcc": mcc,
            "mcc_description": MCC_CODES[mcc],
            "amount": round(amount, 2),
            "currency": card["preferred_currency"],
            "timestamp": ts,
            "city": card["home_city"],
            "latitude": card["home_lat"] + RNG.normal(0, 0.02),
            "longitude": card["home_lon"] + RNG.normal(0, 0.02),
            "is_fraud": 0,
            "fraud_pattern": "none",
        })
    return pd.DataFrame(rows)


def inject_card_testing(df, cards_df, n_cards_affected=60):
    """Many small rapid transactions on the same card in a short window."""
    new_rows = []
    victim_cards = cards_df.sample(n=n_cards_affected, random_state=1)
    txn_counter = 900000
    for _, card in victim_cards.iterrows():
        burst_start = random_timestamp()
        n_burst = RNG.integers(6, 15)
        for j in range(n_burst):
            ts = burst_start + timedelta(seconds=int(RNG.integers(5, 90) * (j + 1)))
            new_rows.append({
                "transaction_id": f"TXN{txn_counter:07d}",
                "card_id": card["card_id"],
                "issuer_bank": card["issuer"],
                "acquirer_bank": ISSUERS[RNG.integers(0, len(ISSUERS))],
                "channel": "Internet Banking",
                "mcc": "5999",
                "mcc_description": MCC_CODES["5999"],
                "amount": round(RNG.uniform(0.5, 3.0), 2),
                "currency": card["preferred_currency"],
                "timestamp": ts,
                "city": card["home_city"],
                "latitude": card["home_lat"] + RNG.normal(0, 0.02),
                "longitude": card["home_lon"] + RNG.normal(0, 0.02),
                "is_fraud": 1,
                "fraud_pattern": "card_testing",
            })
            txn_counter += 1
    return pd.DataFrame(new_rows)


def inject_velocity_abuse(df, cards_df, n_cards_affected=50):
    """Unusually high transaction frequency in a short window, larger amounts."""
    new_rows = []
    victim_cards = cards_df.sample(n=n_cards_affected, random_state=2)
    txn_counter = 910000
    for _, card in victim_cards.iterrows():
        burst_start = random_timestamp()
        n_burst = RNG.integers(5, 10)
        for j in range(n_burst):
            ts = burst_start + timedelta(minutes=int(RNG.integers(1, 20) * (j + 1)))
            new_rows.append({
                "transaction_id": f"TXN{txn_counter:07d}",
                "card_id": card["card_id"],
                "issuer_bank": card["issuer"],
                "acquirer_bank": ISSUERS[RNG.integers(0, len(ISSUERS))],
                "channel": RNG.choice(["POS", "ATM"]),
                "mcc": RNG.choice(MCC_LIST),
                "mcc_description": "Velocity burst",
                "amount": round(card["avg_amount"] * RNG.uniform(1.5, 3.0), 2),
                "currency": card["preferred_currency"],
                "timestamp": ts,
                "city": card["home_city"],
                "latitude": card["home_lat"] + RNG.normal(0, 0.05),
                "longitude": card["home_lon"] + RNG.normal(0, 0.05),
                "is_fraud": 1,
                "fraud_pattern": "velocity_abuse",
            })
            txn_counter += 1
    return pd.DataFrame(new_rows)


def inject_amount_anomaly(cards_df, n_affected=80):
    """Single transaction far above the card's normal spending profile."""
    new_rows = []
    victim_cards = cards_df.sample(n=n_affected, random_state=3)
    txn_counter = 920000
    for _, card in victim_cards.iterrows():
        ts = random_timestamp()
        new_rows.append({
            "transaction_id": f"TXN{txn_counter:07d}",
            "card_id": card["card_id"],
            "issuer_bank": card["issuer"],
            "acquirer_bank": ISSUERS[RNG.integers(0, len(ISSUERS))],
            "channel": RNG.choice(["POS", "Internet Banking", "ZIPIT"]),
            "mcc": RNG.choice(MCC_LIST),
            "mcc_description": "Anomalous high value",
            "amount": round(card["avg_amount"] * RNG.uniform(8, 20), 2),
            "currency": card["preferred_currency"],
            "timestamp": ts,
            "city": card["home_city"],
            "latitude": card["home_lat"] + RNG.normal(0, 0.02),
            "longitude": card["home_lon"] + RNG.normal(0, 0.02),
            "is_fraud": 1,
            "fraud_pattern": "amount_anomaly",
        })
        txn_counter += 1
    return pd.DataFrame(new_rows)


def inject_geo_jump(cards_df, n_affected=70):
    """Two transactions in different, distant cities too close in time."""
    new_rows = []
    victim_cards = cards_df.sample(n=n_affected, random_state=4)
    txn_counter = 930000
    for _, card in victim_cards.iterrows():
        ts1 = random_timestamp()
        ts2 = ts1 + timedelta(minutes=int(RNG.integers(10, 45)))
        far_city = CITIES[RNG.integers(0, len(CITIES))]
        for ts, (city, lat, lon) in [(ts1, (card["home_city"], card["home_lat"], card["home_lon"])),
                                      (ts2, far_city)]:
            new_rows.append({
                "transaction_id": f"TXN{txn_counter:07d}",
                "card_id": card["card_id"],
                "issuer_bank": card["issuer"],
                "acquirer_bank": ISSUERS[RNG.integers(0, len(ISSUERS))],
                "channel": "POS",
                "mcc": RNG.choice(MCC_LIST),
                "mcc_description": "Geo jump",
                "amount": round(card["avg_amount"] * RNG.uniform(1.0, 2.5), 2),
                "currency": card["preferred_currency"],
                "timestamp": ts,
                "city": city,
                "latitude": lat + RNG.normal(0, 0.02),
                "longitude": lon + RNG.normal(0, 0.02),
                "is_fraud": 1,
                "fraud_pattern": "geo_jump",
            })
            txn_counter += 1
    return pd.DataFrame(new_rows)


def inject_odd_hour_high_value(cards_df, n_affected=60):
    """Large transaction at an unusual hour (01:00-04:00)."""
    new_rows = []
    victim_cards = cards_df.sample(n=n_affected, random_state=5)
    txn_counter = 940000
    for _, card in victim_cards.iterrows():
        day_offset = RNG.integers(0, (END_DATE - START_DATE).days)
        odd_hour = RNG.integers(1, 4)
        ts = START_DATE + timedelta(days=int(day_offset), hours=int(odd_hour),
                                     minutes=int(RNG.integers(0, 59)))
        new_rows.append({
            "transaction_id": f"TXN{txn_counter:07d}",
            "card_id": card["card_id"],
            "issuer_bank": card["issuer"],
            "acquirer_bank": ISSUERS[RNG.integers(0, len(ISSUERS))],
            "channel": RNG.choice(["ATM", "Internet Banking", "Mobile Money"]),
            "mcc": "6011",
            "mcc_description": "Odd-hour high value",
            "amount": round(card["avg_amount"] * RNG.uniform(4, 9), 2),
            "currency": card["preferred_currency"],
            "timestamp": ts,
            "city": card["home_city"],
            "latitude": card["home_lat"] + RNG.normal(0, 0.02),
            "longitude": card["home_lon"] + RNG.normal(0, 0.02),
            "is_fraud": 1,
            "fraud_pattern": "odd_hour_high_value",
        })
        txn_counter += 1
    return pd.DataFrame(new_rows)


def main():
    cards_df = make_cards(N_CARDS)
    legit = base_transactions(cards_df, N_TRANSACTIONS)

    fraud_parts = [
        inject_card_testing(legit, cards_df),
        inject_velocity_abuse(legit, cards_df),
        inject_amount_anomaly(cards_df),
        inject_geo_jump(cards_df),
        inject_odd_hour_high_value(cards_df),
    ]

    full = pd.concat([legit] + fraud_parts, ignore_index=True)
    full = full.sort_values("timestamp").reset_index(drop=True)

    fraud_rate = full["is_fraud"].mean()
    print(f"Total transactions: {len(full)}")
    print(f"Fraud transactions: {full['is_fraud'].sum()} ({fraud_rate:.3%})")
    print(full["fraud_pattern"].value_counts())

    full.to_csv("/home/claude/zimswitch_fraud/transactions.csv", index=False)
    cards_df.to_csv("/home/claude/zimswitch_fraud/cards.csv", index=False)
    print("\nSaved transactions.csv and cards.csv")


if __name__ == "__main__":
    main()
