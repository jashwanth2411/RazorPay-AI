import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.investigator import (
    generate_investigation_report,
    print_investigation_report,
    ai_investigate,
    print_ai_investigation
)
import pandas as pd
import joblib

from risk_engine import analyze_transaction


# ============================================================
# RAZORGUARD AI
# REAL TRANSACTION PREDICTION
# ============================================================


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

model = joblib.load(
    "model/fraud_model.pkl"
)

features = joblib.load(
    "model/features.pkl"
)


# ------------------------------------------------------------
# CREATE TRANSACTION
# ------------------------------------------------------------

transaction = {

    "transaction_id": "TXN-DEMO-001",

    "amount": 25000,

    "customer_age": 28,

    "customer_transaction_count": 42,

    "customer_avg_amount": 1200,

    "device_age_days": 2,

    "new_device": 1,

    "failed_attempts": 4,

    "transactions_last_1h": 7,

    "transactions_last_24h": 18,

    "location_changed": 1,

    "hour": 2,

    "payment_method": "upi",

    "merchant_category": "electronics"
}


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

transaction["amount_ratio"] = (
    transaction["amount"] /
    (transaction["customer_avg_amount"] + 1)
)

transaction["is_night"] = int(
    transaction["hour"] <= 4
    or
    transaction["hour"] >= 23
)

transaction["high_velocity"] = int(
    transaction["transactions_last_1h"] >= 5
    or
    transaction["transactions_last_24h"] >= 20
)


# ------------------------------------------------------------
# CONVERT TO DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(
    [transaction]
)


# ------------------------------------------------------------
# ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------

df = pd.get_dummies(
    df,
    columns=[
        "payment_method",
        "merchant_category"
    ],
    dtype=int
)


# ------------------------------------------------------------
# MAKE SURE ALL MODEL FEATURES EXIST
# ------------------------------------------------------------

for feature in features:

    if feature not in df.columns:

        df[feature] = 0


# Keep exactly the same order as training.

X = df[features]


# ------------------------------------------------------------
# MODEL PREDICTION
# ------------------------------------------------------------

fraud_probability = model.predict_proba(
    X
)[0][1]


# ------------------------------------------------------------
# RISK ENGINE
# ------------------------------------------------------------

result = analyze_transaction(
    transaction,
    fraud_probability
)
report = generate_investigation_report(
    transaction=transaction,
    fraud_probability=result["fraud_probability"],
    risk_score=result["risk_score"],
    risk_level=result["risk_level"],
    signals=result["signals"],
    recommended_action=result["recommended_action"]
)

print_investigation_report(report)


ai_report = ai_investigate(report)

print_ai_investigation(ai_report)
# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 65)
print("RAZORGUARD AI - LIVE TRANSACTION ANALYSIS")
print("=" * 65)

print()

print(
    "Transaction ID:",
    transaction["transaction_id"]
)

print(
    f"Transaction Amount: "
    f"₹{transaction['amount']:,.2f}"
)

print()

print(
    f"Fraud Probability: "
    f"{result['fraud_probability'] * 100:.2f}%"
)

print(
    f"Risk Score: "
    f"{result['risk_score']}/100"
)

print(
    f"Risk Level: "
    f"{result['risk_level']}"
)

print()

print("Risk Signals:")

for signal in result["signals"]:

    print(
        f"  ⚠ {signal['message']}"
    )

print()

print(
    "Recommended Action:"
)

print(
    f"  {result['recommended_action']}"
)

print()

print("=" * 65)