import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import razorpay

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(os.path.join(BASE_DIR, ".env"))

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")


# ============================================================
# RAZORPAY CLIENT
# ============================================================

razorpay_client = razorpay.Client(
    auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RazorGuard AI",
    description="Defense-only AI fraud and payment risk detection API",
    version="1.0.0"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "fraud_model.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "features.pkl"
)


model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)


# ============================================================
# JSON SAFE CONVERTER
# ============================================================

def make_json_safe(value):
    """
    Convert NumPy and nested Python values into
    standard JSON-compatible Python values.
    """

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "name": "RazorGuard AI",
        "status": "online",
        "message": "RazorGuard AI fraud detection API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "online",
        "risk_engine": "online"
    }


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_transaction(transaction):
    """
    Convert a transaction dictionary into the same
    feature structure used during model training.
    """

    df = pd.DataFrame([transaction])

    # Amount relative to customer's normal spending
    df["amount_ratio"] = (
        df["amount"] /
        (df["customer_avg_amount"] + 1)
    )

    # Unusual transaction time
    df["is_night"] = (
        (df["hour"] < 6) |
        (df["hour"] >= 23)
    ).astype(int)

    # High transaction velocity
    df["high_velocity"] = (
        df["transactions_last_1h"] >= 5
    ).astype(int)

    # One-hot encode categorical fields
    df = pd.get_dummies(
        df,
        columns=[
            "payment_method",
            "merchant_category"
        ],
        dtype=int
    )

    # Add any missing training columns
    for column in features:
        if column not in df.columns:
            df[column] = 0

    # Keep exact training feature order
    df = df[features]

    return df


# ============================================================
# PREDICTION
# ============================================================

def predict_transaction(transaction):

    df = prepare_transaction(transaction)

    probability = model.predict_proba(df)[0][1]

    return float(probability)


# ============================================================
# RISK ENGINE
# ============================================================

from backend.risk_engine import analyze_transaction


# ============================================================
# RAZORPAY WEBHOOK
# ============================================================

@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):

    # --------------------------------------------------------
    # Read raw request body
    # --------------------------------------------------------

    body = await request.body()

    signature = request.headers.get(
        "X-Razorpay-Signature"
    )

    if not signature:
        raise HTTPException(
            status_code=400,
            detail="Missing X-Razorpay-Signature header"
        )


    # --------------------------------------------------------
    # Verify Razorpay webhook signature
    # --------------------------------------------------------

    try:

        razorpay_client.utility.verify_webhook_signature(
            body.decode("utf-8"),
            signature,
            RAZORPAY_WEBHOOK_SECRET
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid Razorpay webhook signature"
        )


    # --------------------------------------------------------
    # Parse JSON only AFTER signature verification
    # --------------------------------------------------------

    try:

        payload = json.loads(
            body.decode("utf-8")
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload"
        )


    # --------------------------------------------------------
    # Extract event
    # --------------------------------------------------------

    event = payload.get(
        "event",
        "unknown"
    )


    # --------------------------------------------------------
    # Extract payment entity
    # --------------------------------------------------------

    try:

        payment = (
            payload
            ["payload"]
            ["payment"]
            ["entity"]
        )

    except KeyError:

        raise HTTPException(
            status_code=400,
            detail="Payment entity not found in webhook"
        )


    # --------------------------------------------------------
    # Payment information
    # --------------------------------------------------------

    payment_id = payment.get(
        "id",
        "unknown"
    )

    amount_paise = payment.get(
        "amount",
        0
    )

    currency = payment.get(
        "currency",
        "INR"
    )

    payment_method = payment.get(
        "method",
        "unknown"
    )


    # Razorpay sends amount in paise
    amount_rupees = float(amount_paise) / 100


    # --------------------------------------------------------
    # Behavioural baseline
    # --------------------------------------------------------
    #
    # A basic Razorpay payment webhook does not contain all
    # behavioural features used by our ML model.
    #
    # Therefore we use conservative baseline values here.
    # In a production integration these can be replaced with
    # merchant/customer/device behavioural data.
    # --------------------------------------------------------

    transaction = {

        "transaction_id": str(payment_id),

        "amount": amount_rupees,

        "customer_age": 30,

        "customer_transaction_count": 10,

        "customer_avg_amount": max(
            amount_rupees,
            1000
        ),

        "device_age_days": 180,

        "new_device": 0,

        "failed_attempts": 0,

        "transactions_last_1h": 1,

        "transactions_last_24h": 3,

        "location_changed": 0,

        "hour": 12,

        "payment_method": payment_method,

        "merchant_category": "general"
    }


    # --------------------------------------------------------
    # Run RazorGuard AI
    # --------------------------------------------------------

    try:

        fraud_probability = predict_transaction(
            transaction
        )

        result = analyze_transaction(
            transaction,
            fraud_probability
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Risk analysis failed: {str(error)}"
        )


    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    response_data = {

        "status": "processed",

        "event": str(event),

        "payment_id": str(payment_id),

        "currency": str(currency),

        "amount": float(amount_rupees),

        "fraud_probability": float(
            result["fraud_probability"]
        ),

        "risk_score": int(
            result["risk_score"]
        ),

        "risk_level": str(
            result["risk_level"]
        ),

        "signals": result["signals"],

        "recommended_action": str(
            result["recommended_action"]
        )
    }


    # --------------------------------------------------------
    # Convert EVERY nested NumPy value
    # --------------------------------------------------------

    return make_json_safe(
        response_data
    )