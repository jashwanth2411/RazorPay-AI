import os
import json
import hmac
import hashlib

import requests
from dotenv import load_dotenv


# Load .env
load_dotenv()


# Get webhook secret without printing it
secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if not secret:
    raise RuntimeError(
        "RAZORPAY_WEBHOOK_SECRET is missing from .env"
    )


# ---------------------------------------------------------
# Fake Razorpay webhook payload
# ---------------------------------------------------------

payload = {
    "entity": "event",
    "account_id": "acc_TEST_RAZORGUARD",
    "event": "payment.captured",
    "contains": [
        "payment"
    ],
    "payload": {
        "payment": {
            "entity": {
                "id": "pay_RAZORGUARD_TEST_001",
                "entity": "payment",
                "amount": 250000,
                "currency": "INR",
                "status": "captured",
                "method": "upi"
            }
        }
    }
}


# ---------------------------------------------------------
# IMPORTANT:
# Serialize ONCE and sign EXACTLY these bytes.
# ---------------------------------------------------------

body = json.dumps(
    payload,
    separators=(",", ":")
)


# ---------------------------------------------------------
# Generate Razorpay-style HMAC SHA256 signature
# ---------------------------------------------------------

signature = hmac.new(
    secret.encode("utf-8"),
    body.encode("utf-8"),
    hashlib.sha256
).hexdigest()


# ---------------------------------------------------------
# Send webhook to local RazorGuard API
# ---------------------------------------------------------

url = "http://127.0.0.1:8000/webhook/razorpay"

headers = {
    "Content-Type": "application/json",
    "X-Razorpay-Signature": signature
}


response = requests.post(
    url,
    data=body.encode("utf-8"),
    headers=headers,
    timeout=10
)


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RAZORGUARD WEBHOOK TEST")
print("=" * 60)

print("\nHTTP Status:", response.status_code)

print("\nResponse:")

try:
    print(
        json.dumps(
            response.json(),
            indent=2
        )
    )

except Exception:
    print(response.text)

print("\n" + "=" * 60)