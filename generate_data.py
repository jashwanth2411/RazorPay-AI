import pandas as pd
import numpy as np
import os

# ============================================================
# RAZORGUARD AI - SYNTHETIC TRANSACTION DATASET
# ============================================================

np.random.seed(42)

N = 30000

df = pd.DataFrame()


# ============================================================
# BASIC TRANSACTION INFORMATION
# ============================================================

df["transaction_id"] = [
    f"TXN{100000 + i}"
    for i in range(N)
]

df["amount"] = np.round(
    np.random.lognormal(
        mean=7,
        sigma=1,
        size=N
    ),
    2
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

df["customer_age"] = np.random.randint(
    18,
    70,
    N
)

df["customer_transaction_count"] = np.random.randint(
    1,
    300,
    N
)

df["customer_avg_amount"] = np.round(
    np.random.lognormal(
        mean=6.5,
        sigma=0.7,
        size=N
    ),
    2
)


# ============================================================
# DEVICE INFORMATION
# ============================================================

df["device_age_days"] = np.random.randint(
    0,
    1500,
    N
)

df["new_device"] = np.random.binomial(
    1,
    0.12,
    N
)


# ============================================================
# PAYMENT BEHAVIOUR
# ============================================================

df["failed_attempts"] = np.random.poisson(
    0.8,
    N
)

df["transactions_last_1h"] = np.random.poisson(
    1.5,
    N
)

df["transactions_last_24h"] = np.random.poisson(
    5,
    N
)


# ============================================================
# LOCATION
# ============================================================

df["location_changed"] = np.random.binomial(
    1,
    0.08,
    N
)


# ============================================================
# TRANSACTION TIME
# ============================================================

df["hour"] = np.random.randint(
    0,
    24,
    N
)


# ============================================================
# PAYMENT METHOD
# ============================================================

payment_methods = [
    "card",
    "upi",
    "wallet",
    "netbanking"
]

df["payment_method"] = np.random.choice(
    payment_methods,
    N,
    p=[
        0.35,
        0.45,
        0.08,
        0.12
    ]
)


# ============================================================
# MERCHANT CATEGORY
# ============================================================

merchant_categories = [
    "electronics",
    "fashion",
    "food",
    "grocery",
    "travel",
    "services"
]

df["merchant_category"] = np.random.choice(
    merchant_categories,
    N
)


# ============================================================
# FRAUD SIGNAL GENERATION
# ============================================================

fraud_score = np.zeros(N)


# Very large transaction compared
# with customer's normal behaviour

fraud_score += (
    df["amount"]
    >
    df["customer_avg_amount"] * 5
) * 2.5


# New device

fraud_score += (
    df["new_device"] * 2
)


# Multiple failed attempts

fraud_score += (
    df["failed_attempts"] >= 3
) * 2.5


# Location changed

fraud_score += (
    df["location_changed"] * 1.5
)


# Unusual transaction time

fraud_score += (
    (
        (df["hour"] <= 4)
        |
        (df["hour"] >= 23)
    )
) * 1.5


# High transaction velocity

fraud_score += (
    df["transactions_last_1h"] >= 5
) * 2


fraud_score += (
    df["transactions_last_24h"] >= 20
) * 1.5


# Add random variation

fraud_score += np.random.normal(
    0,
    1,
    N
)


# ============================================================
# CONVERT SCORE TO FRAUD PROBABILITY
# ============================================================

fraud_probability = (
    1 /
    (
        1 +
        np.exp(
            -(fraud_score - 6)
        )
    )
)


# ============================================================
# CREATE FRAUD LABEL
# ============================================================

df["is_fraud"] = (
    np.random.random(N)
    <
    fraud_probability
).astype(int)


# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs(
    "data",
    exist_ok=True
)

output_path = "data/transactions.csv"

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("RAZORGUARD AI - DATASET CREATED")
print("=" * 60)

print()

print(
    f"Total Transactions : {len(df)}"
)

print(
    f"Fraud Transactions : {df['is_fraud'].sum()}"
)

print(
    f"Legitimate         : "
    f"{(df['is_fraud'] == 0).sum()}"
)

print(
    f"Fraud Percentage   : "
    f"{df['is_fraud'].mean() * 100:.2f}%"
)

print()

print(
    f"Saved to: {output_path}"
)

print()

print("First 5 transactions:")

print(
    df.head()
)

print()

print("=" * 60)