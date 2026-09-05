import pandas as pd

# ============================================================
# RAZORGUARD AI - DATASET INSPECTION
# ============================================================

print("=" * 60)
print("RAZORGUARD AI - DATASET ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(
    "data/transactions.csv"
)


# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

print("\n1. DATASET SIZE")
print("-" * 40)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ------------------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------------------

print("\n2. FEATURES")
print("-" * 40)

for column in df.columns:
    print(
        f"- {column}"
    )


# ------------------------------------------------------------
# DATA TYPES
# ------------------------------------------------------------

print("\n3. DATA TYPES")
print("-" * 40)

print(
    df.dtypes
)


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\n4. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()

print(
    missing
)


# ------------------------------------------------------------
# FRAUD DISTRIBUTION
# ------------------------------------------------------------

print("\n5. FRAUD DISTRIBUTION")
print("-" * 40)

fraud_count = df["is_fraud"].value_counts()

print(
    fraud_count
)

print()

fraud_percentage = (
    df["is_fraud"]
    .value_counts(
        normalize=True
    )
    * 100
)

print(
    fraud_percentage
)


# ------------------------------------------------------------
# TRANSACTION STATISTICS
# ------------------------------------------------------------

print("\n6. TRANSACTION AMOUNT")
print("-" * 40)

print(
    df["amount"].describe()
)


# ------------------------------------------------------------
# FRAUD VS LEGITIMATE AMOUNT
# ------------------------------------------------------------

print("\n7. FRAUD VS LEGITIMATE TRANSACTION")
print("-" * 40)

print(
    df.groupby(
        "is_fraud"
    )["amount"].mean()
)


# ------------------------------------------------------------
# FAILED ATTEMPTS
# ------------------------------------------------------------

print("\n8. FAILED ATTEMPTS")
print("-" * 40)

print(
    df.groupby(
        "is_fraud"
    )["failed_attempts"].mean()
)


# ------------------------------------------------------------
# NEW DEVICE
# ------------------------------------------------------------

print("\n9. NEW DEVICE")
print("-" * 40)

print(
    df.groupby(
        "is_fraud"
    )["new_device"].mean()
)


# ------------------------------------------------------------
# LOCATION CHANGE
# ------------------------------------------------------------

print("\n10. LOCATION CHANGE")
print("-" * 40)

print(
    df.groupby(
        "is_fraud"
    )["location_changed"].mean()
)


# ------------------------------------------------------------
# TRANSACTION VELOCITY
# ------------------------------------------------------------

print("\n11. TRANSACTIONS IN LAST HOUR")
print("-" * 40)

print(
    df.groupby(
        "is_fraud"
    )["transactions_last_1h"].mean()
)


print("\n12. TRANSACTIONS IN LAST 24 HOURS")
print("-" * 40)

print(
    df.groupby(
        "is_fraud"
    )["transactions_last_24h"].mean()
)


# ------------------------------------------------------------
# NIGHT TRANSACTIONS
# ------------------------------------------------------------

print("\n13. NIGHT TRANSACTIONS")
print("-" * 40)

df["is_night"] = (
    (df["hour"] <= 4)
    |
    (df["hour"] >= 23)
)

print(
    df.groupby(
        "is_fraud"
    )["is_night"].mean()
)


# ------------------------------------------------------------
# PAYMENT METHOD
# ------------------------------------------------------------

print("\n14. PAYMENT METHOD")
print("-" * 40)

print(
    pd.crosstab(
        df["payment_method"],
        df["is_fraud"],
        normalize="index"
    ) * 100
)


# ------------------------------------------------------------
# MERCHANT CATEGORY
# ------------------------------------------------------------

print("\n15. MERCHANT CATEGORY")
print("-" * 40)

print(
    pd.crosstab(
        df["merchant_category"],
        df["is_fraud"],
        normalize="index"
    ) * 100
)


print("\n")
print("=" * 60)
print("DATASET INSPECTION COMPLETE")
print("=" * 60)