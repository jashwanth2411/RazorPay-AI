import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# RAZORGUARD AI
# PHASE 4B - THRESHOLD OPTIMIZATION
# ============================================================


print("=" * 70)
print("RAZORGUARD AI - THRESHOLD ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/transactions.csv"
)


# ============================================================
# 2. RECREATE THE SAME FEATURES USED DURING TRAINING
# ============================================================

df["amount_ratio"] = (
    df["amount"] /
    (df["customer_avg_amount"] + 1)
)

df["is_night"] = (
    (df["hour"] <= 4) |
    (df["hour"] >= 23)
).astype(int)

df["high_velocity"] = (
    (df["transactions_last_1h"] >= 5) |
    (df["transactions_last_24h"] >= 20)
).astype(int)


# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

df = pd.get_dummies(
    df,
    columns=[
        "payment_method",
        "merchant_category"
    ],
    dtype=int
)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "transaction_id",
        "is_fraud"
    ]
)

y = df["is_fraud"]


# ============================================================
# 5. RECREATE THE SAME 80/20 TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "model/fraud_model.pkl"
)


# ============================================================
# 7. GET FRAUD PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 8. TEST MULTIPLE THRESHOLDS
# ============================================================

thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


results = []


print()
print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<10}"
    f"{'FN':<10}"
)

print("-" * 68)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<10}"
        f"{fn:<10}"
    )


# ============================================================
# 9. FIND BEST F1 THRESHOLD
# ============================================================

results_df = pd.DataFrame(
    results
)

best_row = results_df.loc[
    results_df["f1"].idxmax()
]


print()
print("=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print()

print(
    f"Threshold : "
    f"{best_row['threshold']:.2f}"
)

print(
    f"Precision : "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_row['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_row['f1']:.4f}"
)

print(
    f"False Positives : "
    f"{int(best_row['false_positives'])}"
)

print(
    f"False Negatives : "
    f"{int(best_row['false_negatives'])}"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model/threshold_results.csv",
    index=False
)


print()
print(
    "Saved: model/threshold_results.csv"
)

print()
print("=" * 70)