import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# RAZORGUARD AI
# PHASE 4 - FRAUD DETECTION MODEL
# ============================================================


print("=" * 60)
print("RAZORGUARD AI - FRAUD DETECTION MODEL")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/transactions.csv"
)

print()
print("Dataset loaded.")
print(
    "Total transactions:",
    len(df)
)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

print()
print("Creating features...")


# How many times larger is this transaction
# compared with the customer's average?

df["amount_ratio"] = (
    df["amount"] /
    (df["customer_avg_amount"] + 1)
)


# Detect unusual transaction hours

df["is_night"] = (
    (df["hour"] <= 4) |
    (df["hour"] >= 23)
).astype(int)


# Detect unusually high transaction velocity

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


print()
print(
    "Number of model features:",
    X.shape[1]
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print()
print("Creating held-out test set...")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 6. HANDLE CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = (
    negative /
    positive
)

print()
print(
    "Class imbalance ratio:",
    round(scale_pos_weight, 2)
)


# ============================================================
# 7. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42
)


# ============================================================
# 8. TRAIN
# ============================================================

print()
print("Training XGBoost fraud detector...")
print()

model.fit(
    X_train,
    y_train
)

print()
print("Training completed.")


# ============================================================
# 9. PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# Default threshold

threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)


# ============================================================
# 10. EVALUATION
# ============================================================

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

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("RAZORGUARD MODEL PERFORMANCE")
print("=" * 60)

print()

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print()

print("Confusion Matrix:")
print(cm)

print()

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 13. FALSE POSITIVE ANALYSIS
# ============================================================

print("=" * 60)
print("FALSE POSITIVE ANALYSIS")
print("=" * 60)

print()

print(
    "True Negatives :",
    tn
)

print(
    "False Positives:",
    fp
)

print(
    "False Negatives:",
    fn
)

print(
    "True Positives  :",
    tp
)


# Example business assumption.
# We will later make this configurable.

AVERAGE_LEGITIMATE_TRANSACTION = 1500

false_positive_cost = (
    fp *
    AVERAGE_LEGITIMATE_TRANSACTION
)

print()

print(
    "Estimated false-positive exposure: ₹",
    round(
        false_positive_cost,
        2
    )
)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print()

print(
    importance.head(15)
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/fraud_model.pkl"
)

joblib.dump(
    list(X.columns),
    "model/features.pkl"
)

print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "model/fraud_model.pkl"
)

print(
    "model/features.pkl"
)