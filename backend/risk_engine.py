"""
RazorGuard AI - Risk Scoring Engine

Converts ML fraud probability and transaction
behaviour into a merchant-friendly risk assessment.
"""

# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(risk_score):

    if risk_score >= 75:
        return "HIGH"

    elif risk_score >= 40:
        return "MEDIUM"

    return "LOW"


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def get_recommended_action(risk_level):

    if risk_level == "HIGH":
        return "HOLD FOR VERIFICATION"

    elif risk_level == "MEDIUM":
        return "REQUEST ADDITIONAL VERIFICATION"

    return "APPROVE - LOW RISK"


# ============================================================
# DETECT RISK SIGNALS
# ============================================================

def detect_risk_signals(transaction):

    signals = []

    amount = transaction.get("amount", 0)

    customer_avg = transaction.get(
        "customer_avg_amount",
        1
    )

    amount_ratio = (
        amount / max(customer_avg, 1)
    )

    # --------------------------------------------------------
    # Amount deviation
    # --------------------------------------------------------

    if amount_ratio >= 10:

        signals.append({
            "signal": "EXTREME_AMOUNT_DEVIATION",
            "message":
                f"Transaction amount is "
                f"{amount_ratio:.1f}x the customer's average."
        })

    elif amount_ratio >= 5:

        signals.append({
            "signal": "HIGH_AMOUNT_DEVIATION",
            "message":
                f"Transaction amount is "
                f"{amount_ratio:.1f}x the customer's average."
        })

    # --------------------------------------------------------
    # New device
    # --------------------------------------------------------

    if transaction.get("new_device", 0) == 1:

        signals.append({
            "signal": "NEW_DEVICE",
            "message":
                "Transaction originated from a "
                "new device."
        })

    # --------------------------------------------------------
    # Failed attempts
    # --------------------------------------------------------

    failed_attempts = transaction.get(
        "failed_attempts",
        0
    )

    if failed_attempts >= 5:

        signals.append({
            "signal": "EXCESSIVE_FAILED_ATTEMPTS",
            "message":
                f"{failed_attempts} failed payment attempts "
                "were detected."
        })

    elif failed_attempts >= 3:

        signals.append({
            "signal": "MULTIPLE_FAILED_ATTEMPTS",
            "message":
                f"{failed_attempts} failed payment attempts "
                "were detected."
        })

    # --------------------------------------------------------
    # Location change
    # --------------------------------------------------------

    if transaction.get("location_changed", 0) == 1:

        signals.append({
            "signal": "LOCATION_CHANGE",
            "message":
                "Transaction location differs from "
                "recent customer behaviour."
        })

    # --------------------------------------------------------
    # Transaction velocity
    # --------------------------------------------------------

    transactions_1h = transaction.get(
        "transactions_last_1h",
        0
    )

    if transactions_1h >= 10:

        signals.append({
            "signal": "EXTREME_TRANSACTION_VELOCITY",
            "message":
                f"{transactions_1h} transactions occurred "
                "within the last hour."
        })

    elif transactions_1h >= 5:

        signals.append({
            "signal": "HIGH_TRANSACTION_VELOCITY",
            "message":
                f"{transactions_1h} transactions occurred "
                "within the last hour."
        })

    # --------------------------------------------------------
    # Daily transaction velocity
    # --------------------------------------------------------

    transactions_24h = transaction.get(
        "transactions_last_24h",
        0
    )

    if transactions_24h >= 30:

        signals.append({
            "signal": "EXTREME_DAILY_VELOCITY",
            "message":
                f"{transactions_24h} transactions occurred "
                "within the last 24 hours."
        })

    elif transactions_24h >= 20:

        signals.append({
            "signal": "HIGH_DAILY_VELOCITY",
            "message":
                f"{transactions_24h} transactions occurred "
                "within the last 24 hours."
        })

    # --------------------------------------------------------
    # Unusual time
    # --------------------------------------------------------

    hour = transaction.get(
        "hour",
        12
    )

    if hour <= 4 or hour >= 23:

        signals.append({
            "signal": "UNUSUAL_TIME",
            "message":
                f"Transaction occurred at {hour:02d}:00, "
                "an unusual transaction period."
        })

    # --------------------------------------------------------
    # No major signals
    # --------------------------------------------------------

    if not signals:

        signals.append({
            "signal": "NO_MAJOR_SIGNAL",
            "message":
                "No major behavioural risk signals detected."
        })

    return signals


# ============================================================
# CALCULATE RISK SCORE
# ============================================================

def calculate_risk_score(
    fraud_probability,
    signals
):

    # ML probability is the main component.
    base_score = fraud_probability * 100

    signal_weights = {

        "EXTREME_AMOUNT_DEVIATION": 5,
        "HIGH_AMOUNT_DEVIATION": 3,

        "NEW_DEVICE": 3,

        "EXCESSIVE_FAILED_ATTEMPTS": 5,
        "MULTIPLE_FAILED_ATTEMPTS": 3,

        "LOCATION_CHANGE": 2,

        "EXTREME_TRANSACTION_VELOCITY": 5,
        "HIGH_TRANSACTION_VELOCITY": 3,

        "EXTREME_DAILY_VELOCITY": 5,
        "HIGH_DAILY_VELOCITY": 3,

        "UNUSUAL_TIME": 2,

        "NO_MAJOR_SIGNAL": 0
    }

    signal_adjustment = 0

    for signal in signals:

        signal_adjustment += signal_weights.get(
            signal["signal"],
            0
        )

    final_score = (
        base_score +
        signal_adjustment
    )

    # Keep score between 0 and 100.

    final_score = min(
        max(final_score, 0),
        100
    )

    return round(final_score)


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_transaction(
    transaction,
    fraud_probability
):

    signals = detect_risk_signals(
        transaction
    )

    risk_score = calculate_risk_score(
        fraud_probability,
        signals
    )

    risk_level = get_risk_level(
        risk_score
    )

    recommended_action = get_recommended_action(
        risk_level
    )

    return {

        "risk_score": risk_score,

        "fraud_probability":
            round(
                fraud_probability,
                4
            ),

        "risk_level":
            risk_level,

        "signals":
            signals,

        "recommended_action":
            recommended_action
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_transaction = {

        "amount": 50000,

        "customer_avg_amount": 1000,

        "new_device": 1,

        "failed_attempts": 6,

        "location_changed": 1,

        "transactions_last_1h": 8,

        "transactions_last_24h": 25,

        "hour": 2
    }

    # Simulated ML probability
    # for testing the risk engine.

    fraud_probability = 0.87

    result = analyze_transaction(
        test_transaction,
        fraud_probability
    )

    print()
    print("=" * 60)
    print("RAZORGUARD AI - RISK ANALYSIS")
    print("=" * 60)

    print()

    print(
        f"Risk Score: "
        f"{result['risk_score']}/100"
    )

    print(
        f"Fraud Probability: "
        f"{result['fraud_probability'] * 100:.2f}%"
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

    print("Recommended Action:")

    print(
        f"  {result['recommended_action']}"
    )

    print("=" * 60)