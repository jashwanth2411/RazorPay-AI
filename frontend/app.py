import os
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# BACKEND IMPORT
# ============================================================

from backend.risk_engine import analyze_transaction


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RazorGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 44px;
    font-weight: 800;
}

.subtitle {
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 20px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 15px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        PROJECT_ROOT,
        "model",
        "fraud_model.pkl"
    )

    features_path = os.path.join(
        PROJECT_ROOT,
        "model",
        "features.pkl"
    )

    model = joblib.load(model_path)

    features = joblib.load(
        features_path
    )

    return model, features


model, model_features = load_model()


# ============================================================
# SESSION HISTORY
# ============================================================

if "transaction_history" not in st.session_state:

    st.session_state.transaction_history = []


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_transaction(transaction):

    df = pd.DataFrame(
        [transaction]
    )

    # Amount compared with customer's average
    df["amount_ratio"] = (
        df["amount"]
        / (df["customer_avg_amount"] + 1)
    )

    # Night transaction
    df["is_night"] = (
        (df["hour"] < 6)
        | (df["hour"] >= 23)
    ).astype(int)

    # High transaction velocity
    df["high_velocity"] = (
        df["transactions_last_1h"] >= 5
    ).astype(int)

    # Remove transaction ID
    df = df.drop(
        columns=["transaction_id"],
        errors="ignore"
    )

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=[
            "payment_method",
            "merchant_category"
        ]
    )

    # Add missing model columns
    for feature in model_features:

        if feature not in df.columns:

            df[feature] = 0

    # Exact feature order
    df = df[model_features]

    return df


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):

    X = prepare_transaction(
        transaction
    )

    fraud_probability = (
        model.predict_proba(X)[0][1]
    )

    result = analyze_transaction(
        transaction,
        fraud_probability
    )

    return result


# ============================================================
# RISK GAUGE
# ============================================================

def create_risk_gauge(score):

    if score >= 75:

        gauge_color = "#ef4444"

    elif score >= 40:

        gauge_color = "#f59e0b"

    else:

        gauge_color = "#22c55e"

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,

            number={
                "suffix": "/100",
                "font": {
                    "size": 34
                }
            },

            title={
                "text": "RISK SCORE",
                "font": {
                    "size": 20
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "dtick": 20
                },

                "bar": {
                    "color": gauge_color
                },

                "steps": [
                    {
                        "range": [0, 40],
                        "color": "#14532d"
                    },
                    {
                        "range": [40, 75],
                        "color": "#713f12"
                    },
                    {
                        "range": [75, 100],
                        "color": "#7f1d1d"
                    }
                ]
            }
        )
    )

    figure.update_layout(
        height=300,
        margin={
            "l": 20,
            "r": 20,
            "t": 50,
            "b": 10
        },
        paper_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white"
        }
    )

    return figure


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ RazorGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Payment Risk & Fraud Investigation Platform'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM STATUS
# ============================================================

status1, status2, status3, status4 = st.columns(4)

with status1:
    st.success("● MODEL ONLINE")

with status2:
    st.success("● RISK ENGINE ONLINE")

with status3:
    st.success("● ANALYZER ONLINE")

with status4:
    st.info("● TEST MODE")


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ RazorGuard")

st.sidebar.caption(
    "AI-Powered Payment Risk Management"
)

st.sidebar.divider()

st.sidebar.markdown(
    "### SYSTEM PIPELINE"
)

st.sidebar.markdown(
    """
💳 **Transaction**

↓

🧠 **XGBoost Model**

↓

📊 **Fraud Probability**

↓

⚠️ **Risk Engine**

↓

🔍 **Risk Signals**

↓

🛡️ **Recommended Action**

↓

📈 **Analytics**
"""
)

st.sidebar.divider()

st.sidebar.markdown(
    "### ABOUT"
)

st.sidebar.info(
    "RazorGuard AI is a student prototype "
    "for transaction fraud detection and "
    "risk investigation."
)

st.sidebar.warning(
    "Synthetic/test data only. "
    "No real payments are processed."
)


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '💳 Transaction Investigation'
    '</div>',
    unsafe_allow_html=True
)


input_col1, input_col2, input_col3 = st.columns(3)


# ============================================================
# TRANSACTION DETAILS INPUT
# ============================================================

with input_col1:

    st.markdown("#### Transaction")

    transaction_id = st.text_input(
        "Transaction ID",
        value="TXN-DEMO-001"
    )

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        value=25000.0,
        step=500.0
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "upi",
            "credit_card",
            "debit_card",
            "netbanking",
            "wallet"
        ]
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "electronics",
            "fashion",
            "grocery",
            "travel",
            "food",
            "entertainment"
        ]
    )


# ============================================================
# CUSTOMER INPUT
# ============================================================

with input_col2:

    st.markdown(
        "#### Customer Behaviour"
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=28
    )

    customer_transaction_count = st.number_input(
        "Customer Transaction Count",
        min_value=0,
        value=42
    )

    customer_avg_amount = st.number_input(
        "Customer Average Amount (₹)",
        min_value=1.0,
        value=1200.0,
        step=100.0
    )

    location_changed = st.selectbox(
        "Location Changed",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


# ============================================================
# DEVICE / VELOCITY INPUT
# ============================================================

with input_col3:

    st.markdown(
        "#### Device & Velocity"
    )

    device_age_days = st.number_input(
        "Device Age (days)",
        min_value=0,
        value=2
    )

    new_device = st.selectbox(
        "New Device",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    failed_attempts = st.number_input(
        "Failed Payment Attempts",
        min_value=0,
        value=4
    )

    transactions_last_1h = st.number_input(
        "Transactions - Last 1 Hour",
        min_value=0,
        value=7
    )

    transactions_last_24h = st.number_input(
        "Transactions - Last 24 Hours",
        min_value=0,
        value=18
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=2
    )


st.write("")


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 ANALYZE TRANSACTION",
    width="stretch",
    type="primary"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    transaction = {

        "transaction_id":
            transaction_id,

        "amount":
            amount,

        "customer_age":
            customer_age,

        "customer_transaction_count":
            customer_transaction_count,

        "customer_avg_amount":
            customer_avg_amount,

        "device_age_days":
            device_age_days,

        "new_device":
            new_device,

        "failed_attempts":
            failed_attempts,

        "transactions_last_1h":
            transactions_last_1h,

        "transactions_last_24h":
            transactions_last_24h,

        "location_changed":
            location_changed,

        "hour":
            hour,

        "payment_method":
            payment_method,

        "merchant_category":
            merchant_category
    }


    # ========================================================
    # RUN MODEL
    # ========================================================

    try:

        result = predict_transaction(
            transaction
        )

    except Exception as error:

        st.error(
            "Transaction analysis failed."
        )

        st.exception(error)

        st.stop()


    # ========================================================
    # RESULT VALUES
    # ========================================================

    fraud_probability = (
        result["fraud_probability"] * 100
    )

    risk_score = int(
        result["risk_score"]
    )

    risk_level = result[
        "risk_level"
    ]

    recommended_action = result[
        "recommended_action"
    ]

    signals = result.get(
        "signals",
        []
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_record = {

        "Transaction ID":
            str(transaction_id),

        "Amount":
            float(amount),

        "Fraud Probability":
            round(
                fraud_probability,
                2
            ),

        "Risk Score":
            risk_score,

        "Risk Level":
            str(risk_level),

        "Recommended Action":
            str(recommended_action)
    }


    st.session_state.transaction_history.append(
        history_record
    )


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🚨 Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )


    gauge_col, metric_col = st.columns(
        [1, 2]
    )


    # ========================================================
    # GAUGE
    # ========================================================

    with gauge_col:

        st.plotly_chart(
            create_risk_gauge(
                risk_score
            ),
            width="stretch"
        )


    # ========================================================
    # METRICS
    # ========================================================

    with metric_col:

        metric1, metric2 = st.columns(2)

        with metric1:

            st.metric(
                "FRAUD PROBABILITY",
                f"{fraud_probability:.2f}%"
            )

        with metric2:

            st.metric(
                "RISK SCORE",
                f"{risk_score}/100"
            )


        st.write("")


        if risk_level == "HIGH":

            st.error(
                "🔴 HIGH RISK — Immediate verification recommended."
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "🟡 MEDIUM RISK — Additional verification recommended."
            )

        else:

            st.success(
                "🟢 LOW RISK — Transaction appears relatively safe."
            )


    # ========================================================
    # RISK SIGNALS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '⚠️ Detected Risk Signals'
        '</div>',
        unsafe_allow_html=True
    )


    if signals:

        signal_col1, signal_col2 = st.columns(2)


        for index, signal in enumerate(
            signals
        ):

            if isinstance(
                signal,
                dict
            ):

                signal_name = signal.get(
                    "signal",
                    "RISK SIGNAL"
                )

                message = signal.get(
                    "message",
                    "Risk signal detected."
                )

            else:

                signal_name = "RISK SIGNAL"

                message = str(
                    signal
                )


            if index % 2 == 0:

                with signal_col1:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### ⚠️ {signal_name}"
                        )

                        st.write(
                            message
                        )

            else:

                with signal_col2:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### ⚠️ {signal_name}"
                        )

                        st.write(
                            message
                        )

    else:

        st.success(
            "No significant risk signals detected."
        )


    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🛡️ Recommended Action'
        '</div>',
        unsafe_allow_html=True
    )


    if risk_level == "HIGH":

        st.error(
            f"🚨 {recommended_action}"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            f"⚠️ {recommended_action}"
        )

    else:

        st.success(
            f"✅ {recommended_action}"
        )


    # ========================================================
    # INVESTIGATION SUMMARY
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🧠 Investigation Summary'
        '</div>',
        unsafe_allow_html=True
    )


    primary_signal_types = {
        "EXTREME_AMOUNT_DEVIATION",
        "NEW_DEVICE",
        "MULTIPLE_FAILED_ATTEMPTS",
        "LOCATION_CHANGE"
    }


    primary_signals = []

    supporting_signals = []


    for signal in signals:

        if isinstance(
            signal,
            dict
        ):

            signal_type = signal.get(
                "signal",
                ""
            )

            message = signal.get(
                "message",
                ""
            )

            if signal_type in primary_signal_types:

                primary_signals.append(
                    message
                )

            else:

                supporting_signals.append(
                    message
                )


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.markdown(
            "#### 🔴 Primary Evidence"
        )

        if primary_signals:

            for message in primary_signals:

                st.write(
                    f"• {message}"
                )

        else:

            st.write(
                "No primary evidence detected."
            )


    with summary_col2:

        st.markdown(
            "#### 🟡 Supporting Evidence"
        )

        if supporting_signals:

            for message in supporting_signals:

                st.write(
                    f"• {message}"
                )

        else:

            st.write(
                "No supporting evidence detected."
            )


    # ========================================================
    # RAZORGUARD ASSESSMENT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🤖 RazorGuard Assessment'
        '</div>',
        unsafe_allow_html=True
    )


    if risk_level == "HIGH":

        assessment = (
            "The transaction presents a high level of risk "
            "based on the fraud model prediction and multiple "
            "behavioural indicators. Strong customer verification "
            "is recommended before approval."
        )

    elif risk_level == "MEDIUM":

        assessment = (
            "The transaction presents elevated risk. "
            "The machine-learning prediction is supported "
            "by multiple behavioural indicators. Additional "
            "customer verification is recommended before approval."
        )

    else:

        assessment = (
            "The transaction appears relatively low risk "
            "based on the available transaction features "
            "and behavioural indicators."
        )


    st.info(
        assessment
    )


    # ========================================================
    # TRANSACTION DETAILS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📋 Transaction Details'
        '</div>',
        unsafe_allow_html=True
    )


    detail_data = [

        {
            "Parameter": "Transaction ID",
            "Value": str(transaction_id)
        },

        {
            "Parameter": "Amount",
            "Value": f"₹{amount:,.2f}"
        },

        {
            "Parameter": "Customer Average",
            "Value": f"₹{customer_avg_amount:,.2f}"
        },

        {
            "Parameter": "Amount Ratio",
            "Value": (
                f"{amount / (customer_avg_amount + 1):.1f}x"
            )
        },

        {
            "Parameter": "Payment Method",
            "Value": str(payment_method)
        },

        {
            "Parameter": "Merchant Category",
            "Value": str(merchant_category)
        },

        {
            "Parameter": "Device Age",
            "Value": f"{device_age_days} days"
        },

        {
            "Parameter": "New Device",
            "Value": (
                "Yes"
                if new_device == 1
                else "No"
            )
        },

        {
            "Parameter": "Failed Attempts",
            "Value": str(failed_attempts)
        },

        {
            "Parameter": "Transactions / 1h",
            "Value": str(transactions_last_1h)
        },

        {
            "Parameter": "Transactions / 24h",
            "Value": str(transactions_last_24h)
        },

        {
            "Parameter": "Location Changed",
            "Value": (
                "Yes"
                if location_changed == 1
                else "No"
            )
        },

        {
            "Parameter": "Transaction Hour",
            "Value": f"{hour:02d}:00"
        }
    ]


    detail_df = pd.DataFrame(
        detail_data
    )


    st.dataframe(
        detail_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# TRANSACTION ANALYTICS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📊 Transaction Analytics'
    '</div>',
    unsafe_allow_html=True
)


history = (
    st.session_state.transaction_history
)


if history:

    history_df = pd.DataFrame(
        history
    )


    # ========================================================
    # COUNTS
    # ========================================================

    total_transactions = len(
        history_df
    )

    high_risk = len(
        history_df[
            history_df["Risk Level"] == "HIGH"
        ]
    )

    medium_risk = len(
        history_df[
            history_df["Risk Level"] == "MEDIUM"
        ]
    )

    low_risk = len(
        history_df[
            history_df["Risk Level"] == "LOW"
        ]
    )


    # ========================================================
    # ANALYTICS METRICS
    # ========================================================

    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = (
        st.columns(4)
    )


    with analytics_col1:

        st.metric(
            "Transactions Analyzed",
            total_transactions
        )


    with analytics_col2:

        st.metric(
            "🔴 High Risk",
            high_risk
        )


    with analytics_col3:

        st.metric(
            "🟡 Medium Risk",
            medium_risk
        )


    with analytics_col4:

        st.metric(
            "🟢 Low Risk",
            low_risk
        )


    st.write("")


    # ========================================================
    # ANALYTICS CHARTS
    # ========================================================

    chart_col1, chart_col2 = st.columns(2)


    with chart_col1:

        st.markdown(
            "#### Risk Distribution"
        )


        risk_counts = pd.DataFrame(
            {
                "Risk Level": [
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ],

                "Transactions": [
                    low_risk,
                    medium_risk,
                    high_risk
                ]
            }
        )


        st.bar_chart(
            risk_counts.set_index(
                "Risk Level"
            ),
            width="stretch"
        )


    with chart_col2:

        st.markdown(
            "#### Fraud Probability"
        )


        probability_df = history_df[
            [
                "Transaction ID",
                "Fraud Probability"
            ]
        ].copy()


        probability_df = (
            probability_df
            .set_index(
                "Transaction ID"
            )
        )


        st.bar_chart(
            probability_df,
            width="stretch"
        )


    # ========================================================
    # HISTORY TABLE
    # ========================================================

    st.markdown(
        "#### 🧾 Transaction History"
    )


    display_history = (
        history_df.copy()
    )


    display_history["Amount"] = (
        display_history["Amount"]
        .map(
            lambda value:
            f"₹{value:,.2f}"
        )
    )


    display_history["Fraud Probability"] = (
        display_history["Fraud Probability"]
        .map(
            lambda value:
            f"{value:.2f}%"
        )
    )


    display_history["Risk Score"] = (
        display_history["Risk Score"]
        .map(
            lambda value:
            f"{value}/100"
        )
    )


    st.dataframe(
        display_history,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # CLEAR HISTORY
    # ========================================================

    if st.button(
        "🗑️ Clear Transaction History"
    ):

        st.session_state.transaction_history = []

        st.rerun()


else:

    st.info(
        "No transactions analyzed yet. "
        "Analyze a transaction above to populate "
        "the analytics dashboard."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📈 Model Information'
    '</div>',
    unsafe_allow_html=True
)


model_col1, model_col2, model_col3, model_col4 = (
    st.columns(4)
)


with model_col1:

    st.metric(
        "ROC-AUC",
        "0.8325"
    )


with model_col2:

    st.metric(
        "Precision",
        "20.81%"
    )


with model_col3:

    st.metric(
        "Recall",
        "48.36%"
    )


with model_col4:

    st.metric(
        "F1 Score",
        "29.10%"
    )


st.caption(
    "Metrics are from the held-out synthetic test set "
    "at the baseline 0.50 decision threshold. "
    "They represent prototype evaluation results, "
    "not production performance."
)


# ============================================================
# LIMITATION
# ============================================================

st.divider()

st.info(
    "This assessment is based on available transaction "
    "features and model outputs. It is not definitive "
    "proof of fraud."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ RazorGuard AI • Defensive fraud-risk prototype • "
    "Synthetic/Test data only"
)