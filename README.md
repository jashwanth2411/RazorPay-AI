# 🛡️ RazorGuard AI

### AI-Powered Defensive Fraud Risk Detection System

RazorGuard AI is a defensive fraud-risk detection prototype designed to identify potentially suspicious payment transactions and recommend appropriate verification actions.

It combines machine learning with a rule-based risk engine and a FastAPI webhook integration to provide real-time transaction risk assessment.

---

## 🚨 Problem

Online payment systems can face financial losses due to fraudulent transactions, chargebacks and unauthorized payments.

RazorGuard AI addresses this problem by analyzing transaction and behavioral signals such as:

- Transaction amount
- Customer transaction history
- New device usage
- Failed payment attempts
- Transaction velocity
- Location changes
- Unusual transaction time
- Payment method
- Merchant category

The system produces a fraud probability, risk score, risk level and recommended action.

---

## 🎯 Key Features

- 🤖 XGBoost-based fraud detection model
- 📊 Fraud probability prediction
- 🚦 Risk score from 0–100
- 🔴 HIGH / 🟠 MEDIUM / 🟢 LOW risk classification
- 🔍 Explainable risk signals
- 🧠 Evidence-based investigation summary
- ⚡ FastAPI webhook endpoint
- 🔐 Razorpay webhook signature verification
- 📈 Streamlit monitoring dashboard
- 🧪 Automated webhook test
- 🛡️ Defensive-only architecture
- 🔑 Environment variables for secrets

---

## 🏗️ System Architecture

```text
                    Payment Transaction
                            │
                            ▼
                  Razorpay Test Webhook
                            │
                            ▼
                 Webhook Signature Check
                            │
                            ▼
                  Feature Preparation
                            │
                            ▼
                    XGBoost Model
                            │
                            ▼
                  Fraud Probability
                            │
                            ▼
                    Risk Engine
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
           LOW            MEDIUM          HIGH
             │              │              │
             ▼              ▼              ▼
          APPROVE       VERIFY         HOLD
                            │
                            ▼
                  Investigation Summary
                            │
                            ▼
                    Streamlit Dashboard