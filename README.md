# 🛡️ FraudGuard AI – Financial Fraud Detection System

An end-to-end Machine Learning application that analyzes financial transactions and identifies potentially fraudulent activities using trained classification models. Built with Python, XGBoost, Scikit-learn, and Streamlit.

## 🚀 Project Overview

FraudGuard AI is an interactive financial fraud detection platform designed to analyze historical credit card transactions and identify suspicious patterns.

The application provides transaction-level predictions, model performance comparisons, interactive analytics, explainable AI insights, and downloadable evaluation reports through a professional Streamlit dashboard.

**Note:** This is an educational and demonstration project based on historical transaction data. It is not connected to a live banking or payment system.

## 🎯 Project Objectives

- Detect potentially fraudulent financial transactions.
- Address class imbalance in financial datasets.
- Compare multiple machine learning classification algorithms.
- Evaluate model performance using appropriate classification metrics.
- Provide interactive transaction analysis.
- Explain model predictions through feature importance.
- Present results through a user-friendly dashboard.

## ✨ Key Features

### 1. Executive Overview
- Dataset summary and transaction statistics.
- Fraud and legitimate transaction distribution.
- Selected model performance overview.

### 2. Transaction Intelligence
- Explore historical test transactions.
- View model-generated fraud probabilities.
- Filter transactions using probability ranges.
- Download transaction analysis results.

### 3. Fraud Detection Center
- Single transaction analysis.
- Manual transaction input.
- CSV batch prediction.
- Fraud probability and risk classification.

### 4. Advanced Analytics
- Transaction class distribution.
- Transaction amount analysis.
- Transaction activity over time.
- Fraud amount distribution.
- Feature correlation analysis.
- XGBoost prediction intelligence.
- Fraud probability distribution.

### 5. Model Intelligence
- Compare Logistic Regression, Random Forest, and XGBoost.
- View model performance metrics.
- Compare algorithms through interactive charts.

### 6. Model Evaluation
- Accuracy, precision, recall, and F1-score.
- ROC-AUC and PR-AUC.
- Confusion matrix.
- Classification report.
- Prediction-level evaluation results.

### 7. Explainable AI
- Global feature importance.
- Individual transaction explanation.
- Feature contribution indicators.

### 8. Reports & Monitoring
- Downloadable evaluation reports.
- Confusion matrix and performance curves.
- Model comparison reports.
- Dataset monitoring statistics.

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model | XGBoost |
| Visualization | Plotly, Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Development Environment | VS Code |

## 📊 Dataset Information

**Dataset:** European Credit Card Fraud Detection Dataset

The dataset contains anonymized credit card transactions made by European cardholders.

Original dataset:
- Total transactions: 284,807
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Input features: 30
- Target column: `Class`

### Data Preprocessing

- Removed 1,081 duplicate records.
- Cleaned dataset contains 283,726 transactions.
- Legitimate transactions: 283,253.
- Fraudulent transactions: 473.
- Used stratified train-test splitting.
- Applied feature scaling to `Time` and `Amount`.
- Used class imbalance handling during model training.

**Important:** Features V1–V28 are anonymized PCA-transformed components.

## 🤖 Machine Learning Models

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

### Model Comparison

| Metric | XGBoost | Random Forest | Logistic Regression |
|---|---:|---:|---:|
| Accuracy | 99.94% | 99.95% | 97.52% |
| Precision | 83.33% | 93.24% | 5.62