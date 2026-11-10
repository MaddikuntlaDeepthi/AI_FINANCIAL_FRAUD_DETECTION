
import os
import joblib
import pandas as pd
import numpy as np


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "models/best_fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

REPORT_PATH = "reports"

os.makedirs(REPORT_PATH, exist_ok=True)


# ==========================================
# 2. LOAD SAVED MODEL
# ==========================================

def load_model():

    print("\n========== LOADING FRAUD MODEL ==========")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    print("Best fraud model loaded successfully!")
    print("Model:", type(model).__name__)

    return model, scaler, feature_columns


# ==========================================
# 3. VALIDATE TRANSACTION DATA
# ==========================================

def validate_data(df, feature_columns):

    print("\n========== VALIDATING INPUT DATA ==========")

    # Remove target column if supplied
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in feature_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if extra_columns:
        raise ValueError(
            f"Unexpected columns: {extra_columns}"
        )

    # Enforce training feature order
    df = df[feature_columns].copy()

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isnull().any().any():
        raise ValueError(
            "Input contains missing or non-numeric values."
        )

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(
            "Input contains infinite or invalid numerical values."
        )

    print("Input validation successful!")

    return df


# ==========================================
# 4. PREPROCESS NEW TRANSACTIONS
# ==========================================

def preprocess_transactions(df, scaler):

    df = df.copy()

    scale_columns = ["Time", "Amount"]

    df[scale_columns] = scaler.transform(
        df[scale_columns]
    )

    return df


# ==========================================
# 5. PREDICT FRAUD
# ==========================================

def predict_transactions(df):

    model, scaler, feature_columns = load_model()

    df = validate_data(df, feature_columns)

    processed_data = preprocess_transactions(
        df,
        scaler
    )

    print("\n========== GENERATING PREDICTIONS ==========")

    predictions = model.predict(processed_data)

    probabilities = model.predict_proba(
        processed_data
    )[:, 1]

    results = df.copy()

    results["Fraud_Probability"] = probabilities

    results["Prediction"] = np.where(
        predictions == 1,
        "Potential Fraud",
        "Legitimate"
    )

    # Display risk bands.
    # These are presentation categories, not calibrated risk estimates.

    results["Risk_Level"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.30, 0.70, 1.001],
        labels=["Low", "Medium", "High"]
    )

    print("\nPrediction completed!")
    print("Total transactions:", len(results))

    print(
        "Potential fraud:",
        int((predictions == 1).sum())
    )

    print(
        "Legitimate:",
        int((predictions == 0).sum())
    )

    return results


# ==========================================
# 6. SAVE PREDICTION REPORT
# ==========================================

def save_report(results, filename="fraud_predictions.csv"):

    output_path = os.path.join(
        REPORT_PATH,
        filename
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nPrediction report saved:")
    print(output_path)


# ==========================================
# 7. MAIN TEST
# ==========================================

def run_prediction_test():

    print("\n========== FRAUD PREDICTION TEST ==========")

    # Load prepared test data
    data = joblib.load(
        "data/processed/train_test_data.joblib"
    )

    X_test = data["X_test"]

    # Select 10 unseen transactions
    sample = X_test.head(10).copy()

    # Convert scaled values back to original
    # values because prediction applies the scaler.

    scaler = joblib.load(SCALER_PATH)

    sample[["Time", "Amount"]] = scaler.inverse_transform(
        sample[["Time", "Amount"]]
    )

    results = predict_transactions(sample)

    print("\nFirst 10 predictions:")

    print(
        results[
            [
                "Amount",
                "Fraud_Probability",
                "Prediction",
                "Risk_Level"
            ]
        ].to_string(index=False)
    )

    save_report(results)

    print("\nPREDICTION ENGINE WORKING SUCCESSFULLY!")


if __name__ == "__main__":

    run_prediction_test()
