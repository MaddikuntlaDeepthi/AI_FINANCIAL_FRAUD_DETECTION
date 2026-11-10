
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

# ============================================================
# FRAUDGUARD AI - MODEL EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.pkl"

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "train_test_data.joblib"
)

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. LOAD MODEL AND TEST DATA
# ------------------------------------------------------------

def load_evaluation_data():

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded:", type(model).__name__)

    print("\nLoading saved test dataset...")

    dataset = joblib.load(DATA_PATH)

    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    print("Test features:", X_test.shape)
    print("Test labels:", y_test.shape)

    print("\nActual class distribution:")
    print(y_test.value_counts().sort_index())

    return model, X_test, y_test


# ------------------------------------------------------------
# 2. GENERATE PREDICTIONS
# ------------------------------------------------------------

def generate_predictions(model, X_test):

    print("\nGenerating predictions...")

    y_probability = model.predict_proba(X_test)[:, 1]

    y_prediction = (y_probability >= 0.5).astype(int)

    print("Prediction completed.")

    return y_prediction, y_probability


# ------------------------------------------------------------
# 3. CALCULATE METRICS
# ------------------------------------------------------------

def calculate_metrics(y_test, y_prediction, y_probability):

    metrics = {
        "Accuracy": accuracy_score(y_test, y_prediction),
        "Precision": precision_score(
            y_test, y_prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_prediction, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_prediction, zero_division=0
        ),
        "ROC AUC": roc_auc_score(
            y_test, y_probability
        ),
        "PR AUC": average_precision_score(
            y_test, y_probability
        )
    }

    print("\n" + "=" * 55)
    print("FRAUD DETECTION MODEL EVALUATION")
    print("=" * 55)

    for name, value in metrics.items():
        print(f"{name:<20}: {value:.4f}")

    return metrics


# ------------------------------------------------------------
# 4. CONFUSION MATRIX
# ------------------------------------------------------------

def save_confusion_matrix(y_test, y_prediction):

    cm = confusion_matrix(y_test, y_prediction)

    tn, fp, fn, tp = cm.ravel()

    print("\nCONFUSION MATRIX DETAILS")
    print("-" * 40)
    print("True Negatives :", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives :", tp)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.title("XGBoost Fraud Detection - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    output_path = REPORT_DIR / "evaluation_confusion_matrix.png"

    plt.savefig(output_path, dpi=300)

    plt.close()

    print("\nConfusion matrix saved:", output_path)


# ------------------------------------------------------------
# 5. CLASSIFICATION REPORT
# ------------------------------------------------------------

def save_classification_report(y_test, y_prediction):

    report = classification_report(
        y_test,
        y_prediction,
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    output_path = REPORT_DIR / "classification_report.csv"

    report_df.to_csv(output_path)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            y_prediction,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    print("Classification report saved:", output_path)


# ------------------------------------------------------------
# 6. ROC CURVE
# ------------------------------------------------------------

def save_roc_curve(y_test, y_probability):

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    auc_value = roc_auc_score(y_test, y_probability)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"XGBoost ROC-AUC = {auc_value:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - Fraud Detection")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_path = REPORT_DIR / "evaluation_roc_curve.png"

    plt.savefig(output_path, dpi=300)

    plt.close()

    print("ROC curve saved:", output_path)


# ------------------------------------------------------------
# 7. PRECISION RECALL CURVE
# ------------------------------------------------------------

def save_precision_recall_curve(y_test, y_probability):

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_probability
    )

    ap = average_precision_score(
        y_test,
        y_probability
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"Average Precision = {ap:.4f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve - Fraud Detection")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_path = REPORT_DIR / "evaluation_pr_curve.png"

    plt.savefig(output_path, dpi=300)

    plt.close()

    print("Precision-recall curve saved:", output_path)


# ------------------------------------------------------------
# 8. SAVE PREDICTION RESULTS
# ------------------------------------------------------------

def save_prediction_results(
    X_test,
    y_test,
    y_prediction,
    y_probability
):

    results = X_test.copy()

    results["Actual_Class"] = np.asarray(y_test)

    results["Predicted_Class"] = y_prediction

    results["Fraud_Probability"] = y_probability

    results["Prediction"] = np.where(
        y_prediction == 1,
        "Potential Fraud",
        "Legitimate"
    )

    results["Correct_Prediction"] = (
        results["Actual_Class"] == results["Predicted_Class"]
    )

    output_path = REPORT_DIR / "evaluation_predictions.csv"

    results.to_csv(output_path, index=False)

    print("\nPrediction results saved:", output_path)

    return results


# ------------------------------------------------------------
# 9. SAVE METRICS
# ------------------------------------------------------------

def save_metrics(metrics):

    output_path = REPORT_DIR / "evaluation_metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            {
                key: float(value)
                for key, value in metrics.items()
            },
            file,
            indent=4
        )

    print("Evaluation metrics saved:", output_path)


# ------------------------------------------------------------
# 10. MAIN EXECUTION
# ------------------------------------------------------------

def main():

    model, X_test, y_test = load_evaluation_data()

    y_prediction, y_probability = generate_predictions(
        model,
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        y_prediction,
        y_probability
    )

    save_confusion_matrix(
        y_test,
        y_prediction
    )

    save_classification_report(
        y_test,
        y_prediction
    )

    save_roc_curve(
        y_test,
        y_probability
    )

    save_precision_recall_curve(
        y_test,
        y_probability
    )

    save_prediction_results(
        X_test,
        y_test,
        y_prediction,
        y_probability
    )

    save_metrics(metrics)

    print("\n" + "=" * 55)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()
