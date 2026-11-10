
import json
import joblib
import numpy as np
import pandas as pd
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
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "train_test_data.joblib"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

print("\nLoading trained model...")
model = joblib.load(MODEL_PATH)

print("Model loaded:", type(model).__name__)

print("\nLoading saved test dataset...")
data = joblib.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, probabilities)
    ),
    "pr_auc": float(
        average_precision_score(y_test, probabilities)
    )
}

print("\n========== INDEPENDENT TEST EVALUATION ==========")

for name, value in metrics.items():
    print(f"{name.upper():15}: {value:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX")
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

plt.title("XGBoost Independent Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "final_confusion_matrix.png",
    dpi=300
)

plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, probabilities)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost AUC = {metrics['roc_auc']:.4f}"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Independent Test ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "final_roc_curve.png",
    dpi=300
)

plt.close()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"PR-AUC = {metrics['pr_auc']:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Independent Test Precision-Recall Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "final_pr_curve.png",
    dpi=300
)

plt.close()

# Classification report
report = classification_report(
    y_test,
    predictions,
    target_names=["Legitimate", "Fraud"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "final_classification_report.csv"
)

# Evaluation predictions
results = X_test.copy().reset_index(drop=True)

results["Actual_Class"] = np.asarray(y_test)
results["Fraud_Probability"] = probabilities
results["Predicted_Class"] = predictions

results["Prediction"] = np.where(
    predictions == 1,
    "Potential Fraud",
    "Legitimate"
)

results.to_csv(
    REPORT_DIR / "final_evaluation_predictions.csv",
    index=False
)

# Metrics JSON
with open(
    REPORT_DIR / "final_evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print("\nReports generated successfully:")
print("1. final_confusion_matrix.png")
print("2. final_roc_curve.png")
print("3. final_pr_curve.png")
print("4. final_classification_report.csv")
print("5. final_evaluation_predictions.csv")
print("6. final_evaluation_metrics.json")

print("\nINDEPENDENT TEST EVALUATION COMPLETED")
