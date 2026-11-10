
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/processed/train_test_data.joblib"
MODEL_PATH = "models"
REPORT_PATH = "reports"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# 2. LOAD PREPROCESSED DATA
# ==========================================

def load_processed_data():

    print("\n========== LOADING PROCESSED DATA ==========")

    data = joblib.load(DATA_PATH)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    print("\nTraining fraud count:", int(y_train.sum()))
    print("Training legitimate count:", int((y_train == 0).sum()))

    return X_train, X_test, y_train, y_test


# ==========================================
# 3. DEFINE MACHINE LEARNING MODELS
# ==========================================

def create_models(y_train):

    print("\n========== INITIALIZING ML MODELS ==========")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print("XGBoost scale_pos_weight:", scale_pos_weight)

    models = {

        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    return models


# ==========================================
# 4. TRAIN AND EVALUATE MODELS
# ==========================================

def train_and_evaluate(models, X_train, X_test, y_train, y_test):

    print("\n========== MODEL TRAINING STARTED ==========")

    results = []
    trained_models = {}

    for model_name, model in models.items():

        print("\n------------------------------------")
        print("Training:", model_name)
        print("------------------------------------")

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)

        # Fraud probability
        y_probability = model.predict_proba(X_test)[:, 1]

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test, y_probability
        )

        pr_auc = average_precision_score(
            y_test, y_probability
        )

        print("\nModel:", model_name)
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))
        print("ROC-AUC:", round(roc_auc, 4))
        print("PR-AUC:", round(pr_auc, 4))
        print("Training time:", round(training_time, 2), "seconds")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            zero_division=0
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"]
        )

        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")

        plt.savefig(
            f"{REPORT_PATH}/{filename}_confusion_matrix.png",
            dpi=300
        )

        plt.close()

        # Store model
        trained_models[model_name] = model

        joblib.dump(
            model,
            f"{MODEL_PATH}/{filename}.pkl"
        )

        # Store metrics
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Training Time": training_time
        })

    return results, trained_models


# ==========================================
# 5. MODEL COMPARISON
# ==========================================

def compare_models(results):

    print("\n========== MODEL COMPARISON ==========")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="PR-AUC",
        ascending=False
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Save metrics
    results_df.to_csv(
        f"{REPORT_PATH}/model_comparison.csv",
        index=False
    )

    # Visualize model comparison
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="PR-AUC"
    )

    plt.title("Model Comparison - PR-AUC")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("PR-AUC Score")

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_PATH}/model_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nBest model based on PR-AUC:")
    print(results_df.iloc[0]["Model"])

    return results_df


# ==========================================
# 6. SAVE BEST MODEL
# ==========================================

def save_best_model(results_df, trained_models):

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    joblib.dump(
        best_model,
        f"{MODEL_PATH}/best_fraud_model.pkl"
    )

    metadata = {
        "best_model": best_model_name,
        "selection_metric": "PR-AUC",
        "pr_auc": float(results_df.iloc[0]["PR-AUC"]),
        "features": list(
            joblib.load(f"{MODEL_PATH}/feature_columns.pkl")
        )
    }

    with open(
        f"{MODEL_PATH}/model_metadata.json",
        "w"
    ) as file:

        json.dump(metadata, file, indent=4)

    print("\nBest model saved:", best_model_name)


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def run_training():

    X_train, X_test, y_train, y_test = load_processed_data()

    models = create_models(y_train)

    results, trained_models = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results_df = compare_models(results)

    save_best_model(results_df, trained_models)

    print("\n====================================")
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("====================================")


if __name__ == "__main__":

    run_training()
