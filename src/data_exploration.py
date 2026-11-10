
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

DATA_PATH = "data/creditcard.csv"
REPORT_PATH = "reports"

os.makedirs(REPORT_PATH, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    print("\n========== LOADING DATASET ==========")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    return df


# ==========================================
# 3. BASIC DATASET INFORMATION
# ==========================================

def dataset_information(df):

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nDataset information:")
    df.info()


# ==========================================
# 4. MISSING VALUE ANALYSIS
# ==========================================

def missing_value_analysis(df):

    print("\n========== MISSING VALUE ANALYSIS ==========")

    missing = df.isnull().sum()

    print("\nMissing values per column:")
    print(missing[missing > 0] if missing.any() else "No missing values found.")

    print("\nTotal missing values:", missing.sum())


# ==========================================
# 5. DUPLICATE ANALYSIS
# ==========================================

def duplicate_analysis(df):

    print("\n========== DUPLICATE ANALYSIS ==========")

    duplicates = df.duplicated().sum()

    print("Total duplicate rows:", duplicates)

    print("Note: Duplicates will be investigated before deciding whether to remove them.")


# ==========================================
# 6. FRAUD DISTRIBUTION
# ==========================================

def fraud_distribution(df):

    print("\n========== FRAUD DISTRIBUTION ==========")

    class_counts = df["Class"].value_counts().sort_index()

    print("\nTransaction counts:")
    print("Legitimate transactions (0):", class_counts.get(0, 0))
    print("Fraudulent transactions (1):", class_counts.get(1, 0))

    print("\nTransaction percentages:")
    print((class_counts / len(df) * 100).round(4))

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Class"
    )

    plt.title("Legitimate vs Fraudulent Transactions")
    plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_PATH}/fraud_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 7. TRANSACTION AMOUNT ANALYSIS
# ==========================================

def transaction_amount_analysis(df):

    print("\n========== TRANSACTION AMOUNT ANALYSIS ==========")

    print("\nOverall transaction amount statistics:")
    print(df["Amount"].describe())

    print("\nAmount statistics by transaction class:")
    print(df.groupby("Class")["Amount"].describe())

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y="Amount"
    )

    plt.title("Transaction Amount by Class")
    plt.xlabel("Transaction Class")
    plt.ylabel("Transaction Amount")

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_PATH}/amount_analysis.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. CORRELATION ANALYSIS
# ==========================================

def correlation_analysis(df):

    print("\n========== CORRELATION ANALYSIS ==========")

    correlations = df.corr(numeric_only=True)["Class"]

    correlations = correlations.drop("Class").sort_values(
        key=abs,
        ascending=False
    )

    print("\nTop 10 features correlated with fraud:")
    print(correlations.head(10))

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_PATH}/correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 9. GENERATE EDA REPORT
# ==========================================

def run_eda():

    df = load_dataset()

    dataset_information(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    fraud_distribution(df)

    transaction_amount_analysis(df)

    correlation_analysis(df)

    print("\n====================================")
    print("EDA COMPLETED SUCCESSFULLY!")
    print("Charts saved in reports folder.")
    print("====================================")


# ==========================================
# 10. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    run_eda()
