
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/creditcard.csv"
PROCESSED_PATH = "data/processed"
MODEL_PATH = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_data():

    print("\n========== LOADING DATA ==========")

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    return df


# ==========================================
# 3. REMOVE DUPLICATES
# ==========================================

def remove_duplicates(df):

    print("\n========== DUPLICATE REMOVAL ==========")

    original_count = len(df)

    df = df.drop_duplicates().copy()

    removed = original_count - len(df)

    print("Original records:", original_count)
    print("Removed duplicates:", removed)
    print("Remaining records:", len(df))

    return df


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

def separate_features_target(df):

    print("\n========== FEATURE SEPARATION ==========")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    return X, y


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

def split_data(X, y):

    print("\n========== TRAIN TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# ==========================================
# 6. FEATURE SCALING
# ==========================================

def scale_features(X_train, X_test):

    print("\n========== FEATURE SCALING ==========")

    scaler = StandardScaler()

    # Only scale Time and Amount.
    # V1 to V28 are already PCA-transformed.

    scale_columns = ["Time", "Amount"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    X_train[scale_columns] = scaler.fit_transform(
        X_train[scale_columns]
    )

    # Apply the same transformation to test data
    X_test[scale_columns] = scaler.transform(
        X_test[scale_columns]
    )

    print("Scaled columns:", scale_columns)
    print("Scaler fitted on training data only.")

    return X_train, X_test, scaler


# ==========================================
# 7. SAVE PROCESSED DATA
# ==========================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
):

    print("\n========== SAVING PROCESSED DATA ==========")

    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    joblib.dump(
        processed_data,
        f"{PROCESSED_PATH}/train_test_data.joblib"
    )

    joblib.dump(
        scaler,
        f"{MODEL_PATH}/scaler.pkl"
    )

    joblib.dump(
        list(X_train.columns),
        f"{MODEL_PATH}/feature_columns.pkl"
    )

    print("Training and testing data saved.")
    print("Scaler saved.")
    print("Feature columns saved.")


# ==========================================
# 8. COMPLETE PREPROCESSING PIPELINE
# ==========================================

def run_preprocessing():

    df = load_data()

    df = remove_duplicates(df)

    X, y = separate_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )

    print("\n====================================")
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("====================================")


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    run_preprocessing()
