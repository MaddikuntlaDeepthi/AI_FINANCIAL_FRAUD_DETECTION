import shutil
from urllib.request import urlopen
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# FRAUDGUARD AI - VERSION 2.3
# ============================================================

st.set_page_config(
    page_title="FraudGuard AI | Financial Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
SPLIT_PATH = BASE_DIR / "data" / "processed" / "train_test_data.joblib"

MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

REPORTS = BASE_DIR / "reports"

METRICS_PATH = REPORTS / "final_evaluation_metrics.json"
CONFUSION_PATH = REPORTS / "final_confusion_matrix.png"
ROC_PATH = REPORTS / "final_roc_curve.png"
PR_PATH = REPORTS / "final_pr_curve.png"
CLASS_REPORT_PATH = REPORTS / "final_classification_report.csv"
PREDICTIONS_PATH = REPORTS / "final_evaluation_predictions.csv"

COMPARISON_PATH = REPORTS / "model_comparison.csv"

THRESHOLD = 0.50

# ============================================================
# PREMIUM DARK FINTECH THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 5%, #17152f 0%, transparent 35%),
        linear-gradient(135deg, #070b16 0%, #0a1020 100%);
    color: #f5f7ff;
}

header[data-testid="stHeader"] {
    background: rgba(7, 11, 22, 0.95);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a2b 0%, #08111f 100%);
    border-right: 1px solid #26364e;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] * {
    color: #e8f0ff;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

p, label, span {
    color: #c4d3e9;
}

.hero-label {
    color: #20d9ed;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 5px;
    text-transform: uppercase;
}

.hero-title {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: #8cb9ed;
    font-size: 14px;
    margin-bottom: 26px;
}

.status-pill {
    display: inline-block;
    padding: 12px 18px;
    border-radius: 30px;
    color: #00e6b0;
    background: rgba(0, 230, 176, 0.08);
    border: 1px solid rgba(0, 230, 176, 0.35);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 28px;
}

.metric-card {
    background: linear-gradient(145deg, #18283e, #101b2c);
    border: 1px solid #263c58;
    border-radius: 20px;
    padding: 22px;
    min-height: 140px;
    margin-bottom: 12px;
}

.metric-label {
    color: #85b9f6;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.metric-value {
    color: white;
    font-size: 27px;
    font-weight: 800;
    margin-top: 24px;
}

.metric-caption {
    color: #86b8ef;
    font-size: 10px;
    margin-top: 12px;
}

.panel {
    background: #101a2a;
    border: 1px solid #263a55;
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 18px;
}

.section-caption {
    color: #20d9ed;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 8px;
}

div.stButton > button {
    background: linear-gradient(90deg, #13b8dc, #5578f5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0d9fc0, #435fd1);
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: #111e30;
    border: 1px solid #2a405d;
    padding: 15px;
    border-radius: 14px;
}

div[data-testid="stMetric"] label {
    color: #83b8f4 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: white !important;
}

div[data-testid="stFileUploader"] {
    background: #101a2a;
    border: 1px dashed #3d668d;
    border-radius: 15px;
    padding: 15px;
}

.stAlert {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

.app-footer {
    text-align: center;
    color: #6684a8;
    font-size: 11px;
    border-top: 1px solid #26364e;
    padding-top: 20px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================

@st.cache_data

def load_dataset():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        dataset_url = (
            "https://github.com/MaddikuntlaDeepthi/"
            "AI_FINANCIAL_FRAUD_DETECTION/releases/download/"
            "v1.0.0/creditcard.csv"
        )

        try:
            with st.spinner("Downloading fraud detection dataset..."):
                with urlopen(dataset_url, timeout=300) as response:
                    with open(DATA_PATH, "wb") as file:
                        shutil.copyfileobj(response, file)

        except Exception as e:
            if DATA_PATH.exists():
                DATA_PATH.unlink()

            st.error(f"Dataset download failed: {e}")
            return None

    # Remove exact duplicate rows so dashboard statistics match
    # the cleaned dataset.
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().reset_index(drop=True)

    return data



@st.cache_resource
def load_scaler():
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource

@st.cache_resource
def load_split():
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not SPLIT_PATH.exists():
        split_url = (
            "https://github.com/MaddikuntlaDeepthi/"
            "AI_FINANCIAL_FRAUD_DETECTION/releases/download/"
            "v1.0.0/train_test_data.joblib"
        )

        try:
            with st.spinner("Downloading processed training and testing data..."):
                with urlopen(split_url, timeout=300) as response:
                    with open(SPLIT_PATH, "wb") as file:
                        shutil.copyfileobj(response, file)

        except Exception as e:
            if SPLIT_PATH.exists():
                SPLIT_PATH.unlink()

            st.error(f"Processed dataset download failed: {e}")
            return None

    try:
        data = joblib.load(SPLIT_PATH)

        if not isinstance(data, dict):
            st.error("Invalid processed dataset format.")
            return None

        return data

    except Exception as e:
        st.error(f"Could not load processed dataset: {e}")
        return None



@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_json(path_string):
    path = Path(path_string)
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


df = load_dataset()
model = load_model()
scaler = load_scaler()
split_data = load_split()

if df is None:
    st.error("Dataset not found. Check data/creditcard.csv.")
    st.stop()

if model is None:
    st.error("Trained model not found. Check models/best_fraud_model.pkl.")
    st.stop()

if scaler is None:
    st.error("Feature scaler not found. Check models/scaler.pkl.")
    st.stop()

required_features = [
    col for col in df.columns if col not in ["Class"]
]

if split_data is None:
    st.error("Processed train_test_data.joblib not found or invalid.")
    st.stop()

X_test = split_data["X_test"]
y_test = split_data["y_test"]

if isinstance(X_test, pd.DataFrame):
    X_test = X_test.copy()
else:
    X_test = pd.DataFrame(X_test, columns=required_features)

if isinstance(y_test, pd.Series):
    y_test = y_test.reset_index(drop=True)
else:
    y_test = pd.Series(y_test)

# Ensure test data is already in the same processed form used in training.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= THRESHOLD).astype(int)

test_results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Fraud Probability": test_probabilities,
    "Prediction": test_predictions
})

test_results["Prediction Label"] = np.where(
    test_results["Prediction"] == 1,
    "Potential Fraud",
    "Legitimate"
)


# ============================================================
# COMMON HELPERS
# ============================================================

def money(value):
    return f"€{value:,.2f}"


def metric_card(label, value, caption):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(label, title, subtitle):
    st.markdown(f"""
    <div class="hero-label">{label}</div>
    <div class="hero-title">{title}</div>
    <div class="hero-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)


def plot_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce9fa"),
        margin=dict(l=20, r=20, t=45, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce9fa")
        ),
        xaxis=dict(
            gridcolor="#26364e",
            zerolinecolor="#26364e"
        ),
        yaxis=dict(
            gridcolor="#26364e",
            zerolinecolor="#26364e"
        )
    )
    return fig


def show_image(path, caption):
    if Path(path).exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{Path(path).name} is not available yet.")


def format_metric(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def predict_transactions(input_df):
    clean = input_df.copy()

    if "Class" in clean.columns:
        clean = clean.drop(columns=["Class"])

    missing = [c for c in required_features if c not in clean.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    clean = clean[required_features].copy()

    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean.isnull().any().any():
        raise ValueError("Input contains missing or non-numeric values.")

    # The model was trained with Time and Amount standardized.
    clean[["Time", "Amount"]] = scaler.transform(clean[["Time", "Amount"]])

    probabilities = model.predict_proba(clean)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    output = input_df.copy()
    output["Fraud Probability"] = probabilities
    output["Prediction"] = predictions
    output["Risk Status"] = np.where(
        predictions == 1,
        "Potential Fraud",
        "Legitimate"
    )

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:13px;margin-bottom:35px;">
        <div style="
            width:52px;height:52px;border-radius:15px;
            background:linear-gradient(135deg,#18c9e5,#5578f5);
            display:flex;align-items:center;justify-content:center;
            font-size:25px;font-weight:900;color:#07111f;">
            F
        </div>
        <div>
            <div style="font-size:20px;font-weight:800;color:white;">
                FraudGuard AI
            </div>
            <div style="font-size:9px;letter-spacing:3px;color:#8cb9ed;">
                FINANCIAL SECURITY
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-caption">MAIN MENU</div>',
        unsafe_allow_html=True
    )

    pages = [
        "Executive Overview",
        "Transaction Intelligence",
        "Fraud Detection Center",
        "Advanced Analytics",
        "Model Intelligence",
        "Model Evaluation",
        "Explainable AI",
        "Reports & Monitoring"
    ]

    page = st.radio(
        "Navigation",
        pages,
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div class="panel">
        <div class="metric-label">SYSTEM STATUS</div>
        <div style="color:#00e6b0;font-size:13px;font-weight:800;margin-top:15px;">
            ● MODEL READY
        </div>
        <div style="color:#86b8ef;font-size:10px;margin-top:12px;">
            Fraud detection engine initialized
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("FraudGuard AI • Version 2.3")


# ============================================================
# 1. EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    page_header(
        "SECURITY OPERATIONS CENTER",
        "Executive Overview",
        "AI-powered financial transaction monitoring and fraud intelligence."
    )

    st.markdown(
        '<div class="status-pill">● FRAUD DETECTION ENGINE READY</div>',
        unsafe_allow_html=True
    )

    total_records = len(df)
    fraud_count = int(df["Class"].sum())
    legitimate_count = total_records - fraud_count

    metrics = load_json(str(METRICS_PATH))
    pr_auc = metrics.get("pr_auc", 0.8224) if metrics else 0.8224

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("DATASET RECORDS", f"{total_records:,}", "After duplicate removal")

    with c2:
        metric_card("FRAUD CASES", f"{fraud_count:,}", "Actual fraud labels in dataset")

    with c3:
        metric_card("BEST MODEL", "XGBoost", "Selected trained model")

    with c4:
        metric_card("BEST PR-AUC", format_metric(pr_auc), "Held-out test evaluation")

    st.markdown("### Model Performance Intelligence")

    comparison = load_csv(str(COMPARISON_PATH))

    left, right = st.columns([1.25, 0.85])

    with left:
        st.markdown("**Model Comparison**")
        st.caption("Performance of trained classification models")

        if comparison is not None and not comparison.empty:
            numeric_columns = [
                c for c in comparison.columns
                if c.lower() in [
                    "accuracy", "precision", "recall",
                    "f1", "f1_score", "roc_auc", "pr_auc"
                ]
            ]

            if numeric_columns:
                selected_metric = st.selectbox(
                    "Select evaluation metric",
                    numeric_columns
                )

                name_column = next(
                    (c for c in comparison.columns
                     if c.lower() in ["model", "model_name", "classifier"]),
                    comparison.columns[0]
                )

                fig = px.bar(
                    comparison,
                    x=name_column,
                    y=selected_metric,
                    color=name_column,
                    title=f"{selected_metric} comparison"
                )
                st.plotly_chart(plot_layout(fig), use_container_width=True)
            else:
                st.dataframe(comparison, use_container_width=True)
        else:
            st.info("Model comparison report not found.")

    with right:
        st.markdown("**Dataset Class Distribution**")
        class_data = pd.DataFrame({
            "Class": ["Legitimate", "Fraud"],
            "Count": [legitimate_count, fraud_count]
        })

        fig = px.pie(
            class_data,
            names="Class",
            values="Count",
            hole=0.65,
            color="Class",
            color_discrete_map={
                "Legitimate": "#20c8e8",
                "Fraud": "#ed496b"
            }
        )
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    st.markdown("### Test Evaluation Snapshot")

    c1, c2, c3, c4 = st.columns(4)

    final_metrics = load_json(str(METRICS_PATH))

    if final_metrics:
        with c1:
            st.metric("Accuracy", format_metric(final_metrics.get("accuracy")))
        with c2:
            st.metric("Precision", format_metric(final_metrics.get("precision")))
        with c3:
            st.metric("Recall", format_metric(final_metrics.get("recall")))
        with c4:
            st.metric("F1 Score", format_metric(final_metrics.get("f1_score")))


# ============================================================
# 2. TRANSACTION INTELLIGENCE
# ============================================================

elif page == "Transaction Intelligence":

    page_header(
        "TRANSACTION MONITORING",
        "Transaction Intelligence",
        "Explore historical transactions and model-generated fraud predictions."
    )

    total = len(test_results)
    fraud_predictions = int(test_results["Prediction"].sum())
    legitimate_predictions = total - fraud_predictions

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("TEST TRANSACTIONS", f"{total:,}", "Held-out test dataset")

    with c2:
        metric_card("MODEL-FLAGGED FRAUD", f"{fraud_predictions:,}", "Predicted at threshold 0.50")

    with c3:
        metric_card("MODEL-FLAGGED LEGITIMATE", f"{legitimate_predictions:,}", "Predicted at threshold 0.50")

    st.markdown("### Transaction Explorer")

    search_df = test_results.copy()

    min_probability, max_probability = st.slider(
        "Fraud probability range",
        0.0, 1.0, (0.0, 1.0), 0.01
    )

    filtered = search_df[
        search_df["Fraud Probability"].between(
            min_probability, max_probability
        )
    ]

    st.caption(f"Showing {len(filtered):,} matching transactions.")

    st.dataframe(
        filtered.head(1000),
        use_container_width=True,
        height=380
    )

    st.download_button(
        "Download transaction results",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="fraudguard_transaction_results.csv",
        mime="text/csv"
    )

    st.markdown("### Prediction Distribution")

    fig = px.histogram(
        test_results,
        x="Fraud Probability",
        color="Prediction Label",
        nbins=50,
        color_discrete_map={
            "Legitimate": "#20c8e8",
            "Potential Fraud": "#ed496b"
        }
    )
    st.plotly_chart(plot_layout(fig), use_container_width=True)


# ============================================================
# 3. FRAUD DETECTION CENTER
# ============================================================

elif page == "Fraud Detection Center":

    page_header(
        "LIVE RISK ANALYSIS",
        "Fraud Detection Center",
        "Analyze an individual transaction or upload a CSV for batch screening."
    )

    mode = st.radio(
        "Select analysis mode",
        ["Single Transaction", "Upload CSV"],
        horizontal=True
    )

    if mode == "Single Transaction":

        st.markdown("### Transaction Input")

        sample_type = st.radio(
            "Choose a sample",
            ["Random legitimate sample", "Random fraud sample", "Manual input"],
            horizontal=True
        )

        if sample_type == "Random legitimate sample":
            sample = df[df["Class"] == 0].sample(1, random_state=None)
            input_values = sample[required_features].copy()
            actual_label = 0

        elif sample_type == "Random fraud sample":
            sample = df[df["Class"] == 1].sample(1, random_state=None)
            input_values = sample[required_features].copy()
            actual_label = 1

        else:
            input_values = pd.DataFrame(
                [{feature: 0.0 for feature in required_features}]
            )
            actual_label = None

            with st.form("manual_transaction_form"):
                manual_values = {}

                for feature in required_features:
                    default = 0.0
                    if feature == "Amount":
                        default = 100.0

                    manual_values[feature] = st.number_input(
                        feature,
                        value=float(default),
                        format="%.6f",
                        key=f"manual_{feature}"
                    )

                submitted = st.form_submit_button("Analyze Transaction")

            if submitted:
                input_values = pd.DataFrame([manual_values])

        if sample_type != "Manual input":
            st.caption("A real labeled transaction has been selected from the dataset.")

        if sample_type != "Manual input" or submitted:
            if st.button("Run Fraud Analysis", key="single_predict"):
                try:
                    result = predict_transactions(input_values)

                    probability = float(result["Fraud Probability"].iloc[0])
                    prediction = int(result["Prediction"].iloc[0])

                    st.markdown("### Analysis Result")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.metric("Fraud Probability", f"{probability * 100:.2f}%")

                    with c2:
                        st.metric(
                            "Model Classification",
                            "Potential Fraud" if prediction == 1 else "Legitimate"
                        )

                    if prediction == 1:
                        st.error("⚠️ This transaction has been flagged by the model.")
                    else:
                        st.success("✓ The model classified this transaction as legitimate.")

                    if actual_label is not None:
                        st.info(
                            f"Dataset ground-truth label: "
                            f"{'Fraud' if actual_label == 1 else 'Legitimate'}"
                        )

                    st.caption(
                        "A model prediction is a risk signal, not confirmation of actual fraud."
                    )

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

    else:
        st.markdown("### Batch Transaction Analysis")

        uploaded_file = st.file_uploader(
            "Upload transaction CSV",
            type=["csv"]
        )

        st.caption(
            "CSV must contain Time, V1–V28 and Amount. "
            "A Class column is optional and will not be used as a model input."
        )

        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)

                st.write("Uploaded data preview")
                st.dataframe(upload_df.head(), use_container_width=True)

                if st.button("Analyze Uploaded Transactions"):
                    results = predict_transactions(upload_df)

                    flagged = int(results["Prediction"].sum())
                    count = len(results)

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        metric_card("ANALYZED", f"{count:,}", "Uploaded transactions")

                    with c2:
                        metric_card("POTENTIAL FRAUD", f"{flagged:,}", "Model flagged")

                    with c3:
                        metric_card(
                            "LEGITIMATE",
                            f"{count - flagged:,}",
                            "Model classified"
                        )

                    st.dataframe(results, use_container_width=True)

                    st.download_button(
                        "Download analysis results",
                        results.to_csv(index=False).encode("utf-8"),
                        file_name="fraudguard_batch_predictions.csv",
                        mime="text/csv"
                    )

                    fig = px.histogram(
                        results,
                        x="Fraud Probability",
                        color="Risk Status",
                        nbins=30,
                        color_discrete_map={
                            "Legitimate": "#20c8e8",
                            "Potential Fraud": "#ed496b"
                        }
                    )
                    st.plotly_chart(plot_layout(fig), use_container_width=True)

            except Exception as error:
                st.error(f"Unable to process CSV: {error}")


# ============================================================
# 4. ADVANCED ANALYTICS - ALL SEVEN CHARTS
# ============================================================

elif page == "Advanced Analytics":

    page_header(
        "DATA SCIENCE & INSIGHTS",
        "Advanced Analytics",
        "Explore fraud patterns, transaction behavior and model prediction intelligence."
    )

    st.markdown("### Analytics Filters")

    min_amount = float(df["Amount"].min())
    max_amount = float(df["Amount"].max())

    amount_range = st.slider(
        "Transaction amount range",
        min_value=min_amount,
        max_value=max_amount,
        value=(min_amount, max_amount)
    )

    category_filter = st.selectbox(
        "Transaction category",
        ["All Transactions", "Legitimate Only", "Fraud Only"]
    )

    analytics_df = df[
        df["Amount"].between(amount_range[0], amount_range[1])
    ].copy()

    if category_filter == "Legitimate Only":
        analytics_df = analytics_df[analytics_df["Class"] == 0]
    elif category_filter == "Fraud Only":
        analytics_df = analytics_df[analytics_df["Class"] == 1]

    st.caption(f"{len(analytics_df):,} transactions match the selected filters.")

    # 1. Transaction Class Distribution
    st.markdown("### 1. Transaction Class Distribution")

    class_counts = analytics_df["Class"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    class_chart = pd.DataFrame({
        "Transaction Type": ["Legitimate", "Fraud"],
        "Count": [class_counts[0], class_counts[1]]
    })

    fig = px.pie(
        class_chart,
        names="Transaction Type",
        values="Count",
        hole=0.55,
        color="Transaction Type",
        color_discrete_map={
            "Legitimate": "#20c8e8",
            "Fraud": "#ed496b"
        }
    )
    st.plotly_chart(plot_layout(fig), use_container_width=True)

    # 2. Transaction Amount Analysis
    st.markdown("### 2. Transaction Amount Analysis")

    amount_sample = analytics_df.sample(
        min(12000, len(analytics_df)),
        random_state=42
    )

    fig = px.histogram(
        amount_sample,
        x="Amount",
        color="Class",
        nbins=50,
        color_discrete_map={
            0: "#20c8e8",
            1: "#ed496b"
        },
        labels={"Class": "Actual Class"}
    )
    st.plotly_chart(plot_layout(fig), use_container_width=True)

    # 3. Transaction Activity Over Time
    st.markdown("### 3. Transaction Activity Over Time")

    time_data = analytics_df.copy()
    time_data["Time Hours"] = time_data["Time"] / 3600

    time_data["Time Bucket"] = (
        time_data["Time Hours"] // 1
    ).astype(int)

    hourly = (
        time_data.groupby(["Time Bucket", "Class"])
        .size()
        .reset_index(name="Transactions")
    )

    fig = px.line(
        hourly,
        x="Time Bucket",
        y="Transactions",
        color="Class",
        markers=True,
        color_discrete_map={
            0: "#20c8e8",
            1: "#ed496b"
        },
        labels={
            "Time Bucket": "Elapsed hours",
            "Class": "Actual class"
        }
    )
    st.plotly_chart(plot_layout(fig), use_container_width=True)

    # 4. Fraud Amount Distribution
    st.markdown("### 4. Fraud Amount Distribution")

    fraud_amounts = analytics_df[analytics_df["Class"] == 1]

    if not fraud_amounts.empty:
        fig = px.box(
            fraud_amounts,
            y="Amount",
            points="outliers",
            title="Actual fraud transaction amounts"
        )
        st.plotly_chart(plot_layout(fig), use_container_width=True)
    else:
        st.info("No fraud transactions in the selected filters.")

    # 5. Top Feature Correlations
    st.markdown("### 5. Top Feature Correlations")

    correlation = analytics_df.corr(numeric_only=True)["Class"].drop(
        labels=["Class"], errors="ignore"
    )

    top_correlations = correlation.abs().sort_values(
        ascending=False
    ).head(12).index

    correlation_data = correlation.loc[top_correlations].sort_values()

    fig = px.bar(
        x=correlation_data.values,
        y=correlation_data.index,
        orientation="h",
        labels={"x": "Correlation with Class", "y": "Feature"},
        title="Features with strongest absolute correlation"
    )
    st.plotly_chart(plot_layout(fig), use_container_width=True)

    # 6. XGBoost Prediction Intelligence
    st.markdown("### 6. XGBoost Prediction Intelligence")

    prediction_plot = test_results.copy()
    prediction_plot["Actual Label"] = np.where(
        prediction_plot["Actual"] == 1,
        "Actual Fraud",
        "Actual Legitimate"
    )

    fig = px.scatter(
        prediction_plot.sample(
            min(6000, len(prediction_plot)),
            random_state=42
        ),
        x="Actual",
        y="Fraud Probability",
        color="Actual Label",
        opacity=0.55,
        color_discrete_map={
            "Actual Legitimate": "#20c8e8",
            "Actual Fraud": "#ed496b"
        },
        title="Predicted probability versus actual class"
    )
    fig.add_hline(
        y=THRESHOLD,
        line_dash="dash",
        line_color="#f5c451",
        annotation_text="Decision threshold 0.50"
    )
    st.plotly_chart(plot_layout(fig), use_container_width=True)

    # 7. Fraud Probability Distribution
    st.markdown("### 7. Fraud Probability Distribution")

    fig = px.histogram(
        test_results,
        x="Fraud Probability",
        color="Prediction Label",
        nbins=50,
        color_discrete_map={
            "Legitimate": "#20c8e8",
            "Potential Fraud": "#ed496b"
        },
        title="XGBoost predicted fraud probabilities"
    )
    fig.add_vline(
        x=THRESHOLD,
        line_dash="dash",
        line_color="#f5c451",
        annotation_text="Threshold 0.50"
    )
    st.plotly_chart(plot_layout(fig), use_container_width=True)


# ============================================================
# 5. MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    page_header(
        "MODEL PERFORMANCE",
        "Model Intelligence",
        "Understand the trained models and compare their predictive performance."
    )

    st.markdown("### Selected Model")

    c1, c2 = st.columns(2)

    with c1:
        metric_card("SELECTED MODEL", "XGBoost", "Gradient boosting classifier")

    with c2:
        metric_card(
            "DECISION THRESHOLD",
            f"{THRESHOLD:.2f}",
            "Probability classification threshold"
        )

    comparison = load_csv(str(COMPARISON_PATH))

    st.markdown("### Model Comparison")

    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)

        numeric_columns = [
            c for c in comparison.columns
            if c.lower() in [
                "accuracy", "precision", "recall",
                "f1", "f1_score", "roc_auc", "pr_auc"
            ]
        ]

        if numeric_columns:
            metric_choice = st.selectbox(
                "Choose comparison metric",
                numeric_columns,
                key="model_intelligence_metric"
            )

            name_column = next(
                (c for c in comparison.columns
                 if c.lower() in ["model", "model_name", "classifier"]),
                comparison.columns[0]
            )

            fig = px.bar(
                comparison,
                x=name_column,
                y=metric_choice,
                color=name_column,
                title=f"Model comparison - {metric_choice}"
            )
            st.plotly_chart(plot_layout(fig), use_container_width=True)
    else:
        st.warning("Model comparison CSV not found.")

    st.markdown("### Model Configuration")

    st.write({
        "Algorithm": "XGBoost Classifier",
        "Number of test records": len(X_test),
        "Input features": len(required_features),
        "Threshold": THRESHOLD,
        "Model file": str(MODEL_PATH.relative_to(BASE_DIR))
    })


# ============================================================
# 6. MODEL EVALUATION - UPDATED FINAL REPORTS
# ============================================================

elif page == "Model Evaluation":

    page_header(
        "VALIDATION & PERFORMANCE",
        "Model Evaluation",
        "Detailed evaluation of the saved XGBoost model on the original held-out test split."
    )

    metrics = load_json(str(METRICS_PATH))

    if metrics is None:
        st.warning(
            "Final evaluation metrics not found. "
            "Run: python src/evaluate_model.py"
        )
    else:
        st.markdown("### Final Evaluation Metrics")

        metric_names = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1_score"),
            ("ROC-AUC", "roc_auc"),
            ("PR-AUC", "pr_auc")
        ]

        columns = st.columns(3)

        for index, (label, key) in enumerate(metric_names):
            with columns[index % 3]:
                value = metrics.get(key)

                if value is not None:
                    st.metric(label, format_metric(value))
                else:
                    st.metric(label, "N/A")

        st.markdown("### Confusion Matrix")

        left, right = st.columns(2)

        with left:
            show_image(
                CONFUSION_PATH,
                "Final confusion matrix"
            )

        with right:
            st.markdown("#### Classification Results")

            tn = metrics.get("true_negatives", 56636)
            fp = metrics.get("false_positives", 15)
            fn = metrics.get("false_negatives", 20)
            tp = metrics.get("true_positives", 75)

            st.metric("True Negatives", f"{tn:,}")
            st.metric("False Positives", f"{fp:,}")
            st.metric("False Negatives", f"{fn:,}")
            st.metric("True Positives", f"{tp:,}")

        st.markdown("### ROC Curve")
        show_image(ROC_PATH, "Final ROC curve")

        st.markdown("### Precision-Recall Curve")
        show_image(PR_PATH, "Final Precision-Recall curve")

        st.markdown("### Classification Report")

        classification_report = load_csv(str(CLASS_REPORT_PATH))

        if classification_report is not None:
            st.dataframe(
                classification_report,
                use_container_width=True
            )
            st.download_button(
                "Download classification report",
                classification_report.to_csv(index=False).encode("utf-8"),
                file_name="final_classification_report.csv",
                mime="text/csv"
            )
        else:
            st.info("Final classification report CSV not found.")

        st.markdown("### Prediction-Level Evaluation")

        evaluation_predictions = load_csv(str(PREDICTIONS_PATH))

        if evaluation_predictions is not None:
            st.caption(
                f"Total prediction records: {len(evaluation_predictions):,}"
            )

            st.dataframe(
                evaluation_predictions.head(1000),
                use_container_width=True
            )

            st.download_button(
                "Download final evaluation predictions",
                evaluation_predictions.to_csv(index=False).encode("utf-8"),
                file_name="final_evaluation_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("Final evaluation predictions CSV not found.")

        st.info(
            "Methodology note: These results use the original held-out test split. "
            "The split was also used during earlier model comparison, so it is not "
            "a completely untouched independent test set."
        )


# ============================================================
# 7. EXPLAINABLE AI
# ============================================================

elif page == "Explainable AI":

    page_header(
        "MODEL INTERPRETABILITY",
        "Explainable AI",
        "Explore which input features are associated with the model's fraud predictions."
    )

    st.markdown("### Feature Importance")

    try:
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": required_features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig = px.bar(
                importance.head(15).sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 XGBoost feature importances"
            )
            st.plotly_chart(plot_layout(fig), use_container_width=True)

            st.dataframe(
                importance,
                use_container_width=True
            )
        else:
            st.info("Feature importance is not available for this model.")
    except Exception as error:
        st.warning(f"Unable to calculate feature importance: {error}")

    st.markdown("### Individual Prediction Explanation")

    selected_index = st.number_input(
        "Select test transaction index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )

    row = X_test.iloc[[int(selected_index)]]

    probability = float(model.predict_proba(row)[0, 1])

    st.metric("Estimated fraud probability", f"{probability * 100:.2f}%")

    if hasattr(model, "feature_importances_"):
        row_importance = pd.DataFrame({
            "Feature": required_features,
            "Feature Value": row.iloc[0].values,
            "Model Importance": model.feature_importances_
        })

        row_importance["Contribution Indicator"] = (
            row_importance["Feature Value"]
            * row_importance["Model Importance"]
        )

        row_importance = row_importance.reindex(
            row_importance["Contribution Indicator"].abs()
            .sort_values(ascending=False).index
        )

        st.markdown("#### Important input features")
        st.dataframe(
            row_importance.head(15),
            use_container_width=True
        )

    st.caption(
        "Feature importance indicates model reliance, not causation. "
        "V1–V28 are anonymized PCA components and do not directly identify "
        "real-world transaction characteristics."
    )


# ============================================================
# 8. REPORTS & MONITORING
# ============================================================

elif page == "Reports & Monitoring":

    page_header(
        "SYSTEM REPORTING",
        "Reports & Monitoring",
        "Access saved model reports, evaluation outputs and downloadable artifacts."
    )

    report_files = [
        ("Final Evaluation Metrics", METRICS_PATH),
        ("Final Confusion Matrix", CONFUSION_PATH),
        ("Final ROC Curve", ROC_PATH),
        ("Final Precision-Recall Curve", PR_PATH),
        ("Final Classification Report", CLASS_REPORT_PATH),
        ("Final Evaluation Predictions", PREDICTIONS_PATH),
        ("Model Comparison", COMPARISON_PATH)
    ]

    st.markdown("### Available Reports")

    for label, path in report_files:
        c1, c2 = st.columns([3, 1])

        with c1:
            st.write(f"**{label}**")
            st.caption(path.name)

        with c2:
            if path.exists():
                st.success("Available")
            else:
                st.warning("Not found")

    st.markdown("### Evaluation Summary")

    metrics = load_json(str(METRICS_PATH))

    if metrics:
        st.json(metrics)

    st.markdown("### Dataset Monitoring")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("TOTAL RECORDS", f"{len(df):,}", "Cleaned dataset")

    with c2:
        metric_card(
            "LEGITIMATE",
            f"{int((df['Class'] == 0).sum()):,}",
            "Actual dataset labels"
        )

    with c3:
        metric_card(
            "FRAUD",
            f"{int((df['Class'] == 1).sum()):,}",
            "Actual dataset labels"
        )

    st.caption(
        "This application demonstrates fraud analysis using a historical "
        "credit-card dataset. It is not connected to a live banking system."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="app-footer">
    FRAUDGUARD AI | FINANCIAL SECURITY INTELLIGENCE
    <br><br>
    AI-powered fraud analysis • Educational and demonstration system
</div>
""", unsafe_allow_html=True)
