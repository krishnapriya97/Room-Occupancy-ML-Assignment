
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Room Occupancy Prediction",
    page_icon="🏠",
    layout="wide"
)

# Application title
st.title("Room Occupancy Prediction")
st.write(
    "This application predicts the number of people "
    "in a room using environmental sensor data."
)

# Load saved files
MODEL_DIR = "models"

models = {
    "Logistic Regression": joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression.pkl")
    ),
    "Decision Tree": joblib.load(
        os.path.join(MODEL_DIR, "decision_tree.pkl")
    ),
    "Random Forest": joblib.load(
        os.path.join(MODEL_DIR, "random_forest.pkl")
    ),
    "SVM": joblib.load(
        os.path.join(MODEL_DIR, "svm.pkl")
    ),
    "KNN": joblib.load(
        os.path.join(MODEL_DIR, "knn.pkl")
    )
}

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

# Model selection
selected_model_name = st.selectbox(
    "Select a Machine Learning Model",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.subheader("Upload Sensor Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    input_data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview")
    st.dataframe(input_data.head())

    # Check target availability
    target_column = "Room_Occupancy_Count"

    has_target = target_column in input_data.columns

    # Remove unnecessary columns
    input_features = input_data.drop(
        columns=["Date", "Time", target_column],
        errors="ignore"
    )

    # Check required features
    missing_features = [
        col for col in feature_columns
        if col not in input_features.columns
    ]

    if missing_features:

        st.error(
            "Missing required sensor features: "
            + ", ".join(missing_features)
        )

    elif not all(
        pd.api.types.is_numeric_dtype(input_features[col])
        for col in feature_columns
    ):

        st.error("All sensor features must contain numeric values.")

    elif input_features[feature_columns].isnull().any().any():

        st.error("Please remove missing values from the sensor data.")

    else:

        X_input = input_features[feature_columns]

        # Apply scaling where required
        if selected_model_name in [
            "Logistic Regression", "SVM", "KNN"
        ]:
            X_model = scaler.transform(X_input)
        else:
            X_model = X_input

        # Generate predictions
        predictions = selected_model.predict(X_model)

        st.subheader("Prediction Results")

        results = input_data.copy()
        results["Predicted_Occupancy"] = predictions

        st.dataframe(results.head(20))

        st.download_button(
            "Download Predictions",
            results.to_csv(index=False).encode("utf-8"),
            "occupancy_predictions.csv",
            "text/csv"
        )

        # Evaluate if actual target is available
        if has_target:

            y_actual = input_data[target_column]

            if not y_actual.isin([0, 1, 2, 3]).all():
                st.error("Target values must be between 0 and 3.")

            else:

                st.subheader("Model Evaluation Metrics")

                accuracy = accuracy_score(y_actual, predictions)
                precision = precision_score(
                    y_actual, predictions,
                    average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_actual, predictions,
                    average="weighted", zero_division=0
                )
                f1 = f1_score(
                    y_actual, predictions,
                    average="weighted", zero_division=0
                )
                mcc = matthews_corrcoef(y_actual, predictions)

                auc = None

                if hasattr(selected_model, "predict_proba"):
                    probabilities = selected_model.predict_proba(X_model)

                    auc = roc_auc_score(
                        y_actual,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                        labels=selected_model.classes_
                    )

                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Precision", f"{precision:.4f}")
                col3.metric("Recall", f"{recall:.4f}")

                col4, col5, col6 = st.columns(3)

                col4.metric("F1 Score", f"{f1:.4f}")
                col5.metric("MCC", f"{mcc:.4f}")
                col6.metric(
                    "AUC",
                    f"{auc:.4f}" if auc is not None else "N/A"
                )

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_actual,
                    predictions,
                    labels=[0, 1, 2, 3]
                )

                fig, ax = plt.subplots()

                display = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=[0, 1, 2, 3]
                )

                display.plot(
                    ax=ax,
                    cmap="Blues",
                    colorbar=False
                )

                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Classification Report")

                report = classification_report(
                    y_actual,
                    predictions,
                    labels=[0, 1, 2, 3],
                    zero_division=0
                )

                st.text(report)

        else:

            st.info(
                "Actual occupancy values are not available. "
                "Predictions are displayed without evaluation metrics."
            )
