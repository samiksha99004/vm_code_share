"""
Dashboard-ready inference: for each flow, outputs either "Normal" or "Probably <attack type>"
with a confidence score - this is the function your IDS dashboard should call per flow.

Design decisions made here (so you can explain them in your report):
1. Majority attack types (DDoS, DoS-Hulk, Bot, SSH-Bruteforce, DoS-GoldenEye, DoS-Slowloris,
   DDoS-LOIC-UDP) already hit ~99-100% recall/precision from the trained model - no special
   handling needed, they're reported as-is.
2. A few rare classes (Brute Force-Web, DoS-SlowHTTPTest, FTP-BruteForce) are unreliable
   because of tiny training sample sizes - Brute Force-Web in particular has only 19%
   precision (mostly false alarms). Instead of hiding this or retraining on too little data,
   this script applies a CONFIDENCE THRESHOLD: if the model's top prediction is one of these
   shaky classes AND its probability is below MIN_CONFIDENCE_FOR_RARE_CLASS, the dashboard
   shows "Possible Attack - Unconfirmed" instead of confidently naming the wrong attack.
   This cuts down false alarms without silently hiding a real detection.
"""

import pandas as pd
import numpy as np
import joblib
import os

INPUT_DIR = r"F:\all_10_csv_manual"
FEATURES_FILE = os.path.join(INPUT_DIR, "feature_columns.pkl")
HYBRID_FEATURES_FILE = os.path.join(INPUT_DIR, "hybrid_feature_columns.pkl")
IF_MODEL_FILE = os.path.join(INPUT_DIR, "isolation_forest_model.pkl")
XGB_MODEL_FILE = os.path.join(INPUT_DIR, "xgboost_multiclass_model.pkl")
LABEL_ENCODER_FILE = os.path.join(INPUT_DIR, "label_encoder.pkl")

# classes that were unreliable in evaluation (low precision or too few training rows)
LOW_CONFIDENCE_CLASSES = {"Brute Force -Web", "DoS attacks-SlowHTTPTest", "FTP-BruteForce"}
MIN_CONFIDENCE_FOR_RARE_CLASS = 0.60  # tune this based on your dashboard's false-alarm tolerance

feature_cols = joblib.load(FEATURES_FILE)
use_if_score = os.path.exists(HYBRID_FEATURES_FILE)
hybrid_cols = joblib.load(HYBRID_FEATURES_FILE) if use_if_score else feature_cols
if_model = joblib.load(IF_MODEL_FILE) if use_if_score else None
model = joblib.load(XGB_MODEL_FILE)
le = joblib.load(LABEL_ENCODER_FILE)
class_names = list(le.classes_)


def predict_flows(df_flows: pd.DataFrame) -> pd.DataFrame:
    """
    df_flows: DataFrame containing at least the raw feature_cols columns (one row per flow).
    Returns a DataFrame with one row per input flow: prediction, confidence, status.
    """
    X = df_flows[feature_cols].replace([np.inf, -np.inf], np.nan)

    if use_if_score:
        X = X.copy()
        X["if_anomaly_score"] = if_model.decision_function(X[feature_cols]).astype("float32")
        X = X[hybrid_cols]

    proba = model.predict_proba(X)          # shape: (n_rows, n_classes)
    top_idx = proba.argmax(axis=1)
    top_class = np.array(class_names)[top_idx]
    top_conf = proba[np.arange(len(proba)), top_idx]

    statuses = []
    displayed_labels = []
    for cls, conf in zip(top_class, top_conf):
        if cls == "Benign":
            statuses.append("Normal")
            displayed_labels.append("Normal")
        elif cls in LOW_CONFIDENCE_CLASSES and conf < MIN_CONFIDENCE_FOR_RARE_CLASS:
            statuses.append("Possible Attack - Unconfirmed")
            displayed_labels.append("Possible Attack - Unconfirmed")
        else:
            statuses.append("Attack Detected")
            displayed_labels.append(f"Probably: {cls}")

    return pd.DataFrame({
        "predicted_class": top_class,
        "confidence": top_conf.round(4),
        "status": statuses,
        "dashboard_label": displayed_labels,
    })


if __name__ == "__main__":
    # quick demo on a handful of rows from test_dataset.csv
    sample = pd.read_csv(os.path.join(INPUT_DIR, "test_dataset.csv"), nrows=20)
    sample.columns = sample.columns.str.strip()
    result = predict_flows(sample)
    result["actual_label"] = sample["Label"].values
    print(result.to_string(index=False))
