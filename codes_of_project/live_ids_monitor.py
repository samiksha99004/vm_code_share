"""
live_ids_monitor.py

Watches the folder where CICFlowMeter writes its live-capture CSV, and for every
NEW row that appears, runs it through your trained IDS pipeline
(Isolation Forest -> XGBoost multiclass) and prints the result to the terminal.

Run this on the WINDOWS VM (the one being brute-forced), in the same folder as:
  feature_columns.pkl, hybrid_feature_columns.pkl, isolation_forest_model.pkl,
  xgboost_multiclass_model.pkl, label_encoder.pkl

Requires: pandas, numpy, joblib   (pip install pandas numpy joblib)

USAGE:
  1. Start CICFlowMeter in live-capture mode pointed at your network interface,
     writing to CICFLOW_OUTPUT_DIR below (CICFlowMeter flushes flow rows to CSV
     periodically as flows complete/time out).
  2. Run:  python live_ids_monitor.py
  3. From Kali, run Hydra against the Windows SSH service.
  4. Watch this terminal - it prints one line per new flow it sees.
"""

import glob
import os
import time

import joblib
import numpy as np
import pandas as pd

# ---- CONFIG: edit these two paths for your machine ----
MODEL_DIR = r"."                      # folder containing the .pkl files (default: current folder)
CICFLOW_OUTPUT_DIR = r"C:\cicflow_output"   # folder where CICFlowMeter writes its live CSV(s)
POLL_SECONDS = 3                      # how often to check for new rows

LOW_CONFIDENCE_CLASSES = {"Brute Force -Web", "DoS attacks-SlowHTTPTest", "FTP-BruteForce"}
MIN_CONFIDENCE_FOR_RARE_CLASS = 0.60

# ---- load model artifacts once ----
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
hybrid_path = os.path.join(MODEL_DIR, "hybrid_feature_columns.pkl")
use_if_score = os.path.exists(hybrid_path)
hybrid_cols = joblib.load(hybrid_path) if use_if_score else feature_cols
if_model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_model.pkl")) if use_if_score else None
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_multiclass_model.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
class_names = list(le.classes_)


def predict_flows(df_flows: pd.DataFrame) -> pd.DataFrame:
    df_flows.columns = df_flows.columns.str.strip()
    missing = [c for c in feature_cols if c not in df_flows.columns]
    if missing:
        raise ValueError(f"CICFlowMeter output is missing expected columns: {missing[:5]} ...")

    X = df_flows[feature_cols].replace([np.inf, -np.inf], np.nan)

    if use_if_score:
        X = X.copy()
        X["if_anomaly_score"] = if_model.decision_function(X[feature_cols]).astype("float32")
        X = X[hybrid_cols]

    proba = model.predict_proba(X)
    top_idx = proba.argmax(axis=1)
    top_class = np.array(class_names)[top_idx]
    top_conf = proba[np.arange(len(proba)), top_idx]

    labels = []
    for cls, conf in zip(top_class, top_conf):
        if cls == "Benign":
            labels.append(("Normal", cls, conf))
        elif cls in LOW_CONFIDENCE_CLASSES and conf < MIN_CONFIDENCE_FOR_RARE_CLASS:
            labels.append(("Possible Attack - Unconfirmed", cls, conf))
        else:
            labels.append((f"ATTACK DETECTED -> Probably: {cls}", cls, conf))

    out = df_flows.copy()
    out["dashboard_label"] = [l[0] for l in labels]
    out["predicted_class"] = [l[1] for l in labels]
    out["confidence"] = [round(l[2], 4) for l in labels]
    return out


def find_latest_csv():
    files = glob.glob(os.path.join(CICFLOW_OUTPUT_DIR, "*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    print(f"Watching {CICFLOW_OUTPUT_DIR} for new flows... (Ctrl+C to stop)")
    rows_seen = 0
    current_file = None

    while True:
        latest = find_latest_csv()
        if latest is None:
            time.sleep(POLL_SECONDS)
            continue

        if latest != current_file:
            current_file = latest
            rows_seen = 0
            print(f"\n[watching file] {current_file}")

        try:
            df = pd.read_csv(current_file)
        except Exception:
            time.sleep(POLL_SECONDS)
            continue

        if len(df) > rows_seen:
            new_rows = df.iloc[rows_seen:].reset_index(drop=True)
            rows_seen = len(df)

            try:
                result = predict_flows(new_rows)
            except ValueError as e:
                print(f"[schema error] {e}")
                time.sleep(POLL_SECONDS)
                continue

            for i, row in result.iterrows():
                src = row.get("Src IP", row.get("Source IP", "?"))
                dst = row.get("Dst IP", row.get("Destination IP", "?"))
                print(f"Flow: {src} -> {dst} | {row['dashboard_label']} (conf={row['confidence']})")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
