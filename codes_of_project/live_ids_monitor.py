"""
live_ids_monitor.py  (v2 — detailed per-flow report)

Watches the folder where CICFlowMeter writes its live-capture CSV, and for every
NEW row that appears, runs it through the trained IDS pipeline
(Isolation Forest -> XGBoost multiclass) and prints a detailed report:
  - Normal / ANOMALY status
  - full probability breakdown across attack classes (zero-probability classes skipped)
  - source/destination IP + port for that flow

Run this on the WINDOWS VM (the victim), in the same folder as:
  feature_columns.pkl, hybrid_feature_columns.pkl, isolation_forest_model.pkl,
  xgboost_multiclass_model.pkl, label_encoder.pkl

Requires: pandas, numpy, joblib, scikit-learn, xgboost
  pip install pandas numpy joblib scikit-learn xgboost
"""

import glob
import os
import time

import joblib
import numpy as np
import pandas as pd

# ---- CONFIG ----
MODEL_DIR = r"."                            # folder containing the .pkl files
CICFLOW_OUTPUT_DIR = r"C:\cicflow_output"   # folder where CICFlowMeter writes its live CSV
POLL_SECONDS = 3
MIN_PCT_TO_SHOW = 0.01                      # hide classes rounding to 0.00%
BENIGN_LABEL = "Benign"                     # must match label_encoder's spelling for the normal class

# ---- load model artifacts once ----
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
hybrid_path = os.path.join(MODEL_DIR, "hybrid_feature_columns.pkl")
use_if_score = os.path.exists(hybrid_path)
hybrid_cols = joblib.load(hybrid_path) if use_if_score else feature_cols
if_model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_model.pkl")) if use_if_score else None
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_multiclass_model.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
class_names = list(le.classes_)

flow_counter = 0


def build_features(df_flows: pd.DataFrame) -> pd.DataFrame:
    df_flows.columns = df_flows.columns.str.strip()
    missing = [c for c in feature_cols if c not in df_flows.columns]
    if missing:
        raise ValueError(f"CICFlowMeter output is missing expected columns: {missing[:5]} ...")

    X = df_flows[feature_cols].replace([np.inf, -np.inf], np.nan)

    if use_if_score:
        X = X.copy()
        X["if_anomaly_score"] = if_model.decision_function(X[feature_cols]).astype("float32")
        X = X[hybrid_cols]
    return X


def get_conn_info(row: pd.Series) -> str:
    src = row.get("Src IP", row.get("Source IP", "?"))
    src_port = row.get("Src Port", row.get("Source Port", ""))
    dst = row.get("Dst IP", row.get("Destination IP", "?"))
    dst_port = row.get("Dst Port", row.get("Destination Port", ""))
    src_str = f"{src}:{src_port}" if src_port not in ("", None) else str(src)
    dst_str = f"{dst}:{dst_port}" if dst_port not in ("", None) else str(dst)
    return f"{src_str} -> {dst_str}"


def print_flow_report(conn_info: str, proba_row: np.ndarray):
    global flow_counter
    flow_counter += 1

    top_idx = int(proba_row.argmax())
    top_class = class_names[top_idx]
    status = "NORMAL" if top_class == BENIGN_LABEL else "ANOMALY DETECTED"

    # Build sorted, non-zero probability table (attack classes only, Benign shown separately)
    pct_pairs = [(class_names[i], proba_row[i] * 100) for i in range(len(class_names))]
    pct_pairs.sort(key=lambda p: p[1], reverse=True)

    print("\n" + "=" * 60)
    print(f"Flow #{flow_counter}  |  {conn_info}")
    print(f"Status: {status}")
    if status == "ANOMALY DETECTED":
        print("Attack probability breakdown:")
        shown_any = False
        for name, pct in pct_pairs:
            if name == BENIGN_LABEL:
                continue
            if pct < MIN_PCT_TO_SHOW:
                continue
            print(f"  {name:<28} {pct:6.2f}%")
            shown_any = True
        if not shown_any:
            print("  (no attack class scored above 0%)")
    print("=" * 60)


def predict_and_report(df_flows: pd.DataFrame):
    X = build_features(df_flows)
    proba = model.predict_proba(X)

    for i in range(len(df_flows)):
        conn_info = get_conn_info(df_flows.iloc[i])
        print_flow_report(conn_info, proba[i])


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
                predict_and_report(new_rows)
            except ValueError as e:
                print(f"[schema error] {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
