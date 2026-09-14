"""
Evaluate the MULTICLASS hybrid model (Isolation Forest score + XGBoost) on the full
merged_final.csv, streaming in chunks to avoid memory errors. Excludes Infiltration,
matching the model's training.

Note: like the binary full-dataset run, this file includes rows the model already saw
during training (merged_final.csv = train_dataset.csv + test_dataset.csv combined), so
these numbers will look better than true generalization performance. Use this run for the
per-attack-type breakdown on the complete dataset; use the test-only run for the honest
headline number in your report.
"""

import pandas as pd
import numpy as np
import joblib
import os
import time

INPUT_DIR = r"F:\all_10_csv_manual"
DATA_FILE = os.path.join(INPUT_DIR, "merged_final.csv")

FEATURES_FILE = os.path.join(INPUT_DIR, "feature_columns.pkl")
HYBRID_FEATURES_FILE = os.path.join(INPUT_DIR, "hybrid_feature_columns.pkl")
IF_MODEL_FILE = os.path.join(INPUT_DIR, "isolation_forest_model.pkl")
XGB_MODEL_FILE = os.path.join(INPUT_DIR, "xgboost_multiclass_model.pkl")
LABEL_ENCODER_FILE = os.path.join(INPUT_DIR, "label_encoder.pkl")

REPORT_TXT = os.path.join(INPUT_DIR, "multiclass_full_dataset_report.txt")
REPORT_CSV = os.path.join(INPUT_DIR, "multiclass_per_attack_type.csv")

CHUNK_SIZE = 200_000
EXCLUDED_LABELS = ["Infilteration"]

feature_cols = joblib.load(FEATURES_FILE)
use_if_score = os.path.exists(HYBRID_FEATURES_FILE)
hybrid_cols = joblib.load(HYBRID_FEATURES_FILE) if use_if_score else feature_cols
if_model = joblib.load(IF_MODEL_FILE) if use_if_score else None
model = joblib.load(XGB_MODEL_FILE)
le = joblib.load(LABEL_ENCODER_FILE)
class_names = list(le.classes_)
n_classes = len(class_names)
print(f"Loaded multiclass model. Classes ({n_classes}): {class_names}")

usecols = list(dict.fromkeys(feature_cols + ["Label"]))

cm = np.zeros((n_classes, n_classes), dtype="int64")  # rows=actual, cols=predicted
unseen_label_rows = 0
excluded_rows_skipped = 0
rows_done = 0

print("Streaming merged_final.csv in chunks...")
start = time.time()
for chunk in pd.read_csv(DATA_FILE, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        if col == "Label":
            continue
        if chunk[col].dtype == "float64":
            chunk[col] = chunk[col].astype("float32")
        elif chunk[col].dtype == "int64":
            chunk[col] = chunk[col].astype("int32")

    for lbl in EXCLUDED_LABELS:
        keep = chunk["Label"] != lbl
        excluded_rows_skipped += (~keep).sum()
        chunk = chunk[keep]

    known_mask = chunk["Label"].isin(class_names)
    unseen_label_rows += (~known_mask).sum()
    chunk = chunk[known_mask]
    if len(chunk) == 0:
        continue

    raw_label = chunk["Label"]
    y_true_chunk = le.transform(raw_label)

    X_chunk = chunk[feature_cols].replace([np.inf, -np.inf], np.nan)
    if use_if_score:
        scores = if_model.decision_function(X_chunk).astype("float32")
        X_chunk = X_chunk.copy()
        X_chunk["if_anomaly_score"] = scores
        X_chunk = X_chunk[hybrid_cols]

    y_pred_chunk = model.predict(X_chunk)

    # accumulate into confusion matrix
    np.add.at(cm, (y_true_chunk, y_pred_chunk), 1)

    rows_done += len(chunk)
    print(f"  processed {rows_done:,} rows...", end="\r")

print(f"\nDone streaming in {time.time() - start:.1f}s. Rows used: {rows_done:,} "
      f"(excluded {excluded_rows_skipped:,} Infiltration rows, "
      f"skipped {unseen_label_rows:,} rows with unseen labels)")

# ---------- per-class precision/recall/F1 from the confusion matrix ----------
support = cm.sum(axis=1)
predicted_totals = cm.sum(axis=0)
tp = np.diag(cm)

precision = np.divide(tp, predicted_totals, out=np.zeros(n_classes), where=predicted_totals != 0)
recall = np.divide(tp, support, out=np.zeros(n_classes), where=support != 0)
f1 = np.divide(2 * precision * recall, precision + recall,
               out=np.zeros(n_classes), where=(precision + recall) != 0)

accuracy = tp.sum() / cm.sum()
macro_f1 = f1.mean()
weighted_f1 = (f1 * support).sum() / support.sum()

rows = []
for i, name in enumerate(class_names):
    rows.append({
        "label": name,
        "total": int(support[i]),
        "detected_correctly": int(tp[i]),
        "missed": int(support[i] - tp[i]),
        "precision": round(precision[i], 4),
        "recall": round(recall[i], 4),
        "f1": round(f1[i], 4),
    })
per_class_df = pd.DataFrame(rows).sort_values("recall")

# ---------- print + save ----------
lines = []
lines.append(f"MULTICLASS FULL DATASET EVALUATION - {DATA_FILE}")
lines.append(f"Total rows evaluated: {rows_done}")
lines.append(f"Accuracy: {accuracy:.4f}")
lines.append(f"Macro F1: {macro_f1:.4f}   Weighted F1: {weighted_f1:.4f}\n")
lines.append("Per-attack-type detection (sorted worst -> best by recall):")
lines.append(per_class_df.to_string(index=False))
lines.append("\nConfusion Matrix (rows=actual, cols=predicted):")
lines.append(f"Order: {class_names}")
lines.append(str(cm))

report_str = "\n".join(lines)
print("\n" + report_str)

with open(REPORT_TXT, "w") as f:
    f.write(report_str)
per_class_df.to_csv(REPORT_CSV, index=False)

print(f"\nText report saved to: {REPORT_TXT}")
print(f"Per-attack-type CSV saved to: {REPORT_CSV}")
