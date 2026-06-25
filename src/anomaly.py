"""Detect anomalous commits using a trained classifier.

Loads `features.csv` and a pickled classifier (default: `models/codedna_model.pkl`),
computes predicted labels and prediction probabilities, and flags anomalies where
the model's confidence is below a threshold or the predicted label differs from
the recorded label.

Outputs a CSV (`anomalies.csv` by default) with columns: commit_hash, label,
predicted_label, predicted_prob, is_anomaly, reason
"""
import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def detect_anomalies(df: pd.DataFrame, model, threshold: float = 0.5):
    feature_cols = [c for c in df.columns if c not in ('commit_hash', 'label')]
    X = df[feature_cols].fillna(0)

    # Predict labels and probabilities
    try:
        probs = model.predict_proba(X)
        classes = list(model.classes_)
    except Exception:
        # Model may not support predict_proba
        preds = model.predict(X)
        probs = None
        classes = None

    results = []
    for i, row in df.iterrows():
        commit = row.get('commit_hash', '')
        true_label = row.get('label', '')

        if probs is None:
            pred_label = preds[i]
            pred_prob = ''
        else:
            class_idx = int(np.argmax(probs[i]))
            pred_label = classes[class_idx]
            pred_prob = float(probs[i][class_idx])

        is_anom = False
        reasons = []
        if pred_label != true_label:
            is_anom = True
            reasons.append('predicted_label_mismatch')
        if probs is not None and pred_prob < threshold:
            is_anom = True
            reasons.append(f'low_confidence<{threshold}')

        results.append({
            'commit_hash': commit,
            'label': true_label,
            'predicted_label': pred_label,
            'predicted_prob': pred_prob,
            'is_anomaly': is_anom,
            'reason': ';'.join(reasons) if reasons else '',
        })

    return pd.DataFrame(results)


def parse_args():
    p = argparse.ArgumentParser(description='Detect anomalous commits using trained model')
    p.add_argument('--features', '-f', default='features.csv', help='Input features CSV')
    p.add_argument('--model', '-m', default='models/codedna_model.pkl', help='Pickled model file')
    p.add_argument('--output', '-o', default='anomalies.csv', help='Output anomalies CSV')
    p.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.features):
        print(f"Features file not found: {args.features}")
        return
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    df = pd.read_csv(args.features)
    model = load_model(args.model)

    anomalies_df = detect_anomalies(df, model, threshold=args.threshold)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    anomalies_df.to_csv(args.output, index=False)
    print(f"Wrote anomalies to {args.output} (count={len(anomalies_df[anomalies_df['is_anomaly']])})")


if __name__ == '__main__':
    main()
# Load models/codedna_model.pkl and features.csv
# For each commit, predict the author and compare with actual author label
# If predicted author != actual author, flag it as anomalous
# Print a list of flagged suspicious commits with their anomaly score
# Save flagged commits to flagged_commits.csv