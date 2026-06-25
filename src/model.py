"""
Read features.csv and train a Random Forest classifier to predict author_name.
Evaluate with precision, recall, F1 score per author.
Save trained model to models/codedna_model.pkl.
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd


def load_features(csv_path: str) -> tuple:
    """Load features.csv and return X, y, and feature names."""
    df = pd.read_csv(csv_path)
    
    # Extract label column
    y = df['label']
    
    # Feature columns: all except commit_hash and label
    feature_cols = [col for col in df.columns if col not in ['commit_hash', 'label']]
    X = df[feature_cols].fillna(0)
    
    return X, y, feature_cols


def train_model(X, y, test_size=0.2, random_state=42):
    """Train a Random Forest classifier and return model, test split, and test predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    return clf, (X_train, X_test, y_train, y_test), y_pred


def evaluate_model(y_test, y_pred):
    """Print evaluation metrics: precision, recall, F1 per author."""
    print("\n" + "="*70)
    print("Classification Report:")
    print("="*70)
    print(classification_report(y_test, y_pred))
    
    # Per-class metrics
    print("\n" + "="*70)
    print("Per-Author Metrics:")
    print("="*70)
    
    labels = sorted(np.unique(y_test))
    for label in labels:
        precision = precision_score(y_test, y_pred, labels=[label], zero_division=0)
        recall = recall_score(y_test, y_pred, labels=[label], zero_division=0)
        f1 = f1_score(y_test, y_pred, labels=[label], zero_division=0)
        
        print(f"\n{label}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")


def save_model(clf, output_path: str):
    """Save trained model to pickle file."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Random Forest to predict author from commit features")
    p.add_argument('--features', '-f', default='features.csv', help='Input features CSV file')
    p.add_argument('--model', '-m', default='models/codedna_model.pkl', help='Output model pickle file')
    p.add_argument('--test-size', type=float, default=0.2, help='Test set size ratio')
    p.add_argument('--random-seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading features from {args.features}...")
    X, y, feature_names = load_features(args.features)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of authors: {len(y.unique())}")
    print(f"Features: {', '.join(feature_names)}")
    
    print("\nTraining Random Forest classifier...")
    clf, (X_train, X_test, y_train, y_test), y_pred = train_model(X, y, test_size=args.test_size, random_state=args.random_seed)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    evaluate_model(y_test, y_pred)
    
    save_model(clf, args.model)


if __name__ == '__main__':
    main()