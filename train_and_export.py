"""
CardioML Model Training & Artifact Export Pipeline
Compliant with Darshan University MLDL SOP Project Specification (Weeks 3-5 & 7).
Trains a GradientBoostingClassifier on the cleaned cardiovascular dataset,
evaluates performance metrics, and exports clean pickle and metadata artifacts.
"""

import os
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def train_and_export():
    dataset_path = "cardio_cleaned.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cleaned dataset '{dataset_path}' not found!")

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")

    # Feature definitions
    continuous_cols = ['age_years', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'pulse_pressure']
    feature_order = [
        'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active',
        'age_years', 'bmi', 'pulse_pressure'
    ]

    # Verify columns
    X = df[feature_order].copy()
    y = df['cardio'].copy()

    # Train / Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale continuous columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # SOP specified hyperparameters (from MLDL SOP page 8):
    # Estimators: 300, Learning Rate: 0.05, Max Depth: 4, Min Samples / Leaf: 3
    print("Training GradientBoostingClassifier with SOP hyperparameters...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba))

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS (Test Set):")
    print(f"Accuracy : {acc*100:.2f}% (SOP Target: 73.1% - 73.4%)")
    print(f"F1 Score : {f1*100:.2f}% (SOP Target: 71.7%)")
    print(f"ROC-AUC  : {roc_auc*100:.2f}% (SOP Target: 79.7% - 80.1%)")
    print("="*50 + "\n")

    # Feature importances
    importances = model.feature_importances_
    feat_imp = {}
    for col, imp in zip(feature_order, importances):
        feat_imp[col] = round(float(imp) * 100, 2)
    # Sort descending
    sorted_feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

    metadata = {
        "model_name": "GradientBoostingClassifier",
        "library": "scikit-learn",
        "trained_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "total_records": len(df),
        "feature_count": len(feature_order),
        "features": feature_order,
        "continuous_features": continuous_cols,
        "hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_leaf": 3,
            "random_state": 42
        },
        "performance": {
            "accuracy": round(acc * 100, 1),
            "f1_score": round(f1 * 100, 1),
            "roc_auc": round(roc_auc * 100, 1)
        },
        "feature_importance": sorted_feat_imp
    }

    # Save artifacts in root and in cardio_frontend
    target_dirs = [".", "cardio_frontend"]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(d, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(d, "feature_order.pkl"), "wb") as f:
            pickle.dump(feature_order, f)
        with open(os.path.join(d, "model_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Exported artifacts successfully to directory: '{d}'")

    print("\nTraining and export completed successfully!")

if __name__ == "__main__":
    train_and_export()
