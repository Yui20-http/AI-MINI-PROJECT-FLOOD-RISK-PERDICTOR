"""
train_model.py
----------------
Trains a flood-risk classifier on data/flood_risk_dataset_india.csv
and saves the fitted model + encoders into the model/ folder.

Run this once before starting app.py:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_risk_dataset_india.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURES = [
    "Latitude", "Longitude", "Rainfall (mm)", "Temperature (°C)",
    "Humidity (%)", "River Discharge (m³/s)", "Water Level (m)",
    "Elevation (m)", "Land Cover", "Soil Type", "Population Density",
    "Infrastructure", "Historical Floods",
]
TARGET = "Flood Occurred"
CATEGORICAL = ["Land Cover", "Soil Type"]


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.dropna()

    # Encode categorical columns
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features (helps some models / future swaps)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, target_names=["No Flood", "Flood"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    # Feature importance (useful for the README / report)
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "flood_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))
    print(f"\nSaved model + scaler + encoders to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
