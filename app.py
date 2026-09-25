"""
Flood Risk Predictor
====================
This single file contains:
- the Flask web app for the final mini-project
- the practical exercises for the journal
- the shared dataset processing functions
- the model training and evaluation functions

Run the app:
    python app.py

Run all AI/ML practicals from Python:
    from app import run_all_practicals
    run_all_practicals()
"""

import os
from collections import deque

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_risk_dataset_india.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURE_COLUMNS = [
    "Latitude",
    "Longitude",
    "Rainfall (mm)",
    "Temperature (°C)",
    "Humidity (%)",
    "River Discharge (m³/s)",
    "Water Level (m)",
    "Elevation (m)",
    "Land Cover",
    "Soil Type",
    "Population Density",
    "Infrastructure",
    "Historical Floods",
]
BINARY_TARGET = "Flood Occurred"
CATEGORICAL_COLUMNS = ["Land Cover", "Soil Type"]


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset and remove missing rows."""
    df = pd.read_csv(path)
    return df.dropna().copy()


def encode_categorical_columns(
    df: pd.DataFrame,
    columns: list[str] = CATEGORICAL_COLUMNS,
):
    """Encode categorical columns using LabelEncoder and return the encoders."""
    encoded = df.copy()
    encoders = {}
    for col in columns:
        if col in encoded.columns:
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col].astype(str))
            encoders[col] = le
    return encoded, encoders


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Shared train/test split for all experiments."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(pd.unique(y)) > 1 else None,
    )


def scale_features(X_train, X_test):
    """Scale numeric features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# Practical 1: BFS / DFS / A*
def print_grid(grid):
    for row in grid:
        print(" ".join(str(cell) for cell in row))
    print()


def reconstruct_path(parent, start, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    if path and path[0] != start:
        return []
    return path


def bfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    return reconstruct_path(parent, start, goal)


def dfs(grid, start, goal):
    stack = [start]
    visited = {start}
    parent = {start: None}

    while stack:
        r, c = stack.pop()
        if (r, c) == goal:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                stack.append((nr, nc))
    return reconstruct_path(parent, start, goal)


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = {start}
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = min(open_set, key=lambda pos: f_score.get(pos, float("inf")))
        if current == goal:
            break
        open_set.remove(current)

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                neighbor = (nr, nc)
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    open_set.add(neighbor)
    return reconstruct_path(came_from, start, goal)


def practical_01_search():
    print("Practical 1: AI Search (BFS, DFS, A*)")
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    start = (0, 0)
    goal = (4, 4)
    print_grid(grid)
    print(f"Start: {start} | Goal: {goal}\n")
    for name, algo in [("BFS", bfs), ("DFS", dfs), ("A*", astar)]:
        path = algo(grid, start, goal)
        print(f"{name}: path = {path}")
        print(f"{name}: steps = {len(path) - 1 if path else -1}\n")


# Practical 2: Bayes Rule + Naive Bayes
def practical_02_bayes():
    print("\nPractical 2: Bayes' Rule and Naive Bayes")
    P_A = 0.30
    P_B_given_A = 0.80
    P_B_given_not_A = 0.25
    P_not_A = 0.70
    P_B = (P_B_given_A * P_A) + (P_B_given_not_A * P_not_A)
    P_A_given_B = (P_B_given_A * P_A) / P_B
    print("Bayes' Rule worked example (using the Law of Total Probability):")
    print(f"P(A) = {P_A}")
    print(f"P(~A) = {P_not_A}")
    print(f"P(B|A) = {P_B_given_A}")
    print(f"P(B|~A) = {P_B_given_not_A}")
    print(f"P(B) = P(B|A)·P(A) + P(B|~A)·P(~A) = {P_B:.4f}")
    print(f"P(A|B) = P(B|A)·P(A) / P(B) = {P_A_given_B:.4f}\n")

    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Gaussian Naive Bayes Results:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("First 5 predicted probabilities:")
    for row in model.predict_proba(X_test[:5]):
        print(np.round(row, 4))
    print()


# Practical 3: ML Basics
def practical_03_ml_basics():
    print("\nPractical 3: ML Basics")
    df = load_dataset()
    df, encoders = encode_categorical_columns(df)
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

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

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["No Flood", "Flood"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "flood_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "features.pkl"))
    print(f"Model saved to {MODEL_DIR}")
    print()


# Practical 4: Linear Regression
def practical_04_linear_regression():
    print("\nPractical 4: Linear Regression")
    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    target = "Water Level (m)"
    feature_cols = [col for col in FEATURE_COLUMNS if col not in {target, BINARY_TARGET}]
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("Top coefficients:")
    coeffs = pd.Series(model.coef_, index=feature_cols).sort_values(ascending=False)
    print(coeffs.head(8))
    print()


# Practical 5: Logistic Regression
def practical_05_logistic_regression():
    print("\nPractical 5: Logistic Regression")
    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    print(classification_report(y_test, preds, target_names=["No Flood", "Flood"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("First 5 probabilities:")
    for row in model.predict_proba(X_test[:5]):
        print(row)
    print()


# Practical 6: Cross-validation
def practical_06_cross_validation():
    print("\nPractical 6: Cross-validation")
    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET]

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        print(f"{name}: {scores}")
        print(f"Mean = {scores.mean():.4f}, Std = {scores.std():.4f}\n")


# Practical 7: Decision Tree Classifier
def practical_07_decision_tree_classifier():
    print("\nPractical 7: Decision Tree Classifier")
    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("Feature importances:")
    for feature, importance in sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: x[1], reverse=True)[:8]:
        print(f"{feature}: {importance:.4f}")
    print("Decision tree text view:")
    print(tree.export_text(model, feature_names=FEATURE_COLUMNS, max_depth=3))
    print()


# Practical 8: Decision Tree Regression
def practical_08_decision_tree_regression():
    print("\nPractical 8: Decision Tree Regression")
    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    target = "Water Level (m)"
    feature_cols = [col for col in FEATURE_COLUMNS if col not in {target, BINARY_TARGET}]
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("Feature importances:")
    for feature, importance in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)[:8]:
        print(f"{feature}: {importance:.4f}")
    print()


# Practical 9: kNN Classifier
def practical_09_knn():
    print("\nPractical 9: kNN Classifier")
    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    best_model = None
    best_k = None
    best_score = -1.0

    for k in [3, 5, 7, 9, 11]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        print(f"k = {k} -> accuracy = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    print(f"Best k = {best_k} with accuracy {best_score:.4f}")
    print(classification_report(y_test, best_model.predict(X_test_scaled), target_names=["No Flood", "Flood"]))
    print()


# Practical 10: Multiclass Classification
def build_multiclass_target(df: pd.DataFrame) -> pd.Series:
    """Create a genuine 3-class target from rainfall, water level, and discharge."""
    composite = (
        df["Rainfall (mm)"]
        + 1.5 * df["Water Level (m)"]
        + 0.3 * df["River Discharge (m³/s)"]
    )
    q1, q2 = composite.quantile([0.33, 0.66])
    labels = []
    for value in composite:
        if value <= q1:
            labels.append("Low")
        elif value <= q2:
            labels.append("Moderate")
        else:
            labels.append("High")
    return pd.Series(labels, index=df.index)


def practical_10_multiclass():
    print("\nPractical 10: Multiclass Classification")
    df = load_dataset()
    df, _ = encode_categorical_columns(df)
    df["Risk Class"] = build_multiclass_target(df)
    X = df[FEATURE_COLUMNS]
    y = df["Risk Class"]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds, labels=["Low", "Moderate", "High"]))
    print()


def run_all_practicals():
    print("Flood Risk Predictor - Practical Journal Runner")
    print("=" * 60)
    practical_01_search()
    practical_02_bayes()
    practical_03_ml_basics()
    practical_04_linear_regression()
    practical_05_logistic_regression()
    practical_06_cross_validation()
    practical_07_decision_tree_classifier()
    practical_08_decision_tree_regression()
    practical_09_knn()
    practical_10_multiclass()
    print("\nAll practicals completed.")


# Flask app for the PBL mini-project
app = Flask(__name__)

try:
    model = joblib.load(os.path.join(MODEL_DIR, "flood_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    FEATURES = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    LAND_COVER_OPTIONS = list(encoders["Land Cover"].classes_)
    SOIL_TYPE_OPTIONS = list(encoders["Soil Type"].classes_)
except (FileNotFoundError, OSError, ValueError, AttributeError):
    model = None
    scaler = None
    encoders = None
    FEATURES = FEATURE_COLUMNS
    LAND_COVER_OPTIONS = []
    SOIL_TYPE_OPTIONS = []


def risk_label(probability):
    """Convert probability to a human-readable flood risk band."""
    if probability < 0.35:
        return "Low", "low"
    elif probability < 0.65:
        return "Moderate", "moderate"
    else:
        return "High", "high"


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        land_cover_options=LAND_COVER_OPTIONS,
        soil_type_options=SOIL_TYPE_OPTIONS,
        result=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or encoders is None:
        return render_template(
            "index.html",
            land_cover_options=LAND_COVER_OPTIONS,
            soil_type_options=SOIL_TYPE_OPTIONS,
            result={
                "probability": 0.0,
                "label": "Low",
                "css_class": "low",
                "inputs": request.form,
                "warning": "Train the model first by running practical_03_ml_basics() or python train_model.py",
            },
        )

    form = request.form
    row = {
        "Latitude": float(form["latitude"]),
        "Longitude": float(form["longitude"]),
        "Rainfall (mm)": float(form["rainfall"]),
        "Temperature (°C)": float(form["temperature"]),
        "Humidity (%)": float(form["humidity"]),
        "River Discharge (m³/s)": float(form["river_discharge"]),
        "Water Level (m)": float(form["water_level"]),
        "Elevation (m)": float(form["elevation"]),
        "Land Cover": encoders["Land Cover"].transform([form["land_cover"]])[0],
        "Soil Type": encoders["Soil Type"].transform([form["soil_type"]])[0],
        "Population Density": float(form["population_density"]),
        "Infrastructure": int(form["infrastructure"]),
        "Historical Floods": int(form["historical_floods"]),
    }

    X = pd.DataFrame([row], columns=FEATURES)
    X_scaled = scaler.transform(X)
    probability = float(model.predict_proba(X_scaled)[0][1])
    label, css_class = risk_label(probability)

    result = {
        "probability": round(probability * 100, 1),
        "label": label,
        "css_class": css_class,
        "inputs": form,
    }

    return render_template(
        "index.html",
        land_cover_options=LAND_COVER_OPTIONS,
        soil_type_options=SOIL_TYPE_OPTIONS,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
