"""
app.py
------
Flask web app for the Flood Risk Predictor.
Loads the model trained by train_model.py and serves a form where
a user enters conditions for any location and gets a flood-risk
prediction back.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

model = joblib.load(os.path.join(MODEL_DIR, "flood_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
FEATURES = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))

LAND_COVER_OPTIONS = list(encoders["Land Cover"].classes_)
SOIL_TYPE_OPTIONS = list(encoders["Soil Type"].classes_)


def risk_label(probability):
    """Turn a flood probability into a human-readable risk band."""
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

    X = np.array([[row[f] for f in FEATURES]])
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
