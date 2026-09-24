# Flood Risk Predictor (India) — ML Web App

A Flask web app that predicts flood risk for **any location**, not just one city.
You enter rainfall, elevation, water level, land cover, soil type, etc., and a
Random Forest model (trained on 10,000 real-format records across India) returns
a Low / Moderate / High risk band with a probability.

This mirrors the structure of the reference site you shared
(rainfall + elevation + drainage + land-use + historical-flood data → risk level),
but is not limited to Mumbai — it works on the whole dataset's feature set.

---

## 1. Project structure

```
flood-risk-predictor/
├── data/
│   └── flood_risk_dataset_india.csv   # your dataset (10,000 rows)
├── model/                             # created after training (model + encoders)
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── train_model.py                     # trains & saves the ML model
├── app.py                             # Flask web app
├── requirements.txt
└── README.md
```

## 2. Open it in VS Code

1. Unzip the project folder you downloaded.
2. In VS Code: **File → Open Folder →** select `flood-risk-predictor`.
3. Open a terminal in VS Code: **Terminal → New Terminal**.

## 3. Set up a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

VS Code may prompt "Select interpreter" — pick the `venv` one.

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Train the model

```bash
python train_model.py
```

This reads `data/flood_risk_dataset_india.csv`, encodes the categorical columns
(Land Cover, Soil Type), trains a `RandomForestClassifier`, and saves:
- `model/flood_model.pkl`
- `model/scaler.pkl`
- `model/encoders.pkl`
- `model/features.pkl`

It also prints accuracy and a feature-importance list in the terminal — useful
if this is for a college project report/PBL writeup.

## 6. Run the web app

```bash
python app.py
```

Open the link it prints — normally **http://127.0.0.1:5000** — in your browser.
Fill the form and click **Predict Flood Risk**.

---

## 7. About the dataset — read this before you present it

`flood_risk_dataset_india.csv` (from Kaggle) has these columns:

| Column | Meaning |
|---|---|
| Latitude, Longitude | Location |
| Rainfall (mm) | Recent/seasonal rainfall |
| Temperature (°C), Humidity (%) | Weather |
| River Discharge (m³/s) | Volume of water flowing in nearby river |
| Water Level (m) | Current river/water body level |
| Elevation (m) | Land elevation |
| Land Cover | Water Body / Forest / Agricultural / Desert / Urban |
| Soil Type | Clay / Peat / Loam / Sandy / Silt |
| Population Density | People per unit area |
| Infrastructure | 1 = flood defenses/drainage present, 0 = none |
| Historical Floods | 1 = area has flooded before, 0 = no record |
| Flood Occurred | Target: 1 = flood, 0 = no flood |

**Important honesty check:** when I trained on this file, accuracy came out
around **51%** — essentially a coin flip. I checked, and the values in this
particular CSV are randomly generated per row (there's no real statistical
relationship between rainfall/elevation/etc. and the flood outcome). That's
common with several of the "flood risk" datasets on Kaggle — they're meant for
practicing the ML *pipeline*, not for a model that's actually predictive.

So the code and app are fully working, but if you need a model that performs
meaningfully better than chance for a real report, you have two options:

- **Use real government data** instead of/alongside this file — e.g. IMD
  (India Meteorological Department) rainfall data, CWC (Central Water
  Commission) flood/water-level bulletins, or your state disaster management
  authority's historical flood records. These have real cause-effect signal.
- **Keep this dataset for the pipeline/demo** (it's fine for a PBL/mini-project
  showing you can build and deploy the ML + web app), but say explicitly in
  your report that it's a synthetic dataset and note the accuracy honestly —
  examiners generally respect that more than a suspiciously perfect number.

I'm happy to help you swap in a real dataset (e.g. an IMD district-rainfall +
CWC flood-records dataset) if you want a version with genuine predictive power
— just say the word.

## 8. Ideas to extend it (optional, good for the PBL writeup)

- Add a map (Leaflet.js) so users click a point instead of typing lat/long.
- Try `XGBoostClassifier` or `LogisticRegression` and compare accuracy (ties
  in nicely with the "Multiclass Classification" / "Decision Tree" practicals
  in your index).
- Add k-fold cross-validation in `train_model.py` (you already have that as
  practical #6) and report the average accuracy instead of a single split.
- Deploy it (Render, Railway, PythonAnywhere) like the reference site you linked.
