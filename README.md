# Flood Risk Predictor — AI and Application Development Practical Journal

This project contains a complete mini-project for a BSc IT practical journal based on the topic: Flood Risk Predictor.

It includes:
- 10 machine learning / AI practical scripts
- one shared helper file for consistent data loading and preprocessing
- a Flask web app for the final mini-project
- a project runner that executes all practicals in one go
- honest documentation about the nature of the dataset

## Project output

![Flood Signal dashboard output](static/project-output.png)

The screenshot shows the desktop dashboard layout with the location search, rainfall trend area, map context, and prediction workspace.

---

## Project structure

```text
flood-risk-predictor/
├── app.py                              # Flask web app for flood risk prediction
├── common.py                           # Shared helper for all practicals
├── train_model.py                      # Trained binary model pipeline
├── run_all.py                          # Runs practicals 1 to 10 in sequence
├── practical_01_search.py              # BFS, DFS, A*
├── practical_02_bayes.py               # Bayes rule + Naive Bayes
├── practical_03_ml_basics.py           # Data prep + model training + save artifacts
├── practical_04_linear_regression.py   # Predict Water Level
├── practical_05_logistic_regression.py # Binary classification
├── practical_06_cross_validation.py   # 5-fold CV comparison
├── practical_07_decision_tree_classifier.py
├── practical_08_decision_tree_regression.py
├── practical_09_knn.py                # k-nearest neighbours
├── practical_10_multiclass.py         # Low / Moderate / High classes
├── data/
│   └── flood_risk_dataset_india.csv
├── model/
│   ├── flood_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── features.pkl
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Practical list

1. Intro to AI / Search: BFS, DFS, A*
2. Bayes Rule / Naive Bayes
3. ML Basics: preprocessing, train/test split, scaling, RandomForest, save model
4. Linear Regression: predict Water Level
5. Binary Classifier: Logistic Regression
6. k-fold Cross Validation: RandomForest and LogisticRegression
7. Decision Tree Classifier
8. Decision Tree Regression
9. kNN Classifier
10. Multiclass Classification: Low / Moderate / High
11. PBL Mini Project: Flask app with prediction form

---

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Run all practicals

```bash
python run_all.py
```

---

## Run the web app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## About the dataset

The project uses the file `data/flood_risk_dataset_india.csv` with columns such as:
- Latitude
- Longitude
- Rainfall (mm)
- Temperature (°C)
- Humidity (%)
- River Discharge (m³/s)
- Water Level (m)
- Elevation (m)
- Land Cover
- Soil Type
- Population Density
- Infrastructure
- Historical Floods
- Flood Occurred

This dataset is best understood as a synthetic or random-practice dataset rather than a real-world flood forecasting dataset. That means the model may show low or near-chance accuracy. In the report, this should be stated honestly instead of presenting the results as real predictive power.

This is important for a college project because it demonstrates the ML workflow correctly while being transparent about the quality of the dataset.

---

## Honest note for faculty / viva

The app and scripts are working and the project demonstrates all the required practicals, but the dataset itself may not contain true causal patterns between the input features and flood occurrence. Accuracy therefore may not be high. That is not a weakness of the implementation; it is an honest outcome of the data used.

If you want a more realistic predictive model later, the next step would be to replace this dataset with real hydrological and meteorological data from government sources or a verified flood dataset.
