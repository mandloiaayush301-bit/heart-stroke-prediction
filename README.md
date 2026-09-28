# Heart Stroke Prediction

## Project Overview
**Heart Stroke Prediction** is an interactive Machine Learning web application built with **Streamlit** and **scikit-learn** that estimates whether a patient is at **High Risk** or **Low Risk** of heart disease/stroke based on 11 clinical and physiological input parameters.

The repository contains the inference web application (`app.py`) along with the pre-trained serialized model and preprocessing artifacts (`Logistic_heart.pkl`, `scaler.pkl`, and `columns.pkl`).

---

## Problem Statement
Cardiovascular diseases and heart stroke risks depend on multiple interrelated clinical indicators such as blood pressure, cholesterol levels, electrocardiogram (ECG) readings, maximum heart rate, and ST depression during exercise. Manually assessing how these combined indicators map to overall risk can be error-prone. A fast, accessible predictive interface helps demonstrate how supervised classification models evaluate patient risk profiles from structured clinical measurements.

---

## Objective
- Provide an easy-to-use **Streamlit** web interface for entering patient cardiovascular parameters.
- Transform raw user inputs into the exact one-hot encoded 15-feature schema expected by the trained model.
- Standardize features using a pre-fitted `StandardScaler` and classify heart disease risk using a trained `LogisticRegression` model.

---

## Machine Learning Approach
1. **Feature Encoding & Alignment**: Categorical inputs collected from the UI are converted into one-hot encoded indicator columns (`<Feature>_<Category>`) and aligned with the 15-column schema stored in `columns.pkl` (missing dummy columns are filled with `0`).
2. **Feature Scaling**: All 15 ordered features are standardized using the pre-fitted `StandardScaler` (`scaler.pkl`).
3. **Binary Classification**: The standardized feature vector is passed to the pre-trained `LogisticRegression` classifier (`Logistic_heart.pkl`), which outputs a binary prediction:
   - `1`: **High Risk of Heart Disease**
   - `0`: **Low Risk of Heart Disease**

---

## Dataset Description
> **Note:** The raw training dataset file (e.g., CSV) and training notebook are not included in this repository; only the trained model and preprocessing artifacts derived from the training data are stored.

Based on inspection of the serialized `StandardScaler` (`scaler.pkl`) and feature schema (`columns.pkl`):
- **Training Samples Seen (`n_samples_seen_`)**: `615` records
- **Input Feature Schema**: Standard Heart Disease / Failure clinical attributes (11 raw attributes expanding to 15 model features after one-hot encoding with drop-first baseline categories).

---

## Features Used

### Raw Clinical Inputs Collected in `app.py`
| Feature | UI Input Type | Range / Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Age** | Slider | `18` – `100` | `40` | Patient age in years |
| **Sex** | Selectbox | `M`, `F` | `M` | Biological sex |
| **Chest Pain Type** | Selectbox | `ATA`, `NAP`, `TA`, `ASY` | `ATA` | Type of chest pain experienced |
| **Resting Blood Pressure** | Number Input | `80` – `200` mm Hg | `120` | Resting blood pressure (`RestingBP`) |
| **Cholesterol** | Number Input | `100` – `600` mg/dL | `200` | Serum cholesterol (`Cholesterol`) |
| **Fasting Blood Sugar > 120 mg/dL** | Selectbox | `0`, `1` | `0` | `1` if fasting blood sugar > 120 mg/dL, else `0` (`FastingBS`) |
| **Resting ECG** | Selectbox | `Normal`, `ST`, `LVH` | `Normal` | Resting electrocardiogram results |
| **Max Heart Rate** | Slider | `60` – `220` | `150` | Maximum heart rate achieved (`MaxHR`) |
| **Exercise-Induced Angina** | Selectbox | `Y`, `N` | `Y` | Presence of exercise-induced angina |
| **Oldpeak (ST Depression)** | Slider | `0.0` – `6.0` | `1.0` | ST depression induced by exercise relative to rest (`Oldpeak`) |
| **ST Slope** | Selectbox | `Up`, `Flat`, `Down` | `Up` | Slope of the peak exercise ST segment |

### Encoded Model Features (`columns.pkl`)
The model expects the following 15 ordered features:
1. `Age`
2. `RestingBP`
3. `Cholesterol`
4. `FastingBS`
5. `MaxHR`
6. `Oldpeak`
7. `Sex_M`
8. `ChestPainType_ATA`
9. `ChestPainType_NAP`
10. `ChestPainType_TA`
11. `RestingECG_Normal`
12. `RestingECG_ST`
13. `ExerciseAngina_Y`
14. `ST_Slope_Flat`
15. `ST_Slope_Up`

---

## Data Preprocessing
Within `app.py`, user inputs undergo the following preprocessing pipeline before inference:
1. **Dictionary Construction**: Numeric values (`Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`) and active categorical flags (`Sex_<sex>`, `ChestPainType_<chest_pain>`, `RestingECG_<resting_ecg>`, `ExerciseAngina_<exercise_angina>`, `ST_Slope_<st_slope>`) are assembled into a single-row pandas `DataFrame`.
2. **Missing Column Imputation**: Any column in `columns.pkl` not present in the current input row is initialized to `0`.
3. **Column Ordering**: The `DataFrame` columns are reordered to strictly match `expected_columns` from `columns.pkl`.
4. **Standardization**: `scaler.transform(input_df)` standardizes the 15 features using the pre-computed means and standard deviations in `scaler.pkl`.

---

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) and training notebooks are not bundled in this deployment repository. The repository focuses on serving the pre-trained model artifacts via the Streamlit interface.

---

## Model(s) Used
- **Estimator**: `sklearn.linear_model.LogisticRegression` (serialized in `Logistic_heart.pkl`)
- **Hyperparameters**: `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `fit_intercept=True`, `max_iter=100`
- **Target Classes**: `[0, 1]` (`0` = Low Risk of Heart Disease, `1` = High Risk of Heart Disease)

---

## Model Evaluation Metrics & Results
Training/validation split evaluation metrics (such as accuracy, precision, recall, F1-score, or ROC-AUC) are not stored inside the serialized `.pkl` files or source code in this repository. At runtime, the application outputs a direct class prediction:
- **`1` ->** `⚠️ High Risk of Heart Disease`
- **`0` ->** `✅ Low Risk of Heart Disease`

---

## Project Structure
```text
heart-stroke-prediction/
│
├── app.py                 # Streamlit web application for heart stroke/disease risk prediction
├── Logistic_heart.pkl     # Pre-trained scikit-learn LogisticRegression model
├── scaler.pkl             # Pre-fitted scikit-learn StandardScaler
├── columns.pkl            # List of 15 expected feature column names in exact order
├── requirements.txt       # Required Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore rules (excludes .venv, cache, and temporary files)
```

---

## Technologies Used
- **Python 3**
- **Streamlit** (`streamlit`) — Interactive web UI
- **Pandas** (`pandas`) — Tabular input construction and column alignment
- **Joblib** (`joblib`) — Deserializing `.pkl` model and preprocessing artifacts
- **Scikit-learn** (`scikit-learn`) — `LogisticRegression` model and `StandardScaler`

---

## Installation Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mandloiaayush301-bit/heart-stroke-prediction.git
   cd heart-stroke-prediction
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Project

Run the Streamlit application from the project root directory:

```bash
streamlit run app.py
```

Streamlit will start a local web server (typically at `http://localhost:8501`) and open the application in your default web browser.

---

## Example Usage
1. Launch the app using `streamlit run app.py`.
2. Adjust the patient clinical parameters in the UI:
   - Set **Age** (e.g., `40`), **Sex** (`M` or `F`), and **Chest Pain Type** (`ATA`, `NAP`, `TA`, or `ASY`).
   - Enter **Resting Blood Pressure** (e.g., `120` mm Hg), **Cholesterol** (e.g., `200` mg/dL), and **Fasting Blood Sugar > 120 mg/dL** (`0` or `1`).
   - Select **Resting ECG**, **Max Heart Rate**, **Exercise-Induced Angina**, **Oldpeak**, and **ST Slope**.
3. Click the **Predict** button.
4. View the classification banner:
   - `✅ Low Risk of Heart Disease` or `⚠️ High Risk of Heart Disease`.

---

## Future Improvements
- Include the original training notebook and dataset to document EDA and benchmark comparisons across multiple classifiers (e.g., Random Forest, XGBoost, SVM).
- Display prediction probabilities (`model.predict_proba()`) alongside the binary risk classification.
- Add model explainability (e.g., feature contribution / SHAP plots) for individual predictions.
- Pin exact `scikit-learn==1.6.1` version in deployment environments to match the version used when serializing the `.pkl` artifacts.

---

## Disclaimer
> **Medical Disclaimer:** This project is created strictly for **educational and machine learning demonstration purposes**. It is **not** a certified medical device and must **not** be used for actual clinical diagnosis, medical decision-making, or treatment. Always consult a qualified healthcare professional for medical advice.
