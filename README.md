# Industrial Predictive Maintenance & Failure Analysis

A portfolio project that demonstrates data preparation, exploratory analysis, machine learning, model evaluation, SQL analytics, and an interactive dashboard for industrial sensor data.

> **Data note:** The included generator creates synthetic demonstration data so the project runs without downloading a dataset. Results on synthetic data are for demonstrating the workflow only and must not be presented as real-world performance. For a stronger portfolio, replace it with a documented public predictive-maintenance dataset and rerun the pipeline.

## Features

- Generate a reproducible sensor dataset (temperature, vibration, rotational speed, tool wear, operating type, and failure label).
- Clean and validate data with Pandas.
- Train a scikit-learn classification pipeline.
- Evaluate using precision, recall, F1-score, ROC-AUC, confusion matrix, and a classification report.
- Save the trained model and evaluation outputs.
- Explore sensor patterns and model predictions in a Streamlit dashboard.
- Includes SQL schema and example analytics queries.

## Project structure

```text
industrial-predictive-maintenance/
├── app.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── sql/
    ├── schema.sql
    └── analysis_queries.sql
```

## Quick start

Use Python 3.10 or 3.11 recommended.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Generate demo data

```bash
python generate_data.py
```

This creates `data/machine_sensor_data.csv`.

### 4. Train and evaluate the model

```bash
python train_model.py
```

Outputs:
- `models/failure_pipeline.joblib`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `reports/feature_importance.png`

### 5. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens in your browser. It shows dataset summaries, failure rates, sensor distributions, model metrics, and a form for scoring a hypothetical machine reading.

## Model approach

- **Target:** `failure` (0 = no failure, 1 = failure)
- **Features:** `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`, `machine_type`
- **Model:** preprocessing pipeline (median imputation + one-hot encoding) and Random Forest classifier
- **Validation:** stratified train/test split with fixed random seed
- **Metrics:** precision, recall, F1-score, ROC-AUC, confusion matrix

In maintenance use cases, false negatives can be costly because a failure is missed. Do not select a model based only on accuracy; discuss recall, precision, and the trade-off with maintenance teams.

## Use real data for a stronger portfolio

Replace the generated CSV with a properly licensed public predictive-maintenance dataset. Keep the same column names or update `FEATURES` in `train_model.py` and `app.py`. Document the source, license, data limitations, and any preprocessing. Do not claim real operational savings or production readiness without evidence.

## GitHub upload

```bash
git init
git add .
git commit -m "Build industrial predictive maintenance analytics project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/industrial-predictive-maintenance.git
git push -u origin main
```

Create the empty repository on GitHub first. Replace `YOUR-USERNAME` with your GitHub username.

## Interview talking points

1. Why is recall important when detecting rare machine failures?
2. Why use a preprocessing pipeline rather than manually encoding categories?
3. Why report F1-score and ROC-AUC in addition to accuracy?
4. How would you prevent data leakage if records were collected over time?
5. How would you monitor model drift and validate predictions before real deployment?

## Limitations

This is an educational prototype. It is not connected to real equipment, does not issue maintenance instructions, and is not a validated safety-critical system.
