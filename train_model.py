"""Train and evaluate an industrial failure classifier."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/machine_sensor_data.csv")
MODEL_PATH = Path("models/failure_pipeline.joblib")
REPORT_DIR = Path("reports")
NUMERIC_FEATURES = [
    "air_temperature_k", "process_temperature_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min"
]
CATEGORICAL_FEATURES = ["machine_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "failure"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run `python generate_data.py` first.")

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing_cols = sorted(set(required) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "test_rows": int(len(y_test)),
        "failure_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "data_note": "Synthetic demonstration data; metrics are not evidence of real-world performance."
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(y_test, predictions, zero_division=0)
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No failure", "Failure"],
                yticklabels=["No failure", "Failure"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Plot feature importance after preprocessing, including one-hot categories.
    fitted_preprocessor = model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(12)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved reports in: {REPORT_DIR}/")

if __name__ == "__main__":
    main()
