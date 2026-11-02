"""Train and evaluate a reproducible Random Forest on the bundled Iris CSV."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "iris.csv"
MODEL_PATH = ROOT / "iris_model.pkl"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def train_and_evaluate() -> dict[str, object]:
    """Fit the model, print held-out metrics, and save the deployable artifact."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"iris.csv is missing required columns: {sorted(missing)}")
    if data[FEATURES].isna().any().any() or data[TARGET].isna().any():
        raise ValueError("iris.csv contains missing values.")

    X, y = data[FEATURES], data[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    evaluation_model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=sorted(y.unique()))

    # Refit on all rows for the production artifact after measuring held-out accuracy.
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    model.fit(X, y)
    artifact = {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "accuracy": float(accuracy),
        "dataset_rows": int(len(data)),
        "evaluation": {
            "labels": sorted(y.unique()),
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
        },
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"Dataset rows: {len(data)}")
    print(f"Held-out accuracy: {accuracy:.3f}")
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: " + ", ".join(model.classes_) + "):")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    print(f"Saved model: {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train_and_evaluate()
