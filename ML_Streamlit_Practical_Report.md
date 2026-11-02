# Train, Evaluate, and Deploy an Iris Flower Prediction Model

**Machine Learning Practical**  
Python · scikit-learn · Streamlit Community Cloud

## 1. Aim

To train a machine learning model using Python, evaluate its performance, and deploy the trained model as a web application using Streamlit Community Cloud.

## 2. Software requirements

- Python 3.10 or newer
- Streamlit 1.49 or newer, below version 2
- pandas 2.x, scikit-learn 1.x, and joblib 1.x
- Git, a GitHub account, and Streamlit Community Cloud access

## 3. Required files

| File | Purpose |
|---|---|
| `app.py` | Streamlit prediction interface |
| `train.py` | Model training and evaluation |
| `iris_model.pkl` | Joblib model artifact and evaluation metadata |
| `requirements.txt` | Compatible dependency constraints |
| `iris.csv` | 150 labeled Iris examples |
| `README.md` | Setup and deployment instructions |

## 4. Procedure

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Train and evaluate the model: `python train.py`.
3. Confirm that `iris_model.pkl` is generated.
4. Run the app locally: `streamlit run app.py`.
5. Adjust the four measurements and select **Identify this flower**.
6. Push the project to the public GitHub repository [`pranav-a-gsa/ML-Streamlit-Practical`](https://github.com/pranav-a-gsa/ML-Streamlit-Practical), branch `main`.
7. In Streamlit Community Cloud, create an app from that repository and branch, with root-level `app.py` as the main file, then deploy.
8. Open the live app, edit all four measurements, select **Identify this flower**, and observe the predicted species.

## 5. Program 1 - train.py

The script uses an 80/20 stratified split and `random_state=42`. It reports accuracy, a confusion matrix, and a classification report on the holdout set, then trains the final deployable artifact on all dataset rows.

```python
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

```

## 6. Program 2 - app.py

The interface provides editable numeric inputs for sepal length, sepal width, petal length, and petal width. It displays the predicted class, prediction probabilities, held-out accuracy, per-species metrics, and confusion matrix.

```python
"""Interactive Streamlit app for Iris species prediction."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "iris_model.pkl"
SPECIES = {
    "setosa": ("Iris setosa", "A distinctive species with short petals and broad sepals."),
    "versicolor": ("Iris versicolor", "The intermediate variety, with balanced petal dimensions."),
    "virginica": ("Iris virginica", "The largest of the three species, often with long petals."),
}

st.set_page_config(page_title="Iris Flower Prediction", page_icon="🌼", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
    :root { --ink:#202b24; --muted:#788078; --paper:#f7f7f2; --green:#315e46; --line:#e3e5dc; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    .block-container { max-width:1160px; padding-top:2.2rem; padding-bottom:3rem; }
    [data-testid="stHeader"] { background:transparent; }
    .topline { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding:0 0 18px; margin-bottom:48px; }
    .brand { font-size:14px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .brand span { color:#799475; font-size:18px; margin-right:9px; }
    .edition { color:var(--muted); font-family:'DM Mono',monospace; font-size:11px; letter-spacing:.12em; text-transform:uppercase; }
    .eyebrow { color:var(--green); font-family:'DM Mono',monospace; font-size:11px; letter-spacing:.16em; text-transform:uppercase; margin:0 0 15px; }
    h1 { font-family:'Playfair Display',serif !important; font-size:clamp(42px,6vw,68px) !important; line-height:1.03 !important; letter-spacing:-.04em !important; margin:0 0 18px !important; color:var(--ink) !important; }
    .lede { color:var(--muted); font-size:16px; line-height:1.7; max-width:540px; margin-bottom:34px; }
    .panel { background:#fff; border:1px solid var(--line); border-radius:16px; padding:25px 26px 22px; }
    .panel-title { font-family:'Playfair Display',serif; font-size:22px; margin:0 0 5px; }
    .panel-copy { color:var(--muted); font-size:13px; margin:0 0 20px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:13px; padding:16px 18px; min-height:88px; }
    .metric-label { color:var(--muted); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; font-size:10px; }
    .metric-value { color:var(--ink); font-size:21px; font-weight:600; margin-top:6px; }
    .result { background:#edf3ec; border:1px solid #dbe7d9; border-radius:14px; padding:20px 22px; margin-top:17px; }
    .result-label { color:var(--green); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; font-size:10px; }
    .result-name { font-family:'Playfair Display',serif; font-size:27px; color:#254832; margin:5px 0; }
    .result-copy { color:#627266; font-size:13px; }
    .foot { color:#8a9089; font-size:11px; margin-top:42px; padding-top:16px; border-top:1px solid var(--line); }
    div.stButton > button { background:var(--green); color:white; border:0; border-radius:9px; height:46px; font-weight:600; }
    div.stButton > button:hover { background:#254832; color:white; }
    [data-testid="stNumberInput"] label { color:#59645c; font-size:13px; }
    @media (max-width:640px) { .block-container { padding:1.4rem 1rem 2rem; } .topline { margin-bottom:34px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("The model file is missing. Run `python train.py` first.")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("The model artifact has an unsupported format. Run `python train.py` again.")
    return artifact


def predict_species(artifact, measurements: list[float]):
    """Return the predicted class and probabilities, or show a recoverable error."""
    try:
        row = pd.DataFrame([measurements], columns=artifact["features"])
        model = artifact["model"]
        prediction = str(model.predict(row)[0]).lower()
        probabilities = model.predict_proba(row)[0]
        return prediction, probabilities
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        return None


st.markdown(
    '<div class="topline"><div class="brand"><span>✳</span> Fieldnotes · ML</div>'
    '<div class="edition">Classification study &nbsp; / &nbsp; 01</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<p class="eyebrow">A tiny field guide to machine learning</p>', unsafe_allow_html=True)
st.title("Iris Flower Prediction")
st.markdown(
    '<p class="lede">Four measurements. Three possibilities. Adjust the flower’s dimensions '
    'and let a Random Forest model identify its species.</p>',
    unsafe_allow_html=True,
)

try:
    artifact = load_artifact()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown('<div class="panel"><p class="panel-title">Flower measurements</p><p class="panel-copy">Enter dimensions in centimeters</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.number_input("Sepal length", min_value=0.1, max_value=10.0, value=5.1, step=0.1, format="%.1f", help="Length of the green, leaf-like outer part.")
        petal_length = st.number_input("Petal length", min_value=0.1, max_value=10.0, value=1.4, step=0.1, format="%.1f", help="Length of one colored petal.")
    with c2:
        sepal_width = st.number_input("Sepal width", min_value=0.1, max_value=10.0, value=3.5, step=0.1, format="%.1f", help="Width of the green, leaf-like outer part.")
        petal_width = st.number_input("Petal width", min_value=0.1, max_value=10.0, value=0.2, step=0.1, format="%.1f", help="Width of one colored petal.")
    st.markdown('</div>', unsafe_allow_html=True)
    should_predict = st.button("Identify this flower  →", width="stretch")

with right:
    st.markdown('<p class="eyebrow">Your reading</p>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">A considered prediction</div><p class="panel-copy">Based on a model trained on the classic Iris dataset.</p>', unsafe_allow_html=True)
    if should_predict:
        outcome = predict_species(artifact, [sepal_length, sepal_width, petal_length, petal_width])
        if outcome is not None:
            prediction, probabilities = outcome
            confidence = float(max(probabilities))
            name, description = SPECIES.get(prediction, (prediction.title(), "Predicted species."))
            st.markdown(
                f'<div class="result"><div class="result-label">Most likely species</div>'
                f'<div class="result-name">{name}</div><div class="result-copy">{description}</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(confidence, text=f"Model confidence · {confidence:.0%}")
            with st.expander("See probabilities for all species"):
                for label, probability in zip(artifact["classes"], probabilities):
                    st.write(f"{str(label).title()} — {probability:.1%}")
                    st.progress(float(probability))
    else:
        st.markdown('<div class="result"><div class="result-label">Ready when you are</div><div class="result-name">🌱</div><div class="result-copy">Enter the four measurements, then identify your flower.</div></div>', unsafe_allow_html=True)

accuracy = float(artifact.get("accuracy", 0))
rows = int(artifact.get("dataset_rows", 150))
m1, m2, m3 = st.columns(3)
for column, label, value in [
    (m1, "Held-out accuracy", f"{accuracy:.0%}"),
    (m2, "Training examples", f"{rows}"),
    (m3, "Species in study", "03"),
]:
    with column:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

with st.expander("Model evaluation details"):
    evaluation = artifact.get("evaluation", {})
    report = evaluation.get("classification_report", {})
    labels = evaluation.get("labels", [])
    matrix = evaluation.get("confusion_matrix", [])
    if report and labels and matrix:
        st.caption("Held-out test set: 30 samples, stratified across all three species.")
        report_rows = []
        for label in labels:
            scores = report.get(label, {})
            report_rows.append({
                "Species": str(label).title(),
                "Precision": scores.get("precision", 0),
                "Recall": scores.get("recall", 0),
                "F1 score": scores.get("f1-score", 0),
                "Support": int(scores.get("support", 0)),
            })
        macro = report.get("macro avg", {})
        report_rows.append({
            "Species": "Macro average",
            "Precision": macro.get("precision", 0),
            "Recall": macro.get("recall", 0),
            "F1 score": macro.get("f1-score", 0),
            "Support": int(macro.get("support", 0)),
        })
        st.dataframe(pd.DataFrame(report_rows), hide_index=True, width="stretch")
        st.caption("Confusion matrix · rows are actual species; columns are predicted species.")
        st.dataframe(
            pd.DataFrame(matrix, index=[str(v).title() for v in labels], columns=[str(v).title() for v in labels]),
            width="stretch",
        )
    else:
        st.info("Run `python train.py` to add the evaluation details to the model file.")

st.markdown('<div class="foot">A learning demo using the UCI Iris dataset · Random Forest classifier · Predictions are estimates, not botanical advice.</div>', unsafe_allow_html=True)

```

## 7. Deployment and screenshots

The project is deployed on Streamlit Community Cloud from the public GitHub repository, branch `main`, with root-level `app.py` as the app entry point. Cloud logs confirmed successful dependency installation and app startup using scikit-learn 1.9.1, matching the model artifact.

Live application: [https://ml-app-practical-ibtaznkwzetgult5ocxjbn.streamlit.app/](https://ml-app-practical-ibtaznkwzetgult5ocxjbn.streamlit.app/)

A live screenshot was captured during verification and showed the running application. No screenshot image file is available in the local project or workspace to embed in this report. The live page above provides direct access to the application; no illustrative or fabricated screenshot is included.
## 8. Observations

| Activity | Observation |
|---|---|
| Dataset | 150 rows; 120 training and 30 testing rows |
| Held-out accuracy | 93.3% (28/30) |
| Confusion matrix | `[[10, 0, 0], [0, 9, 1], [0, 1, 9]]`; labels: Setosa, Versicolor, Virginica |
| Setosa sample | 5.1, 3.5, 1.4, 0.2 → Setosa |
| Versicolor sample | 6.0, 2.9, 4.5, 1.5 → Versicolor |
| Virginica sample | 6.5, 3.0, 5.2, 2.0 → Virginica |
| UI checks | All four inputs editable; live Setosa, Versicolor, and Virginica predictions passed |
| Refresh check | Application loaded after browser refresh and retained working prediction behavior |
| Cloud deployment | Live on Streamlit Community Cloud; updated dependency and model versions deployed |

## 9. Result

The Random Forest model was trained and evaluated on the Iris dataset. It achieved **93.3% held-out accuracy (28/30)**. The app is deployed at the verified Streamlit URL, accepts all four editable measurements, and correctly predicted Setosa, Versicolor, and Virginica for the recorded sample inputs, including after a browser refresh.

## 10. GitHub repository URL

Verified public repository: [`https://github.com/pranav-a-gsa/ML-Streamlit-Practical`](https://github.com/pranav-a-gsa/ML-Streamlit-Practical)  
Status: public repository on `main`; existing project history and required files are available. The latest model compatibility fix is committed as `973dce9` and pushed.

## 11. Live Streamlit application URL

Verified live application: [`https://ml-app-practical-ibtaznkwzetgult5ocxjbn.streamlit.app/`](https://ml-app-practical-ibtaznkwzetgult5ocxjbn.streamlit.app/)  
Status: deployment succeeded; live predictions and refresh behavior verified.


