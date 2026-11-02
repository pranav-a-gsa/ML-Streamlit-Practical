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
    :root { --ink:#24312a; --muted:#66736b; --paper:#fbfcfb; --green:#426b53; --line:#dfe5e1; }
    .stApp { background:var(--paper); color:var(--ink); }
    .block-container { max-width:980px; padding-top:2rem; padding-bottom:2.5rem; }
    h1 { color:var(--ink) !important; font-size:2.35rem !important; letter-spacing:-.025em !important; margin-bottom:.3rem !important; }
    h2, h3 { color:var(--ink) !important; }
    [data-testid="stCaptionContainer"] { color:var(--muted); }
    [data-testid="stNumberInput"] label { color:#435148; font-size:.92rem; font-weight:500; }
    [data-testid="stNumberInput"] input { background:#fff; border-color:var(--line); border-radius:6px; }
    div.stButton > button { min-height:42px; border-radius:7px; font-weight:600; }
    div.stButton > button[kind="primary"] { background:var(--green); border-color:var(--green); }
    div.stButton > button[kind="primary"]:hover { background:#355b45; border-color:#355b45; }
    .result { background:#f1f5f2; border:1px solid var(--line); border-left:4px solid var(--green); border-radius:7px; padding:15px 17px; margin:12px 0; }
    .result-label { color:var(--muted); font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
    .result-name { color:var(--ink); font-size:22px; font-weight:650; margin:4px 0; }
    .result-copy { color:#536158; font-size:14px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:7px; padding:12px 14px; min-height:76px; }
    .metric-label { color:var(--muted); font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
    .metric-value { color:var(--ink); font-size:19px; font-weight:600; margin-top:5px; }
    .foot { color:var(--muted); font-size:12px; margin-top:28px; padding-top:14px; border-top:1px solid var(--line); }
    @media (max-width:640px) { .block-container { padding:1.25rem 1rem 2rem; } h1 { font-size:1.9rem !important; } }
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


st.title("Iris Flower Prediction")
st.caption("Enter the flower measurements in centimeters and select Predict.")

try:
    artifact = load_artifact()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("Flower measurements")
    st.caption("Use the four sepal and petal dimensions below.")
    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.number_input("Sepal length", min_value=0.1, max_value=10.0, value=5.1, step=0.1, format="%.1f", help="Length of the green, leaf-like outer part.")
        petal_length = st.number_input("Petal length", min_value=0.1, max_value=10.0, value=1.4, step=0.1, format="%.1f", help="Length of one colored petal.")
    with c2:
        sepal_width = st.number_input("Sepal width", min_value=0.1, max_value=10.0, value=3.5, step=0.1, format="%.1f", help="Width of the green, leaf-like outer part.")
        petal_width = st.number_input("Petal width", min_value=0.1, max_value=10.0, value=0.2, step=0.1, format="%.1f", help="Width of one colored petal.")
    should_predict = st.button("Predict", type="primary", width="stretch")

with right:
    st.subheader("Prediction")
    st.caption("Predicted species and model confidence appear here.")
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
        st.markdown('<div class="result"><div class="result-label">Ready to predict</div><div class="result-copy">Enter the measurements and select Predict.</div></div>', unsafe_allow_html=True)

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

st.markdown('<div class="foot">Iris dataset · Random Forest classifier · Use predictions as a learning demonstration.</div>', unsafe_allow_html=True)
