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

st.set_page_config(page_title="Iris Studio · Flower classifier", page_icon="🌼", layout="wide")

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


st.markdown(
    '<div class="topline"><div class="brand"><span>✳</span> Fieldnotes · ML</div>'
    '<div class="edition">Classification study &nbsp; / &nbsp; 01</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<p class="eyebrow">A tiny field guide to machine learning</p>', unsafe_allow_html=True)
st.title("Meet the Iris.")
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
    predict = st.button("Identify this flower  →", use_container_width=True)

with right:
    st.markdown('<p class="eyebrow">Your reading</p>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">A considered prediction</div><p class="panel-copy">Based on a model trained on the classic Iris dataset.</p>', unsafe_allow_html=True)
    if predict:
        row = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=artifact["features"])
        model = artifact["model"]
        prediction = str(model.predict(row)[0]).lower()
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities))
        name, description = SPECIES.get(prediction, (prediction.title(), "Predicted species."))
        st.markdown(
            f'<div class="result"><div class="result-label">Most likely species</div>'
            f'<div class="result-name">{name}</div><div class="result-copy">{description}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(confidence, text=f"Model confidence · {confidence:.0%}")
        with st.expander("See probabilities for all species"):
            for label, probability in zip(model.classes_, probabilities):
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

st.markdown('<div class="foot">A learning demo using the UCI Iris dataset · Random Forest classifier · Predictions are estimates, not botanical advice.</div>', unsafe_allow_html=True)
