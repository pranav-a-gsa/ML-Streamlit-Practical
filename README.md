# Iris Studio

A self-contained Iris species classifier built with Streamlit and scikit-learn. It trains a reproducible Random Forest, reports held-out evaluation metrics, and serves predictions with per-species probabilities.

## Files

- `app.py` — Streamlit interface
- `train.py` — training, evaluation, and model export
- `iris.csv` — bundled Iris measurements
- `iris_model.pkl` — generated model artifact (included for one-click deployment)
- `requirements.txt` — pinned major-version dependency ranges

## Run locally

Use Python 3.10 or newer. From this folder:

```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

The training script uses an 80/20 stratified split and a fixed seed to report held-out accuracy, a classification report, and a confusion matrix. It then refits the model on all available rows for the deployed artifact. The app displays the held-out score and each prediction's class probabilities.

## Deploy to Streamlit Community Cloud

Push this folder's contents to a GitHub repository, then create a Streamlit Community Cloud app pointing to `app.py` at the repository root. The checked-in model means the cloud app does not need a training step during startup. To retrain, run `python train.py` and commit the updated artifact.
