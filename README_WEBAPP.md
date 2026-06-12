NetElixir Web App

Files added:
- app.py — Streamlit dashboard wrapper around existing `src` pipeline

Run locally:

1. Install dependencies (prefer a virtualenv):

```bash
pip install -r requirements.txt
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

Notes:
- Upload raw CSV exports in the Data Ingestion Hub. The app will save uploads to a temporary directory and call `src/generate_features.py` to produce `features.parquet`.
- If `pickle/model.pkl` is missing, use the sidebar "Train Model" button to run `src/train.py` and create the model bundle.
- Sliders in the sidebar adjust per-network spend multipliers (0 = -100%, 1 = historical average, 4 = +300%). Changing sliders re-runs the in-memory simulation using the serialized model.
