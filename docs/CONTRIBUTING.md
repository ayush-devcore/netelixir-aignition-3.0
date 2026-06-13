CONTRIBUTING
============

Thank you for contributing! Before opening a Pull Request, please verify the repository runs end-to-end locally by following these steps.

1) Prerequisites

- Install Python 3.12 and Git.

2) Create and activate a virtual environment

Windows (Powershell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4) (Optional) Generate features locally

This step will normalize CSVs in `./data` and create `data/features.parquet`.

```bash
python src/generate_features.py --data-dir ./data --out data/features.parquet
```

5) Run the smoke test (REQUIRED before opening a PR)

This test runs feature generation, trains the model, and runs inference, then validates the output schema.

```bash
python tests/smoke_test.py
```

If the smoke test exits with code `0` and prints `SUCCESS`, your environment is correct. If it fails, please fix locally and only open a PR when the smoke test passes.

6) (Optional) Run the Streamlit app for manual inspection

```bash
streamlit run app.py
```

Notes
- Avoid committing large binary artifacts. The CI workflow will run the smoke test on every push to `main`.
- If you change the pipeline, update `tests/smoke_test.py` accordingly so CI keeps validating the expected behavior.

Thank you — maintainers will review your PR once the smoke test passes.
