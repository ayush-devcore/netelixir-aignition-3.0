# AIgnition 3.0 — Probabilistic Revenue Forecasting for E-commerce Marketing

**NetElixir AIgnition 3.0 Hackathon Submission**

A practical AI-assisted forecasting utility that predicts e-commerce revenue and ROAS across Google Ads, Meta Ads, and Microsoft Ads using probabilistic quantile regression and an LLM-powered causal inference layer.

---

## Quick Start

```bash
git clone https://github.com/ayush-devcore/netelixir-aignition-3.0.git
cd netelixir-aignition-3.0
pip install -r requirements.txt
./run.sh
```

Output is written to `./output/predictions.csv`.

---

## How to Run

```bash
./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
```

| Argument | Description | Default |
|---|---|---|
| `DATA_DIR` | Folder containing raw CSV data files | `./data` |
| `MODEL_PATH` | Path to the serialised model bundle | `./models/model.pkl` |
| `OUTPUT_PATH` | Destination for the predictions CSV | `./output/predictions.csv` |

**Example (explicit paths):**
```bash
./run.sh ./data ./models/model.pkl ./output/predictions.csv
```

If `DATA_DIR` contains no CSV files, the pipeline auto-generates a synthetic sample dataset so you can verify the full execution flow on a clean clone.

---

## Repository Structure

```
netelixir-aignition-3.0/
├── run.sh                  # Single entry-point (required by submission guide)
├── requirements.txt        # Pinned Python dependencies
├── data/                   # Drop CSV files here; replaced at test time
│   └── sample_marketing_data.csv
├── models/
│   └── model.pkl           # Serialized LightGBM quantile model bundle
├── src/
│   ├── app.py                # Streamlit app wrapper
│   ├── generate_features.py   # Data ingestion & feature engineering
│   ├── train.py               # Model training pipeline
│   └── predict.py             # Probabilistic inference & causal summaries
├── docs/
│   ├── CONTRIBUTING.md
│   └── DEVELOPER_DOCS.md
└── README.md
```

---

## Input Data Format

Place one or more CSV files inside `data/`. The pipeline auto-detects columns, so exact naming is flexible. It looks for:

| Concept | Accepted column names |
|---|---|
| Date | `date`, `day`, `week`, `period` |
| Spend | `spend`, `cost`, `budget` |
| Revenue | `revenue`, `rev`, `conv_val`, `sales`, `value`, `gmv` |
| Campaign | `campaign_name`, `campaign`, `placement`, `ad_group` |
| Channel | detected from **filename** (`google`, `meta`, `facebook`, `microsoft`, `bing`) or an existing `channel` column |

**Supported channel filenames:** `google_ads_*.csv`, `meta_*.csv`, `microsoft_*.csv`, etc.
**GA4 / Shopify files** are also accepted — channel is derived from the `source` / `medium` column.

---

## Output Format

`predictions.csv` contains one row per channel × campaign × horizon combination:

| Column | Description |
|---|---|
| `planning_horizon_days` | 30, 60, or 90 |
| `marketing_channel` | `google_ads` / `meta_ads` / `ms_ads` |
| `campaign_name` | Campaign identifier |
| `simulated_budget_spend` | Projected total spend over the horizon |
| `revenue_forecast_p10` | Conservative (10th-percentile) revenue estimate |
| `revenue_forecast_p50` | Median (baseline) revenue estimate |
| `revenue_forecast_p90` | Optimistic (90th-percentile) revenue estimate |
| `roas_range_p10/p50/p90` | Corresponding ROAS values |
| `ai_assisted_causal_summary` | LLM or heuristic causal narrative |

---

## AI Causal Inference Layer

The inference engine implements a **hybrid online / offline** approach:

- **Online mode** — if `OPENAI_API_KEY` or `GEMINI_API_KEY` is set in the environment, the engine calls GPT-4o-mini / Gemini to generate agency-grade strategic insights per channel, campaign, and horizon.
- **Offline / pipeline mode** — if no API key is present (or the network is unavailable), a structured heuristic matrix produces differentiated causal narratives based on channel type, campaign intent (brand / retargeting / prospecting), and ROAS tier. This guarantees zero-latency, zero-error execution in the automated test pipeline.

---

## Forecasting Methodology

- **Model:** LightGBM `LGBMRegressor` with `objective='quantile'` — three independent models for q10, q50, q90.
- **Horizon scaling:** The model is trained on daily spend values. At inference time, daily spend is fed to the model and the resulting daily revenue estimate is multiplied by the horizon length to produce the correct aggregate-period forecast.
- **ROAS calculation:** `period_revenue / period_spend`, where both numerator and denominator are scaled by the same horizon factor.
- **Planning windows:** 30, 60, and 90 days.

See `DEVELOPER_DOCS.md` for full methodology, architecture, and assumptions.

---

## Python Version

Python **3.10+** recommended. Tested on Python 3.11.

## Dependencies

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
lightgbm==4.3.0
pyarrow==16.1.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Reproducibility

- All random seeds are fixed (`random_state=42` throughout).
- No absolute paths — all file I/O uses the CLI arguments or relative defaults.
- No internet dependency at run time (offline fallback handles the AI layer).
- Tested on a clean clone in a fresh virtual environment.

---

*NetElixir AIgnition 3.0 — June 2026*