\# AIgnition 3.0 - Technical Documentation \& Architecture Overview



\## 1. Technical Documentation



\### Forecasting Methodology \& Model Selection

To satisfy the requirement for \*\*probabilistic range forecasting\*\* rather than a single deterministic value, this utility implements \*\*Quantile Regression\*\* using \*\*LightGBM (`LGBMRegressor`)\*\*. 

\* Traditional linear models assume a symmetric normal distribution of errors, which fails to capture the volatile spikes and seasonality of ecommerce marketing data.

\* By utilizing the `objective='quantile'` loss function, we train three distinct model layers independently to capture structural distribution boundaries:

&#x20; \* \*\*$q\_{10}$ (Low Edge):\*\* Represents a conservative risk-adjusted floor.

&#x20; \* \*\*$q\_{50}$ (Median Edge):\*\* Represents the baseline expected outcome.

&#x20; \* \*\*$q\_{90}$ (High Edge):\*\* Represents maximum scale potential during high-performance or peak seasonal periods.



\### Data Preprocessing Logic \& Campaign Consistency

The preprocessing engine (`generate\_features.py`) scans the target data directory dynamically. It ingests disconnected platform exports (Google Ads, Meta Ads, Microsoft Ads, Shopify, GA4) and executes the following normalization steps:

1\. \*\*Schema Standardization:\*\* Strips whitespace, forces lowercase, and normalizes disparate platform headers into a uniform naming convention.

2\. \*\*Channel Mapping:\*\* Dynamically scans file names and contents to map raw logs to their correct platform attribution channel without hardcoded dependencies.

3\. \*\*Temporal Alignment:\*\* Extracts core seasonal markers (`month`, `day\_of\_week`, `quarter`) to anchor multi-horizon projections.

4\. \*\*Data Integrity Guard:\*\* Validates campaign structures, fixes text fragmentation, and handles missing spend/revenue values gracefully.



\### AI Integration Strategy (Causal Inference Layer)

The solution implements a hybrid \*\*Deterministic/Generative Causal Layer\*\* within `predict.py` to balance analytical depth with the hackathon's "no internet at runtime" constraint:

\* \*\*Online Mode (Demo Workflow):\*\* When a valid LLM API key (`OPENAI\_API\_KEY` or `GEMINI\_API\_KEY`) is present, the script constructs contextual prompts passing the simulated multi-horizon budgets, median revenue, and calculated ROAS ranges to generate agency-ready strategic insights.

\* \*\*Offline Guardrail Mode (Automated Pipeline):\*\* If network calls are unavailable or credentials are omitted, the engine smoothly falls back to a localized rule-based heuristic matrix. This ensures the automated script executes with zero errors or latencies while still delivering descriptive agency rationales.



\### Assumptions and Limitations

\* \*\*Attribution Constancy:\*\* Assumes the underlying multi-channel attribution weights provided in the ingestion files represent the absolute source of truth.

\* \*\*Feature Scope:\*\* Predictions are primarily driven by planned ad spend volume and explicit seasonal calendar markers. Out-of-band market conditions (inventory stockouts, global supply chain shocks) are outside the model's predictive boundary.



\---



\## 2. Architecture Overview

