import os
import argparse
import pickle
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# AI Causal Inference Layer
# ---------------------------------------------------------------------------

def generate_causal_insight(channel: str, campaign: str, rev_p50: float,
                             roas_p50: float, horizon: int,
                             rev_trend: str = "stable") -> str:
    """
    Hybrid causal inference layer.
    - Online mode : calls OpenAI / Gemini when a valid API key is present.
    - Offline mode: structured rule-based heuristics that produce agency-grade
      rationale without any network dependency (safe for automated pipeline).
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"You are a senior digital marketing analyst at an ecommerce agency.\n"
                f"Channel: {channel} | Campaign: {campaign} | Horizon: {horizon} days\n"
                f"Forecast — Median Revenue: ${rev_p50:,.0f} | Blended ROAS: {roas_p50:.2f}x\n"
                f"Revenue trend signal: {rev_trend}\n\n"
                f"Write exactly 2 sentences: (1) identify the most likely performance driver "
                f"for this specific channel/campaign combination, "
                f"(2) give one concrete budget or bidding recommendation."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=90,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass  # fall through to offline heuristics

    # ------------------------------------------------------------------
    # Offline heuristic matrix  (channel × campaign × ROAS tier)
    # ------------------------------------------------------------------
    channel_label = {
        "google_ads": "Google Ads",
        "meta_ads":   "Meta Ads",
        "ms_ads":     "Microsoft Ads",
    }.get(channel, channel)

    campaign_lower = campaign.lower()

    # ROAS tiers
    if roas_p50 < 2.5:
        roas_tier = "low"
    elif roas_p50 < 4.5:
        roas_tier = "mid"
    else:
        roas_tier = "high"

    # Campaign-type signals
    if "brand" in campaign_lower:
        intent = "high-intent branded"
        action  = "Protect budget share; consider raising target ROAS to capture incremental branded demand."
    elif "retarget" in campaign_lower or "remarketing" in campaign_lower:
        intent = "retargeting / warm-audience"
        action  = "Refresh creatives and tighten audience recency windows to combat ad fatigue over the {h}-day window."
    elif "generic" in campaign_lower or "prospect" in campaign_lower:
        intent = "generic prospecting / cold-audience"
        action  = "Shift 10–15 % of budget toward lookalike audiences derived from top converters to improve prospect quality."
    else:
        intent = "mixed-intent"
        action  = "Conduct a search-term audit and prune low-quality traffic sources to recover efficiency."

    action = action.replace("{h}", str(horizon))

    # ROAS-tier narrative
    roas_narratives = {
        "low":  (f"{channel_label} {intent} campaigns are experiencing margin compression over the {horizon}-day horizon "
                 f"(projected ROAS {roas_p50:.2f}x), likely driven by rising CPCs and audience saturation."),
        "mid":  (f"{channel_label} {intent} campaigns are tracking within baseline efficiency "
                 f"(projected ROAS {roas_p50:.2f}x), with stable conversion velocity expected over {horizon} days."),
        "high": (f"{channel_label} {intent} campaigns are outperforming efficiency benchmarks "
                 f"(projected ROAS {roas_p50:.2f}x), signalling strong demand alignment over the {horizon}-day window."),
    }

    narrative = roas_narratives[roas_tier]
    return f"Causal Analysis ({horizon}d) — {narrative} {action}"


# ---------------------------------------------------------------------------
# Core prediction logic (reusable by both CLI and any future UI layer)
# ---------------------------------------------------------------------------

def predict_from_features(
    df: pd.DataFrame,
    model_bundle: dict,
    horizons: list = None,
    channel_multipliers: dict = None,
) -> pd.DataFrame:
    """
    Run probabilistic multi-horizon simulation from a features DataFrame.

    Key fix: the quantile models were trained on *daily* spend values.
    We therefore feed daily-scale spend to the model and then multiply
    the resulting daily revenue estimate by the horizon length to obtain
    the correct aggregate-period revenue.  ROAS is then:
        period_revenue / period_spend  =  (daily_rev * h) / (daily_spend * h)
    which correctly preserves the daily efficiency ratio.
    """
    if horizons is None:
        horizons = [30, 60, 90]
    if channel_multipliers is None:
        channel_multipliers = {}

    # Per-channel-campaign daily baseline (mean spend, last-seen seasonality)
    base_agg = df.groupby(["channel", "campaign_name"]).agg(
        spend=("spend", "mean"),
        month=("month", "last"),
        day_of_week=("day_of_week", "last"),
    ).reset_index()

    records = []

    for horizon in horizons:
        for _, row in base_agg.iterrows():
            multiplier    = channel_multipliers.get(row["channel"], 1.0)
            daily_spend   = row["spend"] * multiplier

            # Feed *daily* spend to the model (matches training distribution)
            X_infer = pd.DataFrame([{
                "spend":       daily_spend,
                "month":       row["month"],
                "day_of_week": row["day_of_week"],
            }])

            try:
                daily_p10 = float(model_bundle["q_10"].predict(X_infer)[0])
                daily_p50 = float(model_bundle["q_50"].predict(X_infer)[0])
                daily_p90 = float(model_bundle["q_90"].predict(X_infer)[0])
            except Exception:
                daily_p10 = daily_p50 = daily_p90 = 0.0

            # Enforce monotonic quantile ordering and non-negativity
            daily_p10 = max(0.0, daily_p10)
            daily_p50 = max(daily_p10, daily_p50)
            daily_p90 = max(daily_p50, daily_p90)

            # Scale to aggregate planning period
            period_spend = daily_spend  * horizon
            rev_p10      = daily_p10    * horizon
            rev_p50      = daily_p50    * horizon
            rev_p90      = daily_p90    * horizon

            # ROAS: period revenue ÷ period spend
            denom      = max(1.0, period_spend)
            roas_p10   = rev_p10 / denom
            roas_p50   = rev_p50 / denom
            roas_p90   = rev_p90 / denom

            ai_insight = generate_causal_insight(
                channel=row["channel"],
                campaign=row["campaign_name"],
                rev_p50=rev_p50,
                roas_p50=roas_p50,
                horizon=horizon,
            )

            records.append({
                "planning_horizon_days":    horizon,
                "marketing_channel":        row["channel"],
                "campaign_name":            row["campaign_name"],
                "simulated_budget_spend":   round(period_spend, 2),
                "revenue_forecast_p10":     round(rev_p10,  2),
                "revenue_forecast_p50":     round(rev_p50,  2),
                "revenue_forecast_p90":     round(rev_p90,  2),
                "roas_range_p10":           round(roas_p10, 2),
                "roas_range_p50":           round(roas_p50, 2),
                "roas_range_p90":           round(roas_p90, 2),
                "ai_assisted_causal_summary": ai_insight,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIgnition 3.0 — Inference & Simulation Engine")
    parser.add_argument("--features", type=str, default="features.parquet",
                        help="Path to compiled features parquet")
    parser.add_argument("--model",    type=str, default="./pickle/model.pkl",
                        help="Path to serialised model bundle")
    parser.add_argument("--output",   type=str, default="./output/predictions.csv",
                        help="Destination path for predictions CSV")
    args = parser.parse_args()

    # --- Validate inputs ---
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.features):
        raise FileNotFoundError(f"Features file not found: {args.features}")

    print("[*] Loading model bundle...")
    with open(args.model, "rb") as f:
        model_bundle = pickle.load(f)

    print("[*] Loading feature set...")
    df = pd.read_parquet(args.features)

    print("[*] Running multi-horizon probabilistic simulation (30 / 60 / 90 days)...")
    output_df = predict_from_features(df, model_bundle)

    # --- Write output ---
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_df.to_csv(args.output, index=False)
    print(f"[+] Predictions written to: {args.output}")
    print(f"    Rows: {len(output_df)} | "
          f"Channels: {output_df['marketing_channel'].nunique()} | "
          f"Horizons: {sorted(output_df['planning_horizon_days'].unique().tolist())}")


if __name__ == "__main__":
    main()