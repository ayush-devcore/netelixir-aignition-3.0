import os
import argparse
import glob
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        str(c).lower().strip().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


# Known channel keywords → canonical name
_CHANNEL_MAP = {
    "google":    "google_ads",
    "gads":      "google_ads",
    "meta":      "meta_ads",
    "facebook":  "meta_ads",
    "fb":        "meta_ads",
    "instagram": "meta_ads",
    "microsoft": "ms_ads",
    "bing":      "ms_ads",
    "msads":     "ms_ads",
}

# GA4 / Shopify files are supporting data; their channel values come from
# an in-row 'source' or 'medium' column rather than the filename.
_SUPPORTING_SOURCES = {"shopify", "ga4", "analytics"}


def identify_channel_from_filename(filename: str) -> str | None:
    """Return canonical channel name from filename, or None for supporting files."""
    lower = filename.lower()
    for keyword, channel in _CHANNEL_MAP.items():
        if keyword in lower:
            return channel
    for src in _SUPPORTING_SOURCES:
        if src in lower:
            return None          # handled row-by-row from content
    return "other"


def map_source_medium_to_channel(source: str) -> str:
    """Map GA4-style source/medium strings to canonical channel."""
    s = str(source).lower()
    for keyword, channel in _CHANNEL_MAP.items():
        if keyword in s:
            return channel
    return "other"


# ---------------------------------------------------------------------------
# Single-file ingestion
# ---------------------------------------------------------------------------

def ingest_file(file_path: str) -> pd.DataFrame | None:
    filename = os.path.basename(file_path)
    print(f"    [*] Processing: {filename}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"    [!] Could not read {filename}: {e}")
        return None

    df = clean_column_names(df)

    # --- Date ---
    date_col = next(
        (c for c in df.columns if any(k in c for k in ("date", "day", "week", "period"))),
        None,
    )
    df["normalized_date"] = (
        pd.to_datetime(df[date_col], errors="coerce")
        if date_col
        else pd.Timestamp.now().normalize()
    )
    df["normalized_date"] = df["normalized_date"].fillna(pd.Timestamp.now().normalize())

    # --- Spend ---
    if "spend" not in df.columns:
        spend_col = next(
            (c for c in df.columns if any(k in c for k in ("cost", "budget", "spend"))),
            None,
        )
        df["spend"] = pd.to_numeric(df[spend_col], errors="coerce").fillna(0.0) if spend_col else 0.0
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0.0)

    # --- Revenue ---
    if "revenue" not in df.columns:
        rev_col = next(
            (c for c in df.columns if any(k in c for k in ("rev", "conv_val", "sales", "value", "gmv"))),
            None,
        )
        df["revenue"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0.0) if rev_col else 0.0
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)

    # --- Campaign name ---
    if "campaign_name" not in df.columns:
        camp_col = next(
            (c for c in df.columns if any(k in c for k in ("campaign", "placement", "ad_group", "adgroup"))),
            None,
        )
        df["campaign_name"] = df[camp_col].astype(str).str.strip() if camp_col else "unclassified_campaign"
    df["campaign_name"] = df["campaign_name"].astype(str).str.strip()

    # --- Channel ---
    file_channel = identify_channel_from_filename(filename)

    if file_channel is None:
        # Supporting file (GA4 / Shopify): derive channel from content
        source_col = next(
            (c for c in df.columns if any(k in c for k in ("source", "medium", "channel"))),
            None,
        )
        if source_col:
            df["channel"] = df[source_col].apply(map_source_medium_to_channel)
        else:
            df["channel"] = "other"
    elif "channel" not in df.columns:
        df["channel"] = file_channel
    # else: file already has a 'channel' column — leave it as-is

    df["channel"] = df["channel"].astype(str).str.strip().str.lower()

    # Drop rows whose channel couldn't be mapped to one of the three ad platforms
    valid_channels = {"google_ads", "meta_ads", "ms_ads"}
    df = df[df["channel"].isin(valid_channels)].copy()

    if df.empty:
        print(f"    [!] No valid ad-platform rows found in {filename} — skipping.")
        return None

    return df[["normalized_date", "channel", "campaign_name", "spend", "revenue"]].copy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIgnition 3.0 — Feature Engineering Pipeline")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory containing raw CSV data files")
    parser.add_argument("--out",      type=str, default="features.parquet",
                        help="Output path for compiled features parquet")
    args = parser.parse_args()

    print(f"[*] Scanning '{args.data_dir}' for CSV files...")
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))

    if not csv_files:
        print("[!] No CSV files found — generating synthetic sample data for local verification...")
        os.makedirs(args.data_dir, exist_ok=True)

        np.random.seed(42)
        n = 120
        mock_df = pd.DataFrame({
            "date":          pd.date_range(end=pd.Timestamp.now(), periods=n),
            "channel":       np.random.choice(["google_ads", "meta_ads", "ms_ads"], n),
            "campaign_name": np.random.choice(["Brand_Search", "Generic_Prospecting", "Retargeting_Q2"], n),
            "spend":         np.random.uniform(50, 2000, n).round(2),
            "revenue":       np.random.uniform(200, 8000, n).round(2),
        })
        mock_path = os.path.join(args.data_dir, "sample_marketing_data.csv")
        mock_df.to_csv(mock_path, index=False)
        csv_files = [mock_path]
        print(f"    [+] Sample data written to {mock_path}")

    # Ingest all files
    frames = []
    for fp in csv_files:
        result = ingest_file(fp)
        if result is not None:
            frames.append(result)

    if not frames:
        raise RuntimeError(
            "Feature engineering failed: no valid ad-platform data could be extracted "
            "from any file in the data directory. Ensure CSVs contain spend/revenue columns."
        )

    master_df = pd.concat(frames, ignore_index=True)

    # Drop rows with zero spend AND zero revenue (no signal)
    master_df = master_df[~((master_df["spend"] == 0) & (master_df["revenue"] == 0))].copy()

    # Temporal features for seasonality modelling
    master_df["month"]       = master_df["normalized_date"].dt.month
    master_df["day_of_week"] = master_df["normalized_date"].dt.dayofweek
    master_df["quarter"]     = master_df["normalized_date"].dt.quarter

    master_df.to_parquet(args.out, index=False)

    print(f"[+] Features written to: {args.out}")
    print(f"    Rows: {len(master_df)} | "
          f"Channels: {master_df['channel'].nunique()} | "
          f"Campaigns: {master_df['campaign_name'].nunique()} | "
          f"Date range: {master_df['normalized_date'].min().date()} → "
          f"{master_df['normalized_date'].max().date()}")


if __name__ == "__main__":
    main()