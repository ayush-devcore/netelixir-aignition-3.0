import os
import argparse
import glob
import pandas as pd
import numpy as np

def clean_column_names(df):
    df.columns = [str(col).lower().strip().replace(" ", "_").replace("-", "_") for col in df.columns]
    return df

def identify_marketing_channel(filename, df):
    fname_lower = filename.lower()
    if 'google' in fname_lower: return 'google_ads'
    elif 'meta' in fname_lower or 'facebook' in fname_lower: return 'meta_ads'
    elif 'microsoft' in fname_lower or 'bing' in fname_lower: return 'ms_ads'
    elif 'shopify' in fname_lower: return 'shopify_sales'
    elif 'ga4' in fname_lower: return 'ga4_attribution'
    return 'other'

def main():
    parser = argparse.ArgumentParser(description="AIgnition 3.0 Data Ingestion Pipeline")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out', type=str, default='features.parquet')
    args = parser.parse_args()

    print(f"[*] Scanning for input files in: {args.data_dir}")
    csv_pattern = os.path.join(args.data_dir, "*.csv")
    all_files = glob.glob(csv_pattern)

    if not all_files:
        print(f"[!] Warning: No CSV data files detected in '{args.data_dir}'.")
        print("[*] Creating a placeholder dataset for local pipeline verification...")
        os.makedirs(args.data_dir, exist_ok=True)
        mock_dates = pd.date_range(end=pd.Timestamp.now(), periods=120)
        mock_df = pd.DataFrame({
            'date': mock_dates,
            'channel': np.random.choice(['google_ads', 'meta_ads', 'ms_ads'], size=120),
            'campaign_name': np.random.choice(['Brand_Search', 'Generic_Prospecting', 'Retargeting_Q2'], size=120),
            'spend': np.random.uniform(50, 2000, size=120),
            'revenue': np.random.uniform(100, 8000, size=120)
        })
        mock_path = os.path.join(args.data_dir, "sample_marketing_data.csv")
        mock_df.to_csv(mock_path, index=False)
        all_files = [mock_path]

    compiled_records = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"[*] Extracting and normalizing: {filename}")
        try:
            df = pd.read_csv(file_path)
            df = clean_column_names(df)
            date_col = next((col for col in df.columns if 'date' in col or 'day' in col), None)
            df['normalized_date'] = pd.to_datetime(df[date_col]) if date_col else pd.Timestamp.now().normalize()
            if 'spend' not in df.columns: df['spend'] = 0.0
            if 'revenue' not in df.columns:
                rev_col = next((col for col in df.columns if 'rev' in col or 'conv_val' in col or 'sales' in col), None)
                df['revenue'] = df[rev_col] if rev_col else 0.0
            if 'campaign_name' not in df.columns:
                camp_col = next((col for col in df.columns if 'camp' in col or 'placement' in col), None)
                df['campaign_name'] = df[camp_col] if camp_col else 'unclassified_campaign'
            df['campaign_name'] = df['campaign_name'].astype(str).str.strip()
            if 'channel' not in df.columns: df['channel'] = identify_marketing_channel(filename, df)
            compiled_records.append(df[['normalized_date', 'channel', 'campaign_name', 'spend', 'revenue']].copy())
        except Exception as e:
            print(f"[!] Critical error processing file {filename}: {str(e)}")
            continue

    master_df = pd.concat(compiled_records, ignore_index=True)
    master_df['month'] = master_df['normalized_date'].dt.month
    master_df['day_of_week'] = master_df['normalized_date'].dt.dayofweek
    master_df['quarter'] = master_df['normalized_date'].dt.quarter

    master_df.to_parquet(args.out, index=False)
    print(f"[+] Data preprocessing engine completed. Features written to: {args.out}")

if __name__ == '__main__':
    main()