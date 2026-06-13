import os
import pickle
import pandas as pd
import lightgbm as lgb

def train_probabilistic_models():
    print("[*] Loading processed features for training...")
    if not os.path.exists('features.parquet'):
        raise FileNotFoundError("Missing 'features.parquet'. Please run the feature generation step first.")
        
    df = pd.read_parquet('features.parquet')
    X = df[['spend', 'month', 'day_of_week']] 
    y = df['revenue']

    quantiles = [0.10, 0.50, 0.90]
    model_bundle = {}

    print("[*] Initializing Quantile Regression models via LightGBM...")
    for q in quantiles:
        quantile_key = f'q_{int(q*100)}'
        print(f"    -> Training engine for Quantile: {q} ({quantile_key})")
        
        model = lgb.LGBMRegressor(
            objective='quantile', 
            alpha=q, 
            random_state=42, 
            n_estimators=50,
            verbosity=-1
        )
        model.fit(X, y)
        model_bundle[quantile_key] = model

    os.makedirs('pickle', exist_ok=True)
    model_output_path = os.path.join('pickle', 'model.pkl')
    
    with open(model_output_path, 'wb') as f:
        pickle.dump(model_bundle, f)
        
    print(f"[+] Multi-quantile model bundle successfully saved to: {model_output_path}")

if __name__ == '__main__':
    train_probabilistic_models()