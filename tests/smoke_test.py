import sys
import runpy
import os
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def run_script(script_relpath, argv=None):
    argv = argv or []
    sys_argv_backup = sys.argv[:]
    try:
        sys.argv = [os.path.basename(script_relpath)] + argv
        runpy.run_path(str(ROOT / script_relpath), run_name='__main__')
    finally:
        sys.argv = sys_argv_backup


def main():
    # Clean previous artifacts to ensure deterministic run
    for p in ['features.parquet', 'pickle/model.pkl', 'output/predictions.csv', 'output/predictions_from_refactor.csv']:
        fp = ROOT / p
        if fp.exists():
            try:
                fp.unlink()
            except Exception:
                # try removing file inside directories
                pass

    # Ensure output directory exists
    (ROOT / 'output').mkdir(parents=True, exist_ok=True)

    print('[*] Running feature generation...')
    run_script('src/generate_features.py', ['--data-dir', './data', '--out', 'features.parquet'])

    if not (ROOT / 'features.parquet').exists():
        print('[ERROR] features.parquet was not created')
        sys.exit(2)

    print('[*] Running training...')
    run_script('src/train.py')
    if not (ROOT / 'pickle' / 'model.pkl').exists():
        print('[ERROR] model.pkl was not created')
        sys.exit(3)

    print('[*] Running prediction CLI...')
    run_script('src/predict.py', ['--features', 'features.parquet', '--model', './pickle/model.pkl', '--output', './output/predictions.csv'])

    out_path = ROOT / 'output' / 'predictions.csv'
    if not out_path.exists():
        print('[ERROR] output/predictions.csv not found')
        sys.exit(4)

    print('[*] Validating output schema...')
    df = pd.read_csv(out_path)
    required = {
        'planning_horizon_days', 'marketing_channel', 'campaign_name', 'simulated_budget_spend',
        'revenue_forecast_p10', 'revenue_forecast_p50', 'revenue_forecast_p90',
        'roas_range_p10', 'roas_range_p50', 'roas_range_p90', 'ai_assisted_causal_summary'
    }

    missing = required - set(df.columns)
    if missing:
        print('[ERROR] Missing required columns in output/predictions.csv:', ','.join(sorted(missing)))
        sys.exit(5)

    print('[SUCCESS] Smoke test passed — pipeline ran end-to-end and output schema validated.')
    sys.exit(0)


if __name__ == '__main__':
    main()
