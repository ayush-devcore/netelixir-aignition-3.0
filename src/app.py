import os
import sys
import tempfile
import subprocess
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure src is importable for using utility functions (safe, relative)
ROOT = Path(__file__).parent
SRC_PATH = ROOT / 'src'
sys.path.insert(0, str(SRC_PATH))

try:
    from predict import predict_from_features, generate_offline_causal_insights
except Exception:
    # graceful fallback if import not possible (function still optional)
    def generate_offline_causal_insights(channel, revenue_mid, roas_mid, horizon):
        if roas_mid < 2.0:
            return f"Causal Analysis ({horizon}d): {channel} shows scale compression. Recommend shifting budget toward high-intent terms."
        elif roas_mid > 5.0:
            return f"Causal Analysis ({horizon}d): strong seasonal efficiency gains for {channel}. Consider increasing budget."
        else:
            return f"Causal Analysis ({horizon}d): {channel} maintaining stable equilibrium. Monitor for fatigue."

    def predict_from_features(df, model_bundle, horizons=None, channel_multipliers=None):
        # minimal local fallback mirrors original behavior (single-horizon safe)
        if horizons is None:
            horizons = [30]
        if channel_multipliers is None:
            channel_multipliers = {}
        # reuse compute_simulation-like behavior
        base_aggregates = df.groupby(['channel', 'campaign_name']).agg({
            'spend': 'mean',
            'month': 'last',
            'day_of_week': 'last'
        }).reset_index()
        records = []
        for h in horizons:
            for _, row in base_aggregates.iterrows():
                projected_spend = row['spend'] * h * channel_multipliers.get(row['channel'], 1.0)
                records.append({
                    'planning_horizon_days': h,
                    'marketing_channel': row['channel'],
                    'campaign_name': row['campaign_name'],
                    'simulated_budget_spend': projected_spend,
                    'revenue_forecast_p10': 0.0,
                    'revenue_forecast_p50': 0.0,
                    'revenue_forecast_p90': 0.0,
                    'roas_range_p10': 0.0,
                    'roas_range_p50': 0.0,
                    'roas_range_p90': 0.0,
                    'ai_assisted_causal_summary': generate_offline_causal_insights(row['channel'], 0.0, 0.0, h)
                })
        import pandas as _pd
        return _pd.DataFrame(records)


st.set_page_config(page_title='NetElixir Forecast Dashboard', layout='wide')


def run_generate_features(uploaded_files, out_path: Path):
    """Save uploaded files to a temp dir and run the existing CLI feature generator against them."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for f in uploaded_files:
            # write bytes to temp file preserving original name
            dest = td_path / f.name
            with open(dest, 'wb') as wf:
                wf.write(f.getbuffer())

        cmd = [sys.executable, str(SRC_PATH / 'generate_features.py'), '--data-dir', str(td_path), '--out', str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout + '\n' + proc.stderr


@st.cache_data(show_spinner=False)
def load_features(path: Path):
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_model(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


    # inference now uses predict_from_features imported from src/predict.py


def make_corridor_figure(sim_df: pd.DataFrame, horizon: int):
    # aggregate totals across campaigns for the horizon
    agg = sim_df.groupby('marketing_channel').agg({
        'revenue_forecast_p10': 'sum',
        'revenue_forecast_p50': 'sum',
        'revenue_forecast_p90': 'sum'
    }).sum()  # sum across channels to get total

    total_p10 = agg['revenue_forecast_p10']
    total_p50 = agg['revenue_forecast_p50']
    total_p90 = agg['revenue_forecast_p90']

    days = np.arange(1, horizon + 1)
    # distribute cumulatively for visual corridor
    p10 = np.linspace(0, total_p10, horizon)
    p50 = np.linspace(0, total_p50, horizon)
    p90 = np.linspace(0, total_p90, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=p90, line=dict(color='rgba(66,133,244,0.6)'), name='P90', hovertemplate='%{y:$,.0f}'))
    fig.add_trace(go.Scatter(x=days, y=p50, line=dict(color='rgba(52,199,89,1)'), name='P50', hovertemplate='%{y:$,.0f}'))
    fig.add_trace(go.Scatter(x=days, y=p10, line=dict(color='rgba(219,68,55,0.6)'), name='P10', hovertemplate='%{y:$,.0f}'))
    fig.update_layout(showlegend=True, xaxis_title='Day', yaxis_title='Cumulative Revenue', title='Revenue Performance Corridor')
    return fig


def main():
    st.title('NetElixir — Probabilistic Forecast Dashboard')

    # Left: file upload / ingestion
    with st.container():
        st.header('Data Ingestion Hub')
        uploaded = st.file_uploader('Upload raw marketing CSVs (GA4, Shopify, ad exports)', accept_multiple_files=True, type=['csv'])
        features_path = ROOT / 'features.parquet'

        if uploaded and st.button('Normalize & Generate Features'):
            with st.spinner('Normalizing and generating features...'):
                code, out = run_generate_features(uploaded, features_path)
            if code == 0 and features_path.exists():
                st.success('Features generated successfully.')
            else:
                st.error('Feature generation reported issues. See logs below.')
                st.code(out)

    # show a preview if features exist
    if features_path.exists():
        try:
            df = load_features(features_path)
            st.subheader('Aggregated Master Timeline Preview')
            preview_cols = [c for c in ['normalized_date', 'channel', 'campaign_name', 'spend', 'revenue'] if c in df.columns]
            st.dataframe(df[preview_cols].sort_values('normalized_date').head(50))
        except Exception as e:
            st.error(f'Unable to read features.parquet: {e}')

    # Sidebar: sliders and horizon
    st.sidebar.header('Scenario & Budget Simulator')
    # default multipliers based on historical averages
    try:
        hist = df.groupby('channel')['spend'].mean().to_dict()
    except Exception:
        hist = {'google_ads': 1000.0, 'meta_ads': 800.0, 'ms_ads': 300.0}

    def slider_for(channel):
        default = 1.0
        if channel in hist and hist[channel] > 0:
            default = 1.0
        return st.sidebar.slider(channel + ' multiplier', 0.0, 4.0, default, step=0.01, help='0 = -100% | 1 = historical | 4 = +300%')

    g_mult = slider_for('google_ads')
    m_mult = slider_for('meta_ads')
    b_mult = slider_for('ms_ads')

    horizon = st.sidebar.radio('Planning Horizon', (30, 60, 90), index=0)

    # Ensure model exists or offer training
    model_path = ROOT / 'pickle' / 'model.pkl'
    if not model_path.exists():
        if st.sidebar.button('Train Model (creates pickle/model.pkl)'):
            with st.sidebar.spinner('Training probabilistic models — this may take a moment...'):
                cmd = [sys.executable, str(SRC_PATH / 'train.py')]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0 and model_path.exists():
                    st.sidebar.success('Model trained and saved.')
                else:
                    st.sidebar.error('Training failed; see logs below')
                    st.sidebar.code(proc.stdout + '\n' + proc.stderr)

    if model_path.exists() and features_path.exists():
        model_bundle = load_model(model_path)
        sim_df = predict_from_features(df, model_bundle, horizons=[horizon], channel_multipliers={
            'google_ads': g_mult,
            'meta_ads': m_mult,
            'ms_ads': b_mult
        })
        # filter to the selected horizon (predict_from_features returns one row per horizon per campaign)
        sim_df = sim_df[sim_df['planning_horizon_days'] == horizon].reset_index(drop=True)

        st.header('Forecasting Grid')
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = make_corridor_figure(sim_df, horizon)
            st.plotly_chart(fig, use_container_width=True)

            # Channel allocation
            st.subheader('Channel Allocation Breakdown')
            chagg = sim_df.groupby('marketing_channel').agg({'revenue_forecast_p50': 'sum', 'simulated_budget_spend': 'sum'}).reset_index()
            chagg['roas_p50'] = chagg['revenue_forecast_p50'] / chagg['simulated_budget_spend'].replace(0, 1)
            bar = px.bar(chagg, x='marketing_channel', y=['revenue_forecast_p50', 'roas_p50'], barmode='group', labels={'value': 'Value', 'marketing_channel': 'Channel'})
            st.plotly_chart(bar, use_container_width=True)

        with col2:
            st.subheader('Blended ROAS Metrics')
            total_spend = sim_df['simulated_budget_spend'].sum()
            roas_p10 = sim_df['revenue_forecast_p10'].sum() / max(1.0, total_spend)
            roas_p50 = sim_df['revenue_forecast_p50'].sum() / max(1.0, total_spend)
            roas_p90 = sim_df['revenue_forecast_p90'].sum() / max(1.0, total_spend)

            def color_for(x):
                if x >= 3.0:
                    return '✅'
                if x >= 2.0:
                    return '🟡'
                return '🔴'

            st.metric('P10 ROAS', f"{roas_p10:.2f}x", delta=None)
            st.metric('P50 ROAS', f"{roas_p50:.2f}x", delta=None)
            st.metric('P90 ROAS', f"{roas_p90:.2f}x", delta=None)

            st.markdown('---')
            st.subheader('AI-Assisted Strategic Causal Analysis')
            # show aggregated AI insights per channel
            for ch, grp in sim_df.groupby('marketing_channel'):
                roas_mid = grp['roas_range_p50'].mean()
                rev_mid = grp['revenue_forecast_p50'].sum()
                insight = generate_offline_causal_insights(ch, rev_mid, roas_mid, horizon)
                st.markdown(f"**{ch}**: > {insight}")

    else:
        st.info('Upload CSVs and generate `features.parquet`, then train a model (if missing) to enable interactive forecasting.')


if __name__ == '__main__':
    main()
