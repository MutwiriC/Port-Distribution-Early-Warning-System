"""
NSE Stock Price Predictor - Streamlit App
Inspired by Ziidi Trader dark theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta, datetime
import pickle

# PAGE CONFIG
st.set_page_config(
    page_title="NSE Stock Price Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS - Dark Trading Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #111118 100%);
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        color: #6b7280;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    .stSelectbox > div > div {
        background: #1a1a24;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# DATA CLEANING 
def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )
    return df


def convert_numeric(df):
    numeric_cols = ['day_price', 'day_high', 'day_low', '12m_high', '12m_low', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('—', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# LOAD DATA & MODELS
@st.cache_resource
def load_data_and_models():
    try:
        data_path = Path("nse-2022-2025-data")
        all_dfs = []
        for csv_file in sorted(data_path.glob("*.csv")):
            df = pd.read_csv(csv_file)
            df = clean_columns(df)
            df = convert_numeric(df)
            all_dfs.append(df)
        
        nse_data = pd.concat(all_dfs, ignore_index=True)
        nse_data['date'] = pd.to_datetime(nse_data['date'], errors='coerce')
        nse_data = nse_data.dropna(subset=['date', 'code', 'day_price'])
        nse_data = nse_data.sort_values(['code', 'date']).reset_index(drop=True)
        
        models_dir = Path("models")
        with open(models_dir / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        models_dict = {}
        for code in metadata['blue_chips']:
            file = models_dir / f"{code}_models.pkl"
            if file.exists():
                with open(file, 'rb') as f:
                    models_dict[code] = pickle.load(f)
        
        latest_date = nse_data['date'].max()
        return nse_data, metadata, models_dict, latest_date
    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        return None, None, None, None


# FEATURE ENGINEERING
@st.cache_data
def prepare_features(df):
    df = df.copy()
    df['day_high'] = df['day_high'].fillna(df['day_price'])
    df['day_low'] = df['day_low'].fillna(df['day_price'])
    df['volume'] = df['volume'].fillna(0)
    df['daily_return'] = df['day_price'].pct_change()
    
    for lag in [1,2,3]:
        df[f'return_lag{lag}'] = df['daily_return'].shift(lag)
        df[f'low_lag{lag}'] = df['day_low'].shift(lag)
        df[f'high_lag{lag}'] = df['day_high'].shift(lag)
    
    df['volume_lag1'] = df['volume'].shift(1)
    df['return_roll5_mean'] = df['daily_return'].shift(1).rolling(5).mean()
    df['return_roll20_mean'] = df['daily_return'].shift(1).rolling(20).mean()
    df['return_roll5_std'] = df['daily_return'].shift(1).rolling(5).std()
    df['return_ewm5'] = df['daily_return'].shift(1).ewm(span=5).mean()
    df['return_ewm20'] = df['daily_return'].shift(1).ewm(span=20).mean()
    
    df['price_pos_52w'] = ((df['day_price'] - df['12m_low']) / (df['12m_high'] - df['12m_low'] + 1e-6)).fillna(0.5)
    df['vol_roll20_mean'] = df['volume'].shift(1).rolling(20).mean()
    df['volume_ratio'] = df['volume_lag1'] / (df['vol_roll20_mean'] + 1e-6)
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0)
    df['dow_num'] = df['date'].dt.dayofweek
    df['range_lag1'] = df['high_lag1'] - df['low_lag1']
    
    return df


# PREDICTION 
def predict_next_day(stock_df, code, models, metadata):
    feature_cols = metadata['feature_cols']
    feature_df = stock_df[feature_cols].dropna().reset_index(drop=True)
    
    if len(feature_df) < 10:
        return None, None, None, "Not enough data"
    
    latest_features = feature_df.iloc[-1:]
    pred_return = models[code]['price'].predict(latest_features)[0]
    pred_low_return = models[code]['low'].predict(latest_features)[0]
    pred_high_return = models[code]['high'].predict(latest_features)[0]
    
    today_price = stock_df.iloc[-1]['day_price']
    pred_price = today_price * (1 + pred_return)
    pred_low = today_price * (1 + pred_low_return)
    pred_high = today_price * (1 + pred_high_return)
    
    pred_low = min(pred_low, pred_price)
    pred_high = max(pred_high, pred_price)
    
    return pred_price, pred_low, pred_high, None


# MAIN APP
def main():
    nse_data, metadata, models_dict, latest_date = load_data_and_models()
    
    if nse_data is None:
        st.error("Failed to load data.")
        return
    
    blue_chips = metadata['blue_chips']
    
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="color: white; margin: 0; font-size: 28px;">NSE Blue-Chips Stock Predictor</h1>
        <p style="color: #6b7280; margin: 4px 0 0 0;">Next-Day Price Predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, _ = st.columns([2, 3])
    with col1:
        stock = st.selectbox("Select Stock", blue_chips, label_visibility="collapsed")
    
    stock_df = nse_data[nse_data['code'] == stock].copy()
    if len(stock_df) < 50:
        st.error("Not enough data for this stock")
        return
    
    stock_df = prepare_features(stock_df)
    latest = stock_df.iloc[-1]
    prev = stock_df.iloc[-2] if len(stock_df) > 1 else latest
    
    today_price = latest['day_price']
    today_high = latest['day_high']
    today_low = latest['day_low']
    today_volume = latest['volume']
    price_change = today_price - prev['day_price']
    price_change_pct = (price_change / prev['day_price']) * 100 if prev['day_price'] != 0 else 0
    
    pred_price, pred_low, pred_high, error = predict_next_day(stock_df, stock, models_dict, metadata)
    if error:
        st.error(error)
        return

    pred_change = pred_price - today_price
    pred_change_pct = (pred_change / today_price) * 100
    
    # Logic for colors
    change_color = "#10b981" if price_change >= 0 else "#ef4444"
    change_bg = "rgba(16, 185, 129, 0.15)" if price_change >= 0 else "rgba(239, 68, 68, 0.15)"
    change_icon = "^" if price_change >= 0 else "v"

    # MAIN STOCK CARD
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #1a1a24 0%, #12121a 100%); border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
            <div>
                <p style="font-size: 28px; font-weight: 700; color: #ffffff; margin: 0;">{stock}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 4px 0 0 0;">Nairobi Securities Exchange</p>
            </div>
            <div style="text-align: right;">
                <p style="font-size: 36px; font-weight: 700; color: #ffffff; margin: 0;">KES {today_price:,.2f}</p>
                <span style="background: {change_bg}; color: {change_color}; padding: 6px 12px; border-radius: 8px; font-size: 14px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;">
                    {change_icon} {abs(price_change):.2f} ({abs(price_change_pct):.2f}%)
                </span>
            </div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 10px; padding: 12px 16px; display: flex; align-items: center; gap: 10px;">
            <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block;"></span>
            <span style="color: #10b981; font-size: 14px;">
                Latest data: {latest['date'].strftime('%B %d, %Y')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Historical", "Details"])
    
    with tab1:
        st.markdown('<p style="font-size: 18px; font-weight: 600; color: #ffffff; margin: 24px 0 16px 0;">Tomorrow\'s Prediction</p>', unsafe_allow_html=True)
        
        p_color = "#10b981" if pred_change >= 0 else "#ef4444"
        p_bg = "linear-gradient(145deg, #1a2e1a 0%, #12201a 100%)" if pred_change >= 0 else "linear-gradient(145deg, #2e1a1a 0%, #201212 100%)"
        p_icon = "^" if pred_change >= 0 else "v"
        
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 20px;">
            <div style="background: #2a1e1e; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(239, 68, 68, 0.2);">
                <p style="font-size: 11px; color: #6b7280; text-transform: uppercase; margin-bottom: 8px;">Predicted Low</p>
                <p style="font-size: 24px; font-weight: 700; color: #ef4444; margin: 0;">KES {pred_low:,.2f}</p>
            </div>
            <div style="background: {p_bg}; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid {p_color}33;">
                <p style="font-size: 11px; color: #6b7280; text-transform: uppercase; margin-bottom: 8px;">Predicted Price</p>
                <p style="font-size: 24px; font-weight: 700; color: {p_color}; margin: 0;">KES {pred_price:,.2f}</p>
                <p style="font-size: 12px; color: {p_color}; margin-top: 4px;">{p_icon} {abs(pred_change_pct):.2f}%</p>
            </div>
            <div style="background: #1e2a1e; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(34, 197, 94, 0.2);">
                <p style="font-size: 11px; color: #6b7280; text-transform: uppercase; margin-bottom: 8px;">Predicted High</p>
                <p style="font-size: 24px; font-weight: 700; color: #22c55e; margin: 0;">KES {pred_high:,.2f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<p style="font-size: 18px; font-weight: 600; color: #ffffff; margin: 24px 0 16px 0;">Price History</p>', unsafe_allow_html=True)
        chart_data = stock_df.tail(30)[['date', 'day_price', 'day_high', 'day_low']].set_index('date')
        st.line_chart(chart_data)
        
    with tab3:
        st.markdown('<p style="font-size: 18px; font-weight: 600; color: #ffffff; margin: 24px 0 16px 0;">Today\'s Statistics</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <table style="width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 16px;">
            <tr>
                <th style="background: rgba(59, 130, 246, 0.1); color: #60a5fa; padding: 12px 16px; text-align: left; font-size: 12px; text-transform: uppercase; border-radius: 8px 0 0 0;">Metric</th>
                <th style="background: rgba(59, 130, 246, 0.1); color: #60a5fa; padding: 12px 16px; text-align: left; font-size: 12px; text-transform: uppercase; border-radius: 0 8px 0 0;">Value</th>
            </tr>
            <tr>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">Day High</td>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">KES {today_high:,.2f}</td>
            </tr>
            <tr>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">Day Low</td>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">KES {today_low:,.2f}</td>
            </tr>
            <tr>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">Volume</td>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">{today_volume:,.0f}</td>
            </tr>
            <tr>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">52W High</td>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">KES {latest['12m_high']:,.2f}</td>
            </tr>
            <tr>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-radius: 0 0 0 8px;">52W Low</td>
                <td style="background: #16161e; color: #ffffff; padding: 14px 16px; border-radius: 0 0 8px 0;">KES {latest['12m_low']:,.2f}</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    # FOOTER DISCLAIMER
    st.markdown("""
    <div style="margin-top: 40px; padding: 20px; background: rgba(59, 130, 246, 0.05); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.1); text-align: center;">
        <p style="color: #60a5fa; font-size: 13px; margin: 0; line-height: 1.6;">
            <strong>Disclaimer:</strong> This tool is for educational purposes only and is not approved by any regulatory authority.
                         It does not constitute financial advice or recommendations.
                         Users should conduct their own research or consult licensed financial advisors before making investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()