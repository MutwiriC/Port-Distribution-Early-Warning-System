"""
NSE Stock Price Predictor - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta
import pickle

# PAGE CONFIG

st.set_page_config(
    page_title="NSE Stock Price Predictor",
    layout="wide"
)
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
    numeric_cols = [
        'day_price', 'day_high', 'day_low',
        '12m_high', '12m_low',
        'volume'
    ]
    
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
        
        # Load models
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
    
    df['price_pos_52w'] = (
        (df['day_price'] - df['12m_low']) /
        (df['12m_high'] - df['12m_low'] + 1e-6)
    ).fillna(0.5)
    
    df['vol_roll20_mean'] = df['volume'].shift(1).rolling(20).mean()
    df['volume_ratio'] = df['volume_lag1'] / (df['vol_roll20_mean'] + 1e-6)
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0)
    
    df['dow_num'] = df['date'].dt.dayofweek
    df['range_lag1'] = df['high_lag1'] - df['low_lag1']
    
    return df

# PREDICTION 

def predict_next_day(stock_df, code, models, metadata):
    
    feature_cols = metadata['feature_cols']
    
    full_df = stock_df.copy()
    
    feature_df = stock_df[feature_cols].dropna().reset_index(drop=True)
    
    if len(feature_df) < 10:
        return None, None, None, "Not enough data"
    
    latest_features = feature_df.iloc[-1:]
    
    pred_return = models[code]['price'].predict(latest_features)[0]
    pred_low_return = models[code]['low'].predict(latest_features)[0]
    pred_high_return = models[code]['high'].predict(latest_features)[0]
    
    today_price = full_df.iloc[-1]['day_price']
    
    pred_price = today_price * (1 + pred_return)
    pred_low = today_price * (1 + pred_low_return)
    pred_high = today_price * (1 + pred_high_return)
    
    pred_low = min(pred_low, pred_price)
    pred_high = max(pred_high, pred_price)
    
    return pred_price, pred_low, pred_high, None

# MAIN APP

def main():
    
    st.title("NSE Stock Price Predictor")
    
    nse_data, metadata, models_dict, _ = load_data_and_models()
    
    if nse_data is None:
        return
    
    blue_chips = metadata['blue_chips']
    
    stock = st.selectbox("Select stock", blue_chips)
    
    stock_df = nse_data[nse_data['code'] == stock].copy()
    
    if len(stock_df) < 50:
        st.error("Not enough data")
        return
    
    stock_df = prepare_features(stock_df)
    
    latest = stock_df.iloc[-1]
    today_price = latest['day_price']
    
    pred_price, pred_low, pred_high, error = predict_next_day(
        stock_df, stock, models_dict, metadata
    )
    
    if error:
        st.error(error)
        return
    
    st.metric("Today's Price", f"KES {today_price:.2f}")
    st.metric("Predicted Price", f"KES {pred_price:.2f}")
    st.metric("Predicted Low", f"KES {pred_low:.2f}")
    st.metric("Predicted High", f"KES {pred_high:.2f}")


# RUN APP

if __name__ == "__main__":
    main()