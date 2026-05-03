"""
Train and Save XGBoost Models for NSE Stocks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# CONFIG
DATA_PATH = Path("nse-2022-2025-data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# CLEANING
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
                .str.replace('nan', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# LOAD DATA

def load_data():
    print(" Loading NSE data...")
    
    all_dfs = []
    for csv_file in sorted(DATA_PATH.glob("*.csv")):
        df = pd.read_csv(csv_file)
        df = clean_columns(df)
        df = convert_numeric(df)
        all_dfs.append(df)
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'code', 'day_price'])
    
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    print(f" Loaded {len(df)} rows")
    return df

# FEATURE ENGINEERING 

def prepare_features(df):
    print("Preparing features...")
    
    df = df.copy()
    
    df['day_high'] = df['day_high'].fillna(df['day_price'])
    df['day_low'] = df['day_low'].fillna(df['day_price'])
    df['volume'] = df['volume'].fillna(0)
    
    df['daily_return'] = df.groupby('code')['day_price'].pct_change()
    
    # Lags
    for lag in [1, 2, 3]:
        df[f'return_lag{lag}'] = df.groupby('code')['daily_return'].shift(lag)
        df[f'low_lag{lag}'] = df.groupby('code')['day_low'].shift(lag)
        df[f'high_lag{lag}'] = df.groupby('code')['day_high'].shift(lag)
    
    df['volume_lag1'] = df.groupby('code')['volume'].shift(1)
    
    # Rolling stats
    df['return_roll5_mean'] = df.groupby('code')['daily_return'].shift(1).rolling(5).mean()
    df['return_roll20_mean'] = df.groupby('code')['daily_return'].shift(1).rolling(20).mean()
    df['return_roll5_std'] = df.groupby('code')['daily_return'].shift(1).rolling(5).std()
    
    # EWM
    df['return_ewm5'] = df.groupby('code')['daily_return'].shift(1).ewm(span=5).mean()
    df['return_ewm20'] = df.groupby('code')['daily_return'].shift(1).ewm(span=20).mean()
    
    # 52-week position
    df['price_pos_52w'] = (
        (df['day_price'] - df['12m_low']) /
        (df['12m_high'] - df['12m_low'] + 1e-6)
    ).fillna(0.5)
    
    # Volume ratio
    df['vol_roll20_mean'] = df.groupby('code')['volume'].shift(1).rolling(20).mean()
    df['volume_ratio'] = df['volume_lag1'] / (df['vol_roll20_mean'] + 1e-6)
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0)
    
    # Time
    df['dow_num'] = df['date'].dt.dayofweek
    
    # Range
    df['range_lag1'] = df['high_lag1'] - df['low_lag1']
    
    return df

# TRAIN MODELS


def train_models(df):
    blue_chips = [
        'EQTY', 'KCB', 'SCOM', 'COOP', 'SCBK',
        'ABSA', 'KNRE', 'KEGN', 'KPLC', 'BRIT'
    ]
    
    missing = [c for c in blue_chips if c not in df['code'].unique()]
    if missing:
        print(f" Missing stocks in dataset: {missing}")
    
    feature_cols = [
        'return_lag1','return_lag2','return_lag3',
        'low_lag1','low_lag2','low_lag3',
        'high_lag1','high_lag2','high_lag3',
        'volume_lag1',
        'return_roll5_mean','return_roll20_mean','return_roll5_std',
        'return_ewm5','return_ewm20',
        'price_pos_52w','volume_ratio',
        'dow_num','range_lag1'
    ]
    
    trained_codes = []
    
    for code in blue_chips:
        print(f"\n Training {code}...")
        
        stock_df = df[df['code'] == code].copy()
        
        # Targets
        stock_df['target_return'] = stock_df['day_price'].pct_change().shift(-1)
        stock_df['target_low'] = stock_df['day_low'].pct_change().shift(-1)
        stock_df['target_high'] = stock_df['day_high'].pct_change().shift(-1)
        
        stock_df = stock_df.dropna()
        
        if len(stock_df) < 50:
            print(f"Skipping {code} (not enough data)")
            continue
        
        X = stock_df[feature_cols]
        
        models = {}
        
        for name, target in zip(
            ['price','low','high'],
            ['target_return','target_low','target_high']
        ):
            y = stock_df[target]
            
            model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X, y)
            
            preds = model.predict(X)
            mae = mean_absolute_error(y, preds)
            
            print(f"{name} MAE: {mae:.6f}")
            
            models[name] = model
        
        # Save model
        with open(MODELS_DIR / f"{code}_models.pkl", 'wb') as f:
            pickle.dump(models, f)
        
        trained_codes.append(code)
    
    # Save metadata AFTER training
    metadata = {
        'feature_cols': feature_cols,
        'blue_chips': trained_codes
    }
    
    with open(MODELS_DIR / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n Models saved successfully!")
    print(f"Trained stocks: {trained_codes}")

# MAIN

if __name__ == "__main__":
    
    df = load_data()
    
    df = prepare_features(df)
    
    train_models(df)