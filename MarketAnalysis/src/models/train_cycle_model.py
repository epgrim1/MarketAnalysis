#!/usr/bin/env python3
"""
Train a RandomForestClassifier for predicting the economic cycle state using
the available daily data files in data/raw/.

For each ticker (SPY, GLD, TIP, UUP, SHY, TLT), the script:
  - Loads the CSV file.
  - Converts the Unix timestamp to a date.
  - Selects the columns: close, ROC, ATR, and RSI.
  - Renames these columns so they are prefixed by the ticker symbol.
Then, the script merges the data on date, derives additional features, generates
a pseudo-label (cycle_state) using rule-based logic, trains the classifier, prints
evaluation metrics, and saves the trained model to src/models/ml_cycle_state_model.pkl.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

# --- Set up base directories using pathlib ---

# __file__ is the path to this script (assumed to be in src/models/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Go up three levels to reach MarketAnalysis/
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_SAVE_PATH = BASE_DIR / "src" / "models" / "ml_cycle_state_model.pkl"

# --- Configuration: Define ticker info ---
tickers_info = {
    "SPY": {"exchange": "AMEX", "file": "AMEX_SPY 1D.csv"},
    "GLD": {"exchange": "AMEX", "file": "AMEX_GLD 1D.csv"},
    "TIP": {"exchange": "AMEX", "file": "AMEX_TIP 1D.csv"},
    "UUP": {"exchange": "AMEX", "file": "AMEX_UUP 1D.csv"},
    "SHY": {"exchange": "NASDAQ", "file": "NASDAQ_SHY 1D.csv"},
    "TLT": {"exchange": "NASDAQ", "file": "NASDAQ_TLT 1D.csv"}
}

# --- Helper Functions ---

def load_and_process_ticker(ticker: str, info: dict) -> pd.DataFrame:
    """
    Load CSV data for a given ticker, convert the timestamp to a date,
    compute ROC if missing, and rename columns to include the ticker as a prefix.
    """
    file_path = RAW_DATA_DIR / info["file"]
    if not file_path.exists():
        raise FileNotFoundError(f"File for {ticker} not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Convert 'time' (assumed Unix timestamp) to datetime and extract the date
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
    
    # Ensure the required columns exist; if ROC is missing, compute it.
    if 'close' not in df.columns:
        raise ValueError(f"Missing 'close' column in {ticker} data.")
    if 'ROC' not in df.columns:
        df['ROC'] = df['close'].pct_change() * 100  # compute percent change in percent
    # ATR and RSI: if missing, they will remain NaN.
    for col in ['ATR', 'RSI']:
        if col not in df.columns:
            df[col] = np.nan

    # Select only the columns of interest: date, close, ROC, ATR, RSI
    df = df[['date', 'close', 'ROC', 'ATR', 'RSI']]
    
    # Rename columns to have the ticker prefix
    df = df.rename(columns={
        'close': f'{ticker}_price',
        'ROC': f'{ticker}_roc',
        'ATR': f'{ticker}_atr',
        'RSI': f'{ticker}_rsi'
    })
    
    # Drop duplicate dates if any and sort by date
    df = df.drop_duplicates(subset=['date']).sort_values(by='date')
    
    return df

def merge_ticker_data(tickers_info: dict) -> pd.DataFrame:
    """
    Load and merge data for all tickers on the 'date' column.
    """
    merged_df = None
    for ticker, info in tickers_info.items():
        df = load_and_process_ticker(ticker, info)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='date', how='inner')
    # Drop any rows with missing values (ensure a complete feature vector)
    merged_df = merged_df.dropna()
    return merged_df

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the merged dataframe, compute derived features:
      - yield_spread = TLT_price - SHY_price
      - market_health = (SPY_roc * 0.4) + (SPY_rsi * 0.3) + ((1 / SPY_atr) * 0.3)
    """
    df = df.copy()
    # Compute derived features
    df['yield_spread'] = df['TLT_price'] - df['SHY_price']
    
    # Avoid division by zero for SPY_atr: set a floor (e.g., 0.01)
    df['SPY_atr_adj'] = df['SPY_atr'].replace(0, 0.01)
    df['market_health'] = (df['SPY_roc'] * 0.4) + (df['SPY_rsi'] * 0.3) + ((1 / df['SPY_atr_adj']) * 0.3)
    df = df.drop(columns=['SPY_atr_adj'])
    
    return df

def generate_labels(df: pd.DataFrame) -> pd.Series:
    """
    Generate pseudo-labels for the economic cycle using rule-based logic.
    """
    def label_row(row):
        if row['yield_spread'] < 0 and row['market_health'] > 60:
            return "Recession"
        elif row['market_health'] > 70:
            return "Early"
        elif row['market_health'] > 50:
            return "Mid"
        else:
            return "Late"
    return df.apply(label_row, axis=1)

# --- Main Training Script ---

def main():
    print("Loading and merging ticker data...")
    try:
        merged_df = merge_ticker_data(tickers_info)
    except Exception as e:
        print(f"Error merging ticker data: {e}")
        return
    
    if merged_df.empty:
        print("Merged dataframe is empty. Check your raw data files.")
        return
    
    print(f"Merged data contains {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
    
    # Derive additional features
    df_features = derive_features(merged_df)
    
    # Generate pseudo-labels using the rule-based logic
    df_features['cycle_state'] = generate_labels(df_features)
    
    # For training, drop the 'date' column and any columns not used as features.
    X = df_features.drop(columns=['date', 'cycle_state'])
    y = df_features['cycle_state']
    
    print("Feature columns used for training:")
    print(X.columns.tolist())
    
    # Split the dataset into training and testing sets (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    
    # Save the trained model
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
