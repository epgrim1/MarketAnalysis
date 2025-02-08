#!/usr/bin/env python3
# src/models/cycle_classifier.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class EconomicCycleStateClassifier:
    """
    Classifies the current economic cycle state using raw data from various ETFs.
    Data is read from CSV files for SPY, GLD, TIP, UUP, SHY, and TLT.
    Derived indicators such as yield spread and a market health composite are computed,
    and a feature vector is built to simulate an economic cycle classification.
    """

    def __init__(self):
        self.tickers = {
            "SPY": {"exchange": "AMEX", "file": "AMEX_SPY 1D.csv"},
            "GLD": {"exchange": "AMEX", "file": "AMEX_GLD 1D.csv"},
            "TIP": {"exchange": "AMEX", "file": "AMEX_TIP 1D.csv"},
            "UUP": {"exchange": "AMEX", "file": "AMEX_UUP 1D.csv"},
            "SHY": {"exchange": "NASDAQ", "file": "NASDAQ_SHY 1D.csv"},
            "TLT": {"exchange": "NASDAQ", "file": "NASDAQ_TLT 1D.csv"}
        }
        # In production, train an ML model on historical data.
        # Here we simulate the cycle classification with rule-based logic.

    def read_csv_for_ticker(self, file_name: str) -> pd.DataFrame:
        file_path = os.path.join("data", "raw", file_name)
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def compute_indicators(self, df: pd.DataFrame) -> dict:
        # Assumes CSV has at least the columns: time, open, high, low, close, (optionally ATR, RSI, ROC).
        if df.empty or 'close' not in df.columns:
            return {}
        last_row = df.iloc[-1]
        price = last_row['close']
        if 'ROC' in df.columns:
            roc = last_row['ROC']
        elif len(df) >= 2:
            prev_price = df.iloc[-2]['close']
            roc = ((price - prev_price) / prev_price) * 100
        else:
            roc = np.nan
        rsi = last_row['RSI'] if 'RSI' in df.columns else np.nan
        atr = last_row['ATR'] if 'ATR' in df.columns else np.nan
        return {
            "price": price,
            "roc": roc,
            "rsi": rsi,
            "atr": atr
        }

    def get_features(self) -> dict:
        features = {}
        for ticker, info in self.tickers.items():
            df = self.read_csv_for_ticker(info["file"])
            indicators = self.compute_indicators(df)
            features[ticker] = indicators
            logging.info(f"Indicators for {ticker}: {indicators}")
        return features

    def derive_indicators(self, features: dict) -> dict:
        # Yield Spread: TLT price - SHY price
        if features.get("TLT") and features.get("SHY"):
            yield_spread = features["TLT"].get("price", np.nan) - features["SHY"].get("price", np.nan)
        else:
            yield_spread = np.nan
        # Market Health Composite using SPY data:
        if features.get("SPY"):
            spy = features["SPY"]
            volatility = spy.get("atr", 0.01) if spy.get("atr", 0) else 0.01
            market_health = (spy.get("roc", 0) * 0.4) + (spy.get("rsi", 50) * 0.3) + ((1 / volatility) * 0.3)
        else:
            market_health = np.nan
        return {"yield_spread": yield_spread, "market_health": market_health}

    def build_feature_vector(self, features: dict, derived: dict) -> pd.DataFrame:
        feature_vector = {}
        for ticker, inds in features.items():
            for key, val in inds.items():
                feature_vector[f"{ticker}_{key}"] = val
        feature_vector.update(derived)
        return pd.DataFrame([feature_vector])

    def simulate_cycle_state(self, feature_vector: dict) -> str:
        ys = feature_vector.get("yield_spread", 0)
        mh = feature_vector.get("market_health", 50)
        # Simple rule-based logic for demonstration.
        if ys < 0 and mh > 60:
            return "Recession"
        elif mh > 70:
            return "Early"
        elif mh > 50:
            return "Mid"
        else:
            return "Late"

    def predict_cycle_state(self) -> dict:
        features = self.get_features()
        derived = self.derive_indicators(features)
        df_features = self.build_feature_vector(features, derived)
        print("\nFeature Vector:")
        print(df_features)
        feature_vector_dict = df_features.iloc[0].to_dict()
        cycle_state = self.simulate_cycle_state(feature_vector_dict)
        # For demonstration, we simulate confidence based on market_health
        confidence = "High" if feature_vector_dict.get("market_health", 50) > 60 else "Medium"
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_state': cycle_state,
            'confidence': {
                'prediction_strength': confidence
            },
            'state_details': feature_vector_dict
        }
        return result

if __name__ == "__main__":
    classifier = EconomicCycleStateClassifier()
    result = classifier.predict_cycle_state()
    print("\nSimulated Economic Cycle State:", result['predicted_state'])
    print("\nConfidence:", result['confidence']['prediction_strength'])
    print("\nTickers and Indicators Being Used:")
    features = classifier.get_features()
    for ticker, inds in features.items():
        print(f"- {ticker}: {inds}")
    print("\nCycle classification complete.")
