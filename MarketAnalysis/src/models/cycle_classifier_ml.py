#!/usr/bin/env python3
# src/models/cycle_classifier_ml.py

import os
import pandas as pd
import joblib
import logging
from datetime import datetime
from src.data.data_processor import DataProcessor

class EconomicCycleStateClassifierML:
    """
    Uses a pre-trained ML model to predict the current economic cycle state.
    """
    def __init__(self, tickers: dict, data_dir: str = "data"):
        self.tickers = tickers
        self.data_processor = DataProcessor(data_dir=data_dir)
        self.model = None
        model_path = os.path.join("src", "models", "ml_cycle_state_model.pkl")
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"ML model not found at {model_path}. Please train the model first. Error: {e}")
            self.model = None

    def get_feature_vector(self) -> dict:
        feature_vector = {}
        for ticker, info in self.tickers.items():
            exchange = info.get("exchange", "AMEX")
            df = self.data_processor.load_sector_data(ticker, exchange=exchange)
            if df is None or df.empty:
                logging.warning(f"No data found for {ticker}")
                continue
            try:
                df_ind = self.data_processor.calculate_technical_indicators(df)
                # Instead of dropping all rows with NaN values, fill missing values
                df_ind = df_ind.fillna(method='ffill').fillna(method='bfill')
                if df_ind.empty:
                    logging.warning(f"After indicator calculation, no data for {ticker}")
                    continue
                last_row = df_ind.iloc[-1]
            except Exception as e:
                logging.error(f"Error processing indicators for {ticker}: {e}")
                continue
            # Extract features: price, ROC, ATR, and RSI.
            # Use the raw columns as produced by DataProcessor (adjust if needed).
            price = last_row.get("close")
            roc = last_row.get("ROC") if last_row.get("ROC") is not None else last_row.get("returns")
            atr = last_row.get("ATR")
            rsi = last_row.get("rsi")
            feature_vector[f"{ticker}_price"] = price
            feature_vector[f"{ticker}_roc"] = roc
            feature_vector[f"{ticker}_atr"] = atr
            feature_vector[f"{ticker}_rsi"] = rsi
        # Derived feature: yield_spread = TLT_price - SHY_price (if both exist)
        if "TLT_price" in feature_vector and "SHY_price" in feature_vector:
            feature_vector["yield_spread"] = feature_vector["TLT_price"] - feature_vector["SHY_price"]
        # Derived feature: market_health using SPY's features if available.
        if "SPY_roc" in feature_vector and "SPY_rsi" in feature_vector and "SPY_atr" in feature_vector:
            spy_atr = feature_vector["SPY_atr"] if feature_vector["SPY_atr"] and feature_vector["SPY_atr"] != 0 else 0.01
            feature_vector["market_health"] = (feature_vector["SPY_roc"] * 0.4) + (feature_vector["SPY_rsi"] * 0.3) + ((1 / spy_atr) * 0.3)
        return feature_vector

    def predict_cycle_state(self) -> dict:
        feature_vector = self.get_feature_vector()
        if not feature_vector:
            logging.error("Feature vector is empty. Cannot predict cycle state.")
            return {}
        if self.model is None:
            logging.error("No ML model loaded. Cannot predict cycle state.")
            return {}
        # Convert the feature vector into a one-row DataFrame.
        df_features = pd.DataFrame([feature_vector])
        try:
            prediction = self.model.predict(df_features)
            probabilities = self.model.predict_proba(df_features)
        except Exception as e:
            logging.error(f"Error during model prediction: {e}")
            return {}
        prob_dict = dict(zip(self.model.classes_, probabilities[0]))
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_state": prediction[0],
            "confidence": prob_dict,
            "feature_vector": feature_vector
        }
        return result

if __name__ == "__main__":
    # Define the ticker configuration.
    tickers_info = {
        "SPY": {"exchange": "AMEX", "file": "AMEX_SPY 1D.csv"},
        "GLD": {"exchange": "AMEX", "file": "AMEX_GLD 1D.csv"},
        "TIP": {"exchange": "AMEX", "file": "AMEX_TIP 1D.csv"},
        "UUP": {"exchange": "AMEX", "file": "AMEX_UUP 1D.csv"},
        "SHY": {"exchange": "NASDAQ", "file": "NASDAQ_SHY 1D.csv"},
        "TLT": {"exchange": "NASDAQ", "file": "NASDAQ_TLT 1D.csv"}
    }
    
    classifier_ml = EconomicCycleStateClassifierML(tickers_info, data_dir="data")
    result = classifier_ml.predict_cycle_state()
    print("\nML Predicted Economic Cycle State:", result.get("predicted_state", "N/A"))
    print("Confidence Distribution:", result.get("confidence", {}))
    print("Feature Vector:", result.get("feature_vector", {}))
