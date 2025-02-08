#!/usr/bin/env python3
# src/models/cycle_classifier.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tradingview_ta import TA_Handler, Interval
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple

class CycleClassifier:
    """
    A class to classify the current economic cycle state using market indicators
    and machine learning.
    """
    
    def __init__(self):
        """Initialize the CycleClassifier with necessary setup."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/cycle_classifier.log'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        
        # Define indicators to track
        self.cycle_indicators = {
            'TNX': {'symbol': 'TLT', 'exchange': 'NASDAQ'},  # Treasury yields
            'UST2Y': {'symbol': 'SHY', 'exchange': 'NASDAQ'},
            'SPY': {'symbol': 'SPY', 'exchange': 'AMEX'},
            'UUP': {'symbol': 'UUP', 'exchange': 'AMEX'},    # Dollar strength
            'GLD': {'symbol': 'GLD', 'exchange': 'AMEX'},    # Gold
            'TIP': {'symbol': 'TIP', 'exchange': 'AMEX'},    # Inflation expectations
            'XLF': {'symbol': 'XLF', 'exchange': 'AMEX'},    # Financials
            'XLK': {'symbol': 'XLK', 'exchange': 'AMEX'},    # Technology
            'XLE': {'symbol': 'XLE', 'exchange': 'AMEX'},    # Energy
            'XLU': {'symbol': 'XLU', 'exchange': 'AMEX'},    # Utilities
            'XLP': {'symbol': 'XLP', 'exchange': 'AMEX'},    # Consumer Staples
            'XLV': {'symbol': 'XLV', 'exchange': 'AMEX'}     # Healthcare
        }
        
        # Initialize and train model
        self.feature_names, self.sample_data = self._generate_sample_data()
        self._train_model()

    def get_indicator_data(self, symbol: str, 
                         screener: str = "america") -> Optional[Dict]:
        """
        Get technical indicator data for a symbol.
        
        Args:
            symbol (str): The symbol to analyze
            screener (str): The market screener to use
            
        Returns:
            Dict or None: Dictionary of indicator data or None if error
        """
        # Special handling for SPY (multiple possible symbols)
        if symbol == "SPY":
            candidates = [
                ("SPY", "AMEX"),
                ("US500", "CAPITALCOM"),
                ("^GSPC", "INDEX"),
                ("SPX500USD", "FOREXCOM")
            ]
            for cand_symbol, exchange in candidates:
                try:
                    handler = TA_Handler(
                        symbol=cand_symbol,
                        screener=screener,
                        exchange=exchange,
                        interval=Interval.INTERVAL_1_DAY
                    )
                    analysis = handler.get_analysis()
                    return self._process_analysis(analysis)
                except Exception as e:
                    logging.error(
                        f"Error getting data for {cand_symbol} "
                        f"on {exchange}: {str(e)}"
                    )
            return None
            
        # Handle other symbols
        indicator_info = self.cycle_indicators.get(symbol)
        if not indicator_info:
            return None
            
        try:
            handler = TA_Handler(
                symbol=indicator_info['symbol'],
                screener=screener,
                exchange=indicator_info['exchange'],
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            return self._process_analysis(analysis)
        except Exception as e:
            logging.error(
                f"Error getting data for {indicator_info['symbol']} "
                f"on {indicator_info['exchange']}: {str(e)}"
            )
            return None

    def _process_analysis(self, analysis: Any) -> Optional[Dict]:
        """
        Process the raw analysis data into standardized metrics.
        
        Returns:
            Dict or None: Processed metrics or None if error
        """
        if not analysis:
            return None
            
        price = analysis.indicators.get('close')
        if not price:
            return None
            
        # Calculate volatility
        atr = analysis.indicators.get('ATR', 0)
        volatility = atr / price if atr else 0.01
        
        # Calculate momentum
        prev_close = analysis.indicators.get('previous_close', price)
        roc = ((price - prev_close) / prev_close * 100) if prev_close else 0
        
        return {
            'price': price,
            'volatility': volatility,
            'rsi': analysis.indicators.get('RSI', 50),
            'momentum': roc
        }

    def _generate_sample_data(self) -> Tuple[List[str], pd.DataFrame]:
        """
        Generate synthetic data for initial model training.
        
        Returns:
            Tuple[List[str], pd.DataFrame]: Feature names and sample data
        """
        n_samples = 1000
        data = []
        feature_names = []
        
        # Generate feature names
        for symbol in self.cycle_indicators.keys():
            for metric in ['price', 'volatility', 'rsi', 'momentum']:
                feature_names.append(f"{symbol}_{metric}")
        
        feature_names.extend(['yield_spread', 'yield_curve_steepness', 'market_health'])
        
        # Generate synthetic samples
        for _ in range(n_samples):
            sample = {}
            for feature in feature_names:
                if 'price' in feature:
                    sample[feature] = np.random.normal(100, 5)
                elif 'volatility' in feature:
                    sample[feature] = abs(np.random.normal(0.02, 0.005))
                elif 'rsi' in feature:
                    sample[feature] = np.random.uniform(30, 70)
                else:
                    sample[feature] = np.random.normal(0, 1)
            
            # Calculate derived features
            sample['yield_spread'] = sample['TNX_price'] - sample['UST2Y_price']
            sample['yield_curve_steepness'] = abs(sample['yield_spread'])
            sample['market_health'] = (
                sample['SPY_momentum'] * 0.4 +
                sample['SPY_rsi'] * 0.3 +
                (1 / max(sample['SPY_volatility'], 0.01)) * 0.3
            )
            
            data.append(sample)
        
        return feature_names, pd.DataFrame(data, columns=feature_names)

    def _train_model(self) -> None:
        """Train the classifier model on synthetic data."""
        X = self.sample_data
        y = np.zeros(len(X))
        
        # Assign synthetic cycle states based on market conditions
        for i in range(len(X)):
            if X['yield_spread'].iloc[i] < -0.5:
                y[i] = 3  # Recession
            elif X['market_health'].iloc[i] > 60:
                y[i] = 1  # Mid-cycle
            elif X['SPY_momentum'].iloc[i] > 0.5:
                y[i] = 0  # Early cycle
            else:
                y[i] = 2  # Late cycle
        
        # Fit scaler and model
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        logging.info("Model and scaler fitted with synthetic sample data")

    def predict_cycle_state(self) -> Optional[Dict]:
        """
        Predict the current economic cycle state.
        
        Returns:
            Dict or None: Prediction results with confidence metrics
        """
        try:
            # Get current market features
            features_df = self.get_cycle_features()
            if features_df is None or features_df.empty:
                logging.error("Failed to get sufficient cycle features")
                return None
            
            # Ensure proper feature order
            features_df = features_df[self.feature_names]
            
            # Make prediction
            scaled_features = self.scaler.transform(features_df)
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            cycle_states = ['Early', 'Mid', 'Late', 'Recession']
            max_prob = np.max(probabilities)
            sorted_probs = np.sort(probabilities)
            prob_gap = max_prob - sorted_probs[-2]
            
            state = int(np.argmax(probabilities))
            
            # Prepare results
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_state': cycle_states[state],
                'confidence': {
                    'probability': float(max_prob),
                    'probability_gap': float(prob_gap),
                    'prediction_strength': 'High' if prob_gap > 0.3 else 'Medium' if prob_gap > 0.15 else 'Low'
                },
                'state_probabilities': {
                    state: float(prob) for state, prob in zip(cycle_states, probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return None

    def get_cycle_features(self) -> pd.DataFrame:
        """
        Get current market features for cycle classification.
        
        Returns:
            pd.DataFrame: Current market features
        """
        features = {}
        
        # Get data for each indicator
        for symbol in self.cycle_indicators.keys():
            data = self.get_indicator_data(symbol)
            if not data:
                data = {
                    'price': 100,
                    'volatility': 0.01,
                    'rsi': 50,
                    'momentum': 0
                }
            
            features[f"{symbol}_price"] = data['price']
            features[f"{symbol}_volatility"] = data['volatility']
            features[f"{symbol}_rsi"] = data['rsi']
            features[f"{symbol}_momentum"] = data['momentum']
        
        # Calculate composite features
        features["yield_spread"] = (
            features.get("TNX_price", 0) - 
            features.get("UST2Y_price", 0)
        )
        features["yield_curve_steepness"] = abs(features["yield_spread"])
        
        # Market health score
        features["market_health"] = (
            features.get("SPY_momentum", 0) * 0.4 +
            features.get("SPY_rsi", 50) * 0.3 +
            (1 / max(features.get("SPY_volatility", 0.01), 0.01)) * 0.3
        )
        
        return pd.DataFrame([features], columns=self.feature_names)