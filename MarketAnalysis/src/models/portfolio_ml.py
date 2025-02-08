#!/usr/bin/env python3
# src/models/portfolio_ml.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Union

# Import live data functions (if used)
from src.data.live_data import get_live_options_snapshot, process_etf_options
# Import the ML cycle classifier (make sure it's correctly aliased in __init__.py)
from src.models.cycle_classifier_ml import EconomicCycleStateClassifierML
# Import the sector rotator for technical scanning
from src.models.sector_rotator import SectorRotatorWizard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SectorAnalyzer:
    def __init__(self, sectors: List[str]):
        self.sectors = sectors
        self.models = {}
        self.feature_importance = {}
        self.predictions = {}
        self.metrics = {}
        # Ticker configuration for the cycle classifier
        tickers_info = {
            "SPY": {"exchange": "AMEX", "file": "AMEX_SPY 1D.csv"},
            "GLD": {"exchange": "AMEX", "file": "AMEX_GLD 1D.csv"},
            "TIP": {"exchange": "AMEX", "file": "AMEX_TIP 1D.csv"},
            "UUP": {"exchange": "AMEX", "file": "AMEX_UUP 1D.csv"},
            "SHY": {"exchange": "NASDAQ", "file": "NASDAQ_SHY 1D.csv"},
            "TLT": {"exchange": "NASDAQ", "file": "NASDAQ_TLT 1D.csv"}
        }
        self.cycle_classifier = EconomicCycleStateClassifierML(tickers_info, data_dir="data")
        self.sector_rotator = SectorRotatorWizard({
            'min_volume': 1000,
            'min_relative_strength': 0.7
        })

    def validate_columns(self, df: pd.DataFrame, sector: str) -> None:
        required_columns = ['time', 'open', 'high', 'low', 'close', 'Upper', 'Basis', 
                            'Lower', 'DEMA', 'HMA', 'Volume', 'VPT', 'KVO', 'ROC', 
                            'ROC.1', 'ROC.2', 'PPO', 'Signal.1']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {sector}: {missing_columns}")

    def calculate_quarterly_metrics(self, group: pd.DataFrame) -> pd.Series:
        metrics = {}
        try:
            metrics['quarterly_return'] = (group['close'].iloc[-1] / group['close'].iloc[0] - 1) * 100
            metrics['avg_band_width'] = ((group['Upper'] - group['Lower']) / group['Basis']).mean()
            metrics['dema_trend'] = (group['DEMA'].iloc[-1] / group['DEMA'].iloc[0] - 1) * 100
            metrics['hma_trend'] = (group['HMA'].iloc[-1] / group['HMA'].iloc[0] - 1) * 100
            metrics['vpt_trend'] = group['VPT'].diff().mean()
            metrics['kvo_trend'] = group['KVO'].diff().mean()
            for col in ['ROC', 'ROC.1', 'ROC.2']:
                metrics[f'{col}_mean'] = group[col].mean()
                metrics[f'{col}_std'] = group[col].std()
            metrics['trend_strength'] = ((group['close'] > group['DEMA']).mean() +
                                         (group['close'] > group['HMA']).mean()) / 2
        except Exception as e:
            print(f"Error in calculate_quarterly_metrics: {e}")
            raise
        return pd.Series(metrics)

    def prepare_sector_data(self, sector: str) -> Tuple[pd.DataFrame, pd.Series]:
        exchange = 'NASDAQ' if sector == 'SMH' else 'AMEX'
        file_path = os.path.join("data", "raw", f"{exchange}_{sector} 1D.csv")
        df = pd.read_csv(file_path)
        self.validate_columns(df, sector)
        df['date'] = pd.to_datetime(df['time'], unit='s')
        quarterly_data = df.groupby(df['date'].dt.to_period('Q')).apply(self.calculate_quarterly_metrics)
        quarterly_data = quarterly_data.reset_index()
        target = quarterly_data['quarterly_return']
        features = quarterly_data.drop(['quarterly_return', 'date'], axis=1)
        return features, target

    def train_sector_model(self, sector: str) -> Dict:
        features, target = self.prepare_sector_data(sector)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, max_depth=12, 
                                      min_samples_split=5, min_samples_leaf=2,
                                      random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.models[sector] = model
        self.metrics[sector] = {
            'train_r2': model.score(X_train, y_train),
            'test_r2': model.score(X_test, y_test),
            'mae': np.mean(np.abs(y_pred - y_test)),
            'rmse': np.sqrt(np.mean((y_pred - y_test)**2))
        }
        self.feature_importance[sector] = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        self.predictions[sector] = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Error': y_pred - y_test.values
        })
        return self.metrics[sector]

    def analyze_all_sectors(self) -> None:
        print(f"Processing {len(self.sectors)} sectors: {', '.join(self.sectors)}")
        for sector in self.sectors:
            print(f"  Processing {sector}...", end='\r')
            self.train_sector_model(sector)
        print(" " * 50, end='\r')

    def get_live_sector_data(self) -> Dict:
        live_data = {}
        for sector in self.sectors:
            try:
                snapshot = get_live_options_snapshot(sector)
                processed = process_etf_options(sector, snapshot)
                if processed:
                    live_data[sector] = processed
            except Exception as e:
                logging.error(f"Error getting live data for {sector}: {e}")
        return live_data

    def has_seasonality_boost(self, sector: str) -> bool:
        current_month = datetime.now().month
        boost_months = [2, 11]
        return current_month in boost_months

    def calculate_expected_hold_time(self, sector: str) -> float:
        """
        Calculate expected hold time (in days) by blending a historical average hold time
        with an ML-predicted trend duration, then adjust dynamically based on technical indicators.
        """
        # Base values
        historical_average = 45.0  # days
        ml_trend_duration = 30.0   # days
        alpha = 0.5                # blending factor
        base_hold_time = alpha * historical_average + (1 - alpha) * ml_trend_duration
        
        # Retrieve dynamic technical data from the sector rotator
        technical_data = next((x for x in self.sector_rotator.scan_sectors() if x['symbol'] == sector), {})
        if technical_data:
            # Use RSI to create a dynamic factor: if RSI is 50, factor is 1.0; higher RSI increases the factor
            rsi = technical_data.get('rsi', 50)
            rsi_factor = 1.0 + ((rsi - 50) / 100.0)  # For example, RSI of 70 gives 1.2; RSI of 30 gives 0.8
            # Use volatility if available to adjust further: for demonstration, if volatility < 1, factor is 1.0; otherwise 0.9
            volatility = technical_data.get('volatility', 1.0)
            vol_factor = 1.0 if volatility < 1 else 0.9
            dynamic_trend_strength = rsi_factor * vol_factor
        else:
            dynamic_trend_strength = 1.0
        
        expected_hold_time = base_hold_time * dynamic_trend_strength
        return expected_hold_time

    def calculate_expected_strike_price(self, sector: str) -> Union[float, None]:
        technical = next((x for x in self.sector_rotator.scan_sectors() if x['symbol'] == sector), {})
        if sector in self.predictions and not self.predictions[sector].empty:
            latest_pred = self.predictions[sector].iloc[-1].to_dict()
        else:
            latest_pred = {}
        if technical and 'price' in technical and latest_pred and 'Predicted' in latest_pred:
            price = technical['price']
            predicted_return = latest_pred['Predicted']
            target_price = price * (1 + predicted_return / 100.0)
            return target_price
        else:
            return None

    def get_rotation_recommendations(self) -> Dict:
        cycle_state = self.cycle_classifier.predict_cycle_state()
        technical_data = self.sector_rotator.scan_sectors()
        live_data = self.get_live_sector_data()
        recommendations = {}
        for sector in self.sectors:
            if not self.has_seasonality_boost(sector):
                logging.info(f"Sector {sector} does not have a seasonality boost; skipping.")
                continue
            technical = next((x for x in technical_data if x['symbol'] == sector), {})
            sector_data = {
                'ml_metrics': self.metrics.get(sector, {}),
                'technical_indicators': technical,
                'live_options': live_data.get(sector, {}),
                'feature_importance': self.feature_importance.get(sector, pd.DataFrame()).head(3).to_dict('records'),
                'latest_prediction': (self.predictions.get(sector, pd.DataFrame()).iloc[-1].to_dict() 
                                        if sector in self.predictions and not self.predictions[sector].empty else {}),
                'expected_hold_time': self.calculate_expected_hold_time(sector),
                'expected_strike_price': self.calculate_expected_strike_price(sector)
            }
            recommendations[sector] = sector_data
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cycle_state': cycle_state,
            'sector_data': recommendations
        }

def main():
    print("Portfolio_ML Analysis Starting...")
    print("-" * 50)
    
    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLP', 'XLI', 'XLB', 'XLU', 'XLY', 'SMH', 'XRT']
    analyzer = SectorAnalyzer(sectors)
    
    analyzer.analyze_all_sectors()
    
    recommendations = analyzer.get_rotation_recommendations()
    
    cycle_state_info = recommendations.get("cycle_state", {})
    if cycle_state_info:
        print(f"\nML Predicted Economic Cycle State: {cycle_state_info.get('predicted_state', 'N/A')}")
        print(f"Cycle Confidence Distribution: {cycle_state_info.get('confidence', {})}")
    print("-" * 50)
    
    print("\nSector Rotation Recommendations:")
    for sector, data in recommendations.get("sector_data", {}).items():
        ml_metrics = data.get('ml_metrics', {})
        latest_pred = data.get('latest_prediction', {})
        technical = data.get('technical_indicators', {})
        expected_hold_time = data.get('expected_hold_time')
        expected_strike_price = data.get('expected_strike_price')
        
        print(f"\n{sector}:")
        print(f"ML Model Accuracy (R²): {ml_metrics.get('test_r2', 0):.3f}")
        print(f"Predicted Return: {latest_pred.get('Predicted', 0):.2f}%")
        if technical:
            print(f"Technical Signal: {technical.get('recommendation', 'NEUTRAL')}")
            print(f"Relative Strength: {technical.get('relative_strength', 0):.2f}")
        print(f"Expected Hold Time (days): {expected_hold_time:.1f}")
        if expected_strike_price is not None:
            print(f"Expected Strike Price: ${expected_strike_price:.2f}")
        else:
            print("Expected Strike Price: N/A")
    
    results_df = pd.DataFrame(recommendations['sector_data']).T
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sector_rotation_recommendations.csv")
    results_df.to_csv(output_file)
    
    print("\nAnalysis complete. Results saved to 'sector_rotation_recommendations.csv'")

if __name__ == "__main__":
    main()
