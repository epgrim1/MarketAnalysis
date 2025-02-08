#!/usr/bin/env python3
# Portfolio_ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from LiveData import get_live_options_snapshot, process_etf_options
from SectorRotatorWizard import SectorRotatorWizard
from EconomicCycleStateClassifier import EconomicCycleStateClassifier

class SectorAnalyzer:
    def __init__(self, sectors: List[str]):
        self.sectors = sectors
        self.models = {}
        self.feature_importance = {}
        self.predictions = {}
        self.metrics = {}
        self.cycle_classifier = EconomicCycleStateClassifier()
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
            # Basic price metrics
            metrics['quarterly_return'] = (group['close'].iloc[-1] / group['close'].iloc[0] - 1) * 100
            
            # Technical indicators
            metrics['avg_band_width'] = ((group['Upper'] - group['Lower']) / group['Basis']).mean()
            metrics['dema_trend'] = (group['DEMA'].iloc[-1] / group['DEMA'].iloc[0] - 1) * 100
            metrics['hma_trend'] = (group['HMA'].iloc[-1] / group['HMA'].iloc[0] - 1) * 100
            metrics['vpt_trend'] = group['VPT'].diff().mean()
            metrics['kvo_trend'] = group['KVO'].diff().mean()
            
            # ROC metrics
            for col in ['ROC', 'ROC.1', 'ROC.2']:
                metrics[f'{col}_mean'] = group[col].mean()
                metrics[f'{col}_std'] = group[col].std()
            
            # Trend strength
            metrics['trend_strength'] = (
                (group['close'] > group['DEMA']).mean() +
                (group['close'] > group['HMA']).mean()
            ) / 2
            
        except Exception as e:
            print(f"Error in calculate_quarterly_metrics: {str(e)}")
            raise
            
        return pd.Series(metrics)

    def prepare_sector_data(self, sector: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Handle different exchanges
        exchange = 'NASDAQ' if sector == 'SMH' else 'AMEX'
        
        # Read sector data
        df = pd.read_csv(f'{exchange}_{sector} 1D.csv')
        self.validate_columns(df, sector)
        
        # Convert time to date
        df['date'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate quarterly metrics
        quarterly_data = df.groupby(df['date'].dt.to_period('Q')).apply(self.calculate_quarterly_metrics)
        quarterly_data = quarterly_data.reset_index()
        
        # Prepare features and target
        target = quarterly_data['quarterly_return']
        features = quarterly_data.drop(['quarterly_return', 'date'], axis=1)
        
        return features, target

    def train_sector_model(self, sector: str) -> Dict:
        features, target = self.prepare_sector_data(sector)
        
        # Split and train model
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=200, max_depth=12, 
                                    min_samples_split=5, min_samples_leaf=2,
                                    random_state=42)
        model.fit(X_train, y_train)
        
        # Store results
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
            print(f"  {sector}", end='\r')
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

    def get_rotation_recommendations(self) -> Dict:
        # Get economic cycle state
        cycle_state = self.cycle_classifier.predict_cycle_state()
        
        # Get sector technical data
        technical_data = self.sector_rotator.scan_sectors()
        
        # Get live options data
        live_data = self.get_live_sector_data()
        
        # Combine all data sources
        recommendations = {}
        for sector in self.sectors:
            sector_data = {
                'ml_metrics': self.metrics.get(sector, {}),
                'technical_indicators': next((x for x in technical_data if x['symbol'] == sector), {}),
                'live_options': live_data.get(sector, {}),
                'feature_importance': self.feature_importance.get(sector, pd.DataFrame()).head(3).to_dict('records'),
                'latest_prediction': self.predictions.get(sector, pd.DataFrame()).iloc[-1].to_dict() if sector in self.predictions else {}
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
    
    # Initialize analyzer
    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLP', 'XLI', 'XLB', 'XLU', 'XLY', 'SMH', 'XRT']
    analyzer = SectorAnalyzer(sectors)
    
    # Run ML analysis
    analyzer.analyze_all_sectors()
    
    # Get comprehensive recommendations
    recommendations = analyzer.get_rotation_recommendations()
    
    # Print cycle state
    print(f"\nEconomic Cycle State: {recommendations['cycle_state']['predicted_state']}")
    print(f"Confidence: {recommendations['cycle_state']['confidence']['prediction_strength']}")
    print("-" * 50)
    
    # Print sector recommendations
    print("\nSector Rotation Recommendations:")
    for sector, data in recommendations['sector_data'].items():
        ml_metrics = data['ml_metrics']
        latest_pred = data['latest_prediction']
        technical = data['technical_indicators']
        
        print(f"\n{sector}:")
        print(f"ML Model Accuracy (R²): {ml_metrics.get('test_r2', 0):.3f}")
        print(f"Predicted Return: {latest_pred.get('Predicted', 0):.2f}%")
        if technical:
            print(f"Technical Signal: {technical.get('recommendation', 'NEUTRAL')}")
            print(f"Relative Strength: {technical.get('relative_strength', 0):.2f}")
    
    # Save results
    results_df = pd.DataFrame(recommendations['sector_data']).T
    results_df.to_csv('sector_rotation_recommendations.csv')
    
    print("\nAnalysis complete. Results saved to 'sector_rotation_recommendations.csv'")

if __name__ == "__main__":
    main()