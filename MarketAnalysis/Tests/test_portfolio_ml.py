#!/usr/bin/env python3
# tests/test_portfolio_ml.py

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from src.models.portfolio_ml import PortfolioML
from src.models.sector_rotator import SectorRotator
from src.models.cycle_classifier import CycleClassifier
from src.data.data_processor import DataProcessor
from src.utils.config import ConfigManager

class TestPortfolioML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample data."""
        # Create temporary directory for test data
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.data_dir = cls.test_dir / "data"
        cls.data_dir.mkdir(parents=True)
        
        # Create test configuration
        cls.config = ConfigManager()
        cls.config.set('paths.data_dir', str(cls.data_dir))
        
        # Generate sample data
        cls._generate_sample_data()
        
        # Initialize components
        cls.portfolio = PortfolioML()
        cls.data_processor = DataProcessor(str(cls.data_dir))

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)

    @classmethod
    def _generate_sample_data(cls):
        """Generate sample market data for testing."""
        dates = pd.date_range(
            start='2023-01-01',
            end='2024-12-31',
            freq='D'
        )
        
        for symbol in ['XLK', 'XLF', 'XLE']:
            # Generate synthetic price data
            np.random.seed(42)  # For reproducibility
            prices = np.random.random(len(dates)) * 10 + 100
            volumes = np.random.random(len(dates)) * 1000000 + 500000
            
            df = pd.DataFrame({
                'time': [int(d.timestamp()) for d in dates],
                'date': dates,
                'open': prices * (1 + np.random.random(len(dates)) * 0.02),
                'high': prices * (1 + np.random.random(len(dates)) * 0.04),
                'low': prices * (1 - np.random.random(len(dates)) * 0.02),
                'close': prices,
                'Volume': volumes,
                'Upper': prices * 1.05,
                'Basis': prices,
                'Lower': prices * 0.95,
                'DEMA': prices * (1 + np.random.random(len(dates)) * 0.01),
                'HMA': prices * (1 + np.random.random(len(dates)) * 0.01),
                'VPT': np.cumsum(np.random.random(len(dates))),
                'KVO': np.random.random(len(dates)) * 100,
                'ROC': np.random.random(len(dates)) * 5,
                'ROC.1': np.random.random(len(dates)) * 5,
                'ROC.2': np.random.random(len(dates)) * 5,
                'PPO': np.random.random(len(dates)) * 2,
                'Signal.1': np.random.random(len(dates)) * 2
            })
            
            # Save to test directory
            file_path = cls.data_dir / f"AMEX_{symbol} 1D.csv"
            df.to_csv(file_path, index=False)

    def test_data_loading(self):
        """Test data loading functionality."""
        df = self.data_processor.load_sector_data('XLK')
        self.assertIsNotNone(df)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(len(df) > 0)
        required_columns = ['time', 'open', 'high', 'low', 'close', 'Volume']
        for col in required_columns:
            self.assertIn(col, df.columns)

    def test_technical_indicators(self):
        """Test technical indicator calculations."""
        df = self.data_processor.load_sector_data('XLK')
        df_with_indicators = self.data_processor.calculate_technical_indicators(df)
        
        # Check if indicators were calculated
        expected_indicators = ['rsi', 'macd', 'macd_signal', 'rolling_std']
        for indicator in expected_indicators:
            self.assertIn(indicator, df_with_indicators.columns)
        
        # Check if values are within expected ranges
        self.assertTrue(all(df_with_indicators['rsi'].between(0, 100)))

    def test_model_training(self):
        """Test model training and predictions."""
        # Prepare data
        features, target = self.portfolio.prepare_sector_data('XLK')
        
        # Check features and target
        self.assertIsNotNone(features)
        self.assertIsNotNone(target)
        self.assertTrue(len(features) > 0)
        self.assertEqual(len(features), len(target))
        
        # Test model training
        metrics = self.portfolio.train_sector_model('XLK')
        self.assertIsNotNone(metrics)
        self.assertIn('test_r2', metrics)
        self.assertIn('mae', metrics)

    def test_sector_rotation(self):
        """Test sector rotation analysis."""
        rotation = self.portfolio.get_rotation_recommendations()
        
        self.assertIsNotNone(rotation)
        self.assertIn('timestamp', rotation)
        self.assertIn('sector_data', rotation)
        
        # Check sector data structure
        sector_data = rotation['sector_data']
        self.assertTrue(isinstance(sector_data, dict))
        self.assertTrue(len(sector_data) > 0)

    def test_cycle_classification(self):
        """Test economic cycle classification."""
        classifier = CycleClassifier()
        cycle_state = classifier.predict_cycle_state()
        
        self.assertIsNotNone(cycle_state)
        self.assertIn('predicted_state', cycle_state)
        self.assertIn('confidence', cycle_state)
        
        # Check valid cycle state
        valid_states = ['Early', 'Mid', 'Late', 'Recession']
        self.assertIn(cycle_state['predicted_state'], valid_states)

    def test_feature_importance(self):
        """Test feature importance analysis."""
        self.portfolio.analyze_all_sectors()
        
        # Check feature importance for each sector
        for sector in ['XLK', 'XLF', 'XLE']:
            importance = self.portfolio.feature_importance.get(sector)
            self.assertIsNotNone(importance)
            self.assertTrue(isinstance(importance, pd.DataFrame))
            self.assertIn('feature', importance.columns)
            self.assertIn('importance', importance.columns)

    def test_correlation_analysis(self):
        """Test sector correlation analysis."""
        correlations = self.data_processor.get_sector_correlations(
            ['XLK', 'XLF', 'XLE']
        )
        
        self.assertIsNotNone(correlations)
        self.assertTrue(isinstance(correlations, pd.DataFrame))
        self.assertEqual(correlations.shape[0], correlations.shape[1])

    def test_market_regime(self):
        """Test market regime analysis."""
        df = self.data_processor.prepare_analysis_data('XLK')
        regime_features = self.data_processor.get_market_regime_features(df)
        
        self.assertIsNotNone(regime_features)
        self.assertTrue(isinstance(regime_features, dict))
        expected_features = ['trend_strength', 'volatility_regime', 'momentum_score']
        for feature in expected_features:
            self.assertIn(feature, regime_features)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid symbol
        with self.assertRaises(Exception):
            self.data_processor.load_sector_data('INVALID')
        
        # Test missing required columns
        invalid_df = pd.DataFrame({'time': [], 'close': []})
        with self.assertRaises(ValueError):
            self.portfolio.validate_columns(invalid_df, 'TEST')

if __name__ == '__main__':
    unittest.main()