#!/usr/bin/env python3
# tests/test_sector_rotator.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.models.sector_rotator import SectorRotator
from src.utils.config import ConfigManager

class TestSectorRotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and configurations."""
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Test configuration
        cls.config = {
            'min_volume': 1000000,
            'min_relative_strength': 0.7,
            'momentum_threshold': 0.5,
            'technical_weights': {
                'trend': 0.4,
                'momentum': 0.3,
                'volatility': 0.3
            }
        }
        
        # Initialize rotator
        cls.rotator = SectorRotator(cls.config)
        
        # Mock response for technical indicators
        cls.mock_indicator_response = {
            'close': 100,
            'volume': 2000000,
            'RSI': 60,
            'TEMA20': 98,
            'TEMA50': 95,
            'TEMA20_1d_ago': 97,
            'TEMA50_1d_ago': 96,
            'ATR': 2,
            'VHF': 0.5
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)

    def test_initialization(self):
        """Test proper initialization of SectorRotator."""
        self.assertIsNotNone(self.rotator)
        self.assertEqual(self.rotator.config['min_volume'], 1000000)
        self.assertEqual(len(self.rotator.sectors), 11)  # Check all sectors are loaded
        self.assertIn('XLK', self.rotator.sectors)

    @patch('tradingview_ta.TA_Handler')
    def test_technical_indicators(self, mock_ta_handler):
        """Test technical indicator retrieval and processing."""
        # Setup mock
        mock_analysis = MagicMock()
        mock_analysis.indicators = self.mock_indicator_response
        mock_analysis.summary = {'RECOMMENDATION': 'BUY'}
        mock_ta_handler.return_value.get_analysis.return_value = mock_analysis

        # Test indicator retrieval
        indicators = self.rotator.get_technical_indicators('XLK')
        
        self.assertIsNotNone(indicators)
        self.assertEqual(indicators['symbol'], 'XLK')
        self.assertGreater(indicators['relative_strength'], 0)
        self.assertEqual(indicators['recommendation'], 'BUY')

    def test_ma_cross_detection(self):
        """Test moving average crossover detection."""
        # Test golden cross
        golden = self.rotator._detect_ma_cross(100, 98, 97, 99)
        self.assertEqual(golden, 'GOLDEN')
        
        # Test death cross
        death = self.rotator._detect_ma_cross(97, 99, 100, 98)
        self.assertEqual(death, 'DEATH')
        
        # Test no cross
        none = self.rotator._detect_ma_cross(100, 95, 99, 94)
        self.assertEqual(none, 'NONE')

    @patch('tradingview_ta.TA_Handler')
    def test_sector_analysis(self, mock_ta_handler):
        """Test individual sector analysis."""
        # Setup mock
        mock_analysis = MagicMock()
        mock_analysis.indicators = self.mock_indicator_response
        mock_analysis.summary = {'RECOMMENDATION': 'BUY'}
        mock_ta_handler.return_value.get_analysis.return_value = mock_analysis

        # Test sector analysis
        result = self.rotator.analyze_sector('XLK')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'XLK')
        self.assertEqual(result['sector'], 'Technology')
        self.assertTrue('relative_strength' in result)
        self.assertTrue('rsi' in result)
        self.assertTrue('ma_cross' in result)
        self.assertTrue('recommendation' in result)

    @patch('tradingview_ta.TA_Handler')
    def test_sector_scan(self, mock_ta_handler):
        """Test full sector scan functionality."""
        # Setup mock
        mock_analysis = MagicMock()
        mock_analysis.indicators = self.mock_indicator_response
        mock_analysis.summary = {'RECOMMENDATION': 'BUY'}
        mock_ta_handler.return_value.get_analysis.return_value = mock_analysis

        # Test full scan
        opportunities = self.rotator.scan_sectors()
        
        self.assertIsNotNone(opportunities)
        self.assertTrue(isinstance(opportunities, list))
        if len(opportunities) > 0:
            self.assertTrue('symbol' in opportunities[0])
            self.assertTrue('sector' in opportunities[0])
            self.assertTrue('relative_strength' in opportunities[0])

    def test_rotation_recommendations(self):
        """Test rotation recommendations generation."""
        with patch.object(SectorRotator, 'scan_sectors') as mock_scan:
            # Setup mock scan results
            mock_scan.return_value = [
                {
                    'symbol': 'XLK',
                    'sector': 'Technology',
                    'relative_strength': 1.2,
                    'rsi': 65,
                    'ma_cross': 'GOLDEN',
                    'recommendation': 'STRONG_BUY'
                },
                {
                    'symbol': 'XLF',
                    'sector': 'Financials',
                    'relative_strength': 0.9,
                    'rsi': 45,
                    'ma_cross': 'NONE',
                    'recommendation': 'NEUTRAL'
                }
            ]

            # Test recommendations
            recommendations = self.rotator.get_rotation_recommendations()
            
            self.assertIsNotNone(recommendations)
            self.assertTrue('timestamp' in recommendations)
            self.assertTrue('opportunities' in recommendations)
            self.assertTrue('top_sectors' in recommendations)
            self.assertTrue('avoid_sectors' in recommendations)

    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test invalid symbol
        with patch('tradingview_ta.TA_Handler') as mock_ta:
            mock_ta.side_effect = Exception("API Error")
            result = self.rotator.get_technical_indicators('INVALID')
            self.assertIsNone(result)

        # Test minimum threshold filtering
        with patch.object(SectorRotator, 'get_technical_indicators') as mock_indicators:
            mock_indicators.return_value = {
                'symbol': 'XLK',
                'price': 100,
                'volume': 100,  # Below minimum
                'rsi': 50,
                'relative_strength': 0.5,  # Below minimum
                'recommendation': 'NEUTRAL'
            }
            result = self.rotator.analyze_sector('XLK')
            self.assertIsNone(result)

    def test_performance_scoring(self):
        """Test sector performance scoring mechanism."""
        test_opportunities = [
            {
                'symbol': 'XLK',
                'relative_strength': 1.2,
                'rsi': 65,
                'ma_cross': 'GOLDEN'
            },
            {
                'symbol': 'XLF',
                'relative_strength': 0.9,
                'rsi': 45,
                'ma_cross': 'NONE'
            }
        ]

        with patch.object(SectorRotator, 'scan_sectors') as mock_scan:
            mock_scan.return_value = test_opportunities
            recommendations = self.rotator.get_rotation_recommendations()
            
            # Verify scoring logic
            self.assertTrue(len(recommendations['top_sectors']) > 0)
            self.assertEqual(recommendations['top_sectors'][0], 'XLK')

    def test_indicator_validation(self):
        """Test validation of technical indicators."""
        # Test with missing indicators
        incomplete_indicators = self.mock_indicator_response.copy()
        del incomplete_indicators['RSI']
        
        with patch('tradingview_ta.TA_Handler') as mock_ta:
            mock_analysis = MagicMock()
            mock_analysis.indicators = incomplete_indicators
            mock_analysis.summary = {'RECOMMENDATION': 'BUY'}
            mock_ta.return_value.get_analysis.return_value = mock_analysis
            
            indicators = self.rotator.get_technical_indicators('XLK')
            self.assertEqual(indicators['rsi'], 50)  # Should use default value

    def test_historical_performance(self):
        """Test historical performance analysis."""
        # Create synthetic historical data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        historical_data = pd.DataFrame({
            'date': dates,
            'close': np.random.random(len(dates)) * 100 + 100,
            'volume': np.random.random(len(dates)) * 1000000,
            'rsi': np.random.random(len(dates)) * 100
        })

        # Test with synthetic data
        with patch.object(SectorRotator, 'get_technical_indicators') as mock_indicators:
            mock_indicators.return_value = {
                'symbol': 'XLK',
                'price': historical_data['close'].iloc[-1],
                'volume': historical_data['volume'].iloc[-1],
                'rsi': historical_data['rsi'].iloc[-1],
                'relative_strength': 1.1,
                'recommendation': 'BUY'
            }
            
            result = self.rotator.analyze_sector('XLK')
            self.assertIsNotNone(result)
            self.assertEqual(result['symbol'], 'XLK')

if __name__ == '__main__':
    unittest.main()