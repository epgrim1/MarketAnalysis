#!/usr/bin/env python3
# src/models/sector_rotator.py

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tradingview_ta import TA_Handler, Interval, Exchange
from typing import Dict, List, Optional

class SectorRotatorWizard:
    """
    A class to analyze and recommend sector rotations based on technical analysis
    and market conditions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SectorRotator with configuration settings.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - min_volume: Minimum trading volume
                - min_relative_strength: Minimum relative strength vs SPY
        """
        self.sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLY': 'Consumer Discretionary',
            'SMH': 'Semiconductors',
            'XRT': 'Retail'
        }
        self.config = config
        
        # Setup logging
        logging.basicConfig(
            filename='logs/sector_rotation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def get_technical_indicators(self, symbol: str) -> Optional[Dict]:
        """
        Get technical indicators for a given symbol using TradingView API.
        
        Args:
            symbol (str): The ticker symbol to analyze
            
        Returns:
            Dict or None: Dictionary of technical indicators or None if error
        """
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener="america",
                exchange="AMEX" if symbol != "SMH" else "NASDAQ",
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            
            # Get SPY data for relative strength
            spy_handler = TA_Handler(
                symbol="SPY",
                screener="america", 
                exchange="AMEX",
                interval=Interval.INTERVAL_1_DAY
            )
            spy_analysis = spy_handler.get_analysis()
            
            # Calculate relative strength
            try:
                relative_strength = (
                    analysis.indicators['close'] / 
                    analysis.indicators['close_1d_ago']
                ) / (
                    spy_analysis.indicators['close'] / 
                    spy_analysis.indicators['close_1d_ago']
                )
            except (ZeroDivisionError, KeyError):
                relative_strength = 1.0
            
            # Get moving averages
            tema_20 = analysis.indicators.get('TEMA20', None)
            tema_50 = analysis.indicators.get('TEMA50', None)
            tema_20_prev = analysis.indicators.get('TEMA20_1d_ago', None)
            tema_50_prev = analysis.indicators.get('TEMA50_1d_ago', None)
            
            return {
                'symbol': symbol,
                'price': analysis.indicators.get('close'),
                'volume': analysis.indicators.get('volume', 0),
                'rsi': analysis.indicators.get('RSI', 50),
                'relative_strength': relative_strength,
                'tema_cross': self._detect_ma_cross(
                    tema_20, tema_50, tema_20_prev, tema_50_prev
                ),
                'recommendation': analysis.summary['RECOMMENDATION'],
                'atr': analysis.indicators.get('ATR', None),
                'volatility_index': analysis.indicators.get('VHF', None)
            }
            
        except Exception as e:
            logging.error(f"Error getting indicators for {symbol}: {e}")
            return None

    @staticmethod
    def _detect_ma_cross(ma1: float, ma2: float, 
                        ma1_prev: float, ma2_prev: float) -> str:
        """
        Detect moving average crossovers.
        
        Returns:
            str: 'GOLDEN', 'DEATH', or 'NONE'
        """
        if None in [ma1, ma2, ma1_prev, ma2_prev]:
            return 'NONE'
            
        if ma1 > ma2 and ma1_prev <= ma2_prev:
            return 'GOLDEN'
        elif ma1 < ma2 and ma1_prev >= ma2_prev:
            return 'DEATH'
        return 'NONE'

    def analyze_sector(self, symbol: str) -> Optional[Dict]:
        """
        Perform comprehensive analysis on a single sector.
        
        Args:
            symbol (str): The sector ETF symbol to analyze
            
        Returns:
            Dict or None: Analysis results or None if error
        """
        try:
            indicators = self.get_technical_indicators(symbol)
            if not indicators:
                return None
                
            # Apply filters
            if indicators['relative_strength'] < self.config['min_relative_strength']:
                logging.info(
                    f"{symbol}: Low relative strength "
                    f"({indicators['relative_strength']:.2f})"
                )
                return None
                
            if indicators['volume'] < self.config['min_volume']:
                logging.info(f"{symbol}: Low volume ({indicators['volume']})")
                return None
                
            return {
                'symbol': symbol,
                'sector': self.sectors[symbol],
                'price': indicators['price'],
                'volume': indicators['volume'],
                'rsi': indicators['rsi'],
                'relative_strength': indicators['relative_strength'],
                'ma_cross': indicators['tema_cross'],
                'recommendation': indicators['recommendation'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logging.error(f"Error analyzing sector {symbol}: {e}")
            return None

    def scan_sectors(self) -> List[Dict]:
        """
        Scan all sectors and return opportunities that meet criteria.
        
        Returns:
            List[Dict]: List of sector opportunities with analysis results
        """
        logging.info("Starting sector scan...")
        opportunities = []
        
        for symbol in self.sectors.keys():
            result = self.analyze_sector(symbol)
            if result:
                opportunities.append(result)
                logging.info(f"Found opportunity in {symbol}")
                
        # Sort by relative strength
        opportunities.sort(
            key=lambda x: x.get('relative_strength', 0), 
            reverse=True
        )
        
        return opportunities

    def get_rotation_recommendations(self) -> Dict:
        """
        Get comprehensive rotation recommendations with rankings.
        
        Returns:
            Dict: Rotation recommendations and analysis
        """
        opportunities = self.scan_sectors()
        
        # Calculate sector scores
        for opp in opportunities:
            score = (
                opp.get('relative_strength', 0) * 0.4 +
                (opp.get('rsi', 50) / 100) * 0.3 +
                (1 if opp.get('ma_cross') == 'GOLDEN' else 0) * 0.3
            )
            opp['rotation_score'] = score
            
        # Sort by rotation score
        opportunities.sort(
            key=lambda x: x.get('rotation_score', 0), 
            reverse=True
        )
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'opportunities': opportunities,
            'top_sectors': [
                opp['symbol'] for opp in opportunities[:3]
            ],
            'avoid_sectors': [
                opp['symbol'] for opp in opportunities[-3:]
            ]
        }