#!/usr/bin/env python3
# src/data/data_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

class DataProcessor:
    """
    Handle data processing tasks for market analysis, including loading,
    cleaning, and feature engineering for sector data.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataProcessor with data directory path.
        
        Args:
            data_dir (str): Path to data directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache: Dict[str, pd.DataFrame] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=self.data_dir / 'data_processing.log'
        )
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_sector_data(self, symbol: str, exchange: str = "AMEX") -> Optional[pd.DataFrame]:
        """
        Load raw sector data from CSV file.
        
        Args:
            symbol (str): Sector ETF symbol
            exchange (str): Exchange name (default: "AMEX")
            
        Returns:
            Optional[pd.DataFrame]: Loaded data or None if error
        """
        try:
            # Check cache first
            cache_key = f"{exchange}_{symbol}"
            if cache_key in self.cache:
                return self.cache[cache_key].copy()
            
            # Load from file
            file_path = self.raw_dir / f"{exchange}_{symbol} 1D.csv"
            df = pd.read_csv(file_path)
            
            # Basic validation
            required_columns = ['time', 'open', 'high', 'low', 'close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in {symbol} data")
            
            # Convert timestamps
            df['date'] = pd.to_datetime(df['time'], unit='s')
            
            # Cache the result
            self.cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for analysis.
        
        Args:
            df (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with additional technical indicators
        """
        try:
            df = df.copy()
            
            # Basic price metrics
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']/df['close'].shift(1))
            
            # Volatility
            df['rolling_std'] = df['returns'].rolling(window=20).std()
            
            # Moving averages
            for window in [20, 50, 200]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # Momentum indicators
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            raise

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def _calculate_macd(prices: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        fast_ema = prices.ewm(span=fast).mean()
        slow_ema = prices.ewm(span=slow).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal).mean()
        
        return macd, signal_line

    def prepare_analysis_data(self, symbol: str, 
                            lookback_days: int = 252) -> Optional[pd.DataFrame]:
        """
        Prepare data for analysis with feature engineering.
        
        Args:
            symbol (str): Sector ETF symbol
            lookback_days (int): Number of historical days to include
            
        Returns:
            Optional[pd.DataFrame]: Processed data or None if error
        """
        try:
            # Load raw data
            exchange = "NASDAQ" if symbol == "SMH" else "AMEX"
            df = self.load_sector_data(symbol, exchange)
            if df is None:
                return None
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Filter for lookback period
            start_date = datetime.now() - timedelta(days=lookback_days)
            df = df[df['date'] >= start_date]
            
            # Handle missing values
            df = df.dropna()
            
            # Save processed data
            output_path = self.processed_dir / f"{symbol}_processed.csv"
            df.to_csv(output_path, index=False)
            
            return df
            
        except Exception as e:
            logging.error(f"Error preparing analysis data for {symbol}: {e}")
            return None

    def get_market_regime_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate market regime indicators.
        
        Args:
            df (pd.DataFrame): Processed price data
            
        Returns:
            Dict[str, float]: Market regime indicators
        """
        try:
            features = {}
            
            # Trend strength
            features['trend_strength'] = (
                (df['close'] > df['sma_200']).mean() * 100
            )
            
            # Volatility regime
            current_vol = df['rolling_std'].iloc[-1]
            avg_vol = df['rolling_std'].mean()
            features['volatility_regime'] = (current_vol / avg_vol - 1) * 100
            
            # Momentum
            features['momentum_score'] = df['rsi'].iloc[-1]
            
            # Volume trend
            features['volume_trend'] = (
                (df['volume_ratio'] > 1).tail(20).mean() * 100
            )
            
            return features
            
        except Exception as e:
            logging.error(f"Error calculating market regime features: {e}")
            return {}

    def get_sector_correlations(self, symbols: List[str], 
                              window: int = 63) -> Optional[pd.DataFrame]:
        """
        Calculate rolling correlations between sectors.
        
        Args:
            symbols (List[str]): List of sector symbols
            window (int): Rolling window in days
            
        Returns:
            Optional[pd.DataFrame]: Correlation matrix or None if error
        """
        try:
            # Get returns for all sectors
            returns_dict = {}
            for symbol in symbols:
                df = self.prepare_analysis_data(symbol)
                if df is not None:
                    returns_dict[symbol] = df['returns']
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate rolling correlations
            correlations = returns_df.rolling(window=window).corr()
            
            # Get latest correlation matrix
            latest_corr = correlations.iloc[-window:]
            latest_corr = latest_corr.reset_index()
            pivot_corr = latest_corr.pivot(
                index='level_0', 
                columns='level_1', 
                values='returns'
            )
            
            return pivot_corr
            
        except Exception as e:
            logging.error(f"Error calculating sector correlations: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cache.clear()
        logging.info("Data cache cleared")

    def update_data(self, symbol: str, new_data: pd.DataFrame) -> None:
        """
        Update data for a symbol in both cache and files.
        
        Args:
            symbol (str): Sector ETF symbol
            new_data (pd.DataFrame): New data to update with
        """
        try:
            exchange = "NASDAQ" if symbol == "SMH" else "AMEX"
            cache_key = f"{exchange}_{symbol}"
            
            # Update cache
            self.cache[cache_key] = new_data.copy()
            
            # Save to file
            file_path = self.raw_dir / f"{exchange}_{symbol} 1D.csv"
            new_data.to_csv(file_path, index=False)
            
            logging.info(f"Data updated for {symbol}")
            
        except Exception as e:
            logging.error(f"Error updating data for {symbol}: {e}")
            raise