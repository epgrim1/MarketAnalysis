#!/usr/bin/env python3
"""
Backtest Sector Rotation Strategy Based on Seasonality (with SPY Comparison)

This script:
  1. Defines sector groups and their assigned periods throughout the year.
  2. Loads historical daily price data for each sector ETF and computes daily returns.
  3. Generates a seasonality signal based on the time of the year and assigns sectors to specific periods.
  4. Backtests the strategy by investing in the assigned sector group for each period.
  5. Loads historical daily price data for the S&P 500 index (SPY) for comparison.
  6. Plots and prints the cumulative return results, including the SPY benchmark.
  
Make sure your raw data files are in the data/raw/ folder.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

RAW_DATA_DIR = os.path.join("data", "raw")
# List of sector ETFs to trade
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLP', 'XLI', 'XLB', 'XLU', 'XLY', 'SMH', 'XRT']

# --- Seasonality Signal ---

def seasonality_signal(date: pd.Timestamp) -> str:
    month = date.month
    if month in [1, 2]:
        return ["XLV", "XLP", "XLU"]  # Group 1
    elif month in [3, 4]:
        return ["XLI", "XLB", "XLY"]  # Group 2
    elif month in [5, 6]:
        return ["XLE", "XLF", "XLRE"]  # Group 3
    elif month in [7, 8]:
        return ["XLK", "XLC", "XLY"]  # Group 4
    elif month in [9, 10]:
        return ["XLV", "XLP", "XLU"]  # Group 5
    else:  # November-December
        return ["XLY", "XRT", "XLI"]  # Group 6

# --- Backtesting the Sector Strategy ---

def backtest_sectors():
    sector_results = {}
    
    for sector in SECTORS:
        exchange = "NASDAQ" if sector == "SMH" else "AMEX"
        sector_file = os.path.join(RAW_DATA_DIR, f"{exchange}_{sector} 1D.csv")
        df = pd.read_csv(sector_file)
        # Convert 'time' to a date-only datetime
        df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'close']].sort_values(by='date')
        # Normalize the index to date-only (time 00:00:00)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index.date)
        df['return'] = df['close'].pct_change()
        
        print(f"{sector} Date Range: {df.index.min()} to {df.index.max()}")
        
        df['seasonality_signal'] = df.index.to_series().apply(lambda date: sector in seasonality_signal(date))
        df['strategy_return'] = df['return'] * df['seasonality_signal']
        df['cum_return_sector'] = (1 + df['return']).cumprod()
        df['cum_return_strategy'] = (1 + df['strategy_return']).cumprod()
        
        sector_results[sector] = df
    
    if not sector_results:
        print("No sector data found. Aborting backtest.")
        return
    
    # Load SPY data for comparison
    spy_file = os.path.join(RAW_DATA_DIR, "AMEX_SPY 1D.csv")
    spy_df = pd.read_csv(spy_file)
    spy_df['date'] = pd.to_datetime(spy_df['time'], unit='s').dt.date
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df = spy_df[['date', 'close']].sort_values(by='date')
    spy_df = spy_df.set_index('date')
    spy_df.index = pd.to_datetime(spy_df.index.date)
    spy_df['return'] = spy_df['close'].pct_change()
    spy_df['cum_return_spy'] = (1 + spy_df['return']).cumprod()
    
    print(f"SPY Date Range: {spy_df.index.min()} to {spy_df.index.max()}")
    print("SPY Performance:")
    print(spy_df[['return', 'cum_return_spy']].tail())
    
    final_spy_return = spy_df['cum_return_spy'].iloc[-1]
    print(f"\nFinal SPY Cumulative Return: {final_spy_return:.2f}x")
    
    # Create the union of all dates from sectors with data.
    all_dates = sorted(set().union(*[set(df.index) for df in sector_results.values()]))
    if not all_dates:
        print("No common dates found across sectors. Aborting backtest.")
        return
    
    portfolio_returns = pd.DataFrame(index=all_dates)
    for sector, df in sector_results.items():
        portfolio_returns[sector] = df['strategy_return'].reindex(all_dates, fill_value=0)
    
    portfolio_returns['portfolio_return'] = portfolio_returns.mean(axis=1)
    portfolio_returns['cum_return_portfolio'] = (1 + portfolio_returns['portfolio_return']).cumprod()
    
    if portfolio_returns.empty or portfolio_returns['cum_return_portfolio'].empty:
        print("Portfolio returns DataFrame is empty. Aborting backtest.")
        return
    
    print("Union of Dates Across Sectors:", portfolio_returns.index.min(), "to", portfolio_returns.index.max())
    print("Backtest Summary for Sector Rotation Strategy:")
    print(portfolio_returns[['portfolio_return', 'cum_return_portfolio']].tail())
    
    final_return = portfolio_returns['cum_return_portfolio'].iloc[-1]
    print("\nFinal Cumulative Portfolio Return: {:.2f}x".format(final_return))
    
    # Merge portfolio returns with SPY returns
    merged_returns = pd.merge(portfolio_returns[['cum_return_portfolio']], spy_df['cum_return_spy'], left_index=True, right_index=True, how='left')
    
    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(merged_returns.index, merged_returns['cum_return_portfolio'], label='Sector Rotation Strategy')
    plt.plot(merged_returns.index, merged_returns['cum_return_spy'], label='SPY')
    plt.title("Cumulative Returns: Sector Rotation Strategy vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for sector, df in sector_results.items():
        plt.plot(df.index, df['cum_return_strategy'], label=sector)
    plt.title("Sector Strategy Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()

def main():
    backtest_sectors()

if __name__ == "__main__":
    main()