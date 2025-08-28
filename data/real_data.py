"""
Real market data fetching and processing using yfinance.
This module provides functionality to download historical stock data
and process it for portfolio optimization.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RealDataFetcher:
    """Fetches and processes real market data using yfinance."""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize the data fetcher.
        
        Args:
            start_date: Start date for data fetching (YYYY-MM-DD)
            end_date: End date for data fetching (YYYY-MM-DD)
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
    def get_stock_data(self, tickers: List[str], period: str = "5y") -> pd.DataFrame:
        """
        Download stock price data for given tickers.
        
        Args:
            tickers: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with adjusted close prices
        """
        try:
            data = yf.download(tickers, start=self.start_date, end=self.end_date, 
                             period=period, progress=False)
            
            if len(tickers) == 1:
                return data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
            else:
                return data['Adj Close']
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_returns(self, tickers: List[str], return_type: str = "log") -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            tickers: List of stock symbols
            return_type: Type of returns ('simple' or 'log')
            
        Returns:
            DataFrame with returns
        """
        prices = self.get_stock_data(tickers)
        
        if return_type == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
            
        return returns.dropna()
    
    def get_sp500_tickers(self, n_stocks: int = 50) -> List[str]:
        """
        Get list of S&P 500 stock tickers.
        
        Args:
            n_stocks: Number of stocks to return (top by market cap)
            
        Returns:
            List of ticker symbols
        """
        # Popular large-cap stocks as default
        default_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
            'ABBV', 'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS',
            'ABT', 'MRK', 'CSCO', 'ACN', 'DHR', 'VZ', 'ADBE', 'NFLX', 'CRM',
            'NKE', 'TXN', 'ORCL', 'INTC', 'QCOM', 'AMD', 'HON', 'UNP', 'RTX',
            'NEE', 'LOW', 'PM', 'T', 'SPGI'
        ]
        
        return default_tickers[:min(n_stocks, len(default_tickers))]
    
    def get_sector_etfs(self) -> List[str]:
        """
        Get list of sector ETF tickers.
        
        Returns:
            List of sector ETF symbols
        """
        return [
            'XLK',  # Technology
            'XLF',  # Financial
            'XLV',  # Health Care
            'XLI',  # Industrial
            'XLY',  # Consumer Discretionary
            'XLP',  # Consumer Staples
            'XLE',  # Energy
            'XLU',  # Utilities
            'XLB',  # Materials
            'XLRE', # Real Estate
            'XLC'   # Communication Services
        ]
    
    def get_international_etfs(self) -> List[str]:
        """
        Get list of international ETF tickers.
        
        Returns:
            List of international ETF symbols
        """
        return [
            'VEA',   # Developed Markets
            'VWO',   # Emerging Markets
            'EFA',   # MSCI EAFE
            'IEMG',  # Core MSCI Emerging Markets
            'SCHF',  # International Equity
            'VXUS',  # Total International Stock
            'EEM',   # Emerging Markets
            'FXI',   # China Large-Cap
            'EWJ',   # Japan
            'EWG'    # Germany
        ]
    
    def get_bond_etfs(self) -> List[str]:
        """
        Get list of bond ETF tickers.
        
        Returns:
            List of bond ETF symbols
        """
        return [
            'BND',   # Total Bond Market
            'AGG',   # Core Bond
            'TLT',   # 20+ Year Treasury
            'IEF',   # 7-10 Year Treasury
            'SHY',   # 1-3 Year Treasury
            'LQD',   # Investment Grade Corporate
            'HYG',   # High Yield Corporate
            'TIP',   # Inflation Protected Securities
            'EMB',   # Emerging Markets Bonds
            'MBB'    # Mortgage-Backed Securities
        ]
    
    def create_diversified_portfolio(self, include_international: bool = True, 
                                   include_bonds: bool = True) -> pd.DataFrame:
        """
        Create a diversified portfolio with stocks, international, and bonds.
        
        Args:
            include_international: Whether to include international ETFs
            include_bonds: Whether to include bond ETFs
            
        Returns:
            DataFrame with returns for diversified portfolio
        """
        tickers = []
        
        # Add US stocks
        tickers.extend(self.get_sp500_tickers(20))
        
        # Add sector ETFs
        tickers.extend(self.get_sector_etfs()[:6])
        
        # Add international exposure
        if include_international:
            tickers.extend(self.get_international_etfs()[:4])
            
        # Add bond exposure
        if include_bonds:
            tickers.extend(self.get_bond_etfs()[:4])
        
        return self.get_returns(tickers)
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the output file
        """
        filepath = f"/Users/mengfanlong/Downloads/Portfolio_Optimization_system/data/{filename}"
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def get_market_data_summary(self, tickers: List[str]) -> Dict:
        """
        Get summary statistics for market data.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary with summary statistics
        """
        returns = self.get_returns(tickers)
        
        summary = {
            'mean_returns': returns.mean(),
            'std_returns': returns.std(),
            'correlation_matrix': returns.corr(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'data_shape': returns.shape,
            'date_range': {
                'start': returns.index.min(),
                'end': returns.index.max()
            }
        }
        
        return summary
    
    def _calculate_max_drawdown(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate maximum drawdown for each asset.
        
        Args:
            returns: DataFrame with returns
            
        Returns:
            Series with max drawdown for each asset
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()


def main():
    """Example usage of the RealDataFetcher."""
    
    # Initialize data fetcher
    fetcher = RealDataFetcher()
    
    # Example 1: Get data for major tech stocks
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    tech_returns = fetcher.get_returns(tech_stocks)
    fetcher.save_data(tech_returns, 'tech_stock_returns.csv')
    
    # Example 2: Create diversified portfolio
    diversified_returns = fetcher.create_diversified_portfolio()
    fetcher.save_data(diversified_returns, 'diversified_portfolio_returns.csv')
    
    # Example 3: Get summary statistics
    summary = fetcher.get_market_data_summary(tech_stocks)
    print("\nTech Stocks Summary:")
    print(f"Data shape: {summary['data_shape']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Average Sharpe ratios:\n{summary['sharpe_ratio'].round(2)}")
    
    # Example 4: Sector ETFs analysis
    sector_etfs = fetcher.get_sector_etfs()
    sector_returns = fetcher.get_returns(sector_etfs)
    fetcher.save_data(sector_returns, 'sector_etf_returns.csv')
    
    print(f"\nGenerated {len(tech_stocks)} tech stock returns")
    print(f"Generated {diversified_returns.shape[1]} diversified portfolio assets")
    print(f"Generated {len(sector_etfs)} sector ETF returns")


if __name__ == "__main__":
    main()