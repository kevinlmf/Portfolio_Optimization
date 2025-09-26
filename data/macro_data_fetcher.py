"""
Macroeconomic Data Fetcher
Advanced data fetching capabilities for macroeconomic indicators from multiple sources.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import time
import os

warnings.filterwarnings('ignore')


class MacroDataFetcher:
    """
    Advanced macroeconomic data fetcher supporting multiple data sources.

    This class provides unified access to macroeconomic data from:
    - FRED API (Federal Reserve Economic Data)
    - Yahoo Finance
    - Alpha Vantage (optional)
    - World Bank API (optional)
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the macro data fetcher.

        Args:
            fred_api_key: FRED API key for accessing Federal Reserve data
        """
        self.fred_api_key = fred_api_key
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"

        # Cache for storing fetched data
        self.data_cache = {}

        # FRED series mappings
        self.fred_series = {
            # Interest Rates
            'DGS10': '10-Year Treasury Rate',
            'DGS3MO': '3-Month Treasury Rate',
            'DGS1MO': '1-Month Treasury Rate',
            'DGS2': '2-Year Treasury Rate',
            'DGS5': '5-Year Treasury Rate',
            'DFEDTARU': 'Federal Funds Target Rate',
            'EFFR': 'Effective Federal Funds Rate',
            'REAINTRATREARAT10Y': '10-Year Real Interest Rate',

            # Inflation Indicators
            'CPIAUCSL': 'Consumer Price Index',
            'CPILFESL': 'Core CPI (Less Food & Energy)',
            'PPIACO': 'Producer Price Index',
            'T5YIE': '5-Year Breakeven Inflation Rate',
            'T10YIE': '10-Year Breakeven Inflation Rate',
            'FPCPITOTLZGUSA': 'Inflation Rate (YoY)',

            # Economic Activity
            'GDP': 'Gross Domestic Product',
            'GDPC1': 'Real GDP',
            'GDPPOT': 'Potential GDP',
            'UNRATE': 'Unemployment Rate',
            'INDPRO': 'Industrial Production Index',
            'UMCSENT': 'Consumer Sentiment Index',
            'PAYEMS': 'Total Non-Farm Payrolls',
            'EMRATIO': 'Employment-Population Ratio',
            'CIVPART': 'Labor Force Participation Rate',

            # Money Supply & Banking
            'M1SL': 'M1 Money Supply',
            'M2SL': 'M2 Money Supply',
            'BOGMBASE': 'Monetary Base',
            'TOTRESNS': 'Total Reserves',
            'EXCSRESNS': 'Excess Reserves',

            # Credit & Bond Markets
            'BAMLH0A0HYM2': 'High Yield Bond Spread',
            'BAA10Y': 'BAA Corporate Bond Spread',
            'AAA10Y': 'AAA Corporate Bond Spread',
            'T10Y3M': '10Y-3M Treasury Spread',
            'T10Y2Y': '10Y-2Y Treasury Spread',

            # Housing Market
            'HOUST': 'Housing Starts',
            'PERMIT': 'Building Permits',
            'CSUSHPINSA': 'Case-Shiller Home Price Index',
            'MORTGAGE30US': '30-Year Mortgage Rate',

            # International & Commodities (some available on FRED)
            'DEXUSEU': 'US/Euro Exchange Rate',
            'DEXJPUS': 'Japan/US Exchange Rate',
            'DEXCHUS': 'China/US Exchange Rate',
            'DCOILWTICO': 'WTI Crude Oil Price',
            'GOLDAMGBD228NLBM': 'Gold Price',

            # Market Indicators
            'VIXCLS': 'VIX Volatility Index',
            'WILL5000INDFC': 'Wilshire 5000 Total Market Index',
            'NASDAQCOM': 'NASDAQ Composite Index',
        }

        # Yahoo Finance mappings for market data
        self.yf_tickers = {
            'VIX': '^VIX',
            'SPY': 'SPY',
            'DXY': 'DX-Y.NYB',  # Dollar Index
            'TNX': '^TNX',      # 10-Year Treasury Yield
            'FVX': '^FVX',      # 5-Year Treasury Yield
            'IRX': '^IRX',      # 3-Month Treasury Yield
            'TYX': '^TYX',      # 30-Year Treasury Yield
            'OIL': 'CL=F',      # Crude Oil
            'GOLD': 'GC=F',     # Gold Futures
            'SILVER': 'SI=F',   # Silver Futures
            'COPPER': 'HG=F',   # Copper Futures
            'EURUSD': 'EURUSD=X',  # EUR/USD
            'GBPUSD': 'GBPUSD=X',  # GBP/USD
            'USDJPY': 'USDJPY=X',  # USD/JPY
            'USDCNY': 'USDCNY=X',  # USD/CNY
        }

    def set_fred_api_key(self, api_key: str) -> None:
        """Set FRED API key."""
        self.fred_api_key = api_key
        print("FRED API key set successfully")

    def fetch_fred_data(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.Series:
        """
        Fetch data from FRED API.

        Args:
            series_id: FRED series identifier
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Pandas Series with the data
        """
        if not self.fred_api_key:
            print(f"Warning: FRED API key not set. Cannot fetch {series_id}")
            return pd.Series()

        try:
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json'
            }

            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date

            response = requests.get(self.fred_base_url, params=params)

            if response.status_code == 200:
                data = response.json()

                if 'observations' in data:
                    observations = data['observations']

                    # Convert to pandas DataFrame
                    df = pd.DataFrame(observations)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[df['value'] != '.']  # Remove missing values
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')

                    # Set date as index and return series
                    df.set_index('date', inplace=True)
                    series = df['value']
                    series.name = series_id

                    print(f"✓ Fetched {len(series)} observations for {series_id}")
                    return series
                else:
                    print(f"✗ No observations found for {series_id}")
                    return pd.Series()

            else:
                print(f"✗ Error fetching {series_id}: HTTP {response.status_code}")
                return pd.Series()

        except Exception as e:
            print(f"✗ Error fetching {series_id} from FRED: {e}")
            return pd.Series()

    def fetch_yf_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.

        Args:
            symbol: Yahoo Finance symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Pandas DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if not data.empty:
                print(f"✓ Fetched {len(data)} observations for {symbol}")
                return data
            else:
                print(f"✗ No data found for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"✗ Error fetching {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()

    def fetch_comprehensive_macro_data(self, period: str = "2y", use_cache: bool = True) -> Dict[str, pd.Series]:
        """
        Fetch comprehensive macroeconomic dataset.

        Args:
            period: Time period for data collection
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary of macroeconomic time series
        """
        print("Fetching comprehensive macroeconomic dataset...")

        if use_cache and self.data_cache:
            print("Using cached data...")
            return self.data_cache

        macro_data = {}

        # Calculate date range
        end_date = datetime.now()
        if period.endswith('y'):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=years * 365)
        elif period.endswith('mo'):
            months = int(period[:-2])
            start_date = end_date - timedelta(days=months * 30)
        else:
            start_date = end_date - timedelta(days=730)  # Default 2 years

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Fetch FRED data if API key is available
        if self.fred_api_key:
            print("\nFetching data from FRED API...")
            for series_id, description in self.fred_series.items():
                try:
                    series = self.fetch_fred_data(series_id, start_date_str, end_date_str)
                    if not series.empty:
                        macro_data[series_id] = series

                    # Rate limiting for FRED API
                    time.sleep(0.1)

                except Exception as e:
                    print(f"Error fetching {series_id}: {e}")
                    continue
        else:
            print("FRED API key not available. Using alternative sources...")

        # Fetch Yahoo Finance data
        print("\nFetching data from Yahoo Finance...")
        for key, symbol in self.yf_tickers.items():
            try:
                data = self.fetch_yf_data(symbol, period)
                if not data.empty:
                    macro_data[f'YF_{key}_CLOSE'] = data['Close']
                    if 'Volume' in data.columns:
                        macro_data[f'YF_{key}_VOLUME'] = data['Volume']

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"Error fetching {key}: {e}")
                continue

        # Calculate derived indicators
        macro_data.update(self._calculate_derived_indicators(macro_data))

        # Cache the results
        if use_cache:
            self.data_cache = macro_data

        print(f"\n✓ Successfully fetched {len(macro_data)} macroeconomic series")
        return macro_data

    def _calculate_derived_indicators(self, macro_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate derived macroeconomic indicators."""
        derived = {}

        try:
            # Yield curve slopes
            if 'DGS10' in macro_data and 'DGS3MO' in macro_data:
                common_dates = macro_data['DGS10'].index.intersection(macro_data['DGS3MO'].index)
                if len(common_dates) > 0:
                    slope = macro_data['DGS10'].loc[common_dates] - macro_data['DGS3MO'].loc[common_dates]
                    derived['YIELD_CURVE_10Y3M'] = slope

            if 'DGS10' in macro_data and 'DGS2' in macro_data:
                common_dates = macro_data['DGS10'].index.intersection(macro_data['DGS2'].index)
                if len(common_dates) > 0:
                    slope = macro_data['DGS10'].loc[common_dates] - macro_data['DGS2'].loc[common_dates]
                    derived['YIELD_CURVE_10Y2Y'] = slope

            # Real interest rates (approximation)
            if 'DGS10' in macro_data and 'T5YIE' in macro_data:
                common_dates = macro_data['DGS10'].index.intersection(macro_data['T5YIE'].index)
                if len(common_dates) > 0:
                    real_rate = macro_data['DGS10'].loc[common_dates] - macro_data['T5YIE'].loc[common_dates]
                    derived['REAL_RATE_10Y'] = real_rate

            # Economic momentum indicators
            for series_name in ['GDP', 'INDPRO', 'PAYEMS']:
                if series_name in macro_data and len(macro_data[series_name]) > 12:
                    series = macro_data[series_name]

                    # Year-over-year growth
                    yoy_growth = series.pct_change(periods=12) * 100
                    derived[f'{series_name}_YOY'] = yoy_growth

                    # 3-month moving average
                    ma3 = series.rolling(window=3).mean()
                    derived[f'{series_name}_MA3'] = ma3

            # Volatility indicators
            for key in ['YF_VIX_CLOSE', 'YF_SPY_CLOSE']:
                if key in macro_data and len(macro_data[key]) > 21:
                    series = macro_data[key]

                    # 21-day moving average
                    ma21 = series.rolling(window=21).mean()
                    derived[f'{key}_MA21'] = ma21

                    # Volatility (for price series)
                    if 'CLOSE' in key:
                        returns = series.pct_change()
                        vol21 = returns.rolling(window=21).std() * np.sqrt(252) * 100
                        derived[f'{key}_VOL21'] = vol21

            print(f"✓ Calculated {len(derived)} derived indicators")

        except Exception as e:
            print(f"Error calculating derived indicators: {e}")

        return derived

    def get_available_series(self) -> Dict[str, Dict[str, str]]:
        """Get information about available data series."""
        return {
            'FRED Series': self.fred_series,
            'Yahoo Finance Tickers': self.yf_tickers
        }

    def create_macro_summary(self, macro_data: Dict[str, pd.Series], latest_only: bool = True) -> pd.DataFrame:
        """
        Create a summary of macroeconomic indicators.

        Args:
            macro_data: Dictionary of macro time series
            latest_only: If True, return only latest values; if False, return full time series

        Returns:
            DataFrame with macro summary
        """
        try:
            if latest_only:
                # Create summary of latest values
                summary_data = []

                for series_name, series_data in macro_data.items():
                    if not series_data.empty:
                        latest_value = series_data.iloc[-1] if len(series_data) > 0 else np.nan
                        latest_date = series_data.index[-1] if len(series_data) > 0 else None

                        # Calculate some basic statistics
                        mean_val = series_data.mean() if len(series_data) > 1 else np.nan
                        std_val = series_data.std() if len(series_data) > 1 else np.nan

                        summary_data.append({
                            'series': series_name,
                            'latest_value': latest_value,
                            'latest_date': latest_date,
                            'mean': mean_val,
                            'std': std_val,
                            'observations': len(series_data)
                        })

                return pd.DataFrame(summary_data)

            else:
                # Return full time series data aligned
                aligned_data = pd.DataFrame()

                for series_name, series_data in macro_data.items():
                    aligned_data[series_name] = series_data

                return aligned_data

        except Exception as e:
            print(f"Error creating macro summary: {e}")
            return pd.DataFrame()

    def export_data(self, macro_data: Dict[str, pd.Series], output_path: str, format: str = 'csv') -> bool:
        """
        Export macroeconomic data to file.

        Args:
            macro_data: Dictionary of macro time series
            output_path: Output file path
            format: Export format ('csv', 'excel', 'json')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create aligned DataFrame
            df = pd.DataFrame()
            for series_name, series_data in macro_data.items():
                df[series_name] = series_data

            # Export based on format
            if format.lower() == 'csv':
                df.to_csv(output_path)
            elif format.lower() == 'excel':
                df.to_excel(output_path)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='index', date_format='iso')
            else:
                print(f"Unsupported format: {format}")
                return False

            print(f"✓ Data exported to {output_path}")
            return True

        except Exception as e:
            print(f"Error exporting data: {e}")
            return False


def main():
    """Example usage of MacroDataFetcher."""

    print("=== Macroeconomic Data Fetcher Example ===")

    # Initialize fetcher
    fetcher = MacroDataFetcher()

    # Note: For full functionality, set FRED API key
    # fetcher.set_fred_api_key("your_fred_api_key_here")

    # Fetch comprehensive data
    macro_data = fetcher.fetch_comprehensive_macro_data(period="1y")

    if macro_data:
        print(f"\n=== Fetched {len(macro_data)} series ===")

        # Create summary
        summary = fetcher.create_macro_summary(macro_data, latest_only=True)
        print("\n=== Latest Values Summary ===")
        print(summary.head(10).to_string(index=False))

        # Export data
        output_path = '/tmp/claude/macro_data_sample.csv'
        fetcher.export_data(macro_data, output_path)

        # Show available series
        print("\n=== Available Data Sources ===")
        available = fetcher.get_available_series()
        for source, series_dict in available.items():
            print(f"{source}: {len(series_dict)} series available")

    else:
        print("No data fetched. Check your API keys and internet connection.")


if __name__ == "__main__":
    main()