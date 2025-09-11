"""
Enhanced Real Data Fetcher for Alpha and Beta Factor Mining
Enhanced version with better data quality, more comprehensive datasets,
and proper preprocessing for factor research.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


class EnhancedDataFetcher:
    """Enhanced data fetcher for comprehensive factor research."""

    def __init__(self, start_date: str = None, end_date: str = None):
        self.start_date = start_date or (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        # Cache
        self.price_cache = {}
        self.fundamental_cache = {}

    # ================= ALPHA ==================
    def get_comprehensive_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV + fundamental + derived metrics."""
        print(f"Fetching comprehensive data for {len(tickers)} tickers...")

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_single_ticker_data, ticker): ticker for ticker in tickers
            }
            ticker_data = {}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_data[ticker] = future.result()
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")

        price_data, volume_data, fundamental_data = {}, {}, {}

        for ticker, data in ticker_data.items():
            if data is not None and not data.empty:
                if "Close" in data.columns:
                    price_data[ticker] = data["Close"]
                elif "Adj Close" in data.columns:
                    price_data[ticker] = data["Adj Close"]

                if "Volume" in data.columns:
                    volume_data[ticker] = data["Volume"]

                if "Market Cap" in data.columns:
                    fundamental_data[ticker] = data[
                        ["Market Cap", "PE Ratio", "Dividend Yield"]
                    ]

        result = {
            "prices": pd.DataFrame(price_data),
            "volumes": pd.DataFrame(volume_data),
            "returns": pd.DataFrame(price_data).pct_change(),
            "fundamentals": fundamental_data,
        }

        # Clean data
        for key in ["prices", "volumes", "returns"]:
            if key in result:
                result[key] = result[key].dropna(how="all").ffill().bfill()

        print(f"Successfully fetched data: {result['prices'].shape} price points")
        return result

    def _fetch_single_ticker_data(self, ticker: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            if hist_data.empty:
                return None

            # Fundamental data
            try:
                info = stock.info
                hist_data["Market Cap"] = info.get("marketCap", np.nan)
                hist_data["PE Ratio"] = info.get("trailingPE", np.nan)
                hist_data["Dividend Yield"] = info.get("dividendYield", np.nan)
            except Exception as e:
                print(f"Warning: Could not fetch fundamentals for {ticker}: {e}")

            # Derived metrics
            hist_data["Returns"] = hist_data["Close"].pct_change()
            hist_data["Log_Returns"] = np.log(hist_data["Close"] / hist_data["Close"].shift(1))
            hist_data["High_Low_Ratio"] = hist_data["High"] / hist_data["Low"] - 1
            hist_data["Close_Open_Ratio"] = hist_data["Close"] / hist_data["Open"] - 1
            return hist_data
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    def get_market_indices_data(self) -> pd.DataFrame:
        """Get market indices for beta calc."""
        indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ", 
            "DIA": "Dow Jones",
            "^VIX": "Volatility Index",  # Fixed VIX symbol
            "TLT": "Treasury",
            "GLD": "Gold",
            "UUP": "Dollar Index",       # Fixed DXY -> UUP (more reliable)
        }

        index_data = {}
        for symbol, name in indices.items():
            try:
                data = yf.download(
                    symbol, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False
                )
                if not data.empty and "Close" in data.columns:
                    close_data = data["Close"]
                    # Handle MultiIndex columns from yfinance
                    if isinstance(close_data, pd.DataFrame):
                        close_data = close_data.iloc[:, 0]  # Take first (and only) column
                    index_data[name] = close_data
                else:
                    print(f"⚠️ No data for {name} ({symbol})")
            except Exception as e:
                print(f"Could not fetch {name} ({symbol}): {e}")
        
        
        if not index_data:
            print("⚠️ No market indices fetched, creating minimal dataset")
            # Create empty DataFrame with proper structure
            return pd.DataFrame(columns=['Market'])
        
        df = pd.DataFrame(index_data)
        return df.ffill() if not df.empty else df

    def get_sector_classification(self, tickers: List[str]) -> Dict[str, str]:
        sector_map = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self._get_ticker_sector, ticker): ticker for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    sector = future.result()
                    if sector:
                        sector_map[ticker] = sector
                except Exception as e:
                    print(f"Could not get sector for {ticker}: {e}")
        return sector_map

    def _get_ticker_sector(self, ticker: str) -> Optional[str]:
        try:
            return yf.Ticker(ticker).info.get("sector", None)
        except:
            return None

    def create_alpha_research_dataset(
        self, tickers: List[str], include_fundamentals: bool = True, include_market_data: bool = True
    ) -> pd.DataFrame:
        print("Creating alpha research dataset...")
        stock_data = self.get_comprehensive_stock_data(tickers)
        market_data = self.get_market_indices_data() if include_market_data else None
        sectors = self.get_sector_classification(tickers)

        dataset_list = []
        for ticker in tickers:
            if ticker not in stock_data["prices"].columns:
                continue

            df = pd.DataFrame(index=stock_data["prices"].index)
            df["close"] = stock_data["prices"][ticker]
            df["volume"] = (
                stock_data["volumes"][ticker] if ticker in stock_data["volumes"].columns else np.nan
            )
            df["returns"] = (
                stock_data["returns"][ticker] if ticker in stock_data["returns"].columns else np.nan
            )
            df["ticker"] = ticker
            df["sector"] = sectors.get(ticker, "Unknown")
            df["date"] = df.index

            if market_data is not None:
                for col in market_data.columns:
                    df[f"market_{col.lower().replace(' ', '_')}"] = market_data[col]

            if include_fundamentals and ticker in stock_data["fundamentals"]:
                for col in stock_data["fundamentals"][ticker].columns:
                    df[f"fundamental_{col.lower().replace(' ', '_')}"] = stock_data[
                        "fundamentals"
                    ][ticker][col]

            dataset_list.append(df)

        full_dataset = pd.concat(dataset_list, ignore_index=True)
        full_dataset = full_dataset.dropna(subset=["close", "ticker"])
        print(f"Alpha research dataset created: {full_dataset.shape}")
        print(f"Date range: {full_dataset['date'].min()} to {full_dataset['date'].max()}")
        print(f"Tickers: {full_dataset['ticker'].nunique()}")
        return full_dataset

    # ================= BETA ==================
    def create_beta_research_dataset(
        self, tickers: List[str], market_factors: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        print("Creating beta research dataset...")
        stock_data = self.get_comprehensive_stock_data(tickers)
        returns = stock_data["returns"]

        market_data = self.get_market_indices_data()
        if market_factors is None:
            market_factors = ["S&P 500", "NASDAQ", "Treasury", "Volatility Index"]

        available_factors = [f for f in market_factors if f in market_data.columns]

        if available_factors:
            factor_returns = market_data[available_factors].pct_change()
        else:
            print("⚠️ Warning: No market factors found, using equal-weighted stock returns as Market")
            market_series = returns.mean(axis=1)
            factor_returns = pd.DataFrame({"Market": market_series}, index=market_series.index)

        # ✅ Ensure at least 1 factor
        if factor_returns.empty or factor_returns.shape[1] == 0:
            market_series = returns.mean(axis=1)
            factor_returns = pd.DataFrame({"Market": market_series}, index=market_series.index)

        # ✅ Align index
        common_idx = returns.index.intersection(factor_returns.index)
        returns = returns.loc[common_idx]
        factor_returns = factor_returns.loc[common_idx]

        return {
            "stock_returns": returns,
            "factor_returns": factor_returns,
            "correlation_matrix": returns.corr(),
            "factor_correlation": factor_returns.corr(),
        }





