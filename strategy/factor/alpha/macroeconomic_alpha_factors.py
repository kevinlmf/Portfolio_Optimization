"""
Macroeconomic Alpha Factors
Macroeconomic indicators and economic environment factors for alpha generation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
import warnings
import requests
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')


class MacroeconomicAlphaFactors:
    """
    Generate macroeconomic-based alpha factors.

    This class implements various macroeconomic indicators and economic environment
    factors that can influence stock returns and portfolio performance.
    """

    def __init__(self, tickers: List[str], period: str = "2y"):
        """
        Initialize with ticker list.

        Args:
            tickers: List of stock symbols
            period: Time period for data collection
        """
        self.tickers = tickers
        self.period = period
        self.macro_data = {}
        self.stock_data = {}

        # FRED API endpoint (requires API key)
        self.fred_api_key = None  # Users should set their own FRED API key
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"

        # Macroeconomic indicators mapping
        self.macro_indicators = {
            # Interest Rates
            'DGS10': '10-Year Treasury Rate',
            'DGS3MO': '3-Month Treasury Rate',
            'DFEDTARU': 'Federal Funds Target Rate',
            'REAINTRATREARAT10Y': '10-Year Real Interest Rate',

            # Inflation
            'CPIAUCSL': 'Consumer Price Index',
            'CPILFESL': 'Core CPI (Less Food & Energy)',
            'PPIACO': 'Producer Price Index',
            'T5YIE': '5-Year Breakeven Inflation Rate',

            # Economic Activity
            'GDP': 'Gross Domestic Product',
            'UNRATE': 'Unemployment Rate',
            'INDPRO': 'Industrial Production Index',
            'UMCSENT': 'Consumer Sentiment Index',
            'PAYEMS': 'Non-Farm Payrolls',

            # Money Supply & Credit
            'M2SL': 'M2 Money Supply',
            'TOTRESNS': 'Total Reserves of Depository Institutions',
            'BAMLH0A0HYM2': 'High Yield Bond Spread',
            'BAA10Y': 'BAA Corporate Bond Yield Spread',

            # Global & Commodity
            'DEXUSEU': 'US/Euro Exchange Rate',
            'DEXJPUS': 'Japan/US Exchange Rate',
            'DCOILWTICO': 'WTI Oil Price',
            'GOLDAMGBD228NLBM': 'Gold Price',

            # Market Sentiment
            'VIXCLS': 'VIX Volatility Index',
            'AAII': 'AAII Investor Sentiment'
        }

    def set_fred_api_key(self, api_key: str) -> None:
        """Set FRED API key for data access."""
        self.fred_api_key = api_key

    def fetch_macro_data_alternative(self) -> None:
        """
        Fetch macroeconomic data using alternative free sources.
        This method uses yfinance and other free APIs as alternatives to FRED.
        """
        print("Fetching macroeconomic data from alternative sources...")

        try:
            # Get Treasury rates from yfinance
            tnx = yf.Ticker("^TNX")  # 10-Year Treasury
            irx = yf.Ticker("^IRX")  # 3-Month Treasury

            self.macro_data['10Y_TREASURY'] = tnx.history(period=self.period)['Close']
            self.macro_data['3M_TREASURY'] = irx.history(period=self.period)['Close']

            # Calculate yield curve slope
            if len(self.macro_data['10Y_TREASURY']) > 0 and len(self.macro_data['3M_TREASURY']) > 0:
                # Align dates
                common_dates = self.macro_data['10Y_TREASURY'].index.intersection(
                    self.macro_data['3M_TREASURY'].index
                )
                if len(common_dates) > 0:
                    slope_data = (self.macro_data['10Y_TREASURY'].loc[common_dates] -
                                self.macro_data['3M_TREASURY'].loc[common_dates])
                    self.macro_data['YIELD_CURVE_SLOPE'] = slope_data

            # Get VIX
            vix = yf.Ticker("^VIX")
            self.macro_data['VIX'] = vix.history(period=self.period)['Close']

            # Get Dollar Index
            dxy = yf.Ticker("DX-Y.NYB")
            self.macro_data['DOLLAR_INDEX'] = dxy.history(period=self.period)['Close']

            # Get Oil Price (WTI)
            oil = yf.Ticker("CL=F")
            self.macro_data['OIL_PRICE'] = oil.history(period=self.period)['Close']

            # Get Gold Price
            gold = yf.Ticker("GC=F")
            self.macro_data['GOLD_PRICE'] = gold.history(period=self.period)['Close']

            # Get SPY for market benchmark
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period=self.period)
            self.macro_data['SPY_CLOSE'] = spy_data['Close']
            self.macro_data['SPY_VOLUME'] = spy_data['Volume']

            print("✓ Alternative macroeconomic data fetched successfully")

        except Exception as e:
            print(f"✗ Error fetching alternative macro data: {e}")

    def fetch_stock_data(self) -> None:
        """Fetch stock price data for all tickers."""
        print("Fetching stock data...")

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                self.stock_data[ticker] = stock.history(period=self.period)
                print(f"✓ {ticker} data fetched")
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
                self.stock_data[ticker] = None

    def calculate_all_factors(self) -> pd.DataFrame:
        """
        Calculate all macroeconomic alpha factors.

        Returns:
            DataFrame with macroeconomic factors for each ticker
        """
        if not self.macro_data:
            self.fetch_macro_data_alternative()

        if not self.stock_data:
            self.fetch_stock_data()

        all_factors = []

        for ticker in self.tickers:
            if self.stock_data.get(ticker) is None:
                continue

            factors = self._calculate_ticker_factors(ticker)
            if factors is not None:
                all_factors.append(factors)

        if all_factors:
            return pd.concat(all_factors, ignore_index=True)
        else:
            return pd.DataFrame()

    def _calculate_ticker_factors(self, ticker: str) -> Optional[pd.DataFrame]:
        """Calculate macroeconomic factors for a single ticker."""
        try:
            stock_data = self.stock_data[ticker]

            if stock_data.empty:
                return None

            # Get latest date
            latest_date = stock_data.index[-1]

            # Initialize factor dictionary
            factors = {
                'date': latest_date.strftime('%Y-%m-%d'),
                'tic': ticker,
            }

            # Calculate different categories of macro factors
            factors.update(self._calculate_interest_rate_factors(stock_data, latest_date))
            factors.update(self._calculate_market_environment_factors(stock_data, latest_date))
            factors.update(self._calculate_commodity_factors(stock_data, latest_date))
            factors.update(self._calculate_currency_factors(stock_data, latest_date))
            factors.update(self._calculate_volatility_factors(stock_data, latest_date))
            factors.update(self._calculate_correlation_factors(stock_data, latest_date))
            factors.update(self._calculate_regime_factors(stock_data, latest_date))

            return pd.DataFrame([factors])

        except Exception as e:
            print(f"Error calculating macro factors for {ticker}: {e}")
            return None

    def _calculate_interest_rate_factors(self, stock_data: pd.DataFrame, latest_date) -> Dict:
        """Calculate interest rate related factors."""
        factors = {}

        try:
            # Current levels
            if '10Y_TREASURY' in self.macro_data and not self.macro_data['10Y_TREASURY'].empty:
                latest_10y = self._get_latest_macro_value('10Y_TREASURY', latest_date)
                factors['treasury_10y_level'] = latest_10y

                # Calculate changes
                factors.update(self._calculate_macro_changes('10Y_TREASURY', latest_date, 'treasury_10y'))

            if '3M_TREASURY' in self.macro_data and not self.macro_data['3M_TREASURY'].empty:
                latest_3m = self._get_latest_macro_value('3M_TREASURY', latest_date)
                factors['treasury_3m_level'] = latest_3m

                factors.update(self._calculate_macro_changes('3M_TREASURY', latest_date, 'treasury_3m'))

            # Yield curve slope
            if 'YIELD_CURVE_SLOPE' in self.macro_data and not self.macro_data['YIELD_CURVE_SLOPE'].empty:
                slope = self._get_latest_macro_value('YIELD_CURVE_SLOPE', latest_date)
                factors['yield_curve_slope'] = slope

                factors.update(self._calculate_macro_changes('YIELD_CURVE_SLOPE', latest_date, 'yield_slope'))

            # Interest rate beta (correlation with rate changes)
            if '10Y_TREASURY' in self.macro_data:
                factors['interest_rate_beta'] = self._calculate_interest_rate_beta(stock_data, latest_date)

        except Exception as e:
            print(f"Error calculating interest rate factors: {e}")

        return factors

    def _calculate_market_environment_factors(self, stock_data: pd.DataFrame, latest_date) -> Dict:
        """Calculate market environment factors."""
        factors = {}

        try:
            # VIX factors
            if 'VIX' in self.macro_data and not self.macro_data['VIX'].empty:
                vix_level = self._get_latest_macro_value('VIX', latest_date)
                factors['vix_level'] = vix_level

                # VIX percentile (relative to recent history)
                vix_data = self._get_aligned_macro_data('VIX', latest_date, window=252)
                if len(vix_data) > 10:
                    factors['vix_percentile'] = (vix_data <= vix_level).mean()

                factors.update(self._calculate_macro_changes('VIX', latest_date, 'vix'))

                # Stock-VIX correlation
                factors['vix_correlation'] = self._calculate_vix_correlation(stock_data, latest_date)

            # Market beta relative to SPY
            if 'SPY_CLOSE' in self.macro_data:
                factors['market_beta'] = self._calculate_market_beta(stock_data, latest_date)

        except Exception as e:
            print(f"Error calculating market environment factors: {e}")

        return factors

    def _calculate_commodity_factors(self, stock_data: pd.DataFrame, latest_date) -> Dict:
        """Calculate commodity exposure factors."""
        factors = {}

        try:
            # Oil factors
            if 'OIL_PRICE' in self.macro_data and not self.macro_data['OIL_PRICE'].empty:
                oil_level = self._get_latest_macro_value('OIL_PRICE', latest_date)
                factors['oil_price_level'] = oil_level

                factors.update(self._calculate_macro_changes('OIL_PRICE', latest_date, 'oil'))
                factors['oil_correlation'] = self._calculate_commodity_correlation(stock_data, 'OIL_PRICE', latest_date)

            # Gold factors
            if 'GOLD_PRICE' in self.macro_data and not self.macro_data['GOLD_PRICE'].empty:
                gold_level = self._get_latest_macro_value('GOLD_PRICE', latest_date)
                factors['gold_price_level'] = gold_level

                factors.update(self._calculate_macro_changes('GOLD_PRICE', latest_date, 'gold'))
                factors['gold_correlation'] = self._calculate_commodity_correlation(stock_data, 'GOLD_PRICE', latest_date)

        except Exception as e:
            print(f"Error calculating commodity factors: {e}")

        return factors

    def _calculate_currency_factors(self, stock_data: pd.DataFrame, latest_date) -> Dict:
        """Calculate currency exposure factors."""
        factors = {}

        try:
            if 'DOLLAR_INDEX' in self.macro_data and not self.macro_data['DOLLAR_INDEX'].empty:
                dxy_level = self._get_latest_macro_value('DOLLAR_INDEX', latest_date)
                factors['dollar_index_level'] = dxy_level

                factors.update(self._calculate_macro_changes('DOLLAR_INDEX', latest_date, 'dollar'))
                factors['dollar_correlation'] = self._calculate_commodity_correlation(stock_data, 'DOLLAR_INDEX', latest_date)

        except Exception as e:
            print(f"Error calculating currency factors: {e}")

        return factors

    def _calculate_volatility_factors(self, stock_data: pd.DataFrame, latest_date) -> Dict:
        """Calculate volatility regime factors."""
        factors = {}

        try:
            # Stock volatility vs market volatility
            stock_vol = self._calculate_realized_volatility(stock_data, 21)
            market_vol = self._calculate_realized_volatility_macro('SPY_CLOSE', 21, latest_date)

            if stock_vol is not None and market_vol is not None:
                factors['relative_volatility'] = stock_vol / market_vol if market_vol > 0 else np.nan

            factors['realized_volatility_21d'] = stock_vol
            factors['realized_volatility_63d'] = self._calculate_realized_volatility(stock_data, 63)

            # Volatility regime indicator
            if stock_vol is not None:
                historical_vols = []
                for window in [21, 63, 126, 252]:
                    vol = self._calculate_realized_volatility(stock_data, window)
                    if vol is not None:
                        historical_vols.append(vol)

                if historical_vols:
                    factors['vol_regime_indicator'] = (stock_vol - np.mean(historical_vols)) / np.std(historical_vols)

        except Exception as e:
            print(f"Error calculating volatility factors: {e}")

        return factors

    def _calculate_correlation_factors(self, stock_data: pd.DataFrame, latest_date) -> Dict:
        """Calculate correlation with macro factors."""
        factors = {}

        try:
            # Calculate rolling correlations with key macro factors
            macro_factors = ['10Y_TREASURY', 'VIX', 'OIL_PRICE', 'GOLD_PRICE', 'DOLLAR_INDEX']

            for macro_factor in macro_factors:
                if macro_factor in self.macro_data:
                    corr = self._calculate_rolling_correlation(stock_data, macro_factor, latest_date, 63)
                    factors[f'{macro_factor.lower()}_corr_63d'] = corr

        except Exception as e:
            print(f"Error calculating correlation factors: {e}")

        return factors

    def _calculate_regime_factors(self, stock_data: pd.DataFrame, latest_date) -> Dict:
        """Calculate economic regime factors."""
        factors = {}

        try:
            # Risk-on vs Risk-off regime
            factors['risk_regime'] = self._calculate_risk_regime(latest_date)

            # Interest rate environment
            factors['rate_environment'] = self._calculate_rate_environment(latest_date)

            # Inflation regime
            factors['inflation_regime'] = self._calculate_inflation_regime(latest_date)

        except Exception as e:
            print(f"Error calculating regime factors: {e}")

        return factors

    def _get_latest_macro_value(self, macro_key: str, reference_date, lookback_days: int = 10) -> Optional[float]:
        """Get the latest available macro value."""
        if macro_key not in self.macro_data or self.macro_data[macro_key].empty:
            return np.nan

        try:
            macro_series = self.macro_data[macro_key]

            # Find the closest date within lookback period
            end_date = reference_date
            start_date = reference_date - timedelta(days=lookback_days)

            # Filter data within the date range
            mask = (macro_series.index >= start_date) & (macro_series.index <= end_date)
            filtered_series = macro_series[mask]

            if len(filtered_series) > 0:
                return filtered_series.iloc[-1]
            else:
                # If no data in recent period, get the most recent available
                if len(macro_series) > 0:
                    return macro_series.iloc[-1]

        except Exception as e:
            print(f"Error getting latest macro value for {macro_key}: {e}")

        return np.nan

    def _calculate_macro_changes(self, macro_key: str, reference_date, prefix: str) -> Dict:
        """Calculate various time period changes for a macro factor."""
        changes = {}

        try:
            if macro_key not in self.macro_data or self.macro_data[macro_key].empty:
                return {f'{prefix}_1d_change': np.nan, f'{prefix}_1w_change': np.nan,
                       f'{prefix}_1m_change': np.nan, f'{prefix}_3m_change': np.nan}

            macro_series = self.macro_data[macro_key]
            current_value = self._get_latest_macro_value(macro_key, reference_date)

            if pd.isna(current_value):
                return {f'{prefix}_1d_change': np.nan, f'{prefix}_1w_change': np.nan,
                       f'{prefix}_1m_change': np.nan, f'{prefix}_3m_change': np.nan}

            # Calculate changes over different periods
            periods = {'1d': 1, '1w': 7, '1m': 30, '3m': 90}

            for period_name, days in periods.items():
                past_date = reference_date - timedelta(days=days)
                past_value = self._get_latest_macro_value(macro_key, past_date)

                if pd.notna(past_value) and past_value != 0:
                    change = (current_value - past_value) / past_value
                    changes[f'{prefix}_{period_name}_change'] = change
                else:
                    changes[f'{prefix}_{period_name}_change'] = np.nan

        except Exception as e:
            print(f"Error calculating macro changes for {macro_key}: {e}")
            changes = {f'{prefix}_1d_change': np.nan, f'{prefix}_1w_change': np.nan,
                      f'{prefix}_1m_change': np.nan, f'{prefix}_3m_change': np.nan}

        return changes

    def _calculate_interest_rate_beta(self, stock_data: pd.DataFrame, reference_date, window: int = 63) -> float:
        """Calculate interest rate beta (sensitivity to rate changes)."""
        try:
            if '10Y_TREASURY' in self.macro_data and not self.macro_data['10Y_TREASURY'].empty:
                # Get aligned data
                stock_returns = stock_data['Close'].pct_change().dropna()
                treasury_data = self.macro_data['10Y_TREASURY']

                # Align dates
                common_dates = stock_returns.index.intersection(treasury_data.index)
                if len(common_dates) < window:
                    return np.nan

                # Get recent data
                recent_dates = common_dates[-window:] if len(common_dates) >= window else common_dates

                stock_ret = stock_returns.loc[recent_dates]
                treasury_changes = treasury_data.loc[recent_dates].diff()

                if len(stock_ret) > 10 and len(treasury_changes) > 10:
                    correlation = np.corrcoef(stock_ret[1:], treasury_changes[1:])[0, 1]
                    return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            print(f"Error calculating interest rate beta: {e}")

        return np.nan

    def _calculate_vix_correlation(self, stock_data: pd.DataFrame, reference_date, window: int = 63) -> float:
        """Calculate correlation with VIX."""
        try:
            if 'VIX' in self.macro_data and not self.macro_data['VIX'].empty:
                stock_returns = stock_data['Close'].pct_change().dropna()
                vix_changes = self.macro_data['VIX'].pct_change().dropna()

                # Align dates
                common_dates = stock_returns.index.intersection(vix_changes.index)
                if len(common_dates) < window:
                    return np.nan

                recent_dates = common_dates[-window:] if len(common_dates) >= window else common_dates

                stock_ret = stock_returns.loc[recent_dates]
                vix_ret = vix_changes.loc[recent_dates]

                if len(stock_ret) > 10 and len(vix_ret) > 10:
                    correlation = np.corrcoef(stock_ret, vix_ret)[0, 1]
                    return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            print(f"Error calculating VIX correlation: {e}")

        return np.nan

    def _calculate_market_beta(self, stock_data: pd.DataFrame, reference_date, window: int = 252) -> float:
        """Calculate market beta relative to SPY."""
        try:
            if 'SPY_CLOSE' in self.macro_data and not self.macro_data['SPY_CLOSE'].empty:
                stock_returns = stock_data['Close'].pct_change().dropna()
                market_returns = self.macro_data['SPY_CLOSE'].pct_change().dropna()

                # Align dates
                common_dates = stock_returns.index.intersection(market_returns.index)
                if len(common_dates) < 30:  # Minimum data requirement
                    return np.nan

                recent_dates = common_dates[-window:] if len(common_dates) >= window else common_dates

                stock_ret = stock_returns.loc[recent_dates]
                market_ret = market_returns.loc[recent_dates]

                if len(stock_ret) > 20 and len(market_ret) > 20:
                    # Calculate beta using covariance/variance
                    covariance = np.cov(stock_ret, market_ret)[0, 1]
                    market_variance = np.var(market_ret)

                    if market_variance > 0:
                        beta = covariance / market_variance
                        return beta

        except Exception as e:
            print(f"Error calculating market beta: {e}")

        return np.nan

    def _calculate_commodity_correlation(self, stock_data: pd.DataFrame, commodity_key: str, reference_date, window: int = 63) -> float:
        """Calculate correlation with commodity prices."""
        try:
            if commodity_key in self.macro_data and not self.macro_data[commodity_key].empty:
                stock_returns = stock_data['Close'].pct_change().dropna()
                commodity_returns = self.macro_data[commodity_key].pct_change().dropna()

                # Align dates
                common_dates = stock_returns.index.intersection(commodity_returns.index)
                if len(common_dates) < window:
                    return np.nan

                recent_dates = common_dates[-window:] if len(common_dates) >= window else common_dates

                stock_ret = stock_returns.loc[recent_dates]
                commodity_ret = commodity_returns.loc[recent_dates]

                if len(stock_ret) > 10 and len(commodity_ret) > 10:
                    correlation = np.corrcoef(stock_ret, commodity_ret)[0, 1]
                    return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            print(f"Error calculating commodity correlation: {e}")

        return np.nan

    def _calculate_realized_volatility(self, stock_data: pd.DataFrame, window: int) -> Optional[float]:
        """Calculate realized volatility."""
        try:
            returns = stock_data['Close'].pct_change().dropna()
            if len(returns) >= window:
                recent_returns = returns.tail(window)
                volatility = recent_returns.std() * np.sqrt(252)  # Annualized
                return volatility
        except Exception as e:
            print(f"Error calculating realized volatility: {e}")

        return np.nan

    def _calculate_realized_volatility_macro(self, macro_key: str, window: int, reference_date) -> Optional[float]:
        """Calculate realized volatility for macro factor."""
        try:
            if macro_key in self.macro_data and not self.macro_data[macro_key].empty:
                macro_returns = self.macro_data[macro_key].pct_change().dropna()
                if len(macro_returns) >= window:
                    recent_returns = macro_returns.tail(window)
                    volatility = recent_returns.std() * np.sqrt(252)
                    return volatility
        except Exception as e:
            print(f"Error calculating macro volatility: {e}")

        return np.nan

    def _calculate_rolling_correlation(self, stock_data: pd.DataFrame, macro_key: str, reference_date, window: int) -> float:
        """Calculate rolling correlation with macro factor."""
        try:
            if macro_key in self.macro_data and not self.macro_data[macro_key].empty:
                stock_returns = stock_data['Close'].pct_change().dropna()
                macro_data = self.macro_data[macro_key]

                if 'PRICE' in macro_key.upper() or 'LEVEL' in macro_key.upper():
                    macro_series = macro_data.pct_change().dropna()
                else:
                    macro_series = macro_data.diff().dropna()

                # Align dates
                common_dates = stock_returns.index.intersection(macro_series.index)
                if len(common_dates) < window:
                    return np.nan

                recent_dates = common_dates[-window:] if len(common_dates) >= window else common_dates

                stock_ret = stock_returns.loc[recent_dates]
                macro_changes = macro_series.loc[recent_dates]

                if len(stock_ret) > 10 and len(macro_changes) > 10:
                    correlation = np.corrcoef(stock_ret, macro_changes)[0, 1]
                    return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            print(f"Error calculating rolling correlation: {e}")

        return np.nan

    def _calculate_risk_regime(self, reference_date) -> float:
        """Calculate risk-on vs risk-off regime indicator."""
        try:
            # Simple regime indicator based on VIX level
            if 'VIX' in self.macro_data and not self.macro_data['VIX'].empty:
                vix_level = self._get_latest_macro_value('VIX', reference_date)

                if pd.notna(vix_level):
                    # Risk-off when VIX > 25, risk-on when VIX < 15, neutral otherwise
                    if vix_level > 25:
                        return -1.0  # Risk-off
                    elif vix_level < 15:
                        return 1.0   # Risk-on
                    else:
                        return 0.0   # Neutral

        except Exception as e:
            print(f"Error calculating risk regime: {e}")

        return 0.0

    def _calculate_rate_environment(self, reference_date) -> float:
        """Calculate interest rate environment indicator."""
        try:
            if '10Y_TREASURY' in self.macro_data and not self.macro_data['10Y_TREASURY'].empty:
                current_rate = self._get_latest_macro_value('10Y_TREASURY', reference_date)

                if pd.notna(current_rate):
                    # Rising rate environment (> 3%), falling rate environment (< 2%), neutral otherwise
                    if current_rate > 3.0:
                        return 1.0   # Rising rates
                    elif current_rate < 2.0:
                        return -1.0  # Falling rates
                    else:
                        return 0.0   # Neutral

        except Exception as e:
            print(f"Error calculating rate environment: {e}")

        return 0.0

    def _calculate_inflation_regime(self, reference_date) -> float:
        """Calculate inflation regime indicator (simplified)."""
        try:
            # This is a simplified version since we don't have real inflation data
            # In practice, you would use actual CPI data

            # Use oil price as a proxy for inflationary pressures
            if 'OIL_PRICE' in self.macro_data and not self.macro_data['OIL_PRICE'].empty:
                current_oil = self._get_latest_macro_value('OIL_PRICE', reference_date)

                # Get oil price 1 year ago for comparison
                past_oil = self._get_latest_macro_value('OIL_PRICE', reference_date - timedelta(days=365))

                if pd.notna(current_oil) and pd.notna(past_oil) and past_oil > 0:
                    oil_change = (current_oil - past_oil) / past_oil

                    if oil_change > 0.3:  # 30% increase suggests inflationary pressure
                        return 1.0   # Inflationary
                    elif oil_change < -0.3:  # 30% decrease suggests deflationary pressure
                        return -1.0  # Deflationary
                    else:
                        return 0.0   # Stable

        except Exception as e:
            print(f"Error calculating inflation regime: {e}")

        return 0.0

    def _get_aligned_macro_data(self, macro_key: str, reference_date, window: int) -> pd.Series:
        """Get aligned macro data for a specific window."""
        if macro_key not in self.macro_data or self.macro_data[macro_key].empty:
            return pd.Series()

        try:
            macro_series = self.macro_data[macro_key]
            end_date = reference_date
            start_date = reference_date - timedelta(days=window + 10)  # Extra buffer

            mask = (macro_series.index >= start_date) & (macro_series.index <= end_date)
            return macro_series[mask].tail(window)

        except Exception as e:
            print(f"Error getting aligned macro data: {e}")
            return pd.Series()

    def create_composite_macro_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Create composite macroeconomic factors."""
        df = factors_df.copy()

        try:
            # Macro sensitivity score
            sensitivity_factors = ['interest_rate_beta', 'vix_correlation', 'oil_correlation', 'dollar_correlation']
            for _, row in df.iterrows():
                scores = []
                for factor in sensitivity_factors:
                    if pd.notna(row.get(factor)):
                        scores.append(abs(row[factor]))  # Use absolute value

                if scores:
                    df.loc[_, 'macro_sensitivity_score'] = np.mean(scores)
                else:
                    df.loc[_, 'macro_sensitivity_score'] = np.nan

            # Rate sensitivity indicator
            rate_factors = ['interest_rate_beta', 'treasury_10y_1m_change']
            for _, row in df.iterrows():
                rate_beta = row.get('interest_rate_beta', 0)
                rate_change = row.get('treasury_10y_1m_change', 0)

                if pd.notna(rate_beta) and pd.notna(rate_change):
                    # Negative for rate-sensitive stocks in rising rate environment
                    df.loc[_, 'rate_sensitivity_indicator'] = rate_beta * rate_change
                else:
                    df.loc[_, 'rate_sensitivity_indicator'] = np.nan

            # Risk regime score
            regime_factors = ['risk_regime', 'rate_environment', 'inflation_regime']
            for _, row in df.iterrows():
                regime_scores = []
                for factor in regime_factors:
                    if pd.notna(row.get(factor)):
                        regime_scores.append(row[factor])

                if regime_scores:
                    df.loc[_, 'macro_regime_score'] = np.mean(regime_scores)
                else:
                    df.loc[_, 'macro_regime_score'] = np.nan

        except Exception as e:
            print(f"Error creating composite macro factors: {e}")

        return df

    def get_factor_categories(self) -> Dict[str, List[str]]:
        """Get factor names grouped by category."""
        return {
            'interest_rates': ['treasury_10y_level', 'treasury_3m_level', 'yield_curve_slope',
                             'treasury_10y_1d_change', 'treasury_10y_1w_change', 'treasury_10y_1m_change',
                             'interest_rate_beta'],

            'market_environment': ['vix_level', 'vix_percentile', 'vix_1d_change', 'vix_1w_change',
                                 'vix_correlation', 'market_beta'],

            'commodities': ['oil_price_level', 'oil_1d_change', 'oil_1w_change', 'oil_correlation',
                          'gold_price_level', 'gold_1d_change', 'gold_1w_change', 'gold_correlation'],

            'currencies': ['dollar_index_level', 'dollar_1d_change', 'dollar_1w_change', 'dollar_correlation'],

            'volatility': ['relative_volatility', 'realized_volatility_21d', 'realized_volatility_63d',
                         'vol_regime_indicator'],

            'correlations': ['10y_treasury_corr_63d', 'vix_corr_63d', 'oil_price_corr_63d',
                           'gold_price_corr_63d', 'dollar_index_corr_63d'],

            'regimes': ['risk_regime', 'rate_environment', 'inflation_regime'],

            'composite': ['macro_sensitivity_score', 'rate_sensitivity_indicator', 'macro_regime_score']
        }


def main():
    """Example usage of MacroeconomicAlphaFactors."""

    # Select diverse tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'GLD', 'TLT']

    print("=== Macroeconomic Alpha Factors Example ===")
    print(f"Analyzing {len(tickers)} stocks: {tickers}")

    # Initialize factor calculator
    macro_factors = MacroeconomicAlphaFactors(tickers, period="2y")

    # Calculate all factors
    print("\nCalculating macroeconomic factors...")
    factors_df = macro_factors.calculate_all_factors()

    if factors_df.empty:
        print("No data available")
        return

    print(f"Generated macro factors for {len(factors_df)} stocks")

    # Create composite factors
    factors_df = macro_factors.create_composite_macro_factors(factors_df)

    # Display some key factors
    print("\n=== Key Macroeconomic Factors ===")
    key_factors = ['tic', 'treasury_10y_level', 'vix_level', 'market_beta',
                   'interest_rate_beta', 'macro_sensitivity_score']

    available_factors = [f for f in key_factors if f in factors_df.columns]
    if available_factors:
        display_df = factors_df[available_factors].round(4)
        print(display_df.to_string(index=False))

    # Factor categories
    print(f"\n=== Factor Categories ===")
    categories = macro_factors.get_factor_categories()
    for category, factor_list in categories.items():
        available_factors = [f for f in factor_list if f in factors_df.columns]
        print(f"{category}: {len(available_factors)} factors")

    print(f"\nTotal factors generated: {len(factors_df.columns)} columns")
    print(f"Data shape: {factors_df.shape}")

    # Save results
    output_path = '/tmp/claude/macroeconomic_factors_example.csv'
    factors_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()