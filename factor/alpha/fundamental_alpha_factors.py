"""
Fundamental Alpha Factors
Financial statement and market-based fundamental factors for alpha generation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class FundamentalAlphaFactors:
    """
    Generate fundamental analysis based alpha factors.
    
    This class implements various fundamental indicators derived from
    financial statements and market data for alpha generation.
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
        self.fundamental_data = {}
        self.market_data = {}
        
    def fetch_fundamental_data(self) -> None:
        """Fetch fundamental data for all tickers."""
        print("Fetching fundamental data...")
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Get financial statements
                self.fundamental_data[ticker] = {
                    'info': stock.info,
                    'financials': stock.financials,
                    'balance_sheet': stock.balance_sheet,
                    'cashflow': stock.cashflow,
                    'quarterly_financials': stock.quarterly_financials,
                    'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                    'quarterly_cashflow': stock.quarterly_cashflow
                }
                
                # Get market data
                self.market_data[ticker] = stock.history(period=self.period)
                
                print(f"✓ {ticker} data fetched")
                
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
                self.fundamental_data[ticker] = None
                self.market_data[ticker] = None
    
    def calculate_all_factors(self) -> pd.DataFrame:
        """
        Calculate all fundamental alpha factors.
        
        Returns:
            DataFrame with fundamental factors
        """
        if not self.fundamental_data:
            self.fetch_fundamental_data()
        
        all_factors = []
        
        for ticker in self.tickers:
            if self.fundamental_data.get(ticker) is None:
                continue
                
            factors = self._calculate_ticker_factors(ticker)
            if factors is not None:
                all_factors.append(factors)
        
        if all_factors:
            return pd.concat(all_factors, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _calculate_ticker_factors(self, ticker: str) -> Optional[pd.DataFrame]:
        """Calculate fundamental factors for a single ticker."""
        try:
            data = self.fundamental_data[ticker]
            market_data = self.market_data[ticker]
            
            if data is None or market_data.empty:
                return None
            
            # Get current market data
            current_price = market_data['Close'].iloc[-1]
            market_cap = data['info'].get('marketCap', 0)
            
            # Initialize factor dictionary
            factors = {
                'date': market_data.index[-1].strftime('%Y-%m-%d'),
                'tic': ticker,
                'current_price': current_price,
                'market_cap': market_cap
            }
            
            # Calculate different categories of factors
            factors.update(self._calculate_valuation_factors(data, current_price))
            factors.update(self._calculate_profitability_factors(data))
            factors.update(self._calculate_leverage_factors(data))
            factors.update(self._calculate_efficiency_factors(data))
            factors.update(self._calculate_growth_factors(data))
            factors.update(self._calculate_quality_factors(data))
            factors.update(self._calculate_market_factors(data, market_data))
            
            return pd.DataFrame([factors])
            
        except Exception as e:
            print(f"Error calculating factors for {ticker}: {e}")
            return None
    
    def _calculate_valuation_factors(self, data: Dict, current_price: float) -> Dict:
        """Calculate valuation-related factors."""
        factors = {}
        info = data['info']
        
        # Basic valuation ratios
        factors['pe_ratio'] = info.get('trailingPE', np.nan)
        factors['forward_pe'] = info.get('forwardPE', np.nan)
        factors['pb_ratio'] = info.get('priceToBook', np.nan)
        factors['ps_ratio'] = info.get('priceToSalesTrailing12Months', np.nan)
        factors['peg_ratio'] = info.get('pegRatio', np.nan)
        
        # Enterprise value ratios
        factors['ev_revenue'] = info.get('enterpriseToRevenue', np.nan)
        factors['ev_ebitda'] = info.get('enterpriseToEbitda', np.nan)
        
        # Book value and tangible book value
        book_value = info.get('bookValue', np.nan)
        factors['price_to_book'] = current_price / book_value if book_value and book_value > 0 else np.nan
        
        # Market cap related
        factors['market_cap_log'] = np.log(info.get('marketCap', 1)) if info.get('marketCap', 0) > 0 else np.nan
        
        return factors
    
    def _calculate_profitability_factors(self, data: Dict) -> Dict:
        """Calculate profitability-related factors."""
        factors = {}
        info = data['info']
        
        # Margin ratios
        factors['profit_margin'] = info.get('profitMargins', np.nan)
        factors['operating_margin'] = info.get('operatingMargins', np.nan)
        factors['gross_margin'] = info.get('grossMargins', np.nan)
        
        # Return ratios
        factors['roe'] = info.get('returnOnEquity', np.nan)
        factors['roa'] = info.get('returnOnAssets', np.nan)
        factors['roic'] = info.get('returnOnCapital', np.nan)
        
        # Earnings ratios
        factors['earnings_growth'] = info.get('earningsGrowth', np.nan)
        factors['earnings_quarterly_growth'] = info.get('earningsQuarterlyGrowth', np.nan)
        
        return factors
    
    def _calculate_leverage_factors(self, data: Dict) -> Dict:
        """Calculate leverage and financial health factors."""
        factors = {}
        info = data['info']
        
        # Debt ratios
        factors['debt_to_equity'] = info.get('debtToEquity', np.nan)
        factors['total_debt'] = info.get('totalDebt', np.nan)
        factors['total_cash'] = info.get('totalCash', np.nan)
        
        # Calculate net debt
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        factors['net_debt'] = total_debt - total_cash
        
        # Interest coverage
        factors['interest_coverage'] = info.get('interestCoverage', np.nan)
        
        # Current ratio and quick ratio
        factors['current_ratio'] = info.get('currentRatio', np.nan)
        factors['quick_ratio'] = info.get('quickRatio', np.nan)
        
        return factors
    
    def _calculate_efficiency_factors(self, data: Dict) -> Dict:
        """Calculate operational efficiency factors."""
        factors = {}
        info = data['info']
        
        # Turnover ratios
        factors['asset_turnover'] = info.get('assetTurnover', np.nan)
        factors['inventory_turnover'] = info.get('inventoryTurnover', np.nan)
        factors['receivables_turnover'] = info.get('receivablesTurnover', np.nan)
        
        # Working capital efficiency
        factors['working_capital'] = info.get('workingCapital', np.nan)
        
        return factors
    
    def _calculate_growth_factors(self, data: Dict) -> Dict:
        """Calculate growth-related factors."""
        factors = {}
        info = data['info']
        
        # Revenue growth
        factors['revenue_growth'] = info.get('revenueGrowth', np.nan)
        factors['revenue_quarterly_growth'] = info.get('revenueQuarterlyGrowth', np.nan)
        
        # Earnings growth
        factors['earnings_growth'] = info.get('earningsGrowth', np.nan)
        
        # Book value growth (if available)
        try:
            if data['balance_sheet'] is not None and len(data['balance_sheet'].columns) >= 2:
                bs = data['balance_sheet']
                if 'Total Stockholder Equity' in bs.index:
                    current_equity = bs.loc['Total Stockholder Equity'].iloc[0]
                    prev_equity = bs.loc['Total Stockholder Equity'].iloc[1]
                    factors['book_value_growth'] = (current_equity / prev_equity - 1) if prev_equity != 0 else np.nan
        except:
            factors['book_value_growth'] = np.nan
        
        return factors
    
    def _calculate_quality_factors(self, data: Dict) -> Dict:
        """Calculate quality and stability factors."""
        factors = {}
        info = data['info']
        
        # Dividend factors
        factors['dividend_yield'] = info.get('dividendYield', np.nan)
        factors['dividend_rate'] = info.get('dividendRate', np.nan)
        factors['payout_ratio'] = info.get('payoutRatio', np.nan)
        
        # Cash flow factors
        factors['operating_cash_flow'] = info.get('operatingCashflow', np.nan)
        factors['free_cash_flow'] = info.get('freeCashflow', np.nan)
        
        # Calculate free cash flow yield
        market_cap = info.get('marketCap', 0)
        free_cash_flow = info.get('freeCashflow', 0)
        factors['fcf_yield'] = free_cash_flow / market_cap if market_cap > 0 else np.nan
        
        # Beta and volatility
        factors['beta'] = info.get('beta', np.nan)
        
        return factors
    
    def _calculate_market_factors(self, data: Dict, market_data: pd.DataFrame) -> Dict:
        """Calculate market-based factors."""
        factors = {}
        info = data['info']
        
        # Trading volume factors
        factors['avg_volume'] = info.get('averageVolume', np.nan)
        factors['avg_volume_10day'] = info.get('averageVolume10days', np.nan)
        
        # Price momentum (calculated from market data)
        if len(market_data) >= 252:  # At least 1 year of data
            returns_1y = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-252] - 1)
            factors['momentum_1y'] = returns_1y
        else:
            factors['momentum_1y'] = np.nan
            
        if len(market_data) >= 63:  # At least 3 months of data
            returns_3m = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-63] - 1)
            factors['momentum_3m'] = returns_3m
        else:
            factors['momentum_3m'] = np.nan
        
        # Volatility (calculated from market data)
        if len(market_data) >= 21:  # At least 1 month of data
            returns = market_data['Close'].pct_change().dropna()
            factors['volatility_21d'] = returns.tail(21).std() * np.sqrt(252)
            factors['volatility_252d'] = returns.tail(252).std() * np.sqrt(252) if len(returns) >= 252 else np.nan
        else:
            factors['volatility_21d'] = np.nan
            factors['volatility_252d'] = np.nan
        
        # 52-week high/low
        if len(market_data) >= 252:
            high_52w = market_data['High'].tail(252).max()
            low_52w = market_data['Low'].tail(252).min()
            current_price = market_data['Close'].iloc[-1]
            
            factors['price_to_52w_high'] = current_price / high_52w
            factors['price_to_52w_low'] = current_price / low_52w
        else:
            factors['price_to_52w_high'] = np.nan
            factors['price_to_52w_low'] = np.nan
        
        return factors
    
    def calculate_sector_relative_factors(self, factors_df: pd.DataFrame, sector_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Calculate sector-relative factors.
        
        Args:
            factors_df: DataFrame with fundamental factors
            sector_mapping: Dictionary mapping tickers to sectors
            
        Returns:
            DataFrame with sector-relative factors
        """
        if sector_mapping is None:
            # Default sector mapping (simplified)
            sector_mapping = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
                'JPM': 'Financial', 'BAC': 'Financial',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare'
            }
        
        # Add sector information
        factors_df['sector'] = factors_df['tic'].map(sector_mapping)
        
        # Calculate sector medians for key factors
        key_factors = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa', 'debt_to_equity']
        
        for factor in key_factors:
            if factor in factors_df.columns:
                sector_medians = factors_df.groupby('sector')[factor].median()
                factors_df[f'{factor}_sector_relative'] = factors_df.apply(
                    lambda row: (row[factor] / sector_medians.get(row['sector'], 1) - 1) 
                    if pd.notna(row[factor]) and row['sector'] in sector_medians else np.nan,
                    axis=1
                )
        
        return factors_df
    
    def create_composite_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Create composite factors from basic factors."""
        df = factors_df.copy()
        
        # Quality score (higher is better)
        quality_factors = ['roe', 'roa', 'profit_margin', 'operating_margin']
        quality_scores = []
        
        for _, row in df.iterrows():
            scores = []
            for factor in quality_factors:
                if pd.notna(row.get(factor)):
                    scores.append(row[factor])
            
            if scores:
                df.loc[_, 'quality_score'] = np.mean(scores)
            else:
                df.loc[_, 'quality_score'] = np.nan
        
        # Value score (lower is better, so we invert)
        value_factors = ['pe_ratio', 'pb_ratio', 'ps_ratio']
        
        for _, row in df.iterrows():
            scores = []
            for factor in value_factors:
                value = row.get(factor)
                if pd.notna(value) and value > 0:
                    scores.append(1 / value)  # Invert so lower ratios = higher scores
            
            if scores:
                df.loc[_, 'value_score'] = np.mean(scores)
            else:
                df.loc[_, 'value_score'] = np.nan
        
        # Growth score
        growth_factors = ['revenue_growth', 'earnings_growth']
        
        for _, row in df.iterrows():
            scores = []
            for factor in growth_factors:
                if pd.notna(row.get(factor)):
                    scores.append(row[factor])
            
            if scores:
                df.loc[_, 'growth_score'] = np.mean(scores)
            else:
                df.loc[_, 'growth_score'] = np.nan
        
        # Financial health score
        health_factors = ['current_ratio', 'quick_ratio']
        negative_health_factors = ['debt_to_equity']
        
        for _, row in df.iterrows():
            scores = []
            
            # Positive factors
            for factor in health_factors:
                if pd.notna(row.get(factor)):
                    scores.append(row[factor])
            
            # Negative factors (invert)
            for factor in negative_health_factors:
                value = row.get(factor)
                if pd.notna(value) and value > 0:
                    scores.append(1 / (1 + value))  # Transform to 0-1 range
            
            if scores:
                df.loc[_, 'financial_health_score'] = np.mean(scores)
            else:
                df.loc[_, 'financial_health_score'] = np.nan
        
        return df
    
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """Get factor names grouped by category."""
        return {
            'valuation': ['pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio',
                         'ev_revenue', 'ev_ebitda', 'price_to_book', 'market_cap_log'],
            
            'profitability': ['profit_margin', 'operating_margin', 'gross_margin',
                            'roe', 'roa', 'roic', 'earnings_growth', 'earnings_quarterly_growth'],
            
            'leverage': ['debt_to_equity', 'total_debt', 'total_cash', 'net_debt',
                        'interest_coverage', 'current_ratio', 'quick_ratio'],
            
            'efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover',
                          'working_capital'],
            
            'growth': ['revenue_growth', 'revenue_quarterly_growth', 'earnings_growth',
                      'book_value_growth'],
            
            'quality': ['dividend_yield', 'dividend_rate', 'payout_ratio',
                       'operating_cash_flow', 'free_cash_flow', 'fcf_yield', 'beta'],
            
            'market': ['avg_volume', 'avg_volume_10day', 'momentum_1y', 'momentum_3m',
                      'volatility_21d', 'volatility_252d', 'price_to_52w_high', 'price_to_52w_low'],
            
            'composite': ['quality_score', 'value_score', 'growth_score', 'financial_health_score']
        }


def main():
    """Example usage of FundamentalAlphaFactors."""
    
    # Select diverse tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'PG', 'XOM', 'V']
    
    print("=== Fundamental Alpha Factors Example ===")
    print(f"Analyzing {len(tickers)} stocks: {tickers}")
    
    # Initialize factor calculator
    fund_factors = FundamentalAlphaFactors(tickers, period="2y")
    
    # Calculate all factors
    print("\nCalculating fundamental factors...")
    factors_df = fund_factors.calculate_all_factors()
    
    if factors_df.empty:
        print("No data available")
        return
    
    print(f"Generated fundamental factors for {len(factors_df)} stocks")
    
    # Add sector relative factors
    factors_df = fund_factors.calculate_sector_relative_factors(factors_df)
    
    # Create composite factors
    factors_df = fund_factors.create_composite_factors(factors_df)
    
    # Display some key factors
    print("\n=== Key Fundamental Factors ===")
    key_factors = ['tic', 'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 
                   'quality_score', 'value_score', 'growth_score']
    
    display_df = factors_df[key_factors].round(4)
    print(display_df.to_string(index=False))
    
    # Factor categories
    print(f"\n=== Factor Categories ===")
    categories = fund_factors.get_factor_categories()
    for category, factor_list in categories.items():
        available_factors = [f for f in factor_list if f in factors_df.columns]
        print(f"{category}: {len(available_factors)} factors")
    
    print(f"\nTotal factors generated: {len(factors_df.columns)} columns")
    print(f"Data shape: {factors_df.shape}")
    
    # Save results
    output_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/fundamental_factors_example.csv'
    factors_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()