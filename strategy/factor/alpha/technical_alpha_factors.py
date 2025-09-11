"""
Technical Alpha Factors
Advanced technical analysis indicators for alpha generation in portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TechnicalAlphaFactors:
    """
    Generate technical analysis based alpha factors.
    
    This class implements various technical indicators that can be used
    as alpha factors for portfolio optimization and stock selection.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with market data.
        
        Args:
            data: DataFrame with columns ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values(['tic', 'date']).reset_index(drop=True)
        
    def calculate_all_factors(self, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        Calculate all technical alpha factors.
        
        Args:
            windows: List of window periods for calculations
            
        Returns:
            DataFrame with all technical factors
        """
        factors = []
        
        for tic in self.data['tic'].unique():
            tic_data = self.data[self.data['tic'] == tic].copy()
            tic_factors = self._calculate_tic_factors(tic_data, windows)
            factors.append(tic_factors)
        
        return pd.concat(factors, ignore_index=True)
    
    def _calculate_tic_factors(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate factors for a single ticker."""
        df = data.copy()
        
        # Basic price factors
        df = self._add_momentum_factors(df, windows)
        df = self._add_volatility_factors(df, windows)
        df = self._add_trend_factors(df, windows)
        df = self._add_oscillator_factors(df, windows)
        df = self._add_volume_factors(df, windows)
        df = self._add_pattern_factors(df, windows)
        df = self._add_statistical_factors(df, windows)
        
        return df
    
    def _add_momentum_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add momentum-based factors."""
        # Price momentum
        for window in windows:
            df[f'momentum_{window}'] = df['close'].pct_change(window)
            df[f'log_momentum_{window}'] = np.log(df['close'] / df['close'].shift(window))
            
        # RSI (Relative Strength Index)
        for window in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def _add_volatility_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add volatility-based factors."""
        # Price volatility
        df['returns'] = df['close'].pct_change()
        for window in windows:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'realized_vol_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
            
        # High-Low volatility
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        for window in windows:
            df[f'hl_volatility_{window}'] = df['hl_ratio'].rolling(window=window).mean()
            
        # True Range and ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        for window in [14, 21]:
            df[f'atr_{window}'] = df['tr'].rolling(window=window).mean()
            
        # Bollinger Bands
        for window in [20, 50]:
            sma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            df[f'bb_squeeze_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / sma
            
        return df
    
    def _add_trend_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add trend-following factors."""
        # Moving averages
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'wma_{window}'] = df['close'].rolling(window=window).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1))
            )
            
        # Price relative to moving averages
        for window in windows:
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
            df[f'price_to_ema_{window}'] = df['close'] / df[f'ema_{window}'] - 1
            
        # Moving average crossovers
        df['ma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ma_cross_10_50'] = (df['sma_10'] > df['sma_50']).astype(int) if 50 in windows else 0
        
        # ADX (Average Directional Index)
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr_14 = df['tr'].rolling(window=14).mean()
        plus_di_14 = pd.Series(plus_dm).rolling(window=14).mean() / tr_14 * 100
        minus_di_14 = pd.Series(minus_dm).rolling(window=14).mean() / tr_14 * 100
        
        dx = np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100
        df['adx'] = dx.rolling(window=14).mean()
        
        return df
    
    def _add_oscillator_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add oscillator-based factors."""
        # Williams %R
        for window in [14, 21]:
            high_n = df['high'].rolling(window=window).max()
            low_n = df['low'].rolling(window=window).min()
            df[f'williams_r_{window}'] = -100 * (high_n - df['close']) / (high_n - low_n)
            
        # CCI (Commodity Channel Index)
        for window in [14, 20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(window=window).mean()
            mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df[f'cci_{window}'] = (tp - sma_tp) / (0.015 * mad)
            
        # ROC (Rate of Change)
        for window in [10, 20]:
            df[f'roc_{window}'] = ((df['close'] - df['close'].shift(window)) / df['close'].shift(window)) * 100
            
        return df
    
    def _add_volume_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add volume-based factors."""
        # Volume moving averages
        for window in windows:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
            
        # Price-Volume relationship
        df['pv_trend'] = (df['close'].pct_change() * df['volume']).rolling(window=5).mean()
        
        # On Balance Volume (OBV)
        obv = []
        obv_value = 0
        for i in range(len(df)):
            if i == 0:
                obv_value = df['volume'].iloc[i]
            else:
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv_value += df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv_value -= df['volume'].iloc[i]
            obv.append(obv_value)
        df['obv'] = obv
        
        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # Accumulation/Distribution Line
        ad_line = []
        ad_value = 0
        for i in range(len(df)):
            if df['high'].iloc[i] != df['low'].iloc[i]:
                clv = ((df['close'].iloc[i] - df['low'].iloc[i]) - (df['high'].iloc[i] - df['close'].iloc[i])) / (df['high'].iloc[i] - df['low'].iloc[i])
                ad_value += clv * df['volume'].iloc[i]
            ad_line.append(ad_value)
        df['ad_line'] = ad_line
        
        return df
    
    def _add_pattern_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add pattern recognition factors."""
        # Gap detection
        df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        
        # Inside/Outside days
        df['inside_day'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        df['outside_day'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
        
        # Doji patterns (simplified)
        body_size = np.abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        df['doji'] = (body_size < 0.1 * total_range).astype(int)
        
        # Hammer/Hanging man patterns (simplified)
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        df['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < 0.5 * body_size)).astype(int)
        
        return df
    
    def _add_statistical_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add statistical factors."""
        # Z-score of price
        for window in windows:
            rolling_mean = df['close'].rolling(window=window).mean()
            rolling_std = df['close'].rolling(window=window).std()
            df[f'price_zscore_{window}'] = (df['close'] - rolling_mean) / rolling_std
            
        # Percentile rank
        for window in windows:
            df[f'percentile_rank_{window}'] = df['close'].rolling(window=window).rank(pct=True)
            
        # Linear regression slope
        for window in [10, 20]:
            def calculate_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                y = series.values
                slope = np.polyfit(x, y, 1)[0]
                return slope / series.iloc[-1]  # Normalize by current price
            
            df[f'price_slope_{window}'] = df['close'].rolling(window=window).apply(calculate_slope)
            
        # R-squared of linear regression
        for window in [10, 20]:
            def calculate_r_squared(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                y = series.values
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            df[f'price_r2_{window}'] = df['close'].rolling(window=window).apply(calculate_r_squared)
            
        return df
    
    def get_factor_names(self) -> Dict[str, List[str]]:
        """Get factor names grouped by category."""
        return {
            'momentum': ['momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 
                        'log_momentum_5', 'log_momentum_10', 'log_momentum_20', 'log_momentum_60',
                        'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
                        'stoch_k', 'stoch_d'],
            
            'volatility': ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
                          'realized_vol_5', 'realized_vol_10', 'realized_vol_20', 'realized_vol_60',
                          'hl_volatility_5', 'hl_volatility_10', 'hl_volatility_20', 'hl_volatility_60',
                          'atr_14', 'atr_21', 'bb_position_20', 'bb_position_50', 
                          'bb_squeeze_20', 'bb_squeeze_50'],
            
            'trend': ['price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_60',
                     'price_to_ema_5', 'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_60',
                     'ma_cross_5_20', 'ma_cross_10_50', 'adx'],
            
            'oscillators': ['williams_r_14', 'williams_r_21', 'cci_14', 'cci_20',
                           'roc_10', 'roc_20'],
            
            'volume': ['volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20', 'volume_ratio_60',
                      'pv_trend', 'obv', 'vpt', 'ad_line'],
            
            'patterns': ['gap_up', 'gap_down', 'inside_day', 'outside_day', 'doji', 'hammer'],
            
            'statistical': ['price_zscore_5', 'price_zscore_10', 'price_zscore_20', 'price_zscore_60',
                           'percentile_rank_5', 'percentile_rank_10', 'percentile_rank_20', 'percentile_rank_60',
                           'price_slope_10', 'price_slope_20', 'price_r2_10', 'price_r2_20']
        }
    
    def calculate_factor_returns(self, factors_df: pd.DataFrame, forward_periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate forward returns for factor analysis.
        
        Args:
            factors_df: DataFrame with factors
            forward_periods: List of forward-looking periods
            
        Returns:
            DataFrame with forward returns
        """
        results = []
        
        for tic in factors_df['tic'].unique():
            tic_data = factors_df[factors_df['tic'] == tic].copy()
            
            for period in forward_periods:
                tic_data[f'forward_return_{period}'] = tic_data['close'].pct_change(period).shift(-period)
                
            results.append(tic_data)
        
        return pd.concat(results, ignore_index=True)
    
    def analyze_factor_performance(self, factors_df: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """
        Analyze factor performance using IC (Information Coefficient).
        
        Args:
            factors_df: DataFrame with factors and forward returns
            factor_names: List of factor names to analyze
            
        Returns:
            DataFrame with factor performance metrics
        """
        results = []
        
        forward_return_cols = [col for col in factors_df.columns if col.startswith('forward_return_')]
        
        for factor in factor_names:
            if factor not in factors_df.columns:
                continue
                
            factor_results = {'factor': factor}
            
            for return_col in forward_return_cols:
                period = return_col.split('_')[-1]
                
                # Calculate IC (correlation between factor and forward returns)
                ic = factors_df[[factor, return_col]].corr().iloc[0, 1]
                factor_results[f'ic_{period}'] = ic
                
                # Calculate IC significance
                valid_data = factors_df[[factor, return_col]].dropna()
                if len(valid_data) > 30:
                    from scipy.stats import pearsonr
                    _, p_value = pearsonr(valid_data[factor], valid_data[return_col])
                    factor_results[f'ic_pvalue_{period}'] = p_value
                else:
                    factor_results[f'ic_pvalue_{period}'] = 1.0
            
            results.append(factor_results)
        
        return pd.DataFrame(results)


def main():
    """Example usage of TechnicalAlphaFactors."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Get price data and convert to required format
    df_list = []
    for ticker in tickers:
        import yfinance as yf
        stock_data = yf.download(ticker, period="1y", progress=False)
        
        for date, row in stock_data.iterrows():
            df_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'tic': ticker,
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            })
    
    df = pd.DataFrame(df_list)
    
    # Initialize factor calculator
    tech_factors = TechnicalAlphaFactors(df)
    
    print("Calculating technical alpha factors...")
    factors_df = tech_factors.calculate_all_factors()
    
    print(f"Generated {len(factors_df)} factor observations")
    print(f"Factor columns: {len([col for col in factors_df.columns if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']])}")
    
    # Calculate forward returns
    factors_with_returns = tech_factors.calculate_factor_returns(factors_df)
    
    # Analyze factor performance
    factor_names = []
    factor_categories = tech_factors.get_factor_names()
    for category, names in factor_categories.items():
        factor_names.extend(names)
    
    performance_df = tech_factors.analyze_factor_performance(factors_with_returns, factor_names[:10])  # Analyze first 10 factors
    
    print("\nTop factors by IC (1-day forward return):")
    performance_df_sorted = performance_df.sort_values('ic_1', key=abs, ascending=False)
    for _, row in performance_df_sorted.head(5).iterrows():
        print(f"{row['factor']}: IC = {row['ic_1']:.4f}, p-value = {row['ic_pvalue_1']:.4f}")
    
    print(f"\nTechnical factors generated successfully!")
    print(f"Data saved with {len(factors_df)} rows and {len(factors_df.columns)} columns")


if __name__ == "__main__":
    main()