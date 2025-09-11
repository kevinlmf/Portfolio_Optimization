"""
Price-Volume Alpha Factors
Advanced price and volume relationship factors for alpha generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PriceVolumeAlphaFactors:
    """
    Generate price-volume relationship based alpha factors.
    
    This class implements sophisticated price-volume analysis techniques
    that capture market microstructure effects and institutional behavior.
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
        Calculate all price-volume alpha factors.
        
        Args:
            windows: List of window periods for calculations
            
        Returns:
            DataFrame with all price-volume factors
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
        
        # Basic preprocessing
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_log'] = np.log(df['volume'] + 1)  # Add 1 to handle zeros
        
        # Add different categories of factors
        df = self._add_volume_momentum_factors(df, windows)
        df = self._add_price_volume_correlation_factors(df, windows)
        df = self._add_volume_profile_factors(df, windows)
        df = self._add_institutional_flow_factors(df, windows)
        df = self._add_liquidity_factors(df, windows)
        df = self._add_volume_pattern_factors(df, windows)
        df = self._add_price_volume_divergence_factors(df, windows)
        df = self._add_microstructure_factors(df, windows)
        
        return df
    
    def _add_volume_momentum_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add volume momentum factors."""
        # Volume moving averages and ratios
        for window in windows:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
            df[f'volume_momentum_{window}'] = df['volume'].pct_change(window)
            
        # Volume acceleration
        for window in [5, 10]:
            df[f'volume_acceleration_{window}'] = df[f'volume_momentum_{window}'].diff()
            
        # Volume trend strength
        for window in [10, 20]:
            def volume_trend_strength(series):
                if len(series) < 3:
                    return 0
                x = np.arange(len(series))
                slope, _, r_value, _, _ = stats.linregress(x, series)
                return slope * r_value
            
            df[f'volume_trend_{window}'] = df['volume'].rolling(window=window).apply(volume_trend_strength)
            
        # Relative volume position
        for window in [20, 60]:
            df[f'volume_percentile_{window}'] = df['volume'].rolling(window=window).rank(pct=True)
            
        return df
    
    def _add_price_volume_correlation_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add price-volume correlation factors."""
        # Price-volume correlation
        for window in windows:
            df[f'pv_correlation_{window}'] = df['returns'].rolling(window=window).corr(df['volume_log'])
            
        # Volume-weighted price measures
        df['vwap_intraday'] = (df['high'] + df['low'] + df['close']) / 3  # Simplified VWAP
        
        for window in [5, 10, 20]:
            # Volume-weighted moving average
            df[f'vwma_{window}'] = (df['close'] * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
            df[f'price_to_vwma_{window}'] = df['close'] / df[f'vwma_{window}'] - 1
            
        # Price-volume elasticity
        for window in [10, 20]:
            def price_volume_elasticity(df_slice):
                if len(df_slice) < 3:
                    return 0
                price_changes = df_slice['returns'].dropna()
                volume_changes = df_slice['volume'].pct_change().dropna()
                
                if len(price_changes) != len(volume_changes) or len(price_changes) < 2:
                    return 0
                
                # Calculate elasticity as correlation * (std(volume_change) / std(price_change))
                corr = price_changes.corr(volume_changes)
                if pd.isna(corr) or np.std(price_changes) == 0:
                    return 0
                
                elasticity = corr * (np.std(volume_changes) / np.std(price_changes))
                return elasticity
            
            df[f'pv_elasticity_{window}'] = df.rolling(window=window).apply(
                lambda x: price_volume_elasticity(x), raw=False
            ).iloc[:, 0]  # Get the first column result
            
        return df
    
    def _add_volume_profile_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add volume profile and distribution factors."""
        # Volume distribution measures
        for window in [20, 60]:
            df[f'volume_skew_{window}'] = df['volume'].rolling(window=window).skew()
            df[f'volume_kurtosis_{window}'] = df['volume'].rolling(window=window).kurt()
            
        # Volume concentration
        for window in [10, 20]:
            # Gini coefficient for volume concentration
            def gini_coefficient(series):
                if len(series) < 2:
                    return 0
                sorted_series = sorted(series)
                n = len(series)
                cumsum_series = np.cumsum(sorted_series)
                gini = (n + 1 - 2 * sum((n + 1 - i) * sorted_series[i] for i in range(n)) / cumsum_series[-1]) / n
                return gini
            
            df[f'volume_gini_{window}'] = df['volume'].rolling(window=window).apply(gini_coefficient)
            
        # Volume clusters (high volume days)
        volume_threshold_75 = df['volume'].rolling(window=60).quantile(0.75)
        volume_threshold_90 = df['volume'].rolling(window=60).quantile(0.90)
        
        df['volume_cluster_75'] = (df['volume'] > volume_threshold_75).astype(int)
        df['volume_cluster_90'] = (df['volume'] > volume_threshold_90).astype(int)
        
        # Volume cluster frequency
        for window in [10, 20]:
            df[f'volume_cluster_freq_{window}'] = df['volume_cluster_75'].rolling(window=window).mean()
            
        return df
    
    def _add_institutional_flow_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add factors that may indicate institutional trading."""
        # Large volume moves (potential institutional activity)
        volume_z_score = (df['volume'] - df['volume'].rolling(window=60).mean()) / df['volume'].rolling(window=60).std()
        df['large_volume_indicator'] = (volume_z_score > 2).astype(int)
        
        # Volume-price impact
        for window in [5, 10]:
            # Measure how much volume is needed to move price
            def volume_price_impact(df_slice):
                if len(df_slice) < 2:
                    return 0
                
                price_moves = np.abs(df_slice['returns'])
                volumes = df_slice['volume_log']
                
                if len(price_moves) != len(volumes) or np.std(volumes) == 0:
                    return 0
                
                # Inverse relationship: higher impact means less volume needed for price moves
                corr = price_moves.corr(volumes)
                return corr if not pd.isna(corr) else 0
            
            df[f'volume_price_impact_{window}'] = df.rolling(window=window).apply(
                lambda x: volume_price_impact(x), raw=False
            ).iloc[:, 0]
            
        # Smart money indicator (volume leading price)
        for window in [5, 10]:
            # Check if volume spikes precede price moves
            volume_spikes = (df['volume'] > df['volume'].rolling(window=20).quantile(0.8)).astype(int)
            future_returns = df['returns'].shift(-1)  # Next day's return
            
            df[f'volume_leads_price_{window}'] = volume_spikes.rolling(window=window).corr(future_returns.abs())
            
        return df
    
    def _add_liquidity_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add liquidity-related factors."""
        # Amihud illiquidity ratio
        for window in [10, 20]:
            price_impact = np.abs(df['returns']) / df['volume_log']
            df[f'amihud_illiquidity_{window}'] = price_impact.rolling(window=window).mean()
            
        # Volume-adjusted volatility
        for window in [10, 20]:
            volatility = df['returns'].rolling(window=window).std()
            avg_volume = df['volume_log'].rolling(window=window).mean()
            df[f'volume_adj_volatility_{window}'] = volatility / avg_volume
            
        # Turnover-based measures
        # Note: We approximate turnover using volume/market_cap if available
        # For now, we use volume as a proxy
        for window in [10, 20]:
            df[f'turnover_volatility_{window}'] = (df['volume'].rolling(window=window).std() / 
                                                  df['volume'].rolling(window=window).mean())
            
        return df
    
    def _add_volume_pattern_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add volume pattern recognition factors."""
        # Volume breakout patterns
        for window in [20, 40]:
            volume_upper_band = df['volume'].rolling(window=window).quantile(0.8)
            volume_lower_band = df['volume'].rolling(window=window).quantile(0.2)
            
            df[f'volume_breakout_up_{window}'] = (df['volume'] > volume_upper_band).astype(int)
            df[f'volume_breakdown_{window}'] = (df['volume'] < volume_lower_band).astype(int)
            
        # Volume trend changes
        for window in [10, 20]:
            volume_ma_short = df['volume'].rolling(window=window//2).mean()
            volume_ma_long = df['volume'].rolling(window=window).mean()
            
            df[f'volume_trend_change_{window}'] = (volume_ma_short > volume_ma_long).astype(int)
            
        # Volume seasonality (day-of-week effect)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Average volume by day of week
        for dow in range(5):  # Monday to Friday
            dow_mask = df['day_of_week'] == dow
            if dow_mask.sum() > 0:
                dow_avg_volume = df[dow_mask]['volume'].mean()
                df[f'volume_vs_dow_{dow}'] = df['volume'] / dow_avg_volume - 1
            else:
                df[f'volume_vs_dow_{dow}'] = 0
                
        return df
    
    def _add_price_volume_divergence_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add price-volume divergence factors."""
        # Price-volume divergence signals
        for window in [10, 20]:
            # Price trend
            price_ma_short = df['close'].rolling(window=window//2).mean()
            price_ma_long = df['close'].rolling(window=window).mean()
            price_trend = (price_ma_short > price_ma_long).astype(int)
            
            # Volume trend
            volume_ma_short = df['volume'].rolling(window=window//2).mean()
            volume_ma_long = df['volume'].rolling(window=window).mean()
            volume_trend = (volume_ma_short > volume_ma_long).astype(int)
            
            # Divergence: price and volume trends in opposite directions
            df[f'pv_divergence_{window}'] = (price_trend != volume_trend).astype(int)
            
        # On Balance Volume (OBV) divergence
        # Calculate OBV
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
        
        # OBV trend vs Price trend
        for window in [10, 20]:
            obv_slope = df['obv'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
            )
            price_slope = df['close'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
            )
            
            # Normalize slopes
            obv_slope_norm = obv_slope / df['obv'].rolling(window=window).mean()
            price_slope_norm = price_slope / df['close'].rolling(window=window).mean()
            
            df[f'obv_price_divergence_{window}'] = obv_slope_norm - price_slope_norm
            
        return df
    
    def _add_microstructure_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add microstructure-related factors."""
        # Intraday intensity approximation
        df['intraday_intensity'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
        df['intraday_intensity'] = df['intraday_intensity'].fillna(0)  # Handle division by zero
        
        # Money flow index approximation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        for window in [14, 21]:
            def money_flow_index(df_slice):
                if len(df_slice) < 2:
                    return 50
                    
                positive_flow = 0
                negative_flow = 0
                
                for i in range(1, len(df_slice)):
                    if df_slice['close'].iloc[i] > df_slice['close'].iloc[i-1]:
                        positive_flow += df_slice['money_flow'].iloc[i]
                    else:
                        negative_flow += df_slice['money_flow'].iloc[i]
                
                if negative_flow == 0:
                    return 100
                
                money_ratio = positive_flow / negative_flow
                mfi = 100 - (100 / (1 + money_ratio))
                return mfi
            
            df['money_flow'] = money_flow
            df[f'mfi_{window}'] = df.rolling(window=window).apply(
                lambda x: money_flow_index(x), raw=False
            ).iloc[:, 0]
            
        # Volume-weighted bid-ask spread approximation
        # Using high-low as a proxy for bid-ask spread
        spread_proxy = (df['high'] - df['low']) / df['close']
        df['volume_weighted_spread'] = spread_proxy * df['volume']
        
        for window in [10, 20]:
            df[f'avg_volume_weighted_spread_{window}'] = df['volume_weighted_spread'].rolling(window=window).mean()
            
        return df
    
    def get_factor_names(self) -> Dict[str, List[str]]:
        """Get factor names grouped by category."""
        return {
            'volume_momentum': [
                'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20', 'volume_ratio_60',
                'volume_momentum_5', 'volume_momentum_10', 'volume_momentum_20', 'volume_momentum_60',
                'volume_acceleration_5', 'volume_acceleration_10',
                'volume_trend_10', 'volume_trend_20',
                'volume_percentile_20', 'volume_percentile_60'
            ],
            
            'price_volume_correlation': [
                'pv_correlation_5', 'pv_correlation_10', 'pv_correlation_20', 'pv_correlation_60',
                'price_to_vwma_5', 'price_to_vwma_10', 'price_to_vwma_20',
                'pv_elasticity_10', 'pv_elasticity_20'
            ],
            
            'volume_profile': [
                'volume_skew_20', 'volume_skew_60', 'volume_kurtosis_20', 'volume_kurtosis_60',
                'volume_gini_10', 'volume_gini_20',
                'volume_cluster_75', 'volume_cluster_90', 'volume_cluster_freq_10', 'volume_cluster_freq_20'
            ],
            
            'institutional_flow': [
                'large_volume_indicator', 'volume_price_impact_5', 'volume_price_impact_10',
                'volume_leads_price_5', 'volume_leads_price_10'
            ],
            
            'liquidity': [
                'amihud_illiquidity_10', 'amihud_illiquidity_20',
                'volume_adj_volatility_10', 'volume_adj_volatility_20',
                'turnover_volatility_10', 'turnover_volatility_20'
            ],
            
            'volume_patterns': [
                'volume_breakout_up_20', 'volume_breakout_up_40',
                'volume_breakdown_20', 'volume_breakdown_40',
                'volume_trend_change_10', 'volume_trend_change_20'
            ],
            
            'price_volume_divergence': [
                'pv_divergence_10', 'pv_divergence_20',
                'obv_price_divergence_10', 'obv_price_divergence_20'
            ],
            
            'microstructure': [
                'intraday_intensity', 'mfi_14', 'mfi_21',
                'avg_volume_weighted_spread_10', 'avg_volume_weighted_spread_20'
            ]
        }
    
    def calculate_factor_alpha(self, factors_df: pd.DataFrame, forward_periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Calculate alpha metrics for factors.
        
        Args:
            factors_df: DataFrame with factors
            forward_periods: Periods for forward returns
            
        Returns:
            DataFrame with alpha metrics
        """
        # Add forward returns
        factors_with_returns = self._add_forward_returns(factors_df, forward_periods)
        
        # Calculate factor performance
        all_factor_names = []
        factor_categories = self.get_factor_names()
        for category, names in factor_categories.items():
            all_factor_names.extend(names)
        
        performance_results = []
        
        for factor in all_factor_names:
            if factor not in factors_with_returns.columns:
                continue
                
            factor_perf = {'factor': factor}
            
            for period in forward_periods:
                return_col = f'forward_return_{period}'
                if return_col in factors_with_returns.columns:
                    # Information Coefficient
                    ic = factors_with_returns[factor].corr(factors_with_returns[return_col])
                    factor_perf[f'ic_{period}d'] = ic
                    
                    # Rank IC (more robust)
                    rank_ic = factors_with_returns[factor].corr(factors_with_returns[return_col], method='spearman')
                    factor_perf[f'rank_ic_{period}d'] = rank_ic
                    
            performance_results.append(factor_perf)
        
        return pd.DataFrame(performance_results)
    
    def _add_forward_returns(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add forward returns to the dataframe."""
        result_dfs = []
        
        for tic in df['tic'].unique():
            tic_data = df[df['tic'] == tic].copy()
            
            for period in periods:
                tic_data[f'forward_return_{period}'] = tic_data['close'].pct_change(period).shift(-period)
            
            result_dfs.append(tic_data)
        
        return pd.concat(result_dfs, ignore_index=True)


def main():
    """Example usage of PriceVolumeAlphaFactors."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("=== Price-Volume Alpha Factors Example ===")
    print(f"Analyzing {len(tickers)} stocks: {tickers}")
    
    # Get data in required format
    df_list = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
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
    pv_factors = PriceVolumeAlphaFactors(df)
    
    # Calculate all factors
    print("\nCalculating price-volume factors...")
    factors_df = pv_factors.calculate_all_factors()
    
    print(f"Generated factors for {len(factors_df)} observations")
    
    # Show factor categories
    factor_categories = pv_factors.get_factor_categories()
    print(f"\n=== Factor Categories ===")
    total_factors = 0
    for category, factor_list in factor_categories.items():
        available_factors = [f for f in factor_list if f in factors_df.columns]
        print(f"{category}: {len(available_factors)} factors")
        total_factors += len(available_factors)
    
    print(f"Total available factors: {total_factors}")
    
    # Calculate factor alpha
    print("\nCalculating factor alpha metrics...")
    alpha_metrics = pv_factors.calculate_factor_alpha(factors_df)
    
    # Show top factors by IC
    print("\n=== Top Factors by 1-Day IC ===")
    alpha_sorted = alpha_metrics.dropna(subset=['ic_1d']).sort_values('ic_1d', key=abs, ascending=False)
    for _, row in alpha_sorted.head(10).iterrows():
        print(f"{row['factor']}: IC = {row['ic_1d']:.4f}")
    
    # Save results
    output_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/price_volume_factors_example.csv'
    factors_df.to_csv(output_path, index=False)
    
    alpha_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/pv_factor_alpha_metrics.csv'
    alpha_metrics.to_csv(alpha_path, index=False)
    
    print(f"\nFactors saved to: {output_path}")
    print(f"Alpha metrics saved to: {alpha_path}")
    print(f"Data shape: {factors_df.shape}")


if __name__ == "__main__":
    main()