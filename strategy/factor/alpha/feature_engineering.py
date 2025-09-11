"""
Financial Feature Engineering Module

Provides a rich set of features for machine learning Alpha extraction,
including technical indicators, fundamental indicators, market microstructure,
and macroeconomic variables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

class FinancialFeatureEngineer:
    """Financial Feature Engineer"""
    
    def __init__(self,
                 lookback_periods: List[int] = [5, 10, 20, 60, 120],
                 volatility_periods: List[int] = [5, 20, 60],
                 momentum_periods: List[int] = [5, 10, 20, 60],
                 standardize_features: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            lookback_periods: periods used for rolling indicators
            volatility_periods: periods used for volatility measures
            momentum_periods: periods used for momentum features
            standardize_features: whether to standardize features
        """
        self.lookback_periods = lookback_periods
        self.volatility_periods = volatility_periods
        self.momentum_periods = momentum_periods
        self.standardize_features = standardize_features
        
        self.scaler = RobustScaler() if standardize_features else None
        self.feature_names_ = []
        
    def create_features(self, 
                       prices: pd.DataFrame,
                       volumes: Optional[pd.DataFrame] = None,
                       market_data: Optional[pd.DataFrame] = None,
                       macro_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Construct the full feature set.
        
        Args:
            prices: price data (T, N)
            volumes: volume data (T, N)
            market_data: market-level variables (VIX, interest rates, etc.)
            macro_data: macroeconomic variables
            
        Returns:
            Feature matrix (T, F*N) where F = number of features, N = number of assets
        """
        logger.info("Building financial features...")
        
        all_features = []
        self.feature_names_ = []
        
        # 1. Price-based technical indicators
        price_features = self._create_price_features(prices)
        all_features.append(price_features)
        
        # 2. Return-based features
        returns = prices.pct_change()
        return_features = self._create_return_features(returns)
        all_features.append(return_features)
        
        # 3. Volatility features
        volatility_features = self._create_volatility_features(returns)
        all_features.append(volatility_features)
        
        # 4. Momentum and reversal
        momentum_features = self._create_momentum_features(prices, returns)
        all_features.append(momentum_features)
        
        # 5. Volume-based features
        if volumes is not None:
            volume_features = self._create_volume_features(prices, volumes)
            all_features.append(volume_features)
        
        # 6. Cross-sectional features (relative ranking across assets)
        cross_sectional_features = self._create_cross_sectional_features(prices, returns)
        all_features.append(cross_sectional_features)
        
        # 7. Market regime features
        if market_data is not None:
            regime_features = self._create_market_regime_features(returns, market_data)
            all_features.append(regime_features)
        
        # 8. Macroeconomic features
        if macro_data is not None:
            macro_features = self._create_macro_features(macro_data, len(prices))
            all_features.append(macro_features)
        
        # Merge all features
        combined_features = pd.concat(all_features, axis=1)
        
        # Handle missing values
        combined_features = self._handle_missing_values(combined_features)
        
        # Standardize if required
        if self.standardize_features:
            combined_features = self._standardize_features(combined_features)
        
        logger.info(f"Feature construction complete: {combined_features.shape[1]} features")
        
        return combined_features
    
    def _create_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create price-based technical indicators"""
        features = []
        
        for period in self.lookback_periods:
            # Simple moving average
            sma = prices.rolling(window=period).mean()
            price_to_sma = prices / sma - 1
            features.append(price_to_sma.add_suffix(f'_price_to_sma_{period}'))
            
            # Exponential moving average
            ema = prices.ewm(span=period).mean()
            price_to_ema = prices / ema - 1
            features.append(price_to_ema.add_suffix(f'_price_to_ema_{period}'))
            
            # Bollinger Bands
            rolling_std = prices.rolling(window=period).std()
            bb_upper = sma + 2 * rolling_std
            bb_lower = sma - 2 * rolling_std
            bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
            features.append(bb_position.add_suffix(f'_bb_position_{period}'))
            
            # Price channel position
            rolling_max = prices.rolling(window=period).max()
            rolling_min = prices.rolling(window=period).min()
            channel_position = (prices - rolling_min) / (rolling_max - rolling_min)
            features.append(channel_position.add_suffix(f'_channel_pos_{period}'))
        
        # RSI
        rsi_features = self._calculate_rsi(prices, period=14)
        features.append(rsi_features)
        
        # MACD
        macd_features = self._calculate_macd(prices)
        features.append(macd_features)
        
        self.feature_names_.extend([col for df in features for col in df.columns])
        return pd.concat(features, axis=1)
    
    def _create_return_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create return-based features"""
        features = []
        
        for period in self.lookback_periods:
            # Cumulative returns
            cum_returns = (1 + returns).rolling(window=period).apply(np.prod) - 1
            features.append(cum_returns.add_suffix(f'_cum_ret_{period}'))
            
            # Return statistics
            mean_returns = returns.rolling(window=period).mean()
            features.append(mean_returns.add_suffix(f'_mean_ret_{period}'))
            
            std_returns = returns.rolling(window=period).std()
            features.append(std_returns.add_suffix(f'_std_ret_{period}'))
            
            # Skewness and kurtosis
            skew_returns = returns.rolling(window=period).skew()
            features.append(skew_returns.add_suffix(f'_skew_ret_{period}'))
            
            kurt_returns = returns.rolling(window=period).kurt()
            features.append(kurt_returns.add_suffix(f'_kurt_ret_{period}'))
            
            # Sharpe ratio approximation
            sharpe_ratio = mean_returns / (std_returns + 1e-8)
            features.append(sharpe_ratio.add_suffix(f'_sharpe_{period}'))
        
        self.feature_names_.extend([col for df in features for col in df.columns])
        return pd.concat(features, axis=1)
    
    def _create_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-related features"""
        features = []
        
        for period in self.volatility_periods:
            # Historical volatility
            hist_vol = returns.rolling(window=period).std() * np.sqrt(252)
            features.append(hist_vol.add_suffix(f'_hist_vol_{period}'))
            
            # Simplified GARCH-style volatility
            garch_vol = self._calculate_garch_volatility(returns, period)
            features.append(garch_vol.add_suffix(f'_garch_vol_{period}'))
            
            # Volatility change
            vol_change = hist_vol / hist_vol.shift(period) - 1
            features.append(vol_change.add_suffix(f'_vol_change_{period}'))
            
            # Volatility rank
            vol_rank = hist_vol.rolling(window=period*2).rank(pct=True)
            features.append(vol_rank.add_suffix(f'_vol_rank_{period}'))
        
        # Realized volatility (example: 20-day window)
        realized_vol = returns.rolling(window=20).std() * np.sqrt(252)
        features.append(realized_vol.add_suffix('_realized_vol_20'))
        
        self.feature_names_.extend([col for df in features for col in df.columns])
        return pd.concat(features, axis=1)
    
    def _create_momentum_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and reversal features"""
        features = []
        
        for period in self.momentum_periods:
            # Price momentum
            price_momentum = prices / prices.shift(period) - 1
            features.append(price_momentum.add_suffix(f'_price_mom_{period}'))
            
            # Return momentum
            return_momentum = returns.rolling(window=period).mean()
            features.append(return_momentum.add_suffix(f'_ret_mom_{period}'))
            
            # Momentum strength
            mom_strength = np.abs(return_momentum) / (returns.rolling(window=period).std() + 1e-8)
            features.append(mom_strength.add_suffix(f'_mom_strength_{period}'))
            
            # Reversal indicator
            reversal = -returns.rolling(window=period).mean()
            features.append(reversal.add_suffix(f'_reversal_{period}'))
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(prices)
        features.append(trend_strength)
        
        self.feature_names_.extend([col for df in features for col in df.columns])
        return pd.concat(features, axis=1)
    
    def _create_volume_features(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        features = []
        returns = prices.pct_change()
        
        for period in [5, 10, 20]:
            # Volume moving average ratio
            vol_ma = volumes.rolling(window=period).mean()
            vol_ratio = volumes / vol_ma
            features.append(vol_ratio.add_suffix(f'_vol_ratio_{period}'))
            
            # Price-volume correlation
            price_volume_corr = returns.rolling(window=period).corr(volumes.pct_change())
            features.append(price_volume_corr.add_suffix(f'_pv_corr_{period}'))
            
            # Money flow index (simplified)
            money_flow = (prices * volumes).rolling(window=period).sum()
            mf_ratio = money_flow / money_flow.shift(period) - 1
            features.append(mf_ratio.add_suffix(f'_money_flow_{period}'))
        
        # On-Balance Volume (OBV)
        obv = self._calculate_obv(prices, volumes)
        features.append(obv)
        
        # Volume Weighted Average Price (VWAP)
        vwap = self._calculate_vwap(prices, volumes)
        price_to_vwap = prices / vwap - 1
        features.append(price_to_vwap.add_suffix('_price_to_vwap'))
        
        self.feature_names_.extend([col for df in features for col in df.columns])
        return pd.concat(features, axis=1)
