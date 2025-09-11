"""
Machine Learning Alpha Factors
Advanced machine learning based alpha factors for portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    print("Warning: XGBoost and/or LightGBM not available. Some features will be limited.")


class MLAlphaFactors:
    """
    Generate machine learning based alpha factors.
    
    This class implements various ML techniques to extract alpha signals
    from market data, including ensemble methods, dimensionality reduction,
    and clustering techniques.
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
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Store fitted models
        self.fitted_models = {}
        
    def calculate_all_factors(self, 
                            prediction_horizons: List[int] = [1, 5, 10, 20],
                            feature_windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate all ML-based alpha factors.
        
        Args:
            prediction_horizons: Forward prediction horizons
            feature_windows: Windows for feature engineering
            
        Returns:
            DataFrame with ML factors
        """
        factors = []
        
        for tic in self.data['tic'].unique():
            tic_data = self.data[self.data['tic'] == tic].copy()
            tic_factors = self._calculate_tic_factors(tic_data, prediction_horizons, feature_windows)
            factors.append(tic_factors)
        
        return pd.concat(factors, ignore_index=True)
    
    def _calculate_tic_factors(self, 
                             data: pd.DataFrame, 
                             prediction_horizons: List[int],
                             feature_windows: List[int]) -> pd.DataFrame:
        """Calculate ML factors for a single ticker."""
        df = data.copy()
        
        # Basic feature engineering
        df = self._engineer_basic_features(df, feature_windows)
        
        # Add different categories of ML factors
        df = self._add_ensemble_factors(df, prediction_horizons, feature_windows)
        df = self._add_dimensionality_reduction_factors(df, feature_windows)
        df = self._add_clustering_factors(df, feature_windows)
        df = self._add_time_series_factors(df, prediction_horizons, feature_windows)
        df = self._add_cross_sectional_factors(df, feature_windows)
        df = self._add_regime_detection_factors(df, feature_windows)
        
        return df
    
    def _engineer_basic_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Engineer basic features for ML models."""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_open_ratio'] = df['close'] / df['open'] - 1
        
        # Rolling statistics
        for window in windows:
            df[f'return_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'return_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'return_skew_{window}'] = df['returns'].rolling(window=window).skew()
            df[f'return_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            
            df[f'price_zscore_{window}'] = ((df['close'] - df['close'].rolling(window=window).mean()) / 
                                          df['close'].rolling(window=window).std())
            
            df[f'volume_zscore_{window}'] = ((df['volume'] - df['volume'].rolling(window=window).mean()) / 
                                            df['volume'].rolling(window=window).std())
        
        # Momentum features
        for window in windows:
            df[f'momentum_{window}'] = df['close'].pct_change(window)
            df[f'momentum_vol_adj_{window}'] = (df[f'momentum_{window}'] / 
                                              df['returns'].rolling(window=window).std())
        
        return df
    
    def _add_ensemble_factors(self, 
                            df: pd.DataFrame, 
                            prediction_horizons: List[int], 
                            feature_windows: List[int]) -> pd.DataFrame:
        """Add ensemble model-based factors."""
        # Prepare features
        feature_columns = []
        for window in feature_windows:
            feature_columns.extend([
                f'return_mean_{window}', f'return_std_{window}', 
                f'return_skew_{window}', f'return_kurt_{window}',
                f'price_zscore_{window}', f'volume_zscore_{window}',
                f'momentum_{window}', f'momentum_vol_adj_{window}'
            ])
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) < 3 or len(df) < 50:
            # Not enough features or data
            for horizon in prediction_horizons:
                df[f'rf_prediction_{horizon}d'] = np.nan
                df[f'gbm_prediction_{horizon}d'] = np.nan
                if ADVANCED_ML_AVAILABLE:
                    df[f'xgb_prediction_{horizon}d'] = np.nan
                    df[f'lgb_prediction_{horizon}d'] = np.nan
            return df
        
        # Prepare target variables
        for horizon in prediction_horizons:
            df[f'target_{horizon}d'] = df['returns'].shift(-horizon)
        
        # Train models for each horizon
        for horizon in prediction_horizons:
            target_col = f'target_{horizon}d'
            
            # Create clean dataset
            clean_data = df[available_features + [target_col]].dropna()
            
            if len(clean_data) < 30:
                continue
            
            X = clean_data[available_features]
            y = clean_data[target_col]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data (time series split)
            train_size = int(0.8 * len(X_scaled))
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            if len(X_train) < 20 or len(X_test) < 5:
                continue
            
            # Random Forest
            try:
                rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_predictions = rf_model.predict(X_scaled)
                df[f'rf_prediction_{horizon}d'] = np.nan
                df.loc[clean_data.index, f'rf_prediction_{horizon}d'] = rf_predictions
            except:
                df[f'rf_prediction_{horizon}d'] = np.nan
            
            # Gradient Boosting
            try:
                gbm_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
                gbm_model.fit(X_train, y_train)
                gbm_predictions = gbm_model.predict(X_scaled)
                df[f'gbm_prediction_{horizon}d'] = np.nan
                df.loc[clean_data.index, f'gbm_prediction_{horizon}d'] = gbm_predictions
            except:
                df[f'gbm_prediction_{horizon}d'] = np.nan
            
            # XGBoost (if available)
            if ADVANCED_ML_AVAILABLE:
                try:
                    xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                    xgb_model.fit(X_train, y_train)
                    xgb_predictions = xgb_model.predict(X_scaled)
                    df[f'xgb_prediction_{horizon}d'] = np.nan
                    df.loc[clean_data.index, f'xgb_prediction_{horizon}d'] = xgb_predictions
                except:
                    df[f'xgb_prediction_{horizon}d'] = np.nan
                
                # LightGBM (if available)
                try:
                    lgb_model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
                    lgb_model.fit(X_train, y_train)
                    lgb_predictions = lgb_model.predict(X_scaled)
                    df[f'lgb_prediction_{horizon}d'] = np.nan
                    df.loc[clean_data.index, f'lgb_prediction_{horizon}d'] = lgb_predictions
                except:
                    df[f'lgb_prediction_{horizon}d'] = np.nan
            
        return df
    
    def _add_dimensionality_reduction_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add dimensionality reduction based factors."""
        # Prepare feature matrix
        feature_columns = ['returns', 'high_low_ratio', 'close_open_ratio']
        
        for window in windows:
            if f'return_mean_{window}' in df.columns:
                feature_columns.extend([
                    f'return_mean_{window}', f'return_std_{window}',
                    f'price_zscore_{window}', f'volume_zscore_{window}'
                ])
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) < 3 or len(df) < 50:
            # Not enough features or data
            for i in range(3):  # 3 PCA components
                df[f'pca_factor_{i+1}'] = np.nan
                df[f'ica_factor_{i+1}'] = np.nan
            return df
        
        # Create clean dataset
        feature_data = df[available_features].dropna()
        
        if len(feature_data) < 20:
            for i in range(3):
                df[f'pca_factor_{i+1}'] = np.nan
                df[f'ica_factor_{i+1}'] = np.nan
            return df
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_data)
        
        # PCA
        try:
            pca = PCA(n_components=min(3, X_scaled.shape[1]))
            pca_components = pca.fit_transform(X_scaled)
            
            for i in range(pca_components.shape[1]):
                df[f'pca_factor_{i+1}'] = np.nan
                df.loc[feature_data.index, f'pca_factor_{i+1}'] = pca_components[:, i]
        except:
            for i in range(3):
                df[f'pca_factor_{i+1}'] = np.nan
        
        # ICA
        try:
            ica = FastICA(n_components=min(3, X_scaled.shape[1]), random_state=42)
            ica_components = ica.fit_transform(X_scaled)
            
            for i in range(ica_components.shape[1]):
                df[f'ica_factor_{i+1}'] = np.nan
                df.loc[feature_data.index, f'ica_factor_{i+1}'] = ica_components[:, i]
        except:
            for i in range(3):
                df[f'ica_factor_{i+1}'] = np.nan
        
        return df
    
    def _add_clustering_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add clustering-based factors."""
        # Market regime clustering based on volatility and returns
        for window in [20, 60]:
            if window not in windows:
                continue
                
            if f'return_std_{window}' not in df.columns or f'return_mean_{window}' not in df.columns:
                continue
            
            # Features for clustering
            cluster_features = df[[f'return_mean_{window}', f'return_std_{window}']].dropna()
            
            if len(cluster_features) < 10:
                df[f'market_regime_{window}'] = np.nan
                continue
            
            # K-means clustering
            try:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cluster_features)
                
                df[f'market_regime_{window}'] = np.nan
                df.loc[cluster_features.index, f'market_regime_{window}'] = cluster_labels
                
                # Regime persistence
                df[f'regime_persistence_{window}'] = (df[f'market_regime_{window}'] == 
                                                    df[f'market_regime_{window}'].shift(1)).astype(int)
                
            except:
                df[f'market_regime_{window}'] = np.nan
                df[f'regime_persistence_{window}'] = np.nan
        
        return df
    
    def _add_time_series_factors(self, 
                               df: pd.DataFrame, 
                               prediction_horizons: List[int], 
                               windows: List[int]) -> pd.DataFrame:
        """Add time series analysis factors."""
        # Autoregressive features
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        # Moving average crossover signals
        if len(windows) >= 2:
            short_window = min(windows)
            long_window = max(windows)
            
            if f'return_mean_{short_window}' in df.columns and f'return_mean_{long_window}' in df.columns:
                df[f'ma_crossover_{short_window}_{long_window}'] = (
                    df[f'return_mean_{short_window}'] > df[f'return_mean_{long_window}']
                ).astype(int)
        
        # Trend strength using linear regression
        for window in [10, 20]:
            if window not in windows:
                continue
                
            def trend_strength(series):
                if len(series) < 5:
                    return 0
                x = np.arange(len(series))
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                    return slope * r_value  # Slope weighted by R-squared
                except:
                    return 0
            
            df[f'trend_strength_{window}'] = df['close'].rolling(window=window).apply(trend_strength)
        
        return df
    
    def _add_cross_sectional_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add cross-sectional factors (would be computed across all stocks)."""
        # Since we're processing one stock at a time, we'll create placeholders
        # These would be filled in by the comprehensive factor calculation
        
        for window in windows:
            df[f'cross_sectional_rank_{window}'] = np.nan  # Placeholder
            df[f'cross_sectional_zscore_{window}'] = np.nan  # Placeholder
        
        return df
    
    def _add_regime_detection_factors(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add regime detection factors."""
        # Volatility regime detection
        for window in [20, 40]:
            if window not in windows:
                continue
                
            if f'return_std_{window}' not in df.columns:
                continue
            
            # High/Low volatility regime
            vol_median = df[f'return_std_{window}'].rolling(window=60).median()
            df[f'high_vol_regime_{window}'] = (df[f'return_std_{window}'] > vol_median).astype(int)
            
            # Regime changes
            df[f'vol_regime_change_{window}'] = (df[f'high_vol_regime_{window}'] != 
                                               df[f'high_vol_regime_{window}'].shift(1)).astype(int)
        
        # Momentum regime detection
        for window in [20, 40]:
            if window not in windows:
                continue
                
            if f'momentum_{window}' not in df.columns:
                continue
            
            # Trending vs. mean-reverting regime
            momentum_abs_mean = df[f'momentum_{window}'].abs().rolling(window=60).mean()
            df[f'trending_regime_{window}'] = (df[f'momentum_{window}'].abs() > momentum_abs_mean).astype(int)
        
        return df
    
    def create_ensemble_alpha_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble alpha factors combining multiple ML approaches."""
        df = factors_df.copy()
        
        # Ensemble of predictions
        prediction_columns = [col for col in df.columns if 'prediction' in col]
        
        if len(prediction_columns) >= 2:
            # Simple average ensemble
            df['ensemble_prediction_avg'] = df[prediction_columns].mean(axis=1)
            
            # Weighted ensemble (weight by recent performance)
            # This is a simplified version - in practice you'd use validation performance
            weights = np.ones(len(prediction_columns)) / len(prediction_columns)
            df['ensemble_prediction_weighted'] = df[prediction_columns].multiply(weights).sum(axis=1)
        
        # Meta-learning factor: consistency across models
        if len(prediction_columns) >= 3:
            df['prediction_consensus'] = df[prediction_columns].std(axis=1)  # Lower std = higher consensus
            df['prediction_consensus'] = 1 / (1 + df['prediction_consensus'])  # Transform to 0-1 range
        
        # Regime-adjusted factors
        regime_columns = [col for col in df.columns if 'regime' in col]
        
        if len(regime_columns) >= 1:
            # Adjust momentum based on regime
            for regime_col in regime_columns:
                if 'momentum' in df.columns:
                    df[f'regime_adj_momentum_{regime_col}'] = df['momentum'] * df[regime_col]
        
        return df
    
    def get_factor_names(self) -> Dict[str, List[str]]:
        """Get factor names grouped by category."""
        factor_names = {
            'ensemble_predictions': [],
            'dimensionality_reduction': ['pca_factor_1', 'pca_factor_2', 'pca_factor_3',
                                       'ica_factor_1', 'ica_factor_2', 'ica_factor_3'],
            'clustering': ['market_regime_20', 'market_regime_60', 
                         'regime_persistence_20', 'regime_persistence_60'],
            'time_series': ['return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
                          'trend_strength_10', 'trend_strength_20'],
            'regime_detection': ['high_vol_regime_20', 'high_vol_regime_40',
                               'vol_regime_change_20', 'vol_regime_change_40',
                               'trending_regime_20', 'trending_regime_40'],
            'ensemble_meta': ['ensemble_prediction_avg', 'ensemble_prediction_weighted',
                            'prediction_consensus']
        }
        
        # Add prediction factors dynamically
        for horizon in [1, 5, 10, 20]:
            factor_names['ensemble_predictions'].extend([
                f'rf_prediction_{horizon}d', f'gbm_prediction_{horizon}d'
            ])
            if ADVANCED_ML_AVAILABLE:
                factor_names['ensemble_predictions'].extend([
                    f'xgb_prediction_{horizon}d', f'lgb_prediction_{horizon}d'
                ])
        
        return factor_names
    
    def evaluate_factor_performance(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate ML factor performance."""
        # Add forward returns for evaluation
        factors_with_returns = self._add_forward_returns(factors_df, [1, 5, 10])
        
        # Get all ML factor names
        all_factors = []
        factor_categories = self.get_factor_names()
        for category, factors in factor_categories.items():
            all_factors.extend(factors)
        
        # Filter factors that exist in the dataframe
        available_factors = [f for f in all_factors if f in factors_with_returns.columns]
        
        # Calculate performance metrics
        results = []
        
        for factor in available_factors:
            factor_result = {'factor': factor, 'category': self._get_factor_category(factor)}
            
            # Information Coefficient with different horizons
            for horizon in [1, 5, 10]:
                return_col = f'forward_return_{horizon}d'
                if return_col in factors_with_returns.columns:
                    ic = factors_with_returns[factor].corr(factors_with_returns[return_col])
                    factor_result[f'ic_{horizon}d'] = ic
                    
                    # Rank IC
                    rank_ic = factors_with_returns[factor].corr(
                        factors_with_returns[return_col], method='spearman'
                    )
                    factor_result[f'rank_ic_{horizon}d'] = rank_ic
            
            # Factor stability (autocorrelation)
            factor_result['stability'] = factors_with_returns[factor].autocorr()
            
            # Factor turnover (change frequency)
            factor_changes = factors_with_returns[factor].diff().abs()
            factor_result['turnover'] = factor_changes.mean() / factors_with_returns[factor].std()
            
            results.append(factor_result)
        
        return pd.DataFrame(results)
    
    def _get_factor_category(self, factor_name: str) -> str:
        """Get category for a factor name."""
        factor_categories = self.get_factor_names()
        for category, factors in factor_categories.items():
            if factor_name in factors:
                return category
        return 'other'
    
    def _add_forward_returns(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """Add forward returns for evaluation."""
        result_dfs = []
        
        for tic in df['tic'].unique():
            tic_data = df[df['tic'] == tic].copy()
            
            for horizon in horizons:
                tic_data[f'forward_return_{horizon}d'] = tic_data['close'].pct_change(horizon).shift(-horizon)
            
            result_dfs.append(tic_data)
        
        return pd.concat(result_dfs, ignore_index=True)


def main():
    """Example usage of MLAlphaFactors."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    print("=== Machine Learning Alpha Factors Example ===")
    print(f"Analyzing {len(tickers)} stocks: {tickers}")
    print(f"Advanced ML libraries available: {ADVANCED_ML_AVAILABLE}")
    
    # Get data in required format
    df_list = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        import yfinance as yf
        stock_data = yf.download(ticker, period="2y", progress=False)
        
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
    
    # Initialize ML factor calculator
    ml_factors = MLAlphaFactors(df)
    
    # Calculate all factors
    print("\nCalculating ML-based factors...")
    factors_df = ml_factors.calculate_all_factors(
        prediction_horizons=[1, 5, 10],
        feature_windows=[5, 10, 20]
    )
    
    print(f"Generated factors for {len(factors_df)} observations")
    
    # Create ensemble factors
    print("Creating ensemble factors...")
    factors_df = ml_factors.create_ensemble_alpha_factors(factors_df)
    
    # Show factor categories
    factor_categories = ml_factors.get_factor_names()
    print(f"\n=== ML Factor Categories ===")
    total_factors = 0
    for category, factor_list in factor_categories.items():
        available_factors = [f for f in factor_list if f in factors_df.columns]
        print(f"{category}: {len(available_factors)} factors")
        total_factors += len(available_factors)
    
    print(f"Total available ML factors: {total_factors}")
    
    # Evaluate factor performance
    print("\nEvaluating factor performance...")
    performance_df = ml_factors.evaluate_factor_performance(factors_df)
    
    # Show top factors
    if not performance_df.empty and 'ic_1d' in performance_df.columns:
        print("\n=== Top ML Factors by 1-Day IC ===")
        top_factors = performance_df.dropna(subset=['ic_1d']).sort_values('ic_1d', key=abs, ascending=False)
        for _, row in top_factors.head(10).iterrows():
            print(f"{row['factor']} ({row['category']}): IC = {row['ic_1d']:.4f}")
    
    # Save results
    output_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/ml_alpha_factors_example.csv'
    factors_df.to_csv(output_path, index=False)
    
    if not performance_df.empty:
        perf_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/ml_factor_performance.csv'
        performance_df.to_csv(perf_path, index=False)
        print(f"Performance metrics saved to: {perf_path}")
    
    print(f"\nML factors saved to: {output_path}")
    print(f"Data shape: {factors_df.shape}")
    print(f"Non-null factor columns: {factors_df.count().sum()}")


if __name__ == "__main__":
    main()