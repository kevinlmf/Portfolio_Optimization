"""
Multi-Factor Risk Models for Beta Estimation
Advanced multi-factor approaches for systematic risk estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MultiFactorModels:
    """
    Multi-factor risk model implementations for beta estimation.
    
    This class provides various multi-factor models including Fama-French,
    Carhart, Principal Component Analysis, and custom factor models.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 factor_data: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize with return data and optional factor data.
        
        Args:
            data: DataFrame with asset returns
            factor_data: DataFrame with factor returns (SMB, HML, etc.)
            risk_free_rate: Annual risk-free rate
        """
        self.data = data.copy()
        self.factor_data = factor_data
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252
        
        # Prepare data
        self.returns_data = self._prepare_returns_data()
        self.factor_returns = self._prepare_factor_data()
        
    def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare returns data."""
        if 'close' in self.data.columns:
            # Convert price data to returns
            returns_list = []
            
            for tic in self.data['tic'].unique():
                tic_data = self.data[self.data['tic'] == tic].copy()
                tic_data = tic_data.sort_values('date')
                tic_data['returns'] = tic_data['close'].pct_change()
                returns_list.append(tic_data[['date', 'tic', 'returns']])
            
            returns_df = pd.concat(returns_list, ignore_index=True)
            returns_wide = returns_df.pivot(index='date', columns='tic', values='returns')
            returns_wide.index = pd.to_datetime(returns_wide.index)
            
            return returns_wide
        else:
            return self.data.copy()
    
    def _prepare_factor_data(self) -> Optional[pd.DataFrame]:
        """Prepare factor data."""
        if self.factor_data is not None:
            factor_df = self.factor_data.copy()
            if 'date' in factor_df.columns:
                factor_df['date'] = pd.to_datetime(factor_df['date'])
                factor_df = factor_df.set_index('date')
            
            return factor_df
        else:
            # Create synthetic factors if none provided
            return self._create_synthetic_factors()
    
    def _create_synthetic_factors(self) -> pd.DataFrame:
        """Create synthetic factors from return data."""
        if self.returns_data.empty or len(self.returns_data.columns) < 5:
            return pd.DataFrame()
        
        # Create synthetic factors using PCA
        returns_clean = self.returns_data.dropna()
        
        if len(returns_clean) < 50:
            return pd.DataFrame()
        
        # Standardize returns
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns_clean)
        
        # Apply PCA to create factors
        n_factors = min(5, returns_scaled.shape[1])
        pca = PCA(n_components=n_factors)
        factor_loadings = pca.fit_transform(returns_scaled)
        
        # Create factor DataFrame
        factor_names = [f'PC{i+1}' for i in range(n_factors)]
        factor_df = pd.DataFrame(
            factor_loadings,
            index=returns_clean.index,
            columns=factor_names
        )
        
        # Add market factor (average of all returns)
        factor_df['Market'] = returns_clean.mean(axis=1)
        
        return factor_df
    
    def estimate_fama_french_model(self, 
                                 asset: str, 
                                 window: int = 252,
                                 rolling: bool = False,
                                 include_momentum: bool = True) -> Union[Dict, pd.DataFrame]:
        """
        Estimate Fama-French (3 or 4 factor) model.
        
        Args:
            asset: Asset symbol
            window: Estimation window
            rolling: Whether to calculate rolling estimates
            include_momentum: Whether to include momentum factor (Carhart 4-factor)
            
        Returns:
            Factor loadings and model statistics
        """
        if asset not in self.returns_data.columns:
            return {'market_beta': np.nan, 'smb_beta': np.nan, 'hml_beta': np.nan}
        
        asset_returns = self.returns_data[asset].dropna()
        
        # Create factors if not provided
        factors = self._get_fama_french_factors(include_momentum)
        
        if factors.empty:
            return {'market_beta': np.nan, 'smb_beta': np.nan, 'hml_beta': np.nan}
        
        # Align data
        common_dates = asset_returns.index.intersection(factors.index)
        asset_returns = asset_returns[common_dates]
        factors = factors.loc[common_dates]
        
        if len(asset_returns) < window:
            return {'market_beta': np.nan, 'smb_beta': np.nan, 'hml_beta': np.nan}
        
        # Excess returns
        y = asset_returns - self.daily_rf_rate
        
        factor_columns = ['Market']
        if 'SMB' in factors.columns:
            factor_columns.append('SMB')
        if 'HML' in factors.columns:
            factor_columns.append('HML')
        if include_momentum and 'MOM' in factors.columns:
            factor_columns.append('MOM')
        
        X = factors[factor_columns]
        
        if rolling:
            # Rolling factor model estimation
            results = []
            
            for i in range(window, len(y)):
                window_y = y.iloc[i-window:i]
                window_X = X.iloc[i-window:i]
                
                try:
                    # Regression
                    regressor = LinearRegression()
                    regressor.fit(window_X, window_y)
                    
                    # Extract coefficients
                    result = {
                        'date': y.index[i],
                        'alpha': regressor.intercept_,
                        'r_squared': r2_score(window_y, regressor.predict(window_X))
                    }
                    
                    for j, factor in enumerate(factor_columns):
                        result[f'{factor.lower()}_beta'] = regressor.coef_[j]
                    
                    results.append(result)
                    
                except:
                    # Handle errors
                    result = {'date': y.index[i], 'alpha': np.nan, 'r_squared': np.nan}
                    for factor in factor_columns:
                        result[f'{factor.lower()}_beta'] = np.nan
                    results.append(result)
            
            return pd.DataFrame(results).set_index('date')
        else:
            # Static estimation
            try:
                recent_y = y.tail(window)
                recent_X = X.tail(window)
                
                regressor = LinearRegression()
                regressor.fit(recent_X, recent_y)
                
                result = {
                    'alpha': regressor.intercept_,
                    'r_squared': r2_score(recent_y, regressor.predict(recent_X)),
                    'observations': len(recent_y)
                }
                
                for i, factor in enumerate(factor_columns):
                    result[f'{factor.lower()}_beta'] = regressor.coef_[i]
                
                # T-statistics
                residuals = recent_y - regressor.predict(recent_X)
                mse = np.mean(residuals**2)
                var_coef = mse * np.linalg.inv(recent_X.T @ recent_X).diagonal()
                
                for i, factor in enumerate(factor_columns):
                    if var_coef[i] > 0:
                        t_stat = regressor.coef_[i] / np.sqrt(var_coef[i])
                        result[f'{factor.lower()}_tstat'] = t_stat
                
                return result
                
            except:
                result = {'alpha': np.nan, 'r_squared': np.nan}
                for factor in factor_columns:
                    result[f'{factor.lower()}_beta'] = np.nan
                return result
    
    def _get_fama_french_factors(self, include_momentum: bool = True) -> pd.DataFrame:
        """Get or create Fama-French factors."""
        if self.factor_returns is not None and not self.factor_returns.empty:
            return self.factor_returns
        
        # Create synthetic Fama-French-like factors
        returns_clean = self.returns_data.dropna()
        
        if len(returns_clean) < 50 or len(returns_clean.columns) < 10:
            return pd.DataFrame()
        
        # Market factor (equal-weighted average)
        market_factor = returns_clean.mean(axis=1)
        
        # Size factor (SMB) - proxy using volatility
        volatilities = returns_clean.rolling(window=20).std().iloc[-1]
        small_stocks = volatilities.nlargest(len(volatilities)//3).index
        big_stocks = volatilities.nsmallest(len(volatilities)//3).index
        
        smb_factor = (returns_clean[small_stocks].mean(axis=1) - 
                     returns_clean[big_stocks].mean(axis=1))
        
        # Value factor (HML) - proxy using momentum
        momentum = returns_clean.rolling(window=60).mean().iloc[-1]
        high_momentum = momentum.nlargest(len(momentum)//3).index
        low_momentum = momentum.nsmallest(len(momentum)//3).index
        
        hml_factor = (returns_clean[low_momentum].mean(axis=1) - 
                     returns_clean[high_momentum].mean(axis=1))
        
        factors = pd.DataFrame({
            'Market': market_factor,
            'SMB': smb_factor,
            'HML': hml_factor
        })
        
        if include_momentum:
            # Momentum factor
            mom_factor = (returns_clean[high_momentum].mean(axis=1) - 
                         returns_clean[low_momentum].mean(axis=1))
            factors['MOM'] = mom_factor
        
        return factors
    
    def estimate_pca_factor_model(self, 
                                asset: str, 
                                n_factors: int = 5,
                                window: int = 252,
                                rolling: bool = False) -> Union[Dict, pd.DataFrame]:
        """
        Estimate PCA-based factor model.
        
        Args:
            asset: Asset symbol
            n_factors: Number of principal components
            window: Estimation window
            rolling: Whether to calculate rolling estimates
            
        Returns:
            Factor loadings and model statistics
        """
        if asset not in self.returns_data.columns:
            return {f'pc{i+1}_loading': np.nan for i in range(n_factors)}
        
        returns_clean = self.returns_data.dropna()
        asset_returns = returns_clean[asset]
        
        if len(returns_clean) < window:
            return {f'pc{i+1}_loading': np.nan for i in range(n_factors)}
        
        if rolling:
            # Rolling PCA factor model
            results = []
            
            for i in range(window, len(returns_clean)):
                window_returns = returns_clean.iloc[i-window:i]
                window_asset = asset_returns.iloc[i-window:i]
                
                try:
                    # PCA on window data
                    other_assets = [col for col in window_returns.columns if col != asset]
                    if len(other_assets) < n_factors:
                        continue
                    
                    factor_data = window_returns[other_assets]
                    scaler = StandardScaler()
                    factor_data_scaled = scaler.fit_transform(factor_data)
                    
                    pca = PCA(n_components=min(n_factors, factor_data_scaled.shape[1]))
                    factors = pca.fit_transform(factor_data_scaled)
                    
                    # Regression of asset returns on factors
                    regressor = LinearRegression()
                    regressor.fit(factors, window_asset)
                    
                    result = {
                        'date': returns_clean.index[i],
                        'alpha': regressor.intercept_,
                        'r_squared': r2_score(window_asset, regressor.predict(factors))
                    }
                    
                    for j in range(len(regressor.coef_)):
                        result[f'pc{j+1}_loading'] = regressor.coef_[j]
                        result[f'pc{j+1}_variance_explained'] = pca.explained_variance_ratio_[j]
                    
                    results.append(result)
                    
                except:
                    result = {'date': returns_clean.index[i], 'alpha': np.nan, 'r_squared': np.nan}
                    for j in range(n_factors):
                        result[f'pc{j+1}_loading'] = np.nan
                    results.append(result)
            
            return pd.DataFrame(results).set_index('date')
        else:
            # Static PCA factor model
            try:
                recent_returns = returns_clean.tail(window)
                recent_asset = asset_returns.tail(window)
                
                # Create factors from other assets
                other_assets = [col for col in recent_returns.columns if col != asset]
                factor_data = recent_returns[other_assets]
                
                scaler = StandardScaler()
                factor_data_scaled = scaler.fit_transform(factor_data)
                
                pca = PCA(n_components=min(n_factors, factor_data_scaled.shape[1]))
                factors = pca.fit_transform(factor_data_scaled)
                
                # Regression
                regressor = LinearRegression()
                regressor.fit(factors, recent_asset)
                
                result = {
                    'alpha': regressor.intercept_,
                    'r_squared': r2_score(recent_asset, regressor.predict(factors)),
                    'total_variance_explained': pca.explained_variance_ratio_.sum()
                }
                
                for i in range(len(regressor.coef_)):
                    result[f'pc{i+1}_loading'] = regressor.coef_[i]
                    result[f'pc{i+1}_variance_explained'] = pca.explained_variance_ratio_[i]
                
                return result
                
            except:
                return {f'pc{i+1}_loading': np.nan for i in range(n_factors)}
    
    def estimate_custom_factor_model(self, 
                                   asset: str, 
                                   custom_factors: List[str],
                                   window: int = 252,
                                   regularization: str = 'none',
                                   alpha: float = 0.01) -> Dict:
        """
        Estimate custom factor model with specified factors.
        
        Args:
            asset: Asset symbol
            custom_factors: List of custom factor names
            window: Estimation window
            regularization: 'none', 'ridge', 'lasso', 'elastic_net'
            alpha: Regularization strength
            
        Returns:
            Factor loadings and model statistics
        """
        if asset not in self.returns_data.columns:
            return {f'{factor}_beta': np.nan for factor in custom_factors}
        
        if self.factor_returns is None or self.factor_returns.empty:
            return {f'{factor}_beta': np.nan for factor in custom_factors}
        
        # Check if custom factors are available
        available_factors = [f for f in custom_factors if f in self.factor_returns.columns]
        
        if not available_factors:
            return {f'{factor}_beta': np.nan for factor in custom_factors}
        
        asset_returns = self.returns_data[asset].dropna()
        factor_data = self.factor_returns[available_factors]
        
        # Align data
        common_dates = asset_returns.index.intersection(factor_data.index)
        asset_returns = asset_returns[common_dates]
        factor_data = factor_data.loc[common_dates]
        
        if len(asset_returns) < window:
            return {f'{factor}_beta': np.nan for factor in custom_factors}
        
        # Use recent data
        recent_y = (asset_returns - self.daily_rf_rate).tail(window)
        recent_X = factor_data.tail(window)
        
        try:
            # Choose regression method
            if regularization == 'ridge':
                regressor = Ridge(alpha=alpha)
            elif regularization == 'lasso':
                regressor = Lasso(alpha=alpha, max_iter=1000)
            elif regularization == 'elastic_net':
                regressor = ElasticNet(alpha=alpha, max_iter=1000)
            else:
                regressor = LinearRegression()
            
            regressor.fit(recent_X, recent_y)
            
            result = {
                'alpha': regressor.intercept_,
                'r_squared': r2_score(recent_y, regressor.predict(recent_X)),
                'regularization': regularization
            }
            
            for i, factor in enumerate(available_factors):
                result[f'{factor}_beta'] = regressor.coef_[i]
            
            # Add NaN for missing factors
            for factor in custom_factors:
                if factor not in available_factors:
                    result[f'{factor}_beta'] = np.nan
            
            return result
            
        except:
            return {f'{factor}_beta': np.nan for factor in custom_factors}
    
    def estimate_factor_portfolio_model(self, 
                                      assets: List[str], 
                                      weights: Optional[np.ndarray] = None,
                                      model_type: str = 'fama_french',
                                      **kwargs) -> Dict:
        """
        Estimate factor model for a portfolio.
        
        Args:
            assets: List of asset symbols
            weights: Portfolio weights
            model_type: 'fama_french', 'pca', 'custom'
            **kwargs: Additional parameters for specific models
            
        Returns:
            Portfolio factor loadings
        """
        if weights is None:
            weights = np.ones(len(assets)) / len(assets)
        
        weights = np.array(weights)
        
        # Estimate individual factor loadings
        individual_results = {}
        
        for asset in assets:
            if model_type == 'fama_french':
                result = self.estimate_fama_french_model(asset, **kwargs)
            elif model_type == 'pca':
                result = self.estimate_pca_factor_model(asset, **kwargs)
            elif model_type == 'custom':
                result = self.estimate_custom_factor_model(asset, **kwargs)
            else:
                result = {}
            
            individual_results[asset] = result
        
        # Aggregate to portfolio level
        portfolio_loadings = {}
        
        # Get all factor names
        all_factors = set()
        for result in individual_results.values():
            if isinstance(result, dict):
                for key in result.keys():
                    if key.endswith('_beta') or key.endswith('_loading'):
                        all_factors.add(key)
        
        # Calculate weighted average loadings
        for factor in all_factors:
            loadings = []
            asset_weights = []
            
            for i, asset in enumerate(assets):
                if asset in individual_results and factor in individual_results[asset]:
                    loading = individual_results[asset][factor]
                    if not np.isnan(loading):
                        loadings.append(loading)
                        asset_weights.append(weights[i])
            
            if loadings:
                asset_weights = np.array(asset_weights)
                asset_weights = asset_weights / asset_weights.sum()  # Renormalize
                portfolio_loadings[factor] = np.sum(np.array(loadings) * asset_weights)
            else:
                portfolio_loadings[factor] = np.nan
        
        return {
            'portfolio_loadings': portfolio_loadings,
            'individual_results': individual_results,
            'effective_weights': dict(zip(assets, weights))
        }
    
    def analyze_factor_exposures(self, 
                               assets: List[str], 
                               model_type: str = 'fama_french') -> pd.DataFrame:
        """
        Analyze factor exposures across multiple assets.
        
        Args:
            assets: List of asset symbols
            model_type: Factor model type
            
        Returns:
            DataFrame with factor exposures
        """
        results = []
        
        for asset in assets:
            if model_type == 'fama_french':
                result = self.estimate_fama_french_model(asset)
            elif model_type == 'pca':
                result = self.estimate_pca_factor_model(asset)
            else:
                continue
            
            if isinstance(result, dict):
                result['asset'] = asset
                results.append(result)
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def perform_factor_attribution(self, 
                                 portfolio_returns: pd.Series,
                                 model_type: str = 'fama_french',
                                 **kwargs) -> Dict:
        """
        Perform factor attribution analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            model_type: Factor model type
            **kwargs: Model parameters
            
        Returns:
            Factor attribution results
        """
        # Create temporary asset for portfolio
        temp_data = self.returns_data.copy()
        temp_data['PORTFOLIO'] = portfolio_returns
        
        # Temporarily update data
        original_data = self.returns_data
        self.returns_data = temp_data
        
        try:
            if model_type == 'fama_french':
                result = self.estimate_fama_french_model('PORTFOLIO', **kwargs)
            elif model_type == 'pca':
                result = self.estimate_pca_factor_model('PORTFOLIO', **kwargs)
            else:
                result = {}
            
            # Calculate attribution
            if isinstance(result, dict) and 'r_squared' in result:
                # Factor contribution to returns
                factor_contribution = {}
                factors = self._get_fama_french_factors()
                
                if not factors.empty:
                    common_dates = portfolio_returns.index.intersection(factors.index)
                    aligned_factors = factors.loc[common_dates]
                    
                    for factor_col in aligned_factors.columns:
                        beta_key = f'{factor_col.lower()}_beta'
                        if beta_key in result and not np.isnan(result[beta_key]):
                            factor_return = aligned_factors[factor_col].mean()
                            contribution = result[beta_key] * factor_return
                            factor_contribution[factor_col] = contribution
                
                result['factor_contributions'] = factor_contribution
                result['unexplained_return'] = portfolio_returns.mean() - sum(factor_contribution.values())
            
            return result
            
        finally:
            # Restore original data
            self.returns_data = original_data


def main():
    """Example usage of MultiFactorModels."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    print("=== Multi-Factor Risk Models Example ===")
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM', 'V']
    
    # Get price data and convert to required format
    df_list = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        import yfinance as yf
        stock_data = yf.download(ticker, period="2y", progress=False)
        
        for date, row in stock_data.iterrows():
            df_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'tic': ticker,
                'close': row['Close']
            })
    
    df = pd.DataFrame(df_list)
    
    # Initialize multi-factor models
    mf_models = MultiFactorModels(df)
    
    print(f"\nEstimating multi-factor models...")
    
    # Test asset
    asset = 'AAPL'
    
    # 1. Fama-French 3-factor model
    print(f"\n1. Fama-French Model for {asset}:")
    ff_result = mf_models.estimate_fama_french_model(asset, include_momentum=False)
    print(f"Market Beta: {ff_result.get('market_beta', 'N/A'):.4f}")
    print(f"SMB Beta: {ff_result.get('smb_beta', 'N/A'):.4f}")
    print(f"HML Beta: {ff_result.get('hml_beta', 'N/A'):.4f}")
    print(f"Alpha: {ff_result.get('alpha', 'N/A'):.6f}")
    print(f"R-squared: {ff_result.get('r_squared', 'N/A'):.4f}")
    
    # 2. Carhart 4-factor model
    print(f"\n2. Carhart 4-Factor Model for {asset}:")
    carhart_result = mf_models.estimate_fama_french_model(asset, include_momentum=True)
    print(f"Market Beta: {carhart_result.get('market_beta', 'N/A'):.4f}")
    print(f"SMB Beta: {carhart_result.get('smb_beta', 'N/A'):.4f}")
    print(f"HML Beta: {carhart_result.get('hml_beta', 'N/A'):.4f}")
    print(f"Momentum Beta: {carhart_result.get('mom_beta', 'N/A'):.4f}")
    print(f"R-squared: {carhart_result.get('r_squared', 'N/A'):.4f}")
    
    # 3. PCA Factor Model
    print(f"\n3. PCA Factor Model for {asset}:")
    pca_result = mf_models.estimate_pca_factor_model(asset, n_factors=3)
    print(f"PC1 Loading: {pca_result.get('pc1_loading', 'N/A'):.4f}")
    print(f"PC2 Loading: {pca_result.get('pc2_loading', 'N/A'):.4f}")
    print(f"PC3 Loading: {pca_result.get('pc3_loading', 'N/A'):.4f}")
    print(f"Total Variance Explained: {pca_result.get('total_variance_explained', 'N/A'):.2%}")
    print(f"R-squared: {pca_result.get('r_squared', 'N/A'):.4f}")
    
    # 4. Portfolio Factor Model
    print(f"\n4. Portfolio Factor Analysis:")
    portfolio_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    portfolio_result = mf_models.estimate_factor_portfolio_model(
        portfolio_assets, 
        model_type='fama_french',
        include_momentum=False
    )
    
    print(f"Portfolio Factor Loadings:")
    for factor, loading in portfolio_result['portfolio_loadings'].items():
        if not np.isnan(loading):
            print(f"  {factor}: {loading:.4f}")
    
    # 5. Factor Exposure Analysis
    print(f"\n5. Factor Exposure Analysis:")
    exposure_df = mf_models.analyze_factor_exposures(
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 
        model_type='fama_french'
    )
    
    if not exposure_df.empty:
        print("Factor exposures across assets:")
        display_cols = ['asset', 'market_beta', 'smb_beta', 'hml_beta', 'r_squared']
        available_cols = [col for col in display_cols if col in exposure_df.columns]
        print(exposure_df[available_cols].round(4).to_string(index=False))
    
    # 6. Rolling Factor Model (sample)
    print(f"\n6. Rolling Factor Model Analysis:")
    rolling_result = mf_models.estimate_fama_french_model(
        asset, 
        window=120, 
        rolling=True, 
        include_momentum=False
    )
    
    if isinstance(rolling_result, pd.DataFrame) and not rolling_result.empty:
        print(f"Rolling factor statistics for {asset}:")
        for col in ['market_beta', 'smb_beta', 'hml_beta']:
            if col in rolling_result.columns:
                col_stats = rolling_result[col].dropna()
                if len(col_stats) > 0:
                    print(f"  {col}: Mean={col_stats.mean():.4f}, Std={col_stats.std():.4f}")
    
    # Save results
    output_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/multi_factor_analysis.csv'
    
    # Create comprehensive results
    all_results = []
    for asset in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
        ff_res = mf_models.estimate_fama_french_model(asset, include_momentum=False)
        pca_res = mf_models.estimate_pca_factor_model(asset, n_factors=3)
        
        result_row = {
            'asset': asset,
            'ff_market_beta': ff_res.get('market_beta', np.nan),
            'ff_smb_beta': ff_res.get('smb_beta', np.nan),
            'ff_hml_beta': ff_res.get('hml_beta', np.nan),
            'ff_alpha': ff_res.get('alpha', np.nan),
            'ff_r_squared': ff_res.get('r_squared', np.nan),
            'pca_pc1_loading': pca_res.get('pc1_loading', np.nan),
            'pca_pc2_loading': pca_res.get('pc2_loading', np.nan),
            'pca_pc3_loading': pca_res.get('pc3_loading', np.nan),
            'pca_r_squared': pca_res.get('r_squared', np.nan)
        }
        all_results.append(result_row)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Multi-factor analysis completed for {len(all_results)} assets")


if __name__ == "__main__":
    main()