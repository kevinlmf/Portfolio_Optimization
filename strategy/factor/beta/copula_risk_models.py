"""
Copula-Based Risk Models for Beta Estimation
Advanced copula methods for modeling dependence structure and systematic risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy import stats, optimize
from scipy.stats import norm, t as student_t
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class CopulaRiskModels:
    """
    Copula-based risk model implementations for beta estimation.
    
    This class provides various copula models for capturing non-linear
    dependence structures and tail dependencies in risk estimation.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 market_index: str = 'SPY',
                 risk_free_rate: float = 0.02):
        """
        Initialize with return data.
        
        Args:
            data: DataFrame with asset returns
            market_index: Market index symbol
            risk_free_rate: Annual risk-free rate
        """
        self.data = data.copy()
        self.market_index = market_index
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252
        
        # Prepare data
        self.returns_data = self._prepare_returns_data()
        self.fitted_marginals = {}
        self.fitted_copulas = {}
        
    def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare returns data."""
        if 'close' in self.data.columns:
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
    
    def fit_marginal_distributions(self, 
                                 assets: List[str], 
                                 distribution: str = 'skew_t') -> Dict:
        """
        Fit marginal distributions to individual asset returns.
        
        Args:
            assets: List of asset symbols
            distribution: 'normal', 't', 'skew_t', 'kde'
            
        Returns:
            Dictionary of fitted marginal parameters
        """
        marginal_params = {}
        
        for asset in assets:
            if asset not in self.returns_data.columns:
                marginal_params[asset] = None
                continue
            
            returns = self.returns_data[asset].dropna()
            
            if len(returns) < 50:
                marginal_params[asset] = None
                continue
            
            try:
                if distribution == 'normal':
                    params = self._fit_normal(returns)
                elif distribution == 't':
                    params = self._fit_t_distribution(returns)
                elif distribution == 'skew_t':
                    params = self._fit_skew_t_distribution(returns)
                elif distribution == 'kde' and SCIPY_AVAILABLE:
                    params = self._fit_kde(returns)
                else:
                    params = self._fit_normal(returns)  # Fallback
                
                params['distribution'] = distribution
                marginal_params[asset] = params
                
            except Exception as e:
                print(f"Error fitting {distribution} to {asset}: {e}")
                # Fallback to normal distribution
                params = self._fit_normal(returns)
                params['distribution'] = 'normal'
                marginal_params[asset] = params
        
        self.fitted_marginals = marginal_params
        return marginal_params
    
    def _fit_normal(self, returns: pd.Series) -> Dict:
        """Fit normal distribution."""
        return {
            'mean': returns.mean(),
            'std': returns.std()
        }
    
    def _fit_t_distribution(self, returns: pd.Series) -> Dict:
        """Fit Student's t-distribution."""
        try:
            # Standardize returns
            standardized = (returns - returns.mean()) / returns.std()
            
            # Fit t-distribution
            df, loc, scale = student_t.fit(standardized)
            
            # Transform back to original scale
            return {
                'df': df,
                'mean': returns.mean() + loc * returns.std(),
                'std': scale * returns.std()
            }
        except:
            return self._fit_normal(returns)
    
    def _fit_skew_t_distribution(self, returns: pd.Series) -> Dict:
        """Fit skewed t-distribution (simplified implementation)."""
        try:
            # Use basic skew-t approximation
            mean = returns.mean()
            std = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Estimate degrees of freedom from kurtosis
            if kurtosis > 0:
                df = 6 / kurtosis + 4
            else:
                df = 10
            
            return {
                'mean': mean,
                'std': std,
                'skewness': skewness,
                'df': df
            }
        except:
            return self._fit_normal(returns)
    
    def _fit_kde(self, returns: pd.Series) -> Dict:
        """Fit kernel density estimation."""
        try:
            kde = gaussian_kde(returns.values)
            return {
                'kde': kde,
                'mean': returns.mean(),
                'std': returns.std()
            }
        except:
            return self._fit_normal(returns)
    
    def transform_to_uniform(self, 
                           returns: pd.Series, 
                           marginal_params: Dict) -> pd.Series:
        """Transform returns to uniform marginals using fitted distribution."""
        if marginal_params is None:
            return pd.Series(np.nan, index=returns.index)
        
        dist_type = marginal_params.get('distribution', 'normal')
        
        try:
            if dist_type == 'normal':
                u = norm.cdf(returns, 
                           loc=marginal_params['mean'], 
                           scale=marginal_params['std'])
            
            elif dist_type == 't':
                standardized = (returns - marginal_params['mean']) / marginal_params['std']
                u = student_t.cdf(standardized, df=marginal_params['df'])
            
            elif dist_type == 'skew_t':
                # Simplified skew-t transformation
                standardized = (returns - marginal_params['mean']) / marginal_params['std']
                u = student_t.cdf(standardized, df=marginal_params.get('df', 10))
                
                # Adjust for skewness (simplified)
                skew = marginal_params.get('skewness', 0)
                if abs(skew) > 0.1:
                    u = np.where(returns > marginal_params['mean'], 
                                u**0.5 if skew > 0 else u**2,
                                u**2 if skew > 0 else u**0.5)
            
            elif dist_type == 'kde' and 'kde' in marginal_params:
                # Empirical CDF approximation
                sorted_returns = np.sort(returns.values)
                kde = marginal_params['kde']
                
                u = np.zeros_like(returns)
                for i, val in enumerate(returns):
                    u[i] = np.mean(sorted_returns <= val)
            
            else:
                u = norm.cdf(returns, 
                           loc=marginal_params['mean'], 
                           scale=marginal_params['std'])
            
            # Ensure values are in (0,1)
            u = np.clip(u, 1e-8, 1-1e-8)
            return pd.Series(u, index=returns.index)
            
        except Exception as e:
            print(f"Error in uniform transformation: {e}")
            # Fallback to empirical CDF
            ranks = returns.rank() / (len(returns) + 1)
            return ranks.clip(1e-8, 1-1e-8)
    
    def fit_gaussian_copula(self, 
                          assets: List[str], 
                          window: int = 252) -> Dict:
        """
        Fit Gaussian copula to asset returns.
        
        Args:
            assets: List of asset symbols
            window: Estimation window
            
        Returns:
            Gaussian copula parameters
        """
        # Check if assets exist
        available_assets = [a for a in assets if a in self.returns_data.columns]
        
        if len(available_assets) < 2:
            return {'correlation_matrix': np.nan, 'assets': available_assets}
        
        # Get returns data
        asset_returns = self.returns_data[available_assets].dropna()
        
        if len(asset_returns) < window:
            recent_returns = asset_returns
        else:
            recent_returns = asset_returns.tail(window)
        
        if len(recent_returns) < 30:
            return {'correlation_matrix': np.nan, 'assets': available_assets}
        
        try:
            # Fit marginal distributions if not already fitted
            if not self.fitted_marginals:
                self.fit_marginal_distributions(available_assets)
            
            # Transform to uniform marginals
            uniform_data = pd.DataFrame(index=recent_returns.index)
            
            for asset in available_assets:
                if asset in self.fitted_marginals and self.fitted_marginals[asset] is not None:
                    uniform_data[asset] = self.transform_to_uniform(
                        recent_returns[asset], 
                        self.fitted_marginals[asset]
                    )
                else:
                    # Fallback to empirical CDF
                    ranks = recent_returns[asset].rank() / (len(recent_returns) + 1)
                    uniform_data[asset] = ranks.clip(1e-8, 1-1e-8)
            
            # Transform to normal variables
            normal_data = uniform_data.apply(lambda x: norm.ppf(x))
            
            # Estimate correlation matrix
            correlation_matrix = normal_data.corr().values
            
            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Regularize small eigenvalues
            correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Normalize to correlation matrix
            diag_sqrt = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
            
            copula_params = {
                'correlation_matrix': correlation_matrix,
                'assets': available_assets,
                'copula_type': 'gaussian',
                'log_likelihood': self._gaussian_copula_log_likelihood(normal_data, correlation_matrix)
            }
            
            self.fitted_copulas[tuple(available_assets)] = copula_params
            return copula_params
            
        except Exception as e:
            print(f"Error fitting Gaussian copula: {e}")
            return {'correlation_matrix': np.nan, 'assets': available_assets}
    
    def fit_t_copula(self, 
                   assets: List[str], 
                   window: int = 252) -> Dict:
        """
        Fit t-copula to asset returns.
        
        Args:
            assets: List of asset symbols
            window: Estimation window
            
        Returns:
            t-copula parameters
        """
        # First fit Gaussian copula to get correlation matrix
        gaussian_params = self.fit_gaussian_copula(assets, window)
        
        if isinstance(gaussian_params.get('correlation_matrix'), np.ndarray):
            correlation_matrix = gaussian_params['correlation_matrix']
            available_assets = gaussian_params['assets']
        else:
            return {'correlation_matrix': np.nan, 'df': np.nan, 'assets': assets}
        
        # Get uniform data
        asset_returns = self.returns_data[available_assets].dropna()
        
        if len(asset_returns) < window:
            recent_returns = asset_returns
        else:
            recent_returns = asset_returns.tail(window)
        
        try:
            # Transform to uniform marginals
            uniform_data = pd.DataFrame(index=recent_returns.index)
            
            for asset in available_assets:
                if asset in self.fitted_marginals and self.fitted_marginals[asset] is not None:
                    uniform_data[asset] = self.transform_to_uniform(
                        recent_returns[asset], 
                        self.fitted_marginals[asset]
                    )
                else:
                    ranks = recent_returns[asset].rank() / (len(recent_returns) + 1)
                    uniform_data[asset] = ranks.clip(1e-8, 1-1e-8)
            
            # Estimate degrees of freedom
            df = self._estimate_t_copula_df(uniform_data, correlation_matrix)
            
            copula_params = {
                'correlation_matrix': correlation_matrix,
                'df': df,
                'assets': available_assets,
                'copula_type': 't',
                'log_likelihood': self._t_copula_log_likelihood(uniform_data, correlation_matrix, df)
            }
            
            return copula_params
            
        except Exception as e:
            print(f"Error fitting t-copula: {e}")
            return {'correlation_matrix': np.nan, 'df': np.nan, 'assets': available_assets}
    
    def _estimate_t_copula_df(self, 
                            uniform_data: pd.DataFrame, 
                            correlation_matrix: np.ndarray) -> float:
        """Estimate degrees of freedom for t-copula."""
        try:
            # Method of moments estimation
            # Transform to t-variables assuming high df initially
            t_data = uniform_data.apply(lambda x: student_t.ppf(x, df=30))
            
            # Calculate sample Kendall's tau
            tau_sample = self._calculate_kendall_tau_matrix(uniform_data)
            
            # For t-copula: tau = (2/π) * arcsin(ρ)
            # where ρ is the correlation parameter
            
            # Use optimization to find best df
            def objective(df):
                if df <= 2:
                    return 1e6
                
                # Theoretical tau for t-copula
                tau_theoretical = (2/np.pi) * np.arcsin(correlation_matrix)
                
                # Adjust for degrees of freedom effect (simplified)
                adjustment = 1 - 2/(df + 2)
                tau_adjusted = tau_theoretical * adjustment
                
                return np.sum((tau_sample - tau_adjusted)**2)
            
            result = optimize.minimize_scalar(objective, bounds=(2.1, 50), method='bounded')
            
            return max(2.1, min(50, result.x))
            
        except:
            return 10.0  # Default value
    
    def _calculate_kendall_tau_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Kendall's tau correlation matrix."""
        n_assets = len(data.columns)
        tau_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    tau_matrix[i, j] = 1.0
                else:
                    x = data.iloc[:, i]
                    y = data.iloc[:, j]
                    tau, _ = stats.kendalltau(x, y)
                    tau_matrix[i, j] = tau if not np.isnan(tau) else 0.0
        
        return tau_matrix
    
    def _gaussian_copula_log_likelihood(self, 
                                     normal_data: pd.DataFrame, 
                                     correlation_matrix: np.ndarray) -> float:
        """Calculate log-likelihood for Gaussian copula."""
        try:
            n_obs, n_dim = normal_data.shape
            
            # Remove invalid observations
            valid_mask = ~(normal_data.isnull().any(axis=1) | 
                          np.isinf(normal_data).any(axis=1))
            clean_data = normal_data[valid_mask]
            
            if len(clean_data) == 0:
                return -np.inf
            
            # Log-likelihood calculation
            inv_corr = np.linalg.inv(correlation_matrix)
            log_det = np.linalg.slogdet(correlation_matrix)[1]
            
            log_likelihood = 0
            for _, row in clean_data.iterrows():
                z = row.values
                log_likelihood += -0.5 * (z.T @ inv_corr @ z - z.T @ z) - 0.5 * log_det
            
            return log_likelihood
            
        except:
            return -np.inf
    
    def _t_copula_log_likelihood(self, 
                               uniform_data: pd.DataFrame, 
                               correlation_matrix: np.ndarray, 
                               df: float) -> float:
        """Calculate log-likelihood for t-copula."""
        try:
            n_obs, n_dim = uniform_data.shape
            
            # Transform to t-variables
            t_data = uniform_data.apply(lambda x: student_t.ppf(x, df=df))
            
            # Remove invalid observations
            valid_mask = ~(t_data.isnull().any(axis=1) | 
                          np.isinf(t_data).any(axis=1))
            clean_data = t_data[valid_mask]
            
            if len(clean_data) == 0:
                return -np.inf
            
            # Log-likelihood calculation for t-copula
            inv_corr = np.linalg.inv(correlation_matrix)
            log_det = np.linalg.slogdet(correlation_matrix)[1]
            
            log_likelihood = 0
            for _, row in clean_data.iterrows():
                t_vec = row.values
                
                # Multivariate t density contribution
                quad_form = t_vec.T @ inv_corr @ t_vec
                individual_quad = np.sum(t_vec**2)
                
                log_likelihood += (stats.loggamma((df + n_dim)/2) - 
                                  stats.loggamma(df/2) - 
                                  0.5 * log_det -
                                  (n_dim/2) * np.log(np.pi * df) +
                                  (df + n_dim)/2 * np.log(1 + quad_form/df) -
                                  sum(stats.loggamma((df + 1)/2) - 
                                      stats.loggamma(df/2) - 
                                      0.5 * np.log(np.pi * df) +
                                      (df + 1)/2 * np.log(1 + t**2/df) 
                                      for t in t_vec))
            
            return log_likelihood
            
        except:
            return -np.inf
    
    def estimate_copula_beta(self, 
                           asset: str, 
                           copula_type: str = 'gaussian',
                           tail_dependence: bool = True,
                           window: int = 252) -> Dict:
        """
        Estimate copula-based beta.
        
        Args:
            asset: Asset symbol
            copula_type: 'gaussian' or 't'
            tail_dependence: Whether to calculate tail dependence coefficients
            window: Estimation window
            
        Returns:
            Copula-based beta estimates
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return {'copula_beta': np.nan, 'linear_beta': np.nan}
        
        assets = [asset, self.market_index]
        
        # Fit marginal distributions
        self.fit_marginal_distributions(assets)
        
        # Fit copula
        if copula_type == 'gaussian':
            copula_params = self.fit_gaussian_copula(assets, window)
        else:
            copula_params = self.fit_t_copula(assets, window)
        
        if isinstance(copula_params.get('correlation_matrix'), np.ndarray):
            # Extract correlation between asset and market
            correlation = copula_params['correlation_matrix'][0, 1]
            
            # Traditional linear beta (for comparison)
            asset_returns = self.returns_data[asset].dropna()
            market_returns = self.returns_data[self.market_index].dropna()
            common_dates = asset_returns.index.intersection(market_returns.index)
            
            if len(common_dates) >= window:
                asset_aligned = asset_returns[common_dates].tail(window)
                market_aligned = market_returns[common_dates].tail(window)
                
                # Excess returns
                asset_excess = asset_aligned - self.daily_rf_rate
                market_excess = market_aligned - self.daily_rf_rate
                
                linear_beta = np.cov(asset_excess, market_excess)[0, 1] / np.var(market_excess)
                
                # Copula beta (correlation-based)
                asset_vol = asset_excess.std()
                market_vol = market_excess.std()
                copula_beta = correlation * (asset_vol / market_vol)
                
                result = {
                    'copula_beta': copula_beta,
                    'linear_beta': linear_beta,
                    'correlation': correlation,
                    'copula_type': copula_type,
                    'asset_vol': asset_vol,
                    'market_vol': market_vol
                }
                
                # Add tail dependence if requested
                if tail_dependence:
                    tail_deps = self._calculate_tail_dependence(asset, self.market_index, window)
                    result.update(tail_deps)
                
                # Add degrees of freedom if t-copula
                if copula_type == 't' and 'df' in copula_params:
                    result['df'] = copula_params['df']
                
                return result
        
        return {'copula_beta': np.nan, 'linear_beta': np.nan}
    
    def _calculate_tail_dependence(self, 
                                 asset: str, 
                                 market: str, 
                                 window: int = 252) -> Dict:
        """Calculate upper and lower tail dependence coefficients."""
        try:
            asset_returns = self.returns_data[asset].dropna()
            market_returns = self.returns_data[market].dropna()
            
            # Align data
            common_dates = asset_returns.index.intersection(market_returns.index)
            asset_aligned = asset_returns[common_dates].tail(window)
            market_aligned = market_returns[common_dates].tail(window)
            
            # Transform to uniform marginals
            if asset in self.fitted_marginals and self.fitted_marginals[asset] is not None:
                u_asset = self.transform_to_uniform(asset_aligned, self.fitted_marginals[asset])
            else:
                u_asset = asset_aligned.rank() / (len(asset_aligned) + 1)
            
            if market in self.fitted_marginals and self.fitted_marginals[market] is not None:
                u_market = self.transform_to_uniform(market_aligned, self.fitted_marginals[market])
            else:
                u_market = market_aligned.rank() / (len(market_aligned) + 1)
            
            # Calculate tail dependence coefficients
            thresholds = [0.9, 0.95, 0.99]
            upper_tail_deps = []
            lower_tail_deps = []
            
            for threshold in thresholds:
                # Upper tail dependence
                upper_mask = u_market > threshold
                if upper_mask.sum() > 0:
                    upper_tail_dep = np.mean(u_asset[upper_mask] > threshold)
                    upper_tail_deps.append(upper_tail_dep)
                
                # Lower tail dependence
                lower_threshold = 1 - threshold
                lower_mask = u_market < lower_threshold
                if lower_mask.sum() > 0:
                    lower_tail_dep = np.mean(u_asset[lower_mask] < lower_threshold)
                    lower_tail_deps.append(lower_tail_dep)
            
            return {
                'upper_tail_dependence_90': upper_tail_deps[0] if upper_tail_deps else np.nan,
                'upper_tail_dependence_95': upper_tail_deps[1] if len(upper_tail_deps) > 1 else np.nan,
                'upper_tail_dependence_99': upper_tail_deps[2] if len(upper_tail_deps) > 2 else np.nan,
                'lower_tail_dependence_90': lower_tail_deps[0] if lower_tail_deps else np.nan,
                'lower_tail_dependence_95': lower_tail_deps[1] if len(lower_tail_deps) > 1 else np.nan,
                'lower_tail_dependence_99': lower_tail_deps[2] if len(lower_tail_deps) > 2 else np.nan
            }
            
        except Exception as e:
            print(f"Error calculating tail dependence: {e}")
            return {
                'upper_tail_dependence_90': np.nan,
                'upper_tail_dependence_95': np.nan,
                'upper_tail_dependence_99': np.nan,
                'lower_tail_dependence_90': np.nan,
                'lower_tail_dependence_95': np.nan,
                'lower_tail_dependence_99': np.nan
            }
    
    def estimate_conditional_copula_beta(self, 
                                       asset: str, 
                                       conditioning_vars: List[str],
                                       copula_type: str = 'gaussian',
                                       window: int = 252) -> Dict:
        """
        Estimate conditional copula beta given conditioning variables.
        
        Args:
            asset: Asset symbol
            conditioning_vars: List of conditioning variable names
            copula_type: Type of copula
            window: Estimation window
            
        Returns:
            Conditional copula beta estimates
        """
        all_assets = [asset, self.market_index] + conditioning_vars
        available_assets = [a for a in all_assets if a in self.returns_data.columns]
        
        if len(available_assets) < 3:  # Need at least asset, market, and one conditioning var
            return {'conditional_copula_beta': np.nan}
        
        # Fit marginal distributions
        self.fit_marginal_distributions(available_assets)
        
        # Fit joint copula
        if copula_type == 'gaussian':
            copula_params = self.fit_gaussian_copula(available_assets, window)
        else:
            copula_params = self.fit_t_copula(available_assets, window)
        
        if not isinstance(copula_params.get('correlation_matrix'), np.ndarray):
            return {'conditional_copula_beta': np.nan}
        
        try:
            # Extract correlation matrix
            corr_matrix = copula_params['correlation_matrix']
            
            # Asset is index 0, market is index 1, conditioning vars are 2+
            asset_idx = 0
            market_idx = 1
            cond_indices = list(range(2, len(available_assets)))
            
            if len(cond_indices) == 0:
                # No conditioning variables available
                return {'conditional_copula_beta': corr_matrix[asset_idx, market_idx]}
            
            # Calculate conditional correlation using partial correlation formula
            # ρ(X,Y|Z) = (ρ(X,Y) - ρ(X,Z)ρ(Y,Z)) / sqrt((1-ρ(X,Z)²)(1-ρ(Y,Z)²))
            
            # For multiple conditioning variables, use matrix approach
            sigma_12 = corr_matrix[asset_idx, market_idx]
            sigma_13 = corr_matrix[asset_idx, cond_indices]
            sigma_23 = corr_matrix[market_idx, cond_indices]
            sigma_33 = corr_matrix[np.ix_(cond_indices, cond_indices)]
            
            # Conditional correlation
            if len(cond_indices) == 1:
                # Simple case with one conditioning variable
                sigma_13_scalar = sigma_13[0]
                sigma_23_scalar = sigma_23[0]
                sigma_33_scalar = sigma_33[0, 0]
                
                conditional_corr = ((sigma_12 - sigma_13_scalar * sigma_23_scalar / sigma_33_scalar) /
                                   np.sqrt((1 - sigma_13_scalar**2 / sigma_33_scalar) * 
                                          (1 - sigma_23_scalar**2 / sigma_33_scalar)))
            else:
                # Multiple conditioning variables - use matrix inverse
                inv_sigma_33 = np.linalg.inv(sigma_33)
                
                numerator = sigma_12 - sigma_13 @ inv_sigma_33 @ sigma_23
                denominator = np.sqrt((1 - sigma_13 @ inv_sigma_33 @ sigma_13) * 
                                     (1 - sigma_23 @ inv_sigma_33 @ sigma_23))
                
                conditional_corr = numerator / denominator
            
            # Convert to conditional beta
            asset_returns = self.returns_data[asset].dropna().tail(window)
            market_returns = self.returns_data[self.market_index].dropna().tail(window)
            
            asset_vol = asset_returns.std()
            market_vol = market_returns.std()
            
            conditional_beta = conditional_corr * (asset_vol / market_vol)
            
            return {
                'conditional_copula_beta': conditional_beta,
                'conditional_correlation': conditional_corr,
                'unconditional_correlation': sigma_12,
                'conditioning_variables': [available_assets[i] for i in cond_indices]
            }
            
        except Exception as e:
            print(f"Error calculating conditional copula beta: {e}")
            return {'conditional_copula_beta': np.nan}
    
    def compare_copula_models(self, 
                            assets: List[str], 
                            window: int = 252) -> pd.DataFrame:
        """
        Compare different copula models using information criteria.
        
        Args:
            assets: List of asset symbols
            window: Estimation window
            
        Returns:
            DataFrame with model comparison results
        """
        results = []
        
        # Fit different copula models
        models_to_fit = ['gaussian', 't']
        
        for copula_type in models_to_fit:
            try:
                if copula_type == 'gaussian':
                    params = self.fit_gaussian_copula(assets, window)
                else:
                    params = self.fit_t_copula(assets, window)
                
                if isinstance(params.get('correlation_matrix'), np.ndarray):
                    log_likelihood = params.get('log_likelihood', np.nan)
                    n_params = len(assets) * (len(assets) - 1) / 2  # Correlation matrix parameters
                    
                    if copula_type == 't':
                        n_params += 1  # Degrees of freedom parameter
                    
                    n_obs = window
                    
                    # Information criteria
                    aic = -2 * log_likelihood + 2 * n_params
                    bic = -2 * log_likelihood + n_params * np.log(n_obs)
                    
                    result = {
                        'copula_type': copula_type,
                        'log_likelihood': log_likelihood,
                        'aic': aic,
                        'bic': bic,
                        'n_parameters': n_params
                    }
                    
                    if copula_type == 't':
                        result['degrees_of_freedom'] = params.get('df', np.nan)
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Error fitting {copula_type} copula: {e}")
        
        if results:
            results_df = pd.DataFrame(results)
            # Add rankings
            results_df['aic_rank'] = results_df['aic'].rank()
            results_df['bic_rank'] = results_df['bic'].rank()
            
            return results_df.sort_values('aic')
        else:
            return pd.DataFrame()


def main():
    """Example usage of CopulaRiskModels."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    print("=== Copula Risk Models Example ===")
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']  # Include market index
    
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
    
    # Initialize copula models
    copula_models = CopulaRiskModels(df, market_index='SPY')
    
    print(f"\nFitting copula-based risk models...")
    
    # Test asset
    asset = 'AAPL'
    
    # 1. Fit marginal distributions
    print(f"\n1. Fitting marginal distributions...")
    marginals = copula_models.fit_marginal_distributions(['AAPL', 'MSFT', 'SPY'], 'skew_t')
    
    for asset_name, params in marginals.items():
        if params is not None:
            print(f"{asset_name}: {params['distribution']} - mean={params.get('mean', 0):.6f}, std={params.get('std', 0):.6f}")
    
    # 2. Gaussian Copula Beta
    print(f"\n2. Gaussian Copula Beta for {asset}:")
    gaussian_beta = copula_models.estimate_copula_beta(asset, 'gaussian', tail_dependence=True)
    
    print(f"Copula Beta: {gaussian_beta.get('copula_beta', 'N/A'):.4f}")
    print(f"Linear Beta: {gaussian_beta.get('linear_beta', 'N/A'):.4f}")
    print(f"Correlation: {gaussian_beta.get('correlation', 'N/A'):.4f}")
    print(f"Upper Tail Dependence (95%): {gaussian_beta.get('upper_tail_dependence_95', 'N/A'):.4f}")
    print(f"Lower Tail Dependence (95%): {gaussian_beta.get('lower_tail_dependence_95', 'N/A'):.4f}")
    
    # 3. t-Copula Beta
    print(f"\n3. t-Copula Beta for {asset}:")
    t_beta = copula_models.estimate_copula_beta(asset, 't', tail_dependence=True)
    
    print(f"Copula Beta: {t_beta.get('copula_beta', 'N/A'):.4f}")
    print(f"Degrees of Freedom: {t_beta.get('df', 'N/A'):.2f}")
    print(f"Upper Tail Dependence (95%): {t_beta.get('upper_tail_dependence_95', 'N/A'):.4f}")
    print(f"Lower Tail Dependence (95%): {t_beta.get('lower_tail_dependence_95', 'N/A'):.4f}")
    
    # 4. Conditional Copula Beta
    print(f"\n4. Conditional Copula Beta:")
    conditional_beta = copula_models.estimate_conditional_copula_beta(
        asset, 
        ['MSFT', 'GOOGL'], 
        'gaussian'
    )
    
    print(f"Conditional Beta: {conditional_beta.get('conditional_copula_beta', 'N/A'):.4f}")
    print(f"Conditional Correlation: {conditional_beta.get('conditional_correlation', 'N/A'):.4f}")
    print(f"Unconditional Correlation: {conditional_beta.get('unconditional_correlation', 'N/A'):.4f}")
    
    # 5. Model Comparison
    print(f"\n5. Copula Model Comparison:")
    comparison_df = copula_models.compare_copula_models(['AAPL', 'SPY'])
    
    if not comparison_df.empty:
        print("Model comparison results:")
        print(comparison_df[['copula_type', 'log_likelihood', 'aic', 'bic']].round(4).to_string(index=False))
    
    # 6. Multiple Asset Analysis
    print(f"\n6. Multiple Asset Copula Analysis:")
    test_assets = ['AAPL', 'MSFT', 'GOOGL']
    all_results = []
    
    for test_asset in test_assets:
        gauss_result = copula_models.estimate_copula_beta(test_asset, 'gaussian')
        t_result = copula_models.estimate_copula_beta(test_asset, 't')
        
        all_results.append({
            'asset': test_asset,
            'gaussian_beta': gauss_result.get('copula_beta', np.nan),
            'gaussian_correlation': gauss_result.get('correlation', np.nan),
            't_beta': t_result.get('copula_beta', np.nan),
            't_correlation': t_result.get('correlation', np.nan),
            't_df': t_result.get('df', np.nan)
        })
    
    results_df = pd.DataFrame(all_results)
    print("Multi-asset copula beta estimates:")
    print(results_df.round(4).to_string(index=False))
    
    # Save results
    output_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/copula_risk_analysis.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Copula risk analysis completed for {len(test_assets)} assets")


if __name__ == "__main__":
    main()