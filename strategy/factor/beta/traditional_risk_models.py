"""
Traditional Risk Models for Beta Estimation
Classical approaches to estimating systematic risk (beta) for portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class TraditionalRiskModels:
    """
    Traditional risk model implementations for beta estimation.
    
    This class provides various classical methods for estimating systematic risk
    including CAPM beta, multi-period beta, conditional beta, and robust estimators.
    """
    
    def __init__(self, data: pd.DataFrame, market_index: str = 'SPY', risk_free_rate: float = 0.02):
        """
        Initialize with return data.
        
        Args:
            data: DataFrame with columns ['date', 'tic', 'close'] or returns
            market_index: Market index symbol for beta calculation
            risk_free_rate: Annual risk-free rate
        """
        self.data = data.copy()
        self.market_index = market_index
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252
        
        # Prepare return data
        self.returns_data = self._prepare_returns_data()
        
    def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare returns data from price data."""
        if 'close' in self.data.columns:
            # Convert price data to returns
            returns_list = []
            
            for tic in self.data['tic'].unique():
                tic_data = self.data[self.data['tic'] == tic].copy()
                tic_data = tic_data.sort_values('date')
                tic_data['returns'] = tic_data['close'].pct_change()
                returns_list.append(tic_data[['date', 'tic', 'returns']])
            
            returns_df = pd.concat(returns_list, ignore_index=True)
            
            # Pivot to wide format
            returns_wide = returns_df.pivot(index='date', columns='tic', values='returns')
            returns_wide.index = pd.to_datetime(returns_wide.index)
            
            return returns_wide
        else:
            # Assume data is already in returns format
            return self.data.copy()
    
    def calculate_capm_beta(self, 
                           asset: str, 
                           window: int = 252,
                           rolling: bool = False) -> Union[float, pd.Series]:
        """
        Calculate CAPM beta using ordinary least squares.
        
        Args:
            asset: Asset symbol
            window: Estimation window in days
            rolling: Whether to calculate rolling beta
            
        Returns:
            Beta estimate or time series of beta estimates
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return np.nan
        
        asset_returns = self.returns_data[asset].dropna()
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        if len(asset_returns) < window:
            return np.nan
        
        # Excess returns
        asset_excess = asset_returns - self.daily_rf_rate
        market_excess = market_returns - self.daily_rf_rate
        
        if rolling:
            # Rolling beta calculation
            beta_series = []
            dates = []
            
            for i in range(window, len(asset_excess)):
                window_asset = asset_excess.iloc[i-window:i]
                window_market = market_excess.iloc[i-window:i]
                
                if len(window_asset) == window and len(window_market) == window:
                    # Calculate beta using covariance method
                    covariance = np.cov(window_asset, window_market)[0, 1]
                    market_variance = np.var(window_market)
                    
                    if market_variance > 0:
                        beta = covariance / market_variance
                    else:
                        beta = np.nan
                    
                    beta_series.append(beta)
                    dates.append(asset_excess.index[i])
                else:
                    beta_series.append(np.nan)
                    dates.append(asset_excess.index[i])
            
            return pd.Series(beta_series, index=dates, name=f'{asset}_beta')
        else:
            # Static beta calculation
            if len(asset_excess) >= window:
                # Use most recent window
                recent_asset = asset_excess.tail(window)
                recent_market = market_excess.tail(window)
                
                covariance = np.cov(recent_asset, recent_market)[0, 1]
                market_variance = np.var(recent_market)
                
                if market_variance > 0:
                    return covariance / market_variance
            
            return np.nan
    
    def calculate_regression_beta(self, 
                                asset: str, 
                                window: int = 252,
                                method: str = 'ols',
                                rolling: bool = False) -> Union[Dict, pd.DataFrame]:
        """
        Calculate beta using regression methods.
        
        Args:
            asset: Asset symbol
            window: Estimation window
            method: 'ols', 'ridge', 'lasso', 'huber'
            rolling: Whether to calculate rolling estimates
            
        Returns:
            Regression results including beta, alpha, R-squared
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}
        
        asset_returns = self.returns_data[asset].dropna()
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        if len(asset_returns) < window:
            return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}
        
        # Excess returns
        y = asset_returns - self.daily_rf_rate
        X = market_returns - self.daily_rf_rate
        
        # Choose regression method
        if method == 'ols':
            regressor = LinearRegression()
        elif method == 'ridge':
            regressor = Ridge(alpha=0.01)
        elif method == 'lasso':
            regressor = Lasso(alpha=0.001, max_iter=1000)
        elif method == 'huber':
            regressor = HuberRegressor()
        else:
            regressor = LinearRegression()
        
        if rolling:
            # Rolling regression
            results = []
            
            for i in range(window, len(y)):
                window_y = y.iloc[i-window:i]
                window_X = X.iloc[i-window:i].values.reshape(-1, 1)
                
                try:
                    regressor.fit(window_X, window_y)
                    
                    beta = regressor.coef_[0]
                    alpha = regressor.intercept_
                    
                    # R-squared
                    y_pred = regressor.predict(window_X)
                    r_squared = r2_score(window_y, y_pred)
                    
                    results.append({
                        'date': y.index[i],
                        'beta': beta,
                        'alpha': alpha,
                        'r_squared': r_squared
                    })
                except:
                    results.append({
                        'date': y.index[i],
                        'beta': np.nan,
                        'alpha': np.nan,
                        'r_squared': np.nan
                    })
            
            return pd.DataFrame(results).set_index('date')
        else:
            # Static regression
            try:
                recent_y = y.tail(window)
                recent_X = X.tail(window).values.reshape(-1, 1)
                
                regressor.fit(recent_X, recent_y)
                
                beta = regressor.coef_[0]
                alpha = regressor.intercept_
                
                y_pred = regressor.predict(recent_X)
                r_squared = r2_score(recent_y, y_pred)
                
                return {
                    'beta': beta,
                    'alpha': alpha,
                    'r_squared': r_squared,
                    'method': method
                }
            except:
                return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}
    
    def calculate_time_varying_beta(self, 
                                  asset: str, 
                                  method: str = 'kalman',
                                  **kwargs) -> pd.Series:
        """
        Calculate time-varying beta using advanced methods.
        
        Args:
            asset: Asset symbol
            method: 'kalman', 'garch', 'exponential_smoothing'
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Time series of beta estimates
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return pd.Series()
        
        asset_returns = self.returns_data[asset].dropna()
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        if method == 'exponential_smoothing':
            return self._exponential_smoothing_beta(asset_returns, market_returns, **kwargs)
        elif method == 'kalman':
            return self._kalman_filter_beta(asset_returns, market_returns, **kwargs)
        elif method == 'garch':
            return self._garch_beta(asset_returns, market_returns, **kwargs)
        else:
            # Fallback to rolling beta
            return self.calculate_capm_beta(asset, rolling=True)
    
    def _exponential_smoothing_beta(self, 
                                  asset_returns: pd.Series, 
                                  market_returns: pd.Series,
                                  decay: float = 0.94) -> pd.Series:
        """Calculate beta using exponential smoothing."""
        y = asset_returns - self.daily_rf_rate
        x = market_returns - self.daily_rf_rate
        
        # Initialize
        beta_series = []
        covariance = 0
        market_variance = 0
        
        for i in range(len(y)):
            if i == 0:
                covariance = y.iloc[i] * x.iloc[i]
                market_variance = x.iloc[i] ** 2
            else:
                covariance = decay * covariance + (1 - decay) * y.iloc[i] * x.iloc[i]
                market_variance = decay * market_variance + (1 - decay) * x.iloc[i] ** 2
            
            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = 1.0  # Default to market beta
            
            beta_series.append(beta)
        
        return pd.Series(beta_series, index=asset_returns.index, name=f'{asset_returns.name}_beta_exp')
    
    def _kalman_filter_beta(self, 
                          asset_returns: pd.Series, 
                          market_returns: pd.Series,
                          process_variance: float = 0.001,
                          observation_variance: float = 0.01) -> pd.Series:
        """Calculate beta using Kalman filter (simplified implementation)."""
        y = (asset_returns - self.daily_rf_rate).values
        x = (market_returns - self.daily_rf_rate).values
        
        # Kalman filter parameters
        n = len(y)
        beta_estimates = np.zeros(n)
        P = np.ones(n)  # Error covariance
        
        # Initial estimates
        beta_estimates[0] = 1.0
        P[0] = 1.0
        
        for t in range(1, n):
            # Predict
            beta_pred = beta_estimates[t-1]
            P_pred = P[t-1] + process_variance
            
            # Update
            if x[t] != 0:
                innovation = y[t] - beta_pred * x[t]
                S = x[t]**2 * P_pred + observation_variance
                K = P_pred * x[t] / S
                
                beta_estimates[t] = beta_pred + K * innovation
                P[t] = P_pred - K * x[t] * P_pred
            else:
                beta_estimates[t] = beta_pred
                P[t] = P_pred
        
        return pd.Series(beta_estimates, index=asset_returns.index, name=f'{asset_returns.name}_beta_kalman')
    
    def _garch_beta(self, 
                  asset_returns: pd.Series, 
                  market_returns: pd.Series,
                  window: int = 60) -> pd.Series:
        """Calculate beta using GARCH-based conditional covariance (simplified)."""
        y = asset_returns - self.daily_rf_rate
        x = market_returns - self.daily_rf_rate
        
        beta_series = []
        
        # Rolling GARCH-like beta (simplified version)
        for i in range(window, len(y)):
            window_y = y.iloc[i-window:i]
            window_x = x.iloc[i-window:i]
            
            # Weight recent observations more heavily (exponential decay)
            weights = np.exp(-0.01 * np.arange(window-1, -1, -1))
            weights = weights / weights.sum()
            
            # Weighted covariance and variance
            mean_y = np.average(window_y, weights=weights)
            mean_x = np.average(window_x, weights=weights)
            
            weighted_cov = np.average((window_y - mean_y) * (window_x - mean_x), weights=weights)
            weighted_var_x = np.average((window_x - mean_x)**2, weights=weights)
            
            if weighted_var_x > 0:
                beta = weighted_cov / weighted_var_x
            else:
                beta = 1.0
            
            beta_series.append(beta)
        
        # Pad with NaNs for initial period
        full_series = [np.nan] * window + beta_series
        return pd.Series(full_series, index=asset_returns.index, name=f'{asset_returns.name}_beta_garch')
    
    def calculate_downside_beta(self, 
                              asset: str, 
                              window: int = 252,
                              rolling: bool = False) -> Union[float, pd.Series]:
        """
        Calculate downside beta (beta during market downturns).
        
        Args:
            asset: Asset symbol
            window: Estimation window
            rolling: Whether to calculate rolling estimates
            
        Returns:
            Downside beta estimate(s)
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return np.nan
        
        asset_returns = self.returns_data[asset].dropna()
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        if len(asset_returns) < window:
            return np.nan
        
        # Excess returns
        asset_excess = asset_returns - self.daily_rf_rate
        market_excess = market_returns - self.daily_rf_rate
        
        if rolling:
            beta_series = []
            dates = []
            
            for i in range(window, len(asset_excess)):
                window_asset = asset_excess.iloc[i-window:i]
                window_market = market_excess.iloc[i-window:i]
                
                # Only use periods when market is down
                down_mask = window_market < 0
                
                if down_mask.sum() > 10:  # Need at least 10 down periods
                    down_asset = window_asset[down_mask]
                    down_market = window_market[down_mask]
                    
                    covariance = np.cov(down_asset, down_market)[0, 1]
                    market_variance = np.var(down_market)
                    
                    if market_variance > 0:
                        beta = covariance / market_variance
                    else:
                        beta = np.nan
                else:
                    beta = np.nan
                
                beta_series.append(beta)
                dates.append(asset_excess.index[i])
            
            return pd.Series(beta_series, index=dates, name=f'{asset}_downside_beta')
        else:
            # Static downside beta
            recent_asset = asset_excess.tail(window)
            recent_market = market_excess.tail(window)
            
            # Only use down periods
            down_mask = recent_market < 0
            
            if down_mask.sum() > 10:
                down_asset = recent_asset[down_mask]
                down_market = recent_market[down_mask]
                
                covariance = np.cov(down_asset, down_market)[0, 1]
                market_variance = np.var(down_market)
                
                if market_variance > 0:
                    return covariance / market_variance
            
            return np.nan
    
    def calculate_beta_portfolio(self, 
                               assets: List[str], 
                               weights: Optional[np.ndarray] = None,
                               method: str = 'capm') -> Dict:
        """
        Calculate portfolio beta.
        
        Args:
            assets: List of asset symbols
            weights: Portfolio weights (equal weight if None)
            method: Beta calculation method
            
        Returns:
            Portfolio beta and component contributions
        """
        if weights is None:
            weights = np.ones(len(assets)) / len(assets)
        
        weights = np.array(weights)
        
        # Calculate individual betas
        individual_betas = {}
        for asset in assets:
            if method == 'capm':
                beta = self.calculate_capm_beta(asset)
            elif method == 'regression':
                beta_result = self.calculate_regression_beta(asset)
                beta = beta_result.get('beta', np.nan)
            else:
                beta = self.calculate_capm_beta(asset)
            
            individual_betas[asset] = beta
        
        # Calculate portfolio beta
        valid_assets = [(asset, beta) for asset, beta in individual_betas.items() if not np.isnan(beta)]
        
        if not valid_assets:
            return {
                'portfolio_beta': np.nan,
                'individual_betas': individual_betas,
                'contributions': {}
            }
        
        # Weighted average of individual betas
        asset_names = [asset for asset, _ in valid_assets]
        asset_betas = [beta for _, beta in valid_assets]
        asset_indices = [assets.index(asset) for asset in asset_names]
        asset_weights = weights[asset_indices]
        
        # Renormalize weights for valid assets
        asset_weights = asset_weights / asset_weights.sum()
        
        portfolio_beta = np.sum(asset_weights * asset_betas)
        
        # Calculate contributions
        contributions = {}
        for i, asset in enumerate(asset_names):
            contributions[asset] = asset_weights[i] * asset_betas[i]
        
        return {
            'portfolio_beta': portfolio_beta,
            'individual_betas': individual_betas,
            'contributions': contributions,
            'effective_weights': dict(zip(asset_names, asset_weights))
        }
    
    def analyze_beta_stability(self, 
                             asset: str, 
                             window_sizes: List[int] = [60, 120, 252],
                             methods: List[str] = ['capm', 'regression']) -> pd.DataFrame:
        """
        Analyze beta stability across different windows and methods.
        
        Args:
            asset: Asset symbol
            window_sizes: Different estimation windows
            methods: Different estimation methods
            
        Returns:
            DataFrame with beta estimates and stability metrics
        """
        results = []
        
        for window in window_sizes:
            for method in methods:
                if method == 'capm':
                    beta = self.calculate_capm_beta(asset, window=window)
                    alpha = np.nan
                    r_squared = np.nan
                else:
                    beta_result = self.calculate_regression_beta(asset, window=window, method='ols')
                    beta = beta_result.get('beta', np.nan)
                    alpha = beta_result.get('alpha', np.nan)
                    r_squared = beta_result.get('r_squared', np.nan)
                
                results.append({
                    'window': window,
                    'method': method,
                    'beta': beta,
                    'alpha': alpha,
                    'r_squared': r_squared
                })
        
        results_df = pd.DataFrame(results)
        
        # Calculate stability metrics
        if len(results_df) > 1:
            results_df['beta_std'] = results_df['beta'].std()
            results_df['beta_range'] = results_df['beta'].max() - results_df['beta'].min()
        
        return results_df


def main():
    """Example usage of TraditionalRiskModels."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    print("=== Traditional Risk Models Example ===")
    
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
    
    # Initialize risk models
    risk_models = TraditionalRiskModels(df, market_index='SPY')
    
    print(f"\nEstimating beta for individual stocks...")
    
    # Calculate different types of beta for AAPL
    asset = 'AAPL'
    
    # 1. CAPM Beta
    capm_beta = risk_models.calculate_capm_beta(asset)
    print(f"\nCAPM Beta for {asset}: {camp_beta:.4f}")
    
    # 2. Regression Beta
    regression_result = risk_models.calculate_regression_beta(asset, method='ols')
    print(f"Regression Beta for {asset}: {regression_result['beta']:.4f}")
    print(f"Alpha: {regression_result['alpha']:.6f}")
    print(f"R-squared: {regression_result['r_squared']:.4f}")
    
    # 3. Time-varying Beta
    print(f"\nCalculating time-varying beta...")
    tv_beta = risk_models.calculate_time_varying_beta(asset, method='exponential_smoothing')
    if not tv_beta.empty:
        print(f"Recent time-varying beta: {tv_beta.tail(1).iloc[0]:.4f}")
        print(f"Beta volatility: {tv_beta.std():.4f}")
    
    # 4. Downside Beta
    downside_beta = risk_models.calculate_downside_beta(asset)
    print(f"Downside Beta for {asset}: {downside_beta:.4f}")
    
    # 5. Portfolio Beta
    portfolio_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    portfolio_result = risk_models.calculate_beta_portfolio(portfolio_assets)
    
    print(f"\n=== Portfolio Analysis ===")
    print(f"Portfolio Beta: {portfolio_result['portfolio_beta']:.4f}")
    print(f"Individual Betas:")
    for asset, beta in portfolio_result['individual_betas'].items():
        contribution = portfolio_result['contributions'].get(asset, 0)
        print(f"  {asset}: {beta:.4f} (contribution: {contribution:.4f})")
    
    # 6. Beta Stability Analysis
    print(f"\n=== Beta Stability Analysis for {asset} ===")
    stability_df = risk_models.analyze_beta_stability(asset)
    print(stability_df.round(4))
    
    # 7. Rolling Beta Analysis
    print(f"\nCalculating rolling beta...")
    rolling_beta = risk_models.calculate_capm_beta(asset, window=60, rolling=True)
    if isinstance(rolling_beta, pd.Series) and not rolling_beta.empty:
        print(f"Rolling beta statistics:")
        print(f"  Mean: {rolling_beta.mean():.4f}")
        print(f"  Std: {rolling_beta.std():.4f}")
        print(f"  Min: {rolling_beta.min():.4f}")
        print(f"  Max: {rolling_beta.max():.4f}")
    
    # Save results
    output_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/traditional_risk_analysis.csv'
    
    # Create summary results
    summary_results = []
    for asset in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
        capm_beta = risk_models.calculate_capm_beta(asset)
        reg_result = risk_models.calculate_regression_beta(asset)
        downside_beta = risk_models.calculate_downside_beta(asset)
        
        summary_results.append({
            'asset': asset,
            'capm_beta': capm_beta,
            'regression_beta': reg_result.get('beta', np.nan),
            'alpha': reg_result.get('alpha', np.nan),
            'r_squared': reg_result.get('r_squared', np.nan),
            'downside_beta': downside_beta
        })
    
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Analysis completed for {len(summary_results)} assets")


if __name__ == "__main__":
    main()