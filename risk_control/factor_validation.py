"""
Comprehensive Factor Validation System
Advanced validation framework for both alpha and beta factors with:
- Statistical significance testing
- Out-of-sample validation
- Economic significance analysis  
- Factor decay analysis
- Portfolio construction validation
- Performance attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactorValidator:
    """
    Comprehensive factor validation system.
    
    Validates both alpha and beta factors through multiple lenses:
    1. Statistical validation (IC, t-stats, p-values)
    2. Economic validation (portfolio performance)
    3. Stability validation (decay analysis, regime testing)
    4. Risk validation (factor loadings, correlations)
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 factor_data: Dict[str, pd.DataFrame],
                 validation_period: int = 252,
                 significance_level: float = 0.05,
                 min_ic_threshold: float = 0.02):
        """
        Initialize factor validator.
        
        Args:
            data: Base market data with returns
            factor_data: Dictionary of factor dataframes {'alpha': df, 'beta': df, etc.}
            validation_period: Out-of-sample validation period
            significance_level: Statistical significance threshold
            min_ic_threshold: Minimum IC for practical significance
        """
        self.data = data
        self.factor_data = factor_data
        self.validation_period = validation_period
        self.significance_level = significance_level
        self.min_ic_threshold = min_ic_threshold
        
        # Storage for validation results
        self.validation_results = {}
        self.performance_metrics = {}
        self.factor_rankings = {}
        self.portfolio_tests = {}
        
        logger.info(f"Factor validator initialized")
        logger.info(f"Available factor sets: {list(factor_data.keys())}")
        
    def validate_all_factors(self) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive validation on all factors.
        
        Returns:
            Dictionary with validation results for each factor type
        """
        logger.info("Starting comprehensive factor validation...")
        
        all_results = {}
        
        for factor_type, factors_df in self.factor_data.items():
            logger.info(f"Validating {factor_type} factors...")
            
            # 1. Statistical validation
            statistical_results = self._validate_statistical_significance(factors_df, factor_type)
            
            # 2. Economic validation
            economic_results = self._validate_economic_significance(factors_df, factor_type)
            
            # 3. Stability validation
            stability_results = self._validate_factor_stability(factors_df, factor_type)
            
            # 4. Out-of-sample validation
            oos_results = self._validate_out_of_sample(factors_df, factor_type)
            
            # Combine all validation results
            combined_results = self._combine_validation_results(
                statistical_results, economic_results, stability_results, oos_results
            )
            
            all_results[factor_type] = combined_results
            
        self.validation_results = all_results
        
        # Create factor rankings
        self._create_factor_rankings()
        
        logger.info("Factor validation completed")
        
        return all_results
    
    def _validate_statistical_significance(self, factors_df: pd.DataFrame, factor_type: str) -> pd.DataFrame:
        """Validate statistical significance of factors."""
        logger.info(f"Statistical validation for {factor_type} factors...")
        
        # Add forward returns if not present
        enhanced_df = self._ensure_forward_returns(factors_df)
        
        # Get factor columns
        meta_cols = ['ticker', 'date', 'close', 'volume', 'returns']
        factor_cols = [col for col in enhanced_df.columns 
                      if col not in meta_cols and 'forward_return' not in col]
        
        statistical_results = []
        
        for factor_col in factor_cols:
            if enhanced_df[factor_col].isna().all():
                continue
                
            factor_stats = {'factor_name': factor_col, 'factor_type': factor_type}
            
            # Calculate ICs for different horizons
            for horizon in [1, 5, 10, 20]:
                return_col = f'forward_return_{horizon}d'
                if return_col not in enhanced_df.columns:
                    continue
                
                # Clean data
                clean_data = enhanced_df[[factor_col, return_col]].dropna()
                if len(clean_data) < 30:
                    continue
                
                # Information Coefficient
                ic = clean_data[factor_col].corr(clean_data[return_col])
                rank_ic = clean_data[factor_col].corr(clean_data[return_col], method='spearman')
                
                # IC t-statistic and p-value
                n = len(clean_data)
                if n > 2 and not np.isnan(ic):
                    ic_t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-8))
                    ic_p_value = 2 * (1 - stats.t.cdf(abs(ic_t_stat), n - 2))
                else:
                    ic_t_stat = np.nan
                    ic_p_value = np.nan
                
                factor_stats.update({
                    f'ic_{horizon}d': ic,
                    f'rank_ic_{horizon}d': rank_ic,
                    f'ic_t_stat_{horizon}d': ic_t_stat,
                    f'ic_p_value_{horizon}d': ic_p_value,
                    f'ic_significant_{horizon}d': ic_p_value < self.significance_level if not np.isnan(ic_p_value) else False
                })
            
            # Cross-sectional analysis if multiple tickers
            if 'ticker' in enhanced_df.columns and enhanced_df['ticker'].nunique() > 10:
                cs_results = self._cross_sectional_analysis(enhanced_df, factor_col)
                factor_stats.update(cs_results)
            
            # Factor distribution analysis
            factor_values = enhanced_df[factor_col].dropna()
            if len(factor_values) > 10:
                factor_stats.update({
                    'factor_mean': factor_values.mean(),
                    'factor_std': factor_values.std(),
                    'factor_skew': factor_values.skew(),
                    'factor_kurt': factor_values.kurtosis(),
                    'factor_coverage': len(factor_values) / len(enhanced_df),
                    'factor_autocorr': factor_values.autocorr() if len(factor_values) > 1 else np.nan
                })
            
            statistical_results.append(factor_stats)
        
        return pd.DataFrame(statistical_results)
    
    def _validate_economic_significance(self, factors_df: pd.DataFrame, factor_type: str) -> pd.DataFrame:
        """Validate economic significance through portfolio construction."""
        logger.info(f"Economic validation for {factor_type} factors...")
        
        enhanced_df = self._ensure_forward_returns(factors_df)
        
        # Get factor columns
        meta_cols = ['ticker', 'date', 'close', 'volume', 'returns']
        factor_cols = [col for col in enhanced_df.columns 
                      if col not in meta_cols and 'forward_return' not in col]
        
        economic_results = []
        
        for factor_col in factor_cols[:20]:  # Limit to top 20 factors for computational efficiency
            if enhanced_df[factor_col].isna().all():
                continue
            
            factor_econ = {'factor_name': factor_col, 'factor_type': factor_type}
            
            # Portfolio-based validation
            portfolio_metrics = self._test_factor_portfolio(enhanced_df, factor_col)
            factor_econ.update(portfolio_metrics)
            
            # Long-short portfolio analysis
            ls_metrics = self._test_long_short_portfolio(enhanced_df, factor_col)
            factor_econ.update(ls_metrics)
            
            # Factor timing analysis
            timing_metrics = self._test_factor_timing(enhanced_df, factor_col)
            factor_econ.update(timing_metrics)
            
            economic_results.append(factor_econ)
        
        return pd.DataFrame(economic_results)
    
    def _validate_factor_stability(self, factors_df: pd.DataFrame, factor_type: str) -> pd.DataFrame:
        """Validate factor stability over time."""
        logger.info(f"Stability validation for {factor_type} factors...")
        
        enhanced_df = self._ensure_forward_returns(factors_df)
        
        # Get factor columns
        meta_cols = ['ticker', 'date', 'close', 'volume', 'returns']
        factor_cols = [col for col in enhanced_df.columns 
                      if col not in meta_cols and 'forward_return' not in col]
        
        stability_results = []
        
        for factor_col in factor_cols[:15]:  # Limit for efficiency
            if enhanced_df[factor_col].isna().all():
                continue
            
            factor_stability = {'factor_name': factor_col, 'factor_type': factor_type}
            
            # IC stability over time
            ic_stability = self._analyze_ic_stability(enhanced_df, factor_col)
            factor_stability.update(ic_stability)
            
            # Factor decay analysis
            decay_analysis = self._analyze_factor_decay(enhanced_df, factor_col)
            factor_stability.update(decay_analysis)
            
            # Regime analysis
            regime_analysis = self._analyze_regime_stability(enhanced_df, factor_col)
            factor_stability.update(regime_analysis)
            
            stability_results.append(factor_stability)
        
        return pd.DataFrame(stability_results)
    
    def _validate_out_of_sample(self, factors_df: pd.DataFrame, factor_type: str) -> pd.DataFrame:
        """Validate factors using out-of-sample testing."""
        logger.info(f"Out-of-sample validation for {factor_type} factors...")
        
        enhanced_df = self._ensure_forward_returns(factors_df)
        
        # Split data
        split_date = enhanced_df['date'].quantile(0.7) if 'date' in enhanced_df.columns else None
        if split_date is None:
            logger.warning("No date column found, skipping out-of-sample validation")
            return pd.DataFrame()
        
        train_data = enhanced_df[enhanced_df['date'] <= split_date]
        test_data = enhanced_df[enhanced_df['date'] > split_date]
        
        if len(test_data) < 50:
            logger.warning("Insufficient out-of-sample data")
            return pd.DataFrame()
        
        # Get factor columns
        meta_cols = ['ticker', 'date', 'close', 'volume', 'returns']
        factor_cols = [col for col in enhanced_df.columns 
                      if col not in meta_cols and 'forward_return' not in col]
        
        oos_results = []
        
        for factor_col in factor_cols[:10]:  # Limit for efficiency
            if enhanced_df[factor_col].isna().all():
                continue
            
            factor_oos = {'factor_name': factor_col, 'factor_type': factor_type}
            
            # In-sample vs out-of-sample IC comparison
            for horizon in [1, 5, 10]:
                return_col = f'forward_return_{horizon}d'
                if return_col not in enhanced_df.columns:
                    continue
                
                # In-sample IC
                train_clean = train_data[[factor_col, return_col]].dropna()
                if len(train_clean) > 20:
                    is_ic = train_clean[factor_col].corr(train_clean[return_col])
                else:
                    is_ic = np.nan
                
                # Out-of-sample IC
                test_clean = test_data[[factor_col, return_col]].dropna()
                if len(test_clean) > 20:
                    oos_ic = test_clean[factor_col].corr(test_clean[return_col])
                else:
                    oos_ic = np.nan
                
                # IC consistency
                ic_consistency = abs(is_ic - oos_ic) if not (np.isnan(is_ic) or np.isnan(oos_ic)) else np.nan
                
                factor_oos.update({
                    f'is_ic_{horizon}d': is_ic,
                    f'oos_ic_{horizon}d': oos_ic,
                    f'ic_consistency_{horizon}d': ic_consistency,
                    f'ic_ratio_{horizon}d': oos_ic / is_ic if not np.isnan(is_ic) and is_ic != 0 else np.nan
                })
            
            # Out-of-sample portfolio performance
            if 'ticker' in test_data.columns:
                oos_portfolio = self._test_oos_portfolio(train_data, test_data, factor_col)
                factor_oos.update(oos_portfolio)
            
            oos_results.append(factor_oos)
        
        return pd.DataFrame(oos_results)
    
    def _cross_sectional_analysis(self, data: pd.DataFrame, factor_col: str) -> Dict:
        """Analyze factor performance cross-sectionally."""
        results = {}
        
        # Group by date and analyze cross-sectional IC
        if 'date' in data.columns and 'ticker' in data.columns:
            ic_by_date = []
            
            for date in data['date'].unique():
                date_data = data[data['date'] == date]
                
                if len(date_data) >= 10 and 'forward_return_1d' in date_data.columns:
                    clean_data = date_data[[factor_col, 'forward_return_1d']].dropna()
                    if len(clean_data) >= 5:
                        ic = clean_data[factor_col].corr(clean_data['forward_return_1d'])
                        if not np.isnan(ic):
                            ic_by_date.append(ic)
            
            if len(ic_by_date) > 0:
                ic_series = pd.Series(ic_by_date)
                results.update({
                    'cs_ic_mean': ic_series.mean(),
                    'cs_ic_std': ic_series.std(),
                    'cs_ic_ir': ic_series.mean() / (ic_series.std() + 1e-8),
                    'cs_ic_hit_rate': (ic_series > 0).mean(),
                    'cs_ic_t_stat': ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)) + 1e-8)
                })
        
        return results
    
    def _test_factor_portfolio(self, data: pd.DataFrame, factor_col: str) -> Dict:
        """Test factor through portfolio construction."""
        results = {}
        
        if 'ticker' not in data.columns or 'date' not in data.columns:
            return results
        
        # Create quintile portfolios based on factor values
        portfolio_returns = []
        dates = []
        
        for date in sorted(data['date'].unique()):
            date_data = data[data['date'] == date]
            
            if len(date_data) < 10:
                continue
            
            # Get factor values and forward returns
            factor_return_data = date_data[[factor_col, 'forward_return_1d']].dropna()
            
            if len(factor_return_data) < 5:
                continue
            
            # Create quintiles
            factor_return_data['quintile'] = pd.qcut(
                factor_return_data[factor_col], 
                q=5, 
                labels=False, 
                duplicates='drop'
            )
            
            # Calculate equal-weighted returns for each quintile
            quintile_rets = factor_return_data.groupby('quintile')['forward_return_1d'].mean()
            
            if len(quintile_rets) >= 2:
                portfolio_returns.append(quintile_rets)
                dates.append(date)
        
        if len(portfolio_returns) > 20:
            portfolio_df = pd.DataFrame(portfolio_returns, index=dates)
            
            # Long-short portfolio (Q5 - Q1)
            if 4 in portfolio_df.columns and 0 in portfolio_df.columns:
                long_short = portfolio_df[4] - portfolio_df[0]
                
                results.update({
                    'portfolio_ls_return': long_short.mean() * 252,  # Annualized
                    'portfolio_ls_vol': long_short.std() * np.sqrt(252),
                    'portfolio_ls_sharpe': long_short.mean() / (long_short.std() + 1e-8) * np.sqrt(252),
                    'portfolio_ls_hit_rate': (long_short > 0).mean(),
                    'portfolio_max_dd': (long_short.cumsum() - long_short.cumsum().cummax()).min()
                })
        
        return results
    
    def _test_long_short_portfolio(self, data: pd.DataFrame, factor_col: str) -> Dict:
        """Test long-short portfolio based on factor."""
        results = {}
        
        if 'ticker' not in data.columns or 'date' not in data.columns:
            return results
        
        # Create top/bottom decile portfolios
        portfolio_returns = []
        
        for date in sorted(data['date'].unique()):
            date_data = data[data['date'] == date]
            
            clean_data = date_data[[factor_col, 'forward_return_1d']].dropna()
            
            if len(clean_data) < 20:
                continue
            
            # Top and bottom deciles
            top_threshold = clean_data[factor_col].quantile(0.9)
            bottom_threshold = clean_data[factor_col].quantile(0.1)
            
            top_stocks = clean_data[clean_data[factor_col] >= top_threshold]['forward_return_1d']
            bottom_stocks = clean_data[clean_data[factor_col] <= bottom_threshold]['forward_return_1d']
            
            if len(top_stocks) > 0 and len(bottom_stocks) > 0:
                long_ret = top_stocks.mean()
                short_ret = bottom_stocks.mean()
                ls_ret = long_ret - short_ret
                portfolio_returns.append(ls_ret)
        
        if len(portfolio_returns) > 10:
            ret_series = pd.Series(portfolio_returns)
            
            results.update({
                'decile_ls_return': ret_series.mean() * 252,
                'decile_ls_vol': ret_series.std() * np.sqrt(252),
                'decile_ls_sharpe': ret_series.mean() / (ret_series.std() + 1e-8) * np.sqrt(252),
                'decile_ls_hit_rate': (ret_series > 0).mean()
            })
        
        return results
    
    def _test_factor_timing(self, data: pd.DataFrame, factor_col: str) -> Dict:
        """Test factor timing ability."""
        results = {}
        
        # Simple factor timing: use factor strength as signal
        if len(data) < 100:
            return results
        
        factor_values = data[factor_col].dropna()
        if len(factor_values) < 50:
            return results
        
        # Calculate rolling factor strength
        factor_strength = factor_values.rolling(20).std()  # Use volatility as proxy for factor strength
        
        if len(factor_strength.dropna()) > 20:
            results.update({
                'factor_strength_mean': factor_strength.mean(),
                'factor_strength_vol': factor_strength.std(),
                'factor_timing_score': factor_strength.mean() / (factor_strength.std() + 1e-8)
            })
        
        return results
    
    def _analyze_ic_stability(self, data: pd.DataFrame, factor_col: str) -> Dict:
        """Analyze IC stability over time."""
        results = {}
        
        if 'date' not in data.columns or 'forward_return_1d' not in data.columns:
            return results
        
        # Calculate rolling IC
        window_size = 60  # 60-day rolling window
        rolling_ics = []
        
        sorted_data = data.sort_values('date')
        
        for i in range(window_size, len(sorted_data)):
            window_data = sorted_data.iloc[i-window_size:i]
            clean_window = window_data[[factor_col, 'forward_return_1d']].dropna()
            
            if len(clean_window) >= 20:
                ic = clean_window[factor_col].corr(clean_window['forward_return_1d'])
                if not np.isnan(ic):
                    rolling_ics.append(ic)
        
        if len(rolling_ics) > 10:
            ic_series = pd.Series(rolling_ics)
            results.update({
                'rolling_ic_mean': ic_series.mean(),
                'rolling_ic_std': ic_series.std(),
                'rolling_ic_stability': 1 - (ic_series.std() / (abs(ic_series.mean()) + 1e-8)),
                'ic_persistence': ic_series.autocorr() if len(ic_series) > 1 else np.nan
            })
        
        return results
    
    def _analyze_factor_decay(self, data: pd.DataFrame, factor_col: str) -> Dict:
        """Analyze how quickly factor signal decays."""
        results = {}
        
        # Test IC at different horizons
        horizons = [1, 2, 3, 5, 10, 20]
        decay_ics = []
        
        for horizon in horizons:
            horizon_col = f'forward_return_{horizon}d'
            if horizon_col in data.columns:
                clean_data = data[[factor_col, horizon_col]].dropna()
                if len(clean_data) >= 30:
                    ic = clean_data[factor_col].corr(clean_data[horizon_col])
                    if not np.isnan(ic):
                        decay_ics.append((horizon, ic))
        
        if len(decay_ics) >= 3:
            horizons_list, ics_list = zip(*decay_ics)
            
            # Fit decay curve (exponential decay)
            try:
                # Simple linear fit to log(IC) vs horizon
                log_ics = np.log(np.abs(ics_list))
                valid_idx = ~np.isnan(log_ics) & ~np.isinf(log_ics)
                
                if np.sum(valid_idx) >= 2:
                    slope, intercept, r_val, _, _ = stats.linregress(
                        np.array(horizons_list)[valid_idx], 
                        log_ics[valid_idx]
                    )
                    
                    results.update({
                        'decay_rate': -slope,  # Negative slope indicates decay
                        'decay_r_squared': r_val ** 2,
                        'half_life': np.log(2) / (-slope) if slope < 0 else np.inf
                    })
            except:
                pass
        
        return results
    
    def _analyze_regime_stability(self, data: pd.DataFrame, factor_col: str) -> Dict:
        """Analyze factor stability across different market regimes."""
        results = {}
        
        if len(data) < 200:
            return results
        
        # Define regimes based on market volatility
        if 'returns' in data.columns:
            market_vol = data['returns'].rolling(20).std()
            high_vol_threshold = market_vol.quantile(0.75)
            low_vol_threshold = market_vol.quantile(0.25)
            
            high_vol_data = data[market_vol > high_vol_threshold]
            normal_vol_data = data[(market_vol >= low_vol_threshold) & (market_vol <= high_vol_threshold)]
            low_vol_data = data[market_vol < low_vol_threshold]
            
            regime_results = {}
            
            for regime_name, regime_data in [('high_vol', high_vol_data), 
                                           ('normal_vol', normal_vol_data), 
                                           ('low_vol', low_vol_data)]:
                if len(regime_data) >= 30 and 'forward_return_1d' in regime_data.columns:
                    clean_data = regime_data[[factor_col, 'forward_return_1d']].dropna()
                    if len(clean_data) >= 20:
                        ic = clean_data[factor_col].corr(clean_data['forward_return_1d'])
                        regime_results[f'{regime_name}_ic'] = ic
            
            if len(regime_results) >= 2:
                results.update(regime_results)
                
                # Calculate regime stability
                ic_values = [v for v in regime_results.values() if not np.isnan(v)]
                if len(ic_values) >= 2:
                    results['regime_ic_stability'] = 1 - (np.std(ic_values) / (np.mean(np.abs(ic_values)) + 1e-8))
        
        return results
    
    def _test_oos_portfolio(self, train_data: pd.DataFrame, test_data: pd.DataFrame, factor_col: str) -> Dict:
        """Test out-of-sample portfolio performance."""
        results = {}
        
        if 'ticker' not in test_data.columns or len(test_data) < 20:
            return results
        
        # Simple out-of-sample test: use factor for portfolio construction
        oos_returns = []
        
        for date in sorted(test_data['date'].unique()):
            date_data = test_data[test_data['date'] == date]
            clean_data = date_data[[factor_col, 'forward_return_1d']].dropna()
            
            if len(clean_data) >= 5:
                # Top quintile portfolio
                top_threshold = clean_data[factor_col].quantile(0.8)
                top_stocks = clean_data[clean_data[factor_col] >= top_threshold]['forward_return_1d']
                
                if len(top_stocks) > 0:
                    oos_returns.append(top_stocks.mean())
        
        if len(oos_returns) > 10:
            ret_series = pd.Series(oos_returns)
            results.update({
                'oos_portfolio_return': ret_series.mean() * 252,
                'oos_portfolio_vol': ret_series.std() * np.sqrt(252),
                'oos_portfolio_sharpe': ret_series.mean() / (ret_series.std() + 1e-8) * np.sqrt(252),
                'oos_portfolio_hit_rate': (ret_series > 0).mean()
            })
        
        return results
    
    def _ensure_forward_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure forward returns are available in the data."""
        if 'forward_return_1d' in data.columns:
            return data
        
        # Try to calculate forward returns
        enhanced_data = data.copy()
        
        if 'ticker' in data.columns and 'date' in data.columns and 'close' in data.columns:
            # Calculate forward returns by ticker
            for ticker in data['ticker'].unique():
                ticker_mask = data['ticker'] == ticker
                ticker_data = data[ticker_mask].sort_values('date')
                
                if len(ticker_data) > 1:
                    # Calculate forward returns
                    for horizon in [1, 5, 10, 20]:
                        forward_rets = ticker_data['close'].pct_change(horizon).shift(-horizon)
                        enhanced_data.loc[ticker_mask, f'forward_return_{horizon}d'] = forward_rets.values
        
        return enhanced_data
    
    def _combine_validation_results(self, *result_dfs) -> pd.DataFrame:
        """Combine validation results from different analyses."""
        if not result_dfs or all(df.empty for df in result_dfs):
            return pd.DataFrame()
        
        # Start with the first non-empty dataframe
        combined = None
        for df in result_dfs:
            if not df.empty:
                if combined is None:
                    combined = df.copy()
                else:
                    # Merge on factor_name
                    combined = combined.merge(df, on=['factor_name', 'factor_type'], how='outer', suffixes=('', '_dup'))
                    # Remove duplicate columns
                    dup_cols = [col for col in combined.columns if col.endswith('_dup')]
                    combined = combined.drop(columns=dup_cols)
        
        return combined if combined is not None else pd.DataFrame()
    
    def _create_factor_rankings(self):
        """Create comprehensive factor rankings."""
        all_rankings = {}
        
        for factor_type, results_df in self.validation_results.items():
            if results_df.empty:
                continue
            
            # Create composite scores
            scores_df = results_df.copy()
            
            # IC-based score
            ic_cols = [col for col in scores_df.columns if 'ic_1d' in col and 'p_value' not in col]
            if ic_cols:
                scores_df['ic_score'] = np.abs(scores_df[ic_cols[0]]).fillna(0)
            else:
                scores_df['ic_score'] = 0
            
            # Economic significance score
            econ_cols = [col for col in scores_df.columns if 'ls_sharpe' in col]
            if econ_cols:
                scores_df['econ_score'] = scores_df[econ_cols[0]].fillna(0)
            else:
                scores_df['econ_score'] = 0
            
            # Stability score
            stability_cols = [col for col in scores_df.columns if 'stability' in col]
            if stability_cols:
                scores_df['stability_score'] = scores_df[stability_cols[0]].fillna(0)
            else:
                scores_df['stability_score'] = 0
            
            # Out-of-sample score
            oos_cols = [col for col in scores_df.columns if 'oos_ic_1d' in col]
            if oos_cols:
                scores_df['oos_score'] = np.abs(scores_df[oos_cols[0]]).fillna(0)
            else:
                scores_df['oos_score'] = 0
            
            # Composite score
            scores_df['composite_score'] = (
                scores_df['ic_score'] * 0.3 +
                scores_df['econ_score'] * 0.3 +
                scores_df['stability_score'] * 0.2 +
                scores_df['oos_score'] * 0.2
            )
            
            # Rank factors
            scores_df['rank'] = scores_df['composite_score'].rank(ascending=False)
            
            all_rankings[factor_type] = scores_df.sort_values('composite_score', ascending=False)
        
        self.factor_rankings = all_rankings
    
    def get_top_factors(self, factor_type: str = None, n_factors: int = 10) -> pd.DataFrame:
        """Get top-ranked factors."""
        if not self.factor_rankings:
            logger.warning("No factor rankings available. Run validate_all_factors() first.")
            return pd.DataFrame()
        
        if factor_type and factor_type in self.factor_rankings:
            return self.factor_rankings[factor_type].head(n_factors)
        else:
            # Combine all factor types
            all_factors = []
            for ft, rankings in self.factor_rankings.items():
                rankings_copy = rankings.copy()
                rankings_copy['original_factor_type'] = ft
                all_factors.append(rankings_copy)
            
            if all_factors:
                combined = pd.concat(all_factors, ignore_index=True)
                return combined.sort_values('composite_score', ascending=False).head(n_factors)
            else:
                return pd.DataFrame()
    
    def generate_validation_report(self, output_dir: str = "data/validation"):
        """Generate comprehensive validation report."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save validation results
        for factor_type, results in self.validation_results.items():
            if not results.empty:
                results.to_csv(f"{output_dir}/validation_{factor_type}.csv", index=False)
        
        # Save factor rankings
        for factor_type, rankings in self.factor_rankings.items():
            if not rankings.empty:
                rankings.to_csv(f"{output_dir}/rankings_{factor_type}.csv", index=False)
        
        # Create summary report
        summary_data = []
        
        for factor_type in self.validation_results.keys():
            if factor_type in self.factor_rankings and not self.factor_rankings[factor_type].empty:
                rankings = self.factor_rankings[factor_type]
                
                # Get top factors
                top_factors = rankings.head(5)
                
                summary_data.append({
                    'Factor_Type': factor_type,
                    'Total_Factors': len(rankings),
                    'Avg_IC': rankings['ic_score'].mean(),
                    'Avg_Economic_Score': rankings['econ_score'].mean(),
                    'Avg_Stability': rankings['stability_score'].mean(),
                    'Top_Factor': top_factors.iloc[0]['factor_name'] if len(top_factors) > 0 else 'N/A',
                    'Top_Factor_Score': top_factors.iloc[0]['composite_score'] if len(top_factors) > 0 else 0
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/validation_summary.csv", index=False)
        
        # Create detailed top factors report
        top_factors_all = self.get_top_factors(n_factors=20)
        if not top_factors_all.empty:
            top_factors_all.to_csv(f"{output_dir}/top_factors_overall.csv", index=False)
        
        logger.info(f"Validation report generated in {output_dir}/")
        
        return summary_df


def main():
    """Example usage of FactorValidator."""
    import sys
    sys.path.append('/Users/mengfanlong/Portfolio_Optimization')
    
    print("=== Factor Validation Demo ===")
    
    # This would normally use real factor data from the alpha and beta miners
    # For demo, create sample data
    np.random.seed(42)
    
    # Sample market data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    market_data = []
    for ticker in tickers:
        ticker_data = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'returns': np.random.normal(0.001, 0.02, len(dates)),
            'close': 100 * np.exp(np.random.normal(0.001, 0.02, len(dates)).cumsum())
        })
        market_data.append(ticker_data)
    
    base_data = pd.concat(market_data, ignore_index=True)
    
    # Sample factor data
    alpha_factors = base_data.copy()
    alpha_factors['momentum_20'] = np.random.normal(0, 1, len(alpha_factors))
    alpha_factors['mean_reversion_10'] = np.random.normal(0, 1, len(alpha_factors))
    alpha_factors['volume_factor'] = np.random.normal(0, 1, len(alpha_factors))
    
    beta_factors = pd.DataFrame({
        'ticker': tickers,
        'market_beta': np.random.uniform(0.5, 1.5, len(tickers)),
        'size_beta': np.random.uniform(-0.5, 0.5, len(tickers)),
        'value_beta': np.random.uniform(-0.3, 0.3, len(tickers))
    })
    
    factor_data = {
        'alpha': alpha_factors,
        'beta': beta_factors
    }
    
    # Initialize validator
    validator = FactorValidator(
        data=base_data,
        factor_data=factor_data,
        validation_period=252,
        min_ic_threshold=0.01
    )
    
    # Run validation
    print("Running comprehensive factor validation...")
    validation_results = validator.validate_all_factors()
    
    print(f"Validation completed for {len(validation_results)} factor types")
    
    # Get top factors
    top_factors = validator.get_top_factors(n_factors=10)
    if not top_factors.empty:
        print(f"\n=== Top 10 Factors ===")
        print(top_factors[['factor_name', 'factor_type', 'composite_score']].head(10).round(4))
    
    # Generate validation report
    print("\nGenerating validation report...")
    summary = validator.generate_validation_report()
    print(f"\nValidation Summary:")
    print(summary.round(3))
    
    print("\nFactor validation completed successfully!")
    

if __name__ == "__main__":
    main()