"""
Beta Estimation Evaluator
Comprehensive evaluation and comparison system for different beta estimation methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our beta estimation modules
from .traditional_risk_models import TraditionalRiskModels
from .multi_factor_models import MultiFactorModels
from .copula_risk_models import CopulaRiskModels
from .cvar_risk_models import CVaRRiskModels


class BetaEvaluator:
    """
    Comprehensive beta estimation evaluation system.
    
    This class provides tools to compare, evaluate, and select the best
    beta estimation methods across different approaches including traditional,
    multi-factor, copula-based, and CVaR methods.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 market_index: str = 'SPY',
                 risk_free_rate: float = 0.02):
        """
        Initialize with market data.
        
        Args:
            data: DataFrame with market data
            market_index: Market index symbol
            risk_free_rate: Annual risk-free rate
        """
        self.data = data.copy()
        self.market_index = market_index
        self.risk_free_rate = risk_free_rate
        
        # Initialize all beta estimation models
        self.traditional_models = TraditionalRiskModels(data, market_index, risk_free_rate)
        self.multi_factor_models = MultiFactorModels(data, risk_free_rate=risk_free_rate)
        self.copula_models = CopulaRiskModels(data, market_index, risk_free_rate)
        self.cvar_models = CVaRRiskModels(data, market_index, risk_free_rate)
        
        # Storage for results
        self.beta_estimates = {}
        self.evaluation_results = None
        
    def estimate_all_betas(self, 
                          assets: List[str], 
                          window: int = 252,
                          include_rolling: bool = False) -> Dict:
        """
        Estimate beta using all available methods.
        
        Args:
            assets: List of asset symbols
            window: Estimation window
            include_rolling: Whether to include rolling estimates
            
        Returns:
            Dictionary with all beta estimates
        """
        print("Estimating betas using all methods...")
        
        all_estimates = {}
        
        for asset in assets:
            print(f"Processing {asset}...")
            
            asset_estimates = {}
            
            # 1. Traditional Methods
            try:
                # CAPM Beta
                capm_beta = self.traditional_models.calculate_capm_beta(asset, window)
                asset_estimates['capm_beta'] = camp_beta
                
                # Regression Beta
                reg_result = self.traditional_models.calculate_regression_beta(asset, window, 'ols')
                asset_estimates['ols_beta'] = reg_result.get('beta', np.nan)
                asset_estimates['ols_alpha'] = reg_result.get('alpha', np.nan)
                asset_estimates['ols_r_squared'] = reg_result.get('r_squared', np.nan)
                
                # Ridge Regression Beta
                ridge_result = self.traditional_models.calculate_regression_beta(asset, window, 'ridge')
                asset_estimates['ridge_beta'] = ridge_result.get('beta', np.nan)
                
                # Downside Beta
                downside_beta = self.traditional_models.calculate_downside_beta(asset, window)
                asset_estimates['downside_beta'] = downside_beta
                
                # Time-varying Beta (exponential smoothing)
                tv_beta = self.traditional_models.calculate_time_varying_beta(asset, 'exponential_smoothing')
                if isinstance(tv_beta, pd.Series) and not tv_beta.empty:
                    asset_estimates['exp_smooth_beta'] = tv_beta.iloc[-1]
                else:
                    asset_estimates['exp_smooth_beta'] = np.nan
                
            except Exception as e:
                print(f"Error in traditional methods for {asset}: {e}")
                asset_estimates.update({
                    'capm_beta': np.nan, 'ols_beta': np.nan, 'ols_alpha': np.nan,
                    'ols_r_squared': np.nan, 'ridge_beta': np.nan, 'downside_beta': np.nan,
                    'exp_smooth_beta': np.nan
                })
            
            # 2. Multi-Factor Methods
            try:
                # Fama-French Beta
                ff_result = self.multi_factor_models.estimate_fama_french_model(asset, window, include_momentum=False)
                asset_estimates['ff_market_beta'] = ff_result.get('market_beta', np.nan)
                asset_estimates['ff_smb_beta'] = ff_result.get('smb_beta', np.nan)
                asset_estimates['ff_hml_beta'] = ff_result.get('hml_beta', np.nan)
                asset_estimates['ff_alpha'] = ff_result.get('alpha', np.nan)
                asset_estimates['ff_r_squared'] = ff_result.get('r_squared', np.nan)
                
                # PCA Factor Model
                pca_result = self.multi_factor_models.estimate_pca_factor_model(asset, n_factors=3, window=window)
                asset_estimates['pca_pc1_loading'] = pca_result.get('pc1_loading', np.nan)
                asset_estimates['pca_r_squared'] = pca_result.get('r_squared', np.nan)
                
            except Exception as e:
                print(f"Error in multi-factor methods for {asset}: {e}")
                asset_estimates.update({
                    'ff_market_beta': np.nan, 'ff_smb_beta': np.nan, 'ff_hml_beta': np.nan,
                    'ff_alpha': np.nan, 'ff_r_squared': np.nan,
                    'pca_pc1_loading': np.nan, 'pca_r_squared': np.nan
                })
            
            # 3. Copula Methods
            try:
                # Gaussian Copula Beta
                gauss_copula = self.copula_models.estimate_copula_beta(asset, 'gaussian', window=window)
                asset_estimates['gaussian_copula_beta'] = gauss_copula.get('copula_beta', np.nan)
                asset_estimates['gaussian_correlation'] = gauss_copula.get('correlation', np.nan)
                
                # t-Copula Beta
                t_copula = self.copula_models.estimate_copula_beta(asset, 't', window=window)
                asset_estimates['t_copula_beta'] = t_copula.get('copula_beta', np.nan)
                asset_estimates['t_copula_df'] = t_copula.get('df', np.nan)
                
            except Exception as e:
                print(f"Error in copula methods for {asset}: {e}")
                asset_estimates.update({
                    'gaussian_copula_beta': np.nan, 'gaussian_correlation': np.nan,
                    't_copula_beta': np.nan, 't_copula_df': np.nan
                })
            
            # 4. CVaR Methods
            try:
                # CVaR Beta (different methods)
                cvar_reg = self.cvar_models.estimate_cvar_beta(asset, 0.95, 'regression', window)
                asset_estimates['cvar_beta_regression'] = cvar_reg.get('cvar_beta', np.nan)
                
                cvar_ratio = self.cvar_models.estimate_cvar_beta(asset, 0.95, 'ratio', window)
                asset_estimates['cvar_beta_ratio'] = cvar_ratio.get('cvar_beta', np.nan)
                
                cvar_cond = self.cvar_models.estimate_cvar_beta(asset, 0.95, 'conditional', window)
                asset_estimates['cvar_beta_conditional'] = cvar_cond.get('cvar_beta', np.nan)
                
                # Downside CVaR Beta
                downside_cvar = self.cvar_models.calculate_downside_cvar_beta(asset, 0.95, window)
                asset_estimates['downside_cvar_beta'] = downside_cvar.get('downside_cvar_beta', np.nan)
                
            except Exception as e:
                print(f"Error in CVaR methods for {asset}: {e}")
                asset_estimates.update({
                    'cvar_beta_regression': np.nan, 'cvar_beta_ratio': np.nan,
                    'cvar_beta_conditional': np.nan, 'downside_cvar_beta': np.nan
                })
            
            all_estimates[asset] = asset_estimates
        
        self.beta_estimates = all_estimates
        print(f"Beta estimation completed for {len(assets)} assets")
        
        return all_estimates
    
    def evaluate_beta_performance(self, 
                                forward_periods: List[int] = [1, 5, 10, 20],
                                evaluation_window: int = 60) -> pd.DataFrame:
        """
        Evaluate beta performance using forward-looking analysis.
        
        Args:
            forward_periods: Forward periods for evaluation
            evaluation_window: Window for rolling evaluation
            
        Returns:
            DataFrame with performance metrics
        """
        if not self.beta_estimates:
            raise ValueError("No beta estimates available. Run estimate_all_betas() first.")
        
        print("Evaluating beta performance...")
        
        results = []
        
        # Prepare return data
        returns_data = self.traditional_models.returns_data
        
        for asset in self.beta_estimates.keys():
            if asset not in returns_data.columns or self.market_index not in returns_data.columns:
                continue
            
            asset_returns = returns_data[asset].dropna()
            market_returns = returns_data[self.market_index].dropna()
            
            # Align data
            common_dates = asset_returns.index.intersection(market_returns.index)
            asset_returns = asset_returns[common_dates]
            market_returns = market_returns[common_dates]
            
            if len(asset_returns) < evaluation_window * 2:
                continue
            
            # Rolling evaluation
            for i in range(evaluation_window, len(asset_returns) - max(forward_periods)):
                # Estimation window
                est_asset = asset_returns.iloc[i-evaluation_window:i]
                est_market = market_returns.iloc[i-evaluation_window:i]
                
                # Forward returns
                forward_returns = {}
                for period in forward_periods:
                    if i + period < len(asset_returns):
                        forward_returns[f'forward_{period}d'] = asset_returns.iloc[i+period]
                
                if not forward_returns:
                    continue
                
                # Calculate benchmark betas for this window
                benchmark_capm = np.cov(est_asset, est_market)[0, 1] / np.var(est_market)
                
                # Evaluate different beta methods
                beta_methods = self.beta_estimates[asset]
                
                for method_name, beta_value in beta_methods.items():
                    if np.isnan(beta_value):
                        continue
                    
                    result = {
                        'asset': asset,
                        'date': asset_returns.index[i],
                        'method': method_name,
                        'beta_estimate': beta_value,
                        'benchmark_capm_beta': benchmark_camp
                    }
                    
                    # Prediction accuracy for each forward period
                    market_current = market_returns.iloc[i]
                    
                    for period_name, forward_return in forward_returns.items():
                        period_days = int(period_name.split('_')[1].replace('d', ''))
                        
                        # Expected return based on beta
                        if i + period_days < len(market_returns):
                            future_market = market_returns.iloc[i+period_days]
                            expected_return = beta_value * (future_market - self.cvar_models.daily_rf_rate)
                            actual_return = forward_return - self.cvar_models.daily_rf_rate
                            
                            prediction_error = actual_return - expected_return
                            result[f'{period_name}_prediction_error'] = prediction_error
                            result[f'{period_name}_abs_prediction_error'] = abs(prediction_error)
                    
                    results.append(result)
        
        if results:
            self.evaluation_results = pd.DataFrame(results)
            return self.evaluation_results
        else:
            return pd.DataFrame()
    
    def rank_beta_methods(self, evaluation_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Rank beta estimation methods by performance.
        
        Args:
            evaluation_df: Evaluation results DataFrame
            
        Returns:
            DataFrame with method rankings
        """
        if evaluation_df is None:
            evaluation_df = self.evaluation_results
        
        if evaluation_df is None or evaluation_df.empty:
            return pd.DataFrame()
        
        # Calculate performance metrics by method
        method_performance = []
        
        for method in evaluation_df['method'].unique():
            method_data = evaluation_df[evaluation_df['method'] == method]
            
            if len(method_data) == 0:
                continue
            
            perf_metrics = {'method': method}
            
            # Prediction accuracy metrics
            error_cols = [col for col in method_data.columns if 'prediction_error' in col and 'abs' not in col]
            abs_error_cols = [col for col in method_data.columns if 'abs_prediction_error' in col]
            
            for col in error_cols:
                if col in method_data.columns:
                    errors = method_data[col].dropna()
                    if len(errors) > 0:
                        perf_metrics[f'{col}_rmse'] = np.sqrt(np.mean(errors**2))
                        perf_metrics[f'{col}_bias'] = np.mean(errors)
            
            for col in abs_error_cols:
                if col in method_data.columns:
                    abs_errors = method_data[col].dropna()
                    if len(abs_errors) > 0:
                        perf_metrics[f'{col}_mae'] = np.mean(abs_errors)
            
            # Stability metrics
            beta_estimates = method_data['beta_estimate'].dropna()
            if len(beta_estimates) > 1:
                perf_metrics['beta_stability'] = 1 / (1 + beta_estimates.std())  # Higher is more stable
                perf_metrics['beta_mean'] = beta_estimates.mean()
                perf_metrics['beta_std'] = beta_estimates.std()
            
            # Coverage (number of valid estimates)
            perf_metrics['coverage'] = len(beta_estimates) / len(method_data)
            perf_metrics['observations'] = len(method_data)
            
            method_performance.append(perf_metrics)
        
        if not method_performance:
            return pd.DataFrame()
        
        perf_df = pd.DataFrame(method_performance)
        
        # Create composite ranking
        ranking_metrics = []
        
        # Find RMSE columns for ranking
        rmse_cols = [col for col in perf_df.columns if 'rmse' in col]
        mae_cols = [col for col in perf_df.columns if 'mae' in col]
        
        if rmse_cols or mae_cols:
            # Rank by prediction accuracy (lower is better)
            for col in rmse_cols + mae_cols:
                perf_df[f'{col}_rank'] = perf_df[col].rank(ascending=True)
                ranking_metrics.append(f'{col}_rank')
        
        # Rank by stability (higher is better)
        if 'beta_stability' in perf_df.columns:
            perf_df['stability_rank'] = perf_df['beta_stability'].rank(ascending=False)
            ranking_metrics.append('stability_rank')
        
        # Rank by coverage (higher is better)
        if 'coverage' in perf_df.columns:
            perf_df['coverage_rank'] = perf_df['coverage'].rank(ascending=False)
            ranking_metrics.append('coverage_rank')
        
        # Composite rank (average of individual ranks)
        if ranking_metrics:
            perf_df['composite_rank'] = perf_df[ranking_metrics].mean(axis=1)
            perf_df = perf_df.sort_values('composite_rank')
        
        return perf_df
    
    def analyze_beta_consensus(self, assets: List[str] = None) -> pd.DataFrame:
        """
        Analyze consensus among different beta estimation methods.
        
        Args:
            assets: List of assets to analyze (all if None)
            
        Returns:
            DataFrame with consensus analysis
        """
        if not self.beta_estimates:
            return pd.DataFrame()
        
        if assets is None:
            assets = list(self.beta_estimates.keys())
        
        consensus_results = []
        
        for asset in assets:
            if asset not in self.beta_estimates:
                continue
            
            asset_betas = self.beta_estimates[asset]
            
            # Extract beta values (focus on main beta estimates)
            main_beta_methods = [
                'capm_beta', 'ols_beta', 'ff_market_beta', 
                'gaussian_copula_beta', 't_copula_beta',
                'cvar_beta_conditional'
            ]
            
            beta_values = []
            method_names = []
            
            for method in main_beta_methods:
                if method in asset_betas and not np.isnan(asset_betas[method]):
                    beta_values.append(asset_betas[method])
                    method_names.append(method)
            
            if len(beta_values) < 2:
                continue
            
            beta_array = np.array(beta_values)
            
            consensus_metrics = {
                'asset': asset,
                'n_methods': len(beta_values),
                'methods': ', '.join(method_names),
                'mean_beta': np.mean(beta_array),
                'median_beta': np.median(beta_array),
                'std_beta': np.std(beta_array),
                'min_beta': np.min(beta_array),
                'max_beta': np.max(beta_array),
                'range_beta': np.max(beta_array) - np.min(beta_array),
                'coef_var': np.std(beta_array) / abs(np.mean(beta_array)) if np.mean(beta_array) != 0 else np.inf
            }
            
            # Consensus score (higher = more consensus)
            if len(beta_values) > 2:
                # Use inverse of coefficient of variation as consensus score
                consensus_metrics['consensus_score'] = 1 / (1 + consensus_metrics['coef_var'])
            else:
                consensus_metrics['consensus_score'] = 1 / (1 + abs(beta_values[0] - beta_values[1]))
            
            consensus_results.append(consensus_metrics)
        
        return pd.DataFrame(consensus_results).sort_values('consensus_score', ascending=False)
    
    def create_beta_comparison_report(self, 
                                    assets: List[str] = None,
                                    save_path: str = None) -> str:
        """
        Generate comprehensive beta comparison report.
        
        Args:
            assets: Assets to include in report
            save_path: Path to save report
            
        Returns:
            Report text
        """
        if not self.beta_estimates:
            return "No beta estimates available. Run estimate_all_betas() first."
        
        if assets is None:
            assets = list(self.beta_estimates.keys())
        
        report = []
        report.append("="*70)
        report.append("COMPREHENSIVE BETA ESTIMATION COMPARISON REPORT")
        report.append("="*70)
        
        # Summary Statistics
        report.append(f"\nASSETS ANALYZED: {len(assets)}")
        report.append(f"ESTIMATION METHODS: {len(list(self.beta_estimates.values())[0].keys()) if self.beta_estimates else 0}")
        
        # Method Performance Ranking
        if self.evaluation_results is not None:
            ranked_methods = self.rank_beta_methods()
            if not ranked_methods.empty:
                report.append(f"\n{'='*70}")
                report.append("METHOD PERFORMANCE RANKING")
                report.append(f"{'='*70}")
                
                for i, (_, method_row) in enumerate(ranked_methods.head(10).iterrows()):
                    report.append(f"\n{i+1}. {method_row['method']}")
                    if 'composite_rank' in method_row:
                        report.append(f"   Composite Rank: {method_row['composite_rank']:.2f}")
                    if 'beta_stability' in method_row:
                        report.append(f"   Stability Score: {method_row['beta_stability']:.4f}")
                    if 'coverage' in method_row:
                        report.append(f"   Coverage: {method_row['coverage']:.2%}")
        
        # Consensus Analysis
        consensus_df = self.analyze_beta_consensus(assets)
        if not consensus_df.empty:
            report.append(f"\n{'='*70}")
            report.append("BETA CONSENSUS ANALYSIS")
            report.append(f"{'='*70}")
            
            report.append(f"\nHighest Consensus Assets:")
            for i, (_, row) in enumerate(consensus_df.head(5).iterrows()):
                report.append(f"{i+1}. {row['asset']}")
                report.append(f"   Consensus Score: {row['consensus_score']:.4f}")
                report.append(f"   Mean Beta: {row['mean_beta']:.4f} ± {row['std_beta']:.4f}")
                report.append(f"   Methods: {row['n_methods']}")
            
            report.append(f"\nLowest Consensus Assets:")
            for i, (_, row) in enumerate(consensus_df.tail(3).iterrows()):
                report.append(f"{i+1}. {row['asset']}")
                report.append(f"   Consensus Score: {row['consensus_score']:.4f}")
                report.append(f"   Beta Range: [{row['min_beta']:.4f}, {row['max_beta']:.4f}]")
        
        # Individual Asset Analysis
        report.append(f"\n{'='*70}")
        report.append("INDIVIDUAL ASSET ANALYSIS")
        report.append(f"{'='*70}")
        
        for asset in assets[:5]:  # Limit to first 5 assets for brevity
            if asset not in self.beta_estimates:
                continue
            
            asset_betas = self.beta_estimates[asset]
            report.append(f"\n{asset}:")
            
            # Traditional methods
            report.append(f"  Traditional Methods:")
            report.append(f"    CAPM Beta: {asset_betas.get('camp_beta', 'N/A'):.4f}")
            report.append(f"    OLS Beta: {asset_betas.get('ols_beta', 'N/A'):.4f} (R²: {asset_betas.get('ols_r_squared', 'N/A'):.3f})")
            report.append(f"    Downside Beta: {asset_betas.get('downside_beta', 'N/A'):.4f}")
            
            # Multi-factor methods
            report.append(f"  Multi-Factor Methods:")
            report.append(f"    FF Market Beta: {asset_betas.get('ff_market_beta', 'N/A'):.4f}")
            if not np.isnan(asset_betas.get('ff_smb_beta', np.nan)):
                report.append(f"    FF SMB Beta: {asset_betas.get('ff_smb_beta', 'N/A'):.4f}")
                report.append(f"    FF HML Beta: {asset_betas.get('ff_hml_beta', 'N/A'):.4f}")
            
            # Advanced methods
            report.append(f"  Advanced Methods:")
            report.append(f"    Gaussian Copula Beta: {asset_betas.get('gaussian_copula_beta', 'N/A'):.4f}")
            report.append(f"    t-Copula Beta: {asset_betas.get('t_copula_beta', 'N/A'):.4f}")
            report.append(f"    CVaR Beta: {asset_betas.get('cvar_beta_conditional', 'N/A'):.4f}")
        
        # Recommendations
        report.append(f"\n{'='*70}")
        report.append("RECOMMENDATIONS")
        report.append(f"{'='*70}")
        
        if not consensus_df.empty:
            high_consensus = consensus_df[consensus_df['consensus_score'] > 0.8]
            if len(high_consensus) > 0:
                report.append(f"• {len(high_consensus)} assets show high beta consensus (score > 0.8)")
            
            volatile_betas = consensus_df[consensus_df['coef_var'] > 0.3]
            if len(volatile_betas) > 0:
                report.append(f"• {len(volatile_betas)} assets show high beta uncertainty (CoV > 0.3)")
                report.append("  Consider using ensemble or robust estimation methods")
        
        if self.evaluation_results is not None:
            ranked_methods = self.rank_beta_methods()
            if not ranked_methods.empty:
                best_method = ranked_methods.iloc[0]['method']
                report.append(f"• Best performing method overall: {best_method}")
        
        report.append(f"\nReport generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
    
    def visualize_beta_comparison(self, 
                                assets: List[str] = None, 
                                methods: List[str] = None,
                                save_path: str = None) -> None:
        """Create visualizations for beta comparison."""
        if not self.beta_estimates:
            print("No beta estimates available.")
            return
        
        if assets is None:
            assets = list(self.beta_estimates.keys())[:6]  # Limit for readability
        
        if methods is None:
            methods = ['capm_beta', 'ols_beta', 'ff_market_beta', 'gaussian_copula_beta', 'cvar_beta_conditional']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Beta estimates comparison
        beta_data = []
        for asset in assets:
            if asset not in self.beta_estimates:
                continue
            
            for method in methods:
                beta_val = self.beta_estimates[asset].get(method, np.nan)
                if not np.isnan(beta_val):
                    beta_data.append({'Asset': asset, 'Method': method, 'Beta': beta_val})
        
        if beta_data:
            beta_df = pd.DataFrame(beta_data)
            pivot_df = beta_df.pivot(index='Asset', columns='Method', values='Beta')
            
            pivot_df.plot(kind='bar', ax=axes[0, 0], width=0.8)
            axes[0, 0].set_title('Beta Estimates by Method and Asset')
            axes[0, 0].set_ylabel('Beta')
            axes[0, 0].legend(rotation=45, bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Beta distribution
        all_betas = []
        for asset_betas in self.beta_estimates.values():
            for method in methods:
                beta_val = asset_betas.get(method, np.nan)
                if not np.isnan(beta_val):
                    all_betas.append(beta_val)
        
        if all_betas:
            axes[0, 1].hist(all_betas, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0, 1].axvline(np.mean(all_betas), color='red', linestyle='--', label=f'Mean: {np.mean(all_betas):.3f}')
            axes[0, 1].axvline(1.0, color='orange', linestyle='--', label='Market Beta = 1.0')
            axes[0, 1].set_title('Distribution of All Beta Estimates')
            axes[0, 1].set_xlabel('Beta')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Consensus analysis
        consensus_df = self.analyze_beta_consensus(assets)
        if not consensus_df.empty:
            axes[1, 0].scatter(consensus_df['mean_beta'], consensus_df['std_beta'], 
                              s=100, alpha=0.6, c='green')
            
            for _, row in consensus_df.iterrows():
                axes[1, 0].annotate(row['asset'], 
                                  (row['mean_beta'], row['std_beta']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8)
            
            axes[1, 0].set_xlabel('Mean Beta')
            axes[1, 0].set_ylabel('Beta Standard Deviation')
            axes[1, 0].set_title('Beta Consensus: Mean vs Uncertainty')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Method performance (if available)
        if self.evaluation_results is not None:
            ranked_methods = self.rank_beta_methods()
            if not ranked_methods.empty and 'composite_rank' in ranked_methods.columns:
                top_methods = ranked_methods.head(8)
                axes[1, 1].barh(range(len(top_methods)), top_methods['composite_rank'])
                axes[1, 1].set_yticks(range(len(top_methods)))
                axes[1, 1].set_yticklabels(top_methods['method'])
                axes[1, 1].set_xlabel('Composite Rank (Lower = Better)')
                axes[1, 1].set_title('Method Performance Ranking')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Performance data\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Method Performance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Run evaluate_beta_performance()\nfor ranking data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Method Performance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()


def main():
    """Example usage of BetaEvaluator."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    print("=== Beta Evaluation System Example ===")
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY']  # Include market index
    
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
    
    # Initialize beta evaluator
    beta_evaluator = BetaEvaluator(df, market_index='SPY')
    
    # Test assets (exclude market index)
    test_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Estimate all betas
    print("\n1. Estimating betas using all methods...")
    all_betas = beta_evaluator.estimate_all_betas(test_assets, window=252)
    
    print(f"Beta estimation completed for {len(all_betas)} assets")
    
    # Consensus analysis
    print("\n2. Analyzing beta consensus...")
    consensus_df = beta_evaluator.analyze_beta_consensus(test_assets)
    
    if not consensus_df.empty:
        print("Beta consensus results:")
        display_cols = ['asset', 'mean_beta', 'std_beta', 'consensus_score', 'n_methods']
        print(consensus_df[display_cols].round(4).to_string(index=False))
    
    # Performance evaluation (simplified version)
    print("\n3. Basic performance comparison...")
    
    # Show beta estimates for first asset
    first_asset = test_assets[0]
    print(f"\nBeta estimates for {first_asset}:")
    asset_betas = all_betas[first_asset]
    
    main_methods = ['capm_beta', 'ols_beta', 'ff_market_beta', 'gaussian_copula_beta', 'cvar_beta_conditional']
    for method in main_methods:
        value = asset_betas.get(method, np.nan)
        if not np.isnan(value):
            print(f"  {method}: {value:.4f}")
    
    # Generate comprehensive report
    print("\n4. Generating comprehensive report...")
    report_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/beta_comparison_report.txt'
    report = beta_evaluator.create_beta_comparison_report(test_assets, save_path=report_path)
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    viz_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/beta_comparison_visualization.png'
    beta_evaluator.visualize_beta_comparison(test_assets, save_path=viz_path)
    
    # Save detailed results
    print("\n6. Saving detailed results...")
    
    # Convert beta estimates to DataFrame for saving
    results_data = []
    for asset, methods in all_betas.items():
        row = {'asset': asset}
        row.update(methods)
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/comprehensive_beta_estimates.csv'
    results_df.to_csv(results_path, index=False)
    
    # Save consensus analysis
    if not consensus_df.empty:
        consensus_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/beta_consensus_analysis.csv'
        consensus_df.to_csv(consensus_path, index=False)
        print(f"Consensus analysis saved to: {consensus_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("BETA EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Assets analyzed: {len(test_assets)}")
    print(f"Methods compared: ~20 different approaches")
    print(f"\nFiles generated:")
    print(f"  • Comprehensive report: {report_path}")
    print(f"  • Beta estimates: {results_path}")
    print(f"  • Visualization: {viz_path}")
    
    # Show summary statistics
    print(f"\n=== Summary Statistics ===")
    all_beta_values = []
    for asset_methods in all_betas.values():
        for method, value in asset_methods.items():
            if not np.isnan(value) and 'beta' in method:
                all_beta_values.append(value)
    
    if all_beta_values:
        print(f"Total beta estimates: {len(all_beta_values)}")
        print(f"Mean beta: {np.mean(all_beta_values):.4f}")
        print(f"Beta range: [{np.min(all_beta_values):.4f}, {np.max(all_beta_values):.4f}]")
        print(f"Beta std: {np.std(all_beta_values):.4f}")


if __name__ == "__main__":
    main()