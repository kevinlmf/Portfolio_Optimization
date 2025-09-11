"""
Alpha Factor Evaluator
Comprehensive evaluation and ranking system for alpha factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our factor generators
from .technical_alpha_factors import TechnicalAlphaFactors
from .fundamental_alpha_factors import FundamentalAlphaFactors
from .price_volume_alpha_factors import PriceVolumeAlphaFactors
from .ml_alpha_factors import MLAlphaFactors


class AlphaFactorEvaluator:
    """
    Comprehensive evaluation system for alpha factors.
    
    This class provides tools to evaluate, rank, and select the best
    alpha factors for portfolio optimization using multiple metrics
    and statistical tests.
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
        
        # Initialize factor generators
        self.tech_factors = TechnicalAlphaFactors(self.data)
        self.pv_factors = PriceVolumeAlphaFactors(self.data)
        self.ml_factors = MLAlphaFactors(self.data)
        
        # Storage for results
        self.all_factors_df = None
        self.evaluation_results = None
        
    def generate_all_factors(self, 
                           tickers: List[str] = None,
                           include_fundamental: bool = False) -> pd.DataFrame:
        """
        Generate all types of alpha factors.
        
        Args:
            tickers: List of tickers for fundamental analysis
            include_fundamental: Whether to include fundamental factors
            
        Returns:
            DataFrame with all factors combined
        """
        print("Generating comprehensive alpha factors...")
        
        # Technical factors
        print("1/4: Calculating technical factors...")
        tech_df = self.tech_factors.calculate_all_factors()
        
        # Price-volume factors
        print("2/4: Calculating price-volume factors...")
        pv_df = self.pv_factors.calculate_all_factors()
        
        # ML factors
        print("3/4: Calculating ML factors...")
        ml_df = self.ml_factors.calculate_all_factors()
        
        # Combine factors
        print("4/4: Combining all factors...")
        
        # Start with technical factors as base
        combined_df = tech_df.copy()
        
        # Merge price-volume factors
        pv_factor_cols = [col for col in pv_df.columns 
                         if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
        if pv_factor_cols:
            combined_df = combined_df.merge(
                pv_df[['date', 'tic'] + pv_factor_cols],
                on=['date', 'tic'],
                how='left'
            )
        
        # Merge ML factors
        ml_factor_cols = [col for col in ml_df.columns 
                         if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
        if ml_factor_cols:
            combined_df = combined_df.merge(
                ml_df[['date', 'tic'] + ml_factor_cols],
                on=['date', 'tic'],
                how='left'
            )
        
        # Fundamental factors (if requested and tickers provided)
        if include_fundamental and tickers:
            print("Adding fundamental factors...")
            try:
                fund_factors = FundamentalAlphaFactors(tickers)
                fund_df = fund_factors.calculate_all_factors()
                
                if not fund_df.empty:
                    # Merge fundamental factors
                    fund_factor_cols = [col for col in fund_df.columns 
                                      if col not in ['date', 'tic', 'current_price', 'market_cap']]
                    if fund_factor_cols:
                        combined_df = combined_df.merge(
                            fund_df[['date', 'tic'] + fund_factor_cols],
                            on=['date', 'tic'],
                            how='left'
                        )
            except Exception as e:
                print(f"Warning: Could not generate fundamental factors: {e}")
        
        self.all_factors_df = combined_df
        print(f"Generated {len(combined_df.columns)} total columns for {len(combined_df)} observations")
        
        return combined_df
    
    def evaluate_factors(self, 
                        factors_df: pd.DataFrame = None,
                        forward_periods: List[int] = [1, 5, 10, 20],
                        min_observations: int = 100) -> pd.DataFrame:
        """
        Comprehensive factor evaluation.
        
        Args:
            factors_df: DataFrame with factors (uses self.all_factors_df if None)
            forward_periods: Periods for forward return calculation
            min_observations: Minimum observations required for evaluation
            
        Returns:
            DataFrame with factor evaluation metrics
        """
        if factors_df is None:
            factors_df = self.all_factors_df
            
        if factors_df is None:
            raise ValueError("No factors available. Run generate_all_factors() first.")
        
        print("Evaluating alpha factors...")
        
        # Add forward returns
        factors_with_returns = self._add_forward_returns(factors_df, forward_periods)
        
        # Get factor columns (exclude basic market data)
        basic_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns']
        forward_return_cols = [col for col in factors_with_returns.columns if col.startswith('forward_return_')]
        factor_cols = [col for col in factors_with_returns.columns 
                      if col not in basic_cols + forward_return_cols]
        
        print(f"Evaluating {len(factor_cols)} factors...")
        
        evaluation_results = []
        
        for i, factor in enumerate(factor_cols):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(factor_cols)} factors evaluated")
            
            factor_result = self._evaluate_single_factor(
                factors_with_returns, factor, forward_return_cols, min_observations
            )
            
            if factor_result is not None:
                evaluation_results.append(factor_result)
        
        self.evaluation_results = pd.DataFrame(evaluation_results)
        print(f"Evaluation completed for {len(self.evaluation_results)} factors")
        
        return self.evaluation_results
    
    def _evaluate_single_factor(self, 
                               df: pd.DataFrame, 
                               factor: str, 
                               return_cols: List[str],
                               min_obs: int) -> Optional[Dict]:
        """Evaluate a single factor."""
        try:
            factor_data = df[factor].dropna()
            
            if len(factor_data) < min_obs:
                return None
            
            result = {
                'factor': factor,
                'category': self._categorize_factor(factor),
                'observations': len(factor_data),
                'non_null_pct': len(factor_data) / len(df) * 100
            }
            
            # Basic statistics
            result.update({
                'mean': factor_data.mean(),
                'std': factor_data.std(),
                'skewness': factor_data.skew(),
                'kurtosis': factor_data.kurt(),
                'min': factor_data.min(),
                'max': factor_data.max()
            })
            
            # Information Coefficient analysis
            for return_col in return_cols:
                period = return_col.split('_')[-1].replace('d', '')
                
                # Pearson IC
                ic_data = df[[factor, return_col]].dropna()
                if len(ic_data) >= min_obs:
                    ic = ic_data[factor].corr(ic_data[return_col])
                    result[f'ic_{period}d'] = ic
                    
                    # Rank IC (Spearman)
                    rank_ic = ic_data[factor].corr(ic_data[return_col], method='spearman')
                    result[f'rank_ic_{period}d'] = rank_ic
                    
                    # IC t-statistic
                    if not np.isnan(ic) and len(ic_data) > 2:
                        ic_t_stat = ic * np.sqrt((len(ic_data) - 2) / (1 - ic**2))
                        result[f'ic_tstat_{period}d'] = ic_t_stat
                        
                        # IC p-value
                        ic_p_val = 2 * (1 - stats.t.cdf(abs(ic_t_stat), len(ic_data) - 2))
                        result[f'ic_pval_{period}d'] = ic_p_val
                else:
                    result[f'ic_{period}d'] = np.nan
                    result[f'rank_ic_{period}d'] = np.nan
                    result[f'ic_tstat_{period}d'] = np.nan
                    result[f'ic_pval_{period}d'] = np.nan
            
            # Factor stability and persistence
            result['autocorr_1'] = factor_data.autocorr(lag=1) if len(factor_data) > 1 else np.nan
            result['autocorr_5'] = factor_data.autocorr(lag=5) if len(factor_data) > 5 else np.nan
            
            # Factor turnover (average absolute change)
            factor_changes = factor_data.diff().abs()
            result['turnover'] = factor_changes.mean() / factor_data.std() if factor_data.std() > 0 else np.nan
            
            # Factor coverage (non-null percentage across stocks and time)
            result['coverage_score'] = self._calculate_coverage_score(df, factor)
            
            # Factor uniqueness (correlation with other factors)
            result['uniqueness_score'] = self._calculate_uniqueness_score(df, factor, return_cols[0])
            
            return result
            
        except Exception as e:
            print(f"Error evaluating factor {factor}: {e}")
            return None
    
    def _categorize_factor(self, factor_name: str) -> str:
        """Categorize factor based on its name."""
        factor_name_lower = factor_name.lower()
        
        if any(term in factor_name_lower for term in ['momentum', 'rsi', 'macd', 'stoch', 'roc']):
            return 'momentum'
        elif any(term in factor_name_lower for term in ['volatility', 'atr', 'bb_', 'realized_vol']):
            return 'volatility'
        elif any(term in factor_name_lower for term in ['sma', 'ema', 'wma', 'ma_', 'trend', 'adx']):
            return 'trend'
        elif any(term in factor_name_lower for term in ['volume', 'obv', 'vpt', 'pv_']):
            return 'volume'
        elif any(term in factor_name_lower for term in ['pe_', 'pb_', 'ps_', 'roe', 'roa', 'debt']):
            return 'fundamental'
        elif any(term in factor_name_lower for term in ['prediction', 'rf_', 'gbm_', 'xgb_', 'lgb_']):
            return 'ml_prediction'
        elif any(term in factor_name_lower for term in ['pca', 'ica', 'cluster', 'regime']):
            return 'ml_feature'
        elif any(term in factor_name_lower for term in ['gap_', 'inside_', 'outside_', 'doji', 'hammer']):
            return 'pattern'
        else:
            return 'other'
    
    def _calculate_coverage_score(self, df: pd.DataFrame, factor: str) -> float:
        """Calculate coverage score for a factor."""
        try:
            total_obs = len(df)
            non_null_obs = df[factor].notna().sum()
            
            # Coverage across time and stocks
            coverage_pct = non_null_obs / total_obs
            
            # Penalty for uneven coverage across stocks
            stock_coverage = df.groupby('tic')[factor].apply(lambda x: x.notna().sum() / len(x))
            coverage_std = stock_coverage.std()
            
            # Higher score for more even coverage
            coverage_score = coverage_pct * (1 - coverage_std)
            
            return max(0, min(1, coverage_score))
        except:
            return 0.0
    
    def _calculate_uniqueness_score(self, df: pd.DataFrame, factor: str, return_col: str) -> float:
        """Calculate uniqueness score for a factor."""
        try:
            # Sample other factors to calculate correlation
            other_factors = []
            factor_cols = [col for col in df.columns 
                          if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
                          and not col.startswith('forward_return_')
                          and col != factor]
            
            # Sample up to 20 other factors for efficiency
            sample_factors = np.random.choice(factor_cols, min(20, len(factor_cols)), replace=False)
            
            correlations = []
            for other_factor in sample_factors:
                corr_data = df[[factor, other_factor]].dropna()
                if len(corr_data) > 10:
                    corr = abs(corr_data[factor].corr(corr_data[other_factor]))
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if not correlations:
                return 1.0
            
            # Higher uniqueness score for lower average correlation
            avg_corr = np.mean(correlations)
            uniqueness_score = 1 - avg_corr
            
            return max(0, min(1, uniqueness_score))
        except:
            return 0.5
    
    def _add_forward_returns(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add forward returns to the dataframe."""
        result_dfs = []
        
        for tic in df['tic'].unique():
            tic_data = df[df['tic'] == tic].copy()
            
            for period in periods:
                tic_data[f'forward_return_{period}d'] = tic_data['close'].pct_change(period).shift(-period)
            
            result_dfs.append(tic_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def rank_factors(self, 
                    evaluation_df: pd.DataFrame = None,
                    ranking_method: str = 'composite') -> pd.DataFrame:
        """
        Rank factors based on evaluation metrics.
        
        Args:
            evaluation_df: Evaluation results (uses self.evaluation_results if None)
            ranking_method: 'ic_abs', 'ic_tstat', 'composite'
            
        Returns:
            DataFrame with ranked factors
        """
        if evaluation_df is None:
            evaluation_df = self.evaluation_results
            
        if evaluation_df is None:
            raise ValueError("No evaluation results available. Run evaluate_factors() first.")
        
        df = evaluation_df.copy()
        
        if ranking_method == 'ic_abs':
            # Rank by absolute IC
            df['rank_score'] = df['ic_1d'].abs()
            
        elif ranking_method == 'ic_tstat':
            # Rank by IC t-statistic
            df['rank_score'] = df['ic_tstat_1d'].abs()
            
        elif ranking_method == 'composite':
            # Composite ranking considering multiple criteria
            
            # Normalize metrics to 0-1 scale
            metrics_to_normalize = ['ic_1d', 'ic_5d', 'ic_tstat_1d', 'coverage_score', 'uniqueness_score']
            
            for metric in metrics_to_normalize:
                if metric in df.columns:
                    df[f'{metric}_norm'] = self._normalize_metric(df[metric])
                else:
                    df[f'{metric}_norm'] = 0
            
            # Composite score with weights
            weights = {
                'ic_1d_norm': 0.3,
                'ic_5d_norm': 0.2,
                'ic_tstat_1d_norm': 0.2,
                'coverage_score_norm': 0.15,
                'uniqueness_score_norm': 0.15
            }
            
            df['rank_score'] = sum(df[metric] * weight for metric, weight in weights.items())
        
        # Rank factors
        df['rank'] = df['rank_score'].rank(ascending=False)
        df = df.sort_values('rank')
        
        return df
    
    def _normalize_metric(self, series: pd.Series) -> pd.Series:
        """Normalize metric to 0-1 scale using absolute values."""
        abs_series = series.abs()
        min_val = abs_series.min()
        max_val = abs_series.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        
        return (abs_series - min_val) / (max_val - min_val)
    
    def select_top_factors(self, 
                          ranked_df: pd.DataFrame = None,
                          top_n: int = 20,
                          diversify_by_category: bool = True) -> pd.DataFrame:
        """
        Select top factors with optional category diversification.
        
        Args:
            ranked_df: Ranked factors DataFrame
            top_n: Number of top factors to select
            diversify_by_category: Whether to ensure category diversity
            
        Returns:
            DataFrame with selected top factors
        """
        if ranked_df is None:
            ranked_df = self.rank_factors()
        
        if diversify_by_category:
            # Select factors ensuring category diversity
            selected_factors = []
            used_categories = set()
            
            for _, row in ranked_df.iterrows():
                if len(selected_factors) >= top_n:
                    break
                
                category = row['category']
                
                # Allow up to 3 factors per category
                category_count = sum(1 for f in selected_factors if f['category'] == category)
                
                if category_count < 3:
                    selected_factors.append(row.to_dict())
                    used_categories.add(category)
            
            # Fill remaining slots with best remaining factors
            remaining_slots = top_n - len(selected_factors)
            selected_factor_names = {f['factor'] for f in selected_factors}
            
            for _, row in ranked_df.iterrows():
                if len(selected_factors) >= top_n:
                    break
                
                if row['factor'] not in selected_factor_names:
                    selected_factors.append(row.to_dict())
            
            return pd.DataFrame(selected_factors)
        else:
            # Simple top-N selection
            return ranked_df.head(top_n)
    
    def generate_factor_report(self, 
                             top_factors_df: pd.DataFrame = None,
                             save_path: str = None) -> str:
        """
        Generate comprehensive factor analysis report.
        
        Args:
            top_factors_df: Top factors DataFrame
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        if top_factors_df is None:
            ranked_df = self.rank_factors()
            top_factors_df = self.select_top_factors(ranked_df)
        
        report = []
        report.append("="*60)
        report.append("ALPHA FACTOR ANALYSIS REPORT")
        report.append("="*60)
        
        # Summary statistics
        total_factors = len(self.evaluation_results) if self.evaluation_results is not None else 0
        report.append(f"\nTOTAL FACTORS EVALUATED: {total_factors}")
        report.append(f"TOP FACTORS SELECTED: {len(top_factors_df)}")
        
        # Category distribution
        if 'category' in top_factors_df.columns:
            category_dist = top_factors_df['category'].value_counts()
            report.append(f"\nCATEGORY DISTRIBUTION:")
            for category, count in category_dist.items():
                report.append(f"  {category}: {count} factors")
        
        # Top factors summary
        report.append(f"\n{'='*60}")
        report.append("TOP ALPHA FACTORS")
        report.append(f"{'='*60}")
        
        for i, (_, factor_row) in enumerate(top_factors_df.head(10).iterrows()):
            report.append(f"\n{i+1}. {factor_row['factor']} ({factor_row['category']})")
            report.append(f"   IC (1d): {factor_row.get('ic_1d', 'N/A'):.4f}")
            report.append(f"   IC (5d): {factor_row.get('ic_5d', 'N/A'):.4f}")
            report.append(f"   Coverage: {factor_row.get('coverage_score', 'N/A'):.2%}")
            report.append(f"   Uniqueness: {factor_row.get('uniqueness_score', 'N/A'):.2%}")
            report.append(f"   Rank Score: {factor_row.get('rank_score', 'N/A'):.4f}")
        
        # Performance statistics
        if 'ic_1d' in top_factors_df.columns:
            ic_stats = top_factors_df['ic_1d'].describe()
            report.append(f"\n{'='*60}")
            report.append("IC PERFORMANCE STATISTICS (1-DAY)")
            report.append(f"{'='*60}")
            report.append(f"Mean IC: {ic_stats['mean']:.4f}")
            report.append(f"Std IC: {ic_stats['std']:.4f}")
            report.append(f"Best IC: {ic_stats['max']:.4f}")
            report.append(f"Worst IC: {ic_stats['min']:.4f}")
        
        # Factor quality metrics
        report.append(f"\n{'='*60}")
        report.append("FACTOR QUALITY METRICS")
        report.append(f"{'='*60}")
        
        if 'coverage_score' in top_factors_df.columns:
            avg_coverage = top_factors_df['coverage_score'].mean()
            report.append(f"Average Coverage Score: {avg_coverage:.2%}")
        
        if 'uniqueness_score' in top_factors_df.columns:
            avg_uniqueness = top_factors_df['uniqueness_score'].mean()
            report.append(f"Average Uniqueness Score: {avg_uniqueness:.2%}")
        
        # Recommendations
        report.append(f"\n{'='*60}")
        report.append("RECOMMENDATIONS")
        report.append(f"{'='*60}")
        
        high_ic_factors = top_factors_df[top_factors_df.get('ic_1d', 0).abs() > 0.05]
        if len(high_ic_factors) > 0:
            report.append(f"• {len(high_ic_factors)} factors show strong predictive power (|IC| > 0.05)")
        
        stable_factors = top_factors_df[top_factors_df.get('autocorr_1', 0) > 0.5]
        if len(stable_factors) > 0:
            report.append(f"• {len(stable_factors)} factors show good stability (autocorr > 0.5)")
        
        unique_factors = top_factors_df[top_factors_df.get('uniqueness_score', 0) > 0.7]
        if len(unique_factors) > 0:
            report.append(f"• {len(unique_factors)} factors are highly unique (uniqueness > 0.7)")
        
        report.append(f"\nReport generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
    
    def create_factor_visualization(self, 
                                  top_factors_df: pd.DataFrame = None,
                                  save_path: str = None) -> None:
        """Create visualizations for factor analysis."""
        if top_factors_df is None:
            ranked_df = self.rank_factors()
            top_factors_df = self.select_top_factors(ranked_df)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. IC Distribution
        if 'ic_1d' in top_factors_df.columns:
            axes[0, 0].hist(top_factors_df['ic_1d'].dropna(), bins=20, alpha=0.7, color='steelblue')
            axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Distribution of 1-Day Information Coefficients')
            axes[0, 0].set_xlabel('IC')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Category Distribution
        if 'category' in top_factors_df.columns:
            category_counts = top_factors_df['category'].value_counts()
            axes[0, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Factor Category Distribution')
        
        # 3. IC vs Coverage Score
        if 'ic_1d' in top_factors_df.columns and 'coverage_score' in top_factors_df.columns:
            scatter_data = top_factors_df[['ic_1d', 'coverage_score']].dropna()
            axes[1, 0].scatter(scatter_data['coverage_score'], scatter_data['ic_1d'].abs(), 
                              alpha=0.6, color='green')
            axes[1, 0].set_xlabel('Coverage Score')
            axes[1, 0].set_ylabel('|IC|')
            axes[1, 0].set_title('Factor Coverage vs Predictive Power')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top Factors Bar Chart
        if 'rank_score' in top_factors_df.columns:
            top_10 = top_factors_df.head(10)
            axes[1, 1].barh(range(len(top_10)), top_10['rank_score'])
            axes[1, 1].set_yticks(range(len(top_10)))
            axes[1, 1].set_yticklabels([f[:20] + '...' if len(f) > 20 else f 
                                      for f in top_10['factor']])
            axes[1, 1].set_xlabel('Rank Score')
            axes[1, 1].set_title('Top 10 Alpha Factors')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()


def main():
    """Example usage of AlphaFactorEvaluator."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    print("=== Alpha Factor Evaluator Example ===")
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
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
    
    # Initialize evaluator
    evaluator = AlphaFactorEvaluator(df)
    
    # Generate all factors
    print("\nGenerating comprehensive alpha factors...")
    all_factors = evaluator.generate_all_factors(tickers=tickers, include_fundamental=False)
    
    # Evaluate factors
    print("\nEvaluating factors...")
    evaluation_results = evaluator.evaluate_factors(min_observations=50)
    
    # Rank factors
    print("\nRanking factors...")
    ranked_factors = evaluator.rank_factors()
    
    # Select top factors
    print("\nSelecting top factors...")
    top_factors = evaluator.select_top_factors(top_n=20)
    
    # Generate report
    print("\nGenerating comprehensive report...")
    report_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/alpha_factor_report.txt'
    report = evaluator.generate_factor_report(save_path=report_path)
    
    # Create visualizations
    print("\nCreating visualizations...")
    viz_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/alpha_factor_analysis.png'
    evaluator.create_factor_visualization(save_path=viz_path)
    
    # Save results
    factors_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/all_alpha_factors.csv'
    all_factors.to_csv(factors_path, index=False)
    
    eval_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/factor_evaluation_results.csv'
    evaluation_results.to_csv(eval_path, index=False)
    
    top_factors_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/top_alpha_factors.csv'
    top_factors.to_csv(top_factors_path, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total factors generated: {len(all_factors.columns)}")
    print(f"Factors evaluated: {len(evaluation_results)}")
    print(f"Top factors selected: {len(top_factors)}")
    print(f"\nFiles generated:")
    print(f"  • All factors: {factors_path}")
    print(f"  • Evaluation results: {eval_path}")
    print(f"  • Top factors: {top_factors_path}")
    print(f"  • Analysis report: {report_path}")
    print(f"  • Visualization: {viz_path}")
    
    # Show top 5 factors
    print(f"\nTOP 5 ALPHA FACTORS:")
    for i, (_, row) in enumerate(top_factors.head(5).iterrows()):
        print(f"{i+1}. {row['factor']} ({row['category']})")
        print(f"   IC: {row.get('ic_1d', 'N/A'):.4f}, Score: {row.get('rank_score', 'N/A'):.4f}")


if __name__ == "__main__":
    main()