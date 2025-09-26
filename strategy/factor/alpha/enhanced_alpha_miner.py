"""
Enhanced Alpha Miner - Separate Technical & Macro Alpha Generation
Advanced alpha factor mining that separately generates and combines technical and macroeconomic factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Import existing components
try:
    from strategy.factor.alpha.real_alpha_miner import RealAlphaMiner
    from strategy.factor.alpha.macroeconomic_alpha_factors import MacroeconomicAlphaFactors
    from strategy.factor.alpha.alpha_combiner import AlphaCombiner
    from data.macro_data_fetcher import MacroDataFetcher
except ImportError as e:
    logger.warning(f"Some imports failed: {e}")


class EnhancedAlphaMiner:
    """
    Enhanced alpha factor mining system that:
    1. Separately mines technical and macroeconomic factors
    2. Intelligently combines them using various strategies
    3. Provides detailed attribution and performance analysis
    4. Supports regime-aware dynamic weighting
    """

    def __init__(self,
                 data: pd.DataFrame,
                 min_ic_threshold: float = 0.02,
                 combination_method: str = 'dynamic_regime',
                 rebalance_frequency: int = 21):
        """
        Initialize Enhanced Alpha Miner.

        Args:
            data: Market data with price, volume, and other features
            min_ic_threshold: Minimum IC threshold for factor validation
            combination_method: Method to combine technical and macro factors
            rebalance_frequency: Days between factor rebalancing
        """
        self.data = data.copy()
        self.min_ic_threshold = min_ic_threshold
        self.combination_method = combination_method
        self.rebalance_frequency = rebalance_frequency

        # Standardize date column
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date']).dt.tz_localize(None)

        # Initialize components
        self.technical_miner = None
        self.macro_factors_calculator = None
        self.alpha_combiner = AlphaCombiner(
            min_ic_threshold=min_ic_threshold,
            rebalance_frequency=rebalance_frequency
        )

        # Storage for results
        self.technical_factors = pd.DataFrame()
        self.macro_factors = pd.DataFrame()
        self.combined_factors = pd.DataFrame()
        self.performance_attribution = {}
        self.factor_performance = pd.DataFrame()

        logger.info(f"Enhanced Alpha Miner initialized")
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        logger.info(f"Tickers: {self.data['ticker'].nunique()}")

    def mine_technical_factors(self) -> pd.DataFrame:
        """Mine technical analysis factors."""
        logger.info("Mining technical analysis factors...")

        try:
            # Initialize technical miner
            self.technical_miner = RealAlphaMiner(
                data=self.data,
                min_ic_threshold=self.min_ic_threshold
            )

            # Mine technical factors (excluding macro)
            technical_factors = self._mine_technical_only()

            if not technical_factors.empty:
                # Add forward returns for evaluation
                technical_factors = self.technical_miner._add_forward_returns(technical_factors)

                self.technical_factors = technical_factors
                logger.info(f"Technical factors mined: {technical_factors.shape}")

                # Get technical factor categories
                tech_cols = [col for col in technical_factors.columns
                            if col not in ['date', 'ticker'] and not col.startswith('forward_return_')]
                logger.info(f"Technical factor categories: {len(tech_cols)} factors")

                return technical_factors
            else:
                logger.warning("No technical factors generated")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Technical factor mining failed: {e}")
            return pd.DataFrame()

    def mine_macro_factors(self) -> pd.DataFrame:
        """Mine macroeconomic factors."""
        logger.info("Mining macroeconomic factors...")

        try:
            # Get unique tickers
            tickers = self.data['ticker'].unique().tolist()

            # Initialize macro factor calculator
            self.macro_factors_calculator = MacroeconomicAlphaFactors(
                tickers=tickers,
                period="2y"
            )

            # Calculate macro factors
            macro_df = self.macro_factors_calculator.calculate_all_factors()

            if macro_df.empty:
                logger.warning("No macroeconomic factors generated")
                return pd.DataFrame()

            # Create composite factors
            macro_df = self.macro_factors_calculator.create_composite_macro_factors(macro_df)

            # Align macro factors with stock data timeline
            aligned_macro = self._align_macro_with_stock_data(macro_df)

            self.macro_factors = aligned_macro
            logger.info(f"Macro factors mined: {aligned_macro.shape}")

            # Get macro factor categories
            macro_cols = [col for col in aligned_macro.columns
                         if col not in ['date', 'ticker']]
            logger.info(f"Macro factor categories: {len(macro_cols)} factors")

            return aligned_macro

        except Exception as e:
            logger.error(f"Macro factor mining failed: {e}")
            return pd.DataFrame()

    def combine_alpha_factors(self,
                            method: str = None,
                            **kwargs) -> pd.DataFrame:
        """
        Combine technical and macro factors intelligently.

        Args:
            method: Combination method (overrides default)
                - 'equal_weight': Simple 50-50 combination
                - 'ic_weighted': Information coefficient weighted
                - 'dynamic_regime': Regime-aware dynamic weighting
                - 'risk_parity': Risk-adjusted weighting
                - 'pca_weighted': Principal component based

        Returns:
            Combined alpha factors dataframe
        """
        if method is None:
            method = self.combination_method

        logger.info(f"Combining alpha factors using method: {method}")

        if self.technical_factors.empty and self.macro_factors.empty:
            logger.warning("No factors to combine")
            return pd.DataFrame()

        # Set factors in combiner
        self.alpha_combiner.set_factors(
            self.technical_factors,
            self.macro_factors
        )

        # Combine factors
        combined = self.alpha_combiner.combine_factors(
            method=method,
            **kwargs
        )

        if not combined.empty:
            self.combined_factors = combined
            logger.info(f"Alpha combination completed: {combined.shape}")

            # Evaluate combination performance
            returns_data = self._prepare_returns_data()
            if not returns_data.empty:
                metrics = self.alpha_combiner.evaluate_combination_performance(
                    combined, returns_data
                )
                self.performance_attribution = metrics
                logger.info(f"Performance attribution calculated: {len(metrics)} metrics")

            return combined
        else:
            logger.warning("Factor combination resulted in empty dataframe")
            return pd.DataFrame()

    def mine_all_alpha_factors(self,
                             separate_analysis: bool = True) -> pd.DataFrame:
        """
        Mine all alpha factors with technical-macro separation.

        Args:
            separate_analysis: If True, mine technical and macro separately then combine
                             If False, use traditional combined approach

        Returns:
            Combined alpha factors
        """
        logger.info("Mining all alpha factors with tech-macro separation...")

        if separate_analysis:
            # Mine technical factors
            tech_factors = self.mine_technical_factors()

            # Mine macro factors
            macro_factors = self.mine_macro_factors()

            # Combine intelligently
            combined = self.combine_alpha_factors()

            # Create comprehensive factor performance report
            self._create_factor_performance_report()

            return combined

        else:
            # Fall back to traditional approach
            logger.info("Using traditional combined approach...")
            miner = RealAlphaMiner(
                data=self.data,
                min_ic_threshold=self.min_ic_threshold
            )
            return miner.mine_all_alpha_factors()

    def _mine_technical_only(self) -> pd.DataFrame:
        """Mine only technical factors, excluding macro factors."""

        # Mine individual categories separately
        technical_parts = []

        # Technical analysis factors
        tech_analysis = self.technical_miner._mine_technical_factors()
        if not tech_analysis.empty:
            technical_parts.append(tech_analysis)

        # Fundamental factors (company-specific, not macro)
        fundamental = self.technical_miner._mine_fundamental_factors()
        if not fundamental.empty:
            technical_parts.append(fundamental)

        # Market microstructure factors
        microstructure = self.technical_miner._mine_microstructure_factors()
        if not microstructure.empty:
            technical_parts.append(microstructure)

        # Cross-sectional factors (relative ranking)
        cross_sectional = self.technical_miner._mine_cross_sectional_factors()
        if not cross_sectional.empty:
            technical_parts.append(cross_sectional)

        # Machine learning factors (based on technical data)
        ml_factors = self.technical_miner._mine_ml_factors()
        if not ml_factors.empty:
            technical_parts.append(ml_factors)

        # Alternative factors (market-based, not macro)
        alternative = self.technical_miner._mine_alternative_factors()
        if not alternative.empty:
            technical_parts.append(alternative)

        # Combine technical parts
        if technical_parts:
            # Normalize dates for safe concatenation
            normalized_parts = [self._normalize_dates(df) for df in technical_parts]
            combined_technical = pd.concat(normalized_parts, axis=1)

            # Remove duplicate columns
            combined_technical = combined_technical.loc[:, ~combined_technical.columns.duplicated()]

            logger.info(f"Technical-only factors combined: {combined_technical.shape}")
            return combined_technical
        else:
            logger.warning("No technical factors generated")
            return pd.DataFrame()

    def _align_macro_with_stock_data(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Align macro factors with stock data dates using forward fill."""

        if macro_df.empty or self.data.empty:
            return macro_df

        aligned_data = []

        for ticker in self.data['ticker'].unique():
            # Get stock dates for this ticker
            ticker_stock_data = self.data[self.data['ticker'] == ticker].copy()
            stock_dates = sorted(ticker_stock_data['date'].unique())

            # Get macro data for this ticker
            ticker_macro_data = macro_df[macro_df['tic'] == ticker] if 'tic' in macro_df.columns else macro_df

            if ticker_macro_data.empty:
                # Create default macro factors if none exist
                for date in stock_dates:
                    aligned_data.append({
                        'date': date,
                        'ticker': ticker,
                        'macro_default': 0.0
                    })
                continue

            # Get latest macro values for forward filling
            latest_macro = ticker_macro_data.iloc[-1]

            # Create aligned data for each stock date
            for date in stock_dates:
                row_data = {
                    'date': date,
                    'ticker': ticker
                }

                # Add macro factors (forward fill latest values)
                for col in ticker_macro_data.columns:
                    if col not in ['date', 'tic']:
                        if pd.notna(latest_macro[col]):
                            row_data[f'macro_{col}'] = latest_macro[col]

                if len(row_data) > 2:  # More than just date and ticker
                    aligned_data.append(row_data)

        if aligned_data:
            result = pd.DataFrame(aligned_data)
            result = self._normalize_dates(result)
            return result
        else:
            return pd.DataFrame()

    def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare returns data for performance evaluation."""

        if 'returns' in self.data.columns:
            return self.data[['date', 'ticker', 'returns']].copy()
        elif 'close' in self.data.columns:
            # Calculate returns from close prices
            returns_data = []
            for ticker in self.data['ticker'].unique():
                ticker_data = self.data[self.data['ticker'] == ticker].copy()
                ticker_data = ticker_data.sort_values('date')
                returns = ticker_data['close'].pct_change()

                for i, (_, row) in enumerate(ticker_data.iterrows()):
                    if i > 0 and pd.notna(returns.iloc[i]):
                        returns_data.append({
                            'date': row['date'],
                            'ticker': ticker,
                            'returns': returns.iloc[i]
                        })

            return pd.DataFrame(returns_data) if returns_data else pd.DataFrame()
        else:
            logger.warning("No price or returns data available for evaluation")
            return pd.DataFrame()

    def _create_factor_performance_report(self) -> None:
        """Create comprehensive factor performance report."""

        logger.info("Creating factor performance report...")

        # Prepare returns for evaluation
        returns_data = self._prepare_returns_data()

        if returns_data.empty:
            logger.warning("Cannot create performance report without returns data")
            return

        performance_data = []

        # Technical factors performance
        if not self.technical_factors.empty:
            tech_cols = [col for col in self.technical_factors.columns
                        if col not in ['date', 'ticker'] and not col.startswith('forward_return_')]

            for col in tech_cols[:20]:  # Limit to top 20 for performance
                try:
                    # Merge factor with returns
                    factor_data = self.technical_factors[['date', 'ticker', col]].copy()
                    merged = factor_data.merge(returns_data, on=['date', 'ticker'], how='inner')

                    if len(merged) > 20:
                        ic = merged[col].corr(merged['returns'], method='spearman')
                        if pd.notna(ic):
                            performance_data.append({
                                'factor': col,
                                'category': 'technical',
                                'ic': ic,
                                'abs_ic': abs(ic),
                                'observations': len(merged)
                            })
                except Exception as e:
                    continue

        # Macro factors performance
        if not self.macro_factors.empty:
            macro_cols = [col for col in self.macro_factors.columns
                         if col not in ['date', 'ticker']]

            for col in macro_cols[:20]:  # Limit to top 20 for performance
                try:
                    factor_data = self.macro_factors[['date', 'ticker', col]].copy()
                    merged = factor_data.merge(returns_data, on=['date', 'ticker'], how='inner')

                    if len(merged) > 20:
                        ic = merged[col].corr(merged['returns'], method='spearman')
                        if pd.notna(ic):
                            performance_data.append({
                                'factor': col,
                                'category': 'macro',
                                'ic': ic,
                                'abs_ic': abs(ic),
                                'observations': len(merged)
                            })
                except Exception as e:
                    continue

        # Combined factors performance
        if not self.combined_factors.empty:
            for alpha_type in ['technical_alpha', 'macro_alpha', 'combined_alpha']:
                if alpha_type in self.combined_factors.columns:
                    try:
                        factor_data = self.combined_factors[['date', 'ticker', alpha_type]].copy()
                        merged = factor_data.merge(returns_data, on=['date', 'ticker'], how='inner')

                        if len(merged) > 20:
                            ic = merged[alpha_type].corr(merged['returns'], method='spearman')
                            if pd.notna(ic):
                                performance_data.append({
                                    'factor': alpha_type,
                                    'category': 'combined',
                                    'ic': ic,
                                    'abs_ic': abs(ic),
                                    'observations': len(merged)
                                })
                    except Exception as e:
                        continue

        if performance_data:
            self.factor_performance = pd.DataFrame(performance_data)
            self.factor_performance = self.factor_performance.sort_values('abs_ic', ascending=False)

            logger.info(f"Factor performance report created: {len(performance_data)} factors")

            # Log top performers by category
            for category in ['technical', 'macro', 'combined']:
                category_factors = self.factor_performance[
                    self.factor_performance['category'] == category
                ].head(5)

                if not category_factors.empty:
                    logger.info(f"Top {category} factors:")
                    for _, row in category_factors.iterrows():
                        logger.info(f"  {row['factor']}: IC={row['ic']:.4f}")

    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize date column to be timezone-naive."""
        if df is None or df.empty:
            return df

        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        return df

    def get_factor_summary(self) -> Dict:
        """Get comprehensive summary of mined factors."""

        summary = {
            'technical_factors': {
                'shape': self.technical_factors.shape if not self.technical_factors.empty else (0, 0),
                'factors': len([c for c in self.technical_factors.columns
                               if c not in ['date', 'ticker']]) if not self.technical_factors.empty else 0
            },
            'macro_factors': {
                'shape': self.macro_factors.shape if not self.macro_factors.empty else (0, 0),
                'factors': len([c for c in self.macro_factors.columns
                               if c not in ['date', 'ticker']]) if not self.macro_factors.empty else 0
            },
            'combined_factors': {
                'shape': self.combined_factors.shape if not self.combined_factors.empty else (0, 0),
                'method': self.combination_method
            },
            'performance_attribution': self.performance_attribution,
            'top_performers': {}
        }

        # Add top performers by category
        if not self.factor_performance.empty:
            for category in ['technical', 'macro', 'combined']:
                top_factors = self.factor_performance[
                    self.factor_performance['category'] == category
                ].head(5)

                summary['top_performers'][category] = [
                    {
                        'factor': row['factor'],
                        'ic': row['ic'],
                        'observations': row['observations']
                    }
                    for _, row in top_factors.iterrows()
                ]

        return summary


def main():
    """Example usage of EnhancedAlphaMiner."""

    print("=== Enhanced Alpha Miner Demo ===")

    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    data_list = []
    for ticker in tickers:
        for date in dates:
            if np.random.random() > 0.1:  # 90% data availability
                data_list.append({
                    'date': date,
                    'ticker': ticker,
                    'close': 100 * (1 + np.random.normal(0, 0.02)),
                    'volume': np.random.randint(1000000, 10000000),
                    'high': 102,
                    'low': 98,
                    'open': 99
                })

    market_data = pd.DataFrame(data_list)
    market_data['returns'] = market_data.groupby('ticker')['close'].pct_change()

    print(f"Sample market data created: {market_data.shape}")

    # Initialize enhanced miner
    miner = EnhancedAlphaMiner(
        data=market_data,
        min_ic_threshold=0.01,
        combination_method='dynamic_regime'
    )

    # Mine all factors with separation
    combined_factors = miner.mine_all_alpha_factors(separate_analysis=True)

    if not combined_factors.empty:
        print(f"Combined factors shape: {combined_factors.shape}")

        # Get summary
        summary = miner.get_factor_summary()
        print("\nFactor Summary:")
        print(f"Technical factors: {summary['technical_factors']['factors']} factors")
        print(f"Macro factors: {summary['macro_factors']['factors']} factors")
        print(f"Combination method: {summary['combined_factors']['method']}")

        # Show sample combined data
        print("\nSample combined factors:")
        display_cols = ['date', 'ticker', 'technical_alpha', 'macro_alpha', 'combined_alpha']
        available_cols = [c for c in display_cols if c in combined_factors.columns]
        if available_cols:
            print(combined_factors[available_cols].head())

    print("\n=== Enhanced Alpha Miner Demo Completed ===")


if __name__ == "__main__":
    main()