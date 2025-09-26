"""
Alpha Combiner - Technical & Macroeconomic Alpha Integration
Intelligent combination of technical analysis factors and macroeconomic factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AlphaCombiner:
    """
    Advanced alpha factor combination system that intelligently merges
    technical analysis factors and macroeconomic factors.

    Key Features:
    1. Multi-scale factor alignment (high-freq technical + low-freq macro)
    2. Regime-aware dynamic weighting
    3. Correlation-based factor selection
    4. Performance-driven weight adaptation
    5. Risk-adjusted combination strategies
    """

    def __init__(self,
                 lookback_window: int = 252,
                 rebalance_frequency: int = 21,
                 min_ic_threshold: float = 0.02):
        """
        Initialize Alpha Combiner.

        Args:
            lookback_window: Historical window for factor evaluation
            rebalance_frequency: Days between weight updates
            min_ic_threshold: Minimum IC for factor inclusion
        """
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.min_ic_threshold = min_ic_threshold

        # Storage for factors and performance
        self.technical_factors = pd.DataFrame()
        self.macro_factors = pd.DataFrame()
        self.combined_factors = pd.DataFrame()
        self.factor_weights = pd.DataFrame()
        self.performance_metrics = {}

        # Factor categories
        self.technical_categories = [
            'momentum', 'mean_reversion', 'volatility', 'volume',
            'price_action', 'oscillator', 'trend'
        ]

        self.macro_categories = [
            'interest_rates', 'market_environment', 'commodities',
            'currencies', 'regimes', 'correlations', 'composite'
        ]

        logger.info("Alpha Combiner initialized")

    def set_factors(self,
                   technical_factors: pd.DataFrame,
                   macro_factors: pd.DataFrame) -> None:
        """
        Set input factors for combination.

        Args:
            technical_factors: Technical analysis factors
            macro_factors: Macroeconomic factors
        """
        self.technical_factors = self._standardize_factor_df(technical_factors)
        self.macro_factors = self._standardize_factor_df(macro_factors)

        logger.info(f"Technical factors: {self.technical_factors.shape}")
        logger.info(f"Macro factors: {self.macro_factors.shape}")

    def _standardize_factor_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize factor dataframe format."""
        if df.empty:
            return df

        df = df.copy()

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        # Sort by ticker and date
        if 'ticker' in df.columns and 'date' in df.columns:
            df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

        return df

    def combine_factors(self,
                       method: str = 'dynamic_regime',
                       **kwargs) -> pd.DataFrame:
        """
        Combine technical and macroeconomic factors.

        Args:
            method: Combination method
                - 'equal_weight': Simple equal weighting
                - 'ic_weighted': Information coefficient weighted
                - 'dynamic_regime': Regime-aware dynamic weighting
                - 'risk_parity': Risk-adjusted equal contribution
                - 'pca_weighted': Principal component weighted

        Returns:
            DataFrame with combined alpha factors
        """
        if self.technical_factors.empty and self.macro_factors.empty:
            raise ValueError("No factors provided. Use set_factors() first.")

        logger.info(f"Combining factors using method: {method}")

        # Align factors on common dates and tickers
        aligned_technical, aligned_macro = self._align_factors()

        if method == 'equal_weight':
            result = self._combine_equal_weight(aligned_technical, aligned_macro)
        elif method == 'ic_weighted':
            result = self._combine_ic_weighted(aligned_technical, aligned_macro, **kwargs)
        elif method == 'dynamic_regime':
            result = self._combine_dynamic_regime(aligned_technical, aligned_macro, **kwargs)
        elif method == 'risk_parity':
            result = self._combine_risk_parity(aligned_technical, aligned_macro, **kwargs)
        elif method == 'pca_weighted':
            result = self._combine_pca_weighted(aligned_technical, aligned_macro, **kwargs)
        else:
            raise ValueError(f"Unknown combination method: {method}")

        self.combined_factors = result
        logger.info(f"Factor combination completed: {result.shape}")

        return result

    def _align_factors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align technical and macro factors on common dates and tickers."""

        # Handle case where one factor set is empty
        if self.technical_factors.empty:
            return pd.DataFrame(), self.macro_factors
        if self.macro_factors.empty:
            return self.technical_factors, pd.DataFrame()

        # Find common tickers and dates
        tech_tickers = set(self.technical_factors['ticker'].unique())
        macro_tickers = set(self.macro_factors['ticker'].unique())
        common_tickers = tech_tickers.intersection(macro_tickers)

        if not common_tickers:
            logger.warning("No common tickers found. Using union of tickers.")
            common_tickers = tech_tickers.union(macro_tickers)

        # Get date ranges
        tech_dates = set(pd.to_datetime(self.technical_factors['date']).dt.date)
        macro_dates = set(pd.to_datetime(self.macro_factors['date']).dt.date)
        common_dates = tech_dates.intersection(macro_dates)

        if not common_dates:
            logger.warning("No common dates found. Using date union with forward fill.")
            all_dates = sorted(tech_dates.union(macro_dates))
        else:
            all_dates = sorted(common_dates)

        # Align technical factors
        aligned_tech = []
        for ticker in common_tickers:
            if ticker in tech_tickers:
                ticker_tech = self.technical_factors[
                    self.technical_factors['ticker'] == ticker
                ].copy()
            else:
                # Create placeholder for missing ticker
                ticker_tech = pd.DataFrame({
                    'ticker': [ticker] * len(all_dates),
                    'date': all_dates
                })

            # Reindex to all dates with forward fill
            ticker_tech['date'] = pd.to_datetime(ticker_tech['date'])
            ticker_tech = ticker_tech.set_index('date').resample('D').ffill()
            ticker_tech['ticker'] = ticker
            ticker_tech = ticker_tech.reset_index()

            aligned_tech.append(ticker_tech)

        # Align macro factors (typically lower frequency)
        aligned_macro = []
        for ticker in common_tickers:
            if ticker in macro_tickers:
                ticker_macro = self.macro_factors[
                    self.macro_factors['ticker'] == ticker
                ].copy()
            else:
                # Create placeholder for missing ticker
                ticker_macro = pd.DataFrame({
                    'ticker': [ticker] * len(all_dates),
                    'date': all_dates
                })

            # Forward fill macro factors (they update less frequently)
            ticker_macro['date'] = pd.to_datetime(ticker_macro['date'])
            ticker_macro = ticker_macro.set_index('date').resample('D').ffill()
            ticker_macro['ticker'] = ticker
            ticker_macro = ticker_macro.reset_index()

            aligned_macro.append(ticker_macro)

        aligned_technical = pd.concat(aligned_tech, ignore_index=True) if aligned_tech else pd.DataFrame()
        aligned_macro = pd.concat(aligned_macro, ignore_index=True) if aligned_macro else pd.DataFrame()

        logger.info(f"Aligned technical factors: {aligned_technical.shape}")
        logger.info(f"Aligned macro factors: {aligned_macro.shape}")

        return aligned_technical, aligned_macro

    def _combine_equal_weight(self,
                             technical: pd.DataFrame,
                             macro: pd.DataFrame) -> pd.DataFrame:
        """Simple equal weight combination."""

        result_data = []

        # Get unique dates and tickers
        all_dates = set()
        all_tickers = set()

        if not technical.empty:
            all_dates.update(technical['date'])
            all_tickers.update(technical['ticker'])
        if not macro.empty:
            all_dates.update(macro['date'])
            all_tickers.update(macro['ticker'])

        for date in sorted(all_dates):
            for ticker in sorted(all_tickers):
                row_data = {'date': date, 'ticker': ticker}

                # Get technical factors for this date/ticker
                tech_data = technical[
                    (technical['date'] == date) & (technical['ticker'] == ticker)
                ] if not technical.empty else pd.DataFrame()

                macro_data = macro[
                    (macro['date'] == date) & (macro['ticker'] == ticker)
                ] if not macro.empty else pd.DataFrame()

                # Combine technical factors (equal weight)
                tech_score = 0
                tech_count = 0

                if not tech_data.empty:
                    tech_row = tech_data.iloc[0]
                    for col in tech_data.columns:
                        if col not in ['date', 'ticker'] and pd.notna(tech_row[col]):
                            tech_score += float(tech_row[col])
                            tech_count += 1

                if tech_count > 0:
                    row_data['technical_alpha'] = tech_score / tech_count
                else:
                    row_data['technical_alpha'] = 0

                # Combine macro factors (equal weight)
                macro_score = 0
                macro_count = 0

                if not macro_data.empty:
                    macro_row = macro_data.iloc[0]
                    for col in macro_data.columns:
                        if col not in ['date', 'ticker'] and pd.notna(macro_row[col]):
                            try:
                                macro_score += float(macro_row[col])
                                macro_count += 1
                            except (ValueError, TypeError):
                                continue

                if macro_count > 0:
                    row_data['macro_alpha'] = macro_score / macro_count
                else:
                    row_data['macro_alpha'] = 0

                # Combined alpha (50-50 weight)
                row_data['combined_alpha'] = (
                    0.5 * row_data['technical_alpha'] +
                    0.5 * row_data['macro_alpha']
                )

                result_data.append(row_data)

        return pd.DataFrame(result_data)

    def _combine_ic_weighted(self,
                           technical: pd.DataFrame,
                           macro: pd.DataFrame,
                           returns_col: str = 'forward_return_1d') -> pd.DataFrame:
        """Information coefficient weighted combination."""

        # Calculate IC for technical factors
        tech_weights = self._calculate_factor_ics(technical, returns_col)
        macro_weights = self._calculate_factor_ics(macro, returns_col)

        # Normalize weights
        total_tech_ic = sum(abs(ic) for ic in tech_weights.values())
        total_macro_ic = sum(abs(ic) for ic in macro_weights.values())

        if total_tech_ic + total_macro_ic == 0:
            logger.warning("No significant ICs found, falling back to equal weight")
            return self._combine_equal_weight(technical, macro)

        # Calculate category weights
        tech_weight = total_tech_ic / (total_tech_ic + total_macro_ic)
        macro_weight = total_macro_ic / (total_tech_ic + total_macro_ic)

        logger.info(f"IC-based weights: Technical={tech_weight:.3f}, Macro={macro_weight:.3f}")

        result_data = []

        # Combine using IC weights
        all_dates = set()
        all_tickers = set()

        if not technical.empty:
            all_dates.update(technical['date'])
            all_tickers.update(technical['ticker'])
        if not macro.empty:
            all_dates.update(macro['date'])
            all_tickers.update(macro['ticker'])

        for date in sorted(all_dates):
            for ticker in sorted(all_tickers):
                row_data = {'date': date, 'ticker': ticker}

                # Technical alpha with IC weighting
                tech_alpha = self._calculate_weighted_alpha(
                    technical, date, ticker, tech_weights
                )

                # Macro alpha with IC weighting
                macro_alpha = self._calculate_weighted_alpha(
                    macro, date, ticker, macro_weights
                )

                row_data['technical_alpha'] = tech_alpha
                row_data['macro_alpha'] = macro_alpha
                row_data['combined_alpha'] = (
                    tech_weight * tech_alpha + macro_weight * macro_alpha
                )

                result_data.append(row_data)

        return pd.DataFrame(result_data)

    def _combine_dynamic_regime(self,
                              technical: pd.DataFrame,
                              macro: pd.DataFrame,
                              **kwargs) -> pd.DataFrame:
        """Dynamic regime-aware combination."""

        # Identify market regimes
        regimes = self._identify_market_regimes(technical, macro)

        result_data = []

        all_dates = set()
        all_tickers = set()

        if not technical.empty:
            all_dates.update(technical['date'])
            all_tickers.update(technical['ticker'])
        if not macro.empty:
            all_dates.update(macro['date'])
            all_tickers.update(macro['ticker'])

        for date in sorted(all_dates):
            # Get regime for this date
            current_regime = regimes.get(date, 'normal')

            # Adjust weights based on regime
            if current_regime == 'high_volatility':
                # In volatile markets, macro factors may be more important
                tech_weight, macro_weight = 0.3, 0.7
            elif current_regime == 'trending':
                # In trending markets, technical factors may dominate
                tech_weight, macro_weight = 0.7, 0.3
            elif current_regime == 'mean_reverting':
                # In mean-reverting markets, balance both
                tech_weight, macro_weight = 0.6, 0.4
            else:  # normal regime
                tech_weight, macro_weight = 0.5, 0.5

            for ticker in sorted(all_tickers):
                row_data = {'date': date, 'ticker': ticker}

                # Get factor values
                tech_data = technical[
                    (technical['date'] == date) & (technical['ticker'] == ticker)
                ] if not technical.empty else pd.DataFrame()

                macro_data = macro[
                    (macro['date'] == date) & (macro['ticker'] == ticker)
                ] if not macro.empty else pd.DataFrame()

                # Calculate alphas
                tech_alpha = self._calculate_simple_alpha(tech_data)
                macro_alpha = self._calculate_simple_alpha(macro_data)

                row_data['technical_alpha'] = tech_alpha
                row_data['macro_alpha'] = macro_alpha
                row_data['regime'] = current_regime
                row_data['tech_weight'] = tech_weight
                row_data['macro_weight'] = macro_weight
                row_data['combined_alpha'] = (
                    tech_weight * tech_alpha + macro_weight * macro_alpha
                )

                result_data.append(row_data)

        return pd.DataFrame(result_data)

    def _combine_risk_parity(self,
                           technical: pd.DataFrame,
                           macro: pd.DataFrame,
                           **kwargs) -> pd.DataFrame:
        """Risk parity combination - equal risk contribution."""

        # Calculate volatility of each factor category
        tech_vol = self._calculate_factor_volatility(technical)
        macro_vol = self._calculate_factor_volatility(macro)

        if tech_vol == 0 and macro_vol == 0:
            return self._combine_equal_weight(technical, macro)

        # Inverse volatility weighting for equal risk contribution
        tech_weight = (1 / tech_vol) if tech_vol > 0 else 0
        macro_weight = (1 / macro_vol) if macro_vol > 0 else 0

        total_weight = tech_weight + macro_weight
        if total_weight > 0:
            tech_weight /= total_weight
            macro_weight /= total_weight
        else:
            tech_weight = macro_weight = 0.5

        logger.info(f"Risk parity weights: Technical={tech_weight:.3f}, Macro={macro_weight:.3f}")

        return self._combine_with_fixed_weights(technical, macro, tech_weight, macro_weight)

    def _combine_pca_weighted(self,
                            technical: pd.DataFrame,
                            macro: pd.DataFrame,
                            n_components: int = 3) -> pd.DataFrame:
        """PCA-based combination to reduce dimensionality."""

        # Combine all factors for PCA
        all_factors = self._merge_factor_dataframes(technical, macro)

        if all_factors.empty:
            return pd.DataFrame()

        # Extract numeric columns
        numeric_cols = [col for col in all_factors.columns
                       if col not in ['date', 'ticker'] and
                       all_factors[col].dtype in [np.float64, np.float32, int]]

        if len(numeric_cols) < n_components:
            logger.warning(f"Not enough factors for PCA, using equal weight")
            return self._combine_equal_weight(technical, macro)

        # Apply PCA
        factor_matrix = all_factors[numeric_cols].fillna(0)
        scaler = StandardScaler()
        factor_matrix_scaled = scaler.fit_transform(factor_matrix)

        pca = PCA(n_components=n_components)
        pca_factors = pca.fit_transform(factor_matrix_scaled)

        # Create result with PCA components
        result = all_factors[['date', 'ticker']].copy()
        for i in range(n_components):
            result[f'pca_factor_{i+1}'] = pca_factors[:, i]

        # Combined alpha as weighted sum of PCA components
        explained_variance = pca.explained_variance_ratio_
        result['combined_alpha'] = np.sum([
            explained_variance[i] * pca_factors[:, i]
            for i in range(n_components)
        ], axis=0)

        logger.info(f"PCA explained variance: {explained_variance}")

        return result

    def _calculate_factor_ics(self,
                            factors: pd.DataFrame,
                            returns_col: str) -> Dict[str, float]:
        """Calculate information coefficients for factors."""
        ics = {}

        if factors.empty or returns_col not in factors.columns:
            return ics

        numeric_cols = [col for col in factors.columns
                       if col not in ['date', 'ticker', returns_col] and
                       factors[col].dtype in [np.float64, np.float32, int]]

        for col in numeric_cols:
            factor_data = factors[[col, returns_col]].dropna()
            if len(factor_data) > 20:
                ic, _ = spearmanr(factor_data[col], factor_data[returns_col])
                if not np.isnan(ic):
                    ics[col] = ic

        return ics

    def _calculate_weighted_alpha(self,
                                factors: pd.DataFrame,
                                date,
                                ticker: str,
                                weights: Dict[str, float]) -> float:
        """Calculate weighted alpha for a specific date/ticker."""

        if factors.empty:
            return 0.0

        data = factors[
            (factors['date'] == date) & (factors['ticker'] == ticker)
        ]

        if data.empty:
            return 0.0

        row = data.iloc[0]
        weighted_sum = 0.0
        total_weight = 0.0

        for col, weight in weights.items():
            if col in row and pd.notna(row[col]):
                try:
                    weighted_sum += abs(weight) * float(row[col])
                    total_weight += abs(weight)
                except (ValueError, TypeError):
                    continue

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_simple_alpha(self, data: pd.DataFrame) -> float:
        """Calculate simple average alpha from factor data."""

        if data.empty:
            return 0.0

        row = data.iloc[0]
        values = []

        for col in data.columns:
            if col not in ['date', 'ticker'] and pd.notna(row[col]):
                try:
                    values.append(float(row[col]))
                except (ValueError, TypeError):
                    continue

        return np.mean(values) if values else 0.0

    def _identify_market_regimes(self,
                               technical: pd.DataFrame,
                               macro: pd.DataFrame) -> Dict:
        """Identify market regimes for dynamic weighting."""

        regimes = {}

        # Simple regime identification based on volatility and trends
        if not technical.empty and 'date' in technical.columns:
            dates = sorted(technical['date'].unique())

            for date in dates:
                # Default to normal regime
                regime = 'normal'

                # Check for high volatility indicators in technical factors
                date_data = technical[technical['date'] == date]

                if not date_data.empty:
                    # Look for volatility indicators
                    vol_cols = [col for col in date_data.columns if 'vol' in col.lower()]
                    if vol_cols:
                        avg_vol = date_data[vol_cols].mean().mean()
                        if avg_vol > 0.02:  # Threshold for high volatility
                            regime = 'high_volatility'

                    # Look for momentum indicators
                    mom_cols = [col for col in date_data.columns if 'momentum' in col.lower()]
                    if mom_cols:
                        avg_mom = date_data[mom_cols].mean().mean()
                        if avg_mom > 0.01:
                            regime = 'trending'
                        elif avg_mom < -0.01:
                            regime = 'mean_reverting'

                regimes[date] = regime

        return regimes

    def _calculate_factor_volatility(self, factors: pd.DataFrame) -> float:
        """Calculate average volatility of factors."""

        if factors.empty:
            return 0.0

        numeric_cols = [col for col in factors.columns
                       if col not in ['date', 'ticker'] and
                       factors[col].dtype in [np.float64, np.float32, int]]

        if not numeric_cols:
            return 0.0

        volatilities = []
        for col in numeric_cols:
            col_data = factors[col].dropna()
            if len(col_data) > 2:
                vol = col_data.std()
                if not np.isnan(vol):
                    volatilities.append(vol)

        return np.mean(volatilities) if volatilities else 0.0

    def _combine_with_fixed_weights(self,
                                  technical: pd.DataFrame,
                                  macro: pd.DataFrame,
                                  tech_weight: float,
                                  macro_weight: float) -> pd.DataFrame:
        """Combine factors with fixed weights."""

        result_data = []

        all_dates = set()
        all_tickers = set()

        if not technical.empty:
            all_dates.update(technical['date'])
            all_tickers.update(technical['ticker'])
        if not macro.empty:
            all_dates.update(macro['date'])
            all_tickers.update(macro['ticker'])

        for date in sorted(all_dates):
            for ticker in sorted(all_tickers):
                row_data = {'date': date, 'ticker': ticker}

                # Get data
                tech_data = technical[
                    (technical['date'] == date) & (technical['ticker'] == ticker)
                ] if not technical.empty else pd.DataFrame()

                macro_data = macro[
                    (macro['date'] == date) & (macro['ticker'] == ticker)
                ] if not macro.empty else pd.DataFrame()

                # Calculate alphas
                tech_alpha = self._calculate_simple_alpha(tech_data)
                macro_alpha = self._calculate_simple_alpha(macro_data)

                row_data['technical_alpha'] = tech_alpha
                row_data['macro_alpha'] = macro_alpha
                row_data['combined_alpha'] = (
                    tech_weight * tech_alpha + macro_weight * macro_alpha
                )

                result_data.append(row_data)

        return pd.DataFrame(result_data)

    def _merge_factor_dataframes(self,
                               technical: pd.DataFrame,
                               macro: pd.DataFrame) -> pd.DataFrame:
        """Merge technical and macro factor dataframes."""

        if technical.empty and macro.empty:
            return pd.DataFrame()
        elif technical.empty:
            return macro.copy()
        elif macro.empty:
            return technical.copy()

        # Merge on date and ticker
        merged = technical.merge(
            macro,
            on=['date', 'ticker'],
            how='outer',
            suffixes=('_tech', '_macro')
        )

        return merged

    def evaluate_combination_performance(self,
                                       combined_factors: pd.DataFrame,
                                       returns_data: pd.DataFrame) -> Dict:
        """Evaluate the performance of factor combination."""

        if combined_factors.empty:
            return {}

        # Merge with returns
        perf_data = combined_factors.merge(
            returns_data,
            on=['date', 'ticker'],
            how='inner'
        )

        if perf_data.empty:
            return {}

        metrics = {}

        # Calculate IC for combined alpha
        if 'combined_alpha' in perf_data.columns and 'returns' in perf_data.columns:
            ic_data = perf_data[['combined_alpha', 'returns']].dropna()
            if len(ic_data) > 20:
                ic, p_value = spearmanr(ic_data['combined_alpha'], ic_data['returns'])
                metrics['combined_ic'] = ic if not np.isnan(ic) else 0
                metrics['ic_p_value'] = p_value if not np.isnan(p_value) else 1

                # IC information ratio
                ic_std = ic_data['combined_alpha'].rolling(21).corr(ic_data['returns']).std()
                metrics['ic_ir'] = ic / ic_std if ic_std > 0 else 0

        # Calculate individual component ICs
        for alpha_type in ['technical_alpha', 'macro_alpha']:
            if alpha_type in perf_data.columns:
                ic_data = perf_data[[alpha_type, 'returns']].dropna()
                if len(ic_data) > 20:
                    ic, _ = spearmanr(ic_data[alpha_type], ic_data['returns'])
                    metrics[f'{alpha_type}_ic'] = ic if not np.isnan(ic) else 0

        # Calculate factor decay
        for horizon in [1, 5, 10, 20]:
            returns_col = f'forward_return_{horizon}d'
            if returns_col in perf_data.columns:
                decay_data = perf_data[['combined_alpha', returns_col]].dropna()
                if len(decay_data) > 20:
                    ic, _ = spearmanr(decay_data['combined_alpha'], decay_data[returns_col])
                    metrics[f'ic_{horizon}d'] = ic if not np.isnan(ic) else 0

        self.performance_metrics = metrics
        logger.info(f"Performance evaluation completed: {len(metrics)} metrics")

        return metrics


def main():
    """Example usage of AlphaCombiner."""

    print("=== Alpha Combiner Example ===")

    # Create sample technical factors
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    tech_data = []
    for ticker in tickers:
        for date in dates[:100]:  # Use 100 days for demo
            tech_data.append({
                'date': date,
                'ticker': ticker,
                'momentum_10d': np.random.normal(0, 0.02),
                'rsi': np.random.uniform(20, 80),
                'volatility_21d': np.random.uniform(0.1, 0.5),
                'forward_return_1d': np.random.normal(0, 0.02)
            })

    technical_factors = pd.DataFrame(tech_data)

    # Create sample macro factors (lower frequency)
    macro_data = []
    for ticker in tickers:
        for i, date in enumerate(dates[:100:5]):  # Every 5 days
            macro_data.append({
                'date': date,
                'ticker': ticker,
                'treasury_10y_level': 4.0 + np.random.normal(0, 0.5),
                'vix_level': 20 + np.random.normal(0, 5),
                'market_beta': 1.0 + np.random.normal(0, 0.3),
                'forward_return_1d': np.random.normal(0, 0.02)
            })

    macro_factors = pd.DataFrame(macro_data)

    print(f"Technical factors shape: {technical_factors.shape}")
    print(f"Macro factors shape: {macro_factors.shape}")

    # Initialize combiner
    combiner = AlphaCombiner()
    combiner.set_factors(technical_factors, macro_factors)

    # Test different combination methods
    methods = ['equal_weight', 'ic_weighted', 'dynamic_regime', 'risk_parity']

    for method in methods:
        print(f"\n--- Testing {method} method ---")
        try:
            combined = combiner.combine_factors(method=method)
            print(f"Combined factors shape: {combined.shape}")

            if not combined.empty:
                print("Sample combined data:")
                print(combined[['date', 'ticker', 'technical_alpha', 'macro_alpha', 'combined_alpha']].head())

                # Evaluate performance if returns available
                returns_data = technical_factors[['date', 'ticker', 'forward_return_1d']].rename(
                    columns={'forward_return_1d': 'returns'}
                )

                metrics = combiner.evaluate_combination_performance(combined, returns_data)
                if metrics:
                    print(f"Performance metrics: {metrics}")

        except Exception as e:
            print(f"Error with {method}: {e}")

    print("\n=== Alpha Combiner Demo Completed ===")


if __name__ == "__main__":
    main()