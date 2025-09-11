import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RealBetaEstimator:
    """
    Simplified Real Beta Estimator
    Provides CAPM and multi-factor style beta estimates.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        risk_model_type: str = "multi_factor",
        estimation_window: int = 252,   # ✅ Keep compatible
        min_observations: int = 50,
    ):
        """
        Args:
            data: dict with keys ['stock_returns', 'factor_returns']
            risk_model_type: type of risk model
            estimation_window: rolling window (keep parameter for now, not enforced)
            min_observations: minimum observations
        """
        self.data = data
        self.risk_model_type = risk_model_type
        self.estimation_window = estimation_window
        self.min_observations = min_observations

        # Validate data
        self.stock_returns = data.get("stock_returns", pd.DataFrame()).dropna(how="all")
        self.factor_returns = data.get("factor_returns", pd.DataFrame()).dropna(how="all")

        if self.stock_returns.empty:
            logger.warning("Stock returns are empty!")
        if self.factor_returns.empty:
            logger.warning("Factor returns are empty!")

        # Align indices
        if not self.stock_returns.empty and not self.factor_returns.empty:
            common_idx = self.stock_returns.index.intersection(self.factor_returns.index)
            self.stock_returns = self.stock_returns.loc[common_idx]
            self.factor_returns = self.factor_returns.loc[common_idx]

        logger.info(
            f"Prepared data: stocks {self.stock_returns.shape}, factors {self.factor_returns.shape}"
        )
        logger.info(f"Beta estimator initialized with {self.stock_returns.shape[1]} stocks")

    # ============================================================
    def estimate_all_betas(self) -> Dict[str, pd.DataFrame]:
        """Run all available beta estimators"""
        logger.info("Estimating betas...")

        results = {}
        results["capm_beta"] = self._estimate_capm_betas()
        results["multi_factor_beta"] = self._estimate_multi_factor_betas()

        # Copula version placeholder for now
        results["copula_beta"] = pd.DataFrame(
            {"ticker": self.stock_returns.columns, "copula_dependence": np.nan}
        )

        logger.info(f"✅ Beta estimation completed: {list(results.keys())}")
        return results

    # ============================================================
    def _estimate_capm_betas(self) -> pd.DataFrame:
        """Single-factor CAPM beta"""
        if self.factor_returns.empty:
            logger.warning("No factor returns available, cannot estimate CAPM betas")
            return pd.DataFrame(columns=["ticker", "beta", "r_squared"])

        market = self.factor_returns.iloc[:, 0]  # First factor as market factor
        betas = []
        for ticker in self.stock_returns.columns:
            y = self.stock_returns[ticker].dropna()
            common_idx = y.index.intersection(market.index)
            if len(common_idx) < self.min_observations:
                betas.append([ticker, np.nan, np.nan])
                continue
            x = market.loc[common_idx]
            y = y.loc[common_idx]

            if x.std() == 0:
                betas.append([ticker, np.nan, np.nan])
                continue

            beta = np.cov(y, x)[0, 1] / np.var(x)
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            betas.append([ticker, beta, r2])

        return pd.DataFrame(betas, columns=["ticker", "beta", "r_squared"])

    def _estimate_multi_factor_betas(self) -> pd.DataFrame:
        """Multi-factor regression"""
        if self.factor_returns.empty:
            logger.warning("No factor returns available, cannot estimate multi-factor betas")
            return pd.DataFrame()

        results = []
        X = self.factor_returns.values
        X = np.column_stack([np.ones(len(X)), X])  # Add constant term

        for ticker in self.stock_returns.columns:
            y = self.stock_returns[ticker].dropna()
            common_idx = y.index.intersection(self.factor_returns.index)
            if len(common_idx) < self.min_observations:
                results.append({"ticker": ticker})
                continue

            y = y.loc[common_idx].values
            X_common = self.factor_returns.loc[common_idx].values
            X_common = np.column_stack([np.ones(len(X_common)), X_common])

            try:
                coeffs = np.linalg.lstsq(X_common, y, rcond=None)[0]
                res = {"ticker": ticker, "alpha": coeffs[0]}
                for i, col in enumerate(self.factor_returns.columns):
                    res[f"beta_{col}"] = coeffs[i + 1]
                results.append(res)
            except Exception as e:
                logger.warning(f"Multi-factor regression failed for {ticker}: {e}")
                results.append({"ticker": ticker})

        return pd.DataFrame(results)






