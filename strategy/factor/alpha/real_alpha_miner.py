"""
Real Alpha Factor Miner — timezone-safe, no circular imports
Comprehensive alpha factor mining from real market data with validation.

Fixes included:
- Unifies all `date` columns to naive `datetime64[ns]` (no timezone)
- Avoids NameError by initializing `factors_list` in each miner
- Uses safe merges and de-duplicates columns
- Keeps logging compact but informative
"""

from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb  # type: ignore
    ADVANCED_ML_AVAILABLE = True
except Exception:
    ADVANCED_ML_AVAILABLE = False


# ---------------------------- utils ---------------------------- #

def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['date'] is timezone-naive for safe joins.
    No-op if column missing.
    """
    if df is None or df.empty:
        return df
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


# ------------------------- main miner -------------------------- #

class RealAlphaMiner:
    """
    Mine alpha factors from real market data with comprehensive validation.

    Pipeline:
      1) Feature engineering from price/volume/fundamental data
      2) Factor validation with IC analysis, turnover, decay, coverage
      3) (Optional) ML predictions as factors
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_windows: List[int] | None = None,
        prediction_horizons: List[int] | None = None,
        validation_method: str = "ic_analysis",
        min_ic_threshold: float = 0.02,
    ) -> None:
        self.feature_windows = feature_windows or [5, 10, 20, 60]
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 20]
        self.validation_method = validation_method
        self.min_ic_threshold = float(min_ic_threshold)

        # base data
        self.data = _normalize_dates(data.copy())
        self.data = self.data.sort_values(["ticker", "date"]).reset_index(drop=True)

        # storage
        self.factor_performance: pd.DataFrame | Dict = {}

        logger.info(
            f"Alpha miner initialized with {len(self.data)} observations"
        )
        logger.info(
            f"Date range: {self.data['date'].min()} to {self.data['date'].max()}"
        )
        logger.info(f"Tickers: {self.data['ticker'].nunique()}")

    # ------------------------ public API ----------------------- #

    def mine_all_alpha_factors(self) -> pd.DataFrame:
        logger.info("Starting comprehensive alpha factor mining...")

        technical = _normalize_dates(self._mine_technical_factors())
        fundamental = _normalize_dates(self._mine_fundamental_factors())
        micro = _normalize_dates(self._mine_microstructure_factors())
        crosssec = _normalize_dates(self._mine_cross_sectional_factors())
        ml = _normalize_dates(self._mine_ml_factors())
        alt = _normalize_dates(self._mine_alternative_factors())
        macro = _normalize_dates(self._mine_macro_factors())

        # Safe horizontal concat; some may be empty
        parts = [df for df in [technical, fundamental, micro, crosssec, ml, alt, macro] if df is not None and not df.empty]
        if not parts:
            raise ValueError("No factors produced — check input data.")

        all_factors = pd.concat(parts, axis=1)
        # drop duplicate columns from multiple joins
        all_factors = all_factors.loc[:, ~all_factors.columns.duplicated()]

        validated = self._validate_all_factors(all_factors)
        logger.info(f"Alpha mining completed: {validated.shape[1]} validated factors")
        return validated

    # --------------------- factor miners ----------------------- #

    def _mine_technical_factors(self) -> pd.DataFrame:
        logger.info("Mining technical analysis factors...")
        factors_list: List[pd.DataFrame] = []

        for ticker in self.data["ticker"].unique():
            td = self.data[self.data["ticker"] == ticker].copy().sort_values("date")
            if len(td) < max(self.feature_windows) + max(self.prediction_horizons):
                continue

            out = pd.DataFrame(index=td.index)
            out["ticker"] = ticker
            out["date"] = td["date"].values

            close = td["close"].astype(float).values
            volume = td["volume"].astype(float).values if "volume" in td.columns else None

            for w in self.feature_windows:
                if len(close) <= w:
                    continue
                s_close = pd.Series(close)
                sma = s_close.rolling(w).mean()
                ema = s_close.ewm(span=w).mean()
                out[f"price_to_sma_{w}"] = close / sma - 1
                out[f"price_to_ema_{w}"] = close / ema - 1
                out[f"sma_ema_spread_{w}"] = (sma - ema) / (ema + 1e-12)

                mom = s_close.pct_change(w)
                out[f"momentum_{w}"] = mom

                vol = s_close.pct_change().rolling(w).std()
                out[f"vol_adj_momentum_{w}"] = mom / (vol + 1e-8)

                rstd = s_close.rolling(w).std()
                bb_up = sma + 2 * rstd
                bb_lo = sma - 2 * rstd
                out[f"bb_position_{w}"] = (s_close - bb_lo) / (bb_up - bb_lo + 1e-12)

                rmax = s_close.rolling(w).max()
                rmin = s_close.rolling(w).min()
                out[f"channel_position_{w}"] = (s_close - rmin) / (rmax - rmin + 1e-12)

                if w >= 14:
                    rsi = self._calculate_rsi(s_close, w)
                    out[f"rsi_{w}"] = rsi
                    out[f"rsi_divergence_{w}"] = rsi - 50

                if volume is not None:
                    s_vol = pd.Series(volume)
                    vma = s_vol.rolling(w).mean()
                    out[f"volume_ratio_{w}"] = s_vol / (vma + 1e-12)
                    pr = s_close.pct_change()
                    vc = s_vol.pct_change()
                    out[f"price_volume_corr_{w}"] = pr.rolling(w).corr(vc)

            # MACD
            macd_line, macd_sig = self._calculate_macd(pd.Series(close))
            out["macd"] = macd_line - macd_sig
            out["macd_signal"] = macd_sig

            # Stoch
            if len(td) >= 14:
                k, d = self._calculate_stochastic(td)
                out["stochastic_k"], out["stochastic_d"] = k, d
                out["stochastic_divergence"] = k - d

            factors_list.append(out)

        if not factors_list:
            return pd.DataFrame()
        df = pd.concat(factors_list, ignore_index=True)
        return _normalize_dates(df)

    def _mine_fundamental_factors(self) -> pd.DataFrame:
        logger.info("Mining fundamental factors...")
        # detect availability
        fundamental_cols = [c for c in self.data.columns if c.startswith("fundamental_") or c in {"market_cap", "pe_ratio", "dividend_yield"}]
        if not fundamental_cols:
            logger.warning("No fundamental data available, creating placeholder factors")
            return pd.DataFrame(index=self.data.index)

        factors_list: List[pd.DataFrame] = []
        for ticker in self.data["ticker"].unique():
            td = self.data[self.data["ticker"] == ticker].copy()
            out = pd.DataFrame(index=td.index)
            out["ticker"], out["date"] = ticker, td["date"].values

            if "fundamental_pe_ratio" in td.columns:
                pe = td["fundamental_pe_ratio"].astype(float)
                out["pe_percentile"] = pe.rolling(252, min_periods=5).rank(pct=True)
                out["pe_change"] = pe.pct_change(20)

            if "fundamental_revenue_growth" in td.columns:
                rg = td["fundamental_revenue_growth"].astype(float)
                out["revenue_growth"] = rg
                out["revenue_growth_stability"] = rg.rolling(60, min_periods=10).std()

            if "fundamental_dividend_yield" in td.columns:
                dy = td["fundamental_dividend_yield"].astype(float)
                out["dividend_yield"] = dy
                out["dividend_yield_change"] = dy.diff(20)

            if "fundamental_market_cap" in td.columns:
                mc = td["fundamental_market_cap"].astype(float)
                out["log_market_cap"] = np.log(mc + 1.0)
                out["market_cap_change"] = mc.pct_change(20)

            factors_list.append(out)

        if not factors_list:
            return pd.DataFrame(index=self.data.index)
        df = pd.concat(factors_list, ignore_index=True)
        logger.info(f"Fundamental factors created: {df.shape}")
        return _normalize_dates(df)

    def _mine_microstructure_factors(self) -> pd.DataFrame:
        logger.info("Mining market microstructure factors...")
        factors_list: List[pd.DataFrame] = []

        for ticker in self.data["ticker"].unique():
            td = self.data[self.data["ticker"] == ticker].copy()
            if len(td) < 60:
                continue

            out = pd.DataFrame(index=td.index)
            out["ticker"], out["date"] = ticker, td["date"].values

            close = td["close"].astype(float)
            returns = close.pct_change()

            for w in [5, 10, 20]:
                out[f"return_autocorr_{w}"] = returns.rolling(w).apply(
                    lambda x: x.autocorr() if len(pd.Series(x).dropna()) > 5 else np.nan
                )
                rstd = returns.rolling(w).std()
                out[f"return_jumps_{w}"] = (returns.abs() > 3 * rstd).astype(int)
                out[f"vol_clustering_{w}"] = rstd / (rstd.rolling(w * 2).mean() + 1e-12)

            if "volume" in td.columns and not td["volume"].isna().all():
                vol = td["volume"].astype(float)
                illiq = returns.abs() / (vol + 1.0)
                for w in [5, 20]:
                    out[f"amihud_illiquidity_{w}"] = illiq.rolling(w).mean()

                vma = vol.rolling(20).mean()
                vratio = vol / (vma + 1e-12)
                out["volume_impact"] = returns * vratio

                for w in [20, 60]:
                    r = returns.rolling(w)
                    if r.count().iloc[-1] >= 3:
                        out[f"kyle_lambda_{w}"] = r.std() / np.sqrt((vol.rolling(w).mean() + 1e-12))

            if {"high", "low", "close"}.issubset(td.columns):
                spread_proxy = (td["high"] - td["low"]) / (td["close"] + 1e-12)
                for w in [5, 20]:
                    out[f"spread_proxy_{w}"] = spread_proxy.rolling(w).mean()

            factors_list.append(out)

        if not factors_list:
            return pd.DataFrame(index=self.data.index)
        df = pd.concat(factors_list, ignore_index=True)
        logger.info(f"Microstructure factors created: {df.shape}")
        return _normalize_dates(df)

    def _mine_cross_sectional_factors(self) -> pd.DataFrame:
        logger.info("Mining cross-sectional factors...")
        factors_list: List[pd.DataFrame] = []

        for date in self.data["date"].unique():
            dd = self.data[self.data["date"] == date].copy()
            if len(dd) < 10:
                continue

            # momentum-based cross-sectional ranks
            for w in [5, 20, 60]:
                start = pd.to_datetime(date) - timedelta(days=w * 2)
                hist = self.data[(self.data["date"] >= start) & (self.data["date"] <= date)]
                if hist.empty:
                    continue
                mom_map: Dict[str, float] = {}
                for t in dd["ticker"].unique():
                    ts = hist[hist["ticker"] == t]["close"].astype(float)
                    if len(ts) >= 2:
                        mom_map[t] = ts.iloc[-1] / (ts.iloc[0] + 1e-12) - 1.0
                if not mom_map:
                    continue
                mser = pd.Series(mom_map)
                ranks = mser.rank(pct=True)
                z = (mser - mser.mean()) / (mser.std() + 1e-12)
                for t in dd["ticker"].unique():
                    if t in ranks:
                        dd.loc[dd["ticker"] == t, f"cs_momentum_rank_{w}"] = ranks[t]
                        dd.loc[dd["ticker"] == t, f"cs_momentum_zscore_{w}"] = z[t]

            factors_list.append(dd)

        if not factors_list:
            return pd.DataFrame(index=self.data.index)
        cross = pd.concat(factors_list, ignore_index=True)
        cross = _normalize_dates(cross)
        cs_cols = [c for c in cross.columns if c.startswith("cs_")]
        if cs_cols:
            res = cross[["ticker", "date"] + cs_cols].copy()
        else:
            res = pd.DataFrame(index=cross.index)
        logger.info(f"Cross-sectional factors created: {res.shape}")
        return res

    def _mine_ml_factors(self) -> pd.DataFrame:
        logger.info("Mining machine learning factors...")
        factors_list: List[pd.DataFrame] = []

        for ticker in self.data["ticker"].unique():
            td = self.data[self.data["ticker"] == ticker].copy().sort_values("date")
            if len(td) < 100:
                continue

            out = pd.DataFrame(index=td.index)
            out["ticker"], out["date"] = ticker, td["date"].values

            close = td["close"].astype(float)
            rets = close.pct_change()
            vol = td["volume"].astype(float) if "volume" in td.columns else pd.Series(index=td.index, dtype=float)

            feats = pd.DataFrame(index=td.index)
            for w in [5, 10, 20]:
                feats[f"return_mean_{w}"] = rets.rolling(w).mean()
                feats[f"return_std_{w}"] = rets.rolling(w).std()
                feats[f"momentum_{w}"] = close.pct_change(w)
                if not vol.isna().all():
                    feats[f"volume_ma_{w}"] = vol.rolling(w).mean()
                    feats[f"volume_std_{w}"] = vol.rolling(w).std()

            for w in [10, 20]:
                sma = close.rolling(w).mean()
                feats[f"price_to_sma_{w}"] = close / (sma + 1e-12) - 1
                rmax = close.rolling(w).max()
                rmin = close.rolling(w).min()
                feats[f"price_position_{w}"] = (close - rmin) / (rmax - rmin + 1e-12)

            for lag in [1, 2, 5]:
                feats[f"return_lag_{lag}"] = rets.shift(lag)

            for horizon in self.prediction_horizons:
                tgt = f"target_{horizon}d"
                feats[tgt] = rets.shift(-horizon)
                feature_cols = [c for c in feats.columns if not c.startswith("target_")]
                clean = feats[feature_cols + [tgt]].dropna()
                if len(clean) < 50:
                    continue

                X = clean[feature_cols]
                y = clean[tgt]
                trn = int(0.7 * len(X))
                if trn < 20:
                    continue

                scaler = RobustScaler()
                X_tr = scaler.fit_transform(X.iloc[:trn])
                X_all = scaler.transform(X)

                models = {
                    "ridge": Ridge(alpha=1.0, random_state=42),
                    "lasso": Lasso(alpha=0.01, random_state=42),
                    "rf": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
                }
                if ADVANCED_ML_AVAILABLE:
                    models["xgb"] = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)

                for mname, model in models.items():
                    try:
                        if mname == "xgb" and ADVANCED_ML_AVAILABLE:
                            model.fit(X.iloc[:trn], y.iloc[:trn])
                            pred = model.predict(X)
                        else:
                            model.fit(X_tr, y.iloc[:trn])
                            pred = model.predict(X_all)
                        s = pd.Series(np.nan, index=out.index)
                        s.loc[clean.index] = pred
                        out[f"ml_{mname}_pred_{horizon}d"] = s
                    except Exception as e:
                        logger.warning(f"Model {mname} failed for {ticker}, horizon {horizon}: {e}")
                        continue

            factors_list.append(out)

        if not factors_list:
            return pd.DataFrame(index=self.data.index)
        df = pd.concat(factors_list, ignore_index=True)
        logger.info(f"ML factors created: {df.shape}")
        return _normalize_dates(df)

    def _mine_alternative_factors(self) -> pd.DataFrame:
        logger.info("Mining alternative data factors...")
        factors_list: List[pd.DataFrame] = []

        # market-wide aggregates
        grp = self.data.groupby("date").agg({"close": "mean", "volume": ("sum" if "volume" in self.data.columns else "count")}).reset_index()
        mret = grp["close"].pct_change()
        mvol = mret.rolling(20).std()
        hi, lo = mvol.quantile(0.75), mvol.quantile(0.25)
        grp["vol_regime"] = 0
        grp.loc[mvol > hi, "vol_regime"] = 1
        grp.loc[mvol < lo, "vol_regime"] = -1
        grp["market_stress"] = (mvol - mvol.rolling(60).mean()) / (mvol.rolling(60).std() + 1e-12)

        enhanced = self.data.merge(_normalize_dates(grp[["date", "vol_regime", "market_stress"]]), on="date", how="left")

        for ticker in enhanced["ticker"].unique():
            td = enhanced[enhanced["ticker"] == ticker].copy()
            out = pd.DataFrame(index=td.index)
            out["ticker"], out["date"] = ticker, td["date"].values
            out["vol_regime"] = td["vol_regime"].values
            out["market_stress"] = td["market_stress"].values
            out["regime_persistence"] = (td["vol_regime"] == td["vol_regime"].shift(1)).astype(int)

            rets = td["close"].astype(float).pct_change()
            for w in [10, 20]:
                mom = rets.rolling(w).mean()
                out[f"stress_adj_momentum_{w}"] = mom * (1 - td["market_stress"].abs() * 0.5)

            factors_list.append(out)

        if not factors_list:
            return pd.DataFrame(index=self.data.index)
        df = pd.concat(factors_list, ignore_index=True)
        logger.info(f"Alternative factors created: {df.shape}")
        return _normalize_dates(df)

    def _mine_macro_factors(self) -> pd.DataFrame:
        """Mine macroeconomic alpha factors."""
        logger.info("Mining macroeconomic factors...")

        try:
            # Import macroeconomic modules
            import sys
            import os
            ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            sys.path.append(ROOT_DIR)

            from strategy.factor.alpha.macroeconomic_alpha_factors import MacroeconomicAlphaFactors
            from data.macro_data_fetcher import MacroDataFetcher

            # Get unique tickers from data
            tickers = self.data['ticker'].unique().tolist()

            # Initialize macroeconomic factor calculator
            macro_factors = MacroeconomicAlphaFactors(tickers, period="2y")

            # Calculate macroeconomic factors
            macro_df = macro_factors.calculate_all_factors()

            if macro_df.empty:
                logger.warning("No macroeconomic factors generated")
                return pd.DataFrame()

            # Create composite factors
            macro_df = macro_factors.create_composite_macro_factors(macro_df)

            # Align with stock data dates
            macro_aligned = []

            for ticker in tickers:
                ticker_stock_data = self.data[self.data['ticker'] == ticker].copy()
                ticker_macro_data = macro_df[macro_df['tic'] == ticker]

                if ticker_macro_data.empty:
                    continue

                # Get the latest macro data for this ticker
                latest_macro = ticker_macro_data.iloc[-1]

                # Create factor dataframe aligned with stock dates
                factor_df = pd.DataFrame()
                factor_df['ticker'] = ticker
                factor_df['date'] = ticker_stock_data['date']

                # Add macro factors (broadcast latest values across all dates)
                macro_factor_cols = [col for col in latest_macro.index
                                   if col not in ['date', 'tic']]

                for col in macro_factor_cols:
                    if pd.notna(latest_macro[col]):
                        factor_df[f'macro_{col}'] = latest_macro[col]

                if len(factor_df) > 0:
                    macro_aligned.append(factor_df)

            if not macro_aligned:
                logger.warning("No aligned macroeconomic factors created")
                return pd.DataFrame()

            result_df = pd.concat(macro_aligned, ignore_index=True)
            result_df = _normalize_dates(result_df)

            logger.info(f"Macroeconomic factors created: {result_df.shape}")
            logger.info(f"Macro factor columns: {len([c for c in result_df.columns if c.startswith('macro_')])}")

            return result_df

        except Exception as e:
            logger.warning(f"Macroeconomic factor mining failed: {e}")
            # Return empty dataframe with correct structure
            empty_df = pd.DataFrame()
            empty_df['ticker'] = self.data['ticker']
            empty_df['date'] = self.data['date']
            return _normalize_dates(empty_df)

    # ------------------ validation & helpers ------------------- #

    def _validate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validating alpha factors...")
        enhanced = self._add_forward_returns(df)
        meta = ["ticker", "date"]
        fac_cols = [c for c in enhanced.columns if c not in meta and not c.startswith("forward_return_")]

        stats: List[Dict] = []
        valid_cols: List[str] = []
        for col in fac_cols:
            if enhanced[col].isna().all():
                continue
            m = self._calculate_factor_metrics(enhanced, col)
            if self._is_factor_valid(m):
                valid_cols.append(col)
                stats.append(m)

        if stats:
            self.factor_performance = pd.DataFrame(stats)
            logger.info("Top factors by IC:")
            top = self.factor_performance.sort_values("ic_1d", key=lambda s: s.abs(), ascending=False).head(10)
            for _, r in top.iterrows():
                logger.info(f"  {r['factor']}: IC={r.get('ic_1d', np.nan):.4f}, IC_IR={r.get('ic_ir_1d', 0):.2f}")

        cols = meta + valid_cols
        cols = [c for c in cols if c in enhanced.columns]
        logger.info(f"Validation completed: {len(valid_cols)} factors passed validation")
        return enhanced[cols] if cols else pd.DataFrame()

    def _calculate_factor_metrics(self, data: pd.DataFrame, factor_col: str) -> Dict:
        metrics: Dict = {"factor": factor_col}
        for h in [1, 5, 10]:
            rc = f"forward_return_{h}d"
            if rc not in data.columns:
                continue
            sub = data[[factor_col, rc]].dropna()
            if len(sub) < 20:
                continue
            ic = sub[factor_col].corr(sub[rc])
            ric = sub[factor_col].corr(sub[rc], method="spearman")
            metrics[f"ic_{h}d"] = ic
            metrics[f"rank_ic_{h}d"] = ric
            if len(sub) > 50:
                rolling_ic = sub[factor_col].rolling(50).corr(sub[rc])
                ic_std = rolling_ic.std()
                metrics[f"ic_ir_{h}d"] = ic / (ic_std + 1e-12)
            hit = (np.sign(sub[factor_col]) == np.sign(sub[rc])).mean()
            metrics[f"hit_rate_{h}d"] = hit

        fv = data[factor_col].dropna()
        if len(fv) > 10:
            metrics["autocorr"] = fv.autocorr()
        diff = data[factor_col].diff().abs()
        sd = data[factor_col].std()
        if sd and sd > 0:
            metrics["turnover"] = float(diff.mean() / sd)
        metrics["coverage"] = float(1 - data[factor_col].isna().mean())
        return metrics

    def _is_factor_valid(self, metrics: Dict) -> bool:
        ic_1d = abs(metrics.get("ic_1d", 0.0))
        if ic_1d < self.min_ic_threshold:
            return False
        if metrics.get("coverage", 0.0) < 0.5:
            return False
        if abs(metrics.get("autocorr", 0.0)) > 0.95:
            return False
        if metrics.get("turnover", float("inf")) > 10:
            return False
        return True

    def _add_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        out: List[pd.DataFrame] = []
        for t in df["ticker"].unique():
            sub = df[df["ticker"] == t].copy().sort_values("date")
            # bring in close from base data
            prices = self.data[self.data["ticker"] == t][["date", "close"]].sort_values("date").copy()
            prices = _normalize_dates(prices)
            for h in [1, 5, 10, 20]:
                if len(prices) > h:
                    fwd = prices["close"].pct_change(h).shift(-h)
                    prices[f"forward_return_{h}d"] = fwd
            sub = sub.merge(prices, on="date", how="left", suffixes=("", "_price"))
            out.append(sub)
        return pd.concat(out, ignore_index=True) if out else df

    # --------------------- indicators -------------------------- #

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def _calculate_stochastic(self, data: pd.DataFrame, k_window: int = 14) -> Tuple[pd.Series, pd.Series]:
        if not {"high", "low", "close"}.issubset(data.columns):
            c = data["close"].astype(float)
            rmax = c.rolling(k_window).max()
            rmin = c.rolling(k_window).min()
            k = 100 * ((c - rmin) / (rmax - rmin + 1e-12))
        else:
            h = data["high"].astype(float)
            l = data["low"].astype(float)
            c = data["close"].astype(float)
            rmax = h.rolling(k_window).max()
            rmin = l.rolling(k_window).min()
            k = 100 * ((c - rmin) / (rmax - rmin + 1e-12))
        d = k.rolling(3).mean()
        return k, d

    # Kyle's lambda simplified already in miner


def main() -> None:
    from data.enhanced_data_fetcher import EnhancedDataFetcher

    print("=== Real Alpha Factor Mining Demo ===")
    fetcher = EnhancedDataFetcher(start_date="2020-01-01", end_date="2023-12-31")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print(f"Fetching data for {len(tickers)} tickers...")
    alpha_data = fetcher.create_alpha_research_dataset(tickers, include_fundamentals=True)

    miner = RealAlphaMiner(data=alpha_data)
    alpha_factors = miner.mine_all_alpha_factors()

    print(f"✅ Factor mining completed, got {alpha_factors.shape[1]-2} factors.")
    print(f"Date range: {alpha_factors['date'].min()} to {alpha_factors['date'].max()}")


if __name__ == "__main__":
    main()

