#!/usr/bin/env python3
"""
Portfolio Optimization System - Enhanced Main Entry Point
========================================================

A complete portfolio optimization system integrating Alpha factor mining, 
Beta risk modeling, and portfolio optimization.

Key Features:
- 📊 Data Acquisition: Automatic fetching of stock prices, fundamentals, and financial data
- 🔍 Alpha Mining: Technical, fundamental, machine learning, and cross-sectional factors
- ⚖️ Beta Estimation: CAPM, multi-factor, Copula, and CVaR risk models  
- 🎯 Portfolio Optimization: Maximum Sharpe, minimum variance, maximum utility, risk parity
- 📈 Strategy Backtesting: Rolling optimization, performance evaluation, risk analysis
- 🔄 Risk Control: Stop-loss, position management, risk budgeting
- ⚡ Real-time Execution: Automated trading, monitoring, rebalancing

Usage:
python start_here.py --mode [quick|full|backtest|live] --tickers AAPL,MSFT,GOOGL
"""

import sys
import os
import logging
import argparse
import warnings
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import traceback

# Ensure local modules can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
np.seterr(invalid='ignore', divide='ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure results directories exist
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Import project modules
MODULES_STATUS = {
    'data_fetcher': False,
    'alpha_miner': False, 
    'beta_estimator': False,
    'optimizer': False,
    'risk_control': False,
    'execution': False
}

try:
    from data.enhanced_data_fetcher import EnhancedDataFetcher
    MODULES_STATUS['data_fetcher'] = True
    logger.info("✅ Data fetcher module loaded")
except ImportError as e:
    logger.warning(f"❌ Data fetcher not available: {e}")

try:
    from strategy.factor.alpha.real_alpha_miner import RealAlphaMiner
    MODULES_STATUS['alpha_miner'] = True  
    logger.info("✅ Alpha miner module loaded")
except ImportError as e:
    logger.warning(f"❌ Alpha miner not available: {e}")

try:
    from strategy.factor.beta.real_beta_estimator import RealBetaEstimator
    MODULES_STATUS['beta_estimator'] = True
    logger.info("✅ Beta estimator module loaded")
except ImportError as e:
    logger.warning(f"❌ Beta estimator not available: {e}")

try:
    from strategy.alpha_beta_optimizer import AlphaBetaOptimizer
    MODULES_STATUS['optimizer'] = True
    logger.info("✅ Portfolio optimizer module loaded")
except ImportError as e:
    logger.warning(f"❌ Portfolio optimizer not available: {e}")

try:
    from risk_control.risk_manager import RiskManager
    MODULES_STATUS['risk_control'] = True
    logger.info("✅ Risk control module loaded")
except ImportError as e:
    logger.warning(f"❌ Risk control not available: {e}")

try:
    from execution_engine.portfolio_executor import PortfolioExecutor
    MODULES_STATUS['execution'] = True
    logger.info("✅ Execution engine module loaded")
except ImportError as e:
    logger.warning(f"❌ Execution engine not available: {e}")


class AdvancedPortfolioSystem:
    """Advanced Portfolio System Main Class"""
    
    def __init__(self, 
                 start_date: str = None,
                 end_date: str = None,
                 market_index: str = 'SPY',
                 initial_capital: float = 1000000.0,
                 risk_budget: float = 0.15):
        """
        Initialize the portfolio system.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            market_index: Market benchmark index
            initial_capital: Initial capital
            risk_budget: Risk budget
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.market_index = market_index
        self.initial_capital = initial_capital
        self.risk_budget = risk_budget
        
        # System status
        self.system_ready = all(MODULES_STATUS[key] for key in ['data_fetcher', 'alpha_miner', 'beta_estimator'])
        
        logger.info(f"🚀 Portfolio System initialized")
        logger.info(f"📅 Period: {self.start_date} to {self.end_date}")
        logger.info(f"📈 Market Index: {self.market_index}")
        logger.info(f"💰 Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"⚠️ Risk Budget: {self.risk_budget:.1%}")
        logger.info(f"🔧 System Ready: {self.system_ready}")
        
        # Core components
        self.data_fetcher = None
        self.alpha_miner = None
        self.beta_estimator = None
        self.optimizer = None
        self.risk_manager = None
        self.executor = None
        
        # Data storage
        self.market_data = {}
        self.alpha_factors = None
        self.beta_estimates = {}
        self.portfolio_weights = {}
        self.performance_metrics = {}
        self.risk_metrics = {}
        
        # Analysis results
        self.results = {
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'modules_status': MODULES_STATUS,
                'parameters': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'market_index': self.market_index,
                    'initial_capital': self.initial_capital,
                    'risk_budget': self.risk_budget
                }
            }
        }

    def check_system_health(self) -> Dict[str, Any]:
        """Check system health status"""
        print("\n" + "="*60)
        print("🔍 SYSTEM HEALTH CHECK")
        print("="*60)
        
        health_status = {
            'overall_status': 'healthy',
            'critical_modules': 0,
            'warning_modules': 0,
            'modules_detail': {},
            'recommendations': []
        }
        
        # Check core modules
        core_modules = ['data_fetcher', 'alpha_miner', 'beta_estimator', 'optimizer']
        for module in core_modules:
            status = MODULES_STATUS.get(module, False)
            health_status['modules_detail'][module] = {
                'available': status,
                'critical': True
            }
            if not status:
                health_status['critical_modules'] += 1
                health_status['recommendations'].append(f"Install/fix {module} module")
                print(f"❌ {module.replace('_', ' ').title()}: Critical module missing")
            else:
                print(f"✅ {module.replace('_', ' ').title()}: Available")
        
        # Check optional modules
        optional_modules = ['risk_control', 'execution']
        for module in optional_modules:
            status = MODULES_STATUS.get(module, False)
            health_status['modules_detail'][module] = {
                'available': status,
                'critical': False
            }
            if not status:
                health_status['warning_modules'] += 1
                health_status['recommendations'].append(f"Consider installing {module} module for enhanced functionality")
                print(f"⚠️ {module.replace('_', ' ').title()}: Optional module missing")
            else:
                print(f"✅ {module.replace('_', ' ').title()}: Available")
        
        # Evaluate overall status
        if health_status['critical_modules'] > 0:
            health_status['overall_status'] = 'critical'
            print(f"\n🚨 System Status: CRITICAL ({health_status['critical_modules']} critical issues)")
        elif health_status['warning_modules'] > 2:
            health_status['overall_status'] = 'warning'
            print(f"\n⚠️ System Status: WARNING ({health_status['warning_modules']} warnings)")
        else:
            print(f"\n✅ System Status: HEALTHY")
        
        # Print recommendations
        if health_status['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(health_status['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        self.results['health_check'] = health_status
        return health_status

    def run_quick_analysis(self, tickers: List[str] = None) -> Dict[str, Any]:
        """Quick analysis - basic functionality demo"""
        print("\n" + "="*80)
        print("⚡ QUICK PORTFOLIO ANALYSIS")
        print("="*80)
        
        if not self.system_ready:
            print("❌ System not ready for analysis")
            return {'error': 'System not ready', 'health_check': self.check_system_health()}
        
        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B"]
        
        print(f"📊 Analyzing: {', '.join(tickers)}")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        
        results = {'analysis_type': 'quick', 'tickers': tickers}
        
        try:
            # 1. Data acquisition
            print(f"\n{'='*50}")
            print("📥 1. DATA ACQUISITION")
            print(f"{'='*50}")
            
            self.data_fetcher = EnhancedDataFetcher(self.start_date, self.end_date)
            
            # Fetch core datasets
            alpha_data = self.data_fetcher.create_alpha_research_dataset(
                tickers, include_fundamentals=True
            )
            beta_data = self.data_fetcher.create_beta_research_dataset(tickers)
            
            print(f"✅ Data acquired: {alpha_data.shape[0]} observations, {len(tickers)} assets")
            
            results['data_summary'] = {
                'observations': alpha_data.shape[0],
                'assets': len(tickers),
                'features': alpha_data.shape[1],
                'date_range': [alpha_data['date'].min().strftime('%Y-%m-%d'), 
                              alpha_data['date'].max().strftime('%Y-%m-%d')]
            }
            
            # 2. Alpha factor mining
            print(f"\n{'='*50}")
            print("🔬 2. ALPHA FACTOR MINING")
            print(f"{'='*50}")
            
            self.alpha_miner = RealAlphaMiner(
                data=alpha_data,
                feature_windows=[5, 10, 20, 60],
                prediction_horizons=[1, 5, 10, 20],
                min_ic_threshold=0.01
            )
            
            print("⚙️ Mining alpha factors...")
            self.alpha_factors = self.alpha_miner.mine_all_alpha_factors()
            
            # Factor performance analysis
            if hasattr(self.alpha_miner, 'factor_performance') and not self.alpha_miner.factor_performance.empty:
                perf_df = self.alpha_miner.factor_performance
                
                # Safely get IC columns
                ic_cols = [col for col in perf_df.columns if 'ic' in col.lower()]
                if ic_cols:
                    ic_col = ic_cols[0]
                    top_factors = perf_df.nlargest(10, ic_col, keep='first')
                    
                    print(f"\n📊 Top 10 Alpha Factors (by {ic_col}):")
                    print("-" * 80)
                    for idx, row in top_factors.iterrows():
                        factor_name = row.get('factor', f'Factor_{idx}')[:35]
                        ic_val = row.get(ic_col, 0)
                        ir_val = row.get('ic_ir_1d', row.get('ic_ir', 0))
                        coverage = row.get('coverage', 0)
                        print(f"  {factor_name:<35} IC:{ic_val:>8.4f} IR:{ir_val:>7.2f} Cov:{coverage:>7.1%}")
                    
                    results['alpha_summary'] = {
                        'total_factors': len(perf_df),
                        'effective_factors': len(self.alpha_factors.columns) - 2,
                        'best_ic': float(top_factors.iloc[0].get(ic_col, 0)),
                        'avg_ic': float(perf_df[ic_col].mean()),
                        'top_factor': top_factors.iloc[0].get('factor', 'Unknown')
                    }
                else:
                    print("⚠️ No IC metrics available")
                    results['alpha_summary'] = {'total_factors': 0, 'warning': 'No IC metrics'}
            else:
                print("⚠️ Factor performance evaluation unavailable")
                results['alpha_summary'] = {'error': 'Performance evaluation failed'}
            
            print(f"✅ Alpha mining completed: {self.alpha_factors.shape[1]-2} factors extracted")
            
            # 3. Beta risk estimation
            print(f"\n{'='*50}")
            print("📊 3. BETA RISK ESTIMATION") 
            print(f"{'='*50}")
            
            self.beta_estimator = RealBetaEstimator(data=beta_data)
            
            print("⚙️ Estimating beta coefficients...")
            self.beta_estimates = self.beta_estimator.estimate_all_betas()
            
            # Display Beta results
            beta_summary = {}
            for method, data in self.beta_estimates.items():
                if not data.empty:
                    print(f"\n📈 {method.upper().replace('_', ' ')} Results:")
                    
                    if method == 'capm_beta':
                        display_data = data[['ticker', 'beta', 'r_squared']].round(4)
                        beta_summary[method] = {
                            'avg_beta': float(data['beta'].mean()),
                            'beta_range': [float(data['beta'].min()), float(data['beta'].max())],
                            'avg_r_squared': float(data['r_squared'].mean())
                        }
                    elif method == 'multi_factor_beta':
                        factor_cols = [c for c in data.columns if c.startswith('beta_')]
                        display_cols = ['ticker', 'alpha'] + factor_cols[:3]  # Show first 3 factors
                        display_data = data[display_cols].round(4)
                        beta_summary[method] = {
                            'factors_count': len(factor_cols),
                            'avg_alpha': float(data['alpha'].mean()) if 'alpha' in data.columns else 0
                        }
                    else:
                        display_data = data.head()
                        beta_summary[method] = {'observations': len(data)}
                    
                    print(display_data.to_string(index=False))
            
            results['beta_summary'] = {
                'methods_available': list(self.beta_estimates.keys()),
                'details': beta_summary
            }
            
            print(f"✅ Beta estimation completed: {len(self.beta_estimates)} methods")
            
            # 4. Quick portfolio optimization
            print(f"\n{'='*50}")
            print("🎯 4. PORTFOLIO OPTIMIZATION")
            print(f"{'='*50}")
            
            if MODULES_STATUS['optimizer']:
                try:
                    self.optimizer = AlphaBetaOptimizer(
                        data=alpha_data,
                        market_index=self.market_index
                    )
                    
                    # Run multiple optimization methods
                    optimization_methods = ['max_sharpe', 'min_variance', 'risk_parity']
                    opt_results = {}
                    
                    for method in optimization_methods:
                        print(f"⚙️ Running {method} optimization...")
                        try:
                            result = self.optimizer.optimize_portfolio(
                                assets=tickers,
                                method=method,
                                risk_aversion=2.0
                            )
                            
                            if result.get('success', False):
                                opt_results[method] = result
                                
                                # Format output
                                ret = result.get('expected_return', 0)
                                vol = result.get('volatility', 0) 
                                sharpe = result.get('sharpe_ratio', 0)
                                
                                print(f"  ✅ {method}: Return={ret:.2%} Vol={vol:.2%} Sharpe={sharpe:.3f}")
                                
                                # Show top holdings
                                weights = np.array(result['weights'])
                                assets = result['assets']
                                top_holdings = sorted(zip(assets, weights), key=lambda x: x[1], reverse=True)[:3]
                                holdings_str = ", ".join([f"{asset}:{weight:.1%}" for asset, weight in top_holdings])
                                print(f"      Top holdings: {holdings_str}")
                            else:
                                print(f"  ❌ {method}: {result.get('message', 'Optimization failed')}")
                        
                        except Exception as e:
                            print(f"  ❌ {method}: Error - {str(e)[:50]}")
                    
                    results['optimization_summary'] = {
                        'successful_methods': len(opt_results),
                        'methods_attempted': len(optimization_methods),
                        'results': {
                            method: {
                                'return': float(res.get('expected_return', 0)),
                                'volatility': float(res.get('volatility', 0)),
                                'sharpe_ratio': float(res.get('sharpe_ratio', 0))
                            }
                            for method, res in opt_results.items()
                        }
                    }
                    
                    # Select best strategy
                    if opt_results:
                        best_method = max(opt_results.keys(), 
                                        key=lambda k: opt_results[k].get('sharpe_ratio', -999))
                        print(f"\n🏆 Best performing method: {best_method}")
                        
                        results['optimization_summary']['best_method'] = best_method
                        
                except Exception as e:
                    logger.error(f"Portfolio optimization error: {e}")
                    results['optimization_summary'] = {'error': str(e)}
                    print(f"❌ Portfolio optimization failed: {e}")
            else:
                print("❌ Portfolio optimizer not available")
                results['optimization_summary'] = {'error': 'Optimizer module not available'}
            
            print(f"✅ Portfolio optimization completed")
            
        except Exception as e:
            logger.error(f"Quick analysis error: {e}")
            logger.error(traceback.format_exc())
            results['error'] = str(e)
            print(f"❌ Analysis failed: {e}")
        
        # 5. Results summary
        print(f"\n{'='*80}")
        print("📋 ANALYSIS SUMMARY")
        print("="*80)
        
        self._print_quick_summary(results)
        
        # Save results
        self.results['quick_analysis'] = results
        self._save_analysis_results('quick')
        
        return results

    def run_comprehensive_analysis(self, 
                                 tickers: List[str],
                                 save_results: bool = True,
                                 run_backtest: bool = True) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        print(f"\n{'='*80}")
        print("🔬 COMPREHENSIVE PORTFOLIO ANALYSIS")  
        print("="*80)
        
        # First run quick analysis
        results = self.run_quick_analysis(tickers)
        
        if 'error' in results:
            return results
        
        try:
            # Risk control analysis
            if MODULES_STATUS['risk_control'] and run_backtest:
                print(f"\n{'='*50}")
                print("⚠️ RISK CONTROL ANALYSIS")
                print(f"{'='*50}")
                
                self.risk_manager = RiskManager(
                    initial_capital=self.initial_capital,
                    risk_budget=self.risk_budget
                )
                
                # Risk metrics calculation
                if hasattr(self.alpha_factors, 'values'):
                    risk_metrics = self.risk_manager.calculate_risk_metrics(
                        returns_data=self.alpha_factors,
                        assets=tickers[:6]  # Limit assets for performance
                    )
                    
                    if risk_metrics:
                        print("✅ Risk metrics calculated:")
                        for metric, value in risk_metrics.items():
                            if isinstance(value, (int, float)):
                                if 'var' in metric.lower() or 'cvar' in metric.lower():
                                    print(f"   {metric}: ${value:,.0f}")
                                else:
                                    print(f"   {metric}: {value:.4f}")
                        
                        results['risk_analysis'] = risk_metrics
                
                print("✅ Risk control analysis completed")
            
            # Strategy backtesting
            if run_backtest and MODULES_STATUS['optimizer'] and self.optimizer:
                print(f"\n{'='*50}")
                print("📈 STRATEGY BACKTESTING")
                print(f"{'='*50}")
                
                # Select best strategy for backtesting
                best_method = results.get('optimization_summary', {}).get('best_method', 'max_sharpe')
                
                if best_method:
                    print(f"⚙️ Backtesting {best_method} strategy...")
                    
                    try:
                        # Use fewer assets for backtesting to improve performance
                        backtest_assets = tickers[:min(6, len(tickers))]
                        
                        backtest_results = self.optimizer.backtest_strategy(
                            assets=backtest_assets,
                            rebalance_frequency='monthly',
                            lookback_period=120,
                            method=best_method
                        )
                        
                        if not backtest_results.empty:
                            # Calculate key metrics
                            total_return = backtest_results['cumulative_return'].iloc[-1] - 1
                            returns = backtest_results['portfolio_return'].dropna()
                            
                            annual_return = returns.mean() * 252
                            annual_vol = returns.std() * np.sqrt(252)
                            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                            
                            max_dd = self._calculate_max_drawdown(backtest_results['cumulative_return'])
                            
                            print(f"✅ Backtest completed ({len(backtest_results)} periods):")
                            print(f"   Total Return: {total_return:.2%}")
                            print(f"   Annual Return: {annual_return:.2%}")
                            print(f"   Annual Volatility: {annual_vol:.2%}")
                            print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
                            print(f"   Maximum Drawdown: {max_dd:.2%}")
                            
                            results['backtest_summary'] = {
                                'strategy': best_method,
                                'total_return': float(total_return),
                                'annual_return': float(annual_return),
                                'annual_volatility': float(annual_vol),
                                'sharpe_ratio': float(sharpe_ratio),
                                'max_drawdown': float(max_dd),
                                'periods': len(backtest_results)
                            }
                            
                            # Save backtest results
                            backtest_path = f'results/backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                            backtest_results.to_csv(backtest_path, index=False)
                            print(f"💾 Backtest results saved: {backtest_path}")
                        
                    except Exception as e:
                        logger.error(f"Backtesting error: {e}")
                        results['backtest_error'] = str(e)
                        print(f"❌ Backtesting failed: {e}")
            
            # Save comprehensive analysis results
            if save_results:
                self.results['comprehensive_analysis'] = results
                self._save_analysis_results('comprehensive')
                self._generate_analysis_report(results)
            
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            results['comprehensive_error'] = str(e)
        
        return results

    def run_live_trading_setup(self, 
                              tickers: List[str],
                              trading_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Live trading setup"""
        print(f"\n{'='*80}")
        print("🔴 LIVE TRADING SETUP")
        print("="*80)
        
        if not MODULES_STATUS['execution']:
            print("❌ Execution engine not available for live trading")
            return {'error': 'Execution engine not available'}
        
        if trading_config is None:
            trading_config = {
                'rebalance_frequency': 'daily',
                'max_position_size': 0.15,
                'stop_loss': 0.05,
                'take_profit': 0.20
            }
        
        print("⚠️  WARNING: Live trading involves real money and risks!")
        print("⚠️  This is a demonstration setup only.")
        print(f"📊 Assets: {', '.join(tickers)}")
        print(f"⚙️  Config: {trading_config}")
        
        try:
            # Initialize execution engine
            self.executor = PortfolioExecutor(
                initial_capital=self.initial_capital,
                config=trading_config
            )
            
            # Setup monitoring system
            setup_result = self.executor.setup_live_trading(
                assets=tickers,
                strategy_config=trading_config
            )
            
            results = {
                'live_trading_setup': setup_result,
                'config': trading_config,
                'status': 'ready' if setup_result.get('success') else 'failed'
            }
            
            if setup_result.get('success'):
                print("✅ Live trading setup completed")
                print("💡 Use separate monitoring script to start live trading")
            else:
                print(f"❌ Live trading setup failed: {setup_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Live trading setup error: {e}")
            results = {'error': str(e)}
            print(f"❌ Live trading setup failed: {e}")
        
        return results

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns / peak) - 1
            return float(drawdown.min())
        except:
            return 0.0

    def _print_quick_summary(self, results: Dict[str, Any]):
        """Print quick analysis summary"""
        data_summary = results.get('data_summary', {})
        alpha_summary = results.get('alpha_summary', {})
        beta_summary = results.get('beta_summary', {})
        opt_summary = results.get('optimization_summary', {})
        
        print(f"📊 Data: {data_summary.get('observations', 0)} obs, {data_summary.get('assets', 0)} assets")
        print(f"🔬 Alpha: {alpha_summary.get('effective_factors', 0)} factors (IC: {alpha_summary.get('best_ic', 0):.3f})")
        print(f"📈 Beta: {len(beta_summary.get('methods_available', []))} methods")
        print(f"🎯 Optimization: {opt_summary.get('successful_methods', 0)}/{opt_summary.get('methods_attempted', 0)} successful")
        
        if 'backtest_summary' in results:
            bt = results['backtest_summary']
            print(f"📈 Backtest: {bt.get('total_return', 0):.2%} total return, {bt.get('sharpe_ratio', 0):.3f} Sharpe")
        
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _save_analysis_results(self, analysis_type: str):
        """Save analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save main results file
            results_file = f'results/analysis_{analysis_type}_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"💾 Analysis results saved: {results_file}")
            
            # Save key data files
            if self.alpha_factors is not None and not self.alpha_factors.empty:
                alpha_file = f'results/alpha_factors_{timestamp}.csv'
                self.alpha_factors.to_csv(alpha_file, index=False)
                print(f"💾 Alpha factors saved: {alpha_file}")
            
            if self.beta_estimates:
                for method, data in self.beta_estimates.items():
                    if not data.empty:
                        beta_file = f'results/beta_{method}_{timestamp}.csv'
                        data.to_csv(beta_file, index=False)
                print(f"💾 Beta estimates saved: results/beta_*_{timestamp}.csv")
            
        except Exception as e:
            logger.warning(f"Error saving results: {e}")

    def _generate_analysis_report(self, results: Dict[str, Any]):
        """Generate analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'results/analysis_report_{timestamp}.md'
        
        try:
            with open(report_file, 'w') as f:
                f.write(f"# Portfolio Analysis Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # System information
                f.write(f"## System Information\n")
                f.write(f"- Analysis Period: {self.start_date} to {self.end_date}\n")
                f.write(f"- Market Index: {self.market_index}\n")
                f.write(f"- Initial Capital: ${self.initial_capital:,.0f}\n\n")
                
                # Data summary
                data_summary = results.get('data_summary', {})
                f.write(f"## Data Summary\n")
                f.write(f"- Observations: {data_summary.get('observations', 0)}\n")
                f.write(f"- Assets: {data_summary.get('assets', 0)}\n")
                f.write(f"- Features: {data_summary.get('features', 0)}\n\n")
                
                # Alpha factors
                alpha_summary = results.get('alpha_summary', {})
                f.write(f"## Alpha Factor Analysis\n")
                f.write(f"- Total Factors: {alpha_summary.get('total_factors', 0)}\n")
                f.write(f"- Effective Factors: {alpha_summary.get('effective_factors', 0)}\n")
                f.write(f"- Best IC: {alpha_summary.get('best_ic', 0):.4f}\n")
                f.write(f"- Average IC: {alpha_summary.get('avg_ic', 0):.4f}\n\n")
                
                # Portfolio optimization
                opt_summary = results.get('optimization_summary', {})
                f.write(f"## Portfolio Optimization\n")
                f.write(f"- Successful Methods: {opt_summary.get('successful_methods', 0)}\n")
                
                if 'results' in opt_summary:
                    f.write(f"\n### Optimization Results\n")
                    for method, metrics in opt_summary['results'].items():
                        f.write(f"**{method.replace('_', ' ').title()}:**\n")
                        f.write(f"- Expected Return: {metrics.get('return', 0):.2%}\n")
                        f.write(f"- Volatility: {metrics.get('volatility', 0):.2%}\n")
                        f.write(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n\n")
                
                # Backtest results
                if 'backtest_summary' in results:
                    bt = results['backtest_summary']
                    f.write(f"## Backtest Results\n")
                    f.write(f"- Strategy: {bt.get('strategy', 'N/A')}\n")
                    f.write(f"- Total Return: {bt.get('total_return', 0):.2%}\n")
                    f.write(f"- Annual Return: {bt.get('annual_return', 0):.2%}\n")
                    f.write(f"- Annual Volatility: {bt.get('annual_volatility', 0):.2%}\n")
                    f.write(f"- Sharpe Ratio: {bt.get('sharpe_ratio', 0):.3f}\n")
                    f.write(f"- Max Drawdown: {bt.get('max_drawdown', 0):.2%}\n\n")
            
            print(f"📄 Analysis report generated: {report_file}")
            
        except Exception as e:
            logger.warning(f"Error generating report: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Advanced Portfolio Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_here.py --mode quick --tickers AAPL,MSFT,GOOGL
  python start_here.py --mode full --tickers AAPL,MSFT,GOOGL,AMZN --save
  python start_here.py --mode backtest --tickers QQQ,SPY --start-date 2023-01-01
  python start_here.py --mode live --tickers AAPL,MSFT --capital 100000
        """
    )
    
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'backtest', 'live', 'health'],
                       help='Run mode: quick(fast), full(comprehensive), backtest(backtesting), live(trading), health(health check)')
    
    parser.add_argument('--tickers', type=str, 
                       default='AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA',
                       help='Stock tickers, comma separated (e.g., AAPL,MSFT,GOOGL)')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    
    parser.add_argument('--market-index', type=str, default='SPY',
                       help='Market benchmark index')
    
    parser.add_argument('--capital', type=float, default=1000000.0,
                       help='Initial capital (default: 1,000,000)')
    
    parser.add_argument('--risk-budget', type=float, default=0.15,
                       help='Risk budget (default: 0.15)')
    
    parser.add_argument('--save', action='store_true',
                       help='Save analysis results')
    
    parser.add_argument('--no-backtest', action='store_true',
                       help='Skip backtesting (for full mode only)')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (JSON format)')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def main():
    """Main function"""
    print("=" * 80)
    print("🚀 ADVANCED PORTFOLIO OPTIMIZATION SYSTEM")
    print("   Multi-Factor Alpha Mining + Risk Modeling + Portfolio Optimization")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Process arguments
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(',')]
    
    print(f"🎯 Mode: {args.mode.upper()}")
    print(f"📊 Assets: {', '.join(tickers)}")
    print(f"📅 Period: {args.start_date or 'Auto'} to {args.end_date or 'Current'}")
    print(f"💰 Capital: ${args.capital:,.0f}")
    print(f"⚠️ Risk Budget: {args.risk_budget:.1%}")
    
    # Create system instance
    system = AdvancedPortfolioSystem(
        start_date=args.start_date,
        end_date=args.end_date,
        market_index=args.market_index,
        initial_capital=args.capital,
        risk_budget=args.risk_budget
    )
    
    try:
        # Run analysis based on mode
        if args.mode == 'health':
            results = system.check_system_health()
            
        elif args.mode == 'quick':
            results = system.run_quick_analysis(tickers)
            
        elif args.mode == 'full':
            results = system.run_comprehensive_analysis(
                tickers=tickers,
                save_results=args.save,
                run_backtest=not args.no_backtest
            )
            
        elif args.mode == 'backtest':
            # Dedicated backtesting mode
            results = system.run_comprehensive_analysis(
                tickers=tickers,
                save_results=args.save,
                run_backtest=True
            )
            
        elif args.mode == 'live':
            # Live trading setup
            trading_config = config.get('trading', {
                'rebalance_frequency': 'daily',
                'max_position_size': 0.15,
                'stop_loss': 0.05,
                'take_profit': 0.20
            })
            
            results = system.run_live_trading_setup(tickers, trading_config)
        
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Display final results
        print(f"\n{'='*80}")
        if results.get('error'):
            print("❌ ANALYSIS FAILED")
            print(f"Error: {results['error']}")
            return 1
        else:
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        if args.save or args.mode in ['full', 'backtest']:
            print("💾 Results have been saved to 'results/' directory")
            print("📄 Check the generated report for detailed analysis")
        
        print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"System error: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ System error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)