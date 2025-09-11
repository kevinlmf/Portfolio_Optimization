+  #!/usr/bin/env python3
        2    """
        3 -  Portfolio Optimization System - Main Entry Point
        3 +  Portfolio Optimization System - Enhanced 
          +  Main Entry Point
        4    ====================================================
            ============================
        5 -  一个完整的投资组合优化系统，整合Alpha因子挖掘、Beta风
     险建模和组合优化
        5 +  完整的投资组合优化系统，集成Alpha因子挖掘、Beta风险建
     模和投资组合优化
        6    
        7    主要功能:
        8 -  - Alpha因子挖掘：技术分析、基本面、微观结构、机器学
          - 习、跨截面因子
        9 -  - 
          - Beta风险估计：CAPM、多因子模型、Copula模型、CVaR模型 
          -  
       10 -  - 投资组合优化：夏普比率最大化、最小方差、效用最大化
          - 、风险平价
       11 -  - 策略回测：滚动优化、业绩评估、风险分析
        8 +  - 📊 数据获取: 股价、基本面、财务数据自动获取
        9 +  - 🔍 Alpha挖掘: 技术、基本面、机器学习、跨截面因子
       10 +  - ⚖️ Beta估计: CAPM、多因子、Copula、CVaR风险模型  
       11 +  - 🎯 组合优化: 
          + 夏普最大、最小方差、效用最大、风险平价
       12 +  - 📈 策略回测: 滚动优化、业绩评估、风险分析
       13 +  - 🔄 风险控制: 止损、仓位管理、风险预算
       14 +  - ⚡ 实时执行: 自动下单、监控、调仓
       15    
       16    使用方法:
       17 -  python start_here.py --mode [demo|full|custom
          -  ] --tickers AAPL,MSFT,GOOGL
       17 +  python start_here.py --mode [quick|full|
          +  backtest|live] --tickers AAPL,MSFT,GOOGL
       18    """
       19    
       20    import sys
     ...
        18    import logging
        19    import argparse
        20    import warnings
        21 +  import json
        22    import numpy as np
        23    import pandas as pd
        24    from datetime import datetime, timedelta
        25 -  from typing import List, Dict, Optional
        25 +  from typing import List, Dict, Optional, Tuple, Any
        26 +  from pathlib import Path
        27 +  import traceback
        28    
        29    # 确保能找到本地模块
        30 -  sys.path.append(
           -  os.path.dirname(os.path.abspath(__file__)))
        31 -  sys.path.append(".")
        30 +  sys.path.insert(0, 
           +  os.path.dirname(os.path.abspath(__file__)))
        31 +  sys.path.insert(0, ".")
        32    
        33    # 抑制警告
        34 -  warnings.filterwarnings('ignore')
        34 +  warnings.filterwarnings('ignore'
           +  , category=FutureWarning)
        35 +  warnings.filterwarnings('ignore', 
           + category=UserWarning)
        36 +  np.seterr(invalid='ignore', divide='ignore')
        37    
        38    # 配置日志
        39    logging.basicConfig(
        40 -      level=logging.INFO, 
        41 -      format='%(asctime)s - %(name)s - %(levelname)s 
           -  - %(message)s'
        40 +      level=logging.INFO,
        41 +      format='%(asctime)s - %(name)s - %(levelname)s 
           +  - %(message)s',
        42 +      handlers=[
        43 +          
           + logging.FileHandler('portfolio_system.log'),
        44 +          logging.StreamHandler(sys.stdout)
        45 +      ]
        46    )
        47    logger = logging.getLogger(__name__)
        48    
        49 +  # 确保结果目录存在
        50 +  os.makedirs('results', exist_ok=True)
        51 +  os.makedirs('logs', exist_ok=True)
        52 +  
        53    # 导入项目模块
        54 +  MODULES_STATUS = {
        55 +      'data_fetcher': False,
        56 +      'alpha_miner': False, 
        57 +      'beta_estimator': False,
        58 +      'optimizer': False,
        59 +      'risk_control': False,
        60 +      'execution': False
        61 +  }
        62 +  
        63    try:
        64        from data.enhanced_data_fetcher import 
             EnhancedDataFetcher
        65 -      from strategy.factor.alpha.real_alpha_miner 
           - import RealAlphaMiner  
        65 +      MODULES_STATUS['data_fetcher'] = True
        66 +      logger.info("✅ Data fetcher module loaded")
        67 +  except ImportError as e:
        68 +      logger.warning(f"❌ Data fetcher not available:
           +  {e}")
        69 +  
        70 +  try:
        71 +      from strategy.factor.alpha.real_alpha_miner 
           + import RealAlphaMiner
        72 +      MODULES_STATUS['alpha_miner'] = True  
        73 +      logger.info("✅ Alpha miner module loaded")
        74 +  except ImportError as e:
        75 +      logger.warning(f"❌ Alpha miner not available: 
           + {e}")
        76 +  
        77 +  try:
        78        from strategy.factor.beta.real_beta_estimator 
             import RealBetaEstimator
        79 +      MODULES_STATUS['beta_estimator'] = True
        80 +      logger.info("✅ Beta estimator module loaded")
        81 +  except ImportError as e:
        82 +      logger.warning(f"❌ Beta estimator not 
           + available: {e}")
        83 +  
        84 +  try:
        85        from strategy.alpha_beta_optimizer import 
             AlphaBetaOptimizer
        86 -      DATA_MODULES_AVAILABLE = True
        86 +      MODULES_STATUS['optimizer'] = True
        87 +      logger.info("✅ Portfolio optimizer module 
           + loaded")
        88    except ImportError as e:
        89 -      logger.warning(f"Some modules 
           -  not available: {e}")
        90 -      DATA_MODULES_AVAILABLE = False
        89 +      logger.warning(f"❌ Portfolio optimizer 
           +  not available: {e}")
        90    
        91 +  try:
        92 +      from risk_control.risk_manager import 
           + RiskManager
        93 +      MODULES_STATUS['risk_control'] = True
        94 +      logger.info("✅ Risk control module loaded")
        95 +  except ImportError as e:
        96 +      logger.warning(f"❌ Risk control not available:
           +  {e}")
        97    
        98 -  class PortfolioOptimizationEngine:
        99 -      """投资组合优化引擎主类"""
        98 +  try:
        99 +      from execution_engine.portfolio_executor import
           +  PortfolioExecutor
       100 +      MODULES_STATUS['execution'] = True
       101 +      logger.info("✅ Execution engine module 
           + loaded")
       102 +  except ImportError as e:
       103 +      logger.warning(f"❌ Execution engine not 
           + available: {e}")
       104 +  
       105 +  
       106 +  class AdvancedPortfolioSystem:
       107 +      """高级投资组合系统主类"""
       108        
       109        def __init__(self, 
       110 -                   start_date: str = None, 
       110 +                   start_date: str = None,
       111                     end_date: str = None,
       112 -                   market_index: str = 'SPY'):
       112 +                   market_index: str = 'SPY',
       113 +                   initial_capital: float = 
           + 1000000.0,
       114 +                   risk_budget: float = 0.15):
       115            """
       116 -          初始化优化引擎
       116 +          初始化投资组合系统
       117            
       118            Args:
       119 -              start_date: 开始日期 (格式: YYYY-MM-DD)
       120 -              end_date: 结束日期 (格式: YYYY-MM-DD) 
       121 -              market_index: 市场指数基准
       119 +              start_date: 开始日期 (YYYY-MM-DD)
       120 +              end_date: 结束日期 (YYYY-MM-DD) 
       121 +              market_index: 市场基准指数
       122 +              initial_capital: 初始资金
       123 +              risk_budget: 风险预算
       124            """
       125 -          self.start_date = start_date or 
           -  (datetime.now() - timedelta(days=365*2
           -  )).strftime('%Y-%m-%d')
       125 +          self.start_date = start_date or 
           +  (datetime.now() - timedelta(days=730
           +  )).strftime('%Y-%m-%d')
       126            self.end_date = end_date or 
             datetime.now().strftime('%Y-%m-%d')
       127            self.market_index = market_index
       128 +          self.initial_capital = initial_capital
       129 +          self.risk_budget = risk_budget
       130            
       131 -          logger.info(f"Portfolio Engine initialized 
           - - Period: {self.start_date} to {self.end_date}")
       132 -          logger.info(f"Market Index: 
           - {self.market_index}")
       131 +          # 系统状态
       132 +          self.system_ready = all(MODULES_STATUS[key]
           +  for key in ['data_fetcher', 'alpha_miner', 
           + 'beta_estimator'])
       133            
       134 -          # 存储组件
       134 +          logger.info(f"🚀 Portfolio System 
           + initialized")
       135 +          logger.info(f"📅 Period: {self.start_date} 
           + to {self.end_date}")
       136 +          logger.info(f"📈 Market Index: 
           + {self.market_index}")
       137 +          logger.info(f"💰 Initial Capital: 
           + ${self.initial_capital:,.0f}")
       138 +          logger.info(f"⚠️ Risk Budget: 
           + {self.risk_budget:.1%}")
       139 +          logger.info(f"🔧 System Ready: 
           + {self.system_ready}")
       140 +          
       141 +          # 核心组件
       142            self.data_fetcher = None
       143            self.alpha_miner = None
       144            self.beta_estimator = None
       145            self.optimizer = None
       146 +          self.risk_manager = None
       147 +          self.executor = None
       148            
       149 -          # 存储结果
       150 -          self.raw_data = None
       149 +          # 数据存储
       150 +          self.market_data = {}
       151            self.alpha_factors = None
       152 -          self.beta_estimates = None
       153 -          self.optimization_results = {}
       152 +          self.beta_estimates = {}
       153 +          self.portfolio_weights = {}
       154 +          self.performance_metrics = {}
       155 +          self.risk_metrics = {}
       156 +          
       157 +          # 分析结果
       158 +          self.results = {
       159 +              'system_info': {
       160 +                  'timestamp': 
           + datetime.now().isoformat(),
       161 +                  'modules_status': MODULES_STATUS,
       162 +                  'parameters': {
       163 +                      'start_date': self.start_date,
       164 +                      'end_date': self.end_date,
       165 +                      'market_index': 
           + self.market_index,
       166 +                      'initial_capital': 
           + self.initial_capital,
       167 +                      'risk_budget': self.risk_budget
       168 +                  }
       169 +              }
       170 +          }
       171    
       172 -      def run_demo_analysis(self, tickers: List[str] 
           - = None) -> Dict:
       173 -          """
       174 -          运行演示分析 - 快速展示系统功能
       172 +      def check_system_health(self) -> Dict[str, 
           + Any]:
       173 +          """检查系统健康状态"""
       174 +          print("\n" + "="*60)
       175 +          print("🔍 SYSTEM HEALTH CHECK")
       176 +          print("="*60)
       177            
       178 -          Args:
       179 -              tickers: 股票代码列表
       180 -              
       181 -          Returns:
       182 -              分析结果字典
       183 -          """
       178 +          health_status = {
       179 +              'overall_status': 'healthy',
       180 +              'critical_modules': 0,
       181 +              'warning_modules': 0,
       182 +              'modules_detail': {},
       183 +              'recommendations': []
       184 +          }
       185 +          
       186 +          # 检查核心模块
       187 +          core_modules = ['data_fetcher', 
           + 'alpha_miner', 'beta_estimator', 'optimizer']
       188 +          for module in core_modules:
       189 +              status = MODULES_STATUS.get(module, 
           + False)
       190 +              health_status['modules_detail'][module]
           +  = {
       191 +                  'available': status,
       192 +                  'critical': True
       193 +              }
       194 +              if not status:
       195 +                  health_status['critical_modules'] 
           + += 1
       196 +                  health_status['recommendations'].ap
           + pend(f"Install/fix {module} module")
       197 +                  print(f"❌ {module.replace('_', ' 
           + ').title()}: Critical module missing")
       198 +              else:
       199 +                  print(f"✅ {module.replace('_', ' 
           + ').title()}: Available")
       200 +          
       201 +          # 检查可选模块
       202 +          optional_modules = ['risk_control', 
           + 'execution']
       203 +          for module in optional_modules:
       204 +              status = MODULES_STATUS.get(module, 
           + False)
       205 +              health_status['modules_detail'][module]
           +  = {
       206 +                  'available': status,
       207 +                  'critical': False
       208 +              }
       209 +              if not status:
       210 +                  health_status['warning_modules'] +=
           +  1
       211 +                  
           + health_status['recommendations'].append(f"Consider 
           + installing {module} module for enhanced 
           + functionality")
       212 +                  print(f"⚠️ {module.replace('_', ' 
           + ').title()}: Optional module missing")
       213 +              else:
       214 +                  print(f"✅ {module.replace('_', ' 
           + ').title()}: Available")
       215 +          
       216 +          # 评估整体状态
       217 +          if health_status['critical_modules'] > 0:
       218 +              health_status['overall_status'] = 
           + 'critical'
       219 +              print(f"\n🚨 System Status: CRITICAL 
           + ({health_status['critical_modules']} critical 
           + issues)")
       220 +          elif health_status['warning_modules'] > 2:
       221 +              health_status['overall_status'] = 
           + 'warning'
       222 +              print(f"\n⚠️ System Status: WARNING 
           + ({health_status['warning_modules']} warnings)")
       223 +          else:
       224 +              print(f"\n✅ System Status: HEALTHY")
       225 +          
       226 +          # 打印推荐
       227 +          if health_status['recommendations']:
       228 +              print(f"\n💡 Recommendations:")
       229 +              for i, rec in 
           + enumerate(health_status['recommendations'], 1):
       230 +                  print(f"   {i}. {rec}")
       231 +          
       232 +          self.results['health_check'] = 
           + health_status
       233 +          return health_status
       234 +  
       235 +      def run_quick_analysis(self, tickers: List[str]
           +  = None) -> Dict[str, Any]:
       236 +          """快速分析 - 基础功能演示"""
       237            print("\n" + "="*80)
       238 -          print("🚀 PORTFOLIO OPTIMIZATION DEMO")
       238 +          print("⚡ QUICK PORTFOLIO ANALYSIS")
       239            print("="*80)
       240            
       241 -          # 默认股票池
       241 +          if not self.system_ready:
       242 +              print("❌ System not ready for 
           + analysis")
       243 +              return {'error': 'System not ready', 
           + 'health_check': self.check_system_health()}
       244 +          
       245            if tickers is None:
       246 -              tickers = ["AAPL", "MSFT", "GOOGL", 
           -  "AMZN", "TSLA", "NVDA"]
       247 -              
       248 -          print(f"📊 分析股票: {', '.join(tickers)}")
       249 -          print(f"⏰ 分析时间段: {self.start_date} 到
           -  {self.end_date}")
       246 +              tickers = ["AAPL", "MSFT", "GOOGL", 
           +  "AMZN", "TSLA", "NVDA", "META", "BRK-B"]
       247            
       248 -          results = {}
       248 +          print(f"📊 Analyzing: {', 
           + '.join(tickers)}")
       249 +          print(f"📅 Period: {self.start_date} to 
           + {self.end_date}")
       250            
       251 +          results = {'analysis_type': 'quick', 
           + 'tickers': tickers}
       252 +          
       253            try:
       254                # 1. 数据获取
       255                print(f"\n{'='*50}")
       256 -              print("📥 1. 数据获取阶段")
       256 +              print("📥 1. DATA ACQUISITION")
       257                print(f"{'='*50}")
       258                
       259                self.data_fetcher = 
             EnhancedDataFetcher(self.start_date, self.end_date)
       260                
       261 -              # 获取Alpha研究数据集
       262 -              print("🔍 获取Alpha因子研究数据...")
       263 -              alpha_dataset = 
           - self.data_fetcher.create_alpha_research_dataset(
       261 +              # 获取核心数据集
       262 +              alpha_data = 
           + self.data_fetcher.create_alpha_research_dataset(
       263                    tickers, include_fundamentals=True
       264                )
       265 +              beta_data = self.data_fetcher.create_be
           + ta_research_dataset(tickers)
       266                
       267 -              # 获取Beta风险建模数据集  
       268 -              print("⚖️ 获取Beta风险建模数据...")
       269 -              beta_dataset = self.data_fetcher.create
           - _beta_research_dataset(tickers)
       267 +              print(f"✅ Data acquired: 
           + {alpha_data.shape[0]} observations, {len(tickers)} 
           + assets")
       268                
       269 -              self.raw_data = alpha_dataset
       269                results['data_summary'] = {
       270 -                  'tickers_count': len(tickers),
       271 -                  'alpha_data_shape': 
           - alpha_dataset.shape,
       272 -                  'beta_data_keys': 
           - list(beta_dataset.keys()),
       273 -                  'date_range': 
           - [alpha_dataset['date'].min(), 
           - alpha_dataset['date'].max()]
       270 +                  'observations': 
           + alpha_data.shape[0],
       271 +                  'assets': len(tickers),
       272 +                  'features': alpha_data.shape[1],
       273 +                  'date_range': 
           + [alpha_data['date'].min().strftime('%Y-%m-%d'), 
       274 +                                
           + alpha_data['date'].max().strftime('%Y-%m-%d')]
       275                }
       276                
       277 -              print(f"✅ 数据获取完成")
       278 -              print(f"   - Alpha数据: 
           - {alpha_dataset.shape}")
       279 -              print(f"   - Beta数据集: 
           - {list(beta_dataset.keys())}")
       280 -              
       277                # 2. Alpha因子挖掘
       278                print(f"\n{'='*50}")
       279 -              print("🔬 2. Alpha因子挖掘")
       279 +              print("🔬 2. ALPHA FACTOR MINING")
       280                print(f"{'='*50}")
       281                
       282                self.alpha_miner = RealAlphaMiner(
       283 -                  data=alpha_dataset,
       284 -                  feature_windows=[5, 10, 20],
       285 -                  prediction_horizons=[1, 5, 10],
       283 +                  data=alpha_data,
       284 +                  feature_windows=[5, 10, 20, 60],
       285 +                  prediction_horizons=[1, 5, 10, 20],
       286                    min_ic_threshold=0.01
       287                )
       288                
       289 -              print("⚙️ 开始因子挖掘...")
       289 +              print("⚙️ Mining alpha factors...")
       290                self.alpha_factors = 
             self.alpha_miner.mine_all_alpha_factors()
       291                
       292 -              # 展示Top因子
       293 -              if hasattr(self.alpha_miner, 
           -  'factor_performance') and \
       294 -                 
           - isinstance(self.alpha_miner.factor_performance, 
           - dict) == False and \
       295 -                 not 
           - self.alpha_miner.factor_performance.empty:
       292 +              # 因子性能分析
       293 +              if hasattr(self.alpha_miner, 
           +  'factor_performance') and 
           +  not self.alpha_miner.factor_performance.empty:
       294 +                  perf_df = 
           + self.alpha_miner.factor_performance
       295                    
       296 -                  print("\n📊 Top 10 Alpha因子 
           - (按|IC|排序):")
       297 -                  # 安全地找到IC列
       298 -                  ic_col = None
       299 -                  for col in ['ic_1d', 'IC', 
           - 'ic_mean']:
       300 -                      if col in 
           - self.alpha_miner.factor_performance.columns:
       301 -                          ic_col = col
       302 -                          break
       303 -                  
       304 -                  if ic_col:
       305 -                      top_factors = 
           - self.alpha_miner.factor_performance.sort_values(
       306 -                          ic_col, key=lambda x: 
           - x.abs(), ascending=False
       307 -                      ).head(10)
       308 -                  else:
       309 -                      # 
           - 如果找不到IC列，按第一个数值列排序
       310 -                      numeric_cols = 
           - self.alpha_miner.factor_performance.select_dtypes(in
           - clude=[np.number]).columns
       311 -                      if len(numeric_cols) > 0:
       312 -                          top_factors = 
           - self.alpha_miner.factor_performance.sort_values(
       313 -                              numeric_cols[0], 
           - ascending=False
       314 -                          ).head(10)
       315 -                      else:
       316 -                          top_factors = 
           - self.alpha_miner.factor_performance.head(10)
       317 -                  
       318 -                  for _, row in 
           - top_factors.iterrows():
       319 -                      ic = row.get(ic_col, 0) if 
           - ic_col else 0
       320 -                      ic_ir = row.get('ic_ir_1d', 0) 
       321 -                      coverage = row.get('coverage', 
           - 0)
       322 -                      print(f"   
           - {row['factor'][:30]:<30} IC={ic:>7.4f} 
           - IR={ic_ir:>6.2f} Coverage={coverage:>6.1%}")
       296 +                  # 安全地获取IC列
       297 +                  ic_cols = [col for col in 
           + perf_df.columns if 'ic' in col.lower()]
       298 +                  if ic_cols:
       299 +                      ic_col = ic_cols[0]
       300 +                      top_factors = 
           + perf_df.nlargest(10, ic_col, keep='first')
       301                        
       302 -                  results['alpha_summary'] = {
       303 -                      'total_factors': 
           - len(self.alpha_miner.factor_performance),
       304 -                      'top_factor': 
           - top_factors.iloc[0]['factor'],
       305 -                      'best_ic': 
           - top_factors.iloc[0].get(ic_col, 0) if ic_col else 0
       306 -                  }
       302 +                      print(f"\n📊 Top 10 Alpha 
           + Factors (by {ic_col}):")
       303 +                      print("-" * 80)
       304 +                      for idx, row in 
           + top_factors.iterrows():
       305 +                          factor_name = 
           + row.get('factor', f'Factor_{idx}')[:35]
       306 +                          ic_val = row.get(ic_col, 0)
       307 +                          ir_val = 
           + row.get('ic_ir_1d', row.get('ic_ir', 0))
       308 +                          coverage = 
           + row.get('coverage', 0)
       309 +                          print(f"  {factor_name:<35}
           +  IC:{ic_val:>8.4f} IR:{ir_val:>7.2f} 
           + Cov:{coverage:>7.1%}")
       310 +                      
       311 +                      results['alpha_summary'] = {
       312 +                          'total_factors': 
           + len(perf_df),
       313 +                          'effective_factors': 
           + len(self.alpha_factors.columns) - 2,
       314 +                          'best_ic': 
           + float(top_factors.iloc[0].get(ic_col, 0)),
       315 +                          'avg_ic': 
           + float(perf_df[ic_col].mean()),
       316 +                          'top_factor': 
           + top_factors.iloc[0].get('factor', 'Unknown')
       317 +                      }
       318 +                  else:
       319 +                      print("⚠️ No IC metrics 
           + available")
       320 +                      results['alpha_summary'] = 
           + {'total_factors': 0, 'warning': 'No IC metrics'}
       321                else:
       322 -                  print("⚠️ 因子性能评估结果不可用")
       323 -                  results['alpha_summary'] = {'
           -  total_factors': 0}
       322 +                  print("⚠️ Factor performance 
           + evaluation unavailable")
       323 +                  results['alpha_summary'] = {'error
           +  ': 'Performance evaluation failed'}
       324                
       325 -              print(f"✅ Alpha因子挖掘完成
           -  : {self.alpha_factors.shape[1]-2} 个有效因子")
       325 +              print(f"✅ Alpha mining completed
           +  : {self.alpha_factors.shape[1]-2} factors extracted
           +  ")
       326                
       327 -              # 3. Beta风险估计  
       327 +              # 3. Beta风险估计
       328                print(f"\n{'='*50}")
       329 -              print("📈 3. Beta风险估计")
       329 +              print("📊 3. BETA RISK ESTIMATION") 
       330                print(f"{'='*50}")
       331                
       332 -              self.beta_estimator = 
           -  RealBetaEstimator(data=beta_dataset)
       332 +              self.beta_estimator = 
           +  RealBetaEstimator(data=beta_data)
       333                
       334 -              print("⚙️ 开始Beta估计...")
       334 +              print("⚙️ Estimating beta coefficients
           +  ...")
       335                self.beta_estimates = 
             self.beta_estimator.estimate_all_betas()
       336                
       337                # 展示Beta结果
       338 +              beta_summary = {}
       339                for method, data in 
             self.beta_estimates.items():
       340                    if not data.empty:
       341 -                      print(f"\n📊 {method.upper()} 
           -  结果:")
       341 +                      print(f"\n📈 {method.upper()
           +  .replace('_', ' ')} Results:")
       342 +                      
       343                        if method == 'capm_beta':
       344 -                          display_cols = [
           -  'ticker', 'beta', 'r_squared']
       345 -                          display_cols = [c for c in 
           - display_cols if c in data.columns]
       346 -                          print(data[display_cols].ro
           - und(4).to_string(index=False))
       344 +                          display_data = data[[
           +  'ticker', 'beta', 'r_squared', 'alpha']].round(4)
       345 +                          beta_summary[method] = {
       346 +                              'avg_beta': 
           + float(data['beta'].mean()),
       347 +                              'beta_range': 
           + [float(data['beta'].min()), 
           + float(data['beta'].max())],
       348 +                              'avg_r_squared': 
           + float(data['r_squared'].mean())
       349 +                          }
       350                        elif method == 
             'multi_factor_beta':
       351 -                          display_cols = [
           -  'ticker', 'alpha'] + [
           -  c for c in data.columns if c.startswith('beta_')]
       352 -                          display_cols = [c for c in 
           - display_cols if c in data.columns]
       353 -                          if display_cols:
       354 -                              print(data[display_cols
           - ].round(4).to_string(index=False))
       355 -                          
       351 +                          factor_cols = [
           +  c for c in data.columns if c.startswith('beta_')]
       352 +                          display_cols = ['ticker', 
           + 'alpha'] + factor_cols[:3]  # 显示前3个因子
       353 +                          display_data = data[
           +  display_cols].round(4)
       354 +                          beta_summary[method] = {
       355 +                              'factors_count': 
           + len(factor_cols),
       356 +                              'avg_alpha': 
           + float(data['alpha'].mean()) if 'alpha' in 
           + data.columns else 0
       357 +                          }
       358 +                      else:
       359 +                          display_data = data.head()
       360 +                          beta_summary[method] = 
           + {'observations': len(data)}
       361 +                      
       362 +                      
           + print(display_data.to_string(index=False))
       363 +              
       364                results['beta_summary'] = {
       365 -                  'methods
           -  ': list(self.beta_estimates.keys()),
       366 -                  'capm_available': not 
           - self.beta_estimates.get('capm_beta', 
           - pd.DataFrame()).empty
       365 +                  'methods_available
           +  ': list(self.beta_estimates.keys()),
       366 +                  'details': beta_summary
       367                }
       368                
       369 -              print(f"✅ Beta风险估计完成
           -  : {len(self.beta_estimates)} 种方法")
       369 +              print(f"✅ Beta estimation completed
           +  : {len(self.beta_estimates)} methods")
       370                
       371 -              # 4. 投资组合优化
       371 +              # 4. 快速投资组合优化
       372                print(f"\n{'='*50}")
       373 -              print("🎯 4. 投资组合优化") 
       373 +              print("🎯 4. PORTFOLIO OPTIMIZATION")
       374                print(f"{'='*50}")
       375                
       376 -              if DATA_MODULES_AVAILABLE:
       376 +              if MODULES_STATUS['optimizer']:
       377                    try:
       378 -                      # 创建优化器
       378                        self.optimizer = 
             AlphaBetaOptimizer(
       379 -                          data=alpha_dataset,
       379 +                          data=alpha_data,
       380                            
             market_index=self.market_index
       381                        )
       382                        
       383 -                      # 运行不同的优化方法
       383 +                      # 运行多种优化方法
       384                        optimization_methods = 
             ['max_sharpe', 'min_variance', 'risk_parity']
       385 +                      opt_results = {}
       386                        
       387                        for method in 
             optimization_methods:
       388 -                          print(f"⚙️ 运行 {method} 优化
           -  ...")
       388 +                          print(f"⚙️ Running
           +   {method} optimization...")
       389                            try:
       390                                result = 
             self.optimizer.optimize_portfolio(
       391                                    assets=tickers,
       392                                    method=method,
       393 -                                  risk_aversion=2.0
           -   if method == 'max_utility' else 1.0
       393 +                                  risk_aversion=2.0
       394                                )
       395                                
       396                                if 
             result.get('success', False):
       397 -                                  
           -  self.optimization_results[method] = result
       398 -                                  print(f"✅ {method}
           -  优化成功")
       399 -                                  print(f"   
           - 预期收益: {result.get('expected_return', 0):.2%}")
       400 -                                  print(f"   波动率: 
           - {result.get('volatility', 0):.2%}")
       401 -                                  print(f"   
           - 夏普比率: {result.get('sharpe_ratio', 0):.4f}")
       397 +                                  opt_results
           +  [method] = result
       398                                    
       399 -                                  # 显示前三大持仓
       400 -                                  weights = result['
           -  weights']
       399 +                                  # 格式化输出
       400 +                                  ret = result.get('
           +  expected_return', 0)
       401 +                                  vol = 
           + result.get('volatility', 0) 
       402 +                                  sharpe = 
           + result.get('sharpe_ratio', 0)
       403 +                                  
       404 +                                  print(f"  ✅ 
           + {method}: Return={ret:.2%} Vol={vol:.2%} 
           + Sharpe={sharpe:.3f}")
       405 +                                  
       406 +                                  # 显示主要持仓
       407 +                                  weights = 
           + np.array(result['weights'])
       408                                    assets = 
             result['assets']
       409 -                                  top_3
           -   = sorted(zip(assets, weights), key=lambda x: x[1],
           -   reverse=True)[:3]
       410 -                                  print("   
           - 前三大持仓:")
       411 -                                  for asset, weight 
           - in top_3:
       412 -                                      print(f"     
           - {asset}: {weight:.2%}")
       409 +                                  top_holdings
           +   = sorted(zip(assets, weights), key=lambda x: x[1],
           +   reverse=True)[:3]
       410 +                                  holdings_str = ", 
           + ".join([f"{asset}:{weight:.1%}" for asset, weight in
           +  top_holdings])
       411 +                                  print(f"      Top 
           + holdings: {holdings_str}")
       412                                else:
       413 -                                  print(f"❌ {method}
           -   优化失败: {result.get('message', '未知错误')}")
       414 -                                  
       413 +                                  print(f"  
           +  ❌ {method}: {result.get('message', '
           +  Optimization failed')}")
       414 +                          
       415                            except Exception as e:
       416 -                              print(f"❌ {method} 
     优化出错
           -  : {e}")
       416 +                              print(f"  ❌ {method}: 
           +  Error - {str(e)[:50]}")
       417                        
       418                        results['optimization_summary']
              = {
       419 -                          'successful_methods': 
           - len([r for r in self.optimization_results.values() 
           - if r.get('success')]),
       420 -                          'total_methods
           -  ': len(optimization_methods),
       421 -                          'best_method': 
           - max(self.optimization_results.keys(), 
       422 -                                           key=lambda
           -  k: self.optimization_results[k].get('sharpe_ratio',
           -  -999)) 
       423 -                                           if 
           - self.optimization_results else None
       419 +                          'successful_methods': 
           + len(opt_results),
       420 +                          'methods_attempted
           +  ': len(optimization_methods),
       421 +                          'results': {
       422 +                              method: {
       423 +                                  'return': 
           + float(res.get('expected_return', 0)),
       424 +                                  'volatility': 
           + float(res.get('volatility', 0)),
       425 +                                  'sharpe_ratio': 
           + float(res.get('sharpe_ratio', 0))
       426 +                              }
       427 +                              for method, res in 
           + opt_results.items()
       428 +                          }
       429                        }
       430                        
       431 +                      # 选择最佳策略
       432 +                      if opt_results:
       433 +                          best_method = 
           + max(opt_results.keys(), 
       434 +                                          key=lambda 
           + k: opt_results[k].get('sharpe_ratio', -999))
       435 +                          print(f"\n🏆 Best 
           + performing method: {best_method}")
       436 +                          
           + results['optimization_summary']['best_method'] = 
           + best_method
       437 +                          
       438                    except Exception as e:
       439 -                      print(f"❌ 组合优化模块出错: 
           - {e}")
       439 +                      logger.error(f"Portfolio 
           + optimization error: {e}")
       440                        results['optimization_summary']
              = {'error': str(e)}
       441 +                      print(f"❌ Portfolio 
           + optimization failed: {e}")
       442                else:
       443 -                  print("⚠️ 
           - 优化模块不可用，跳过投资组合优化")
       444 -                  results['optimization_summary'] = 
           -  {'error': 'Module not available'}
       445 -                  
       446 -              print(f"✅ 投资组合优化完成")
       443 +                  print("❌ Portfolio optimizer not 
           + available")
       444 +                  results['optimization_summary'] = 
           +  {'error': 'Optimizer module not available'}
       445                
       446 +              print(f"✅ Portfolio optimization 
           + completed")
       447 +              
       448            except Exception as e:
       449 -              logger.error(f"Demo分析出错: {e}")
       449 +              logger.error(f"Quick analysis error
           +  : {e}")
       450 +              logger.error(traceback.format_exc())
       451                results['error'] = str(e)
       452 -              
       453 -          # 5. 总结报告
       452 +              print(f"❌ Analysis failed: {e}")
       453 +          
       454 +          # 5. 结果总结
       455            print(f"\n{'='*80}")
       456 -          print("📋 5. 分析总结")
       457 -          print(f"{'='*80}")
       456 +          print("📋 ANALYSIS SUMMARY")
       457 +          print("="*80)
       458            
       459 -          self._print_summary_report(results)
       459 +          self._print_quick_summary(results)
       460            
       461 +          # 保存结果
       462 +          self.results['quick_analysis'] = results
       463 +          self._save_analysis_results('quick')
       464 +          
       465            return results
       466    
       467 -      def run_full_analysis(self, 
       468 -                           tickers: List[str],
       469 -                           optimization_methods: 
           - List[str] = None,
       470 -                           save_results: bool = True)
           -  -> Dict:
       471 -          """
       472 -          运行完整分析流程
       473 -          
       474 -          Args:
       475 -              tickers: 股票代码列表
       476 -              optimization_methods: 优化方法列表
       477 -              save_results: 是否保存结果
       478 -              
       479 -          Returns:
       480 -              完整分析结果
       481 -          """
       467 +      def run_comprehensive_analysis(self, 
       468 +                                   tickers: 
           + List[str],
       469 +                                   save_results: bool
           +  = True,
       470 +                                   run_backtest: bool
           +  = True) -> Dict[str, Any]:
       471 +          """运行全面分析"""
       472            print(f"\n{'='*80}")
       473 -          print("🔬 COMPREHENSIVE PORTFOLIO 
           -  ANALYSIS")
       474 -          print(f"{'='*80}")
       473 +          print("🔬 COMPREHENSIVE PORTFOLIO 
           +  ANALYSIS")  
       474 +          print("="*80)
       475            
       476 -          if optimization_methods is None:
       477 -              optimization_methods = ['max_sharpe', 
           - 'min_variance', 'max_utility', 'risk_parity']
       478 -              
       479 -          # 运行基础分析
       480 -          results = self.run_demo_analysis(tickers)
       476 +          # 首先运行快速分析
       477 +          results = self.run_quick_analysis(tickers)
       478            
       479 -          if self.optimization_results and 
           - DATA_MODULES_AVAILABLE:
       480 -              try:
       481 -                  # 策略回测
       479 +          if 'error' in results:
       480 +              return results
       481 +          
       482 +          try:
       483 +              # 风险控制分析
       484 +              if MODULES_STATUS['risk_control'] and 
           + run_backtest:
       485                    print(f"\n{'='*50}")
       486 -                  print("📈 策略回测")
       486 +                  print("⚠️ RISK CONTROL ANALYSIS")
       487                    print(f"{'='*50}")
       488                    
       489 -                  best_method = 
           - results.get('optimization_summary', 
           - {}).get('best_method', 'max_sharpe')
       490 -                  if best_method in 
           - self.optimization_results:
       491 -                      print(f"⚙️ 使用 {best_method} 
           - 策略进行回测...")
       492 -                      
       493 -                      # 
           - 选择部分股票进行回测（提高速度）
       494 -                      test_assets = tickers[:min(6, 
           - len(tickers))]
       495 -                      
       496 -                      backtest_results = 
           - self.optimizer.backtest_strategy(
       497 -                          assets=test_assets,
       498 -                          
           - rebalance_frequency='monthly',
       499 -                          lookback_period=120,
       500 -                          method=best_method
       489 +                  self.risk_manager = RiskManager(
       490 +                      
           + initial_capital=self.initial_capital,
       491 +                      risk_budget=self.risk_budget
       492 +                  )
       493 +                  
       494 +                  # 风险度量计算
       495 +                  if hasattr(self.alpha_factors, 
           + 'values'):
       496 +                      risk_metrics = 
           + self.risk_manager.calculate_risk_metrics(
       497 +                          
           + returns_data=self.alpha_factors,
       498 +                          assets=tickers[:6]  # 
           + 限制资产数量提高性能
       499                        )
       500                        
       501 -                      if not backtest_results.empty:
       502 -                          total_ret = 
           - backtest_results['cumulative_return'].iloc[-1] - 1
       503 -                          annual_ret = 
           - backtest_results['portfolio_return'].mean() * 252
       504 -                          annual_vol = 
           - backtest_results['portfolio_return'].std() * 
           - (252**0.5)
       505 -                          sharpe = annual_ret / 
           - annual_vol if annual_vol > 0 else 0
       501 +                      if risk_metrics:
       502 +                          print("✅ Risk metrics 
           + calculated:")
       503 +                          for metric, value in 
           + risk_metrics.items():
       504 +                              if isinstance(value, 
           + (int, float)):
       505 +                                  if 'var' in 
           + metric.lower() or 'cvar' in metric.lower():
       506 +                                      print(f"   
           + {metric}: ${value:,.0f}")
       507 +                                  else:
       508 +                                      print(f"   
           + {metric}: {value:.4f}")
       509                            
       510 -                          print(f"✅ 回测完成:")
       511 -                          print(f"   总回报: 
           - {total_ret:.2%}")
       512 -                          print(f"   年化收益: 
           - {annual_ret:.2%}")
       513 -                          print(f"   年化波动: 
           - {annual_vol:.2%}")
       514 -                          print(f"   夏普比率: 
           - {sharpe:.4f}")
       515 -                          
       516 -                          results['backtest_summary']
           -  = {
       517 -                              'total_return': 
           - total_ret,
       518 -                              'annual_return': 
           - annual_ret,
       519 -                              'annual_volatility': 
           - annual_vol,
       520 -                              'sharpe_ratio': sharpe,
       521 -                              'periods': 
           - len(backtest_results)
       522 -                          }
       510 +                          results['risk_analysis'] = 
           + risk_metrics
       511                    
       512 -                  # 保存结果
       513 -                  if save_results:
       514 -                      self._save_results(results)
       512 +                  print("✅ Risk control analysis 
           + completed")
       513 +              
       514 +              # 策略回测
       515 +              if run_backtest and 
           + MODULES_STATUS['optimizer'] and self.optimizer:
       516 +                  print(f"\n{'='*50}")
       517 +                  print("📈 STRATEGY BACKTESTING")
       518 +                  print(f"{'='*50}")
       519 +                  
       520 +                  # 选择最佳策略进行回测
       521 +                  best_method = 
           + results.get('optimization_summary', 
           + {}).get('best_method', 'max_sharpe')
       522 +                  
       523 +                  if best_method:
       524 +                      print(f"⚙️ Backtesting 
           + {best_method} strategy...")
       525                        
       526 -              except Exception as e:
       527 -                  logger.error(f"完整分析出错: {e}")
       528 -                  results['backtest_error'] = str(e)
       526 +                      try:
       527 +                          # 
           + 使用较少资产进行回测以提高性能
       528 +                          backtest_assets = 
           + tickers[:min(6, len(tickers))]
       529 +                          
       530 +                          backtest_results = 
           + self.optimizer.backtest_strategy(
       531 +                              assets=backtest_assets,
       532 +                              
           + rebalance_frequency='monthly',
       533 +                              lookback_period=120,
       534 +                              method=best_method
       535 +                          )
       536 +                          
       537 +                          if not 
           + backtest_results.empty:
       538 +                              # 计算关键指标
       539 +                              total_return = 
           + backtest_results['cumulative_return'].iloc[-1] - 1
       540 +                              returns = 
           + backtest_results['portfolio_return'].dropna()
       541 +                              
       542 +                              annual_return = 
           + returns.mean() * 252
       543 +                              annual_vol = 
           + returns.std() * np.sqrt(252)
       544 +                              sharpe_ratio = 
           + annual_return / annual_vol if annual_vol > 0 else 0
       545 +                              
       546 +                              max_dd = 
           + self._calculate_max_drawdown(backtest_results['cumul
           + ative_return'])
       547 +                              
       548 +                              print(f"✅ Backtest 
           + completed ({len(backtest_results)} periods):")
       549 +                              print(f"   Total 
           + Return: {total_return:.2%}")
       550 +                              print(f"   Annual 
           + Return: {annual_return:.2%}")
       551 +                              print(f"   Annual 
           + Volatility: {annual_vol:.2%}")
       552 +                              print(f"   Sharpe 
           + Ratio: {sharpe_ratio:.3f}")
       553 +                              print(f"   Maximum 
           + Drawdown: {max_dd:.2%}")
       554 +                              
       555 +                              
           + results['backtest_summary'] = {
       556 +                                  'strategy': 
           + best_method,
       557 +                                  'total_return': 
           + float(total_return),
       558 +                                  'annual_return': 
           + float(annual_return),
       559 +                                  
           + 'annual_volatility': float(annual_vol),
       560 +                                  'sharpe_ratio': 
           + float(sharpe_ratio),
       561 +                                  'max_drawdown': 
           + float(max_dd),
       562 +                                  'periods': 
           + len(backtest_results)
       563 +                              }
       564 +                              
       565 +                              # 保存回测结果
       566 +                              backtest_path = 
           + f'results/backtest_results_{datetime.now().strftime(
           + "%Y%m%d_%H%M%S")}.csv'
       567 +                              
           + backtest_results.to_csv(backtest_path, index=False)
       568 +                              print(f"💾 Backtest 
           + results saved: {backtest_path}")
       569 +                          
       570 +                      except Exception as e:
       571 +                          logger.error(f"Backtesting 
           + error: {e}")
       572 +                          results['backtest_error'] =
           +  str(e)
       573 +                          print(f"❌ Backtesting 
           + failed: {e}")
       574 +              
       575 +              # 保存全面分析结果
       576 +              if save_results:
       577 +                  
           + self.results['comprehensive_analysis'] = results
       578 +                  
           + self._save_analysis_results('comprehensive')
       579 +                  
           + self._generate_analysis_report(results)
       580 +              
       581 +          except Exception as e:
       582 +              logger.error(f"Comprehensive analysis 
           + error: {e}")
       583 +              results['comprehensive_error'] = str(e)
       584            
       585            return results
       586    
       587 -      def run_custom_analysis(self, config: Dict) -> 
           - Dict:
       588 -          """
       589 -          运行自定义分析
       590 -          
       591 -          Args:
       592 -              config: 自定义配置字典
       593 -              
       594 -          Returns:
       595 -              分析结果
       596 -          """
       587 +      def run_live_trading_setup(self, 
       588 +                                tickers: List[str],
       589 +                                trading_config: 
           + Dict[str, Any] = None) -> Dict[str, Any]:
       590 +          """实盘交易设置"""
       591            print(f"\n{'='*80}")
       592 -          print("⚙️ CUSTOM PORTFOLIO ANALYSIS")
       593 -          print(f"{'='*80}")
       592 +          print("🔴 LIVE TRADING SETUP")
       593 +          print("="*80)
       594            
       595 -          tickers = config.get('tickers', ['AAPL', 
           - 'MSFT', 'GOOGL'])
       596 -          methods = 
           - config.get('optimization_methods', ['max_sharpe'])
       595 +          if not MODULES_STATUS['execution']:
       596 +              print("❌ Execution engine not 
           + available for live trading")
       597 +              return {'error': 'Execution engine not 
           + available'}
       598            
       599 -          return self.run_full_analysis(tickers, 
           - methods, config.get('save_results', True))
       599 +          if trading_config is None:
       600 +              trading_config = {
       601 +                  'rebalance_frequency': 'daily',
       602 +                  'max_position_size': 0.15,
       603 +                  'stop_loss': 0.05,
       604 +                  'take_profit': 0.20
       605 +              }
       606 +          
       607 +          print("⚠️  WARNING: Live trading involves 
           + real money and risks!")
       608 +          print("⚠️  This is a demonstration setup 
           + only.")
       609 +          print(f"📊 Assets: {', '.join(tickers)}")
       610 +          print(f"⚙️  Config: {trading_config}")
       611 +          
       612 +          try:
       613 +              # 初始化执行引擎
       614 +              self.executor = PortfolioExecutor(
       615 +                  
           + initial_capital=self.initial_capital,
       616 +                  config=trading_config
       617 +              )
       618 +              
       619 +              # 设置监控系统
       620 +              setup_result = 
           + self.executor.setup_live_trading(
       621 +                  assets=tickers,
       622 +                  strategy_config=trading_config
       623 +              )
       624 +              
       625 +              results = {
       626 +                  'live_trading_setup': setup_result,
       627 +                  'config': trading_config,
       628 +                  'status': 'ready' if 
           + setup_result.get('success') else 'failed'
       629 +              }
       630 +              
       631 +              if setup_result.get('success'):
       632 +                  print("✅ Live trading setup 
           + completed")
       633 +                  print("💡 Use separate monitoring 
           + script to start live trading")
       634 +              else:
       635 +                  print(f"❌ Live trading setup 
           + failed: {setup_result.get('error', 'Unknown 
           + error')}")
       636 +              
       637 +          except Exception as e:
       638 +              logger.error(f"Live trading setup 
           + error: {e}")
       639 +              results = {'error': str(e)}
       640 +              print(f"❌ Live trading setup failed: 
           + {e}")
       641 +          
       642 +          return results
       643    
       644 -      def _print_summary_report(self, results: Dict):
       645 -          """打印总结报告"""
       646 -          print(f"🎯 分析股票数量: 
           - {results.get('data_summary', 
           - {}).get('tickers_count', 0)}")
       647 -          print(f"📊 有效Alpha因子: 
           - {results.get('alpha_summary', 
           - {}).get('total_factors', 0)}")
       648 -          print(f"📈 Beta估计方法: 
           - {len(results.get('beta_summary', {}).get('methods', 
           - []))}")
       649 -          print(f"🎯 成功优化方法: 
           - {results.get('optimization_summary', 
           - {}).get('successful_methods', 0)}")
       644 +      def _calculate_max_drawdown(self, 
           + cumulative_returns: pd.Series) -> float:
       645 +          """计算最大回撤"""
       646 +          try:
       647 +              peak = 
           + cumulative_returns.expanding(min_periods=1).max()
       648 +              drawdown = (cumulative_returns / peak) 
           + - 1
       649 +              return float(drawdown.min())
       650 +          except:
       651 +              return 0.0
       652 +  
       653 +      def _print_quick_summary(self, results: 
           + Dict[str, Any]):
       654 +          """打印快速分析总结"""
       655 +          data_summary = results.get('data_summary', 
           + {})
       656 +          alpha_summary = 
           + results.get('alpha_summary', {})
       657 +          beta_summary = results.get('beta_summary', 
           + {})
       658 +          opt_summary = 
           + results.get('optimization_summary', {})
       659            
       660 +          print(f"📊 Data: 
           + {data_summary.get('observations', 0)} obs, 
           + {data_summary.get('assets', 0)} assets")
       661 +          print(f"🔬 Alpha: 
           + {alpha_summary.get('effective_factors', 0)} factors 
           + (IC: {alpha_summary.get('best_ic', 0):.3f})")
       662 +          print(f"📈 Beta: 
           + {len(beta_summary.get('methods_available', []))} 
           + methods")
       663 +          print(f"🎯 Optimization: 
           + {opt_summary.get('successful_methods', 
           + 0)}/{opt_summary.get('methods_attempted', 0)} 
           + successful")
       664 +          
       665            if 'backtest_summary' in results:
       666                bt = results['backtest_summary']
       667 -              print(f"📈 回测总回报
           -  : {bt.get('total_return', 0):.2%}")
       668 -              print(f"📊 回测夏普比率: 
           - {bt.get('sharpe_ratio', 0):.4f}")
       669 -              
       670 -          print(f"⏰ 分析完成时间: 
           - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
       667 +              print(f"📈 Backtest
           +  : {bt.get('total_return', 0):.2%}
           +   total return, {bt.get('sharpe_ratio', 0):.3f} 
           +  Sharpe")
       668 +          
       669 +          print(f"⏰ Completed: 
           + {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
       670    
       671 -      def _save_results(self, results: Dict):
       671 +      def _save_analysis_results(self, analysis_type:
           +  str):
       672            """保存分析结果"""
       673            timestamp = 
             datetime.now().strftime('%Y%m%d_%H%M%S')
       674            
       675            try:
       676 -              # 保存Alpha因子
       677 -              if self.alpha_factors is not None:
       678 -                  alpha_path = 
           - f'results/alpha_factors_{timestamp}.csv'
       679 -                  
           - self.alpha_factors.to_csv(alpha_path, index=False)
       680 -                  print(f"💾 Alpha因子已保存: 
           - {alpha_path}")
       676 +              # 保存主要结果文件
       677 +              results_file = 
           + f'results/analysis_{analysis_type}_{timestamp}.json'
       678 +              with open(results_file, 'w') as f:
       679 +                  json.dump(self.results, f, 
           + indent=2, default=str)
       680                
       681 -              # 保存Beta估计
       681 +              print(f"💾 Analysis results saved: 
           + {results_file}")
       682 +              
       683 +              # 保存关键数据文件
       684 +              if self.alpha_factors is not None and 
           + not self.alpha_factors.empty:
       685 +                  alpha_file = 
           + f'results/alpha_factors_{timestamp}.csv'
       686 +                  
           + self.alpha_factors.to_csv(alpha_file, index=False)
       687 +                  print(f"💾 Alpha factors saved: 
           + {alpha_file}")
       688 +              
       689                if self.beta_estimates:
       690                    for method, data in 
             self.beta_estimates.items():
       691                        if not data.empty:
       692 -                          beta_path
           -   = f'results/beta_{method}_{timestamp}.csv'
       693 -                          data.to_csv(beta_path
           -  , index=False)
       694 -                  print(f"💾 Beta估计已保存
           -  : results/beta_*_{timestamp}.csv")
       692 +                          beta_file
           +   = f'results/beta_{method}_{timestamp}.csv'
       693 +                          data.to_csv(beta_file
           +  , index=False)
       694 +                  print(f"💾 Beta estimates saved
           +  : results/beta_*_{timestamp}.csv")
       695                
       696 -              # 保存优化结果
       697 -              if self.optimization_results:
       698 -                  import json
       699 -                  opt_path = 
           - f'results/optimization_results_{timestamp}.json'
       700 -                  # 转换numpy数组为列表以便JSON序列化
       701 -                  save_results = {}
       702 -                  for method, result in 
           - self.optimization_results.items():
       703 -                      save_results[method] = {
       704 -                          k: v.tolist() if hasattr(v,
           -  'tolist') else v 
       705 -                          for k, v in result.items()
       706 -                      }
       707 -                      
       708 -                  with open(opt_path, 'w') as f:
       709 -                      json.dump(save_results, f, 
           - indent=2, default=str)
       710 -                  print(f"💾 优化结果已保存: 
           - {opt_path}")
       696 +          except Exception as e:
       697 +              logger.warning(f"Error saving results: 
           + {e}")
       698 +  
       699 +      def _generate_analysis_report(self, results: 
           + Dict[str, Any]):
       700 +          """生成分析报告"""
       701 +          timestamp = 
           + datetime.now().strftime('%Y%m%d_%H%M%S')
       702 +          report_file = 
           + f'results/analysis_report_{timestamp}.md'
       703 +          
       704 +          try:
       705 +              with open(report_file, 'w') as f:
       706 +                  f.write(f"# Portfolio Analysis 
           + Report\n\n")
       707 +                  f.write(f"**Generated:** 
           + {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
       708                    
       709 +                  # 系统信息
       710 +                  f.write(f"## System Information\n")
       711 +                  f.write(f"- Analysis Period: 
           + {self.start_date} to {self.end_date}\n")
       712 +                  f.write(f"- Market Index: 
           + {self.market_index}\n")
       713 +                  f.write(f"- Initial Capital: 
           + ${self.initial_capital:,.0f}\n\n")
       714 +                  
       715 +                  # 数据摘要
       716 +                  data_summary = 
           + results.get('data_summary', {})
       717 +                  f.write(f"## Data Summary\n")
       718 +                  f.write(f"- Observations: 
           + {data_summary.get('observations', 0)}\n")
       719 +                  f.write(f"- Assets: 
           + {data_summary.get('assets', 0)}\n")
       720 +                  f.write(f"- Features: 
           + {data_summary.get('features', 0)}\n\n")
       721 +                  
       722 +                  # Alpha因子
       723 +                  alpha_summary = 
           + results.get('alpha_summary', {})
       724 +                  f.write(f"## Alpha Factor 
           + Analysis\n")
       725 +                  f.write(f"- Total Factors: 
           + {alpha_summary.get('total_factors', 0)}\n")
       726 +                  f.write(f"- Effective Factors: 
           + {alpha_summary.get('effective_factors', 0)}\n")
       727 +                  f.write(f"- Best IC: 
           + {alpha_summary.get('best_ic', 0):.4f}\n")
       728 +                  f.write(f"- Average IC: 
           + {alpha_summary.get('avg_ic', 0):.4f}\n\n")
       729 +                  
       730 +                  # 投资组合优化
       731 +                  opt_summary = 
           + results.get('optimization_summary', {})
       732 +                  f.write(f"## Portfolio 
           + Optimization\n")
       733 +                  f.write(f"- Successful Methods: 
           + {opt_summary.get('successful_methods', 0)}\n")
       734 +                  
       735 +                  if 'results' in opt_summary:
       736 +                      f.write(f"\n### Optimization 
           + Results\n")
       737 +                      for method, metrics in 
           + opt_summary['results'].items():
       738 +                          
           + f.write(f"**{method.replace('_', ' 
           + ').title()}:**\n")
       739 +                          f.write(f"- Expected 
           + Return: {metrics.get('return', 0):.2%}\n")
       740 +                          f.write(f"- Volatility: 
           + {metrics.get('volatility', 0):.2%}\n")
       741 +                          f.write(f"- Sharpe Ratio: 
           + {metrics.get('sharpe_ratio', 0):.3f}\n\n")
       742 +                  
       743 +                  # 回测结果
       744 +                  if 'backtest_summary' in results:
       745 +                      bt = 
           + results['backtest_summary']
       746 +                      f.write(f"## Backtest 
           + Results\n")
       747 +                      f.write(f"- Strategy: 
           + {bt.get('strategy', 'N/A')}\n")
       748 +                      f.write(f"- Total Return: 
           + {bt.get('total_return', 0):.2%}\n")
       749 +                      f.write(f"- Annual Return: 
           + {bt.get('annual_return', 0):.2%}\n")
       750 +                      f.write(f"- Annual Volatility: 
           + {bt.get('annual_volatility', 0):.2%}\n")
       751 +                      f.write(f"- Sharpe Ratio: 
           + {bt.get('sharpe_ratio', 0):.3f}\n")
       752 +                      f.write(f"- Max Drawdown: 
           + {bt.get('max_drawdown', 0):.2%}\n\n")
       753 +              
       754 +              print(f"📄 Analysis report generated: 
           + {report_file}")
       755 +              
       756            except Exception as e:
       757 -              logger.warning(f"保存结果时出错: {e}")
       757 +              logger.warning(f"
           +  Error generating report: {e}")
       758    
       759    
       760    def parse_arguments():
       761        """解析命令行参数"""
       762 -      parser = argparse.ArgumentParser(
           -  description='Portfolio Optimization System')
       762 +      parser = argparse.ArgumentParser(
       763 +          description='Advanced Portfolio 
           + Optimization System',
       764 +          formatter_class=argparse.RawDescriptionHelp
           + Formatter,
       765 +          epilog="""
       766 +  Examples:
       767 +    python start_here.py --mode quick --tickers 
           + AAPL,MSFT,GOOGL
       768 +    python start_here.py --mode full --tickers 
           + AAPL,MSFT,GOOGL,AMZN --save
       769 +    python start_here.py --mode backtest --tickers 
           + QQQ,SPY --start-date 2023-01-01
       770 +    python start_here.py --mode live --tickers 
           + AAPL,MSFT --capital 100000
       771 +          """
       772 +      )
       773        
       774 -      parser.add_argument('--mode', type=str, 
           -  default='demo', 
       775 -                         choices=['demo', 'full', '
           -  custom'],
       776 -                         help='运行模式: demo(演示), 
     full(
           -  完整分析), custom(自定义)')
       774 +      parser.add_argument('--mode', type=str, 
           +  default='quick',
       775 +                         choices=['quick', 'full', '
           +  backtest', 'live', 'health'],
       776 +                         help='运行模式: quick(快速), 
     full(
           +  全面), backtest(回测), live(实盘), health(健康检查)')
       777        
       778 -      parser.add_argument('--tickers', type=str, 
           -  default='AAPL,MSFT,GOOGL,AMZN,TSLA',
       779 -                         help='股票代码，用逗号分隔 
     (例:
           -  AAPL,MSFT,GOOGL)')
       778 +      parser.add_argument('--tickers', type=str, 
       779 +                         default='AAPL,MSFT,GOOGL
           +  ,AMZN,TSLA,NVDA',
       780 +                         help='股票代码，逗号分隔 
           + (例: AAPL,MSFT,GOOGL)')
       781        
       782        parser.add_argument('--start-date', type=str, 
             default=None,
       783 -                         help='开始日期 
     (YYYY-MM-DD格式)')
       783 +                         help='开始日期 (YYYY-MM-DD)')
       784        
       785        parser.add_argument('--end-date', type=str, 
             default=None,
       786 -                         help='结束日期 
     (YYYY-MM-DD格式)')
       786 +                         help='结束日期 (YYYY-MM-DD)')
       787        
       788        parser.add_argument('--market-index', type=str,
              default='SPY',
       789 -                         help='市场基准指数 (默认: 
     SPY)')
       789 +                         help='市场基准指数')
       790        
       791 +      parser.add_argument('--capital', type=float, 
           + default=1000000.0,
       792 +                         help='初始资金 (默认: 
           + 1,000,000)')
       793 +      
       794 +      parser.add_argument('--risk-budget', 
           + type=float, default=0.15,
       795 +                         help='风险预算 (默认: 
           + 0.15)')
       796 +      
       797        parser.add_argument('--save', 
             action='store_true',
       798 -                         help='保存分析结果到文件')
       798 +                         help='保存分析结果')
       799        
       800 -      parser.add_argument('--methods', type=str, 
           - default='max_sharpe,min_variance,risk_parity',
       801 -                         help='优化方法，用逗号分隔')
       800 +      parser.add_argument('--no-backtest', 
           + action='store_true',
       801 +                         help='跳过回测 
     (仅用于full模式)')
       802        
       803 +      parser.add_argument('--config', type=str, 
           + default=None,
       804 +                         help='配置文件路径 
           + (JSON格式)')
       805 +      
       806        return parser.parse_args()
       807    
       808    
       809 +  def load_config(config_path: str) -> Dict[str, 
           + Any]:
       810 +      """加载配置文件"""
       811 +      try:
       812 +          with open(config_path, 'r') as f:
       813 +              return json.load(f)
       814 +      except Exception as e:
       815 +          logger.warning(f"Failed to load config from
           +  {config_path}: {e}")
       816 +          return {}
       817 +  
       818 +  
       819    def main():
       820        """主函数"""
       821 -      print("="*80)
       822 -      print("🚀 Portfolio Optimization System")  
       823 -      print("   Alpha Factor Mining + Beta 
           -  Risk Modeling + Portfolio Optimization")
       824 -      print("="*80)
       821 +      print("=" * 80)
       822 +      print("🚀 ADVANCED PORTFOLIO OPTIMIZATION 
           + SYSTEM")
       823 +      print("   Multi-Factor Alpha Mining + 
           +  Risk Modeling + Portfolio Optimization")
       824 +      print("=" * 80)
       825        
       826 -      # 检查模块可用性
       827 -      if not DATA_MODULES_AVAILABLE:
       828 -          print("❌ 错误: 
           - 缺少必要的模块，请检查安装")
       829 -          print("   请确保以下模块可用:")
       830 -          print("   - data.enhanced_data_fetcher")
       831 -          print("   - 
           - strategy.factor.alpha.real_alpha_miner")  
       832 -          print("   - 
           - strategy.factor.beta.real_beta_estimator")
       833 -          return
       834 -          
       826        # 解析参数
       827        args = parse_arguments()
       828        
       829 -      # 处理股票列表
       829 +      # 加载配置
       830 +      config = {}
       831 +      if args.config:
       832 +          config = load_config(args.config)
       833 +      
       834 +      # 处理参数
       835        tickers = [ticker.strip().upper() for ticker in
              args.tickers.split(',')]
       836 -      methods = [method.strip() for method in 
           - args.methods.split(',')]
       836        
       837 -      print(f"📊 分析股票: {', '.join(tickers)}")
       838 -      print(f"🎯 优化方法: {', '.join(methods)}")
       839 -      print(f"📅 分析时间: {args.start_date or '自动'} 
     到
           -   {args.end_date or '当前'}")
       840 -      print(f"📈 市场基准: {args.market_index}")
       837 +      print(f"🎯 Mode: {args.mode.upper()}")
       838 +      print(f"📊 Assets: {', '.join(tickers)}")
       839 +      print(f"📅 Period: {args.start_date or 'Auto'} 
           +  to {args.end_date or 'Current'}")
       840 +      print(f"💰 Capital: ${args.capital:,.0f}")
       841 +      print(f"⚠️ Risk Budget: 
           + {args.risk_budget:.1%}")
       842        
       843 -      # 创建引擎
       844 -      engine = PortfolioOptimizationEngine(
       843 +      # 创建系统实例
       844 +      system = AdvancedPortfolioSystem(
       845            start_date=args.start_date,
       846 -          end_date=args.end_date, 
       847 -          market_index=args.market_index
       846 +          end_date=args.end_date,
       847 +          market_index=args.market_index,
       848 +          initial_capital=args.capital,
       849 +          risk_budget=args.risk_budget
       850        )
       851        
       852        try:
       853            # 根据模式运行分析
       854 -          if args.mode == 'demo':
       855 -              results = 
           - engine.run_demo_analysis(tickers)
       854 +          if args.mode == 'health':
       855 +              results = system.check_system_health()
       856                
       857 +          elif args.mode == 'quick':
       858 +              results = 
           + system.run_quick_analysis(tickers)
       859 +              
       860            elif args.mode == 'full':
       861 -              results = engine.run_full_analysis(
       861 +              results = 
           + system.run_comprehensive_analysis(
       862                    tickers=tickers,
       863 -                  optimization_methods=methods,
       864 -                  save_results=args.save
       863 +                  save_results=args.save,
       864 +                  run_backtest=not args.no_backtest
       865                )
       866                
       867 -          elif args.mode == 'custom':
       868 -              config = {
       869 -                  'tickers': tickers,
       870 -                  'optimization_methods': methods,
       871 -                  'save_results': args.save
       872 -              }
       873 -              results = 
           - engine.run_custom_analysis(config)
       867 +          elif args.mode == 'backtest':
       868 +              # 专门的回测模式
       869 +              results = 
           + system.run_comprehensive_analysis(
       870 +                  tickers=tickers,
       871 +                  save_results=args.save,
       872 +                  run_backtest=True
       873 +              )
       874 +              
       875 +          elif args.mode == 'live':
       876 +              # 实盘交易设置
       877 +              trading_config = config.get('trading', 
           + {
       878 +                  'rebalance_frequency': 'daily',
       879 +                  'max_position_size': 0.15,
       880 +                  'stop_loss': 0.05,
       881 +                  'take_profit': 0.20
       882 +              })
       883 +              
       884 +              results = 
           + system.run_live_trading_setup(tickers, 
           + trading_config)
       885            
       886 -          # 最终总结
       886 +          else:
       887 +              raise ValueError(f"Unknown mode: 
           + {args.mode}")
       888 +          
       889 +          # 显示最终结果
       890            print(f"\n{'='*80}")
       891 -          print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
       892 -          print(f"{'='*80}")
       891 +          if results.get('error'):
       892 +              print("❌ ANALYSIS FAILED")
       893 +              print(f"Error: {results['error']}")
       894 +              return 1
       895 +          else:
       896 +              print("✅ ANALYSIS COMPLETED 
           + SUCCESSFULLY")
       897 +          print("="*80)
       898            
       899 -          if args.save:
       900 -              print("💾 Results 
           -  saved to 'results/' directory")
       901 -              
       899 +          if args.save or args.mode in ['full', 
           + 'backtest']:
       900 +              print("💾 Results have been 
           +  saved to 'results/' directory")
       901 +              print("📄 Check the generated report 
           + for detailed analysis")
       902 +          
       903 +          print(f"🕐 Finished at: 
           + {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
       904 +          
       905        except KeyboardInterrupt:
       906 -          print(f"\n⚠️ 用户中断分析")
       906 +          print(f"\n⚠️ Analysis interrupted by user")
       907 +          return 1
       908            
       909        except Exception as e:
       910 -          logger.error(f"分析过程出错: {e}")
       911 -          print(f"❌ 分析失败: {e}")
       910 +          logger.error(f"System error: {e}")
       911 +          logger.error(traceback.format_exc())
       912 +          print(f"❌ System error: {e}")
       913            return 1
       914 -          
       914 +      
       915        return 0
       916    
       917    


