# 智能投资组合优化器 (Smart Portfolio Optimizer)

## 概述

这是一个先进的投资组合优化系统，能够自动发现最强的alpha因子和beta风险模型，然后优化资产配置以获得最大的风险调整收益。

## 核心功能

### 1. 自动因子挖掘
- **技术因子**: 动量、均值回归、技术指标等
- **基本面因子**: PE比率、营收增长、分红收益率等  
- **微观结构因子**: 流动性、价格影响、波动率聚集等
- **横截面因子**: 相对强弱、行业轮动等
- **机器学习因子**: 使用RF、XGBoost等算法预测收益

### 2. 多种Beta估计方法
- **CAPM Beta**: 传统市场beta
- **多因子Beta**: Fama-French等多因子模型
- **Copula Beta**: 基于copula的依赖结构建模
- **CVaR Beta**: 条件风险价值模型

### 3. 优化目标选择
- **最大夏普比率**: 风险调整收益最大化
- **最小方差**: 风险最小化
- **最大效用**: 基于风险厌恶的效用最大化
- **风险平价**: 等风险贡献配置

### 4. 全面回测和报告
- 动态再平衡回测
- 风险分解分析
- 可视化报告生成

## 快速开始

### 运行优化器

```bash
cd scripts
python smart_portfolio_optimizer.py
```

### 自定义股票池

编辑 `smart_portfolio_optimizer.py` 中的 `tickers` 列表：

```python
tickers = [
    # 大盘科技股
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    # 金融股
    'JPM', 'BAC', 'WFC',
    # 消费股
    'PG', 'KO', 'WMT',
    # 你的其他股票...
]
```

### 调整优化参数

```python
optimizer = SmartPortfolioOptimizer(
    start_date="2020-01-01",     # 数据开始日期
    end_date="2024-01-01",       # 数据结束日期
    risk_free_rate=0.03          # 无风险利率
)

# 因子挖掘参数
alpha_factors = optimizer.mine_alpha_factors(
    min_ic_threshold=0.015,      # 最小信息系数阈值
    top_n_factors=25             # 选择前N个因子
)

# 优化参数
result = optimizer.optimize_portfolio(
    objective='max_sharpe',      # 优化目标
    constraints={
        'max_weight': 0.25,      # 单一资产最大权重
        'min_weight': 0.01       # 单一资产最小权重
    },
    alpha_weight=0.7             # alpha因子权重 (vs beta权重)
)
```

## 输出文件说明

运行后会在 `results/smart_optimizer/` 目录下生成以下文件：

### 1. `optimization_report.txt`
完整的优化报告，包含：
- 数据摘要
- 因子挖掘结果
- 最优权重配置
- 回测业绩指标

### 2. `optimal_weights.csv`
最优投资组合权重，字段包括：
- `ticker`: 股票代码
- `weight`: 投资权重
- `expected_return`: 预期收益率
- `contribution`: 收益贡献度

### 3. `backtest_results.csv`
回测结果明细，字段包括：
- `date`: 日期
- `portfolio_return`: 组合收益率
- `cumulative_return`: 累计收益率
- `rebalance_date`: 再平衡日期

### 4. `portfolio_weights.png`
投资组合权重可视化图表

### 5. `backtest_performance.png`
回测业绩图表，包含：
- 累计收益曲线
- 滚动夏普比率

## 高级用法

### 1. 程序化调用

```python
from smart_portfolio_optimizer import SmartPortfolioOptimizer

# 初始化
optimizer = SmartPortfolioOptimizer()

# 获取数据
optimizer.fetch_market_data(['AAPL', 'MSFT', 'GOOGL'])

# 挖掘因子
optimizer.mine_alpha_factors()

# 估计风险模型
optimizer.estimate_risk_models()

# 优化组合
result = optimizer.optimize_portfolio(objective='max_sharpe')

# 回测
backtest = optimizer.backtest_strategy(rebalance_frequency='monthly')

# 生成报告
report = optimizer.generate_report()
```

### 2. 批量测试不同策略

```python
objectives = ['max_sharpe', 'min_variance', 'max_utility']
results = {}

for obj in objectives:
    result = optimizer.optimize_portfolio(objective=obj)
    results[obj] = result
    print(f"{obj}: Sharpe = {result.get('sharpe_ratio', 0):.3f}")
```

### 3. 自定义约束条件

```python
custom_constraints = {
    'max_weight': 0.20,          # 单一持仓不超过20%
    'min_weight': 0.02,          # 单一持仓不少于2%
    'max_concentration': 0.6,     # 赫芬达尔指数不超过0.6
    'min_diversification': 8      # 至少8只有效股票
}

result = optimizer.optimize_portfolio(constraints=custom_constraints)
```

## 性能指标说明

### 主要指标

- **Total Return**: 总收益率
- **Annual Return**: 年化收益率  
- **Annual Volatility**: 年化波动率
- **Sharpe Ratio**: 夏普比率 = (收益率 - 无风险利率) / 波动率
- **Max Drawdown**: 最大回撤
- **Win Rate**: 胜率（正收益日占比）

### 风险指标

- **Effective Assets**: 有效资产数量 = 1/Σ(weight²)
- **Concentration**: 集中度指数（赫芬达尔指数）
- **Risk Contributions**: 各资产风险贡献度

## 注意事项

1. **数据质量**: 确保网络连接正常以获取实时数据
2. **计算时间**: 大量股票和长时间序列可能需要较长计算时间
3. **风险提示**: 历史业绩不代表未来表现，投资需谨慎

## 故障排除

### 常见问题

1. **模块导入错误**: 确保所有依赖包已安装
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib seaborn yfinance
   ```

2. **数据获取失败**: 检查网络连接和股票代码有效性

3. **优化失败**: 可能是约束条件过于严格，尝试放宽限制

### 日志调试

设置日志级别查看详细信息：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 技术架构

```
SmartPortfolioOptimizer
├── 数据层
│   ├── EnhancedDataFetcher (增强数据获取)
│   └── yfinance fallback (备用数据源)
├── 因子层
│   ├── RealAlphaMiner (Alpha因子挖掘)  
│   ├── RealBetaEstimator (Beta估计)
│   └── FactorValidator (因子验证)
├── 优化层
│   ├── 期望收益计算
│   ├── 协方差矩阵构建
│   └── 约束优化求解
└── 输出层
    ├── 回测引擎
    ├── 报告生成
    └── 可视化图表
```

## 更新日志

### v1.0 (当前版本)
- 初始发布
- 支持多种alpha因子挖掘
- 集成多种beta估计方法
- 完整的回测和报告功能

---

如有问题或建议，请联系开发者或提交Issue。