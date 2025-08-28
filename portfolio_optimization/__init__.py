"""
Portfolio Optimization Module

Advanced portfolio optimization system combining alpha factor analysis
and comprehensive beta risk modeling for optimal portfolio construction.

Components:
- Alpha ML: Advanced alpha factor mining and evaluation
- Beta Statistics: Comprehensive systematic risk estimation  
- Alpha-Beta Optimizer: Integrated optimization framework
"""

from .alpha_beta_optimizer import AlphaBetaOptimizer

# Import alpha and beta submodules
from . import alpha_ml
from . import beta_statistics

__all__ = [
    'AlphaBetaOptimizer',
    'alpha_ml',
    'beta_statistics'
]

__version__ = '1.0.0'