"""
Strategy package init
Keep it lightweight to avoid circular imports.
"""

def __getattr__(name):
    if name == "AlphaBetaOptimizer":
        from .alpha_beta_optimizer import AlphaBetaOptimizer
        return AlphaBetaOptimizer
    if name == "alpha_ml":
        from . import alpha_ml
        return alpha_ml
    raise AttributeError(name)

