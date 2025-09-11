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
    if name == "beta_statistics":
        from . import beta_statistics
        return beta_statistics
    raise AttributeError(f"module {__name__} has no attribute {name}")
