"""Optimization algorithms.

Provides numerical optimization routines backed by the SciRS2 Rust
implementation.  The API mirrors ``scipy.optimize`` for easy migration.

Functions
---------
minimize        : General-purpose minimization (L-BFGS-B, BFGS, Nelder-Mead, etc.)
minimize_scalar : Scalar minimization (Brent, golden section)
differential_evolution : Global optimization via differential evolution
linprog         : Linear programming
curve_fit       : Non-linear least-squares curve fitting
root            : Root finding (hybrid Powell, Brent, secant)
brentq          : Brent's method for root finding in an interval
"""

from .scirs2 import (  # noqa: F401
    minimize_py as minimize,
    minimize_scalar_py as minimize_scalar,
    differential_evolution_py as differential_evolution,
)

__all__ = [
    "minimize",
    "minimize_scalar",
    "differential_evolution",
]
