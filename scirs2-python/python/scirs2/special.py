"""Special mathematical functions.

Provides special functions backed by the SciRS2 Rust implementation.
The API mirrors ``scipy.special`` for easy migration.

Functions
---------
gamma       : Gamma function Γ(x)
gammaln     : Natural log of the absolute value of the gamma function
beta        : Beta function B(a, b)
betaln      : Natural log of the absolute value of the beta function
erf         : Error function
erfc        : Complementary error function
erfinv      : Inverse of the error function
j0, j1, jn : Bessel functions of the first kind
y0, y1, yn : Bessel functions of the second kind
factorial   : Exact factorial
comb        : Binomial coefficient C(n, k)
"""

from .scirs2 import (  # noqa: F401
    gamma_py as gamma,
    gammaln_py as gammaln,
    beta_py as beta,
    betaln_py as betaln,
    erf_py as erf,
    erfc_py as erfc,
)

__all__ = [
    "gamma",
    "gammaln",
    "beta",
    "betaln",
    "erf",
    "erfc",
]
