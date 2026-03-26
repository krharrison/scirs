"""Numerical integration.

Provides numerical integration routines backed by the SciRS2 Rust
implementation.  The API mirrors ``scipy.integrate`` for easy migration.

Functions
---------
quad        : General-purpose numerical integration of a Python callable
dblquad     : Double integral over a rectangular region
tplquad     : Triple integral
odeint      : Integrate a system of ODEs (LSODA-style)
solve_ivp   : Solve an initial value problem for a system of ODEs
trapz       : Integrate using the composite trapezoidal rule
simps       : Integrate using Simpson's rule
"""

from .scirs2 import (  # noqa: F401
    quad_py as quad,
    trapz_py as trapz,
    simps_py as simps,
)

__all__ = [
    "quad",
    "trapz",
    "simps",
]
