"""SciRS2 — Scientific computing in Rust with Python bindings.

This package provides Python bindings for the SciRS2 scientific computing
library, a high-performance SciPy alternative implemented in pure Rust.

Modules
-------
linalg : Linear algebra (matmul, det, inv, eig, svd, solve, norm, qr, lu, cholesky)
stats  : Statistical distributions and hypothesis tests
fft    : Fast Fourier Transforms (DFT, DCT, DST, STFT)
optimize : Optimization algorithms (BFGS, L-BFGS-B, differential evolution)
special  : Special mathematical functions (gamma, beta, bessel, etc.)
integrate : Numerical integration (quad, dblquad, odeint)
interpolate : Interpolation (1D, 2D, RBF, splines)
signal  : Signal processing (filters, wavelets, spectral analysis)
spatial : Spatial algorithms (KD-tree, Delaunay, ConvexHull)
sparse  : Sparse matrix operations
ndimage : N-dimensional image processing
cluster : Clustering algorithms (K-Means, DBSCAN, hierarchical)
graph   : Graph algorithms
metrics : ML evaluation metrics
io      : File I/O (HDF5, Parquet, Zarr, NetCDF)
datasets : Dataset loading and synthetic generators
transform : Data transformations and preprocessing
text    : Text processing and NLP
vision  : Computer vision algorithms
series  : Time series analysis (ARIMA, GARCH, state-space)
neural  : Neural network layers
autograd : Automatic differentiation
"""

from .scirs2 import *  # noqa: F401,F403 — re-export all C-extension symbols
from .scirs2 import __version__, __author__  # noqa: F401

# Convenience sub-namespace imports — mirrors the flat API exposed by the
# compiled extension, grouped for discoverability.
from . import linalg  # noqa: F401
from . import stats  # noqa: F401
from . import fft  # noqa: F401
from . import optimize  # noqa: F401
from . import special  # noqa: F401
from . import integrate  # noqa: F401
from . import signal  # noqa: F401
from . import sparse  # noqa: F401
from . import ndimage  # noqa: F401
from . import cluster  # noqa: F401

__all__ = [
    "__version__",
    "__author__",
    "linalg",
    "stats",
    "fft",
    "optimize",
    "special",
    "integrate",
    "signal",
    "sparse",
    "ndimage",
    "cluster",
]
