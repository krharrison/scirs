"""Linear algebra functions.

Provides high-performance linear algebra routines backed by the SciRS2 Rust
implementation.  The API mirrors ``scipy.linalg`` and ``numpy.linalg`` for
easy migration.

Functions
---------
matmul      : Matrix multiplication
det         : Matrix determinant
inv         : Matrix inverse
eig         : Eigenvalues and eigenvectors
svd         : Singular value decomposition
solve       : Solve a linear system Ax = b
norm        : Vector or matrix norm
qr          : QR decomposition
lu          : LU decomposition
cholesky    : Cholesky decomposition
trace       : Matrix trace
pinv        : Moore-Penrose pseudo-inverse
lstsq       : Least-squares solution
matrix_rank : Matrix rank
cond        : Condition number
"""

from .scirs2 import (  # noqa: F401
    matmul_py as matmul,
    det_py as det,
    inv_py as inv,
    eig_py as eig,
    svd_py as svd,
    solve_py as solve,
    norm_py as norm,
    qr_py as qr,
    lu_py as lu,
    cholesky_py as cholesky,
    trace_py as trace,
    pinv_py as pinv,
    lstsq_py as lstsq,
    matrix_rank_py as matrix_rank,
    cond_py as cond,
)

__all__ = [
    "matmul",
    "det",
    "inv",
    "eig",
    "svd",
    "solve",
    "norm",
    "qr",
    "lu",
    "cholesky",
    "trace",
    "pinv",
    "lstsq",
    "matrix_rank",
    "cond",
]
