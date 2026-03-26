"""Sparse matrix operations.

Provides sparse matrix construction and arithmetic backed by the SciRS2 Rust
implementation.  The API mirrors ``scipy.sparse`` for easy migration.

Classes
-------
csr_matrix  : Compressed Sparse Row matrix
csc_matrix  : Compressed Sparse Column matrix
coo_matrix  : COOrdinate format sparse matrix

Functions
---------
eye         : Sparse identity matrix
diags       : Sparse matrix from diagonals
spsolve     : Solve a sparse linear system
splu        : Sparse LU factorisation
linalg      : Sparse linear algebra utilities (eigsh, svds)
"""

from .scirs2 import (  # noqa: F401
    PyCsrMatrix as csr_matrix,
    PyCscMatrix as csc_matrix,
)

__all__ = [
    "csr_matrix",
    "csc_matrix",
]
