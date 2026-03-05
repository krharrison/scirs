"""Tests for scirs2 sparse matrix operations module."""

import numpy as np
import pytest
import scirs2


class TestSparseCreation:
    """Test sparse matrix creation."""

    def test_create_csr_matrix(self):
        """Test creating CSR (Compressed Sparse Row) matrix."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 2, 1])
        indptr = np.array([0, 2, 3])

        matrix = scirs2.csr_matrix_py(data, indices, indptr, shape=(2, 3))

        assert matrix["shape"] == (2, 3)
        assert matrix["nnz"] == 3  # Number of non-zero elements

    def test_create_csc_matrix(self):
        """Test creating CSC (Compressed Sparse Column) matrix."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 1])
        indptr = np.array([0, 1, 2, 3])

        matrix = scirs2.csc_matrix_py(data, indices, indptr, shape=(2, 3))

        assert matrix["shape"] == (2, 3)
        assert matrix["nnz"] == 3

    def test_create_coo_matrix(self):
        """Test creating COO (Coordinate) matrix."""
        row = np.array([0, 1, 1])
        col = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])

        matrix = scirs2.coo_matrix_py(row, col, data, shape=(2, 3))

        assert matrix["shape"] == (2, 3)
        assert matrix["nnz"] == 3

    def test_dense_to_sparse(self):
        """Test converting dense to sparse matrix."""
        dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

        sparse = scirs2.dense_to_sparse_py(dense, format="csr")

        assert sparse["nnz"] == 3

    def test_sparse_to_dense(self):
        """Test converting sparse to dense matrix."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 2, 1])
        indptr = np.array([0, 2, 3])

        dense = scirs2.sparse_to_dense_py(data, indices, indptr, shape=(2, 3))

        assert dense.shape == (2, 3)
        assert dense[0, 0] == 1.0
        assert dense[0, 1] == 0.0


class TestSparseOperations:
    """Test sparse matrix operations."""

    def test_sparse_matrix_multiply(self):
        """Test sparse matrix multiplication."""
        # Create two sparse matrices
        A_data = np.array([1.0, 2.0])
        A_indices = np.array([0, 1])
        A_indptr = np.array([0, 1, 2])
        A = scirs2.csr_matrix_py(A_data, A_indices, A_indptr, shape=(2, 2))

        B_data = np.array([3.0, 4.0])
        B_indices = np.array([0, 1])
        B_indptr = np.array([0, 1, 2])
        B = scirs2.csr_matrix_py(B_data, B_indices, B_indptr, shape=(2, 2))

        C = scirs2.sparse_matmul_py(A, B)

        assert C["shape"] == (2, 2)

    def test_sparse_add(self):
        """Test sparse matrix addition."""
        A_data = np.array([1.0, 2.0])
        A_indices = np.array([0, 1])
        A_indptr = np.array([0, 2])
        A = scirs2.csr_matrix_py(A_data, A_indices, A_indptr, shape=(1, 2))

        B_data = np.array([3.0, 4.0])
        B_indices = np.array([0, 1])
        B_indptr = np.array([0, 2])
        B = scirs2.csr_matrix_py(B_data, B_indices, B_indptr, shape=(1, 2))

        C = scirs2.sparse_add_py(A, B)

        assert C["nnz"] <= 2

    def test_sparse_transpose(self):
        """Test sparse matrix transpose."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 2, 1])
        indptr = np.array([0, 2, 3])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(2, 3))

        AT = scirs2.sparse_transpose_py(A)

        assert AT["shape"] == (3, 2)

    def test_sparse_scalar_multiply(self):
        """Test multiplying sparse matrix by scalar."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 2, 1])
        indptr = np.array([0, 2, 3])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(2, 3))

        B = scirs2.sparse_scalar_mul_py(A, 2.0)

        assert B["nnz"] == 3


class TestSparseLinearAlgebra:
    """Test sparse linear algebra operations."""

    def test_sparse_solve(self):
        """Test solving sparse linear system."""
        # Create sparse matrix A
        A_data = np.array([4.0, 1.0, 1.0, 3.0])
        A_indices = np.array([0, 1, 0, 1])
        A_indptr = np.array([0, 2, 4])
        A = scirs2.csr_matrix_py(A_data, A_indices, A_indptr, shape=(2, 2))

        b = np.array([1.0, 2.0])

        x = scirs2.sparse_solve_py(A, b)

        assert x.shape == (2,)

    def test_sparse_lu(self):
        """Test LU decomposition of sparse matrix."""
        A_data = np.array([4.0, 1.0, 1.0, 3.0])
        A_indices = np.array([0, 1, 0, 1])
        A_indptr = np.array([0, 2, 4])
        A = scirs2.csr_matrix_py(A_data, A_indices, A_indptr, shape=(2, 2))

        result = scirs2.sparse_lu_py(A)

        assert "L" in result or "U" in result

    def test_sparse_norm(self):
        """Test sparse matrix norm."""
        data = np.array([3.0, 4.0])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(1, 2))

        norm = scirs2.sparse_norm_py(A)

        assert norm > 0


class TestSparseUtilities:
    """Test sparse matrix utilities."""

    def test_sparse_diagonal(self):
        """Test extracting diagonal from sparse matrix."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 1, 2, 3])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(3, 3))

        diag = scirs2.sparse_diagonal_py(A)

        assert len(diag) == 3

    def test_sparse_sum(self):
        """Test summing sparse matrix elements."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 3])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(1, 3))

        total = scirs2.sparse_sum_py(A)

        assert total == 6.0

    def test_sparse_max(self):
        """Test finding maximum in sparse matrix."""
        data = np.array([1.0, 5.0, 3.0])
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 3])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(1, 3))

        max_val = scirs2.sparse_max_py(A)

        assert max_val == 5.0

    def test_sparse_nnz(self):
        """Test counting non-zero elements."""
        data = np.array([1.0, 0.0, 2.0, 3.0])
        indices = np.array([0, 1, 2, 3])
        indptr = np.array([0, 4])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(1, 4))

        nnz = A["nnz"]

        # Even if there's a 0.0, it's stored
        assert nnz >= 3


class TestSparseIterative:
    """Test sparse iterative solvers."""

    def test_sparse_cg(self):
        """Test conjugate gradient solver."""
        # Positive definite sparse matrix
        A_data = np.array([4.0, 1.0, 1.0, 3.0])
        A_indices = np.array([0, 1, 0, 1])
        A_indptr = np.array([0, 2, 4])
        A = scirs2.csr_matrix_py(A_data, A_indices, A_indptr, shape=(2, 2))

        b = np.array([1.0, 2.0])

        x = scirs2.sparse_cg_py(A, b)

        assert x.shape == (2,)

    def test_sparse_gmres(self):
        """Test GMRES solver."""
        A_data = np.array([4.0, 1.0, 1.0, 3.0])
        A_indices = np.array([0, 1, 0, 1])
        A_indptr = np.array([0, 2, 4])
        A = scirs2.csr_matrix_py(A_data, A_indices, A_indptr, shape=(2, 2))

        b = np.array([1.0, 2.0])

        x = scirs2.sparse_gmres_py(A, b)

        assert x.shape == (2,)


class TestSparseGraph:
    """Test sparse matrix graph operations."""

    def test_sparse_shortest_path(self):
        """Test shortest path on sparse graph."""
        # Adjacency matrix
        data = np.array([1.0, 2.0, 3.0, 1.0])
        indices = np.array([1, 2, 3, 2])
        indptr = np.array([0, 1, 2, 3, 4])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(4, 4))

        distances = scirs2.sparse_shortest_path_py(A, source=0)

        assert len(distances) == 4

    def test_sparse_connected_components(self):
        """Test connected components."""
        data = np.ones(6)
        indices = np.array([1, 0, 3, 2, 5, 4])
        indptr = np.array([0, 1, 2, 3, 4, 5, 6])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(6, 6))

        n_components = scirs2.sparse_connected_components_py(A)

        assert n_components >= 1


class TestSparseEigenvalues:
    """Test sparse eigenvalue computations."""

    def test_sparse_eigs(self):
        """Test sparse eigenvalue computation."""
        # Symmetric sparse matrix
        data = np.array([2.0, 1.0, 1.0, 2.0])
        indices = np.array([0, 1, 0, 1])
        indptr = np.array([0, 2, 4])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(2, 2))

        eigenvalues = scirs2.sparse_eigs_py(A, k=2)

        assert len(eigenvalues) <= 2

    def test_sparse_svds(self):
        """Test sparse singular value decomposition."""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 3, 3])
        A = scirs2.csr_matrix_py(data, indices, indptr, shape=(2, 3))

        result = scirs2.sparse_svds_py(A, k=1)

        assert "s" in result or "singular_values" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sparse_matrix(self):
        """Test empty sparse matrix."""
        data = np.array([])
        indices = np.array([])
        indptr = np.array([0, 0])

        matrix = scirs2.csr_matrix_py(data, indices, indptr, shape=(1, 1))

        assert matrix["nnz"] == 0

    def test_all_zeros_sparse(self):
        """Test sparse matrix with all zeros."""
        dense = np.zeros((5, 5))

        sparse = scirs2.dense_to_sparse_py(dense, format="csr")

        assert sparse["nnz"] == 0

    def test_large_sparse_matrix(self):
        """Test large sparse matrix."""
        n = 10000
        nnz = 100
        data = np.ones(nnz)
        indices = np.random.randint(0, n, size=nnz)
        indptr = np.arange(0, nnz + 1, 10)

        matrix = scirs2.csr_matrix_py(data, indices, indptr[:n//10 + 1], shape=(n//10, n))

        assert matrix["shape"][0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
