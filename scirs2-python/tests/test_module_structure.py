"""Tests for scirs2 module structure and availability.

These tests verify the module structure, what's available, and provide
baseline mathematical helper tests that run independently of the extension.
These tests ALWAYS run regardless of whether the Rust extension is built.
"""

import numpy as np
import pytest
import sys


class TestModuleImport:
    """Test module importability."""

    def test_scirs2_importable(self):
        """scirs2 should be importable without errors."""
        import scirs2
        assert scirs2 is not None

    def test_numpy_available(self):
        """numpy should be available."""
        import numpy as np
        assert np.__version__ is not None


class TestMathematicalHelpers:
    """Test mathematical helper functions using pure numpy.

    These verify the mathematical formulas used in the test suite
    without requiring the Rust extension.
    """

    def test_determinant_formula_2x2(self):
        """Verify 2x2 determinant formula: ad - bc."""
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        assert abs(det - 8.0) < 1e-10

    def test_matrix_inverse_formula(self):
        """Verify 2x2 inverse formula."""
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        det = np.linalg.det(a)
        a_inv = np.linalg.inv(a)
        # Check A @ A^-1 = I
        assert np.allclose(a @ a_inv, np.eye(2), atol=1e-10)

    def test_eigenvalue_formula_symmetric(self):
        """Verify eigenvalues of [[2,1],[1,2]] are 1 and 3."""
        a = np.array([[2.0, 1.0], [1.0, 2.0]])
        eigenvalues = np.linalg.eigvalsh(a)
        assert abs(eigenvalues[0] - 1.0) < 1e-10
        assert abs(eigenvalues[1] - 3.0) < 1e-10

    def test_cholesky_formula(self):
        """Verify Cholesky decomposition A = L @ L.T."""
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        l = np.linalg.cholesky(a)
        assert np.allclose(l @ l.T, a, atol=1e-10)

    def test_svd_reconstruction(self):
        """Verify SVD: A = U @ diag(s) @ Vt."""
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        reconstructed = u @ np.diag(s) @ vt
        assert np.allclose(reconstructed, a, atol=1e-10)

    def test_fft_dc_component(self):
        """FFT DC component equals sum of signal."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        fft_result = np.fft.fft(data)
        assert abs(fft_result[0].real - 10.0) < 1e-10
        assert abs(fft_result[0].imag) < 1e-10

    def test_fft_parseval(self):
        """Verify Parseval's theorem."""
        data = np.random.default_rng(42).standard_normal(32)
        fft_result = np.fft.fft(data)
        time_energy = np.sum(data ** 2)
        freq_energy = np.sum(np.abs(fft_result) ** 2) / len(data)
        assert abs(time_energy - freq_energy) < 1e-8

    def test_fft_roundtrip(self):
        """FFT followed by IFFT recovers original."""
        data = np.random.default_rng(42).standard_normal(16)
        reconstructed = np.fft.ifft(np.fft.fft(data)).real
        assert np.allclose(data, reconstructed, atol=1e-10)

    def test_mean_formula(self):
        """Mean = sum / count."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(np.mean(data) - 3.0) < 1e-10

    def test_std_formula(self):
        """Standard deviation formula."""
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        std = np.std(data, ddof=0)
        assert abs(std - 2.0) < 1e-10

    def test_correlation_formula(self):
        """Pearson correlation formula."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x  # Perfect correlation
        corr = np.corrcoef(x, y)[0, 1]
        assert abs(corr - 1.0) < 1e-10

    def test_geometric_mean_formula(self):
        """Geometric mean = (product)^(1/n)."""
        data = np.array([1.0, 2.0, 4.0, 8.0])
        gmean = (1.0 * 2.0 * 4.0 * 8.0) ** 0.25
        assert abs(gmean - 2.0 ** 1.5) < 1e-10

    def test_harmonic_mean_formula(self):
        """Harmonic mean = n / sum(1/x_i)."""
        data = np.array([1.0, 2.0, 4.0])
        hmean = 3.0 / (1.0 / 1.0 + 1.0 / 2.0 + 1.0 / 4.0)
        assert abs(hmean - 12.0 / 7.0) < 1e-10

    def test_zscore_formula(self):
        """Z-score normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        zscores = (data - np.mean(data)) / np.std(data, ddof=0)
        assert abs(np.mean(zscores)) < 1e-10
        assert abs(np.std(zscores) - 1.0) < 1e-10


class TestExtensionAvailability:
    """Check what scirs2 functions are available."""

    def test_module_exists(self):
        """scirs2 module should exist."""
        import scirs2
        assert scirs2 is not None

    def test_report_available_functions(self):
        """Report which functions are available (informational)."""
        import scirs2
        available = [attr for attr in dir(scirs2) if attr.endswith('_py')]
        # This test always passes - it just reports what's available
        assert isinstance(available, list)

    def test_linalg_functions_available(self):
        """Check if linalg functions are available."""
        import scirs2
        linalg_funcs = ['det_py', 'inv_py', 'trace_py', 'lu_py', 'qr_py', 'svd_py']
        available = [f for f in linalg_funcs if hasattr(scirs2, f)]
        # Just report - don't fail if not available
        if not available:
            pytest.skip("No linalg functions available - extension not built")
        assert len(available) > 0

    def test_stats_functions_available(self):
        """Check if stats functions are available."""
        import scirs2
        stats_funcs = ['mean_py', 'std_py', 'var_py', 'median_py', 'describe_py']
        available = [f for f in stats_funcs if hasattr(scirs2, f)]
        if not available:
            pytest.skip("No stats functions available - extension not built")
        assert len(available) > 0

    def test_fft_functions_available(self):
        """Check if FFT functions are available."""
        import scirs2
        fft_funcs = ['fft_py', 'ifft_py', 'rfft_py', 'dct_py']
        available = [f for f in fft_funcs if hasattr(scirs2, f)]
        if not available:
            pytest.skip("No FFT functions available - extension not built")
        assert len(available) > 0


class TestNumericalPrecision:
    """Test numerical precision requirements from test suite."""

    def test_frobenius_norm(self):
        """Frobenius norm of [[3,4],[0,0]] = 5."""
        a = np.array([[3.0, 4.0], [0.0, 0.0]])
        norm = np.linalg.norm(a, 'fro')
        assert abs(norm - 5.0) < 1e-10

    def test_l2_norm_345(self):
        """L2 norm of [3, 4] = 5 (Pythagorean triple)."""
        x = np.array([3.0, 4.0])
        norm = np.linalg.norm(x, 2)
        assert abs(norm - 5.0) < 1e-10

    def test_l1_norm(self):
        """L1 norm of [3, -4] = 7."""
        x = np.array([3.0, -4.0])
        norm = np.linalg.norm(x, 1)
        assert abs(norm - 7.0) < 1e-10

    def test_identity_cond(self):
        """Condition number of identity = 1."""
        cond = np.linalg.cond(np.eye(3))
        assert abs(cond - 1.0) < 1e-10

    def test_pseudoinverse_property(self):
        """Verify A @ pinv(A) @ A = A using numpy."""
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        a_pinv = np.linalg.pinv(a)
        assert np.allclose(a @ a_pinv @ a, a, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
