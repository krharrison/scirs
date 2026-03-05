"""
Pure-algorithm test suite for SciRS2.PureAlgorithms.

Tests all pure-Julia implementations without requiring the SciRS2 shared library.
Each test verifies correctness against known analytical solutions.

To run standalone:
    julia --project=/path/to/julia/SciRS2 test/test_pure_algorithms.jl
"""

using Test
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "..", "src", "PureAlgorithms.jl"))
using .PureAlgorithms

# Helper: approximate equality
approx(a, b; atol=1e-8) = abs(a - b) < atol
approx_vec(a, b; atol=1e-8) = all(abs.(a .- b) .< atol)

@testset "SciRS2 PureAlgorithms" begin

    # =========================================================================
    @testset "Romberg Integration" begin

        @testset "x^2 on [0, 1] = 1/3" begin
            val = romberg_integrate(x -> x^2, 0.0, 1.0)
            @test approx(val, 1.0 / 3.0, atol=1e-10)
        end

        @testset "sin on [0, π] = 2" begin
            val = romberg_integrate(sin, 0.0, π)
            @test approx(val, 2.0, atol=1e-10)
        end

        @testset "exp on [0, 1] = e - 1" begin
            val = romberg_integrate(exp, 0.0, 1.0)
            @test approx(val, ℯ - 1.0, atol=1e-10)
        end

        @testset "constant function = length of interval" begin
            val = romberg_integrate(x -> 3.0, -2.0, 5.0)
            @test approx(val, 21.0, atol=1e-10)
        end

        @testset "cos on [0, π/2] = 1" begin
            val = romberg_integrate(cos, 0.0, π / 2)
            @test approx(val, 1.0, atol=1e-10)
        end

        @testset "1/x on [1, e] = 1" begin
            val = romberg_integrate(x -> 1.0 / x, 1.0, ℯ)
            @test approx(val, 1.0, atol=1e-8)
        end

        @testset "polynomial x^5 - 3x^2 + 1 on [-1, 1]" begin
            # ∫₋₁¹ (x^5 - 3x^2 + 1) dx = 0 - 2 + 2 = 0
            val = romberg_integrate(x -> x^5 - 3x^2 + 1.0, -1.0, 1.0)
            @test approx(val, 0.0, atol=1e-10)
        end

    end  # Romberg

    # =========================================================================
    @testset "Gauss-Legendre Quadrature" begin

        @testset "nodes and weights n=1" begin
            nodes, weights = gauss_legendre_nodes_weights(1)
            @test length(nodes) == 1
            @test approx(nodes[1], 0.0, atol=1e-14)
            @test approx(weights[1], 2.0, atol=1e-14)
        end

        @testset "weights sum to 2 for various n" begin
            for n in [2, 3, 4, 5, 8, 10]
                _, weights = gauss_legendre_nodes_weights(n)
                @test approx(sum(weights), 2.0, atol=1e-12)
            end
        end

        @testset "nodes sorted ascending and in (-1, 1)" begin
            for n in [3, 5, 7]
                nodes, _ = gauss_legendre_nodes_weights(n)
                @test issorted(nodes)
                @test all(-1.0 .< nodes .< 1.0)
            end
        end

        @testset "exact for polynomial of degree 2n-1" begin
            # 5-point GL is exact for degree ≤ 9
            # ∫₋₁¹ x^9 dx = 0 (odd function)
            nodes, weights = gauss_legendre_nodes_weights(5)
            val = sum(weights .* nodes .^ 9)
            @test approx(val, 0.0, atol=1e-12)

            # ∫₋₁¹ x^8 dx = 2/9
            val2 = sum(weights .* nodes .^ 8)
            @test approx(val2, 2.0 / 9.0, atol=1e-12)
        end

        @testset "gauss_legendre_integrate transforms to [a, b]" begin
            # ∫₀² exp(x) dx = e² - 1
            val = gauss_legendre_integrate(exp, 0.0, 2.0, 10)
            @test approx(val, ℯ^2 - 1.0, atol=1e-10)
        end

        @testset "n=2 nodes are ±1/√3" begin
            nodes, weights = gauss_legendre_nodes_weights(2)
            @test approx(abs(nodes[1]), 1.0 / sqrt(3.0), atol=1e-12)
            @test approx(abs(nodes[2]), 1.0 / sqrt(3.0), atol=1e-12)
            @test approx(weights[1], 1.0, atol=1e-12)
            @test approx(weights[2], 1.0, atol=1e-12)
        end

    end  # Gauss-Legendre

    # =========================================================================
    @testset "Adaptive Simpson Integration" begin

        @testset "sqrt on [0, 1] = 2/3" begin
            val = adaptive_simpson(sqrt, 0.0, 1.0)
            @test approx(val, 2.0 / 3.0, atol=1e-8)
        end

        @testset "sin on [0, π] = 2" begin
            val = adaptive_simpson(sin, 0.0, π)
            @test approx(val, 2.0, atol=1e-8)
        end

        @testset "Runge function on [-1, 1]" begin
            # ∫₋₁¹ 1/(1 + 25x²) dx = 2/5 * arctan(5)
            f_runge = x -> 1.0 / (1.0 + 25.0 * x^2)
            expected = 2.0 / 5.0 * atan(5.0)
            val = adaptive_simpson(f_runge, -1.0, 1.0; tol=1e-8)
            @test approx(val, expected, atol=1e-7)
        end

    end  # Adaptive Simpson

    # =========================================================================
    @testset "Moran's I Spatial Autocorrelation" begin

        @testset "positive autocorrelation" begin
            # 4 locations: similar neighbors -> positive I
            W = Float64[0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0]
            x = [1.0, 1.0, 5.0, 5.0]  # clear cluster
            I = moran_i(x, W)
            @test I > 0.0
        end

        @testset "negative autocorrelation (checkerboard)" begin
            W = Float64[0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0]
            # Checkerboard pattern -> dissimilar neighbors -> negative I
            x = [1.0, 5.0, 1.0, 5.0]
            I = moran_i(x, W)
            @test I < 0.0
        end

        @testset "constant x raises error" begin
            W = Float64[0 1; 1 0]
            x = [3.0, 3.0]
            @test_throws ErrorException moran_i(x, W)
        end

        @testset "dimension mismatch raises error" begin
            W = Float64[0 1; 1 0]
            x = [1.0, 2.0, 3.0]
            @test_throws DimensionMismatch moran_i(x, W)
        end

        @testset "1x1 grid: trivial" begin
            W = reshape([0.0], 1, 1)
            @test_throws ErrorException moran_i([1.0], W)  # all-zero weight matrix
        end

    end  # Moran's I

    # =========================================================================
    @testset "Kernel Density Estimation" begin

        @testset "density integrates to ≈ 1" begin
            data = Float64.(1:20)
            bw = silverman_bandwidth(data)
            result = kernel_density(data, bw; n_points=500)
            # Approximate integral via trapezoidal rule
            dx = result.x[2] - result.x[1]
            integral = sum(result.density) * dx
            @test approx(integral, 1.0, atol=0.01)
        end

        @testset "density is non-negative" begin
            data = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0]
            bw = silverman_bandwidth(data)
            result = kernel_density(data, bw; n_points=100)
            @test all(result.density .>= 0.0)
        end

        @testset "KDE peaks near data concentration" begin
            data = [3.0, 3.1, 3.0, 2.9, 3.2, 3.0]  # concentrated at 3
            result = kernel_density(data, 0.1; n_points=200)
            peak_x = result.x[argmax(result.density)]
            @test approx(peak_x, 3.0, atol=0.2)
        end

        @testset "silverman_bandwidth gives positive result" begin
            data = randn(100) .* 2.0 .+ 5.0  # N(5, 4)
            bw = silverman_bandwidth(data)
            @test bw > 0.0
        end

        @testset "kernel_density_evaluate at known point" begin
            # Single Dirac-like sample: density near sample ≈ 1/(n*bw*sqrt(2π))
            data = [0.0]
            bw = 1.0
            density_at_0 = kernel_density_evaluate(data, bw, [0.0])[1]
            expected = 1.0 / (1.0 * bw * sqrt(2π))
            @test approx(density_at_0, expected, atol=1e-10)
        end

        @testset "empty data raises error" begin
            @test_throws ArgumentError kernel_density(Float64[], 1.0)
        end

        @testset "negative bandwidth raises error" begin
            @test_throws ArgumentError kernel_density([1.0, 2.0], -0.1)
        end

    end  # KDE

    # =========================================================================
    @testset "Exponential Smoothing" begin

        @testset "SES: alpha=1 reproduces data with 1-lag shift" begin
            y = [1.0, 3.0, 5.0, 7.0, 9.0]
            res = ets_ses(y; alpha=1.0)
            # With alpha=1, l_t = y_t, so fitted[t] = l_{t-1} = y_{t-1}
            @test approx(res.fitted[2], 1.0, atol=1e-10)
            @test approx(res.fitted[3], 3.0, atol=1e-10)
            @test approx(res.forecast, 9.0, atol=1e-10)
        end

        @testset "SES: alpha=0.5 weighted average" begin
            y = [4.0, 4.0, 4.0, 4.0]   # constant -> forecast = 4
            res = ets_ses(y; alpha=0.5)
            @test approx(res.forecast, 4.0, atol=1e-10)
        end

        @testset "SES: invalid alpha raises error" begin
            @test_throws ArgumentError ets_ses([1.0, 2.0]; alpha=0.0)
            @test_throws ArgumentError ets_ses([1.0, 2.0]; alpha=1.5)
        end

        @testset "Holt: linear trend" begin
            # y = [1, 2, 3, 4, 5]: perfect linear trend
            y = Float64.(1:5)
            res = ets_holt(y; alpha=0.9, beta=0.9, h=1)
            # Should forecast 6.0 (approximately)
            @test res.forecast > 5.0
            @test res.forecast < 8.0
        end

        @testset "Holt: constant series has near-zero trend" begin
            y = ones(Float64, 10) .* 7.0
            res = ets_holt(y; alpha=0.3, beta=0.1)
            @test approx(abs(res.trend[end]), 0.0, atol=1e-6)
            @test approx(res.level[end], 7.0, atol=1e-3)
        end

        @testset "Holt: invalid beta raises error" begin
            @test_throws ArgumentError ets_holt([1.0, 2.0, 3.0]; beta=0.0)
        end

        @testset "Holt-Winters: additive, period=4, constant seasonal" begin
            # Quarterly data with period=4, no trend, constant seasonals [1,-1,1,-1]
            period = 4
            seasonal_pattern = [1.0, -1.0, 1.0, -1.0]
            level_val = 10.0
            y = [level_val + seasonal_pattern[((t - 1) % period) + 1] for t in 1:16]
            # We need at least 2 * period = 8 points
            res = ets_holt_winters(y; alpha=0.3, beta=0.1, gamma=0.3, period=period)
            @test length(res.fitted) == 16
            # Level should stay near 10
            @test approx(res.level[end], 10.0, atol=1.5)
        end

        @testset "Holt-Winters: too short raises error" begin
            @test_throws ArgumentError ets_holt_winters([1.0, 2.0, 3.0]; period=4)
        end

    end  # Exponential Smoothing

    # =========================================================================
    @testset "Levinson-Durbin AR Estimation" begin

        @testset "AR(1) with φ=0.9: recovered coefficient" begin
            # Autocorrelation sequence for AR(1) with φ=0.9: r(k) = φ^k
            phi_true = 0.9
            r = [phi_true^k for k in 0:3]   # r[1]=1, r[2]=φ, r[3]=φ², r[4]=φ³
            phi_est = levinson_durbin(r)
            # First coefficient should match φ_true closely
            @test approx(phi_est[1], phi_true, atol=1e-10)
            # Higher-order coefficients should be near zero
            @test abs(phi_est[2]) < 0.01
            @test abs(phi_est[3]) < 0.01
        end

        @testset "white noise: near-zero AR coefficients" begin
            r = [1.0, 0.0, 0.0, 0.0, 0.0]  # white noise ACF
            phi = levinson_durbin(r)
            @test all(abs.(phi) .< 1e-10)
        end

        @testset "error on zero variance" begin
            @test_throws ErrorException levinson_durbin([0.0, 0.0, 0.0])
        end

    end  # Levinson-Durbin

    # =========================================================================
    @testset "Polynomial Roots" begin

        @testset "x^2 - 1 = (x-1)(x+1)" begin
            roots = poly_roots([1.0, 0.0, -1.0])
            real_roots = sort(real.(filter(r -> abs(imag(r)) < 1e-8, roots)))
            @test length(real_roots) == 2
            @test approx(real_roots[1], -1.0, atol=1e-10)
            @test approx(real_roots[2],  1.0, atol=1e-10)
        end

        @testset "x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)" begin
            roots = poly_roots([1.0, -6.0, 11.0, -6.0])
            real_roots = sort(real.(filter(r -> abs(imag(r)) < 1e-8, roots)))
            @test isapprox(real_roots, [1.0, 2.0, 3.0], atol=1e-8)
        end

        @testset "x^2 + 1 = (x-i)(x+i): complex roots" begin
            roots = poly_roots([1.0, 0.0, 1.0])
            @test length(roots) == 2
            imag_parts = sort(abs.(imag.(roots)))
            @test approx(imag_parts[1], 1.0, atol=1e-10)
            @test approx(imag_parts[2], 1.0, atol=1e-10)
        end

        @testset "linear polynomial: x - 5" begin
            roots = poly_roots([1.0, -5.0])
            @test length(roots) == 1
            @test approx(real(roots[1]), 5.0, atol=1e-10)
        end

        @testset "leading coefficient zero raises error" begin
            @test_throws ArgumentError poly_roots([0.0, 1.0, -1.0])
        end

    end  # Polynomial Roots

    # =========================================================================
    @testset "poly_eval" begin

        @testset "constant polynomial" begin
            @test approx(poly_eval([7.0], 3.0), 7.0)
        end

        @testset "linear polynomial 2x + 3 at x=5" begin
            @test approx(poly_eval([2.0, 3.0], 5.0), 13.0)
        end

        @testset "x^2 - 1 at x=3" begin
            @test approx(poly_eval([1.0, 0.0, -1.0], 3.0), 8.0)
        end

    end  # poly_eval

    # =========================================================================
    @testset "Toeplitz Solve" begin

        @testset "2x2 symmetric Toeplitz" begin
            # T = [[4, 1], [1, 4]]
            # Inverse: 1/15 * [[4, -1], [-1, 4]]
            # T * x = [5, 5] -> x = [1, 1] (by symmetry)
            c = [4.0, 1.0]
            r = [4.0, 1.0]
            b = [5.0, 5.0]
            x = toeplitz_solve(c, r, b)
            @test approx(x[1], 1.0, atol=1e-10)
            @test approx(x[2], 1.0, atol=1e-10)
        end

        @testset "3x3 Toeplitz roundtrip" begin
            c = [4.0, 1.0, 0.5]
            r = [4.0, 1.0, 0.5]
            # Construct T and a known x, compute b = T*x, then solve
            T = [c[abs(i-j)+1] for i in 1:3, j in 1:3]
            x_true = [1.0, 2.0, 3.0]
            b_vec = T * x_true
            x_sol = toeplitz_solve(c, r, b_vec)
            @test approx_vec(x_sol, x_true, atol=1e-10)
        end

        @testset "dimension mismatch raises error" begin
            @test_throws DimensionMismatch toeplitz_solve([1.0, 2.0], [1.0], [1.0, 2.0])
        end

        @testset "c[1] ≠ r[1] raises error" begin
            @test_throws ArgumentError toeplitz_solve([4.0, 1.0], [3.0, 1.0], [1.0, 2.0])
        end

    end  # Toeplitz Solve

    # =========================================================================
    @testset "symmetric_toeplitz_solve" begin

        @testset "4x4 symmetric Toeplitz roundtrip" begin
            t = [5.0, 2.0, 1.0, 0.5]
            T = [t[abs(i-j)+1] for i in 1:4, j in 1:4]
            x_true = [1.0, -1.0, 2.0, 0.5]
            b_vec = T * x_true
            x_sol = symmetric_toeplitz_solve(t, b_vec)
            @test approx_vec(x_sol, x_true, atol=1e-8)
        end

    end  # symmetric_toeplitz_solve

    # =========================================================================
    @testset "Nelder-Mead Optimization" begin

        @testset "1D quadratic: minimize (x-3)^2" begin
            res = nelder_mead(v -> (v[1] - 3.0)^2, [0.0]; tol=1e-10, max_iter=5000)
            @test approx(res.x[1], 3.0, atol=1e-5)
            @test approx(res.f_val, 0.0, atol=1e-10)
        end

        @testset "2D quadratic: minimize x^2 + y^2" begin
            res = nelder_mead(v -> v[1]^2 + v[2]^2, [2.0, 3.0]; tol=1e-10)
            @test approx(res.x[1], 0.0, atol=1e-4)
            @test approx(res.x[2], 0.0, atol=1e-4)
            @test approx(res.f_val, 0.0, atol=1e-8)
        end

        @testset "2D Rosenbrock" begin
            rosenbrock(v) = (1.0 - v[1])^2 + 100.0 * (v[2] - v[1]^2)^2
            res = nelder_mead(rosenbrock, [0.0, 0.0]; tol=1e-10, max_iter=50000)
            @test approx(res.x[1], 1.0, atol=1e-3)
            @test approx(res.x[2], 1.0, atol=1e-3)
            @test approx(res.f_val, 0.0, atol=1e-5)
        end

        @testset "converged flag is set when tolerance met" begin
            res = nelder_mead(v -> v[1]^2, [1.0]; tol=1e-8, max_iter=10000)
            @test res.converged
        end

    end  # Nelder-Mead

    # =========================================================================
    @testset "Conjugate Gradient" begin

        @testset "2x2 SPD system" begin
            A = [4.0 1.0; 1.0 3.0]
            b = [1.0, 2.0]
            res = conjugate_gradient(A, b; tol=1e-12)
            x_ref = A \ b
            @test approx_vec(res.x, x_ref, atol=1e-10)
            @test res.converged
        end

        @testset "5x5 diagonal SPD system" begin
            d = [2.0, 3.0, 4.0, 5.0, 6.0]
            A = diagm(d)
            b = [1.0, 2.0, 3.0, 4.0, 5.0]
            x_expected = b ./ d
            res = conjugate_gradient(A, b; tol=1e-12)
            @test approx_vec(res.x, x_expected, atol=1e-10)
        end

        @testset "zero rhs gives zero solution" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [0.0, 0.0]
            res = conjugate_gradient(A, b)
            @test approx_vec(res.x, [0.0, 0.0])
        end

    end  # Conjugate Gradient

    # =========================================================================
    @testset "Autocorrelation" begin

        @testset "lag-0 is always 1" begin
            x = [1.0, 3.0, 2.0, 5.0, 4.0]
            acf = autocorrelation(x, 3)
            @test approx(acf[1], 1.0, atol=1e-14)
        end

        @testset "perfectly correlated AR(1)-like: lag-1 near φ" begin
            # White noise has near-zero ACF at non-zero lags
            x = Float64.(1:10)  # perfect trend, ACF should be high
            acf = autocorrelation(x, 5)
            @test acf[2] > 0.5  # positive correlation at lag 1
        end

        @testset "length matches max_lag + 1" begin
            x = collect(1.0:20.0)
            acf = autocorrelation(x, 7)
            @test length(acf) == 8
        end

        @testset "invalid max_lag raises error" begin
            x = [1.0, 2.0, 3.0]
            @test_throws ArgumentError autocorrelation(x, 3)  # lag >= n
        end

    end  # Autocorrelation

    # =========================================================================
    @testset "Cross-Correlation" begin

        @testset "CCF of identical signals: max at lag 0" begin
            x = [1.0, 2.0, 3.0, 2.0, 1.0]
            result = cross_correlation(x, x, 3)
            idx_zero = findfirst(==(0), result.lags)
            @test result.ccf[idx_zero] == maximum(result.ccf)
        end

        @testset "CCF lags span -max_lag to max_lag" begin
            x = ones(Float64, 10)
            y = ones(Float64, 10)
            result = cross_correlation(x, y, 4)
            @test result.lags == -4:4
            @test length(result.ccf) == 9
        end

        @testset "dimension mismatch raises error" begin
            @test_throws DimensionMismatch cross_correlation([1.0, 2.0], [1.0, 2.0, 3.0], 1)
        end

    end  # Cross-Correlation

    # =========================================================================
    @testset "Periodogram" begin

        @testset "DC signal: all power at frequency 0" begin
            x = ones(Float64, 8)
            result = periodogram(x; fs=1.0)
            @test result.power[1] > 0.0
            # All higher-frequency bins should be near zero
            for p in result.power[2:end]
                @test abs(p) < 1e-10
            end
        end

        @testset "frequency length" begin
            n = 10
            x = collect(Float64, 1:n)
            result = periodogram(x)
            @test length(result.frequencies) == div(n, 2) + 1
            @test length(result.power) == div(n, 2) + 1
        end

        @testset "frequencies are non-negative and bounded by fs/2" begin
            fs = 100.0
            x = randn(64)
            result = periodogram(x; fs=fs)
            @test all(result.frequencies .>= 0.0)
            @test result.frequencies[end] <= fs / 2.0 + eps()
        end

    end  # Periodogram

end  # SciRS2 PureAlgorithms

println("\nAll PureAlgorithms tests completed.")
