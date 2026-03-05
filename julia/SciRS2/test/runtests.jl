"""
SciRS2 Julia wrapper test suite.

Tests are organized into groups matching the SciRS2 functional domains:
- Library loading and version introspection
- Linear algebra (det, inv, solve, svd, eig, matmul)
- Statistics (mean, std, median, percentile, variance, correlation)
- FFT (forward, inverse, rfft, roundtrip)
- Optimization (minimize_scalar, minimize_brent, minimize_nelder_mead, find_root)

NOTE: These tests require the SciRS2 shared library to be built first.
      Run `scripts/build_shared_lib.sh` from the workspace root before
      executing this test suite.

To run:
    julia --project=julia/SciRS2 julia/SciRS2/test/runtests.jl
"""

using Test

# ---- Load the SciRS2 module ----
# We import from source so tests work without the package being registered.
include(joinpath(@__DIR__, "..", "src", "SciRS2.jl"))
using .SciRS2

# Helper: approximate equality with tolerance
approx(a, b; atol=1e-8) = abs(a - b) < atol
approx_vec(a, b; atol=1e-8) = all(abs.(a .- b) .< atol)
approx_mat(a, b; atol=1e-8) = all(abs.(a .- b) .< atol)

@testset "SciRS2 Julia Wrapper" begin

    # -----------------------------------------------------------------------
    @testset "Library version" begin
        ver = SciRS2.version()
        @test isa(ver, String)
        @test !isempty(ver)
        # Version string should look like "0.3.0" or similar
        @test occursin(r"\d+\.\d+", ver)
    end

    # -----------------------------------------------------------------------
    @testset "Linear algebra" begin

        @testset "matmul" begin
            A = [1.0 2.0; 3.0 4.0]
            B = [5.0 6.0; 7.0 8.0]
            C = SciRS2.matmul(A, B)
            @test size(C) == (2, 2)
            @test approx(C[1, 1], 19.0)
            @test approx(C[1, 2], 22.0)
            @test approx(C[2, 1], 43.0)
            @test approx(C[2, 2], 50.0)
        end

        @testset "matmul dimension mismatch" begin
            A = [1.0 2.0 3.0; 4.0 5.0 6.0]   # 2x3
            B = [1.0 2.0; 3.0 4.0]             # 2x2
            @test_throws DimensionMismatch SciRS2.matmul(A, B)
        end

        @testset "det 2x2" begin
            # det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
            A = [1.0 2.0; 3.0 4.0]
            d = SciRS2.det(A)
            @test approx(d, -2.0)
        end

        @testset "det 3x3 identity" begin
            I3 = Matrix{Float64}(I, 3, 3)
            d = SciRS2.det(I3)
            @test approx(d, 1.0)
        end

        @testset "det non-square errors" begin
            A = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3
            @test_throws ErrorException SciRS2.det(A)
        end

        @testset "inv 2x2" begin
            # inv([[4,7],[2,6]]) = [[0.6,-0.7],[-0.2,0.4]]
            A = [4.0 7.0; 2.0 6.0]
            Ainv = SciRS2.inv(A)
            @test size(Ainv) == (2, 2)
            @test approx(Ainv[1, 1], 0.6)
            @test approx(Ainv[1, 2], -0.7)
            @test approx(Ainv[2, 1], -0.2)
            @test approx(Ainv[2, 2], 0.4)
        end

        @testset "inv roundtrip: A * inv(A) == I" begin
            A = [2.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
            Ainv = SciRS2.inv(A)
            product = A * Ainv
            I3 = Matrix{Float64}(I, 3, 3)
            @test approx_mat(product, I3; atol=1e-8)
        end

        @testset "solve Ax = b (2x2)" begin
            # [[2,1],[1,3]] * x = [5, 7]  => x = [1.6, 1.8]
            A = [2.0 1.0; 1.0 3.0]
            b = [5.0, 7.0]
            x = SciRS2.solve(A, b)
            @test length(x) == 2
            @test approx(x[1], 1.6)
            @test approx(x[2], 1.8)
        end

        @testset "solve Ax = b (3x3)" begin
            A = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]  # diagonal
            b = [1.0, 2.0, 4.0]
            x = SciRS2.solve(A, b)
            @test approx_vec(x, [1.0, 1.0, 1.0])
        end

        @testset "svd 2x2 identity" begin
            I2 = Matrix{Float64}(I, 2, 2)
            result = SciRS2.svd(I2)
            @test haskey(result, :U)
            @test haskey(result, :S)
            @test haskey(result, :Vt)
            @test length(result.S) == 2
            s_sorted = sort(result.S, rev=true)
            @test approx(s_sorted[1], 1.0)
            @test approx(s_sorted[2], 1.0)
        end

        @testset "svd reconstruction: A ≈ U * Diagonal(S) * Vt" begin
            A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            result = SciRS2.svd(A)
            A_reconstructed = result.U * Diagonal(result.S) * result.Vt
            @test approx_mat(A_reconstructed, A; atol=1e-8)
        end

        @testset "eig diagonal matrix" begin
            # [[3,0],[0,5]] eigenvalues should be 3 and 5
            A = [3.0 0.0; 0.0 5.0]
            result = SciRS2.eig(A)
            @test haskey(result, :values)
            @test haskey(result, :vectors)
            @test length(result.values) == 2
            real_parts = sort(real.(result.values))
            @test approx(real_parts[1], 3.0; atol=1e-8)
            @test approx(real_parts[2], 5.0; atol=1e-8)
            # Imaginary parts should be zero
            for v in result.values
                @test abs(imag(v)) < 1e-8
            end
        end

    end  # @testset "Linear algebra"

    # -----------------------------------------------------------------------
    @testset "Statistics" begin

        @testset "mean" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test approx(SciRS2.mean(x), 3.0)
        end

        @testset "mean single element" begin
            x = [42.0]
            @test approx(SciRS2.mean(x), 42.0)
        end

        @testset "mean empty errors" begin
            @test_throws ErrorException SciRS2.mean(Float64[])
        end

        @testset "std sample (ddof=1)" begin
            # [2,4,4,4,5,5,7,9] => sample std = 2
            x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
            s = SciRS2.std(x; ddof=1)
            @test approx(s, 2.0; atol=1e-10)
        end

        @testset "std population (ddof=0)" begin
            x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
            s = SciRS2.std(x; ddof=0)
            # population variance = 4 * 7/8 = 3.5, std = sqrt(3.5) ≈ 1.8708
            @test approx(s, sqrt(3.5); atol=1e-8)
        end

        @testset "median odd length" begin
            x = [3.0, 1.0, 4.0, 1.0, 5.0]
            @test approx(SciRS2.median(x), 3.0)
        end

        @testset "median even length" begin
            x = [1.0, 2.0, 3.0, 4.0]
            @test approx(SciRS2.median(x), 2.5)
        end

        @testset "percentile 50th == median" begin
            x = collect(1.0:100.0)
            p50 = SciRS2.percentile(x, 50.0)
            med = SciRS2.median(x)
            @test approx(p50, med; atol=1e-8)
        end

        @testset "percentile 0th and 100th" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test approx(SciRS2.percentile(x, 0.0), 1.0)
            @test approx(SciRS2.percentile(x, 100.0), 5.0)
        end

        @testset "percentile out of range errors" begin
            x = [1.0, 2.0, 3.0]
            @test_throws ErrorException SciRS2.percentile(x, -1.0)
            @test_throws ErrorException SciRS2.percentile(x, 101.0)
        end

        @testset "variance" begin
            x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
            v = SciRS2.variance(x; ddof=1)
            @test approx(v, 4.0; atol=1e-10)
        end

        @testset "correlation perfect positive" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.0, 6.0, 8.0, 10.0]
            c = SciRS2.correlation(x, y)
            @test approx(c, 1.0; atol=1e-10)
        end

        @testset "correlation perfect negative" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [-1.0, -2.0, -3.0, -4.0, -5.0]
            c = SciRS2.correlation(x, y)
            @test approx(c, -1.0; atol=1e-10)
        end

        @testset "correlation uncorrelated" begin
            x = [1.0, 2.0, 3.0]
            y = [1.0, 1.0, 1.0]  # constant y => zero variance => error
            @test_throws ErrorException SciRS2.correlation(x, y)
        end

    end  # @testset "Statistics"

    # -----------------------------------------------------------------------
    @testset "FFT" begin

        @testset "fft_forward unit impulse" begin
            # Unit impulse at index 1 => flat spectrum of all ones
            n = 8
            x = zeros(Float64, n)
            x[1] = 1.0
            X = SciRS2.fft_forward(x)
            @test length(X) == n
            for i in 1:n
                @test approx(real(X[i]), 1.0; atol=1e-10)
                @test approx(imag(X[i]), 0.0; atol=1e-10)
            end
        end

        @testset "fft_forward DC signal" begin
            # All-ones signal => DC component = n, all others = 0
            n = 8
            x = ones(Float64, n)
            X = SciRS2.fft_forward(x)
            @test approx(real(X[1]), Float64(n); atol=1e-10)
            for i in 2:n
                @test abs(X[i]) < 1e-10
            end
        end

        @testset "fft roundtrip: IFFT(FFT(x)) ≈ x" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            X = SciRS2.fft_forward(x)
            x_reconstructed = SciRS2.fft_inverse(real.(X), imag.(X))
            for i in eachindex(x)
                @test approx(real(x_reconstructed[i]), x[i]; atol=1e-10)
                @test abs(imag(x_reconstructed[i])) < 1e-10
            end
        end

        @testset "fft_forward arbitrary length (non-power-of-2)" begin
            # 7-point FFT via Bluestein
            x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # impulse
            X = SciRS2.fft_forward(x)
            @test length(X) == 7
            for i in 1:7
                @test approx(real(X[i]), 1.0; atol=1e-8)
                @test approx(imag(X[i]), 0.0; atol=1e-8)
            end
        end

        @testset "rfft equals fft_forward on real input" begin
            x = [1.0, 2.0, 3.0, 4.0]
            X_forward = SciRS2.fft_forward(x)
            X_rfft = SciRS2.rfft(x)
            @test length(X_rfft) == length(X_forward)
            for i in eachindex(X_forward)
                @test approx(real(X_rfft[i]), real(X_forward[i]); atol=1e-10)
                @test approx(imag(X_rfft[i]), imag(X_forward[i]); atol=1e-10)
            end
        end

        @testset "fft_forward empty input" begin
            X = SciRS2.fft_forward(Float64[])
            @test length(X) == 0
        end

        @testset "fft_inverse empty input" begin
            x = SciRS2.fft_inverse(Float64[], Float64[])
            @test length(x) == 0
        end

    end  # @testset "FFT"

    # -----------------------------------------------------------------------
    @testset "Optimization" begin

        @testset "minimize_scalar quadratic" begin
            # f(x) = (x - 3)^2, minimum at x=3
            result = SciRS2.minimize_scalar(x -> (x - 3.0)^2, 0.0, 6.0; tol=1e-10)
            @test approx(result.x_min, 3.0; atol=1e-6)
            @test approx(result.f_min, 0.0; atol=1e-10)
        end

        @testset "minimize_scalar cosine" begin
            # min(cos(x)) on [3, 4] => minimum is cos(π) = -1 at x = π ≈ 3.14159
            result = SciRS2.minimize_scalar(x -> cos(x), 3.0, 4.0; tol=1e-10)
            @test approx(result.x_min, π; atol=1e-5)
            @test approx(result.f_min, -1.0; atol=1e-8)
        end

        @testset "minimize_scalar invalid interval errors" begin
            @test_throws ErrorException SciRS2.minimize_scalar(x -> x^2, 5.0, 3.0)
        end

        @testset "minimize_brent quadratic" begin
            result = SciRS2.minimize_brent(x -> (x - 2.5)^2, 0.0, 5.0; tol=1e-10)
            @test approx(result.x_min, 2.5; atol=1e-6)
            @test approx(result.f_min, 0.0; atol=1e-10)
        end

        @testset "minimize_brent is more accurate than golden section on smooth functions" begin
            # Quartic: f(x) = (x - 1)^4, sharp minimum
            result_brent = SciRS2.minimize_brent(x -> (x - 1.0)^4, -2.0, 4.0; tol=1e-12)
            @test approx(result_brent.x_min, 1.0; atol=1e-4)
        end

        @testset "minimize_nelder_mead 1D" begin
            # f(x) = (x[1] - 3)^2
            result = SciRS2.minimize_nelder_mead(
                v -> (v[1] - 3.0)^2, [0.0];
                tol=1e-10, max_iter=5000,
            )
            @test approx(result.x_min[1], 3.0; atol=1e-4)
            @test approx(result.f_min, 0.0; atol=1e-8)
        end

        @testset "minimize_nelder_mead 2D Rosenbrock" begin
            # Rosenbrock: minimum at (1, 1) with f=0
            rosenbrock(v) = (1.0 - v[1])^2 + 100.0 * (v[2] - v[1]^2)^2
            result = SciRS2.minimize_nelder_mead(
                rosenbrock, [0.0, 0.0];
                tol=1e-10, max_iter=50000,
            )
            @test approx(result.x_min[1], 1.0; atol=1e-3)
            @test approx(result.x_min[2], 1.0; atol=1e-3)
            @test approx(result.f_min, 0.0; atol=1e-5)
        end

        @testset "find_root quadratic" begin
            # x^2 - 4 = 0 => root at x=2 on [0, 10]
            root = SciRS2.find_root(x -> x^2 - 4.0, 0.0, 10.0; tol=1e-12)
            @test approx(root, 2.0; atol=1e-10)
        end

        @testset "find_root negative bracket" begin
            # x^2 - 4 = 0 => root at x=-2 on [-10, 0]
            root = SciRS2.find_root(x -> x^2 - 4.0, -10.0, 0.0; tol=1e-12)
            @test approx(root, -2.0; atol=1e-10)
        end

        @testset "find_root transcendental" begin
            # sin(x) = 0 => root at x=π on [2, 4]
            root = SciRS2.find_root(x -> sin(x), 2.0, 4.0; tol=1e-12)
            @test approx(root, π; atol=1e-10)
        end

        @testset "find_root no sign change errors" begin
            # x^2 on [1, 2] has no sign change => error
            @test_throws ErrorException SciRS2.find_root(x -> x^2, 1.0, 2.0; tol=1e-12)
        end

    end  # @testset "Optimization"

end  # @testset "SciRS2 Julia Wrapper"
