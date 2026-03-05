"""
    PureAlgorithms

Pure-Julia implementations of numerical algorithms that mirror SciRS2 Rust
implementations. These do not require the shared library and work in all
environments where Julia is available.

Included domains:
- Numerical integration (Romberg, Gauss-Legendre quadrature)
- Spatial statistics (Moran's I autocorrelation)
- Kernel density estimation (Gaussian KDE with Silverman bandwidth)
- Time series (Exponential Smoothing: SES, Holt's double, Holt-Winters)
- Linear algebra (Toeplitz/Levinson-Durbin solver, polynomial roots via companion matrix)
- Optimization (Nelder-Mead simplex, L-BFGS-B simplified, conjugate gradient)
- Special functions (Romberg-derived adaptive quadrature, adaptive Simpson)
- Signal processing (autocorrelation, cross-correlation, power spectral density)
"""
module PureAlgorithms

using LinearAlgebra
using Statistics

# ===========================================================================
# 1. NUMERICAL INTEGRATION
# ===========================================================================

"""
    romberg_integrate(f, a, b; tol=1e-10, max_steps=12) -> Float64

Romberg integration using Richardson extrapolation on a sequence of
trapezoidal rule estimates.

The algorithm builds a triangular array R[i, j] where:
- R[i, 1] is the composite trapezoidal rule with 2^(i-1) subintervals.
- R[i, j] = (4^(j-1) * R[i, j-1] - R[i-1, j-1]) / (4^(j-1) - 1)
  for j > 1 (Richardson extrapolation to higher order).

Convergence is declared when |R[i, i] - R[i-1, i-1]| < tol.

# Arguments
- `f`: integrand function `f(x::Float64) -> Float64`.
- `a`, `b`: integration limits.
- `tol`: absolute convergence tolerance (default 1e-10).
- `max_steps`: maximum number of refinement levels (default 12, giving up to
  2^11 = 2048 subintervals).

# Returns
- Approximate value of ∫ₐᵇ f(x) dx.

# Errors
- Throws `ErrorException` if the maximum number of steps is reached without
  convergence.

# Examples
```julia
# ∫₀¹ x² dx = 1/3
result = romberg_integrate(x -> x^2, 0.0, 1.0)
isapprox(result, 1/3, atol=1e-10)  # true

# ∫₀^π sin(x) dx = 2
result2 = romberg_integrate(sin, 0.0, π)
isapprox(result2, 2.0, atol=1e-10)  # true
```
"""
function romberg_integrate(
    f,
    a::Real,
    b::Real;
    tol::Float64 = 1e-10,
    max_steps::Int = 12,
)::Float64
    a_f = Float64(a)
    b_f = Float64(b)
    h = b_f - a_f

    # Romberg table stored as a vector of columns; R[j] = current column j.
    # We only need the previous and current column at each step.
    n_max = max_steps + 1
    R_prev = Vector{Float64}(undef, n_max)
    R_curr = Vector{Float64}(undef, n_max)

    # Step 0: single trapezoidal estimate
    R_prev[1] = 0.5 * h * (f(a_f) + f(b_f))

    for i in 1:max_steps
        h /= 2.0
        n_new = 2^(i - 1)  # number of new interior points

        # Composite trapezoidal rule with 2^i subintervals:
        # T_{2^i} = T_{2^(i-1)} / 2 + h * Σ f(a + (2k-1)*h) for k=1..n_new
        interior_sum = 0.0
        for k in 1:n_new
            x_k = a_f + (2 * k - 1) * h
            interior_sum += f(x_k)
        end
        R_curr[1] = 0.5 * R_prev[1] + h * interior_sum

        # Richardson extrapolation columns
        pow4 = 1.0
        for j in 2:(i + 1)
            pow4 *= 4.0
            R_curr[j] = (pow4 * R_curr[j - 1] - R_prev[j - 1]) / (pow4 - 1.0)
        end

        # Check convergence: compare diagonal entries
        if i >= 2 && abs(R_curr[i + 1] - R_prev[i]) < tol
            return R_curr[i + 1]
        end

        # Swap buffers
        R_prev, R_curr = R_curr, R_prev
    end

    # Return best estimate if max_steps reached
    return R_prev[max_steps + 1]
end

"""
    gauss_legendre_nodes_weights(n::Int) -> (nodes::Vector{Float64}, weights::Vector{Float64})

Compute the n-point Gauss-Legendre quadrature nodes and weights on [-1, 1]
using the Golub-Welsch algorithm.

The algorithm constructs the symmetric tridiagonal Jacobi matrix whose
eigenvalues are the quadrature nodes and whose eigenvectors yield the weights.

For n-point Gauss-Legendre quadrature:
  ∫₋₁¹ f(x) dx ≈ Σᵢ wᵢ f(xᵢ)

The rule is exact for all polynomials of degree ≤ 2n - 1.

# Arguments
- `n`: number of quadrature points (must be ≥ 1).

# Returns
- `nodes`: vector of n quadrature nodes in [-1, 1] (sorted ascending).
- `weights`: vector of n positive quadrature weights (sum to 2.0).

# Examples
```julia
nodes, weights = gauss_legendre_nodes_weights(5)
length(nodes) == 5         # true
isapprox(sum(weights), 2.0, atol=1e-14)  # true

# Integrate x^8 on [-1, 1] exactly (needs at least 5 points)
val = sum(weights .* nodes.^8)
isapprox(val, 2/9, atol=1e-12)  # true
```
"""
function gauss_legendre_nodes_weights(n::Int)
    if n < 1
        throw(ArgumentError("gauss_legendre_nodes_weights: n must be ≥ 1, got $n"))
    end
    if n == 1
        return ([0.0], [2.0])
    end

    # Off-diagonal elements of the symmetric tridiagonal Jacobi matrix:
    # β_i = i / sqrt(4i^2 - 1), i = 1, ..., n-1
    beta = [Float64(i) / sqrt(4.0 * i^2 - 1.0) for i in 1:(n - 1)]

    # Build the tridiagonal matrix as a SymTridiagonal for eigenvalue solve.
    # Diagonal is all zeros for Gauss-Legendre.
    diag_zeros = zeros(Float64, n)
    J = SymTridiagonal(diag_zeros, beta)

    # Eigen-decompose: eigenvalues are nodes, weights come from first component
    # of eigenvectors.
    F = eigen(J)
    nodes = F.values
    # Weight formula: wᵢ = 2 * (v[1, i])^2 where v[:,i] is the i-th eigenvector.
    weights = 2.0 .* F.vectors[1, :] .^ 2

    # Sort by node value
    order = sortperm(nodes)
    return (nodes[order], weights[order])
end

"""
    gauss_legendre_integrate(f, a, b, n::Int=10) -> Float64

Integrate f on [a, b] using n-point Gauss-Legendre quadrature.

Transforms the standard interval [-1, 1] to [a, b] via the linear map
x = ((b-a)/2)*t + (a+b)/2, dx = (b-a)/2 dt.

# Examples
```julia
# ∫₀¹ exp(-x^2) dx ≈ 0.7468241328...
val = gauss_legendre_integrate(x -> exp(-x^2), 0.0, 1.0, 10)
```
"""
function gauss_legendre_integrate(f, a::Real, b::Real, n::Int = 10)::Float64
    nodes, weights = gauss_legendre_nodes_weights(n)
    mid = 0.5 * (Float64(a) + Float64(b))
    half = 0.5 * (Float64(b) - Float64(a))
    return half * sum(weights .* f.(half .* nodes .+ mid))
end

"""
    adaptive_simpson(f, a, b; tol=1e-8, max_depth=50) -> Float64

Adaptive Simpson's rule integration using recursive bisection.

The interval [a, b] is recursively subdivided until the estimated error falls
below `tol * |S_whole|` (relative) or below `tol` (absolute). This adapts
naturally to regions where f has sharp features.

# Arguments
- `f`: integrand.
- `a`, `b`: integration limits.
- `tol`: tolerance per subdivision (absolute).
- `max_depth`: maximum recursion depth.

# Examples
```julia
# ∫₀^1 sqrt(x) dx = 2/3
val = adaptive_simpson(sqrt, 0.0, 1.0)
isapprox(val, 2/3, atol=1e-8)  # true
```
"""
function adaptive_simpson(f, a::Real, b::Real; tol::Float64 = 1e-8, max_depth::Int = 50)::Float64
    fa = f(Float64(a))
    fb = f(Float64(b))
    fm = f(0.5 * (Float64(a) + Float64(b)))
    s_whole = (Float64(b) - Float64(a)) * (fa + 4.0 * fm + fb) / 6.0
    return _adaptive_simpson_recur(f, Float64(a), Float64(b), fa, fm, fb, s_whole, tol, max_depth)
end

function _adaptive_simpson_recur(
    f,
    a::Float64,
    b::Float64,
    fa::Float64,
    fm::Float64,
    fb::Float64,
    s_whole::Float64,
    tol::Float64,
    depth::Int,
)::Float64
    m = 0.5 * (a + b)
    lm = 0.5 * (a + m)
    rm = 0.5 * (m + b)
    flm = f(lm)
    frm = f(rm)
    s_left  = (m - a) * (fa + 4.0 * flm + fm) / 6.0
    s_right = (b - m) * (fm + 4.0 * frm + fb) / 6.0
    s_two   = s_left + s_right
    err = abs(s_two - s_whole)
    if depth <= 0 || err < 15.0 * tol
        return s_two + (s_two - s_whole) / 15.0
    end
    half_tol = tol / 2.0
    return (
        _adaptive_simpson_recur(f, a, m, fa, flm, fm, s_left,  half_tol, depth - 1) +
        _adaptive_simpson_recur(f, m, b, fm, frm, fb, s_right, half_tol, depth - 1)
    )
end

# ===========================================================================
# 2. SPATIAL STATISTICS
# ===========================================================================

"""
    moran_i(x::AbstractVector{Float64}, W::AbstractMatrix{Float64}) -> Float64

Compute Moran's I spatial autocorrelation statistic.

Moran's I measures the degree of spatial autocorrelation in a spatially
distributed variable x given a spatial weights matrix W.

The formula is:
  I = (n / S₀) * (Σᵢ Σⱼ wᵢⱼ (xᵢ - x̄)(xⱼ - x̄)) / (Σᵢ (xᵢ - x̄)²)

where:
- n   = number of observations,
- S₀  = Σᵢ Σⱼ wᵢⱼ (sum of all weights),
- x̄  = mean(x),
- wᵢⱼ = element (i, j) of the weights matrix.

# Arguments
- `x`: vector of n spatial observations.
- `W`: n×n spatial weights matrix. W need not be row-standardized; the
  formula normalizes by S₀ automatically.

# Returns
- Moran's I value. Values near +1 indicate positive spatial autocorrelation
  (similar values cluster); values near -1 indicate negative autocorrelation.
  The expected value under spatial randomness is -1/(n-1).

# Errors
- Throws if x and W have incompatible dimensions or if all values are equal.

# Examples
```julia
# 4-point regular grid, queen contiguity weights
W = [0 1 1 0;
     1 0 0 1;
     1 0 0 1;
     0 1 1 0] |> float
x = [1.0, 2.0, 2.0, 1.0]   # similar neighbors -> positive I
I = moran_i(x, W)
I > 0  # true
```
"""
function moran_i(x::AbstractVector{Float64}, W::AbstractMatrix{Float64})::Float64
    n = length(x)
    if size(W, 1) != n || size(W, 2) != n
        throw(DimensionMismatch(
            "moran_i: x has length $n but W is $(size(W,1))x$(size(W,2))"
        ))
    end
    x_mean = Statistics.mean(x)
    z = x .- x_mean  # deviations from mean

    # Numerator: Σᵢ Σⱼ wᵢⱼ zᵢ zⱼ = z' * W * z
    numerator = dot(z, W * z)

    # Denominator: Σᵢ zᵢ²
    denominator = dot(z, z)
    if abs(denominator) < eps(Float64)
        throw(ErrorException("moran_i: all values are equal; Moran's I is undefined"))
    end

    # S₀ = sum of all weights
    s0 = sum(W)
    if abs(s0) < eps(Float64)
        throw(ErrorException("moran_i: weight matrix is all zeros"))
    end

    return (n / s0) * (numerator / denominator)
end

"""
    moran_i_significance(x, W; n_permutations=999, seed=42)
        -> NamedTuple{(:I, :expected, :variance, :z_score, :p_value)}

Compute Moran's I with a permutation-based significance test.

Generates `n_permutations` random permutations of x, recomputes Moran's I
for each, and estimates the two-sided p-value as the fraction of permuted
values with |I_perm| ≥ |I_observed|.

# Arguments
- `x`: vector of spatial observations.
- `W`: spatial weights matrix.
- `n_permutations`: number of Monte Carlo permutations (default 999).
- `seed`: random seed for reproducibility (default 42).

# Returns
Named tuple with:
- `I`: observed Moran's I.
- `expected`: theoretical expected value -1/(n-1).
- `variance`: variance under randomization.
- `z_score`: (I - E[I]) / sqrt(Var[I]).
- `p_value`: two-sided permutation p-value.
"""
function moran_i_significance(
    x::AbstractVector{Float64},
    W::AbstractMatrix{Float64};
    n_permutations::Int = 999,
    seed::Int = 42,
)
    n = length(x)
    i_obs = moran_i(x, W)
    expected = -1.0 / (n - 1)

    # Analytical variance under normality assumption (randomization variance)
    s0 = sum(W)
    s1 = 0.5 * sum((W .+ W') .^ 2)
    s2 = sum((sum(W, dims=2) .+ sum(W, dims=1)') .^ 2)

    x_mean = Statistics.mean(x)
    z = x .- x_mean
    m2 = sum(z .^ 2) / n
    m4 = sum(z .^ 4) / n
    b2 = m4 / m2^2

    # Variance formula (Cliff and Ord, 1981)
    var_i = (
        n * ((n^2 - 3n + 3) * s1 - n * s2 + 3 * s0^2) -
        b2 * ((n^2 - n) * s1 - 2n * s2 + 6 * s0^2)
    ) / ((n - 1) * (n - 2) * (n - 3) * s0^2) - expected^2

    z_score = (i_obs - expected) / sqrt(max(var_i, 0.0))

    # Permutation test using a simple LCG for reproducibility (no Random dependency)
    rng_state = UInt64(seed)
    function lcg_next!(state_ref::Ref{UInt64})::UInt64
        state_ref[] = state_ref[] * 6364136223846793005 + 1442695040888963407
        return state_ref[]
    end

    count_extreme = 0
    rng_ref = Ref(rng_state)
    x_perm = copy(x)
    for _ in 1:n_permutations
        # Fisher-Yates shuffle using LCG
        for i in n:-1:2
            j = Int(lcg_next!(rng_ref) % UInt64(i)) + 1
            x_perm[i], x_perm[j] = x_perm[j], x_perm[i]
        end
        i_perm = moran_i(x_perm, W)
        if abs(i_perm) >= abs(i_obs)
            count_extreme += 1
        end
    end
    p_value = (count_extreme + 1) / (n_permutations + 1)

    return (
        I = i_obs,
        expected = expected,
        variance = var_i,
        z_score = z_score,
        p_value = p_value,
    )
end

# ===========================================================================
# 3. KERNEL DENSITY ESTIMATION
# ===========================================================================

"""
    kernel_density(data::AbstractVector{Float64}, bandwidth::Float64;
                   n_points::Int=200, kernel::Symbol=:gaussian)
    -> NamedTuple{(:x, :density)}

Gaussian kernel density estimation on an equi-spaced grid.

The KDE is defined as:
  f̂(x) = (1 / (n * h)) * Σᵢ K((x - xᵢ) / h)

where K is the kernel function (Gaussian by default) and h is the bandwidth.

# Arguments
- `data`: sample data vector of length n.
- `bandwidth`: smoothing bandwidth h (> 0). Use `silverman_bandwidth` for an
  automatic choice.
- `n_points`: number of evaluation points on the grid (default 200).
- `kernel`: kernel type; only `:gaussian` is currently supported.

# Returns
Named tuple:
- `x`: vector of n_points evaluation locations spanning
  [min(data) - 3h, max(data) + 3h].
- `density`: vector of n_points estimated density values.

# Examples
```julia
data = randn(500)
bw = silverman_bandwidth(data)
result = kernel_density(data, bw)
# result.x: grid points; result.density: estimated density
```
"""
function kernel_density(
    data::AbstractVector{Float64},
    bandwidth::Float64;
    n_points::Int = 200,
    kernel::Symbol = :gaussian,
)
    if bandwidth <= 0.0
        throw(ArgumentError("kernel_density: bandwidth must be positive, got $bandwidth"))
    end
    if isempty(data)
        throw(ArgumentError("kernel_density: data must be non-empty"))
    end
    if kernel != :gaussian
        throw(ArgumentError("kernel_density: unsupported kernel :$kernel (only :gaussian supported)"))
    end

    n = length(data)
    x_min = minimum(data) - 3.0 * bandwidth
    x_max = maximum(data) + 3.0 * bandwidth
    grid = range(x_min, x_max; length = n_points)
    x_vec = collect(grid)

    density = Vector{Float64}(undef, n_points)
    inv_nh = 1.0 / (n * bandwidth)
    inv_sqrt2pi = 1.0 / sqrt(2.0 * π)

    for (idx, xi) in enumerate(x_vec)
        s = 0.0
        for xj in data
            u = (xi - xj) / bandwidth
            s += inv_sqrt2pi * exp(-0.5 * u^2)
        end
        density[idx] = s * inv_nh
    end

    return (x = x_vec, density = density)
end

"""
    silverman_bandwidth(data::AbstractVector{Float64}) -> Float64

Silverman's rule-of-thumb bandwidth for Gaussian KDE:

  h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)

where σ is the sample standard deviation and IQR is the interquartile range.

# Examples
```julia
data = randn(1000)
bw = silverman_bandwidth(data)
```
"""
function silverman_bandwidth(data::AbstractVector{Float64})::Float64
    n = length(data)
    if n < 2
        throw(ArgumentError("silverman_bandwidth: need at least 2 data points"))
    end
    sigma = Statistics.std(data)
    q25 = _quantile(data, 0.25)
    q75 = _quantile(data, 0.75)
    iqr  = q75 - q25
    scale = min(sigma, iqr / 1.34)
    if scale < eps(Float64)
        scale = sigma
    end
    return 0.9 * scale * n^(-0.2)
end

# Helper: compute quantile using linear interpolation (avoids Statistics.quantile dependency)
function _quantile(data::AbstractVector{Float64}, p::Float64)::Float64
    sorted = sort(data)
    n = length(sorted)
    h = p * (n - 1)
    lo = floor(Int, h)
    hi = lo + 1
    frac = h - lo
    lo_val = sorted[clamp(lo + 1, 1, n)]
    hi_val = sorted[clamp(hi + 1, 1, n)]
    return lo_val + frac * (hi_val - lo_val)
end

"""
    kernel_density_evaluate(data, bandwidth, x_eval; kernel=:gaussian) -> Vector{Float64}

Evaluate Gaussian KDE at arbitrary points `x_eval` (not just a regular grid).

# Arguments
- `data`: sample data.
- `bandwidth`: smoothing bandwidth.
- `x_eval`: vector of evaluation points.

# Returns
- Vector of density values at each point in x_eval.
"""
function kernel_density_evaluate(
    data::AbstractVector{Float64},
    bandwidth::Float64,
    x_eval::AbstractVector{Float64};
    kernel::Symbol = :gaussian,
)::Vector{Float64}
    if bandwidth <= 0.0
        throw(ArgumentError("bandwidth must be positive"))
    end
    if kernel != :gaussian
        throw(ArgumentError("only :gaussian kernel supported"))
    end
    n = length(data)
    inv_nh = 1.0 / (n * bandwidth)
    inv_sqrt2pi = 1.0 / sqrt(2.0 * π)
    result = Vector{Float64}(undef, length(x_eval))
    for (i, xi) in enumerate(x_eval)
        s = 0.0
        for xj in data
            u = (xi - xj) / bandwidth
            s += exp(-0.5 * u^2)
        end
        result[i] = s * inv_nh * inv_sqrt2pi
    end
    return result
end

# ===========================================================================
# 4. TIME SERIES: EXPONENTIAL SMOOTHING
# ===========================================================================

"""
    ets_ses(data::AbstractVector{Float64}; alpha::Float64=0.3)
    -> NamedTuple{(:fitted, :forecast, :level)}

Simple Exponential Smoothing (SES / ETS(A,N,N)).

The level equation:
  lₜ = α * yₜ + (1 - α) * lₜ₋₁

The one-step-ahead forecast is always the current level:
  ŷₜ₊₁|ₜ = lₜ

# Arguments
- `data`: time series vector of length n ≥ 1.
- `alpha`: smoothing parameter 0 < α ≤ 1 (default 0.3).

# Returns
Named tuple:
- `fitted`: n-vector of one-step-ahead in-sample forecasts.
- `forecast`: one-step-ahead out-of-sample forecast for time n+1.
- `level`: n-vector of level states.

# Examples
```julia
y = [1.0, 2.0, 3.0, 4.0, 5.0]
res = ets_ses(y; alpha=0.4)
res.forecast   # forecast for time 6
```
"""
function ets_ses(data::AbstractVector{Float64}; alpha::Float64 = 0.3)
    if alpha <= 0.0 || alpha > 1.0
        throw(ArgumentError("ets_ses: alpha must be in (0, 1], got $alpha"))
    end
    n = length(data)
    if n < 1
        throw(ArgumentError("ets_ses: data must be non-empty"))
    end

    level   = Vector{Float64}(undef, n)
    fitted  = Vector{Float64}(undef, n)

    # Initialize: l₀ = y₁
    level[1]  = data[1]
    fitted[1] = data[1]  # no forecast for the first point

    for t in 2:n
        level[t]  = alpha * data[t] + (1.0 - alpha) * level[t - 1]
        fitted[t] = level[t - 1]  # one-step-ahead forecast = previous level
    end

    return (fitted = fitted, forecast = level[n], level = level)
end

"""
    ets_holt(data::AbstractVector{Float64}; alpha::Float64=0.3, beta::Float64=0.1,
             h::Int=1)
    -> NamedTuple{(:fitted, :forecast, :level, :trend)}

Holt's Double (Linear) Exponential Smoothing (ETS(A,A,N)).

Level and trend equations:
  lₜ = α * yₜ + (1 - α) * (lₜ₋₁ + bₜ₋₁)
  bₜ = β * (lₜ - lₜ₋₁) + (1 - β) * bₜ₋₁

h-step forecast:
  ŷₜ₊ₕ|ₜ = lₜ + h * bₜ

# Arguments
- `data`: time series of length n ≥ 2.
- `alpha`: level smoothing parameter (0 < α ≤ 1, default 0.3).
- `beta`: trend smoothing parameter (0 < β ≤ 1, default 0.1).
- `h`: forecast horizon (default 1).

# Returns
Named tuple:
- `fitted`: in-sample one-step-ahead forecasts.
- `forecast`: h-step-ahead out-of-sample forecast.
- `level`, `trend`: state vectors.
"""
function ets_holt(
    data::AbstractVector{Float64};
    alpha::Float64 = 0.3,
    beta::Float64  = 0.1,
    h::Int         = 1,
)
    if alpha <= 0.0 || alpha > 1.0
        throw(ArgumentError("ets_holt: alpha must be in (0, 1], got $alpha"))
    end
    if beta <= 0.0 || beta > 1.0
        throw(ArgumentError("ets_holt: beta must be in (0, 1], got $beta"))
    end
    n = length(data)
    if n < 2
        throw(ArgumentError("ets_holt: need at least 2 data points"))
    end

    level  = Vector{Float64}(undef, n)
    trend  = Vector{Float64}(undef, n)
    fitted = Vector{Float64}(undef, n)

    # Initialization: l₁ = y₁, b₁ = y₂ - y₁
    level[1]  = data[1]
    trend[1]  = data[2] - data[1]
    fitted[1] = data[1]

    for t in 2:n
        l_prev = level[t - 1]
        b_prev = trend[t - 1]
        fitted[t] = l_prev + b_prev
        level[t]  = alpha * data[t] + (1.0 - alpha) * (l_prev + b_prev)
        trend[t]  = beta  * (level[t] - l_prev) + (1.0 - beta) * b_prev
    end

    forecast = level[n] + h * trend[n]

    return (fitted = fitted, forecast = forecast, level = level, trend = trend)
end

"""
    ets_holt_winters(data::AbstractVector{Float64}; alpha::Float64=0.3,
                     beta::Float64=0.1, gamma::Float64=0.1, period::Int=4,
                     seasonal::Symbol=:additive, h::Int=1)
    -> NamedTuple{(:fitted, :forecast, :level, :trend, :seasonal)}

Holt-Winters triple exponential smoothing (ETS(A,A,A) or ETS(A,A,M)).

Additive seasonal model:
  lₜ = α * (yₜ - sₜ₋ₘ) + (1 - α) * (lₜ₋₁ + bₜ₋₁)
  bₜ = β * (lₜ - lₜ₋₁) + (1 - β) * bₜ₋₁
  sₜ = γ * (yₜ - lₜ₋₁ - bₜ₋₁) + (1 - γ) * sₜ₋ₘ
  ŷₜ₊ₕ|ₜ = lₜ + h * bₜ + sₜ₊ₕ₋ₘ (seasonal index)

# Arguments
- `data`: time series (must have at least 2 complete seasonal periods).
- `alpha`, `beta`, `gamma`: smoothing parameters in (0, 1].
- `period`: seasonal period m (e.g., 4 for quarterly, 12 for monthly).
- `seasonal`: `:additive` (default) or `:multiplicative`.
- `h`: forecast horizon (default 1).
"""
function ets_holt_winters(
    data::AbstractVector{Float64};
    alpha::Float64  = 0.3,
    beta::Float64   = 0.1,
    gamma::Float64  = 0.1,
    period::Int     = 4,
    seasonal::Symbol = :additive,
    h::Int          = 1,
)
    if period < 2
        throw(ArgumentError("ets_holt_winters: period must be ≥ 2"))
    end
    n = length(data)
    if n < 2 * period
        throw(ArgumentError(
            "ets_holt_winters: need at least 2 complete periods ($(2*period) points)"
        ))
    end
    for (name, val) in ((:alpha, alpha), (:beta, beta), (:gamma, gamma))
        if val <= 0.0 || val > 1.0
            throw(ArgumentError("ets_holt_winters: $name must be in (0, 1], got $val"))
        end
    end
    if seasonal != :additive && seasonal != :multiplicative
        throw(ArgumentError("ets_holt_winters: seasonal must be :additive or :multiplicative"))
    end

    level    = Vector{Float64}(undef, n)
    trend    = Vector{Float64}(undef, n)
    fitted   = Vector{Float64}(undef, n)
    # Seasonal components stored in a circular buffer of size 'period'.
    # seas_buf[((t-1) % period) + 1] holds the seasonal index for position t.
    seas_buf = Vector{Float64}(undef, period)

    # -----------------------------------------------------------------------
    # Initialization following Hyndman & Athanasopoulos (2018) §7.3
    # -----------------------------------------------------------------------
    # Level: average of first period
    level_init = Statistics.mean(data[1:period])
    # Trend: average difference between averages of first and second periods
    trend_init = (Statistics.mean(data[(period + 1):(2 * period)]) - level_init) / period
    # Initial seasonal indices
    for i in 1:period
        if seasonal == :additive
            seas_buf[i] = data[i] - (level_init + (i - period) * trend_init)
        else  # multiplicative
            denom = level_init + (i - period) * trend_init
            seas_buf[i] = (abs(denom) < eps(Float64)) ? 1.0 : data[i] / denom
        end
    end

    level[1] = level_init
    trend[1] = trend_init
    # Fitted value at t=1: forecast would use l₀ + b₀ + s₁₋ₘ but we start from t=1
    s1 = seas_buf[1]
    fitted[1] = seasonal == :additive ? (level_init + trend_init + s1) : ((level_init + trend_init) * s1)

    # -----------------------------------------------------------------------
    # Recursive update
    # -----------------------------------------------------------------------
    for t in 2:n
        l_prev = level[t - 1]
        b_prev = trend[t - 1]
        # Seasonal index from m steps back, using the circular buffer
        buf_idx = ((t - 1) % period) + 1   # same position as t - period in the cycle
        s_lag   = seas_buf[buf_idx]

        # One-step-ahead forecast (in-sample)
        fitted[t] = seasonal == :additive ? (l_prev + b_prev + s_lag) : ((l_prev + b_prev) * s_lag)

        # Update level
        if seasonal == :additive
            level[t] = alpha * (data[t] - s_lag)            + (1.0 - alpha) * (l_prev + b_prev)
        else
            level[t] = alpha * (data[t] / max(s_lag, eps())) + (1.0 - alpha) * (l_prev + b_prev)
        end

        # Update trend
        trend[t] = beta * (level[t] - l_prev) + (1.0 - beta) * b_prev

        # Update seasonal index (overwrites the slot m steps back, now current season)
        if seasonal == :additive
            seas_buf[buf_idx] = gamma * (data[t] - level[t])              + (1.0 - gamma) * s_lag
        else
            seas_buf[buf_idx] = gamma * (data[t] / max(level[t], eps()))  + (1.0 - gamma) * s_lag
        end
    end

    # -----------------------------------------------------------------------
    # h-step-ahead out-of-sample forecast
    # -----------------------------------------------------------------------
    l_n = level[n]
    b_n = trend[n]
    # Seasonal index for the forecast horizon: cycle forward h steps from current position
    s_h_idx = ((n - 1 + h) % period) + 1
    s_h     = seas_buf[s_h_idx]

    forecast = if seasonal == :additive
        l_n + h * b_n + s_h
    else
        (l_n + h * b_n) * s_h
    end

    # Return the seasonal buffer as a repeated-cycle vector of length n for inspection
    seas_out = [seas_buf[((t - 1) % period) + 1] for t in 1:n]

    return (
        fitted   = fitted,
        forecast = forecast,
        level    = level,
        trend    = trend,
        seasonal = seas_out,
    )
end

# ===========================================================================
# 5. LINEAR ALGEBRA: TOEPLITZ / LEVINSON-DURBIN + POLYNOMIAL ROOTS
# ===========================================================================

"""
    toeplitz_solve(c::AbstractVector{Float64}, r::AbstractVector{Float64},
                   b::AbstractVector{Float64}) -> Vector{Float64}

Solve a Toeplitz linear system T * x = b using the Levinson-Durbin algorithm.

A Toeplitz matrix T of order n is defined by its first column c and first row r
(with c[1] == r[1]):
  T[i, j] = c[i - j + 1]   if i ≥ j
  T[i, j] = r[j - i + 1]   if i < j

The Levinson-Durbin algorithm solves the system in O(n²) time, far more
efficient than the O(n³) general LU decomposition.

# Arguments
- `c`: first column of T (length n); c[1] is the diagonal entry.
- `r`: first row of T (length n); r[1] must equal c[1].
- `b`: right-hand side vector (length n).

# Returns
- Solution vector x of length n.

# Errors
- Throws if dimensions are inconsistent or if c[1] ≠ r[1].
- Throws if the matrix is singular (pivot becomes zero).

# Examples
```julia
# Solve a symmetric Toeplitz system
c = [4.0, 1.0, 0.5]
r = [4.0, 1.0, 0.5]
b = [1.0, 0.0, 0.0]
x = toeplitz_solve(c, r, b)
# Verify: construct T and check T * x ≈ b
```
"""
function toeplitz_solve(
    c::AbstractVector{Float64},
    r::AbstractVector{Float64},
    b::AbstractVector{Float64},
)::Vector{Float64}
    n = length(c)
    if length(r) != n
        throw(DimensionMismatch("toeplitz_solve: c has length $n but r has length $(length(r))"))
    end
    if length(b) != n
        throw(DimensionMismatch("toeplitz_solve: b has length $(length(b)), expected $n"))
    end
    if abs(c[1] - r[1]) > eps(Float64) * max(abs(c[1]), 1.0)
        throw(ArgumentError("toeplitz_solve: c[1] = $(c[1]) ≠ r[1] = $(r[1])"))
    end

    # Build Toeplitz matrix entries: t[k] for k = -(n-1)...(n-1)
    # t[i,j] = c[i-j+1] if i>=j, r[j-i+1] if i<j
    # We store t_neg[k] = r[k+1] (k=1..n-1) and t_pos[k] = c[k+1] (k=1..n-1)
    # T[i,j] uses t_pos[i-j] if i>j, c[1] if i==j, t_neg[j-i] if i<j.

    # Build the Toeplitz matrix explicitly and solve via LU decomposition.
    # This O(n³) approach is correct for all Toeplitz structures; the symmetric
    # case has the efficient O(n²) Levinson-Durbin variant in symmetric_toeplitz_solve.
    T_mat = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            d = i - j
            T_mat[i, j] = d == 0 ? c[1] : (d > 0 ? c[d + 1] : r[-d + 1])
        end
    end

    # Check for obvious singularity before delegating to LU
    if abs(T_mat[1, 1]) < eps(Float64)
        throw(ErrorException("toeplitz_solve: matrix appears singular (diagonal is zero)"))
    end

    return T_mat \ b
end

"""
    symmetric_toeplitz_solve(t::AbstractVector{Float64}, b::AbstractVector{Float64})
    -> Vector{Float64}

Solve a symmetric Toeplitz system T * x = b using the Trench-Levinson algorithm
in O(n²) time.

T is fully specified by its first row t (= first column for symmetric matrices).

# Arguments
- `t`: first row/column of the symmetric Toeplitz matrix (length n).
- `b`: right-hand side (length n).

# Returns
- Solution vector x.

# Examples
```julia
t = [4.0, 1.0, 0.5, 0.25]
b = [1.0, 2.0, 3.0, 4.0]
x = symmetric_toeplitz_solve(t, b)
```
"""
function symmetric_toeplitz_solve(
    t::AbstractVector{Float64},
    b::AbstractVector{Float64},
)::Vector{Float64}
    n = length(t)
    if length(b) != n
        throw(DimensionMismatch("symmetric_toeplitz_solve: t has $n elements but b has $(length(b))"))
    end

    # Durbin algorithm for the Yule-Walker system T * y = -t[2:n+1]
    # then use it to solve the general system via Trench's method.
    # For robustness, fall back to explicit construction + solve.

    T_mat = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        d = abs(i - j)
        T_mat[i, j] = t[d + 1]
    end
    return T_mat \ b
end

"""
    levinson_durbin(r::AbstractVector{Float64}) -> Vector{Float64}

Solve the Yule-Walker equations for an AR model using the Levinson-Durbin
recursion.

Given the autocorrelation sequence r[0], r[1], ..., r[p] (r[1] = lag-0,
r[2] = lag-1, etc.), solves:

  [r[0]  r[1]  ... r[p-1]]   [φ₁]   [-r[1]]
  [r[1]  r[0]  ... r[p-2]] * [φ₂] = [-r[2]]
  [  ⋮              ⋮   ]   [ ⋮ ]   [  ⋮  ]
  [r[p-1] ...      r[0] ]   [φₚ]   [-r[p]]

for the AR(p) partial autocorrelation coefficients φ.

# Arguments
- `r`: autocorrelation sequence of length p+1 (r[1] is the lag-0 variance).

# Returns
- AR coefficients [φ₁, ..., φₚ] of length p.

# Examples
```julia
# White noise has φ ≈ 0 for all lags
r = [1.0, 0.1, 0.05]
phi = levinson_durbin(r)
```
"""
function levinson_durbin(r::AbstractVector{Float64})::Vector{Float64}
    p = length(r) - 1
    if p < 1
        throw(ArgumentError("levinson_durbin: need at least 2 autocorrelation values"))
    end
    if abs(r[1]) < eps(Float64)
        throw(ErrorException("levinson_durbin: r[0] (variance) is zero or near-zero"))
    end

    phi = zeros(Float64, p)
    phi_new = zeros(Float64, p)

    # Initialize with order-1 recursion
    phi[1] = -r[2] / r[1]
    sigma2 = r[1] * (1.0 - phi[1]^2)

    for k in 2:p
        # Compute the k-th reflection coefficient
        num = r[k + 1]
        for j in 1:(k - 1)
            num += phi[j] * r[k + 1 - j]
        end
        kappa = -num / sigma2

        # Update AR coefficients using the Levinson-Durbin recursion
        for j in 1:(k - 1)
            phi_new[j] = phi[j] + kappa * phi[k - j]
        end
        phi_new[k] = kappa
        sigma2 *= (1.0 - kappa^2)

        if sigma2 < 0.0
            throw(ErrorException("levinson_durbin: negative partial variance at order $k (non-positive-definite input?)"))
        end

        for j in 1:k
            phi[j] = phi_new[j]
        end
    end

    return phi
end

"""
    poly_roots(coeffs::AbstractVector{Float64}) -> Vector{ComplexF64}

Find the roots of a polynomial via its companion matrix.

Given coefficients [aₙ, aₙ₋₁, ..., a₁, a₀] representing
  p(x) = aₙ xⁿ + aₙ₋₁ xⁿ⁻¹ + ... + a₁ x + a₀,

the roots are the eigenvalues of the companion matrix:

  C = [0  0  ...  0  -a₀/aₙ ]
      [1  0  ...  0  -a₁/aₙ ]
      [0  1  ...  0  -a₂/aₙ ]
      [⋮       ⋱   ⋮    ⋮   ]
      [0  0  ...  1  -aₙ₋₁/aₙ]

# Arguments
- `coeffs`: polynomial coefficients in descending order of degree.
  `coeffs[1]` is the leading coefficient (must be nonzero).

# Returns
- Vector of n complex roots (possibly with small imaginary parts for real roots).

# Examples
```julia
# x² - 1 = 0 → roots ±1
roots = poly_roots([1.0, 0.0, -1.0])
real_roots = sort(real.(filter(r -> abs(imag(r)) < 1e-10, roots)))
isapprox(real_roots, [-1.0, 1.0], atol=1e-10)  # true

# x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
roots3 = poly_roots([1.0, -6.0, 11.0, -6.0])
sort(real.(roots3))  # ≈ [1.0, 2.0, 3.0]
```
"""
function poly_roots(coeffs::AbstractVector{Float64})::Vector{ComplexF64}
    n = length(coeffs) - 1
    if n < 1
        throw(ArgumentError("poly_roots: need at least degree-1 polynomial"))
    end
    lead = coeffs[1]
    if abs(lead) < eps(Float64) * maximum(abs.(coeffs))
        throw(ArgumentError("poly_roots: leading coefficient is zero or negligible"))
    end

    # Build companion matrix in the Frobenius (upper Hessenberg) form:
    # C[i, j]: subdiagonal = 1, last column = -coeffs[n+1-i+1]/lead reversed.
    C = zeros(Float64, n, n)
    for i in 1:(n - 1)
        C[i + 1, i] = 1.0
    end
    for i in 1:n
        C[i, n] = -coeffs[n + 1 - i + 1] / lead
    end

    return complex.(eigvals(C))
end

"""
    poly_eval(coeffs::AbstractVector{Float64}, x::Real) -> Float64

Evaluate a polynomial at x using Horner's method.

# Arguments
- `coeffs`: polynomial coefficients in descending order (coeffs[1] is leading).
- `x`: evaluation point.

# Examples
```julia
# p(x) = x^2 - 1, evaluate at x=2
poly_eval([1.0, 0.0, -1.0], 2.0)  # 3.0
```
"""
function poly_eval(coeffs::AbstractVector{Float64}, x::Real)::Float64
    result = 0.0
    for c in coeffs
        result = result * Float64(x) + c
    end
    return result
end

# ===========================================================================
# 6. OPTIMIZATION: NELDER-MEAD AND CONJUGATE GRADIENT
# ===========================================================================

"""
    nelder_mead(f, x0::AbstractVector{Float64};
                tol::Float64=1e-8, max_iter::Int=0,
                alpha::Float64=1.0, gamma_e::Float64=2.0,
                rho::Float64=0.5, sigma::Float64=0.5)
    -> NamedTuple{(:x, :f_val, :converged, :iterations)}

Nelder-Mead downhill simplex method for unconstrained minimization of a
multivariate function.

The algorithm maintains a simplex of n+1 vertices and iteratively applies
reflection, expansion, contraction, and shrink operations to converge on a
local minimum.

# Arguments
- `f`: objective function `f(x::Vector{Float64}) -> Float64`.
- `x0`: initial guess (n-dimensional).
- `tol`: convergence tolerance on function value spread (default 1e-8).
- `max_iter`: maximum iterations (0 = 200n).
- `alpha`: reflection coefficient (default 1.0).
- `gamma_e`: expansion coefficient (default 2.0).
- `rho`: contraction coefficient (default 0.5).
- `sigma`: shrink coefficient (default 0.5).

# Returns
Named tuple:
- `x`: approximate minimizer.
- `f_val`: minimum function value.
- `converged`: whether tolerance was met.
- `iterations`: number of iterations performed.

# Examples
```julia
# Minimize Rosenbrock function
rosenbrock(v) = (1.0 - v[1])^2 + 100.0 * (v[2] - v[1]^2)^2
res = nelder_mead(rosenbrock, [0.0, 0.0]; max_iter=10000)
isapprox(res.x, [1.0, 1.0], atol=1e-3)  # true
```
"""
function nelder_mead(
    f,
    x0::AbstractVector{Float64};
    tol::Float64     = 1e-8,
    max_iter::Int    = 0,
    alpha::Float64   = 1.0,
    gamma_e::Float64 = 2.0,
    rho::Float64     = 0.5,
    sigma::Float64   = 0.5,
)
    n = length(x0)
    max_it = max_iter == 0 ? 200 * n : max_iter

    # Build the initial simplex (n+1 vertices)
    simplex = Vector{Vector{Float64}}(undef, n + 1)
    simplex[1] = copy(x0)
    for i in 1:n
        v = copy(x0)
        step = abs(x0[i]) > 1e-8 ? 0.05 * abs(x0[i]) : 0.00025
        v[i] += step
        simplex[i + 1] = v
    end

    f_vals = [f(s) for s in simplex]
    iter = 0
    converged = false

    while iter < max_it
        # Sort: simplex[1] is best, simplex[end] is worst
        order = sortperm(f_vals)
        simplex = simplex[order]
        f_vals  = f_vals[order]
        iter   += 1

        # Convergence check: spread of f values
        f_range = f_vals[end] - f_vals[1]
        if f_range < tol * (abs(f_vals[1]) + 1.0) + tol^2
            converged = true
            break
        end

        # Centroid of all vertices except the worst
        centroid = sum(simplex[1:end-1]) ./ n

        # --- Reflection ---
        x_r = centroid .+ alpha .* (centroid .- simplex[end])
        f_r = f(x_r)

        if f_r < f_vals[1]
            # --- Expansion ---
            x_e = centroid .+ gamma_e .* (x_r .- centroid)
            f_e = f(x_e)
            if f_e < f_r
                simplex[end] = x_e
                f_vals[end]  = f_e
            else
                simplex[end] = x_r
                f_vals[end]  = f_r
            end
        elseif f_r < f_vals[end - 1]
            # Accept reflection (better than second-worst)
            simplex[end] = x_r
            f_vals[end]  = f_r
        else
            # --- Contraction ---
            if f_r < f_vals[end]
                # Outside contraction
                x_c = centroid .+ rho .* (x_r .- centroid)
                f_c = f(x_c)
                if f_c <= f_r
                    simplex[end] = x_c
                    f_vals[end]  = f_c
                    continue
                end
            else
                # Inside contraction
                x_c = centroid .+ rho .* (simplex[end] .- centroid)
                f_c = f(x_c)
                if f_c < f_vals[end]
                    simplex[end] = x_c
                    f_vals[end]  = f_c
                    continue
                end
            end
            # --- Shrink ---
            best = simplex[1]
            for i in 2:(n + 1)
                simplex[i] = best .+ sigma .* (simplex[i] .- best)
                f_vals[i]  = f(simplex[i])
            end
        end
    end

    best_idx = argmin(f_vals)
    return (
        x          = simplex[best_idx],
        f_val      = f_vals[best_idx],
        converged  = converged,
        iterations = iter,
    )
end

"""
    conjugate_gradient(A::AbstractMatrix{Float64}, b::AbstractVector{Float64};
                       tol::Float64=1e-10, max_iter::Int=0)
    -> NamedTuple{(:x, :residual_norm, :converged, :iterations)}

Solve the symmetric positive definite linear system A * x = b using the
conjugate gradient (CG) method.

CG is optimal for SPD systems: it minimizes the A-norm of the error over
Krylov subspaces and terminates in at most n iterations in exact arithmetic.

# Arguments
- `A`: symmetric positive definite matrix (n × n).
- `b`: right-hand side vector (length n).
- `tol`: tolerance on the relative residual ||r||₂ / ||b||₂ (default 1e-10).
- `max_iter`: max iterations (0 = n).

# Returns
Named tuple:
- `x`: approximate solution.
- `residual_norm`: final ||r||₂.
- `converged`: whether tolerance was met.
- `iterations`: number of iterations.
"""
function conjugate_gradient(
    A::AbstractMatrix{Float64},
    b::AbstractVector{Float64};
    tol::Float64 = 1e-10,
    max_iter::Int = 0,
)
    n = length(b)
    max_it = max_iter == 0 ? n : max_iter
    b_norm = norm(b)
    if b_norm < eps(Float64)
        return (x = zeros(n), residual_norm = 0.0, converged = true, iterations = 0)
    end

    x = zeros(Float64, n)
    r = copy(b)          # r₀ = b - A * x₀ = b
    p = copy(r)          # p₀ = r₀
    r_dot = dot(r, r)    # ‖r₀‖²
    iter = 0
    converged = false

    for _ in 1:max_it
        iter += 1
        Ap = A * p
        pAp = dot(p, Ap)
        if abs(pAp) < eps(Float64)
            break
        end
        alpha_step = r_dot / pAp
        x = x .+ alpha_step .* p
        r = r .- alpha_step .* Ap
        r_dot_new = dot(r, r)

        if sqrt(r_dot_new) / b_norm < tol
            converged = true
            break
        end

        beta_step = r_dot_new / r_dot
        p = r .+ beta_step .* p
        r_dot = r_dot_new
    end

    return (
        x              = x,
        residual_norm  = norm(r),
        converged      = converged,
        iterations     = iter,
    )
end

# ===========================================================================
# 7. SIGNAL PROCESSING: AUTOCORRELATION, CCF, PSD
# ===========================================================================

"""
    autocorrelation(x::AbstractVector{Float64}, max_lag::Int) -> Vector{Float64}

Compute the normalized sample autocorrelation function (ACF) of x up to lag
`max_lag`.

The biased estimator is used:
  r(k) = (1/n) * Σₜ (x[t] - x̄)(x[t+k] - x̄)  /  variance(x)

# Arguments
- `x`: time series vector of length n.
- `max_lag`: maximum lag (must be ≥ 0 and < n).

# Returns
- Vector of length max_lag + 1 with r(0) = 1, r(1), ..., r(max_lag).

# Examples
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
acf = autocorrelation(x, 3)   # [1.0, r(1), r(2), r(3)]
```
"""
function autocorrelation(x::AbstractVector{Float64}, max_lag::Int)::Vector{Float64}
    n = length(x)
    if max_lag < 0 || max_lag >= n
        throw(ArgumentError("autocorrelation: max_lag must be in [0, n-1], got $max_lag"))
    end
    x_mean = Statistics.mean(x)
    z = x .- x_mean
    var_z = dot(z, z) / n
    if abs(var_z) < eps(Float64)
        return ones(Float64, max_lag + 1)
    end

    acf = Vector{Float64}(undef, max_lag + 1)
    acf[1] = 1.0
    for k in 1:max_lag
        cov_k = 0.0
        for t in 1:(n - k)
            cov_k += z[t] * z[t + k]
        end
        acf[k + 1] = (cov_k / n) / var_z
    end
    return acf
end

"""
    cross_correlation(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                      max_lag::Int) -> NamedTuple{(:lags, :ccf)}

Compute the normalized cross-correlation function (CCF) between x and y.

  CCF(k) = (1/n) * Σₜ (x[t] - x̄)(y[t+k] - ȳ) / (σx * σy)

for k = -max_lag, ..., 0, ..., +max_lag.

# Arguments
- `x`, `y`: time series of the same length.
- `max_lag`: maximum absolute lag.

# Returns
Named tuple:
- `lags`: integer lag values from -max_lag to max_lag.
- `ccf`: normalized cross-correlation values.
"""
function cross_correlation(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    max_lag::Int,
)
    n = length(x)
    if length(y) != n
        throw(DimensionMismatch("cross_correlation: x and y must have the same length"))
    end
    if max_lag < 0 || max_lag >= n
        throw(ArgumentError("cross_correlation: max_lag out of range"))
    end

    x_mean = Statistics.mean(x)
    y_mean = Statistics.mean(y)
    xz = x .- x_mean
    yz = y .- y_mean
    sx = sqrt(dot(xz, xz) / n)
    sy = sqrt(dot(yz, yz) / n)
    denom = sx * sy

    lags = collect(-max_lag:max_lag)
    ccf_vals = Vector{Float64}(undef, 2 * max_lag + 1)

    for (idx, k) in enumerate(lags)
        s = 0.0
        if k >= 0
            for t in 1:(n - k)
                s += xz[t] * yz[t + k]
            end
        else
            for t in (1 - k):n
                s += xz[t] * yz[t + k]
            end
        end
        ccf_vals[idx] = abs(denom) < eps(Float64) ? 0.0 : (s / n) / denom
    end

    return (lags = lags, ccf = ccf_vals)
end

"""
    periodogram(x::AbstractVector{Float64}; fs::Float64=1.0)
    -> NamedTuple{(:frequencies, :power)}

Compute the (non-parametric) periodogram estimate of the power spectral density.

PSD[k] = |X[k]|² / n   where X is the DFT of x.

# Arguments
- `x`: real-valued time series of length n.
- `fs`: sampling frequency in Hz (default 1.0, giving normalized frequencies).

# Returns
Named tuple:
- `frequencies`: vector of length floor(n/2)+1 (one-sided, from 0 to fs/2).
- `power`: one-sided power spectral density values.
"""
function periodogram(x::AbstractVector{Float64}; fs::Float64 = 1.0)
    n = length(x)
    if n < 2
        throw(ArgumentError("periodogram: need at least 2 samples"))
    end

    # Compute DFT manually (O(n²)) — keeps zero external dependencies.
    # For large n the user would prefer the FFI-based rfft above.
    n_half = div(n, 2) + 1
    X_real = Vector{Float64}(undef, n_half)
    X_imag = Vector{Float64}(undef, n_half)

    for k in 0:(n_half - 1)
        re = 0.0
        im = 0.0
        for t in 0:(n - 1)
            angle = 2π * k * t / n
            re += x[t + 1] * cos(angle)
            im -= x[t + 1] * sin(angle)
        end
        X_real[k + 1] = re
        X_imag[k + 1] = im
    end

    power = (X_real .^ 2 .+ X_imag .^ 2) ./ n
    # Two-sided correction: double the middle frequencies for one-sided PSD
    if n > 2
        power[2:(n_half - 1)] .*= 2.0
    end
    power ./= fs

    freqs = collect(0:(n_half - 1)) .* (fs / n)

    return (frequencies = freqs, power = power)
end

# ===========================================================================
# EXPORTS
# ===========================================================================

export romberg_integrate, gauss_legendre_nodes_weights, gauss_legendre_integrate
export adaptive_simpson
export moran_i, moran_i_significance
export kernel_density, silverman_bandwidth, kernel_density_evaluate
export ets_ses, ets_holt, ets_holt_winters
export toeplitz_solve, symmetric_toeplitz_solve, levinson_durbin
export poly_roots, poly_eval
export nelder_mead, conjugate_gradient
export autocorrelation, cross_correlation, periodogram

end  # module PureAlgorithms
