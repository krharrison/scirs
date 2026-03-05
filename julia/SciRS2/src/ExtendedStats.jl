"""
    ExtendedStats

Extended statistical functions for SciRS2.

Provides:
- Higher moments: skewness, excess kurtosis
- Covariance and correlation matrices
- Histogram computation with various binning strategies
- z-score standardization and normalization
- Running (streaming) statistics
- Quantile functions and descriptive summaries
- Weighted statistics
- Bootstrap confidence intervals
"""
module ExtendedStats

using LinearAlgebra
using Statistics

# ===========================================================================
# 1. HIGHER-ORDER MOMENTS
# ===========================================================================

"""
    skewness(x::AbstractVector{Float64}; bias::Bool=true) -> Float64

Compute the skewness (third standardized moment) of a sample.

For `bias=true` (biased estimator, default):
  skewness = (1/n) Σ ((x[i] - μ) / σ)³

For `bias=false` (Fisher's adjusted estimator, matches SciPy default):
  skewness_adj = skewness * √(n*(n-1)) / (n-2)

# Arguments
- `x`: vector with at least 3 elements.
- `bias`: if `false`, apply small-sample correction (default `true`).

# Returns
- Skewness value.

# Examples
```julia
x = [1.0, 2.0, 3.0, 4.0, 100.0]
skewness(x)               # positively skewed
skewness(x; bias=false)   # adjusted estimate
```
"""
function skewness(x::AbstractVector{Float64}; bias::Bool=true)::Float64
    n = length(x)
    if n < 3
        throw(ArgumentError("skewness: need at least 3 elements, got $n"))
    end
    mu = Statistics.mean(x)
    sigma = Statistics.std(x; corrected=false)
    if sigma < eps(Float64)
        throw(ArgumentError("skewness: zero variance, skewness is undefined"))
    end
    s = sum((xi - mu)^3 for xi in x) / (n * sigma^3)
    if !bias
        s = s * sqrt(n * (n - 1)) / (n - 2)
    end
    return s
end

"""
    kurtosis(x::AbstractVector{Float64}; excess::Bool=true, bias::Bool=true) -> Float64

Compute the kurtosis (fourth standardized moment) of a sample.

For `excess=true` (default), returns excess kurtosis (kurtosis - 3), so that
a normal distribution has kurtosis = 0.

For `bias=false`, applies Fisher's small-sample correction.

# Arguments
- `x`: vector with at least 4 elements.
- `excess`: if `true` (default), subtract 3 (excess kurtosis).
- `bias`: if `false`, apply small-sample correction.

# Returns
- (Excess) kurtosis value.

# Examples
```julia
# Normal distribution: excess kurtosis ≈ 0
using Random; x = randn(10000)
kurtosis(x)   # ≈ 0
```
"""
function kurtosis(
    x::AbstractVector{Float64};
    excess::Bool=true,
    bias::Bool=true,
)::Float64
    n = length(x)
    if n < 4
        throw(ArgumentError("kurtosis: need at least 4 elements, got $n"))
    end
    mu = Statistics.mean(x)
    sigma = Statistics.std(x; corrected=false)
    if sigma < eps(Float64)
        throw(ArgumentError("kurtosis: zero variance, kurtosis is undefined"))
    end
    kurt = sum((xi - mu)^4 for xi in x) / (n * sigma^4)
    if !bias
        kurt = ((n + 1) * kurt - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3
    end
    return excess ? kurt - 3.0 : kurt
end

"""
    moments(x::AbstractVector{Float64}; max_order::Int=4)
    -> NamedTuple{(:mean, :variance, :skewness, :kurtosis)}

Compute the first four sample moments in one pass.

# Arguments
- `x`: sample vector with at least 4 elements.
- `max_order`: maximum moment order to compute (2, 3, or 4; default 4).

# Returns
Named tuple with fields `mean`, `variance`, `skewness`, `kurtosis`
(excess kurtosis), or a subset depending on `max_order`.
"""
function moments(x::AbstractVector{Float64}; max_order::Int=4)
    n = length(x)
    if n < max(2, max_order)
        throw(ArgumentError("moments: need at least $max_order elements"))
    end

    mu = Statistics.mean(x)
    m2 = sum((xi - mu)^2 for xi in x) / n
    var = m2 * n / (n - 1)  # sample variance (ddof=1)

    if max_order < 3
        return (mean=mu, variance=var)
    end

    sigma = sqrt(m2)
    if sigma < eps(Float64)
        return (mean=mu, variance=0.0, skewness=0.0, kurtosis=0.0)
    end

    m3 = sum((xi - mu)^3 for xi in x) / (n * sigma^3)
    if max_order < 4
        return (mean=mu, variance=var, skewness=m3)
    end

    m4 = sum((xi - mu)^4 for xi in x) / (n * sigma^4) - 3.0
    return (mean=mu, variance=var, skewness=m3, kurtosis=m4)
end

# ===========================================================================
# 2. COVARIANCE AND CORRELATION MATRICES
# ===========================================================================

"""
    covariance_matrix(X::AbstractMatrix{Float64}; ddof::Int=1) -> Matrix{Float64}

Compute the sample covariance matrix of a dataset.

# Arguments
- `X`: (n × p) data matrix with n observations and p variables.
- `ddof`: delta degrees of freedom (default 1 = sample covariance).

# Returns
- (p × p) symmetric positive semi-definite covariance matrix.

# Examples
```julia
X = [1.0 2.0; 3.0 4.0; 5.0 6.0]   # 3 observations, 2 variables
C = covariance_matrix(X)             # 2x2 covariance matrix
```
"""
function covariance_matrix(X::AbstractMatrix{Float64}; ddof::Int=1)::Matrix{Float64}
    n, p = size(X)
    if n <= ddof
        throw(ArgumentError("covariance_matrix: need more than ddof=$ddof observations"))
    end
    mu = vec(Statistics.mean(X; dims=1))
    Xc = X .- mu'
    return (Xc' * Xc) / (n - ddof)
end

"""
    correlation_matrix(X::AbstractMatrix{Float64}) -> Matrix{Float64}

Compute the Pearson correlation matrix of a dataset.

# Arguments
- `X`: (n × p) data matrix.

# Returns
- (p × p) correlation matrix with ones on the diagonal.

# Examples
```julia
X = [1.0 2.0; 3.0 6.0; 5.0 10.0]
R = correlation_matrix(X)   # 2x2 with R[1,2] = R[2,1] = 1.0 (perfect corr.)
```
"""
function correlation_matrix(X::AbstractMatrix{Float64})::Matrix{Float64}
    C = covariance_matrix(X; ddof=1)
    p = size(C, 1)
    D = sqrt.(max.(0.0, diag(C)))
    R = similar(C)
    for i in 1:p, j in 1:p
        denom = D[i] * D[j]
        R[i, j] = denom < eps(Float64) ? (i == j ? 1.0 : 0.0) : C[i, j] / denom
    end
    return R
end

# ===========================================================================
# 3. HISTOGRAM
# ===========================================================================

"""
    histogram(x::AbstractVector{Float64}; bins::Union{Int,AbstractVector{Float64}}=10,
              density::Bool=false, range::Tuple{Float64,Float64}=(-Inf, Inf))
    -> NamedTuple{(:counts, :edges)}

Compute a histogram of the data.

# Arguments
- `x`: data vector.
- `bins`: number of bins (Int) or explicit bin edges (Vector). Default 10.
- `density`: if `true`, normalize so that the total area sums to 1.
- `range`: (min, max) tuple to clip data range; defaults to (minimum(x), maximum(x)).

# Returns
Named tuple:
- `counts`: bin counts (or densities if `density=true`); length = number of bins.
- `edges`: bin edge positions; length = number of bins + 1.

# Examples
```julia
x = randn(1000)
h = histogram(x; bins=20, density=true)
# h.edges[i] to h.edges[i+1] is the i-th bin; h.counts[i] is its value.
```
"""
function histogram(
    x::AbstractVector{Float64};
    bins::Union{Int,AbstractVector{Float64}}=10,
    density::Bool=false,
    range::Tuple{Float64,Float64}=(-Inf, Inf),
)
    xv = collect(Float64, x)
    lo = isinf(range[1]) ? minimum(xv) : range[1]
    hi = isinf(range[2]) ? maximum(xv) : range[2]

    if lo >= hi
        throw(ArgumentError("histogram: invalid range [$lo, $hi]"))
    end

    # Compute bin edges
    edges = if isa(bins, Int)
        if bins < 1
            throw(ArgumentError("histogram: bins must be ≥ 1, got $bins"))
        end
        collect(range_vec(lo, hi, bins + 1))
    else
        collect(Float64, bins)
    end

    n_bins = length(edges) - 1
    counts = zeros(Float64, n_bins)

    for v in xv
        if v < lo || v > hi
            continue
        end
        # Binary search for the bin
        i = searchsortedlast(edges, v)
        # Last point goes in last bin
        i = clamp(i, 1, n_bins)
        counts[i] += 1.0
    end

    if density
        bin_widths = diff(edges)
        total_area = sum(counts .* bin_widths)
        if total_area > eps(Float64)
            counts ./= total_area
        end
    end

    return (counts=counts, edges=edges)
end

"""Generate n+1 linearly spaced values from a to b."""
function range_vec(a::Float64, b::Float64, n::Int)::Vector{Float64}
    return collect(range(a, stop=b, length=n))
end

# ===========================================================================
# 4. NORMALIZATION AND STANDARDIZATION
# ===========================================================================

"""
    zscore(x::AbstractVector{Float64}; ddof::Int=0) -> Vector{Float64}

Standardize a vector to have zero mean and unit variance (z-score normalization).

  z[i] = (x[i] - μ) / σ

# Arguments
- `x`: input vector.
- `ddof`: delta degrees of freedom for σ computation (default 0 = population std).

# Returns
- z-scores.

# Throws
- `ArgumentError` if the standard deviation is zero.
"""
function zscore(x::AbstractVector{Float64}; ddof::Int=0)::Vector{Float64}
    mu = Statistics.mean(x)
    sigma = Statistics.std(x; corrected=(ddof == 1))
    if sigma < eps(Float64)
        throw(ArgumentError("zscore: zero standard deviation"))
    end
    return (x .- mu) ./ sigma
end

"""
    minmax_scale(x::AbstractVector{Float64};
                 feature_range::Tuple{Float64,Float64}=(0.0, 1.0)) -> Vector{Float64}

Scale a vector to a specified range (default [0, 1]).

  x_scaled[i] = (x[i] - min(x)) / (max(x) - min(x)) * (hi - lo) + lo

# Arguments
- `x`: input vector.
- `feature_range`: target range (lo, hi) (default (0.0, 1.0)).

# Returns
- Scaled vector.

# Throws
- `ArgumentError` if all values are equal (range is zero).
"""
function minmax_scale(
    x::AbstractVector{Float64};
    feature_range::Tuple{Float64,Float64}=(0.0, 1.0),
)::Vector{Float64}
    lo, hi = feature_range
    if lo >= hi
        throw(ArgumentError("minmax_scale: feature_range must have lo < hi"))
    end
    xmin = minimum(x)
    xmax = maximum(x)
    span = xmax - xmin
    if span < eps(Float64)
        throw(ArgumentError("minmax_scale: all values are equal (range is zero)"))
    end
    return (x .- xmin) ./ span .* (hi - lo) .+ lo
end

"""
    robust_scale(x::AbstractVector{Float64}; quantile_range::Tuple{Float64,Float64}=(25.0, 75.0))
    -> Vector{Float64}

Scale a vector using the inter-quartile range (IQR), robust to outliers.

  x_scaled[i] = (x[i] - median(x)) / IQR(x)

# Arguments
- `x`: input vector.
- `quantile_range`: (lo_percentile, hi_percentile) defining the IQR (default (25.0, 75.0)).

# Returns
- Robustly scaled vector.
"""
function robust_scale(
    x::AbstractVector{Float64};
    quantile_range::Tuple{Float64,Float64}=(25.0, 75.0),
)::Vector{Float64}
    sorted = sort(x)
    n = length(sorted)
    med = Statistics.median(x)

    function quantile_val(p::Float64)
        idx = (p / 100.0) * (n - 1)
        lo = floor(Int, idx)
        frac = idx - lo
        if lo + 1 >= n
            return sorted[n]
        end
        return sorted[lo + 1] + frac * (sorted[lo + 2] - sorted[lo + 1])
    end

    q_lo = quantile_val(quantile_range[1])
    q_hi = quantile_val(quantile_range[2])
    iqr = q_hi - q_lo

    if iqr < eps(Float64)
        throw(ArgumentError("robust_scale: IQR is zero, cannot scale"))
    end

    return (x .- med) ./ iqr
end

# ===========================================================================
# 5. WEIGHTED STATISTICS
# ===========================================================================

"""
    weighted_mean(x::AbstractVector{Float64}, w::AbstractVector{Float64}) -> Float64

Compute the weighted arithmetic mean.

  μ_w = Σ w[i] * x[i] / Σ w[i]

# Arguments
- `x`: data values.
- `w`: non-negative weights (must have the same length as `x`).

# Returns
- Weighted mean.
"""
function weighted_mean(
    x::AbstractVector{Float64},
    w::AbstractVector{Float64},
)::Float64
    if length(x) != length(w)
        throw(DimensionMismatch("weighted_mean: x and w must have the same length"))
    end
    total_w = sum(w)
    if total_w < eps(Float64)
        throw(ArgumentError("weighted_mean: sum of weights is zero"))
    end
    return sum(w .* x) / total_w
end

"""
    weighted_variance(x::AbstractVector{Float64}, w::AbstractVector{Float64};
                      ddof::Float64=0.0) -> Float64

Compute the weighted sample variance.

# Arguments
- `x`: data values.
- `w`: non-negative weights.
- `ddof`: delta degrees of freedom (default 0.0 for frequency weights).

# Returns
- Weighted variance.
"""
function weighted_variance(
    x::AbstractVector{Float64},
    w::AbstractVector{Float64};
    ddof::Float64=0.0,
)::Float64
    mu = weighted_mean(x, w)
    total_w = sum(w)
    var = sum(w .* (x .- mu).^2) / (total_w - ddof)
    return var
end

"""
    weighted_std(x::AbstractVector{Float64}, w::AbstractVector{Float64};
                 ddof::Float64=0.0) -> Float64

Compute the weighted standard deviation.
"""
function weighted_std(
    x::AbstractVector{Float64},
    w::AbstractVector{Float64};
    ddof::Float64=0.0,
)::Float64
    return sqrt(weighted_variance(x, w; ddof=ddof))
end

# ===========================================================================
# 6. RUNNING (STREAMING) STATISTICS
# ===========================================================================

"""
    RunningStats

Online (streaming) statistics using Welford's algorithm.

Maintains running mean, variance, min, and max in O(1) memory.

# Usage
```julia
rs = RunningStats()
for x in data
    update!(rs, x)
end
stats = finalize(rs)
```
"""
mutable struct RunningStats
    count::Int
    mean::Float64
    m2::Float64    # sum of squared deviations (for variance)
    minimum::Float64
    maximum::Float64

    RunningStats() = new(0, 0.0, 0.0, Inf, -Inf)
end

"""
    update!(rs::RunningStats, x::Float64)

Update running statistics with a new observation.
"""
function update!(rs::RunningStats, x::Float64)
    rs.count += 1
    delta = x - rs.mean
    rs.mean += delta / rs.count
    rs.m2 += delta * (x - rs.mean)
    rs.minimum = min(rs.minimum, x)
    rs.maximum = max(rs.maximum, x)
end

"""
    finalize(rs::RunningStats) -> NamedTuple

Extract final statistics from a `RunningStats` accumulator.

# Returns
Named tuple with:
- `count`: number of observations.
- `mean`: sample mean.
- `variance`: sample variance (ddof=1).
- `std`: sample standard deviation.
- `minimum`: minimum value seen.
- `maximum`: maximum value seen.
"""
function finalize(rs::RunningStats)
    if rs.count == 0
        throw(ArgumentError("finalize: no observations added to RunningStats"))
    end
    var = rs.count > 1 ? rs.m2 / (rs.count - 1) : 0.0
    return (
        count=rs.count,
        mean=rs.mean,
        variance=var,
        std=sqrt(max(0.0, var)),
        minimum=rs.minimum,
        maximum=rs.maximum,
    )
end

# ===========================================================================
# 7. DESCRIPTIVE SUMMARY STATISTICS
# ===========================================================================

"""
    describe(x::AbstractVector{Float64}) -> NamedTuple

Compute a comprehensive descriptive statistics summary.

# Returns
Named tuple with:
- `count`: number of observations.
- `mean`: arithmetic mean.
- `std`: sample standard deviation (ddof=1).
- `min`: minimum.
- `q25`: 25th percentile.
- `median`: 50th percentile (median).
- `q75`: 75th percentile.
- `max`: maximum.
- `iqr`: inter-quartile range (q75 - q25).
- `skewness`: Fisher's adjusted skewness.
- `kurtosis`: excess kurtosis.

# Examples
```julia
x = randn(100)
d = describe(x)
println("Mean: \$(d.mean), Std: \$(d.std)")
```
"""
function describe(x::AbstractVector{Float64})
    n = length(x)
    if n < 4
        throw(ArgumentError("describe: need at least 4 elements for full summary"))
    end

    sorted = sort(x)

    function pctile(p)
        idx = (p / 100.0) * (n - 1)
        lo = floor(Int, idx)
        frac = idx - lo
        lo_idx = clamp(lo + 1, 1, n)
        hi_idx = clamp(lo + 2, 1, n)
        return sorted[lo_idx] + frac * (sorted[hi_idx] - sorted[lo_idx])
    end

    q25 = pctile(25.0)
    med = pctile(50.0)
    q75 = pctile(75.0)
    mu  = Statistics.mean(x)
    sd  = Statistics.std(x; corrected=true)
    sk  = length(x) >= 3 ? skewness(x; bias=false) : NaN
    ku  = length(x) >= 4 ? kurtosis(x; excess=true, bias=false) : NaN

    return (
        count=n,
        mean=mu,
        std=sd,
        min=sorted[1],
        q25=q25,
        median=med,
        q75=q75,
        max=sorted[n],
        iqr=q75 - q25,
        skewness=sk,
        kurtosis=ku,
    )
end

# ===========================================================================
# 8. BOOTSTRAP CONFIDENCE INTERVALS
# ===========================================================================

"""
    bootstrap_ci(x::AbstractVector{Float64}, statistic::Function;
                 n_boot::Int=1000, confidence::Float64=0.95,
                 seed::UInt64=12345) -> Tuple{Float64, Float64}

Compute a bootstrap confidence interval for a statistic.

Uses the percentile bootstrap method: the interval is the
[α/2, 1-α/2] quantiles of the bootstrap distribution.

# Arguments
- `x`: observed data.
- `statistic`: function that maps a vector to a scalar (e.g., `mean`, `median`).
- `n_boot`: number of bootstrap resamples (default 1000).
- `confidence`: confidence level (default 0.95 → 95% CI).
- `seed`: random seed for reproducibility (LCG-based, no external deps).

# Returns
- `(lo, hi)` tuple for the confidence interval.

# Examples
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
lo, hi = bootstrap_ci(x, Statistics.mean; n_boot=2000)
```
"""
function bootstrap_ci(
    x::AbstractVector{Float64},
    statistic::Function;
    n_boot::Int=1000,
    confidence::Float64=0.95,
    seed::UInt64=12345,
)::Tuple{Float64,Float64}
    n = length(x)
    if n < 2
        throw(ArgumentError("bootstrap_ci: need at least 2 observations"))
    end
    if confidence <= 0.0 || confidence >= 1.0
        throw(ArgumentError("bootstrap_ci: confidence must be in (0, 1)"))
    end

    # LCG random number generator (no external dependencies)
    state = Ref{UInt64}(seed)
    function lcg_rand_int(maxval::Int)::Int
        state[] = state[] * 6364136223846793005 + 1442695040888963407
        return Int(state[] >> 33) % maxval + 1
    end

    boot_stats = Vector{Float64}(undef, n_boot)
    sample_buf = Vector{Float64}(undef, n)

    for b in 1:n_boot
        for i in 1:n
            sample_buf[i] = x[lcg_rand_int(n)]
        end
        boot_stats[b] = statistic(sample_buf)
    end

    sort!(boot_stats)
    alpha = 1.0 - confidence
    lo_idx = max(1, round(Int, alpha / 2.0 * n_boot))
    hi_idx = min(n_boot, round(Int, (1.0 - alpha / 2.0) * n_boot))

    return (boot_stats[lo_idx], boot_stats[hi_idx])
end

# ===========================================================================
# 9. TRIMMED STATISTICS
# ===========================================================================

"""
    trimmed_mean(x::AbstractVector{Float64}; proportiontocut::Float64=0.1) -> Float64

Compute the trimmed mean, discarding a fraction of extreme values from each tail.

# Arguments
- `x`: data vector.
- `proportiontocut`: fraction of data to remove from each tail (default 0.1 = 10%).

# Returns
- Mean of the remaining data.
"""
function trimmed_mean(
    x::AbstractVector{Float64};
    proportiontocut::Float64=0.1,
)::Float64
    n = length(x)
    if proportiontocut < 0.0 || proportiontocut >= 0.5
        throw(ArgumentError("trimmed_mean: proportiontocut must be in [0, 0.5)"))
    end
    k = floor(Int, proportiontocut * n)
    sorted = sort(x)
    trimmed = sorted[(k + 1):(n - k)]
    if isempty(trimmed)
        throw(ArgumentError("trimmed_mean: too many values trimmed, none remain"))
    end
    return Statistics.mean(trimmed)
end

"""
    winsorized_mean(x::AbstractVector{Float64}; limits::Tuple{Float64,Float64}=(0.1, 0.1))
    -> Float64

Compute the Winsorized mean, replacing extreme values with the nearest non-extreme values.

# Arguments
- `x`: data vector.
- `limits`: (lower, upper) proportions to Winsorize (default (0.1, 0.1) = 10% each tail).

# Returns
- Winsorized mean.
"""
function winsorized_mean(
    x::AbstractVector{Float64};
    limits::Tuple{Float64,Float64}=(0.1, 0.1),
)::Float64
    n = length(x)
    lo_prop, hi_prop = limits
    k_lo = floor(Int, lo_prop * n)
    k_hi = floor(Int, hi_prop * n)
    sorted = sort(x)
    lo_val = sorted[k_lo + 1]
    hi_val = sorted[n - k_hi]
    winsorized = clamp.(sorted, lo_val, hi_val)
    return Statistics.mean(winsorized)
end

# ===========================================================================
# EXPORTS
# ===========================================================================

export skewness, kurtosis, moments
export covariance_matrix, correlation_matrix
export histogram
export zscore, minmax_scale, robust_scale
export weighted_mean, weighted_variance, weighted_std
export RunningStats, update!, finalize
export describe
export bootstrap_ci
export trimmed_mean, winsorized_mean

end  # module ExtendedStats
