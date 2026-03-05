"""
    Interpolate

Pure-Julia interpolation algorithms for SciRS2.

Implements:
- Cubic spline interpolation (natural and clamped boundary conditions)
- Akima spline (robust to outliers)
- Piecewise linear interpolation
- Radial Basis Function (RBF) interpolation in nD
- Barycentric rational interpolation (Floater-Hormann weights)
- Monotone cubic (PCHIP) interpolation
- Bilinear interpolation on a 2D grid
"""
module Interpolate

using LinearAlgebra

# ===========================================================================
# 1. PIECEWISE LINEAR INTERPOLATION
# ===========================================================================

"""
    interp_linear(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                  xi::AbstractVector{Float64}) -> Vector{Float64}

Piecewise linear interpolation at query points `xi`.

Uses linear interpolation between the two nearest known points. For points
outside the data range, linear extrapolation is applied from the nearest
endpoint interval.

# Arguments
- `x`: sorted knot abscissae (strictly increasing).
- `y`: knot values, `length(y) == length(x)`.
- `xi`: query points (may be unsorted).

# Returns
- Interpolated values at each point in `xi`.

# Throws
- `ArgumentError` if `x` has fewer than 2 points or if `x` is not sorted.

# Examples
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [0.0, 1.0, 4.0, 9.0]
interp_linear(x, y, [0.5, 1.5, 2.5])   # [0.5, 2.5, 6.5]
```
"""
function interp_linear(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    xi::AbstractVector{Float64},
)::Vector{Float64}
    n = length(x)
    if n < 2
        throw(ArgumentError("interp_linear: need at least 2 knots, got $n"))
    end
    if length(y) != n
        throw(DimensionMismatch("interp_linear: x and y must have the same length"))
    end
    for i in 2:n
        if x[i] <= x[i - 1]
            throw(ArgumentError("interp_linear: x must be strictly increasing"))
        end
    end

    out = Vector{Float64}(undef, length(xi))
    for (qi, xq) in enumerate(xi)
        # Find interval via bisection
        idx = searchsortedlast(x, xq)
        if idx <= 0
            idx = 1
        elseif idx >= n
            idx = n - 1
        end
        # Linear interpolation
        t = (xq - x[idx]) / (x[idx + 1] - x[idx])
        out[qi] = (1.0 - t) * y[idx] + t * y[idx + 1]
    end
    return out
end

"""
    interp_linear(x, y, xi::Float64) -> Float64

Scalar version of piecewise linear interpolation.
"""
function interp_linear(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    xi::Float64,
)::Float64
    return interp_linear(x, y, [xi])[1]
end

# ===========================================================================
# 2. NATURAL CUBIC SPLINE
# ===========================================================================

"""
    SplineCoeffs

Coefficients for a cubic spline over n-1 intervals.

For the k-th interval [x[k], x[k+1]], the spline is:
  s(x) = a[k] + b[k]*(x-x[k]) + c[k]*(x-x[k])^2 + d[k]*(x-x[k])^3
"""
struct SplineCoeffs
    x::Vector{Float64}
    a::Vector{Float64}   # y values at knots
    b::Vector{Float64}   # first derivative coefficients
    c::Vector{Float64}   # second derivative / 2
    d::Vector{Float64}   # third derivative / 6
end

"""
    cubic_spline(x::AbstractVector{Float64}, y::AbstractVector{Float64};
                 bc::Symbol=:natural) -> SplineCoeffs

Fit a cubic spline through the points (x, y).

# Arguments
- `x`: strictly increasing knot abscissae (length n ≥ 2).
- `y`: knot values.
- `bc`: boundary condition, one of:
  - `:natural` (default): second derivative = 0 at endpoints.
  - `:not_a_knot`: the third derivative is continuous at the second and
    second-to-last interior knots (SciPy default for `CubicSpline`).
  - `:clamped` (requires keyword `dy_left` and `dy_right`).
- `dy_left`: left endpoint first derivative (only for `bc=:clamped`).
- `dy_right`: right endpoint first derivative (only for `bc=:clamped`).

# Returns
- `SplineCoeffs` struct that can be evaluated with `spline_eval`.

# Examples
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [0.0, 1.0, 4.0, 9.0]
sp = cubic_spline(x, y)
spline_eval(sp, 1.5)   # ≈ 2.25
```
"""
function cubic_spline(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64};
    bc::Symbol=:natural,
    dy_left::Float64=0.0,
    dy_right::Float64=0.0,
)::SplineCoeffs
    n = length(x)
    if n < 2
        throw(ArgumentError("cubic_spline: need at least 2 knots"))
    end
    if length(y) != n
        throw(DimensionMismatch("cubic_spline: x and y must have the same length"))
    end
    for i in 2:n
        if x[i] <= x[i - 1]
            throw(ArgumentError("cubic_spline: x must be strictly increasing"))
        end
    end

    xv = collect(Float64, x)
    yv = collect(Float64, y)

    # Special case: 2 knots => linear
    if n == 2
        h = xv[2] - xv[1]
        slope = (yv[2] - yv[1]) / h
        return SplineCoeffs(xv, yv, [slope, slope], [0.0, 0.0], [0.0, 0.0])
    end

    # Compute interval lengths
    h = diff(xv)   # length n-1

    # Build the tridiagonal system for second derivatives σ[i] (c coefficients times 2)
    # Standard cubic spline system: Thomas algorithm
    n_sys = n
    diag  = Vector{Float64}(undef, n_sys)
    upper = Vector{Float64}(undef, n_sys - 1)
    lower = Vector{Float64}(undef, n_sys - 1)
    rhs   = Vector{Float64}(undef, n_sys)

    # Interior rows (i = 2..n-1 in 1-based)
    for i in 2:(n - 1)
        lower[i - 1] = h[i - 1]
        diag[i]      = 2.0 * (h[i - 1] + h[i])
        upper[i]     = h[i]
        rhs[i] = 3.0 * ((yv[i + 1] - yv[i]) / h[i] - (yv[i] - yv[i - 1]) / h[i - 1])
    end

    # Boundary conditions
    if bc == :natural
        diag[1]   = 1.0
        upper[1]  = 0.0
        rhs[1]    = 0.0
        lower[n - 1] = 0.0
        diag[n]   = 1.0
        rhs[n]    = 0.0
    elseif bc == :clamped
        diag[1]  = 2.0 * h[1]
        upper[1] = h[1]
        rhs[1]   = 3.0 * ((yv[2] - yv[1]) / h[1] - dy_left)
        lower[n - 1] = h[n - 1]
        diag[n]  = 2.0 * h[n - 1]
        rhs[n]   = 3.0 * (dy_right - (yv[n] - yv[n - 1]) / h[n - 1])
    elseif bc == :not_a_knot
        # Not-a-knot: spline segments 1 and 2 share the same cubic polynomial
        # at x[2]; similarly segments n-2 and n-1 at x[n-1].
        # Left: d[1] = d[2]  =>  (c[2]-c[1])/h[1] = (c[3]-c[2])/h[2]
        diag[1]  = h[2]
        upper[1] = -(h[1] + h[2])
        rhs[1]   = 0.0
        # Not-a-knot right boundary
        lower[n - 1] = -(h[n - 2] + h[n - 1])
        diag[n]   = h[n - 2]
        rhs[n]    = 0.0
    else
        throw(ArgumentError("cubic_spline: unknown bc=$bc (use :natural, :clamped, :not_a_knot)"))
    end

    # Thomas algorithm (tridiagonal forward elimination)
    # The tridiagonal matrix has: lower[i-1], diag[i], upper[i]
    # We only need lower[1..n-1], diag[1..n], upper[1..n-1]
    c = zeros(Float64, n)
    c[1] = rhs[1] / diag[1]
    diag_mod = copy(diag)
    for i in 2:n
        li = (i > 1) ? lower[i - 1] : 0.0
        m = li / diag_mod[i - 1]
        diag_mod[i] -= m * (i <= n - 1 ? upper[i - 1] : 0.0)
        rhs[i] -= m * rhs[i - 1]
        c[i] = rhs[i] / diag_mod[i]
    end
    # Back substitution
    for i in (n - 1):-1:1
        c[i] = (rhs[i] - upper[i] * c[i + 1]) / diag_mod[i]
    end
    # c now contains σ = S'' / 2

    # Compute b and d from c
    b = Vector{Float64}(undef, n - 1)
    d = Vector{Float64}(undef, n - 1)
    for i in 1:(n - 1)
        b[i] = (yv[i + 1] - yv[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    end

    return SplineCoeffs(xv, yv, b, c[1:n-1], d)
end

"""
    spline_eval(sp::SplineCoeffs, xi::AbstractVector{Float64}) -> Vector{Float64}

Evaluate a cubic spline at query points `xi`.

Points outside the data range are extrapolated using the nearest end interval.

# Arguments
- `sp`: spline coefficients from `cubic_spline`.
- `xi`: query points.

# Returns
- Interpolated values.
"""
function spline_eval(sp::SplineCoeffs, xi::AbstractVector{Float64})::Vector{Float64}
    n = length(sp.x)
    out = Vector{Float64}(undef, length(xi))
    for (qi, xq) in enumerate(xi)
        idx = searchsortedlast(sp.x, xq)
        if idx <= 0
            idx = 1
        elseif idx >= n
            idx = n - 1
        end
        dx = xq - sp.x[idx]
        out[qi] = sp.a[idx] + sp.b[idx] * dx + sp.c[idx] * dx^2 + sp.d[idx] * dx^3
    end
    return out
end

"""
    spline_eval(sp::SplineCoeffs, xi::Float64) -> Float64

Scalar evaluation of a cubic spline.
"""
function spline_eval(sp::SplineCoeffs, xi::Float64)::Float64
    return spline_eval(sp, [xi])[1]
end

"""
    spline_deriv(sp::SplineCoeffs, xi::AbstractVector{Float64}; order::Int=1)
    -> Vector{Float64}

Evaluate the derivative of a cubic spline at query points `xi`.

# Arguments
- `sp`: spline coefficients.
- `xi`: query points.
- `order`: derivative order (1, 2, or 3).

# Returns
- Derivative values.
"""
function spline_deriv(
    sp::SplineCoeffs,
    xi::AbstractVector{Float64};
    order::Int=1,
)::Vector{Float64}
    if order < 1 || order > 3
        throw(ArgumentError("spline_deriv: order must be 1, 2, or 3"))
    end
    n = length(sp.x)
    out = Vector{Float64}(undef, length(xi))
    for (qi, xq) in enumerate(xi)
        idx = searchsortedlast(sp.x, xq)
        if idx <= 0
            idx = 1
        elseif idx >= n
            idx = n - 1
        end
        dx = xq - sp.x[idx]
        if order == 1
            out[qi] = sp.b[idx] + 2.0 * sp.c[idx] * dx + 3.0 * sp.d[idx] * dx^2
        elseif order == 2
            out[qi] = 2.0 * sp.c[idx] + 6.0 * sp.d[idx] * dx
        else
            out[qi] = 6.0 * sp.d[idx]
        end
    end
    return out
end

"""
    spline_integrate(sp::SplineCoeffs, a::Float64, b::Float64) -> Float64

Compute the definite integral of a cubic spline over [a, b].

Uses the exact antiderivative of the piecewise cubic.

# Examples
```julia
x = [0.0, 1.0, 2.0]
y = [0.0, 1.0, 4.0]
sp = cubic_spline(x, y)
spline_integrate(sp, 0.0, 2.0)   # ≈ integral of spline over [0, 2]
```
"""
function spline_integrate(sp::SplineCoeffs, a::Float64, b::Float64)::Float64
    if a > b
        return -spline_integrate(sp, b, a)
    end

    n = length(sp.x)
    total = 0.0

    # Find indices of intervals overlapping [a, b]
    i_start = searchsortedlast(sp.x, a)
    i_start = clamp(i_start, 1, n - 1)
    i_end   = searchsortedlast(sp.x, b)
    i_end   = clamp(i_end, 1, n - 1)

    for i in i_start:i_end
        lo = max(a, sp.x[i])
        hi = min(b, sp.x[i + 1])
        if lo >= hi
            continue
        end
        # Antiderivative at hi and lo relative to sp.x[i]
        function antideriv(t)
            sp.a[i] * t + sp.b[i] * t^2 / 2.0 + sp.c[i] * t^3 / 3.0 + sp.d[i] * t^4 / 4.0
        end
        total += antideriv(hi - sp.x[i]) - antideriv(lo - sp.x[i])
    end

    return total
end

# ===========================================================================
# 3. AKIMA SPLINE
# ===========================================================================

"""
    akima_spline(x::AbstractVector{Float64}, y::AbstractVector{Float64}) -> SplineCoeffs

Fit an Akima spline through the data points.

Akima splines are less sensitive to outliers than natural cubic splines because
the slopes at each knot are estimated locally from nearby data, not via a global
linear system.

The slope at knot i is:
  m[i] = (|d[i+1] - d[i]| * d[i-1] + |d[i-1] - d[i-2]| * d[i]) /
         (|d[i+1] - d[i]| + |d[i-1] - d[i-2]|)

where d[k] = (y[k+1] - y[k]) / (x[k+1] - x[k]) are the finite differences.

At the endpoints, phantom points are added by mirror extrapolation.

# Arguments
- `x`: strictly increasing knot abscissae (length n ≥ 3).
- `y`: knot values.

# Returns
- `SplineCoeffs` struct (same type as `cubic_spline`).

# Examples
```julia
x = collect(0.0:0.5:3.0)
y = sin.(x)
sp = akima_spline(x, y)
spline_eval(sp, [0.25, 1.25, 2.75])
```
"""
function akima_spline(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
)::SplineCoeffs
    n = length(x)
    if n < 3
        throw(ArgumentError("akima_spline: need at least 3 knots, got $n"))
    end
    if length(y) != n
        throw(DimensionMismatch("akima_spline: x and y must have the same length"))
    end
    for i in 2:n
        if x[i] <= x[i - 1]
            throw(ArgumentError("akima_spline: x must be strictly increasing"))
        end
    end

    xv = collect(Float64, x)
    yv = collect(Float64, y)
    h = diff(xv)
    d = diff(yv) ./ h   # finite differences, length n-1

    # Extend d with 2 phantom points on each side using mirror extrapolation:
    # d_ext = [d[1]+d[1]-d[2], d[1], d[1..n-1], d[n-1], d[n-1]+d[n-1]-d[n-2]]
    d_ext = vcat(
        [2.0 * d[1] - d[2], 2.0 * d[1] - d[2]],
        d,
        [2.0 * d[end] - d[end-1], 2.0 * d[end] - d[end-1]],
    )
    # Shift: d_ext[k] corresponds to d at index k-2 (k = 1..n+3 for n-1 original + 4 phantom)
    # We need indices 1..n+1 in the extended d to compute m[1..n]
    # d_ext[1]   = d[-1]  (phantom left-2)
    # d_ext[2]   = d[0]   (phantom left-1)
    # d_ext[3]   = d[1]
    # d_ext[n+1] = d[n-1]
    # d_ext[n+2] = d[n]   (phantom right+1)
    # d_ext[n+3] = d[n+1] (phantom right+2)

    # Compute slopes m[i] for i = 1..n
    m = Vector{Float64}(undef, n)
    for i in 1:n
        # Use d_ext indices: d_{i-2} = d_ext[i], d_{i-1} = d_ext[i+1],
        #                    d_i     = d_ext[i+2], d_{i+1} = d_ext[i+3]
        w1 = abs(d_ext[i + 3] - d_ext[i + 2])
        w2 = abs(d_ext[i + 1] - d_ext[i])
        if w1 + w2 < eps(Float64)
            m[i] = (d_ext[i + 1] + d_ext[i + 2]) / 2.0
        else
            m[i] = (w1 * d_ext[i + 1] + w2 * d_ext[i + 2]) / (w1 + w2)
        end
    end

    # Build cubic coefficients from slopes
    b = Vector{Float64}(undef, n - 1)
    c = Vector{Float64}(undef, n - 1)
    dv = Vector{Float64}(undef, n - 1)

    for i in 1:(n - 1)
        b[i] = m[i]
        c[i] = (3.0 * d[i] - 2.0 * m[i] - m[i + 1]) / h[i]
        dv[i] = (m[i] + m[i + 1] - 2.0 * d[i]) / (h[i]^2)
    end

    return SplineCoeffs(xv, yv, b, c, dv)
end

# ===========================================================================
# 4. PCHIP (MONOTONE CUBIC HERMITE INTERPOLATION)
# ===========================================================================

"""
    pchip(x::AbstractVector{Float64}, y::AbstractVector{Float64}) -> SplineCoeffs

Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).

This interpolation method preserves monotonicity in each interval — if the data
is monotone increasing on [x[i], x[i+1]], so is the interpolant. It avoids
spurious oscillations that can occur with natural or Akima splines.

The derivative at each knot is estimated using the method of Fritsch & Carlson
(1980) with the Hyman (1983) modification for monotonicity.

# Arguments
- `x`: strictly increasing knot abscissae (length n ≥ 2).
- `y`: knot values.

# Returns
- `SplineCoeffs` struct.

# Examples
```julia
x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y = [0.0, 0.5, 0.9, 0.95, 1.0, 1.0]  # monotone increasing, levelling off
sp = pchip(x, y)
spline_eval(sp, [0.5, 2.5, 4.5])
```
"""
function pchip(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
)::SplineCoeffs
    n = length(x)
    if n < 2
        throw(ArgumentError("pchip: need at least 2 knots"))
    end
    if length(y) != n
        throw(DimensionMismatch("pchip: x and y must have the same length"))
    end

    xv = collect(Float64, x)
    yv = collect(Float64, y)
    h = diff(xv)
    d = diff(yv) ./ h   # finite differences

    # Compute initial slopes using the weighted harmonic mean
    m = Vector{Float64}(undef, n)

    # Interior slopes (Fritsch-Carlson monotone method)
    for i in 2:(n - 1)
        # Check sign agreement
        if d[i - 1] * d[i] <= 0.0
            m[i] = 0.0
        else
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        end
    end

    # Endpoint slopes: one-sided three-point estimates
    m[1] = pchip_endpoint_slope(h[1], h[2], d[1], d[2])
    m[n] = pchip_endpoint_slope(h[n - 1], n >= 3 ? h[n - 2] : h[n - 1],
                                 d[n - 1], n >= 3 ? d[n - 2] : d[n - 1])

    # Hyman (1983) monotonicity check: limit slopes to prevent overshoots
    for i in 1:n
        li = (i > 1) ? d[i - 1] : d[1]
        ri = (i < n) ? d[i] : d[n - 1]
        if i == 1
            ref = d[1]
        elseif i == n
            ref = d[n - 1]
        else
            ref = (sign(li) == sign(ri)) ? min(abs(li), abs(ri)) : 0.0
        end
        if abs(m[i]) > 3.0 * abs(ref)
            m[i] = 3.0 * ref
        end
    end

    # Build cubic coefficients
    b = Vector{Float64}(undef, n - 1)
    c = Vector{Float64}(undef, n - 1)
    dv = Vector{Float64}(undef, n - 1)

    for i in 1:(n - 1)
        b[i] = m[i]
        c[i] = (3.0 * d[i] - 2.0 * m[i] - m[i + 1]) / h[i]
        dv[i] = (m[i] + m[i + 1] - 2.0 * d[i]) / (h[i]^2)
    end

    return SplineCoeffs(xv, yv, b, c, dv)
end

"""Compute one-sided endpoint slope for PCHIP using three-point formula."""
function pchip_endpoint_slope(h1::Float64, h2::Float64, d1::Float64, d2::Float64)::Float64
    slope = ((2.0 * h1 + h2) * d1 - h1 * d2) / (h1 + h2)
    # Enforce sign consistency
    if sign(slope) != sign(d1)
        return 0.0
    elseif sign(d1) != sign(d2) && abs(slope) > abs(3.0 * d1)
        return 3.0 * d1
    end
    return slope
end

# ===========================================================================
# 5. BARYCENTRIC RATIONAL INTERPOLATION (Floater-Hormann)
# ===========================================================================

"""
    BarycentricInterp

Barycentric rational interpolant with Floater-Hormann weights.

This representation allows O(n) evaluation per query point after O(n²)
precomputation of the weights.
"""
struct BarycentricInterp
    x::Vector{Float64}   # nodes
    y::Vector{Float64}   # values
    w::Vector{Float64}   # barycentric weights
end

"""
    floater_hormann(x::AbstractVector{Float64}, y::AbstractVector{Float64};
                    d::Int=3) -> BarycentricInterp

Build a Floater-Hormann barycentric rational interpolant of blending degree d.

The interpolant is a rational function of degree (n-1, n-1-d) that is exact
on the given nodes and avoids the Runge phenomenon.

For d=0, this degenerates to piecewise linear interpolation. For d=n-1, it
is the standard polynomial interpolation (Lagrange form). The recommended
range for d is 3 to 5.

# Arguments
- `x`: nodes (need not be sorted but must be distinct).
- `y`: values.
- `d`: blending degree (0 ≤ d ≤ n-1; default 3).

# Returns
- `BarycentricInterp` struct.

# References
- Floater & Hormann (2007), "Barycentric rational interpolation with no poles
  and high rates of approximation", Numerische Mathematik.

# Examples
```julia
x = collect(range(0, π, length=10))
y = sin.(x)
interp = floater_hormann(x, y; d=3)
bary_eval(interp, [0.5, 1.0, 2.0])
```
"""
function floater_hormann(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64};
    d::Int=3,
)::BarycentricInterp
    n = length(x)
    if n < 2
        throw(ArgumentError("floater_hormann: need at least 2 nodes"))
    end
    if length(y) != n
        throw(DimensionMismatch("floater_hormann: x and y must have the same length"))
    end
    if d < 0 || d >= n
        throw(ArgumentError("floater_hormann: d must be in [0, n-1], got $d"))
    end

    xv = collect(Float64, x)
    yv = collect(Float64, y)

    # Compute Floater-Hormann weights
    # w[i] = (-1)^i * Σ_{k=max(0,i-d)}^{min(i,n-1-d)} (1/Π_{j∈I_k,j≠i} |x[i]-x[j]|)
    # where I_k = {k, k+1, ..., k+d}
    w = zeros(Float64, n)
    for i in 1:n
        for k in max(1, i - d):min(i, n - d)
            # I_k in 1-based is {k, k+1, ..., k+d}
            prod = 1.0
            for j in k:(k + d)
                if j != i
                    diff_val = xv[i] - xv[j]
                    if abs(diff_val) < eps(Float64)
                        throw(ArgumentError("floater_hormann: duplicate node at index $i and $j"))
                    end
                    prod *= abs(diff_val)
                end
            end
            w[i] += 1.0 / prod
        end
        # Sign: (-1)^(i-1)
        if (i - 1) % 2 == 1
            w[i] = -w[i]
        end
    end

    return BarycentricInterp(xv, yv, w)
end

"""
    bary_eval(interp::BarycentricInterp, xi::AbstractVector{Float64}) -> Vector{Float64}

Evaluate a barycentric rational interpolant at query points.

# Arguments
- `interp`: precomputed interpolant from `floater_hormann`.
- `xi`: query points.

# Returns
- Interpolated values.

# Throws
- `ErrorException` if any query point coincides with a node (exact node matches
  are handled via exact return of the node value).
"""
function bary_eval(interp::BarycentricInterp, xi::AbstractVector{Float64})::Vector{Float64}
    out = Vector{Float64}(undef, length(xi))
    for (qi, xq) in enumerate(xi)
        # Check for exact node match (avoid division by zero)
        node_idx = findfirst(t -> abs(t - xq) < eps(Float64) * (1.0 + abs(xq)), interp.x)
        if node_idx !== nothing
            out[qi] = interp.y[node_idx]
            continue
        end

        # Barycentric formula: P(xq) = Σ w[i]/(xq-x[i]) * y[i] / Σ w[i]/(xq-x[i])
        num = 0.0
        den = 0.0
        for i in eachindex(interp.x)
            t = interp.w[i] / (xq - interp.x[i])
            num += t * interp.y[i]
            den += t
        end
        out[qi] = num / den
    end
    return out
end

"""
    bary_eval(interp::BarycentricInterp, xi::Float64) -> Float64

Scalar evaluation of a barycentric rational interpolant.
"""
function bary_eval(interp::BarycentricInterp, xi::Float64)::Float64
    return bary_eval(interp, [xi])[1]
end

# ===========================================================================
# 6. RADIAL BASIS FUNCTION (RBF) INTERPOLATION
# ===========================================================================

"""
    RBFInterp

Precomputed RBF interpolant: y(x) = Σ λ[i] * φ(‖x - x[i]‖) + p(x)

where p(x) is an optional polynomial trend (degree 0 or 1).
"""
struct RBFInterp
    centers::Matrix{Float64}     # (n_centers, dim)
    weights::Vector{Float64}     # λ coefficients
    poly_coeffs::Vector{Float64} # polynomial coefficients (intercept + linear terms)
    kernel::Symbol               # :gaussian, :multiquadric, :inv_multiquadric,
                                 # :thin_plate, :linear, :cubic
    epsilon::Float64             # shape parameter (for gaussian, multiquadric, etc.)
    poly_degree::Int             # polynomial degree: -1 (none), 0 (constant), 1 (linear)
end

"""
    rbf_build(centers::AbstractMatrix{Float64}, values::AbstractVector{Float64};
              kernel::Symbol=:thin_plate_spline, epsilon::Float64=1.0,
              poly_degree::Int=1) -> RBFInterp

Build a Radial Basis Function interpolant.

# Arguments
- `centers`: (n × d) matrix of data site coordinates (n points in d dimensions).
- `values`: n-dimensional vector of function values at the centers.
- `kernel`: RBF kernel function, one of:
  - `:gaussian`: φ(r) = exp(-(εr)²)
  - `:multiquadric`: φ(r) = √(1 + (εr)²)
  - `:inv_multiquadric`: φ(r) = 1/√(1 + (εr)²)
  - `:thin_plate_spline` (default): φ(r) = r² log(r), r > 0; 0 otherwise
  - `:linear`: φ(r) = r
  - `:cubic`: φ(r) = r³
  - `:quintic`: φ(r) = r⁵
- `epsilon`: shape parameter (default 1.0; only meaningful for gaussian,
  multiquadric, and inv_multiquadric).
- `poly_degree`: degree of the polynomial augmentation:
  - `-1`: no polynomial (may cause ill-conditioning for some kernels)
  - `0`: constant term (1 extra basis function)
  - `1` (default): linear trend (1 + d extra basis functions)

# Returns
- `RBFInterp` struct to use with `rbf_eval`.

# Examples
```julia
# 1D example
centers = reshape([0.0, 1.0, 2.0, 3.0], 4, 1)
values  = [0.0, 1.0, 4.0, 9.0]
interp  = rbf_build(centers, values; poly_degree=1)
rbf_eval(interp, reshape([0.5, 1.5], 2, 1))   # ≈ [0.25, 2.25]

# 2D example
pts = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]   # unit square corners
vals = [0.0, 1.0, 1.0, 2.0]                    # f(x,y) = x + y
interp = rbf_build(pts, vals)
rbf_eval(interp, [0.5 0.5])                    # ≈ [1.0]
```
"""
function rbf_build(
    centers::AbstractMatrix{Float64},
    values::AbstractVector{Float64};
    kernel::Symbol=:thin_plate_spline,
    epsilon::Float64=1.0,
    poly_degree::Int=1,
)::RBFInterp
    n, dim = size(centers)
    if length(values) != n
        throw(DimensionMismatch("rbf_build: centers has $n rows but values has $(length(values)) elements"))
    end
    if n < 2
        throw(ArgumentError("rbf_build: need at least 2 centers"))
    end

    c_mat = collect(Float64, centers)

    # Polynomial augmentation size
    n_poly = if poly_degree == -1
        0
    elseif poly_degree == 0
        1
    else  # degree 1
        1 + dim
    end

    n_sys = n + n_poly

    # Build RBF distance matrix Φ (n × n)
    Phi = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        r = norm(c_mat[i, :] - c_mat[j, :])
        Phi[i, j] = rbf_kernel(kernel, r, epsilon)
    end

    if n_poly == 0
        # No polynomial: solve Φ λ = f
        lam = Phi \ collect(values)
        return RBFInterp(c_mat, lam, Float64[], kernel, epsilon, poly_degree)
    end

    # Polynomial block P (n × n_poly)
    P = build_poly_block(c_mat, n, dim, poly_degree)

    # Extended system: [Φ  P; P' 0] * [λ; c] = [f; 0]
    A = zeros(Float64, n_sys, n_sys)
    A[1:n, 1:n] = Phi
    A[1:n, (n + 1):n_sys] = P
    A[(n + 1):n_sys, 1:n] = P'

    rhs = zeros(Float64, n_sys)
    rhs[1:n] = values

    sol = A \ rhs
    lam = sol[1:n]
    poly_c = sol[(n + 1):n_sys]

    return RBFInterp(c_mat, lam, poly_c, kernel, epsilon, poly_degree)
end

"""Evaluate the RBF kernel φ(r)."""
function rbf_kernel(kernel::Symbol, r::Float64, epsilon::Float64)::Float64
    if kernel == :gaussian
        return exp(-(epsilon * r)^2)
    elseif kernel == :multiquadric
        return sqrt(1.0 + (epsilon * r)^2)
    elseif kernel == :inv_multiquadric
        return 1.0 / sqrt(1.0 + (epsilon * r)^2)
    elseif kernel == :thin_plate_spline
        return r < eps(Float64) ? 0.0 : r^2 * log(r)
    elseif kernel == :linear
        return r
    elseif kernel == :cubic
        return r^3
    elseif kernel == :quintic
        return r^5
    else
        throw(ArgumentError("rbf_kernel: unknown kernel $kernel"))
    end
end

"""Build polynomial basis block P for RBF augmentation."""
function build_poly_block(
    centers::Matrix{Float64},
    n::Int,
    dim::Int,
    poly_degree::Int,
)::Matrix{Float64}
    if poly_degree == 0
        return ones(Float64, n, 1)
    else  # degree 1
        return hcat(ones(Float64, n, 1), centers)
    end
end

"""
    rbf_eval(interp::RBFInterp, xi::AbstractMatrix{Float64}) -> Vector{Float64}

Evaluate an RBF interpolant at query points.

# Arguments
- `interp`: precomputed RBF interpolant from `rbf_build`.
- `xi`: (m × d) matrix of query coordinates.

# Returns
- Vector of m interpolated values.
"""
function rbf_eval(
    interp::RBFInterp,
    xi::AbstractMatrix{Float64},
)::Vector{Float64}
    m, dim = size(xi)
    n = size(interp.centers, 1)

    xi_mat = collect(Float64, xi)
    out = zeros(Float64, m)

    for qi in 1:m
        xq = xi_mat[qi, :]

        # RBF part: Σ λ[i] * φ(‖xq - c[i]‖)
        s = 0.0
        for i in 1:n
            r = norm(xq - interp.centers[i, :])
            s += interp.weights[i] * rbf_kernel(interp.kernel, r, interp.epsilon)
        end

        # Polynomial part
        if interp.poly_degree >= 0
            s += interp.poly_coeffs[1]  # constant
        end
        if interp.poly_degree >= 1
            for d in 1:dim
                s += interp.poly_coeffs[1 + d] * xq[d]
            end
        end

        out[qi] = s
    end

    return out
end

"""
    rbf_eval(interp::RBFInterp, xi::AbstractVector{Float64}) -> Float64

Evaluate an RBF interpolant at a single query point (treated as 1D).
"""
function rbf_eval(interp::RBFInterp, xi::AbstractVector{Float64})::Float64
    return rbf_eval(interp, reshape(collect(Float64, xi), 1, length(xi)))[1]
end

# ===========================================================================
# 7. BILINEAR INTERPOLATION ON A REGULAR 2D GRID
# ===========================================================================

"""
    bilinear_interp(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                    z::AbstractMatrix{Float64},
                    xi::AbstractVector{Float64},
                    yi::AbstractVector{Float64}) -> Vector{Float64}

Bilinear interpolation on a regular 2D grid.

The grid is defined by coordinate vectors `x` (column axis, length n) and
`y` (row axis, length m), with values `z[i, j]` at the point (x[j], y[i]).

# Arguments
- `x`: column coordinates (strictly increasing, length n).
- `y`: row coordinates (strictly increasing, length m).
- `z`: (m × n) matrix of values.
- `xi`, `yi`: query x and y coordinates (must have the same length).

# Returns
- Vector of interpolated values at (xi[k], yi[k]).

# Throws
- `DimensionMismatch` if `z` size doesn't match `x` and `y`.
- `ArgumentError` if xi and yi have different lengths.

# Examples
```julia
x = [0.0, 1.0, 2.0]
y = [0.0, 1.0]
z = [0.0 1.0 4.0;   # row y=0
     1.0 2.0 5.0]   # row y=1
bilinear_interp(x, y, z, [0.5, 1.5], [0.5, 0.5])
```
"""
function bilinear_interp(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    z::AbstractMatrix{Float64},
    xi::AbstractVector{Float64},
    yi::AbstractVector{Float64},
)::Vector{Float64}
    m, n = size(z)
    if length(x) != n
        throw(DimensionMismatch("bilinear_interp: x has $(length(x)) elements but z has $n columns"))
    end
    if length(y) != m
        throw(DimensionMismatch("bilinear_interp: y has $(length(y)) elements but z has $m rows"))
    end
    if length(xi) != length(yi)
        throw(ArgumentError("bilinear_interp: xi and yi must have the same length"))
    end

    xv = collect(Float64, x)
    yv = collect(Float64, y)
    out = Vector{Float64}(undef, length(xi))

    for qi in eachindex(xi)
        # Find bounding cell in x
        ix = searchsortedlast(xv, xi[qi])
        ix = clamp(ix, 1, n - 1)
        # Find bounding cell in y
        iy = searchsortedlast(yv, yi[qi])
        iy = clamp(iy, 1, m - 1)

        tx = (xi[qi] - xv[ix]) / (xv[ix + 1] - xv[ix])
        ty = (yi[qi] - yv[iy]) / (yv[iy + 1] - yv[iy])

        # Bilinear formula
        out[qi] = (
            (1.0 - tx) * (1.0 - ty) * z[iy, ix] +
            tx          * (1.0 - ty) * z[iy, ix + 1] +
            (1.0 - tx) * ty          * z[iy + 1, ix] +
            tx          * ty          * z[iy + 1, ix + 1]
        )
    end

    return out
end

# ===========================================================================
# EXPORTS
# ===========================================================================

export interp_linear
export SplineCoeffs, cubic_spline, spline_eval, spline_deriv, spline_integrate
export akima_spline
export pchip
export BarycentricInterp, floater_hormann, bary_eval
export RBFInterp, rbf_build, rbf_eval
export bilinear_interp

end  # module Interpolate
