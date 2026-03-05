"""
    ExtendedOptimize

Extended optimization algorithms for SciRS2.

Provides:
- L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- Adam optimizer (adaptive moment estimation)
- Gradient descent with Armijo line search
- Coordinate descent
- Simulated annealing
- Differential evolution
- Powell's method (derivative-free, conjugate direction search)
- Penalty method for constrained optimization
- Multi-start optimization wrapper
"""
module ExtendedOptimize

using LinearAlgebra

# ===========================================================================
# 1. GRADIENT DESCENT WITH ARMIJO LINE SEARCH
# ===========================================================================

"""
    gradient_descent(f::Function, grad_f::Function, x0::AbstractVector{Float64};
                     tol::Float64=1e-6, max_iter::Int=10000,
                     alpha0::Float64=1.0, beta::Float64=0.5, c::Float64=1e-4)
    -> NamedTuple{(:x_min, :f_min, :converged, :iterations)}

Minimize a differentiable function using gradient descent with Armijo backtracking
line search.

At each step, the search direction is the negative gradient. The step size is
determined by backtracking from `alpha0` until the Armijo condition is satisfied:
  f(x - α * ∇f(x)) ≤ f(x) - c * α * ‖∇f(x)‖²

# Arguments
- `f`: objective function `f(x::Vector{Float64}) -> Float64`.
- `grad_f`: gradient function `grad_f(x::Vector{Float64}) -> Vector{Float64}`.
- `x0`: initial guess.
- `tol`: convergence tolerance on gradient norm (default 1e-6).
- `max_iter`: maximum number of iterations (default 10000).
- `alpha0`: initial step size for line search (default 1.0).
- `beta`: line search backtracking factor (0 < β < 1; default 0.5).
- `c`: Armijo sufficient decrease constant (default 1e-4).

# Returns
Named tuple with `x_min`, `f_min`, `converged`, `iterations`.

# Examples
```julia
f(x) = (x[1] - 3.0)^2 + (x[2] + 1.0)^2
grad_f(x) = [2.0*(x[1] - 3.0), 2.0*(x[2] + 1.0)]
result = gradient_descent(f, grad_f, [0.0, 0.0])
result.x_min   # ≈ [3.0, -1.0]
```
"""
function gradient_descent(
    f::Function,
    grad_f::Function,
    x0::AbstractVector{Float64};
    tol::Float64=1e-6,
    max_iter::Int=10000,
    alpha0::Float64=1.0,
    beta::Float64=0.5,
    c::Float64=1e-4,
)
    x = collect(Float64, x0)
    f_val = f(x)
    converged = false

    for iter in 1:max_iter
        g = grad_f(x)
        g_norm = norm(g)

        if g_norm < tol
            converged = true
            return (x_min=x, f_min=f_val, converged=true, iterations=iter)
        end

        # Armijo backtracking line search
        alpha = alpha0
        for _ in 1:50
            x_new = x .- alpha .* g
            f_new = f(x_new)
            if f_new <= f_val - c * alpha * g_norm^2
                break
            end
            alpha *= beta
        end

        x = x .- alpha .* g
        f_val = f(x)
    end

    return (x_min=x, f_min=f_val, converged=converged, iterations=max_iter)
end

# ===========================================================================
# 2. L-BFGS
# ===========================================================================

"""
    lbfgs(f::Function, grad_f::Function, x0::AbstractVector{Float64};
          tol::Float64=1e-6, max_iter::Int=1000, m::Int=10,
          c1::Float64=1e-4, c2::Float64=0.9)
    -> NamedTuple{(:x_min, :f_min, :converged, :iterations)}

Minimize a smooth function using the L-BFGS algorithm.

L-BFGS (Limited-memory BFGS) is a quasi-Newton method that approximates the
inverse Hessian using the last `m` gradient/step pairs. It achieves superlinear
convergence on smooth objectives without storing the full Hessian.

The Wolfe strong line search conditions are used to ensure convergence:
  Armijo: f(x + α*p) ≤ f(x) + c1*α*(∇f(x)ᵀp)
  Curvature: |∇f(x + α*p)ᵀp| ≤ c2 * |∇f(x)ᵀp|

# Arguments
- `f`: objective function `f(x::Vector) -> Float64`.
- `grad_f`: gradient `grad_f(x::Vector) -> Vector{Float64}`.
- `x0`: initial guess.
- `tol`: gradient norm convergence tolerance (default 1e-6).
- `max_iter`: maximum number of L-BFGS iterations (default 1000).
- `m`: memory parameter — number of curvature pairs stored (default 10).
- `c1`, `c2`: Wolfe condition constants.

# Returns
Named tuple with `x_min`, `f_min`, `converged`, `iterations`.

# Examples
```julia
# Rosenbrock function
f(x) = (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
grad_f(x) = [
    -2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2),
    200*(x[2] - x[1]^2)
]
result = lbfgs(f, grad_f, [0.0, 0.0]; tol=1e-8, max_iter=2000)
result.x_min   # ≈ [1.0, 1.0]
```
"""
function lbfgs(
    f::Function,
    grad_f::Function,
    x0::AbstractVector{Float64};
    tol::Float64=1e-6,
    max_iter::Int=1000,
    m::Int=10,
    c1::Float64=1e-4,
    c2::Float64=0.9,
)
    x = collect(Float64, x0)
    f_val = f(x)
    g = grad_f(x)

    # Storage for L-BFGS curvature pairs
    s_list = Vector{Vector{Float64}}()  # x differences
    y_list = Vector{Vector{Float64}}()  # gradient differences
    rho_list = Vector{Float64}()

    converged = false

    for iter in 1:max_iter
        g_norm = norm(g)
        if g_norm < tol
            converged = true
            break
        end

        # Two-loop L-BFGS recursion to compute search direction p = -H * g
        q = copy(g)
        alpha_vals = Vector{Float64}(undef, length(s_list))

        for i in length(s_list):-1:1
            alpha_vals[i] = rho_list[i] * dot(s_list[i], q)
            q .-= alpha_vals[i] .* y_list[i]
        end

        # Initial Hessian approximation: H₀ = (sᵀy / yᵀy) * I
        if !isempty(y_list)
            y_last = y_list[end]
            s_last = s_list[end]
            gamma = dot(s_last, y_last) / dot(y_last, y_last)
            r = gamma .* q
        else
            r = q ./ max(g_norm, eps(Float64))
        end

        for i in 1:length(s_list)
            beta_val = rho_list[i] * dot(y_list[i], r)
            r .+= s_list[i] .* (alpha_vals[i] - beta_val)
        end

        p = -r  # search direction

        # Wolfe line search
        alpha = wolfe_line_search(f, grad_f, x, p, g, f_val; c1=c1, c2=c2)

        # Update
        x_new = x .+ alpha .* p
        g_new = grad_f(x_new)
        f_val_new = f(x_new)

        s_k = x_new .- x
        y_k = g_new .- g

        sy = dot(s_k, y_k)

        if sy > eps(Float64) * norm(y_k)^2
            # Store curvature pair
            push!(s_list, s_k)
            push!(y_list, y_k)
            push!(rho_list, 1.0 / sy)

            # Evict old pairs if memory exceeded
            if length(s_list) > m
                popfirst!(s_list)
                popfirst!(y_list)
                popfirst!(rho_list)
            end
        end

        x = x_new
        g = g_new
        f_val = f_val_new
    end

    return (x_min=x, f_min=f_val, converged=converged, iterations=max_iter)
end

"""Zoom phase of Wolfe line search: find α satisfying strong Wolfe conditions."""
function _wolfe_zoom(f, grad_f, x, p, g0, f0, alpha_lo, alpha_hi, f_lo, f_hi; c1, c2)
    for _ in 1:20
        alpha = (alpha_lo + alpha_hi) / 2.0
        f_new = f(x .+ alpha .* p)
        dg0 = dot(g0, p)

        if f_new > f0 + c1 * alpha * dg0 || f_new >= f_lo
            alpha_hi = alpha
            f_hi = f_new
        else
            g_new = grad_f(x .+ alpha .* p)
            dg_new = dot(g_new, p)
            if abs(dg_new) <= -c2 * dg0
                return alpha
            end
            if dg_new * (alpha_hi - alpha_lo) >= 0.0
                alpha_hi = alpha_lo
                f_hi = f_lo
            end
            alpha_lo = alpha
            f_lo = f_new
        end
    end
    return (alpha_lo + alpha_hi) / 2.0
end

"""Wolfe line search returning a step size satisfying strong Wolfe conditions."""
function wolfe_line_search(f, grad_f, x, p, g, f0; c1=1e-4, c2=0.9, alpha_max=10.0)
    dg0 = dot(g, p)
    if dg0 >= 0.0
        return 1.0  # Not a descent direction; return default step
    end

    alpha_prev = 0.0
    alpha = 1.0
    f_prev = f0

    for i in 1:20
        f_new = f(x .+ alpha .* p)

        if f_new > f0 + c1 * alpha * dg0 || (i > 1 && f_new >= f_prev)
            return _wolfe_zoom(f, grad_f, x, p, g, f0, alpha_prev, alpha, f_prev, f_new; c1=c1, c2=c2)
        end

        g_new = grad_f(x .+ alpha .* p)
        dg_new = dot(g_new, p)

        if abs(dg_new) <= -c2 * dg0
            return alpha
        end

        if dg_new >= 0.0
            return _wolfe_zoom(f, grad_f, x, p, g, f0, alpha, alpha_prev, f_new, f_prev; c1=c1, c2=c2)
        end

        alpha_prev = alpha
        f_prev = f_new
        alpha = min(2.0 * alpha, alpha_max)
    end

    return alpha
end

# ===========================================================================
# 3. ADAM OPTIMIZER
# ===========================================================================

"""
    adam(grad_f::Function, x0::AbstractVector{Float64};
         lr::Float64=0.001, beta1::Float64=0.9, beta2::Float64=0.999,
         epsilon::Float64=1e-8, max_iter::Int=10000, tol::Float64=1e-6)
    -> NamedTuple{(:x_min, :iterations, :converged)}

Minimize a function using the Adam (Adaptive Moment Estimation) optimizer.

Adam maintains per-parameter adaptive learning rates using exponentially
decaying averages of past gradients (first moment) and squared gradients
(second moment).

Update rule:
  m_t = β₁ * m_{t-1} + (1-β₁) * g_t           (biased first moment)
  v_t = β₂ * v_{t-1} + (1-β₂) * g_t²           (biased second moment)
  m̂_t = m_t / (1 - β₁^t)                        (bias-corrected)
  v̂_t = v_t / (1 - β₂^t)                        (bias-corrected)
  x_{t+1} = x_t - lr * m̂_t / (√v̂_t + ε)

# Arguments
- `grad_f`: gradient function.
- `x0`: initial parameter vector.
- `lr`: learning rate (default 0.001).
- `beta1`: exponential decay rate for first moment (default 0.9).
- `beta2`: exponential decay rate for second moment (default 0.999).
- `epsilon`: numerical stability constant (default 1e-8).
- `max_iter`: maximum iterations (default 10000).
- `tol`: gradient norm convergence tolerance (default 1e-6).

# Returns
Named tuple with `x_min`, `iterations`, `converged`.

# Examples
```julia
# Minimize (x-3)²
grad_f(x) = [2.0 * (x[1] - 3.0)]
result = adam(grad_f, [0.0]; lr=0.1, max_iter=5000)
result.x_min   # ≈ [3.0]
```
"""
function adam(
    grad_f::Function,
    x0::AbstractVector{Float64};
    lr::Float64=0.001,
    beta1::Float64=0.9,
    beta2::Float64=0.999,
    epsilon::Float64=1e-8,
    max_iter::Int=10000,
    tol::Float64=1e-6,
)
    x = collect(Float64, x0)
    m = zeros(Float64, length(x))   # first moment
    v = zeros(Float64, length(x))   # second moment
    converged = false

    for t in 1:max_iter
        g = grad_f(x)
        g_norm = norm(g)

        if g_norm < tol
            converged = true
            return (x_min=x, iterations=t, converged=true)
        end

        m = beta1 .* m .+ (1.0 - beta1) .* g
        v = beta2 .* v .+ (1.0 - beta2) .* g .^ 2

        # Bias correction
        m_hat = m ./ (1.0 - beta1^t)
        v_hat = v ./ (1.0 - beta2^t)

        x .-= lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
    end

    return (x_min=x, iterations=max_iter, converged=converged)
end

# ===========================================================================
# 4. COORDINATE DESCENT
# ===========================================================================

"""
    coordinate_descent(f::Function, x0::AbstractVector{Float64};
                       tol::Float64=1e-6, max_iter::Int=10000,
                       step::Float64=0.01)
    -> NamedTuple{(:x_min, :f_min, :converged, :iterations)}

Minimize a function by cyclic coordinate descent.

Each iteration cycles through all coordinates, minimizing along each axis
using a golden-section search with a bracket of ±`step * max(1, |x|)`.

# Arguments
- `f`: multivariate objective function.
- `x0`: initial guess.
- `tol`: convergence tolerance (on f value change per cycle; default 1e-6).
- `max_iter`: maximum number of full cycles.
- `step`: relative search step for 1D minimization (default 0.01).

# Returns
Named tuple with `x_min`, `f_min`, `converged`, `iterations`.
"""
function coordinate_descent(
    f::Function,
    x0::AbstractVector{Float64};
    tol::Float64=1e-6,
    max_iter::Int=10000,
    step::Float64=0.01,
)
    x = collect(Float64, x0)
    n = length(x)
    f_val = f(x)
    converged = false

    for iter in 1:max_iter
        f_prev = f_val
        for i in 1:n
            # 1D golden-section search along coordinate i
            xi = x[i]
            bracket = max(1.0, abs(xi)) * step
            lo = xi - bracket * 10.0
            hi = xi + bracket * 10.0

            phi = (sqrt(5.0) - 1.0) / 2.0
            a, b = lo, hi
            x1 = b - phi * (b - a)
            x2 = a + phi * (b - a)

            x[i] = x1; f1 = f(x)
            x[i] = x2; f2 = f(x)

            for _ in 1:50
                if abs(b - a) < tol
                    break
                end
                if f1 < f2
                    b = x2; x2 = x1; f2 = f1
                    x1 = b - phi * (b - a)
                    x[i] = x1; f1 = f(x)
                else
                    a = x1; x1 = x2; f1 = f2
                    x2 = a + phi * (b - a)
                    x[i] = x2; f2 = f(x)
                end
            end
            x[i] = (a + b) / 2.0
            f_val = f(x)
        end

        if abs(f_prev - f_val) < tol * (abs(f_prev) + 1.0)
            converged = true
            break
        end
    end

    return (x_min=x, f_min=f_val, converged=converged, iterations=max_iter)
end

# ===========================================================================
# 5. SIMULATED ANNEALING
# ===========================================================================

"""
    simulated_annealing(f::Function, x0::AbstractVector{Float64};
                        t_init::Float64=1.0, t_final::Float64=1e-6,
                        cooling_rate::Float64=0.995, max_iter::Int=100000,
                        step_size::Float64=0.1, seed::UInt64=42)
    -> NamedTuple{(:x_min, :f_min, :iterations)}

Minimize a function using Simulated Annealing (SA).

SA is a metaheuristic that allows uphill moves with probability exp(-ΔE/T),
where ΔE is the increase in objective and T is the current temperature. This
allows escaping local minima.

The temperature schedule follows geometric cooling: T_{k+1} = cooling_rate * T_k.

# Arguments
- `f`: objective function.
- `x0`: initial solution.
- `t_init`: initial temperature (default 1.0; should be on the order of typical ΔE).
- `t_final`: stopping temperature (default 1e-6).
- `cooling_rate`: temperature reduction factor per step (default 0.995).
- `max_iter`: maximum number of SA steps (default 100000).
- `step_size`: perturbation step size (default 0.1).
- `seed`: random seed for reproducibility (LCG generator).

# Returns
Named tuple with `x_min`, `f_min`, `iterations`.
"""
function simulated_annealing(
    f::Function,
    x0::AbstractVector{Float64};
    t_init::Float64=1.0,
    t_final::Float64=1e-6,
    cooling_rate::Float64=0.995,
    max_iter::Int=100000,
    step_size::Float64=0.1,
    seed::UInt64=42,
)
    x_curr = collect(Float64, x0)
    f_curr = f(x_curr)
    x_best = copy(x_curr)
    f_best = f_curr
    n = length(x_curr)
    T = t_init

    # LCG random number generator
    state = Ref{UInt64}(seed)
    function lcg_rand()::Float64
        state[] = state[] * 6364136223846793005 + 1442695040888963407
        return (Float64(state[] >> 11) / Float64(1 << 53))
    end
    function lcg_randn()::Float64
        # Box-Muller transform using two uniform samples
        u1 = max(lcg_rand(), eps(Float64))
        u2 = lcg_rand()
        return sqrt(-2.0 * log(u1)) * cos(2π * u2)
    end

    for iter in 1:max_iter
        if T < t_final
            break
        end

        # Generate neighbor by Gaussian perturbation
        x_new = x_curr .+ step_size .* [lcg_randn() for _ in 1:n]
        f_new = f(x_new)
        delta = f_new - f_curr

        # Accept with Metropolis criterion
        if delta < 0.0 || lcg_rand() < exp(-delta / T)
            x_curr = x_new
            f_curr = f_new

            if f_curr < f_best
                x_best = copy(x_curr)
                f_best = f_curr
            end
        end

        T *= cooling_rate
    end

    return (x_min=x_best, f_min=f_best, iterations=max_iter)
end

# ===========================================================================
# 6. DIFFERENTIAL EVOLUTION
# ===========================================================================

"""
    differential_evolution(f::Function, bounds::Vector{Tuple{Float64,Float64}};
                            pop_size::Int=15, max_iter::Int=1000,
                            F::Float64=0.8, CR::Float64=0.9, seed::UInt64=42)
    -> NamedTuple{(:x_min, :f_min, :iterations, :converged)}

Minimize a function over a bounded domain using Differential Evolution.

DE is a stochastic population-based algorithm that is effective for
non-convex, discontinuous, and multimodal optimization problems.

For each target vector x_i, a trial vector is created:
1. Mutation: v = x_r1 + F * (x_r2 - x_r3) (rand/1 strategy)
2. Crossover: u_j = v_j if rand < CR, else x_i_j (binomial crossover)
3. Selection: keep u if f(u) ≤ f(x_i), otherwise keep x_i

# Arguments
- `f`: objective function (takes a vector, returns Float64).
- `bounds`: vector of (lo, hi) tuples specifying the search domain for each dimension.
- `pop_size`: population size (default 15 per dimension).
- `max_iter`: maximum number of generations.
- `F`: differential weight (mutation factor, default 0.8; ∈ [0, 2]).
- `CR`: crossover probability (default 0.9; ∈ [0, 1]).
- `seed`: random seed.

# Returns
Named tuple with `x_min`, `f_min`, `iterations`, `converged`.

# Examples
```julia
# Minimize Ackley function on [-5, 5]²
ackley(x) = -20*exp(-0.2*sqrt(0.5*(x[1]^2+x[2]^2))) -
             exp(0.5*(cos(2π*x[1])+cos(2π*x[2]))) + exp(1) + 20
result = differential_evolution(ackley, [(-5.0,5.0), (-5.0,5.0)])
result.x_min   # ≈ [0.0, 0.0]
```
"""
function differential_evolution(
    f::Function,
    bounds::Vector{Tuple{Float64,Float64}};
    pop_size::Int=15,
    max_iter::Int=1000,
    F::Float64=0.8,
    CR::Float64=0.9,
    seed::UInt64=42,
)
    n_dim = length(bounds)
    n_pop = pop_size

    # LCG random number generator
    state = Ref{UInt64}(seed)
    function lcg_rand()::Float64
        state[] = state[] * 6364136223846793005 + 1442695040888963407
        return Float64(state[] >> 11) / Float64(1 << 53)
    end
    function lcg_rand_int(lo::Int, hi::Int)::Int
        return lo + Int(floor(lcg_rand() * (hi - lo + 1)))
    end

    # Initialize population uniformly in bounds
    population = Matrix{Float64}(undef, n_pop, n_dim)
    for i in 1:n_pop, j in 1:n_dim
        lo, hi = bounds[j]
        population[i, j] = lo + lcg_rand() * (hi - lo)
    end

    fitness = [f(population[i, :]) for i in 1:n_pop]
    best_idx = argmin(fitness)
    converged = false

    for gen in 1:max_iter
        for i in 1:n_pop
            # Select 3 distinct random indices ≠ i
            idxs = collect(1:n_pop)
            filter!(x -> x != i, idxs)
            r1 = idxs[lcg_rand_int(1, length(idxs))]
            filter!(x -> x != r1, idxs)
            r2 = idxs[lcg_rand_int(1, length(idxs))]
            filter!(x -> x != r2, idxs)
            r3 = idxs[lcg_rand_int(1, length(idxs))]

            # Mutation
            v = population[r1, :] .+ F .* (population[r2, :] .- population[r3, :])

            # Clip to bounds
            for j in 1:n_dim
                lo, hi = bounds[j]
                v[j] = clamp(v[j], lo, hi)
            end

            # Binomial crossover
            j_rand = lcg_rand_int(1, n_dim)
            u = copy(population[i, :])
            for j in 1:n_dim
                if lcg_rand() < CR || j == j_rand
                    u[j] = v[j]
                end
            end

            # Selection
            f_u = f(u)
            if f_u <= fitness[i]
                population[i, :] = u
                fitness[i] = f_u
                if f_u < fitness[best_idx]
                    best_idx = i
                end
            end
        end

        if fitness[best_idx] < 1e-10
            converged = true
            return (x_min=population[best_idx, :], f_min=fitness[best_idx],
                    iterations=gen, converged=true)
        end
    end

    return (x_min=population[best_idx, :], f_min=fitness[best_idx],
            iterations=max_iter, converged=converged)
end

# ===========================================================================
# 7. POWELL'S METHOD
# ===========================================================================

"""
    powell_minimize(f::Function, x0::AbstractVector{Float64};
                    tol::Float64=1e-8, max_iter::Int=1000)
    -> NamedTuple{(:x_min, :f_min, :converged, :iterations)}

Minimize a function using Powell's conjugate direction method.

Powell's method does not require gradients. It iteratively minimizes along
a set of conjugate directions, updating the direction set to include the
most successful direction from each cycle.

# Arguments
- `f`: objective function.
- `x0`: initial guess (n-dimensional).
- `tol`: convergence tolerance on function value.
- `max_iter`: maximum number of outer cycles.

# Returns
Named tuple with `x_min`, `f_min`, `converged`, `iterations`.
"""
function powell_minimize(
    f::Function,
    x0::AbstractVector{Float64};
    tol::Float64=1e-8,
    max_iter::Int=1000,
)
    x = collect(Float64, x0)
    n = length(x)
    converged = false

    # Initial directions: coordinate axes
    directions = Matrix{Float64}(I, n, n)

    function line_min(p::Vector{Float64})::Float64
        # Golden section search along direction p from current x
        lo, hi = -1.0, 1.0
        # Expand bracket
        for _ in 1:50
            f_lo = f(x .+ lo .* p)
            f_hi = f(x .+ hi .* p)
            f_mid = f(x .+ ((lo + hi) / 2) .* p)
            if f_lo < f_mid && f_lo < f_hi
                lo *= 2.0
            elseif f_hi < f_mid && f_hi < f_lo
                hi *= 2.0
            else
                break
            end
        end

        phi = (sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)
        f1 = f(x .+ x1 .* p)
        f2 = f(x .+ x2 .* p)

        for _ in 1:100
            if abs(b - a) < tol
                break
            end
            if f1 < f2
                b = x2; x2 = x1; f2 = f1
                x1 = b - phi * (b - a)
                f1 = f(x .+ x1 .* p)
            else
                a = x1; x1 = x2; f1 = f2
                x2 = a + phi * (b - a)
                f2 = f(x .+ x2 .* p)
            end
        end
        return (a + b) / 2.0
    end

    f_val = f(x)

    for iter in 1:max_iter
        x_start = copy(x)
        f_start = f_val
        max_decrease = 0.0
        max_dir_idx = 1

        # Minimize along each direction
        for i in 1:n
            p = directions[:, i]
            f_before = f(x)
            alpha = line_min(p)
            x .+= alpha .* p
            f_after = f(x)
            decrease = f_before - f_after
            if decrease > max_decrease
                max_decrease = decrease
                max_dir_idx = i
            end
        end

        f_val = f(x)

        # Convergence check
        if 2.0 * abs(f_start - f_val) < tol * (abs(f_start) + abs(f_val) + eps(Float64))
            converged = true
            break
        end

        # Update direction set: replace the direction of maximum decrease
        new_dir = x .- x_start
        dir_norm = norm(new_dir)
        if dir_norm > eps(Float64)
            directions[:, max_dir_idx] = new_dir ./ dir_norm
        end
    end

    return (x_min=x, f_min=f_val, converged=converged, iterations=max_iter)
end

# ===========================================================================
# 8. PENALTY METHOD FOR CONSTRAINED OPTIMIZATION
# ===========================================================================

"""
    penalty_minimize(f::Function, constraints::Vector{Function},
                     x0::AbstractVector{Float64};
                     tol::Float64=1e-6, max_iter::Int=1000,
                     mu_init::Float64=1.0, mu_factor::Float64=10.0,
                     inner_method::Symbol=:nelder_mead)
    -> NamedTuple{(:x_min, :f_min, :constraint_violation, :converged)}

Minimize a constrained problem using the exterior penalty method.

Problem form:
  min f(x)  subject to  gᵢ(x) ≤ 0 for all i

The penalty function is:
  P(x, μ) = f(x) + μ * Σ max(0, gᵢ(x))²

The penalty parameter μ is increased geometrically until feasibility is achieved.

# Arguments
- `f`: objective function.
- `constraints`: vector of constraint functions gᵢ(x); constraint is gᵢ(x) ≤ 0.
- `x0`: initial guess (need not be feasible).
- `tol`: feasibility tolerance.
- `max_iter`: maximum outer iterations.
- `mu_init`: initial penalty parameter (default 1.0).
- `mu_factor`: penalty growth factor (default 10.0).
- `inner_method`: inner minimization method (`:nelder_mead` or `:coordinate_descent`).

# Returns
Named tuple with `x_min`, `f_min`, `constraint_violation`, `converged`.
"""
function penalty_minimize(
    f::Function,
    constraints::Vector{Function},
    x0::AbstractVector{Float64};
    tol::Float64=1e-6,
    max_iter::Int=1000,
    mu_init::Float64=1.0,
    mu_factor::Float64=10.0,
    inner_method::Symbol=:nelder_mead,
)
    x = collect(Float64, x0)
    mu = mu_init
    converged = false

    function penalty_f(xv, mu_val)
        violation = sum(max(0.0, g(xv))^2 for g in constraints)
        return f(xv) + mu_val * violation
    end

    function total_violation(xv)
        return sum(max(0.0, g(xv)) for g in constraints)
    end

    for outer in 1:max_iter
        # Inner minimization
        pen_f = xv -> penalty_f(xv, mu)

        result = if inner_method == :nelder_mead
            _nelder_mead_inner(pen_f, x; tol=1e-8, max_iter=10000)
        else
            coordinate_descent(pen_f, x; tol=1e-8, max_iter=10000)
        end

        x = result.x_min

        viol = total_violation(x)
        if viol < tol
            converged = true
            break
        end

        mu *= mu_factor
    end

    return (x_min=x, f_min=f(x), constraint_violation=total_violation(x), converged=converged)
end

"""Nelder-Mead inner optimizer (minimal, for penalty method use)."""
function _nelder_mead_inner(f, x0; tol=1e-8, max_iter=10000)
    n = length(x0)
    step = 0.05
    simplex = [copy(x0) for _ in 1:(n + 1)]
    for i in 1:n
        simplex[i + 1][i] += (abs(x0[i]) > 1e-8 ? 0.05 * abs(x0[i]) : step)
    end

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    f_vals = [f(s) for s in simplex]

    for _ in 1:max_iter
        order = sortperm(f_vals)
        simplex = simplex[order]; f_vals = f_vals[order]

        if (f_vals[end] - f_vals[1]) < tol * (abs(f_vals[1]) + 1.0)
            break
        end

        centroid = sum(simplex[1:end-1]) / n
        x_r = centroid + alpha * (centroid - simplex[end])
        f_r = f(x_r)

        if f_r < f_vals[1]
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            simplex[end] = f_e < f_r ? x_e : x_r
            f_vals[end] = f_e < f_r ? f_e : f_r
        elseif f_r < f_vals[end]
            simplex[end] = x_r; f_vals[end] = f_r
        else
            x_c = centroid + rho * (simplex[end] - centroid)
            f_c = f(x_c)
            if f_c < f_vals[end]
                simplex[end] = x_c; f_vals[end] = f_c
            else
                best = simplex[1]
                for i in 2:(n + 1)
                    simplex[i] = best + sigma * (simplex[i] - best)
                    f_vals[i] = f(simplex[i])
                end
            end
        end
    end

    best_idx = argmin(f_vals)
    return (x_min=simplex[best_idx], f_min=f_vals[best_idx])
end

# ===========================================================================
# 9. MULTI-START OPTIMIZATION
# ===========================================================================

"""
    multi_start(f::Function, bounds::Vector{Tuple{Float64,Float64}};
                n_starts::Int=20, method::Symbol=:nelder_mead,
                seed::UInt64=42, kwargs...)
    -> NamedTuple{(:x_min, :f_min, :n_restarts)}

Minimize a function from multiple random starting points.

This improves the chance of finding the global optimum for multimodal functions
by running the local optimizer from `n_starts` uniformly sampled initial points.

# Arguments
- `f`: objective function.
- `bounds`: search domain as vector of (lo, hi) tuples.
- `n_starts`: number of starting points (default 20).
- `method`: local optimizer (`:nelder_mead`, `:coordinate_descent`, or `:powell`).
- `seed`: random seed.
- `kwargs`: additional arguments passed to the local optimizer.

# Returns
Named tuple with `x_min` (best minimizer), `f_min` (best objective), `n_restarts`.
"""
function multi_start(
    f::Function,
    bounds::Vector{Tuple{Float64,Float64}};
    n_starts::Int=20,
    method::Symbol=:nelder_mead,
    seed::UInt64=42,
    kwargs...,
)
    n_dim = length(bounds)
    state = Ref{UInt64}(seed)

    function lcg_rand()::Float64
        state[] = state[] * 6364136223846793005 + 1442695040888963407
        return Float64(state[] >> 11) / Float64(1 << 53)
    end

    best_x = zeros(Float64, n_dim)
    best_f = Inf

    for _ in 1:n_starts
        x0 = [lo + lcg_rand() * (hi - lo) for (lo, hi) in bounds]

        result = if method == :nelder_mead
            _nelder_mead_inner(f, x0; kwargs...)
        elseif method == :coordinate_descent
            coordinate_descent(f, x0; kwargs...)
        elseif method == :powell
            powell_minimize(f, x0; kwargs...)
        else
            throw(ArgumentError("multi_start: unknown method '$method'"))
        end

        if result.f_min < best_f
            best_f = result.f_min
            best_x = result.x_min
        end
    end

    return (x_min=best_x, f_min=best_f, n_restarts=n_starts)
end

# ===========================================================================
# EXPORTS
# ===========================================================================

export gradient_descent
export lbfgs
export adam
export coordinate_descent
export simulated_annealing
export differential_evolution
export powell_minimize
export penalty_minimize
export multi_start

end  # module ExtendedOptimize
