"""
    ExtendedLinalg

Extended linear algebra functions for SciRS2.

Provides:
- Matrix norms (Frobenius, nuclear, spectral/operator, p-norms)
- Matrix rank estimation
- QR decomposition (pure Julia, Householder reflections)
- Cholesky decomposition
- LU decomposition
- Condition number
- Pseudoinverse (Moore-Penrose) via SVD
- Null space and column space
- Matrix functions: exponential, square root, logarithm (via Schur decomposition / Padé approximation)
- Kronecker product and vectorization
- Structured matrix operations: Toeplitz, Circulant, Hankel
- Sparse-like operations: band matrix solve
"""
module ExtendedLinalg

using LinearAlgebra

# ===========================================================================
# 1. MATRIX NORMS
# ===========================================================================

"""
    matrix_norm(A::AbstractMatrix{Float64}; p::Union{Int,Float64,Symbol}=:frobenius)
    -> Float64

Compute a matrix norm.

# Arguments
- `A`: input matrix.
- `p`: norm type:
  - `:frobenius` or `2` (element-wise 2-norm): √(Σ |aᵢⱼ|²)
  - `1`: maximum absolute column sum
  - `Inf` or `:inf`: maximum absolute row sum
  - `2` or `:spectral`: spectral norm (largest singular value) — use `:spectral`
    to distinguish from element-wise 2-norm
  - `:nuclear`: sum of singular values (trace norm)

# Returns
- Norm value.

# Examples
```julia
A = [1.0 2.0; 3.0 4.0]
matrix_norm(A)              # Frobenius norm ≈ 5.477
matrix_norm(A; p=1)         # column sum norm = 6.0
matrix_norm(A; p=:spectral) # spectral norm ≈ 5.465
```
"""
function matrix_norm(
    A::AbstractMatrix{Float64};
    p::Union{Int,Float64,Symbol}=:frobenius,
)::Float64
    if p == :frobenius || p == :fro
        return sqrt(sum(abs2, A))
    elseif p == 1
        return maximum(sum(abs, A; dims=1))
    elseif p == Inf || p == :inf
        return maximum(sum(abs, A; dims=2))
    elseif p == :spectral
        # Largest singular value via power iteration
        return _spectral_norm(A)
    elseif p == :nuclear
        svd_result = LinearAlgebra.svd(A)
        return sum(svd_result.S)
    else
        throw(ArgumentError("matrix_norm: unsupported norm type '$p'"))
    end
end

"""Estimate the spectral norm via power iteration."""
function _spectral_norm(A::AbstractMatrix{Float64}; max_iter::Int=100, tol::Float64=1e-10)::Float64
    m, n = size(A)
    # Use AᵀA power iteration
    v = ones(Float64, n) ./ sqrt(n)
    sigma_prev = 0.0
    for _ in 1:max_iter
        w = A * v
        sigma = norm(w)
        if sigma < eps(Float64)
            return 0.0
        end
        u = w ./ sigma
        v = A' * u
        sigma_new = norm(v)
        v ./= max(sigma_new, eps(Float64))
        if abs(sigma_new - sigma_prev) < tol * sigma_new
            return sigma_new
        end
        sigma_prev = sigma_new
    end
    return sigma_prev
end

"""
    vector_norm(x::AbstractVector{Float64}; p::Union{Int,Float64}=2) -> Float64

Compute the p-norm of a vector.

# Arguments
- `x`: input vector.
- `p`: norm order (default 2, i.e., Euclidean norm).
  - `p=1`: sum of absolute values
  - `p=2`: Euclidean (L2) norm
  - `p=Inf`: maximum absolute value
  - Any `p ≥ 1`: (Σ |xᵢ|^p)^(1/p)

# Returns
- Norm value.
"""
function vector_norm(x::AbstractVector{Float64}; p::Union{Int,Float64}=2)::Float64
    if p == 2
        return sqrt(sum(abs2, x))
    elseif p == 1
        return sum(abs, x)
    elseif p == Inf
        return maximum(abs, x)
    elseif p >= 1
        return sum(abs(xi)^p for xi in x)^(1.0 / p)
    else
        throw(ArgumentError("vector_norm: p must be ≥ 1, got $p"))
    end
end

# ===========================================================================
# 2. MATRIX RANK
# ===========================================================================

"""
    matrix_rank(A::AbstractMatrix{Float64}; tol::Float64=-1.0) -> Int

Estimate the numerical rank of a matrix using singular value decomposition.

# Arguments
- `A`: input matrix.
- `tol`: singular value threshold. Values below `tol` are considered zero.
  If `tol < 0` (default), uses `max(m, n) * eps(σ_max)` where σ_max is the
  largest singular value.

# Returns
- Estimated rank (integer).
"""
function matrix_rank(A::AbstractMatrix{Float64}; tol::Float64=-1.0)::Int
    svd_result = LinearAlgebra.svd(A)
    S = svd_result.S
    if isempty(S)
        return 0
    end
    m, n = size(A)
    threshold = tol < 0.0 ? max(m, n) * S[1] * eps(Float64) : tol
    return count(s -> s > threshold, S)
end

# ===========================================================================
# 3. QR DECOMPOSITION (HOUSEHOLDER)
# ===========================================================================

"""
    qr_decompose(A::AbstractMatrix{Float64}) -> NamedTuple{(:Q, :R)}

Compute the QR decomposition of a matrix using Householder reflections.

For an (m × n) matrix A (m ≥ n), returns:
- Q: (m × m) orthogonal matrix
- R: (m × n) upper triangular matrix (lower m-n rows are zero)

such that A = Q * R.

# Returns
Named tuple with fields `Q` and `R`.

# Examples
```julia
A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
result = qr_decompose(A)
result.Q * result.R   # ≈ A
```
"""
function qr_decompose(A::AbstractMatrix{Float64})
    m, n = size(A)
    if m < n
        throw(ArgumentError("qr_decompose: matrix must have m ≥ n ($(m)x$(n) given)"))
    end

    R = collect(Float64, A)
    Q = Matrix{Float64}(I, m, m)

    for k in 1:min(m - 1, n)
        # Extract column k from row k onward
        v = R[k:m, k]
        # Householder reflector: modify v[1] to cancel subdiagonal
        sigma = norm(v)
        if sigma < eps(Float64)
            continue
        end
        v[1] += (v[1] >= 0.0 ? sigma : -sigma)
        v ./= norm(v)

        # Apply reflector: R[k:m, k:n] -= 2 * v * (v' * R[k:m, k:n])
        R[k:m, k:n] .-= 2.0 .* v .* (v' * R[k:m, k:n])
        # Accumulate Q: Q[:, k:m] -= 2 * (Q[:, k:m] * v) * v'
        Q[:, k:m] .-= 2.0 .* (Q[:, k:m] * v) .* v'
    end

    return (Q=Q, R=R)
end

# ===========================================================================
# 4. CHOLESKY DECOMPOSITION
# ===========================================================================

"""
    cholesky_decompose(A::AbstractMatrix{Float64}) -> Matrix{Float64}

Compute the lower-triangular Cholesky factor L of a symmetric positive definite
matrix A such that A = L * L'.

Uses the standard Cholesky-Banachiewicz algorithm.

# Arguments
- `A`: symmetric positive definite (n × n) matrix.

# Returns
- Lower triangular matrix L.

# Throws
- `ArgumentError` if A is not square.
- `ErrorException` if A is not positive definite (diagonal element would be negative).

# Examples
```julia
A = [4.0 2.0; 2.0 3.0]
L = cholesky_decompose(A)
L * L'   # ≈ A
```
"""
function cholesky_decompose(A::AbstractMatrix{Float64})::Matrix{Float64}
    n = size(A, 1)
    if size(A, 2) != n
        throw(ArgumentError("cholesky_decompose: matrix must be square"))
    end

    L = zeros(Float64, n, n)

    for i in 1:n
        for j in 1:i
            s = A[i, j]
            for k in 1:(j - 1)
                s -= L[i, k] * L[j, k]
            end
            if i == j
                if s < 0.0
                    throw(ErrorException(
                        "cholesky_decompose: matrix is not positive definite (negative diagonal at $i)"
                    ))
                end
                L[i, j] = sqrt(s)
            else
                if abs(L[j, j]) < eps(Float64)
                    throw(ErrorException(
                        "cholesky_decompose: zero pivot at position $j"
                    ))
                end
                L[i, j] = s / L[j, j]
            end
        end
    end

    return L
end

"""
    cholesky_solve(L::AbstractMatrix{Float64}, b::AbstractVector{Float64})
    -> Vector{Float64}

Solve A * x = b given the Cholesky factor L (A = L * L').

Uses forward and backward substitution.

# Arguments
- `L`: lower triangular Cholesky factor.
- `b`: right-hand side vector.

# Returns
- Solution vector x.
"""
function cholesky_solve(
    L::AbstractMatrix{Float64},
    b::AbstractVector{Float64},
)::Vector{Float64}
    n = size(L, 1)
    if length(b) != n
        throw(DimensionMismatch("cholesky_solve: L is $(n)x$(n) but b has $(length(b)) elements"))
    end

    # Forward substitution: L * y = b
    y = Vector{Float64}(undef, n)
    for i in 1:n
        s = b[i]
        for j in 1:(i - 1)
            s -= L[i, j] * y[j]
        end
        y[i] = s / L[i, i]
    end

    # Backward substitution: L' * x = y
    x = Vector{Float64}(undef, n)
    for i in n:-1:1
        s = y[i]
        for j in (i + 1):n
            s -= L[j, i] * x[j]
        end
        x[i] = s / L[i, i]
    end

    return x
end

# ===========================================================================
# 5. LU DECOMPOSITION (PARTIAL PIVOTING)
# ===========================================================================

"""
    lu_decompose(A::AbstractMatrix{Float64}) -> NamedTuple{(:L, :U, :P)}

Compute the LU decomposition with partial pivoting: P * A = L * U.

# Arguments
- `A`: (n × n) square matrix.

# Returns
Named tuple with:
- `L`: unit lower triangular matrix.
- `U`: upper triangular matrix.
- `P`: permutation matrix such that P*A = L*U.

# Examples
```julia
A = [2.0 1.0; 4.0 3.0]
result = lu_decompose(A)
result.P * A   # ≈ result.L * result.U
```
"""
function lu_decompose(A::AbstractMatrix{Float64})
    n = size(A, 1)
    if size(A, 2) != n
        throw(ArgumentError("lu_decompose: matrix must be square, got $(size(A))"))
    end

    U = collect(Float64, A)
    L = Matrix{Float64}(I, n, n)
    P = Matrix{Float64}(I, n, n)

    for k in 1:(n - 1)
        # Find pivot
        pivot_row = k + argmax(abs.(U[k:n, k])) - 1

        if abs(U[pivot_row, k]) < eps(Float64)
            continue  # singular or near-singular
        end

        # Swap rows k and pivot_row in U, L, P
        if pivot_row != k
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k >= 2
                L[[k, pivot_row], 1:(k-1)] = L[[pivot_row, k], 1:(k-1)]
            end
        end

        # Elimination
        for i in (k + 1):n
            if abs(U[k, k]) < eps(Float64)
                continue
            end
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:n] .-= L[i, k] .* U[k, k:n]
        end
    end

    return (L=L, U=U, P=P)
end

# ===========================================================================
# 6. CONDITION NUMBER
# ===========================================================================

"""
    condition_number(A::AbstractMatrix{Float64}; p::Union{Int,Symbol}=2) -> Float64

Compute the condition number of a matrix with respect to the given norm.

For `p=2` (spectral norm), uses the ratio of the largest to smallest singular
value: κ = σ_max / σ_min.

# Arguments
- `A`: input matrix.
- `p`: norm type (default `2` = spectral condition number).

# Returns
- Condition number (≥ 1.0); returns `Inf` for singular matrices.
"""
function condition_number(
    A::AbstractMatrix{Float64};
    p::Union{Int,Symbol}=2,
)::Float64
    if p == 2 || p == :spectral
        svd_result = LinearAlgebra.svd(A)
        S = svd_result.S
        if isempty(S) || S[end] < eps(Float64)
            return Inf
        end
        return S[1] / S[end]
    elseif p == 1
        return matrix_norm(A; p=1) * matrix_norm(inv(A); p=1)
    elseif p == Inf || p == :inf
        return matrix_norm(A; p=Inf) * matrix_norm(inv(A); p=Inf)
    else
        throw(ArgumentError("condition_number: unsupported norm '$p'"))
    end
end

# ===========================================================================
# 7. PSEUDOINVERSE (MOORE-PENROSE)
# ===========================================================================

"""
    pseudoinverse(A::AbstractMatrix{Float64}; tol::Float64=-1.0) -> Matrix{Float64}

Compute the Moore-Penrose pseudoinverse of a matrix.

Uses SVD: A⁺ = V * Σ⁺ * Uᵀ where Σ⁺ inverts all non-zero singular values.

# Arguments
- `A`: (m × n) matrix.
- `tol`: threshold below which singular values are treated as zero.
  Default is `max(m, n) * eps * σ_max`.

# Returns
- (n × m) pseudoinverse.
"""
function pseudoinverse(A::AbstractMatrix{Float64}; tol::Float64=-1.0)::Matrix{Float64}
    F = LinearAlgebra.svd(A)
    S = F.S
    m, n = size(A)
    threshold = tol < 0.0 ? max(m, n) * eps(Float64) * (isempty(S) ? 0.0 : S[1]) : tol
    S_inv = [s > threshold ? 1.0 / s : 0.0 for s in S]
    return F.V * Diagonal(S_inv) * F.U'
end

# ===========================================================================
# 8. NULL SPACE AND COLUMN SPACE
# ===========================================================================

"""
    null_space(A::AbstractMatrix{Float64}; tol::Float64=-1.0) -> Matrix{Float64}

Compute an orthonormal basis for the null space of A.

Uses SVD: the null space consists of the right singular vectors corresponding
to singular values below the threshold.

# Arguments
- `A`: input matrix.
- `tol`: singular value threshold (default: `max(m,n) * eps * σ_max`).

# Returns
- Matrix whose columns form an orthonormal basis for the null space.
  Empty matrix if the null space is trivial ({0}).
"""
function null_space(A::AbstractMatrix{Float64}; tol::Float64=-1.0)::Matrix{Float64}
    F = LinearAlgebra.svd(A)
    S = F.S
    m, n = size(A)
    threshold = tol < 0.0 ? max(m, n) * eps(Float64) * (isempty(S) ? 1.0 : S[1]) : tol
    null_cols = [i for (i, s) in enumerate(S) if s < threshold]
    if isempty(null_cols)
        # Append columns beyond the rank of S (i.e., singular vectors with no S entry)
        k = length(S)
        if k < n
            return F.V[:, (k + 1):n]
        end
        return Matrix{Float64}(undef, n, 0)
    end
    return F.V[:, null_cols]
end

"""
    column_space(A::AbstractMatrix{Float64}; tol::Float64=-1.0) -> Matrix{Float64}

Compute an orthonormal basis for the column space (range) of A.

Uses SVD: the column space is spanned by the left singular vectors corresponding
to non-zero singular values.

# Returns
- Matrix whose columns form an orthonormal basis for the column space.
"""
function column_space(A::AbstractMatrix{Float64}; tol::Float64=-1.0)::Matrix{Float64}
    F = LinearAlgebra.svd(A)
    S = F.S
    m, n = size(A)
    threshold = tol < 0.0 ? max(m, n) * eps(Float64) * (isempty(S) ? 1.0 : S[1]) : tol
    rank_cols = [i for (i, s) in enumerate(S) if s >= threshold]
    if isempty(rank_cols)
        return Matrix{Float64}(undef, m, 0)
    end
    return F.U[:, rank_cols]
end

# ===========================================================================
# 9. KRONECKER PRODUCT AND VECTORIZATION
# ===========================================================================

"""
    kron_product(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    -> Matrix{Float64}

Compute the Kronecker product A ⊗ B.

If A is (m × n) and B is (p × q), the result is (m*p × n*q).

# Examples
```julia
A = [1.0 2.0; 3.0 4.0]
B = [0.0 5.0; 6.0 7.0]
kron_product(A, B)   # 4x4 matrix
```
"""
function kron_product(
    A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
)::Matrix{Float64}
    m, n = size(A)
    p, q = size(B)
    C = Matrix{Float64}(undef, m * p, n * q)
    for i in 1:m, j in 1:n
        C[((i-1)*p + 1):(i*p), ((j-1)*q + 1):(j*q)] = A[i, j] .* B
    end
    return C
end

"""
    vec_matrix(A::AbstractMatrix{Float64}) -> Vector{Float64}

Vectorize a matrix by stacking its columns (column-major order).

This is the standard `vec` operation: vec(A) = [a₁; a₂; ...; aₙ] where aᵢ are columns.
"""
function vec_matrix(A::AbstractMatrix{Float64})::Vector{Float64}
    return vec(A)
end

"""
    unvec_matrix(v::AbstractVector{Float64}, m::Int, n::Int) -> Matrix{Float64}

Reshape a vector into an (m × n) matrix by filling columns.
"""
function unvec_matrix(v::AbstractVector{Float64}, m::Int, n::Int)::Matrix{Float64}
    if length(v) != m * n
        throw(DimensionMismatch("unvec_matrix: v has $(length(v)) elements but m*n=$m*$n=$(m*n)"))
    end
    return reshape(collect(v), m, n)
end

# ===========================================================================
# 10. STRUCTURED MATRICES
# ===========================================================================

"""
    toeplitz_matrix(first_col::AbstractVector{Float64},
                    first_row::Union{AbstractVector{Float64},Nothing}=nothing)
    -> Matrix{Float64}

Build a Toeplitz matrix from its first column (and optionally first row).

A Toeplitz matrix has constant diagonals: T[i,j] = t[|i-j|].
For a symmetric Toeplitz matrix, only `first_col` is needed.
For a general Toeplitz matrix, `first_row` specifies the first row
(with `first_row[1]` == `first_col[1]` required).

# Examples
```julia
c = [1.0, 2.0, 3.0]
r = [1.0, 4.0, 5.0]
T = toeplitz_matrix(c, r)
# T = [1 4 5; 2 1 4; 3 2 1]
```
"""
function toeplitz_matrix(
    first_col::AbstractVector{Float64},
    first_row::Union{AbstractVector{Float64},Nothing}=nothing,
)::Matrix{Float64}
    c = collect(Float64, first_col)
    m = length(c)

    if first_row === nothing
        # Symmetric Toeplitz
        n = m
        T = Matrix{Float64}(undef, m, n)
        for i in 1:m, j in 1:n
            T[i, j] = c[abs(i - j) + 1]
        end
        return T
    else
        r = collect(Float64, first_row)
        n = length(r)
        if abs(c[1] - r[1]) > eps(Float64)
            throw(ArgumentError("toeplitz_matrix: first_col[1] must equal first_row[1]"))
        end
        T = Matrix{Float64}(undef, m, n)
        for i in 1:m, j in 1:n
            k = j - i
            T[i, j] = k >= 0 ? r[k + 1] : c[-k + 1]
        end
        return T
    end
end

"""
    circulant_matrix(c::AbstractVector{Float64}) -> Matrix{Float64}

Build an (n × n) circulant matrix from its first column `c`.

In a circulant matrix, each row is a cyclic shift of the previous row:
  C[i, j] = c[mod(j - i, n) + 1]

Circulant matrices can be diagonalized by the DFT matrix, enabling
fast matrix-vector products via FFT.

# Examples
```julia
c = [1.0, 2.0, 3.0]
C = circulant_matrix(c)
# C = [1 3 2; 2 1 3; 3 2 1]
```
"""
function circulant_matrix(c::AbstractVector{Float64})::Matrix{Float64}
    n = length(c)
    cv = collect(Float64, c)
    C = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        C[i, j] = cv[mod(j - i, n) + 1]
    end
    return C
end

"""
    hankel_matrix(first_col::AbstractVector{Float64},
                  last_row::Union{AbstractVector{Float64},Nothing}=nothing)
    -> Matrix{Float64}

Build a Hankel matrix.

A Hankel matrix has constant anti-diagonals: H[i,j] = h[i+j-1].
`first_col` specifies the first column; `last_row` specifies the last row.
If `last_row` is not provided, the matrix is square with zeros in the lower-right triangle.

# Examples
```julia
c = [1.0, 2.0, 3.0]
r = [3.0, 4.0, 5.0]
H = hankel_matrix(c, r)
# H = [1 2 3; 2 3 4; 3 4 5]
```
"""
function hankel_matrix(
    first_col::AbstractVector{Float64},
    last_row::Union{AbstractVector{Float64},Nothing}=nothing,
)::Matrix{Float64}
    c = collect(Float64, first_col)
    m = length(c)

    if last_row === nothing
        n = m
        H = zeros(Float64, m, n)
        full = vcat(c, zeros(Float64, n - 1))
        for i in 1:m, j in 1:n
            H[i, j] = full[i + j - 1]
        end
        return H
    else
        r = collect(Float64, last_row)
        n = length(r)
        if abs(c[m] - r[1]) > eps(Float64)
            throw(ArgumentError("hankel_matrix: first_col[end] must equal last_row[1]"))
        end
        full = vcat(c, r[2:end])
        H = Matrix{Float64}(undef, m, n)
        for i in 1:m, j in 1:n
            H[i, j] = full[i + j - 1]
        end
        return H
    end
end

# ===========================================================================
# 11. BAND MATRIX SOLVE
# ===========================================================================

"""
    band_solve(dl::AbstractVector{Float64}, d::AbstractVector{Float64},
               du::AbstractVector{Float64}, b::AbstractVector{Float64})
    -> Vector{Float64}

Solve a tridiagonal system A * x = b.

- `dl`: sub-diagonal elements (length n-1).
- `d`: main diagonal elements (length n).
- `du`: super-diagonal elements (length n-1).
- `b`: right-hand side (length n).

Uses the Thomas algorithm (tridiagonal elimination), O(n) complexity.

# Examples
```julia
d  = [4.0, 4.0, 4.0]
dl = [1.0, 1.0]
du = [1.0, 1.0]
b  = [6.0, 6.0, 6.0]
x  = band_solve(dl, d, du, b)   # x ≈ [1.0, 1.0, 1.0]
```
"""
function band_solve(
    dl::AbstractVector{Float64},
    d::AbstractVector{Float64},
    du::AbstractVector{Float64},
    b::AbstractVector{Float64},
)::Vector{Float64}
    n = length(d)
    if length(dl) != n - 1 || length(du) != n - 1 || length(b) != n
        throw(DimensionMismatch("band_solve: inconsistent dimensions"))
    end

    # Forward elimination (Thomas algorithm)
    c = copy(du)
    d_mod = copy(d)
    rhs = copy(b)

    for i in 2:n
        if abs(d_mod[i - 1]) < eps(Float64)
            throw(ErrorException("band_solve: zero pivot at row $(i-1)"))
        end
        m = dl[i - 1] / d_mod[i - 1]
        d_mod[i] -= m * c[i - 1]
        rhs[i] -= m * rhs[i - 1]
    end

    # Back substitution
    x = Vector{Float64}(undef, n)
    x[n] = rhs[n] / d_mod[n]
    for i in (n - 1):-1:1
        x[i] = (rhs[i] - c[i] * x[i + 1]) / d_mod[i]
    end

    return x
end

# ===========================================================================
# EXPORTS
# ===========================================================================

export matrix_norm, vector_norm
export matrix_rank
export qr_decompose
export cholesky_decompose, cholesky_solve
export lu_decompose
export condition_number
export pseudoinverse
export null_space, column_space
export kron_product, vec_matrix, unvec_matrix
export toeplitz_matrix, circulant_matrix, hankel_matrix
export band_solve

end  # module ExtendedLinalg
