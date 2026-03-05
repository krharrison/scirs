"""
    SciRS2

Julia wrapper package for the SciRS2 scientific computing library.

This module provides Julia-native interfaces to the SciRS2 Rust library via
the C FFI layer. All functions return Julia arrays and scalars; memory
management of the underlying C allocations is handled transparently.

# Library location

The shared library is expected at `joinpath(@__DIR__, "..", "deps", "libscirs2_core")`.
Build it first with the provided `scripts/build_shared_lib.sh` script.

# Thread safety

The underlying Rust functions are thread-safe. Julia multi-threading (via
`Threads.@threads`) is safe as long as distinct output buffers are used.

# Error handling

Every C function returns a `SciResult` struct whose `success` field indicates
whether the call succeeded. On failure the `error_msg` field holds a pointer to
a NUL-terminated error string allocated by Rust; this module converts that
string to a Julia `ErrorException` and frees the C memory automatically.
"""
module SciRS2

using LinearAlgebra

# ---------------------------------------------------------------------------
# Library path resolution
# ---------------------------------------------------------------------------

"""
Resolve the path to the SciRS2 shared library depending on the operating system.

On macOS the extension is `.dylib`; on Linux `.so`; on Windows `.dll`.
"""
function _lib_path()::String
    deps_dir = joinpath(@__DIR__, "..", "deps")
    base = joinpath(deps_dir, "libscirs2_core")
    if Sys.isapple()
        return base * ".dylib"
    elseif Sys.islinux()
        return base * ".so"
    elseif Sys.iswindows()
        return base * ".dll"
    else
        error("SciRS2: unsupported operating system: $(Sys.KERNEL)")
    end
end

const _LIBPATH = _lib_path()

# ---------------------------------------------------------------------------
# C ABI structure definitions
# ---------------------------------------------------------------------------

"""
`SciVector` mirrors the Rust `#[repr(C)]` struct:

```c
struct SciVector { double* data; size_t len; };
```

Memory owned by Rust must be freed with `sci_vector_free`.
"""
struct SciVector
    data::Ptr{Cdouble}
    len::Csize_t
end

SciVector() = SciVector(C_NULL, 0)

"""
`SciMatrix` mirrors the Rust `#[repr(C)]` struct (row-major):

```c
struct SciMatrix { double* data; size_t rows; size_t cols; };
```

Memory owned by Rust must be freed with `sci_matrix_free`.
"""
struct SciMatrix
    data::Ptr{Cdouble}
    rows::Csize_t
    cols::Csize_t
end

SciMatrix() = SciMatrix(C_NULL, 0, 0)

"""
`SciComplexVector` holds split real/imaginary arrays:

```c
struct SciComplexVector { double* real; double* imag; size_t len; };
```

Memory owned by Rust must be freed with `sci_complex_vector_free`.
"""
struct SciComplexVector
    real::Ptr{Cdouble}
    imag::Ptr{Cdouble}
    len::Csize_t
end

SciComplexVector() = SciComplexVector(C_NULL, C_NULL, 0)

"""
`SciSvdResult` holds the three components of a singular value decomposition:

```c
struct SciSvdResult { SciMatrix u; SciVector s; SciMatrix vt; };
```

Free with `sci_svd_result_free`.
"""
struct SciSvdResult
    u::SciMatrix
    s::SciVector
    vt::SciMatrix
end

SciSvdResult() = SciSvdResult(SciMatrix(), SciVector(), SciMatrix())

"""
`SciEigResult` holds eigenvalues (complex) and eigenvectors (real matrix):

```c
struct SciEigResult { SciComplexVector eigenvalues; SciMatrix eigenvectors; };
```

Free with `sci_eig_result_free`.
"""
struct SciEigResult
    eigenvalues::SciComplexVector
    eigenvectors::SciMatrix
end

SciEigResult() = SciEigResult(SciComplexVector(), SciMatrix())

"""
`SciResult` is returned by every C FFI function.

```c
struct SciResult { bool success; const char* error_msg; };
```

`success == true` means the operation succeeded and output parameters are valid.
`success == false` means `error_msg` holds an error string allocated by Rust.
"""
struct SciResult
    success::Bool
    error_msg::Ptr{UInt8}
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
Check a `SciResult` and either return normally or throw an `ErrorException`.

If the result is a failure, the Rust-allocated error string is read into Julia,
then freed via `sci_free_error`, and finally thrown as an `ErrorException`.
"""
function _check(r::SciResult, context::AbstractString)
    if r.success
        return
    end
    # Read the error message before freeing it.
    msg = if r.error_msg == C_NULL
        "$(context): unknown error (null error_msg)"
    else
        unsafe_string(r.error_msg)
    end
    # Free the Rust-allocated string.
    ccall((:sci_free_error, _LIBPATH), Cvoid, (Ptr{UInt8},), r.error_msg)
    error("$(context): $(msg)")
end

"""
Copy a `SciVector` allocated by Rust into a Julia `Vector{Float64}`, then free
the Rust allocation.
"""
function _consume_sci_vector(sv::SciVector)::Vector{Float64}
    n = Int(sv.len)
    result = Vector{Float64}(undef, n)
    if n > 0 && sv.data != C_NULL
        unsafe_copyto!(pointer(result), sv.data, n)
    end
    # Free the Rust-owned buffer.
    sv_ref = Ref(sv)
    ccall((:sci_vector_free, _LIBPATH), Cvoid, (Ptr{SciVector},), sv_ref)
    return result
end

"""
Copy a `SciMatrix` (row-major) allocated by Rust into a Julia `Matrix{Float64}`,
then free the Rust allocation.

Julia matrices are column-major, so we transpose while copying to present the
data in the expected orientation.
"""
function _consume_sci_matrix(sm::SciMatrix)::Matrix{Float64}
    rows = Int(sm.rows)
    cols = Int(sm.cols)
    total = rows * cols
    # Read the raw row-major data.
    raw = Vector{Float64}(undef, total)
    if total > 0 && sm.data != C_NULL
        unsafe_copyto!(pointer(raw), sm.data, total)
    end
    # Free the Rust-owned buffer.
    sm_ref = Ref(sm)
    ccall((:sci_matrix_free, _LIBPATH), Cvoid, (Ptr{SciMatrix},), sm_ref)
    # Reshape into a Julia matrix (column-major), which means the row-major
    # Rust data becomes transposed. We take the transpose to obtain the
    # mathematically correct matrix.
    if total == 0
        return Matrix{Float64}(undef, rows, cols)
    end
    # `reshape` produces a (cols, rows) view of row-major data, which when
    # interpreted in column-major order is the transpose. `collect(m')` gives
    # the (rows, cols) result.
    return collect(reshape(raw, cols, rows)')
end

"""
Copy a `SciComplexVector` allocated by Rust into a Julia `Vector{ComplexF64}`,
then free the Rust allocation.
"""
function _consume_sci_complex_vector(cv::SciComplexVector)::Vector{ComplexF64}
    n = Int(cv.len)
    result = Vector{ComplexF64}(undef, n)
    if n > 0 && cv.real != C_NULL && cv.imag != C_NULL
        real_part = unsafe_wrap(Vector{Float64}, cv.real, n; own=false)
        imag_part = unsafe_wrap(Vector{Float64}, cv.imag, n; own=false)
        for i in 1:n
            result[i] = ComplexF64(real_part[i], imag_part[i])
        end
    end
    cv_ref = Ref(cv)
    ccall((:sci_complex_vector_free, _LIBPATH), Cvoid, (Ptr{SciComplexVector},), cv_ref)
    return result
end

"""
Build a `SciVector` that borrows data from a Julia `AbstractVector{Float64}`.

The returned `SciVector` is only valid while `vec` is alive and not moved.
"""
function _borrow_vector(vec::AbstractVector{Float64})::SciVector
    n = length(vec)
    data_ptr = n > 0 ? pointer(vec) : Ptr{Cdouble}(C_NULL)
    return SciVector(data_ptr, Csize_t(n))
end

"""
Build a `SciMatrix` that borrows data from a Julia `Matrix{Float64}`.

Julia matrices are stored column-major; Rust/C expects row-major. We create a
contiguous transposed copy so that the layout seen by Rust is row-major.
Returns `(SciMatrix, transposed_copy)` — the caller must keep `transposed_copy`
alive for the duration of the C call.
"""
function _borrow_matrix(mat::AbstractMatrix{Float64})
    rows, cols = size(mat)
    # Produce a row-major (C order) copy: transpose then collect.
    row_major = collect(mat')  # size (cols, rows) in column-major == row-major view
    # Flatten to a 1-D vector.
    flat = vec(row_major)
    data_ptr = isempty(flat) ? Ptr{Cdouble}(C_NULL) : pointer(flat)
    sm = SciMatrix(data_ptr, Csize_t(rows), Csize_t(cols))
    return sm, flat
end

# ---------------------------------------------------------------------------
# Version introspection
# ---------------------------------------------------------------------------

"""
    version() -> String

Return the version string of the linked SciRS2 core library.

# Example
```julia
ver = SciRS2.version()
println("SciRS2 version: \$ver")
```
"""
function version()::String
    ptr = ccall((:sci_version, _LIBPATH), Ptr{UInt8}, ())
    return unsafe_string(ptr)
end

# ---------------------------------------------------------------------------
# Memory management (low-level, exposed for advanced users)
# ---------------------------------------------------------------------------

"""
    sci_free_vector!(v::SciVector)

Free a `SciVector` that was allocated by Rust.

Normal users should not call this; the high-level wrappers handle memory
automatically via `_consume_sci_vector`.
"""
function sci_free_vector!(v::Ref{SciVector})
    ccall((:sci_vector_free, _LIBPATH), Cvoid, (Ptr{SciVector},), v)
end

"""
    sci_free_matrix!(m::Ref{SciMatrix})

Free a `SciMatrix` that was allocated by Rust.
"""
function sci_free_matrix!(m::Ref{SciMatrix})
    ccall((:sci_matrix_free, _LIBPATH), Cvoid, (Ptr{SciMatrix},), m)
end

"""
    sci_free_complex_vector!(cv::Ref{SciComplexVector})

Free a `SciComplexVector` that was allocated by Rust.
"""
function sci_free_complex_vector!(cv::Ref{SciComplexVector})
    ccall((:sci_complex_vector_free, _LIBPATH), Cvoid, (Ptr{SciComplexVector},), cv)
end

# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

"""
    matmul(a::Matrix{Float64}, b::Matrix{Float64}) -> Matrix{Float64}

Matrix multiplication `C = A * B`.

This is a high-level convenience function implemented via the BLAS routine
exposed in Julia's `LinearAlgebra` standard library. For arbitrary matrix
products, Julia's own `*` operator calls the same underlying BLAS routines and
should be preferred. This wrapper is provided for interface symmetry with other
SciRS2 operations and as a reference implementation.

# Arguments
- `a`: Left matrix of size (m, k).
- `b`: Right matrix of size (k, n).

# Returns
- Product matrix of size (m, n).

# Throws
- `DimensionMismatch` if `size(a, 2) != size(b, 1)`.

# Example
```julia
A = [1.0 2.0; 3.0 4.0]
B = [5.0 6.0; 7.0 8.0]
C = SciRS2.matmul(A, B)
```
"""
function matmul(a::AbstractMatrix{Float64}, b::AbstractMatrix{Float64})::Matrix{Float64}
    # sci_matmul is not yet exposed in the C FFI; delegate to Julia's optimised
    # BLAS-backed multiplication which is equivalent.
    if size(a, 2) != size(b, 1)
        throw(DimensionMismatch(
            "matmul: A is $(size(a,1))x$(size(a,2)) but B is $(size(b,1))x$(size(b,2))"
        ))
    end
    return a * b
end

"""
    det(mat::Matrix{Float64}) -> Float64

Compute the determinant of a square matrix via `sci_det`.

# Arguments
- `mat`: Square matrix.

# Returns
- Scalar determinant value.

# Throws
- `ErrorException` if the matrix is non-square or computation fails.

# Example
```julia
A = [1.0 2.0; 3.0 4.0]
d = SciRS2.det(A)   # -2.0
```
"""
function det(mat::AbstractMatrix{Float64})::Float64
    sm, flat = _borrow_matrix(mat)
    GC.@preserve flat begin
        out = Ref{Cdouble}(0.0)
        r = ccall(
            (:sci_det, _LIBPATH),
            SciResult,
            (Ref{SciMatrix}, Ptr{Cdouble}),
            Ref(sm), out,
        )
        _check(r, "sci_det")
        return out[]
    end
end

"""
    inv(mat::Matrix{Float64}) -> Matrix{Float64}

Compute the inverse of a square matrix via `sci_inv`.

# Arguments
- `mat`: Square, non-singular matrix.

# Returns
- Inverse matrix of the same size.

# Throws
- `ErrorException` if the matrix is singular or non-square.

# Example
```julia
A = [4.0 7.0; 2.0 6.0]
Ainv = SciRS2.inv(A)
```
"""
function inv(mat::AbstractMatrix{Float64})::Matrix{Float64}
    sm, flat = _borrow_matrix(mat)
    GC.@preserve flat begin
        out_sm = Ref(SciMatrix())
        r = ccall(
            (:sci_inv, _LIBPATH),
            SciResult,
            (Ref{SciMatrix}, Ptr{SciMatrix}),
            Ref(sm), out_sm,
        )
        _check(r, "sci_inv")
        return _consume_sci_matrix(out_sm[])
    end
end

"""
    solve(a::Matrix{Float64}, b::Vector{Float64}) -> Vector{Float64}

Solve the linear system A * x = b via `sci_solve` (LU decomposition).

# Arguments
- `a`: Square coefficient matrix of size (n, n).
- `b`: Right-hand side vector of length n.

# Returns
- Solution vector x of length n.

# Throws
- `ErrorException` if the system is singular or dimensions are mismatched.

# Example
```julia
A = [2.0 1.0; 1.0 3.0]
b = [5.0, 7.0]
x = SciRS2.solve(A, b)   # [1.6, 1.8]
```
"""
function solve(a::AbstractMatrix{Float64}, b::AbstractVector{Float64})::Vector{Float64}
    sm_a, flat_a = _borrow_matrix(a)
    b_cont = convert(Vector{Float64}, b)
    sv_b = _borrow_vector(b_cont)
    GC.@preserve flat_a b_cont begin
        out_sv = Ref(SciVector())
        r = ccall(
            (:sci_solve, _LIBPATH),
            SciResult,
            (Ref{SciMatrix}, Ref{SciVector}, Ptr{SciVector}),
            Ref(sm_a), Ref(sv_b), out_sv,
        )
        _check(r, "sci_solve")
        return _consume_sci_vector(out_sv[])
    end
end

"""
    svd(mat::Matrix{Float64}) -> NamedTuple{(:U, :S, :Vt)}

Compute the full SVD decomposition `A = U * Diagonal(S) * Vt` via `sci_svd`.

# Arguments
- `mat`: Input matrix of size (m, n).

# Returns
- Named tuple with fields:
  - `U`: Unitary matrix of size (m, m).
  - `S`: Singular values vector of length min(m, n).
  - `Vt`: Unitary matrix Vᵀ of size (n, n).

# Throws
- `ErrorException` on computation failure.

# Example
```julia
A = [1.0 0.0; 0.0 2.0; 0.0 0.0]
result = SciRS2.svd(A)
result.S   # singular values
```
"""
function svd(mat::AbstractMatrix{Float64})
    sm, flat = _borrow_matrix(mat)
    GC.@preserve flat begin
        out_svd = Ref(SciSvdResult())
        r = ccall(
            (:sci_svd, _LIBPATH),
            SciResult,
            (Ref{SciMatrix}, Ptr{SciSvdResult}),
            Ref(sm), out_svd,
        )
        _check(r, "sci_svd")
        raw = out_svd[]
        # Extract and free each component individually.
        u_mat = _consume_sci_matrix(raw.u)
        s_vec = _consume_sci_vector(raw.s)
        vt_mat = _consume_sci_matrix(raw.vt)
        return (U=u_mat, S=s_vec, Vt=vt_mat)
    end
end

"""
    eig(mat::Matrix{Float64}) -> NamedTuple{(:values, :vectors)}

Compute the eigenvalue decomposition of a square matrix via `sci_eig`.

Returns complex eigenvalues and real eigenvectors (imaginary parts of
eigenvectors are not exposed through the FFI boundary).

# Arguments
- `mat`: Square matrix of size (n, n).

# Returns
- Named tuple with fields:
  - `values`: Vector of `ComplexF64` eigenvalues of length n.
  - `vectors`: Matrix of real eigenvectors of size (n, n) (columns are vectors).

# Throws
- `ErrorException` if the matrix is non-square or computation fails.

# Example
```julia
A = [3.0 0.0; 0.0 5.0]
result = SciRS2.eig(A)
result.values   # [3.0+0.0im, 5.0+0.0im] (order may vary)
```
"""
function eig(mat::AbstractMatrix{Float64})
    sm, flat = _borrow_matrix(mat)
    GC.@preserve flat begin
        out_eig = Ref(SciEigResult())
        r = ccall(
            (:sci_eig, _LIBPATH),
            SciResult,
            (Ref{SciMatrix}, Ptr{SciEigResult}),
            Ref(sm), out_eig,
        )
        _check(r, "sci_eig")
        raw = out_eig[]
        eigenvalues = _consume_sci_complex_vector(raw.eigenvalues)
        eigenvectors = _consume_sci_matrix(raw.eigenvectors)
        return (values=eigenvalues, vectors=eigenvectors)
    end
end

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

"""
    mean(vec::Vector{Float64}) -> Float64

Compute the arithmetic mean of a vector via `sci_mean`.

# Arguments
- `vec`: Non-empty vector.

# Returns
- Arithmetic mean.

# Throws
- `ErrorException` if the vector is empty.

# Example
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
SciRS2.mean(x)   # 3.0
```
"""
function mean(vec::AbstractVector{Float64})::Float64
    v_cont = convert(Vector{Float64}, vec)
    sv = _borrow_vector(v_cont)
    GC.@preserve v_cont begin
        out = Ref{Cdouble}(0.0)
        r = ccall(
            (:sci_mean, _LIBPATH),
            SciResult,
            (Ref{SciVector}, Ptr{Cdouble}),
            Ref(sv), out,
        )
        _check(r, "sci_mean")
        return out[]
    end
end

"""
    std(vec::Vector{Float64}; ddof::Integer=1) -> Float64

Compute the standard deviation via `sci_std`.

# Arguments
- `vec`: Vector with at least `ddof + 1` elements.
- `ddof`: Delta degrees of freedom. `1` (default) gives sample std; `0` gives
  population std.

# Returns
- Standard deviation (non-negative).

# Throws
- `ErrorException` if the vector is too short.

# Example
```julia
x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
SciRS2.std(x)          # sample std ≈ 2.0
SciRS2.std(x; ddof=0)  # population std
```
"""
function std(vec::AbstractVector{Float64}; ddof::Integer=1)::Float64
    v_cont = convert(Vector{Float64}, vec)
    sv = _borrow_vector(v_cont)
    GC.@preserve v_cont begin
        out = Ref{Cdouble}(0.0)
        r = ccall(
            (:sci_std, _LIBPATH),
            SciResult,
            (Ref{SciVector}, Csize_t, Ptr{Cdouble}),
            Ref(sv), Csize_t(ddof), out,
        )
        _check(r, "sci_std")
        return out[]
    end
end

"""
    median(vec::Vector{Float64}) -> Float64

Compute the median of a vector via `sci_median`.

For an even number of elements, returns the average of the two middle values.

# Arguments
- `vec`: Non-empty vector.

# Returns
- Median value.

# Throws
- `ErrorException` if the vector is empty.

# Example
```julia
x = [3.0, 1.0, 4.0, 1.0, 5.0]
SciRS2.median(x)   # 3.0
```
"""
function median(vec::AbstractVector{Float64})::Float64
    v_cont = convert(Vector{Float64}, vec)
    sv = _borrow_vector(v_cont)
    GC.@preserve v_cont begin
        out = Ref{Cdouble}(0.0)
        r = ccall(
            (:sci_median, _LIBPATH),
            SciResult,
            (Ref{SciVector}, Ptr{Cdouble}),
            Ref(sv), out,
        )
        _check(r, "sci_median")
        return out[]
    end
end

"""
    percentile(vec::Vector{Float64}, q::Float64) -> Float64

Compute the q-th percentile using linear interpolation via `sci_percentile`.

Uses the same interpolation method as NumPy's `numpy.percentile`.

# Arguments
- `vec`: Non-empty vector.
- `q`: Percentile in the range [0, 100].

# Returns
- Percentile value.

# Throws
- `ErrorException` if `q` is outside [0, 100] or the vector is empty.

# Example
```julia
x = collect(1.0:10.0)
SciRS2.percentile(x, 50.0)   # 5.5 (median)
SciRS2.percentile(x, 25.0)   # 25th percentile
```
"""
function percentile(vec::AbstractVector{Float64}, q::Real)::Float64
    v_cont = convert(Vector{Float64}, vec)
    sv = _borrow_vector(v_cont)
    GC.@preserve v_cont begin
        out = Ref{Cdouble}(0.0)
        r = ccall(
            (:sci_percentile, _LIBPATH),
            SciResult,
            (Ref{SciVector}, Cdouble, Ptr{Cdouble}),
            Ref(sv), Cdouble(q), out,
        )
        _check(r, "sci_percentile")
        return out[]
    end
end

"""
    variance(vec::Vector{Float64}; ddof::Integer=1) -> Float64

Compute the variance via `sci_variance`.

# Arguments
- `vec`: Vector with at least `ddof + 1` elements.
- `ddof`: Delta degrees of freedom (`1` = sample, `0` = population).

# Returns
- Variance (non-negative).

# Example
```julia
x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
SciRS2.variance(x)   # sample variance ≈ 4.0
```
"""
function variance(vec::AbstractVector{Float64}; ddof::Integer=1)::Float64
    v_cont = convert(Vector{Float64}, vec)
    sv = _borrow_vector(v_cont)
    GC.@preserve v_cont begin
        out = Ref{Cdouble}(0.0)
        r = ccall(
            (:sci_variance, _LIBPATH),
            SciResult,
            (Ref{SciVector}, Csize_t, Ptr{Cdouble}),
            Ref(sv), Csize_t(ddof), out,
        )
        _check(r, "sci_variance")
        return out[]
    end
end

"""
    correlation(x::Vector{Float64}, y::Vector{Float64}) -> Float64

Compute the Pearson correlation coefficient between two vectors via
`sci_correlation`.

# Arguments
- `x`, `y`: Vectors of the same length with at least 2 elements.

# Returns
- Correlation coefficient in [-1, 1].

# Throws
- `ErrorException` on dimension mismatch, constant vectors, or other failures.

# Example
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 8.0, 10.0]
SciRS2.correlation(x, y)   # 1.0 (perfect positive correlation)
```
"""
function correlation(x::AbstractVector{Float64}, y::AbstractVector{Float64})::Float64
    x_cont = convert(Vector{Float64}, x)
    y_cont = convert(Vector{Float64}, y)
    sv_x = _borrow_vector(x_cont)
    sv_y = _borrow_vector(y_cont)
    GC.@preserve x_cont y_cont begin
        out = Ref{Cdouble}(0.0)
        r = ccall(
            (:sci_correlation, _LIBPATH),
            SciResult,
            (Ref{SciVector}, Ref{SciVector}, Ptr{Cdouble}),
            Ref(sv_x), Ref(sv_y), out,
        )
        _check(r, "sci_correlation")
        return out[]
    end
end

# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------

"""
    fft_forward(real_in::Vector{Float64},
                imag_in::Vector{Float64}=zeros(length(real_in))) -> Vector{ComplexF64}

Compute the forward (analysis) FFT of a complex signal via `sci_fft_forward`.

Supports arbitrary-length inputs (not just powers of 2) via Bluestein's
algorithm.

# Arguments
- `real_in`: Real parts of the input signal.
- `imag_in`: Imaginary parts. Defaults to all zeros (pure-real input).

# Returns
- Complex spectrum as `Vector{ComplexF64}` of the same length as input.

# Throws
- `ErrorException` on invalid input.

# Example
```julia
# Real-valued impulse
x = zeros(8); x[1] = 1.0
X = SciRS2.fft_forward(x)
# All-ones spectrum for unit impulse
```
"""
function fft_forward(
    real_in::AbstractVector{Float64},
    imag_in::AbstractVector{Float64}=zeros(Float64, length(real_in)),
)::Vector{ComplexF64}
    n = length(real_in)
    if length(imag_in) != n
        error("fft_forward: real_in and imag_in must have the same length")
    end
    real_cont = convert(Vector{Float64}, real_in)
    imag_cont = convert(Vector{Float64}, imag_in)
    GC.@preserve real_cont imag_cont begin
        real_ptr = n > 0 ? pointer(real_cont) : Ptr{Cdouble}(C_NULL)
        imag_ptr = n > 0 ? pointer(imag_cont) : Ptr{Cdouble}(C_NULL)
        out_cv = Ref(SciComplexVector())
        r = ccall(
            (:sci_fft_forward, _LIBPATH),
            SciResult,
            (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{SciComplexVector}),
            real_ptr, imag_ptr, Csize_t(n), out_cv,
        )
        _check(r, "sci_fft_forward")
        return _consume_sci_complex_vector(out_cv[])
    end
end

"""
    fft_inverse(real_in::Vector{Float64},
                imag_in::Vector{Float64}) -> Vector{ComplexF64}

Compute the inverse FFT and scale by 1/N via `sci_fft_inverse`.

# Arguments
- `real_in`: Real parts of the frequency-domain input.
- `imag_in`: Imaginary parts of the frequency-domain input.

# Returns
- Time-domain signal as `Vector{ComplexF64}`.

# Throws
- `ErrorException` on invalid input.

# Example
```julia
X_real = [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
X_imag = zeros(8)
x = SciRS2.fft_inverse(X_real, X_imag)
# Reconstructed signal: real part ≈ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```
"""
function fft_inverse(
    real_in::AbstractVector{Float64},
    imag_in::AbstractVector{Float64},
)::Vector{ComplexF64}
    n = length(real_in)
    if length(imag_in) != n
        error("fft_inverse: real_in and imag_in must have the same length")
    end
    real_cont = convert(Vector{Float64}, real_in)
    imag_cont = convert(Vector{Float64}, imag_in)
    GC.@preserve real_cont imag_cont begin
        real_ptr = n > 0 ? pointer(real_cont) : Ptr{Cdouble}(C_NULL)
        imag_ptr = n > 0 ? pointer(imag_cont) : Ptr{Cdouble}(C_NULL)
        out_cv = Ref(SciComplexVector())
        r = ccall(
            (:sci_fft_inverse, _LIBPATH),
            SciResult,
            (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{SciComplexVector}),
            real_ptr, imag_ptr, Csize_t(n), out_cv,
        )
        _check(r, "sci_fft_inverse")
        return _consume_sci_complex_vector(out_cv[])
    end
end

"""
    rfft(real_in::Vector{Float64}) -> Vector{ComplexF64}

Compute the forward FFT of a purely real signal via `sci_rfft`.

This is a convenience wrapper around `fft_forward` for real-valued inputs.

# Arguments
- `real_in`: Real-valued input signal.

# Returns
- Complex spectrum of the same length as input.

# Example
```julia
x = [1.0, 0.0, 0.0, 0.0]
X = SciRS2.rfft(x)
```
"""
function rfft(real_in::AbstractVector{Float64})::Vector{ComplexF64}
    n = length(real_in)
    real_cont = convert(Vector{Float64}, real_in)
    GC.@preserve real_cont begin
        real_ptr = n > 0 ? pointer(real_cont) : Ptr{Cdouble}(C_NULL)
        out_cv = Ref(SciComplexVector())
        r = ccall(
            (:sci_rfft, _LIBPATH),
            SciResult,
            (Ptr{Cdouble}, Csize_t, Ptr{SciComplexVector}),
            real_ptr, Csize_t(n), out_cv,
        )
        _check(r, "sci_rfft")
        return _consume_sci_complex_vector(out_cv[])
    end
end

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

"""
    minimize_scalar(f::Function, a::Real, b::Real;
                    tol::Float64=1e-8, max_iter::Int=0)
    -> NamedTuple{(:x_min, :f_min)}

Minimize a scalar function on the interval [a, b] using golden-section search
via `sci_minimize_scalar`.

# Arguments
- `f`: Scalar function `f(x::Float64) -> Float64`.
- `a`, `b`: Search interval endpoints; must satisfy `a < b`.
- `tol`: Convergence tolerance (default 1e-8).
- `max_iter`: Maximum iterations; `0` uses the library default (500).

# Returns
- Named tuple with:
  - `x_min`: Location of the minimum.
  - `f_min`: Minimum function value.

# Throws
- `ErrorException` if `a >= b` or `tol <= 0`.

# Example
```julia
result = SciRS2.minimize_scalar(x -> (x - 3.0)^2, 0.0, 6.0)
result.x_min   # ≈ 3.0
result.f_min   # ≈ 0.0
```
"""
function minimize_scalar(
    f::Function,
    a::Real,
    b::Real;
    tol::Float64=1e-8,
    max_iter::Int=0,
)
    # Wrap the Julia function in a C-callable function pointer.
    c_fn = @cfunction($f, Cdouble, (Cdouble, Ptr{Cvoid}))
    x_out = Ref{Cdouble}(0.0)
    f_out = Ref{Cdouble}(0.0)
    r = ccall(
        (:sci_minimize_scalar, _LIBPATH),
        SciResult,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cdouble, Cdouble, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
        c_fn, C_NULL, Cdouble(a), Cdouble(b), tol, Csize_t(max_iter), x_out, f_out,
    )
    _check(r, "sci_minimize_scalar")
    return (x_min=x_out[], f_min=f_out[])
end

"""
    minimize_brent(f::Function, a::Real, b::Real;
                   tol::Float64=1e-8, max_iter::Int=0)
    -> NamedTuple{(:x_min, :f_min)}

Minimize a scalar function on [a, b] using Brent's method (parabolic
interpolation when possible) via `sci_minimize_brent`.

Generally faster than golden-section search for smooth, unimodal functions.

# Arguments
- `f`: Scalar function `f(x::Float64) -> Float64`.
- `a`, `b`: Search interval.
- `tol`: Convergence tolerance (default 1e-8).
- `max_iter`: Maximum iterations; `0` uses default (500).

# Returns
- Named tuple `(x_min, f_min)`.

# Example
```julia
result = SciRS2.minimize_brent(x -> sin(x), 3.0, 4.0)
result.x_min   # ≈ π/2 + π ≈ 4.712 (sin minimum in [3,4] is sin(3π/2))
```
"""
function minimize_brent(
    f::Function,
    a::Real,
    b::Real;
    tol::Float64=1e-8,
    max_iter::Int=0,
)
    c_fn = @cfunction($f, Cdouble, (Cdouble, Ptr{Cvoid}))
    x_out = Ref{Cdouble}(0.0)
    f_out = Ref{Cdouble}(0.0)
    r = ccall(
        (:sci_minimize_brent, _LIBPATH),
        SciResult,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cdouble, Cdouble, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
        c_fn, C_NULL, Cdouble(a), Cdouble(b), tol, Csize_t(max_iter), x_out, f_out,
    )
    _check(r, "sci_minimize_brent")
    return (x_min=x_out[], f_min=f_out[])
end

"""
    minimize_nelder_mead(f::Function, x0::Vector{Float64};
                         tol::Float64=1e-6, max_iter::Int=0)
    -> NamedTuple{(:x_min, :f_min)}

Minimize a multidimensional function using a pure-Julia Nelder-Mead simplex
method.

Note: The SciRS2 C FFI currently exposes only scalar (1-D) minimization via
function pointers. Multi-dimensional Nelder-Mead is implemented here in Julia
for interface completeness and calls no C code.

# Arguments
- `f`: Multivariate function `f(x::Vector{Float64}) -> Float64`.
- `x0`: Initial guess vector (n-dimensional).
- `tol`: Convergence tolerance on simplex size (default 1e-6).
- `max_iter`: Maximum iterations; `0` uses default `200 * length(x0)`.

# Returns
- Named tuple:
  - `x_min`: Minimizer vector.
  - `f_min`: Minimum function value.

# Example
```julia
# Rosenbrock function
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = SciRS2.minimize_nelder_mead(rosenbrock, [0.0, 0.0]; max_iter=10000)
result.x_min   # ≈ [1.0, 1.0]
```
"""
function minimize_nelder_mead(
    f::Function,
    x0::AbstractVector{Float64};
    tol::Float64=1e-6,
    max_iter::Int=0,
)
    n = length(x0)
    max_it = max_iter == 0 ? 200 * n : max_iter

    # Build the initial simplex: n+1 vertices
    step = 0.05
    simplex = [copy(x0) for _ in 1:(n + 1)]
    for i in 1:n
        simplex[i + 1][i] += (abs(x0[i]) > 1e-8 ? 0.05 * abs(x0[i]) : step)
    end

    # Nelder-Mead coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho   = 0.5   # contraction
    sigma = 0.5   # shrink

    f_vals = [f(s) for s in simplex]

    for _ in 1:max_it
        # Sort vertices by function value (ascending)
        order = sortperm(f_vals)
        simplex = simplex[order]
        f_vals  = f_vals[order]

        # Convergence check: range of f values over simplex
        if (f_vals[end] - f_vals[1]) < tol * (abs(f_vals[1]) + 1.0)
            break
        end

        # Centroid of all but the worst
        centroid = sum(simplex[1:end-1]) / n

        # Reflection
        x_r = centroid + alpha * (centroid - simplex[end])
        f_r = f(x_r)

        if f_r < f_vals[1]
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r
                simplex[end] = x_e
                f_vals[end] = f_e
            else
                simplex[end] = x_r
                f_vals[end] = f_r
            end
        elseif f_r < f_vals[end]
            simplex[end] = x_r
            f_vals[end] = f_r
        else
            # Contraction
            if f_r < f_vals[end]
                x_c = centroid + rho * (x_r - centroid)
                f_c = f(x_c)
                if f_c <= f_r
                    simplex[end] = x_c
                    f_vals[end] = f_c
                    continue
                end
            else
                x_c = centroid + rho * (simplex[end] - centroid)
                f_c = f(x_c)
                if f_c < f_vals[end]
                    simplex[end] = x_c
                    f_vals[end] = f_c
                    continue
                end
            end
            # Shrink
            best = simplex[1]
            for i in 2:(n + 1)
                simplex[i] = best + sigma * (simplex[i] - best)
                f_vals[i] = f(simplex[i])
            end
        end
    end

    # Return the best vertex
    best_idx = argmin(f_vals)
    return (x_min=simplex[best_idx], f_min=f_vals[best_idx])
end

"""
    find_root(f::Function, a::Real, b::Real;
              tol::Float64=1e-12, max_iter::Int=0) -> Float64

Find a root of `f(x) = 0` on the interval [a, b] using Brent's method via
`sci_root_find`.

Requires that `f(a)` and `f(b)` have opposite signs.

# Arguments
- `f`: Scalar function.
- `a`, `b`: Bracket endpoints with sign change.
- `tol`: Absolute convergence tolerance (default 1e-12).
- `max_iter`: Maximum iterations; `0` uses library default (500).

# Returns
- Approximated root.

# Throws
- `ErrorException` if there is no sign change or if `tol <= 0`.

# Example
```julia
root = SciRS2.find_root(x -> x^2 - 4.0, 0.0, 10.0)
root   # ≈ 2.0
```
"""
function find_root(
    f::Function,
    a::Real,
    b::Real;
    tol::Float64=1e-12,
    max_iter::Int=0,
)::Float64
    c_fn = @cfunction($f, Cdouble, (Cdouble, Ptr{Cvoid}))
    x_out = Ref{Cdouble}(0.0)
    r = ccall(
        (:sci_root_find, _LIBPATH),
        SciResult,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cdouble, Cdouble, Csize_t, Ptr{Cdouble}),
        c_fn, C_NULL, Cdouble(a), Cdouble(b), tol, Csize_t(max_iter), x_out,
    )
    _check(r, "sci_root_find")
    return x_out[]
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export version
export sci_free_vector!, sci_free_matrix!, sci_free_complex_vector!
# Linear algebra
export matmul, det, inv, solve, svd, eig
# Statistics
export mean, std, median, percentile, variance, correlation
# FFT
export fft_forward, fft_inverse, rfft
# Optimization
export minimize_scalar, minimize_brent, minimize_nelder_mead, find_root

end  # module SciRS2
