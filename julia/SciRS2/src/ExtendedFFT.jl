"""
    ExtendedFFT

Extended FFT and signal processing functions for SciRS2.

Provides:
- Power spectrum and power spectral density
- Short-Time Fourier Transform (STFT) / spectrogram
- Discrete Cosine Transform (DCT) types I–IV
- Discrete Sine Transform (DST) types I–IV
- Hilbert transform and analytic signal
- Windowing functions (Hann, Hamming, Blackman, Kaiser, Tukey, Bartlett, Flattop)
- Frequency axis generation utilities
- Welch's power spectral density estimate
- Chirp Z-transform (CZT) for arbitrary frequency grids
"""
module ExtendedFFT

using LinearAlgebra

# Internal DFT implementation (O(n²)) for use when n is small or for testing.
# For production use, the FFI-backed fft_forward/fft_inverse are preferred.

"""Compute the N-point DFT of a real/complex signal (O(n²), pure Julia)."""
function _dft(x::AbstractVector{<:Number})::Vector{ComplexF64}
    n = length(x)
    W = [exp(-2π * im * k * j / n) for k in 0:(n-1), j in 0:(n-1)]
    return W * collect(ComplexF64, x)
end

"""Compute the inverse DFT."""
function _idft(X::AbstractVector{ComplexF64})::Vector{ComplexF64}
    n = length(X)
    W = [exp(2π * im * k * j / n) for k in 0:(n-1), j in 0:(n-1)]
    return (W * X) ./ n
end

# ===========================================================================
# 1. WINDOWING FUNCTIONS
# ===========================================================================

"""
    window_hann(n::Int) -> Vector{Float64}

Return a Hann (raised cosine) window of length n.

  w[k] = 0.5 * (1 - cos(2π*k/(n-1))),  k = 0, ..., n-1

The Hann window reduces spectral leakage with a modest amplitude reduction.
"""
function window_hann(n::Int)::Vector{Float64}
    if n < 1
        throw(ArgumentError("window_hann: n must be ≥ 1"))
    end
    if n == 1
        return [1.0]
    end
    return [0.5 * (1.0 - cos(2π * k / (n - 1))) for k in 0:(n-1)]
end

"""
    window_hamming(n::Int) -> Vector{Float64}

Return a Hamming window of length n.

  w[k] = 0.54 - 0.46 * cos(2π*k/(n-1))

The Hamming window has better sidelobe level than Hann but a slightly wider
main lobe.
"""
function window_hamming(n::Int)::Vector{Float64}
    if n < 1
        throw(ArgumentError("window_hamming: n must be ≥ 1"))
    end
    if n == 1
        return [1.0]
    end
    return [0.54 - 0.46 * cos(2π * k / (n - 1)) for k in 0:(n-1)]
end

"""
    window_blackman(n::Int) -> Vector{Float64}

Return a Blackman window of length n.

  w[k] = 0.42 - 0.5*cos(2π*k/(n-1)) + 0.08*cos(4π*k/(n-1))

The Blackman window has the lowest sidelobes of common windows but a wider
main lobe.
"""
function window_blackman(n::Int)::Vector{Float64}
    if n < 1
        throw(ArgumentError("window_blackman: n must be ≥ 1"))
    end
    if n == 1
        return [1.0]
    end
    return [0.42 - 0.5 * cos(2π * k / (n - 1)) + 0.08 * cos(4π * k / (n - 1)) for k in 0:(n-1)]
end

"""
    window_bartlett(n::Int) -> Vector{Float64}

Return a Bartlett (triangular) window of length n.
"""
function window_bartlett(n::Int)::Vector{Float64}
    if n < 1
        throw(ArgumentError("window_bartlett: n must be ≥ 1"))
    end
    if n == 1
        return [1.0]
    end
    return [1.0 - abs(2.0 * k / (n - 1) - 1.0) for k in 0:(n-1)]
end

"""
    window_flattop(n::Int) -> Vector{Float64}

Return a flat-top window of length n.

The flat-top window has a very flat passband (amplitude accuracy) at the
cost of wide sidelobes.
"""
function window_flattop(n::Int)::Vector{Float64}
    if n < 1
        throw(ArgumentError("window_flattop: n must be ≥ 1"))
    end
    if n == 1
        return [1.0]
    end
    a0, a1, a2, a3, a4 = 1.0, 1.93, 1.29, 0.388, 0.032
    return [a0 - a1 * cos(2π*k/(n-1)) + a2 * cos(4π*k/(n-1)) -
            a3 * cos(6π*k/(n-1)) + a4 * cos(8π*k/(n-1)) for k in 0:(n-1)]
end

"""
    window_tukey(n::Int; alpha::Float64=0.5) -> Vector{Float64}

Return a Tukey (cosine-tapered) window of length n.

The Tukey window tapers the first and last α/2 fraction of the window
with a raised cosine and leaves the middle flat.
"""
function window_tukey(n::Int; alpha::Float64=0.5)::Vector{Float64}
    if n < 1
        throw(ArgumentError("window_tukey: n must be ≥ 1"))
    end
    if n == 1
        return [1.0]
    end
    alpha_c = clamp(alpha, 0.0, 1.0)
    w = ones(Float64, n)
    for k in 0:(n-1)
        t = k / (n - 1)
        if t < alpha_c / 2.0
            w[k + 1] = 0.5 * (1.0 - cos(2π * t / alpha_c))
        elseif t > 1.0 - alpha_c / 2.0
            w[k + 1] = 0.5 * (1.0 - cos(2π * (1.0 - t) / alpha_c))
        end
    end
    return w
end

"""
    window_kaiser(n::Int; beta::Float64=8.6) -> Vector{Float64}

Return a Kaiser window of length n with shape parameter beta.

The Kaiser window provides a near-optimal compromise between main lobe width
and sidelobe level, controlled by `beta`:
- beta ≈ 0: rectangular window
- beta ≈ 5: Hamming-like
- beta ≈ 8.6: ≈ -60 dB sidelobes (default)
- beta ≈ 14: ≈ -100 dB sidelobes

Uses the zeroth-order modified Bessel function I₀.
"""
function window_kaiser(n::Int; beta::Float64=8.6)::Vector{Float64}
    if n < 1
        throw(ArgumentError("window_kaiser: n must be ≥ 1"))
    end
    if n == 1
        return [1.0]
    end
    i0_beta = _bessel_i0(beta)
    return [_bessel_i0(beta * sqrt(1.0 - ((2.0 * k / (n - 1)) - 1.0)^2)) / i0_beta
            for k in 0:(n-1)]
end

"""Compute the zeroth-order modified Bessel function I₀(x) via series expansion."""
function _bessel_i0(x::Float64)::Float64
    y = x / 2.0
    s = 1.0
    term = 1.0
    for k in 1:30
        term *= (y / k)^2
        s += term
        if term < eps(Float64) * s
            break
        end
    end
    return s
end

"""
    get_window(name::Symbol, n::Int; kwargs...) -> Vector{Float64}

Get a window by name symbol.

Supported names: `:hann`, `:hamming`, `:blackman`, `:bartlett`, `:flattop`,
`:rectangular`, `:tukey`, `:kaiser`.
"""
function get_window(name::Symbol, n::Int; kwargs...)::Vector{Float64}
    if name == :hann
        return window_hann(n)
    elseif name == :hamming
        return window_hamming(n)
    elseif name == :blackman
        return window_blackman(n)
    elseif name == :bartlett
        return window_bartlett(n)
    elseif name == :flattop
        return window_flattop(n)
    elseif name == :rectangular
        return ones(Float64, n)
    elseif name == :tukey
        alpha = get(kwargs, :alpha, 0.5)
        return window_tukey(n; alpha=Float64(alpha))
    elseif name == :kaiser
        beta = get(kwargs, :beta, 8.6)
        return window_kaiser(n; beta=Float64(beta))
    else
        throw(ArgumentError("get_window: unknown window '$name'"))
    end
end

# ===========================================================================
# 2. POWER SPECTRUM AND PERIODOGRAM
# ===========================================================================

"""
    power_spectrum(x::AbstractVector{Float64}; fs::Float64=1.0,
                   window::Symbol=:hann, onesided::Bool=true)
    -> NamedTuple{(:frequencies, :power)}

Compute the power spectrum of a real signal using the FFT periodogram.

The window function is applied before the FFT to reduce spectral leakage.
The power is normalized so that Parseval's theorem holds (in the discrete sense).

# Arguments
- `x`: real-valued time series.
- `fs`: sampling frequency in Hz (default 1.0, giving normalized frequency).
- `window`: windowing function symbol (default `:hann`).
- `onesided`: if `true` (default), return only positive frequencies (n/2+1 bins);
  if `false`, return the full two-sided spectrum.

# Returns
Named tuple:
- `frequencies`: frequency bins in Hz.
- `power`: power spectral density values.

# Examples
```julia
# 100 Hz sampling, 1 second of a 10 Hz sine + noise
fs = 100.0
t = collect(0.0:(1/fs):(1.0 - 1/fs))
x = sin.(2π * 10.0 .* t) .+ 0.1 .* randn(length(t))
ps = power_spectrum(x; fs=fs)
```
"""
function power_spectrum(
    x::AbstractVector{Float64};
    fs::Float64=1.0,
    window::Symbol=:hann,
    onesided::Bool=true,
)
    n = length(x)
    if n < 2
        throw(ArgumentError("power_spectrum: need at least 2 samples"))
    end

    w = get_window(window, n)
    xw = x .* w

    # FFT via DFT (pure Julia; for large n use the FFI-backed version)
    X = _dft(xw)

    # Normalization factor: preserve power
    win_norm = sum(w .^ 2)

    if onesided
        n_half = div(n, 2) + 1
        freqs = collect(0:(n_half - 1)) .* (fs / n)
        power = abs.(X[1:n_half]) .^ 2 ./ (fs * win_norm)
        # Double non-DC and non-Nyquist bins for one-sided PSD
        power[2:(n_half - 1)] .*= 2.0
        return (frequencies=freqs, power=power)
    else
        freqs = collect(0:(n - 1)) .* (fs / n)
        power = abs.(X) .^ 2 ./ (fs * win_norm)
        return (frequencies=freqs, power=power)
    end
end

"""
    welch_psd(x::AbstractVector{Float64}; fs::Float64=1.0, nperseg::Int=256,
              noverlap::Union{Int,Nothing}=nothing, window::Symbol=:hann)
    -> NamedTuple{(:frequencies, :power)}

Estimate the power spectral density using Welch's method.

The signal is divided into overlapping segments, a windowed periodogram is
computed for each segment, and the results are averaged. This reduces variance
compared to a single periodogram.

# Arguments
- `x`: real-valued time series.
- `fs`: sampling frequency in Hz.
- `nperseg`: length of each segment (default 256, or length(x) if shorter).
- `noverlap`: number of overlapping samples between segments
  (default: half of `nperseg`).
- `window`: windowing function (default `:hann`).

# Returns
Named tuple with `frequencies` and `power` (averaged PSD).
"""
function welch_psd(
    x::AbstractVector{Float64};
    fs::Float64=1.0,
    nperseg::Int=256,
    noverlap::Union{Int,Nothing}=nothing,
    window::Symbol=:hann,
)
    n = length(x)
    seg_len = min(nperseg, n)
    overlap = isnothing(noverlap) ? div(seg_len, 2) : noverlap

    if overlap < 0 || overlap >= seg_len
        throw(ArgumentError("welch_psd: noverlap must be in [0, nperseg-1]"))
    end

    step = seg_len - overlap
    n_half = div(seg_len, 2) + 1
    freqs = collect(0:(n_half - 1)) .* (fs / seg_len)

    w = get_window(window, seg_len)
    win_norm = sum(w .^ 2)

    power_sum = zeros(Float64, n_half)
    n_segs = 0

    start = 1
    while start + seg_len - 1 <= n
        seg = x[start:(start + seg_len - 1)] .* w
        X = _dft(seg)
        power = abs.(X[1:n_half]) .^ 2 ./ (fs * win_norm)
        power[2:(n_half - 1)] .*= 2.0
        power_sum .+= power
        n_segs += 1
        start += step
    end

    if n_segs == 0
        throw(ArgumentError("welch_psd: signal too short for nperseg=$nperseg"))
    end

    return (frequencies=freqs, power=power_sum ./ n_segs)
end

# ===========================================================================
# 3. SHORT-TIME FOURIER TRANSFORM (STFT) AND SPECTROGRAM
# ===========================================================================

"""
    stft(x::AbstractVector{Float64}; fs::Float64=1.0, nperseg::Int=256,
         noverlap::Union{Int,Nothing}=nothing, window::Symbol=:hann)
    -> NamedTuple{(:frequencies, :times, :Zxx)}

Compute the Short-Time Fourier Transform of a signal.

# Arguments
- `x`: real-valued input signal.
- `fs`: sampling frequency in Hz.
- `nperseg`: FFT segment length (default 256).
- `noverlap`: overlap in samples (default: `nperseg ÷ 2`).
- `window`: windowing function (default `:hann`).

# Returns
Named tuple:
- `frequencies`: frequency bins (length n_half = nperseg÷2+1).
- `times`: center times of each STFT frame.
- `Zxx`: complex STFT matrix of shape (n_half, n_frames).

# Examples
```julia
x = sin.(2π .* collect(0:0.01:4.0))
result = stft(x; fs=100.0, nperseg=64)
magnitude = abs.(result.Zxx)
```
"""
function stft(
    x::AbstractVector{Float64};
    fs::Float64=1.0,
    nperseg::Int=256,
    noverlap::Union{Int,Nothing}=nothing,
    window::Symbol=:hann,
)
    n = length(x)
    seg_len = min(nperseg, n)
    overlap = isnothing(noverlap) ? div(seg_len, 2) : noverlap

    if overlap < 0 || overlap >= seg_len
        throw(ArgumentError("stft: noverlap must be in [0, nperseg-1]"))
    end

    step = seg_len - overlap
    n_half = div(seg_len, 2) + 1
    w = get_window(window, seg_len)

    # Collect frames
    frame_starts = collect(1:step:(n - seg_len + 1))
    n_frames = length(frame_starts)

    if n_frames == 0
        throw(ArgumentError("stft: signal too short for nperseg=$seg_len"))
    end

    Zxx = Matrix{ComplexF64}(undef, n_half, n_frames)
    times = Vector{Float64}(undef, n_frames)

    for (fi, start) in enumerate(frame_starts)
        seg = x[start:(start + seg_len - 1)] .* w
        X = _dft(seg)
        Zxx[:, fi] = X[1:n_half]
        times[fi] = (start - 1 + seg_len / 2) / fs
    end

    freqs = collect(0:(n_half - 1)) .* (fs / seg_len)
    return (frequencies=freqs, times=times, Zxx=Zxx)
end

"""
    spectrogram(x::AbstractVector{Float64}; fs::Float64=1.0, nperseg::Int=256,
                noverlap::Union{Int,Nothing}=nothing, window::Symbol=:hann)
    -> NamedTuple{(:frequencies, :times, :power)}

Compute a power spectrogram (squared magnitude of STFT).

# Returns
Named tuple:
- `frequencies`, `times`: same as `stft`.
- `power`: real-valued (n_half × n_frames) matrix of power values.
"""
function spectrogram(
    x::AbstractVector{Float64};
    fs::Float64=1.0,
    nperseg::Int=256,
    noverlap::Union{Int,Nothing}=nothing,
    window::Symbol=:hann,
)
    result = stft(x; fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    seg_len = min(nperseg, length(x))
    w = get_window(window, seg_len)
    win_norm = sum(w .^ 2)
    power = abs.(result.Zxx) .^ 2 ./ (fs * win_norm)
    power[2:(end-1), :] .*= 2.0
    return (frequencies=result.frequencies, times=result.times, power=power)
end

# ===========================================================================
# 4. DISCRETE COSINE TRANSFORM (DCT)
# ===========================================================================

"""
    dct(x::AbstractVector{Float64}; type::Int=2, norm::Bool=true) -> Vector{Float64}

Compute the Discrete Cosine Transform of type 1, 2, 3, or 4.

- **DCT-I**: `X[k] = x[1]/2 + x[n]/2 + Σ_{j=2}^{n-1} x[j] cos(π*(k-1)*(j-1)/(n-1))`
- **DCT-II** (most common, `dct` in MATLAB/SciPy):
  `X[k] = Σ_{j=1}^{n} x[j] cos(π*(k-1)*(2j-1)/(2n))`
- **DCT-III** (inverse of DCT-II, up to scaling): transform of DCT-II output
- **DCT-IV**: `X[k] = Σ_{j=1}^{n} x[j] cos(π*(2k-1)*(2j-1)/(4n))`

# Arguments
- `x`: input vector.
- `type`: DCT type (1, 2, 3, or 4; default 2).
- `norm`: if `true`, use orthonormal normalization (SciPy default; default `true`).

# Returns
- DCT coefficients.

# Examples
```julia
x = [1.0, 2.0, 3.0, 4.0]
X = dct(x)                # DCT-II, orthonormal
x_rec = dct(X; type=3)    # DCT-III = inverse DCT-II (when both are orthonormal)
```
"""
function dct(x::AbstractVector{Float64}; type::Int=2, norm::Bool=true)::Vector{Float64}
    n = length(x)
    if n < 1
        throw(ArgumentError("dct: empty input"))
    end

    if type == 1
        return _dct1(x, norm)
    elseif type == 2
        return _dct2(x, norm)
    elseif type == 3
        return _dct3(x, norm)
    elseif type == 4
        return _dct4(x, norm)
    else
        throw(ArgumentError("dct: type must be 1, 2, 3, or 4, got $type"))
    end
end

function _dct1(x::AbstractVector{Float64}, norm::Bool)::Vector{Float64}
    n = length(x)
    if n < 2
        return copy(x)
    end
    X = Vector{Float64}(undef, n)
    for k in 0:(n-1)
        s = x[1] / 2.0 + x[n] * (k % 2 == 0 ? 0.5 : -0.5)
        for j in 2:(n-1)
            s += x[j] * cos(π * k * (j - 1) / (n - 1))
        end
        X[k + 1] = norm ? s * sqrt(2.0 / (n - 1)) : 2.0 * s
    end
    if norm
        X[1] /= sqrt(2.0)
        X[n] /= sqrt(2.0)
    end
    return X
end

function _dct2(x::AbstractVector{Float64}, norm::Bool)::Vector{Float64}
    n = length(x)
    X = Vector{Float64}(undef, n)
    for k in 0:(n-1)
        s = sum(x[j] * cos(π * k * (2.0 * j - 1) / (2.0 * n)) for j in 1:n)
        if norm
            factor = k == 0 ? sqrt(1.0 / (4.0 * n)) : sqrt(1.0 / (2.0 * n))
            X[k + 1] = 2.0 * factor * s
        else
            X[k + 1] = 2.0 * s
        end
    end
    return X
end

function _dct3(x::AbstractVector{Float64}, norm::Bool)::Vector{Float64}
    n = length(x)
    X = Vector{Float64}(undef, n)
    for j in 0:(n-1)
        s = if norm
            x[1] * sqrt(1.0 / (4.0 * n)) +
            sum(x[k + 1] * sqrt(1.0 / (2.0 * n)) * cos(π * k * (2.0 * j + 1) / (2.0 * n)) for k in 1:(n-1))
        else
            x[1] / 2.0 +
            sum(x[k + 1] * cos(π * k * (2.0 * j + 1) / (2.0 * n)) for k in 1:(n-1))
        end
        X[j + 1] = norm ? 2.0 * s : s
    end
    return X
end

function _dct4(x::AbstractVector{Float64}, norm::Bool)::Vector{Float64}
    n = length(x)
    X = Vector{Float64}(undef, n)
    for k in 0:(n-1)
        s = sum(x[j] * cos(π * (2.0*k + 1) * (2.0*j - 1) / (4.0*n)) for j in 1:n)
        X[k + 1] = norm ? s * sqrt(2.0 / n) : 2.0 * s
    end
    return X
end

"""
    idct(X::AbstractVector{Float64}; type::Int=2, norm::Bool=true) -> Vector{Float64}

Compute the inverse DCT.

The inverse of DCT-II is DCT-III (and vice versa), scaled appropriately.
DCT-I and DCT-IV are self-inverse (with appropriate scaling).
"""
function idct(X::AbstractVector{Float64}; type::Int=2, norm::Bool=true)::Vector{Float64}
    if type == 2
        return dct(X; type=3, norm=norm)
    elseif type == 3
        return dct(X; type=2, norm=norm)
    elseif type == 1
        n = length(X)
        scale = norm ? 1.0 : 1.0 / (2.0 * (n - 1))
        return dct(X; type=1, norm=norm) .* scale
    elseif type == 4
        return dct(X; type=4, norm=norm)
    else
        throw(ArgumentError("idct: type must be 1, 2, 3, or 4"))
    end
end

# ===========================================================================
# 5. DISCRETE SINE TRANSFORM (DST)
# ===========================================================================

"""
    dst(x::AbstractVector{Float64}; type::Int=2, norm::Bool=true) -> Vector{Float64}

Compute the Discrete Sine Transform of type 1, 2, 3, or 4.

# Arguments
- `x`: input vector.
- `type`: DST type (1, 2, 3, or 4; default 2).
- `norm`: if `true`, use orthonormal normalization.

# Returns
- DST coefficients.
"""
function dst(x::AbstractVector{Float64}; type::Int=2, norm::Bool=true)::Vector{Float64}
    n = length(x)
    if n < 1
        throw(ArgumentError("dst: empty input"))
    end

    if type == 1
        X = [sum(x[j] * sin(π * k * j / (n + 1)) for j in 1:n) for k in 1:n]
        return norm ? X .* sqrt(2.0 / (n + 1)) : 2.0 .* X
    elseif type == 2
        X = [sum(x[j] * sin(π * k * (2j - 1) / (2n)) for j in 1:n) for k in 1:n]
        if norm
            factor = [k == n ? sqrt(1.0 / (4.0 * n)) : sqrt(1.0 / (2.0 * n)) for k in 1:n]
            return 2.0 .* factor .* X
        else
            return 2.0 .* X
        end
    elseif type == 3
        X = Vector{Float64}(undef, n)
        for j in 0:(n-1)
            s = norm ?
                sum(x[k] * (k == n ? sqrt(1.0/(4n)) : sqrt(1.0/(2n))) *
                    sin(π * k * (2j + 1) / (2n)) for k in 1:n) :
                (x[n] * (-1)^(j) / 2.0 +
                 sum(x[k] * sin(π * k * (2j + 1) / (2n)) for k in 1:(n-1)))
            X[j + 1] = norm ? 2.0 * s : s
        end
        return X
    elseif type == 4
        X = [sum(x[j] * sin(π * (2k - 1) * (2j - 1) / (4n)) for j in 1:n) for k in 1:n]
        return norm ? X .* sqrt(2.0 / n) : 2.0 .* X
    else
        throw(ArgumentError("dst: type must be 1, 2, 3, or 4, got $type"))
    end
end

# ===========================================================================
# 6. HILBERT TRANSFORM AND ANALYTIC SIGNAL
# ===========================================================================

"""
    hilbert_transform(x::AbstractVector{Float64}) -> Vector{ComplexF64}

Compute the analytic signal via the Hilbert transform.

The analytic signal is:
  z[n] = x[n] + i * H{x}[n]

where H{x} is the Hilbert transform of x, computed via the FFT:
1. Compute the FFT: X = FFT(x)
2. Set negative frequencies to zero (one-sided spectrum): H = 2*X for k>0, X[0] for k=0
3. Compute the IFFT: z = IFFT(H)

The real part of z equals x; the imaginary part is the Hilbert transform.

# Arguments
- `x`: real-valued input signal.

# Returns
- Complex analytic signal of the same length.

# Examples
```julia
x = sin.(2π .* collect(0.0:0.1:9.9))
z = hilbert_transform(x)
envelope = abs.(z)     # instantaneous amplitude
phase = angle.(z)      # instantaneous phase
```
"""
function hilbert_transform(x::AbstractVector{Float64})::Vector{ComplexF64}
    n = length(x)
    if n < 2
        throw(ArgumentError("hilbert_transform: need at least 2 samples"))
    end

    X = _dft(collect(ComplexF64, x))

    # One-sided spectrum: multiply negative frequencies by 0, positive by 2
    h = zeros(ComplexF64, n)
    if n % 2 == 0
        h[1] = X[1]           # DC
        h[2:div(n,2)] = 2.0 .* X[2:div(n,2)]
        h[div(n,2) + 1] = X[div(n,2) + 1]  # Nyquist
        # negative frequencies: h[div(n,2)+2 : n] = 0
    else
        h[1] = X[1]
        h[2:div(n+1,2)] = 2.0 .* X[2:div(n+1,2)]
        # negative frequencies zero
    end

    return _idft(h)
end

"""
    instantaneous_frequency(z::AbstractVector{ComplexF64}; fs::Float64=1.0)
    -> Vector{Float64}

Compute the instantaneous frequency from an analytic signal z.

  f_inst[n] = (1 / 2π) * d(phase(z[n])) / dt

Computed via finite differences on the unwrapped phase.

# Arguments
- `z`: analytic signal (from `hilbert_transform`).
- `fs`: sampling frequency in Hz (default 1.0).

# Returns
- Instantaneous frequency at each sample (in Hz).
"""
function instantaneous_frequency(
    z::AbstractVector{ComplexF64};
    fs::Float64=1.0,
)::Vector{Float64}
    n = length(z)
    if n < 2
        throw(ArgumentError("instantaneous_frequency: need at least 2 samples"))
    end

    # Compute instantaneous phase
    phase = angle.(z)

    # Unwrap phase to avoid 2π jumps
    unwrapped = copy(phase)
    for i in 2:n
        diff_val = unwrapped[i] - unwrapped[i - 1]
        if diff_val > π
            unwrapped[i:end] .-= 2π
        elseif diff_val < -π
            unwrapped[i:end] .+= 2π
        end
    end

    # Finite differences
    freq = Vector{Float64}(undef, n)
    freq[1] = (unwrapped[2] - unwrapped[1]) * fs / (2π)
    freq[n] = (unwrapped[n] - unwrapped[n-1]) * fs / (2π)
    for i in 2:(n-1)
        freq[i] = (unwrapped[i + 1] - unwrapped[i - 1]) * fs / (4π)
    end

    return freq
end

# ===========================================================================
# 7. FREQUENCY AXIS UTILITIES
# ===========================================================================

"""
    fftfreq(n::Int; fs::Float64=1.0) -> Vector{Float64}

Return the DFT sample frequencies for a signal of length n.

Output has the form [0, 1, 2, ..., n/2-1, -n/2, ..., -1] * (fs/n)
(same convention as NumPy's `fftfreq`).

# Arguments
- `n`: signal length.
- `fs`: sampling frequency (default 1.0 gives normalized frequencies in cycles/sample).
"""
function fftfreq(n::Int; fs::Float64=1.0)::Vector{Float64}
    freqs = Vector{Float64}(undef, n)
    n_half = div(n, 2)
    for i in 0:(n_half - 1)
        freqs[i + 1] = Float64(i) * fs / n
    end
    for i in n_half:(n - 1)
        freqs[i + 1] = Float64(i - n) * fs / n
    end
    return freqs
end

"""
    rfftfreq(n::Int; fs::Float64=1.0) -> Vector{Float64}

Return DFT sample frequencies for a real FFT of length n.

Returns n÷2+1 non-negative frequencies.
"""
function rfftfreq(n::Int; fs::Float64=1.0)::Vector{Float64}
    n_half = div(n, 2) + 1
    return collect(0:(n_half-1)) .* (fs / n)
end

"""
    fftshift(x::AbstractVector) -> Vector

Shift the zero-frequency component to the center of the spectrum.

Equivalent to NumPy's `fftshift`.
"""
function fftshift(x::AbstractVector)::Vector
    n = length(x)
    pivot = div(n + 1, 2)
    return vcat(x[(pivot + 1):n], x[1:pivot])
end

# ===========================================================================
# 8. CHIRP Z-TRANSFORM (CZT)
# ===========================================================================

"""
    czt(x::AbstractVector{Float64}; m::Int=length(x),
        w::ComplexF64=exp(-2π*im/length(x)), a::ComplexF64=1.0+0.0im)
    -> Vector{ComplexF64}

Compute the Chirp Z-Transform (CZT).

The CZT evaluates the Z-transform at m equally spaced points on a spiral
contour in the complex plane:
  z_k = A * W^(-k),  k = 0, 1, ..., m-1

Setting `W = exp(-2πi/n)` and `A = 1` gives the standard DFT.

The Bluestein algorithm allows arbitrary m, n combinations with O(n log n)
complexity (via FFT).

# Arguments
- `x`: input sequence of length n.
- `m`: number of output samples (default n).
- `w`: complex ratio between successive contour points (default exp(-2πi/n),
  giving the standard DFT contour).
- `a`: starting point on the contour (default 1.0, i.e., unit circle).

# Returns
- Complex CZT output of length m.

# Applications
- Zoom FFT: evaluate DFT over a subband with higher frequency resolution.
- Arbitrary-length DFT: compute n-point DFT in O(n log n) for any n.

# Examples
```julia
# Standard DFT via CZT
x = [1.0, 2.0, 3.0, 4.0]
X_czt = czt(x)
X_dft = _dft(x)
all(isapprox.(X_czt, X_dft; atol=1e-10))   # true

# Zoom FFT: 100-point evaluation over [0.4, 0.6] normalized frequency
n = 1024
x = sin.(2π .* 0.5 .* collect(0:(n-1)))  # 0.5 normalized freq signal
m_zoom = 100
f1, f2 = 0.4, 0.6
w_zoom = exp(-2π*im*(f2-f1)/m_zoom)
a_zoom = exp(2π*im*f1)
X_zoom = czt(x; m=m_zoom, w=w_zoom, a=a_zoom)
```
"""
function czt(
    x::AbstractVector{Float64};
    m::Int=length(x),
    w::ComplexF64=exp(-2π * im / length(x)),
    a::ComplexF64=1.0 + 0.0im,
)::Vector{ComplexF64}
    return czt_complex(collect(ComplexF64, x); m=m, w=w, a=a)
end

"""
    czt_complex(x::AbstractVector{ComplexF64}; m::Int, w::ComplexF64, a::ComplexF64)
    -> Vector{ComplexF64}

Chirp Z-Transform for complex input (Bluestein's algorithm).
"""
function czt_complex(
    x::AbstractVector{ComplexF64};
    m::Int,
    w::ComplexF64,
    a::ComplexF64,
)::Vector{ComplexF64}
    n = length(x)
    if m < 1 || n < 1
        return ComplexF64[]
    end

    L = n + m - 1
    # Next power of 2 >= L for efficient FFT
    fft_len = nextpow(2, L)

    # Precompute chirp: w_n = W^(n²/2), W^(-(n²/2))
    yn = zeros(ComplexF64, fft_len)
    for k in 0:(n-1)
        yn[k + 1] = x[k + 1] * a^(-k) * w^(k^2 / 2.0)
    end

    hn = zeros(ComplexF64, fft_len)
    for k in 0:(m-1)
        hn[k + 1] = w^(-k^2 / 2.0)
    end
    for k in (fft_len - n + 1):fft_len
        j = fft_len - k
        hn[k + 1] = w^(-(j^2) / 2.0)
    end

    # Convolve via DFT
    Yn = _dft(yn)
    Hn = _dft(hn)
    Gn = _idft(Yn .* Hn)

    result = Vector{ComplexF64}(undef, m)
    for k in 0:(m-1)
        result[k + 1] = Gn[k + 1] * w^(k^2 / 2.0)
    end
    return result
end

"""Find the next power of 2 >= n."""
function nextpow(base::Int, n::Int)::Int
    p = 1
    while p < n
        p *= base
    end
    return p
end

# ===========================================================================
# EXPORTS
# ===========================================================================

export window_hann, window_hamming, window_blackman, window_bartlett
export window_flattop, window_tukey, window_kaiser, get_window
export power_spectrum, welch_psd
export stft, spectrogram
export dct, idct, dst
export hilbert_transform, instantaneous_frequency
export fftfreq, rfftfreq, fftshift
export czt, czt_complex

end  # module ExtendedFFT
