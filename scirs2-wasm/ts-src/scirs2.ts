/**
 * @file scirs2.ts
 * @description High-level TypeScript convenience wrappers for scirs2-wasm.
 *
 * This module provides ergonomic TypeScript classes that wrap the raw WASM
 * bindings with automatic memory management helpers, NumPy-like interfaces,
 * and richer structured result types.
 *
 * Consumers should import from this file rather than from the raw pkg output.
 *
 * @example
 * ```ts
 * import { loadScirs2, Matrix, SignalProcessor, Statistics } from "./scirs2";
 *
 * await loadScirs2();
 *
 * // Matrix operations
 * const m = Matrix.fromArray2D([[1, 2], [3, 4]]);
 * const inv = m.inverse();
 * console.log(inv.toArray2D());
 *
 * // Statistics
 * const stats = Statistics.describe(new Float64Array([1, 2, 3, 4, 5]));
 *
 * // Signal processing
 * const sp = new SignalProcessor(1000);  // 1 kHz sample rate
 * const psd = sp.psd(signal, 256);
 * const spec = sp.stft(signal, 128, 64);
 * ```
 */

// ---------------------------------------------------------------------------
// Internal module handle
// ---------------------------------------------------------------------------

type WasmModule = typeof import("../pkg/scirs2_wasm");
let _wasm: WasmModule | null = null;

function requireWasm(): WasmModule {
    if (_wasm === null) {
        throw new Error(
            "scirs2-wasm not initialised. Call `await loadScirs2()` before using any API."
        );
    }
    return _wasm;
}

/**
 * Load and initialise the scirs2-wasm module.
 *
 * Must be called (and awaited) exactly once before using any other export
 * from this module.
 *
 * @param wasmPath — Optional path / URL override for the `.wasm` binary.
 */
export async function loadScirs2(wasmPath?: string): Promise<void> {
    const mod = (await import(/* @vite-ignore */ "../pkg/scirs2_wasm.js")) as WasmModule & {
        default?: (path?: string) => Promise<unknown>;
    };
    if (typeof mod.default === "function") {
        await mod.default(wasmPath);
    }
    // Invoke the wasm_bindgen start function to install panic hooks
    if (typeof (mod as Record<string, unknown>)["init"] === "function") {
        (mod as Record<string, unknown>)["init"];
    }
    _wasm = mod;
}

// ===========================================================================
// Matrix class — NumPy-like 2-D array
// ===========================================================================

/**
 * A 2-D dense matrix backed by a `Float64Array` (row-major layout).
 *
 * Provides a NumPy-like API for matrix construction, inspection, and
 * arithmetic.  For computationally intensive operations the matrix is
 * automatically lifted to a `WasmMatrix` object that executes in WASM.
 *
 * All methods that return a new `Matrix` create a fresh copy and do not
 * mutate the receiver.
 *
 * @example
 * ```ts
 * // Construct from a 2-D array
 * const a = Matrix.fromArray2D([[1, 2], [3, 4]]);
 *
 * // NumPy-like interface
 * console.log(a.shape);   // [2, 2]
 * console.log(a.T);       // transposed view
 *
 * // Arithmetic
 * const b = a.multiply(a);    // matrix product A²
 * const c = a.add(b);         // element-wise add
 *
 * // Decompositions
 * const d = a.det();
 * const inv = a.inverse();
 * const evs = a.eigenvalues();
 *
 * // Convert back to JS
 * const arr = a.toArray2D();
 * const flat = a.data;
 * ```
 */
export class Matrix {
    readonly data: Float64Array;
    readonly rows: number;
    readonly cols: number;

    // -- Construction ---------------------------------------------------------

    constructor(data: Float64Array | number[], rows: number, cols: number) {
        if (data.length !== rows * cols) {
            throw new RangeError(
                `Data length ${data.length} does not match shape [${rows}, ${cols}]`
            );
        }
        this.data = data instanceof Float64Array ? data : new Float64Array(data);
        this.rows = rows;
        this.cols = cols;
    }

    /** Create an all-zeros matrix. */
    static zeros(rows: number, cols: number): Matrix {
        return new Matrix(new Float64Array(rows * cols), rows, cols);
    }

    /** Create an n×n identity matrix. */
    static identity(n: number): Matrix {
        const d = new Float64Array(n * n);
        for (let i = 0; i < n; i++) d[i * n + i] = 1;
        return new Matrix(d, n, n);
    }

    /** Create a matrix filled with a constant value. */
    static full(rows: number, cols: number, value: number): Matrix {
        return new Matrix(new Float64Array(rows * cols).fill(value), rows, cols);
    }

    /** Create a diagonal matrix from a vector. */
    static diag(values: number[]): Matrix {
        const n = values.length;
        const d = new Float64Array(n * n);
        for (let i = 0; i < n; i++) d[i * n + i] = values[i];
        return new Matrix(d, n, n);
    }

    /**
     * Construct from a row-of-rows 2-D array.
     * @throws {RangeError} if rows are jagged (different lengths).
     */
    static fromArray2D(data: number[][]): Matrix {
        const rows = data.length;
        if (rows === 0) throw new RangeError("Empty 2-D array");
        const cols = data[0].length;
        const flat = new Float64Array(rows * cols);
        for (let r = 0; r < rows; r++) {
            if (data[r].length !== cols) {
                throw new RangeError(`Row ${r} has length ${data[r].length}, expected ${cols}`);
            }
            for (let c = 0; c < cols; c++) {
                flat[r * cols + c] = data[r][c];
            }
        }
        return new Matrix(flat, rows, cols);
    }

    /** Create from a `WasmMatrix` object (copies data into JS). */
    static fromWasmMatrix(m: ReturnType<WasmModule["WasmMatrix"]["prototype"]["transpose"]>): Matrix {
        // WasmMatrix.to_vec() returns number[]
        const vec = m.to_vec() as number[];
        return new Matrix(new Float64Array(vec), m.rows() as number, m.cols() as number);
    }

    // -- Shape / access -------------------------------------------------------

    /** Shape as `[rows, cols]`. */
    get shape(): [number, number] {
        return [this.rows, this.cols];
    }

    /** Transposed view (creates a new Matrix). */
    get T(): Matrix {
        const wasm = requireWasm();
        const out = wasm.wasm_transpose(this.data, this.rows, this.cols);
        return new Matrix(new Float64Array(out), this.cols, this.rows);
    }

    get(r: number, c: number): number {
        return this.data[r * this.cols + c];
    }

    set(r: number, c: number, v: number): void {
        this.data[r * this.cols + c] = v;
    }

    /** Extract a single row as `Float64Array`. */
    row(r: number): Float64Array {
        return this.data.slice(r * this.cols, (r + 1) * this.cols);
    }

    /** Extract a single column as `Float64Array`. */
    col(c: number): Float64Array {
        const out = new Float64Array(this.rows);
        for (let r = 0; r < this.rows; r++) out[r] = this.data[r * this.cols + c];
        return out;
    }

    /** Convert to a 2-D array of arrays. */
    toArray2D(): number[][] {
        return Array.from({ length: this.rows }, (_, r) =>
            Array.from(this.data.slice(r * this.cols, (r + 1) * this.cols))
        );
    }

    /** Lift to a `WasmMatrix` (must be freed by caller). */
    toWasmMatrix(): ReturnType<WasmModule["WasmMatrix"]["prototype"]["transpose"]> {
        const wasm = requireWasm();
        return wasm.WasmMatrix.from_vec(this.rows, this.cols, Array.from(this.data));
    }

    // -- Arithmetic -----------------------------------------------------------

    /** Matrix–matrix product (requires this.cols === other.rows). */
    multiply(other: Matrix): Matrix {
        if (this.cols !== other.rows) {
            throw new RangeError(
                `Shape mismatch for multiply: [${this.rows},${this.cols}] vs [${other.rows},${other.cols}]`
            );
        }
        const wasm = requireWasm();
        const out = wasm.wasm_matmul(this.data, other.data, this.rows, this.cols, other.cols);
        return new Matrix(new Float64Array(out), this.rows, other.cols);
    }

    /** Element-wise addition (shapes must match). */
    add(other: Matrix): Matrix {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new RangeError("Shape mismatch for element-wise add");
        }
        const wasm = requireWasm();
        const out = wasm.wasm_vec_add(this.data, other.data);
        return new Matrix(new Float64Array(out), this.rows, this.cols);
    }

    /** Element-wise subtraction (shapes must match). */
    sub(other: Matrix): Matrix {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new RangeError("Shape mismatch for element-wise sub");
        }
        const wasm = requireWasm();
        const out = wasm.wasm_vec_sub(this.data, other.data);
        return new Matrix(new Float64Array(out), this.rows, this.cols);
    }

    /** Scalar multiplication. */
    scale(s: number): Matrix {
        const wasm = requireWasm();
        const out = wasm.wasm_vec_scale(this.data, s);
        return new Matrix(new Float64Array(out), this.rows, this.cols);
    }

    // -- Properties -----------------------------------------------------------

    /** Frobenius norm. */
    norm(): number {
        const wasm = requireWasm();
        return wasm.wasm_frobenius_norm(this.data);
    }

    /** Trace (sum of diagonal). Requires square matrix. */
    trace(): number {
        if (this.rows !== this.cols) throw new RangeError("Trace requires a square matrix");
        const wm = this.toWasmMatrix();
        try {
            return wm.trace() as number;
        } finally {
            wm.free();
        }
    }

    /** Determinant. Requires square matrix. */
    det(): number {
        if (this.rows !== this.cols) throw new RangeError("Determinant requires a square matrix");
        const wm = this.toWasmMatrix();
        try {
            return wm.determinant() as number;
        } finally {
            wm.free();
        }
    }

    /**
     * Matrix inverse. Requires square, non-singular matrix.
     * @throws {Error} if the matrix is singular or not square.
     */
    inverse(): Matrix {
        if (this.rows !== this.cols) throw new RangeError("Inverse requires a square matrix");
        const wm = this.toWasmMatrix();
        try {
            const inv = wm.inverse();
            const result = Matrix.fromWasmMatrix(inv);
            inv.free();
            return result;
        } finally {
            wm.free();
        }
    }

    /** Eigenvalues (real parts) via QR iteration. */
    eigenvalues(): number[] {
        if (this.rows !== this.cols) throw new RangeError("Eigenvalues require a square matrix");
        const wm = this.toWasmMatrix();
        try {
            return wm.eigenvalues() as number[];
        } finally {
            wm.free();
        }
    }

    /**
     * Thin QR decomposition.
     * Returns { Q: Matrix, R: Matrix } where Q has orthonormal columns and R is upper-triangular.
     */
    qr(): { Q: Matrix; R: Matrix } {
        const wasm = requireWasm();
        const k = Math.min(this.rows, this.cols);
        const flat = new Float64Array(wasm.wasm_qr(this.data, this.rows, this.cols));
        return {
            Q: new Matrix(flat.slice(0, this.rows * k), this.rows, k),
            R: new Matrix(flat.slice(this.rows * k), k, this.cols),
        };
    }

    /**
     * Power-iteration thin SVD.
     * Returns { U, S, V, k } for the `nComponents` largest singular triplets.
     */
    svd(nComponents?: number): { U: Matrix; S: Float64Array; V: Matrix; k: number } {
        const wasm = requireWasm();
        const k = Math.min(nComponents ?? Math.min(this.rows, this.cols), Math.min(this.rows, this.cols));
        const flat = new Float64Array(wasm.wasm_svd_power(this.data, this.rows, this.cols, k));
        return {
            U: new Matrix(flat.slice(0, this.rows * k), this.rows, k),
            S: flat.slice(this.rows * k, this.rows * k + k),
            V: new Matrix(flat.slice(this.rows * k + k), this.cols, k),
            k,
        };
    }

    /**
     * Solve the linear system A·x = b.
     * Requires square matrix.
     */
    solve(b: Float64Array | number[]): Float64Array {
        const wasm = requireWasm();
        const bArr = b instanceof Float64Array ? b : new Float64Array(b);
        const x = wasm.wasm_matrix_solve(this.data, bArr, this.rows);
        return new Float64Array(x);
    }
}

// ===========================================================================
// Statistics class
// ===========================================================================

/**
 * Structured result of {@link Statistics.describe}.
 */
export interface DescriptiveStatsResult {
    readonly mean: number;
    readonly variance: number;
    readonly std_dev: number;
    readonly min: number;
    readonly max: number;
    readonly median: number;
    readonly skewness: number;
    readonly kurtosis: number;
    readonly count: number;
    readonly sum: number;
    readonly q25: number;
    readonly q75: number;
    readonly iqr: number;
}

/**
 * Structured result of {@link Statistics.linReg}.
 */
export interface LinearModelResult {
    readonly slope: number;
    readonly intercept: number;
    readonly r_squared: number;
    readonly std_err_slope: number;
    readonly std_err_intercept: number;
    readonly n: number;
    /** Predict ŷ for a given x. */
    predict(x: number): number;
}

/**
 * Structured result of hypothesis tests.
 */
export interface HypothesisTestResult {
    readonly t_stat: number;
    readonly p_value: number;
    readonly df: number;
    readonly significant_at_05: boolean;
    readonly significant_at_01: boolean;
}

/**
 * Structured result of the K-S normality test.
 */
export interface KsTestResult {
    readonly statistic: number;
    readonly p_value: number;
    readonly is_normal_at_05: boolean;
}

/**
 * Structured histogram result.
 */
export interface HistogramResult {
    readonly edges: Float64Array;
    readonly counts: Uint32Array;
    readonly density: Float64Array;
    readonly n_bins: number;
    readonly total: number;
    readonly bin_centers: Float64Array;
    readonly bin_width: number;
}

/**
 * Namespace of static statistical methods.
 *
 * All methods return plain JS objects (no WASM memory management needed by
 * the caller — memory is freed internally).
 *
 * @example
 * ```ts
 * const ds = Statistics.describe(new Float64Array([1, 2, 3, 4, 5]));
 * console.log(ds.mean, ds.skewness, ds.iqr);
 *
 * const model = Statistics.linReg(xs, ys);
 * console.log(model.slope, model.r_squared);
 * console.log(model.predict(6.0));
 * ```
 */
export class Statistics {
    private constructor() {}

    /**
     * Compute comprehensive descriptive statistics.
     * Automatically frees WASM memory before returning.
     */
    static describe(data: Float64Array | number[]): DescriptiveStatsResult {
        const wasm = requireWasm();
        const arr = data instanceof Float64Array ? data : new Float64Array(data);
        const ds = wasm.compute_descriptive_stats(arr);
        try {
            return {
                mean: ds.mean,
                variance: ds.variance,
                std_dev: ds.std_dev,
                min: ds.min,
                max: ds.max,
                median: ds.median,
                skewness: ds.skewness,
                kurtosis: ds.kurtosis,
                count: ds.count,
                sum: ds.sum,
                q25: ds.q25,
                q75: ds.q75,
                iqr: ds.iqr,
            };
        } finally {
            ds.free();
        }
    }

    /**
     * Fit OLS simple linear regression y = slope·x + intercept.
     * Returns a plain object including a `predict(x)` method.
     */
    static linReg(
        x: Float64Array | number[],
        y: Float64Array | number[]
    ): LinearModelResult {
        const wasm = requireWasm();
        const xa = x instanceof Float64Array ? x : new Float64Array(x);
        const ya = y instanceof Float64Array ? y : new Float64Array(y);
        const model = wasm.wasm_linear_regression_typed(xa, ya);
        try {
            const slope = model.slope;
            const intercept = model.intercept;
            return {
                slope,
                intercept,
                r_squared: model.r_squared,
                std_err_slope: model.std_err_slope,
                std_err_intercept: model.std_err_intercept,
                n: model.n,
                predict(xVal: number): number {
                    return slope * xVal + intercept;
                },
            };
        } finally {
            model.free();
        }
    }

    /**
     * One-sample Student's t-test (H₀: mean = mu).
     */
    static tTestOne(data: Float64Array | number[], mu: number): HypothesisTestResult {
        const wasm = requireWasm();
        const arr = data instanceof Float64Array ? data : new Float64Array(data);
        const r = wasm.wasm_t_test_one_sample_typed(arr, mu);
        try {
            return {
                t_stat: r.t_stat,
                p_value: r.p_value,
                df: r.df,
                significant_at_05: r.significant_at_05,
                significant_at_01: r.significant_at_01,
            };
        } finally {
            r.free();
        }
    }

    /**
     * Two-sample Welch t-test (H₀: mean_x = mean_y).
     */
    static tTestTwo(
        x: Float64Array | number[],
        y: Float64Array | number[]
    ): HypothesisTestResult {
        const wasm = requireWasm();
        const xa = x instanceof Float64Array ? x : new Float64Array(x);
        const ya = y instanceof Float64Array ? y : new Float64Array(y);
        const r = wasm.wasm_t_test_two_sample_typed(xa, ya);
        try {
            return {
                t_stat: r.t_stat,
                p_value: r.p_value,
                df: r.df,
                significant_at_05: r.significant_at_05,
                significant_at_01: r.significant_at_01,
            };
        } finally {
            r.free();
        }
    }

    /**
     * Kolmogorov–Smirnov normality test.
     */
    static ksTest(data: Float64Array | number[]): KsTestResult {
        const wasm = requireWasm();
        const arr = data instanceof Float64Array ? data : new Float64Array(data);
        const r = wasm.wasm_ks_test_normality(arr);
        try {
            return {
                statistic: r.statistic,
                p_value: r.p_value,
                is_normal_at_05: r.is_normal_at_05,
            };
        } finally {
            r.free();
        }
    }

    /**
     * Compute a histogram with n_bins equal-width bins.
     */
    static histogram(data: Float64Array | number[], nBins: number): HistogramResult {
        const wasm = requireWasm();
        const arr = data instanceof Float64Array ? data : new Float64Array(data);
        const h = wasm.wasm_histogram(arr, nBins);
        try {
            return {
                edges: new Float64Array(h.edges),
                counts: new Uint32Array(h.counts),
                density: new Float64Array(h.density),
                n_bins: h.n_bins,
                total: h.total,
                bin_centers: new Float64Array(h.bin_centers()),
                bin_width: h.bin_width(),
            };
        } finally {
            h.free();
        }
    }

    /**
     * Pearson correlation coefficient ρ ∈ [-1, 1].
     */
    static pearsonR(x: Float64Array | number[], y: Float64Array | number[]): number {
        const wasm = requireWasm();
        const xa = x instanceof Float64Array ? x : new Float64Array(x);
        const ya = y instanceof Float64Array ? y : new Float64Array(y);
        return wasm.wasm_pearson_correlation(xa, ya) as number;
    }

    /**
     * Spearman rank correlation coefficient ρ ∈ [-1, 1].
     */
    static spearmanR(x: Float64Array | number[], y: Float64Array | number[]): number {
        const wasm = requireWasm();
        const xa = x instanceof Float64Array ? x : new Float64Array(x);
        const ya = y instanceof Float64Array ? y : new Float64Array(y);
        return wasm.wasm_spearman_correlation_typed(xa, ya) as number;
    }

    /**
     * Arithmetic mean. Returns `NaN` for empty arrays.
     */
    static mean(data: Float64Array | number[]): number {
        const wasm = requireWasm();
        const arr = data instanceof Float64Array ? data : new Float64Array(data);
        return wasm.wasm_mean(arr) as number;
    }

    /**
     * Sample standard deviation (ddof=1). Returns `NaN` for fewer than 2 samples.
     */
    static std(data: Float64Array | number[]): number {
        const wasm = requireWasm();
        const arr = data instanceof Float64Array ? data : new Float64Array(data);
        return wasm.wasm_std(arr) as number;
    }
}

// ===========================================================================
// SignalProcessor class
// ===========================================================================

/**
 * Paired FFT result with interleaved spectrum and convenience magnitude/phase.
 */
export interface FftResult {
    /** Interleaved (re, im) pairs. Length = 2 * N (N = padded signal length). */
    readonly complex: Float64Array;
    /** Magnitude spectrum. Length = N. */
    readonly magnitude: Float64Array;
    /** Phase spectrum in radians. Length = N. */
    readonly phase: Float64Array;
    /** Length of the padded FFT (N). */
    readonly fftSize: number;
}

/**
 * PSD result from {@link SignalProcessor.psd}.
 */
export interface PsdResult {
    /** One-sided PSD values (units: signal²/Hz). */
    readonly values: Float64Array;
    /** Frequency axis in Hz. */
    readonly frequencies: Float64Array;
}

/**
 * STFT result from {@link SignalProcessor.stft}.
 */
export interface StftResult {
    /**
     * Flattened magnitude spectrogram (row-major).
     * Access element (frame, bin): `data[frame * n_freq + bin]`.
     */
    readonly data: Float64Array;
    /** Number of time frames. */
    readonly n_frames: number;
    /** Number of frequency bins (= window_size / 2 + 1). */
    readonly n_freq: number;
    /** Frequency bin centres in Hz. */
    readonly frequencies: Float64Array;
    /** Time stamp in seconds for each frame centre. */
    readonly times: Float64Array;
}

/**
 * Signal processor with a bound sample rate.
 *
 * Wraps the enhanced signal-processing WASM bindings with a NumPy/SciPy-like
 * interface.  All methods are stateless except for the configured `sampleRate`.
 *
 * @example
 * ```ts
 * const sp = new SignalProcessor(44100);  // 44.1 kHz audio
 *
 * // Forward FFT
 * const { complex, magnitude } = sp.fft(signal);
 *
 * // Power spectral density
 * const { values, frequencies } = sp.psd(signal, 1024);
 *
 * // STFT spectrogram
 * const { data, n_frames, n_freq, frequencies: freqs, times } =
 *     sp.stft(signal, 512, 256);
 *
 * // Butterworth low-pass filter
 * const filtered = sp.butterLowpass(signal, 4000);  // -3dB at 4 kHz
 *
 * // Moving average
 * const smoothed = sp.movingAverage(signal, 10);
 *
 * // Convolution
 * const conv = sp.convolve(signal, kernel);
 * ```
 */
export class SignalProcessor {
    readonly sampleRate: number;

    constructor(sampleRate: number) {
        if (sampleRate <= 0) throw new RangeError("sampleRate must be positive");
        this.sampleRate = sampleRate;
    }

    /**
     * Forward FFT of a real-valued signal.
     *
     * The signal is zero-padded to the next power of two internally.
     * Returns interleaved complex output plus magnitude and phase for convenience.
     */
    fft(signal: Float64Array | number[]): FftResult {
        const wasm = requireWasm();
        const arr = signal instanceof Float64Array ? signal : new Float64Array(signal);
        const complex = new Float64Array(wasm.wasm_fft_real(arr));
        const fftSize = complex.length / 2;
        const magnitude = new Float64Array(wasm.wasm_fft_magnitude(complex));
        const phase = new Float64Array(wasm.wasm_fft_phase(complex));
        return { complex, magnitude, phase, fftSize };
    }

    /**
     * Inverse FFT (real part only).
     *
     * @param spectrum — interleaved (re, im) array from {@link fft}.
     */
    ifft(spectrum: Float64Array | number[]): Float64Array {
        const wasm = requireWasm();
        const arr = spectrum instanceof Float64Array ? spectrum : new Float64Array(spectrum);
        return new Float64Array(wasm.wasm_ifft_real(arr));
    }

    /**
     * One-sided Power Spectral Density using Welch's method.
     *
     * @param signal      — real-valued input signal.
     * @param segmentLen  — Welch segment length (rounded to next power of two).
     */
    psd(signal: Float64Array | number[], segmentLen: number): PsdResult {
        const wasm = requireWasm();
        const arr = signal instanceof Float64Array ? signal : new Float64Array(signal);
        const values = new Float64Array(wasm.wasm_power_spectral_density(arr, this.sampleRate, segmentLen));
        // Frequency axis: nfft is the next pow2 >= segmentLen
        let nfft = 1;
        while (nfft < segmentLen) nfft <<= 1;
        const frequencies = new Float64Array(wasm.wasm_fft_frequencies(nfft, this.sampleRate));
        return { values, frequencies };
    }

    /**
     * Short-Time Fourier Transform magnitude spectrogram.
     *
     * @param signal     — real-valued input signal.
     * @param windowSize — Hann window size in samples.
     * @param hopSize    — step between consecutive frames in samples.
     */
    stft(signal: Float64Array | number[], windowSize: number, hopSize: number): StftResult {
        const wasm = requireWasm();
        const arr = signal instanceof Float64Array ? signal : new Float64Array(signal);
        const data = new Float64Array(wasm.wasm_stft(arr, windowSize, hopSize));

        // Derive shape
        let nfft = 1;
        while (nfft < windowSize) nfft <<= 1;
        const n_freq = nfft / 2 + 1;
        const n_frames = data.length > 0 ? data.length / n_freq : 0;

        // Frequency axis
        const frequencies = new Float64Array(wasm.wasm_fft_frequencies(nfft, this.sampleRate));

        // Time stamps: centre of each frame
        const times = Float64Array.from({ length: n_frames }, (_, i) =>
            ((i * hopSize + windowSize / 2) / this.sampleRate)
        );

        return { data, n_frames, n_freq, frequencies, times };
    }

    /**
     * Linear convolution of `signal` and `kernel` (full mode).
     * Automatically selects direct or FFT-based computation.
     */
    convolve(signal: Float64Array | number[], kernel: Float64Array | number[]): Float64Array {
        const wasm = requireWasm();
        const sa = signal instanceof Float64Array ? signal : new Float64Array(signal);
        const ka = kernel instanceof Float64Array ? kernel : new Float64Array(kernel);
        return new Float64Array(wasm.wasm_convolution_1d(sa, ka));
    }

    /**
     * Causal boxcar moving average.
     *
     * @param signal — input signal.
     * @param window — averaging window length.
     */
    movingAverage(signal: Float64Array | number[], window: number): Float64Array {
        const wasm = requireWasm();
        const arr = signal instanceof Float64Array ? signal : new Float64Array(signal);
        return new Float64Array(wasm.wasm_moving_average_simple(arr, window));
    }

    /**
     * Second-order Butterworth low-pass IIR filter.
     *
     * @param signal    — input real-valued signal.
     * @param cutoffHz  — -3 dB cutoff frequency in Hz.
     */
    butterLowpass(signal: Float64Array | number[], cutoffHz: number): Float64Array {
        const wasm = requireWasm();
        const arr = signal instanceof Float64Array ? signal : new Float64Array(signal);
        return new Float64Array(wasm.wasm_butter_lowpass(arr, cutoffHz, this.sampleRate));
    }

    /**
     * Compute frequency axis for an FFT of `n` samples.
     *
     * @returns `Float64Array` of length `n / 2 + 1` with values in Hz.
     */
    fftFrequencies(n: number): Float64Array {
        const wasm = requireWasm();
        return new Float64Array(wasm.wasm_fft_frequencies(n, this.sampleRate));
    }

    /**
     * Magnitude (absolute value) of an interleaved FFT output.
     */
    magnitude(spectrum: Float64Array | number[]): Float64Array {
        const wasm = requireWasm();
        const arr = spectrum instanceof Float64Array ? spectrum : new Float64Array(spectrum);
        return new Float64Array(wasm.wasm_fft_magnitude(arr));
    }

    /**
     * Phase (argument) in radians of an interleaved FFT output.
     */
    phase(spectrum: Float64Array | number[]): Float64Array {
        const wasm = requireWasm();
        const arr = spectrum instanceof Float64Array ? spectrum : new Float64Array(spectrum);
        return new Float64Array(wasm.wasm_fft_phase(arr));
    }
}

// ===========================================================================
// Convenience re-exports
// ===========================================================================

export { requireWasm as _requireWasm };
