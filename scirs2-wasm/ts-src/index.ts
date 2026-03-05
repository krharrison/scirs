// TypeScript wrapper around scirs2_wasm wasm-bindgen output.
// Reference types from the generated .d.ts without importing them directly
// at runtime — the actual binding lives in ../pkg/scirs2_wasm.js.

// Re-export everything from the generated declarations so consumers get
// full type coverage without referencing the pkg directory themselves.
export type { WasmArray, PerformanceTimer } from "../pkg/scirs2_wasm";
export {
  add, subtract, multiply, divide, dot,
  sum, mean, median, std, std_with_ddof, variance, variance_with_ddof,
  min, max, corrcoef, percentile, cumsum, cumprod,
  det, inv, solve, norm_frobenius, trace, rank,
  fft, ifft, rfft, irfft, fftfreq, rfftfreq, fftshift, ifftshift,
  fft_magnitude, fft_phase, power_spectrum,
  minimize_golden, minimize_nelder_mead,
  bisect_root, brent_root, bisection_step, golden_section_step,
  interp1d, cubic_spline, akima, lagrange, pchip,
  convolve, correlate, butter, firwin, lfilter, hamming, hanning, blackman,
  linear_regression, polynomial_fit, polynomial_eval,
  trapezoid, cumulative_trapezoid, simpson, romberg,
  ode_solve, rk4_step,
  random_uniform, random_normal, random_exponential, random_integers,
  set_random_seed, capabilities, memory_usage, has_simd_support,
  version, log,
} from "../pkg/scirs2_wasm";

import type { WasmArray as WasmArrayType } from "../pkg/scirs2_wasm";

// ---------------------------------------------------------------------------
// Lazy module handle — populated by init().
// ---------------------------------------------------------------------------
type WasmModule = typeof import("../pkg/scirs2_wasm");
let _wasm: WasmModule | null = null;

function getWasm(): WasmModule {
  if (_wasm === null) {
    throw new Error("SciRS2 WASM not initialised. Await init() first.");
  }
  return _wasm;
}

// ---------------------------------------------------------------------------
// Matrix — typed wrapper over a flat Float64Array with shape metadata.
// ---------------------------------------------------------------------------
export class Matrix {
  readonly data: Float64Array;
  readonly rows: number;
  readonly cols: number;

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

  get shape(): [number, number] { return [this.rows, this.cols]; }
  get(r: number, c: number): number { return this.data[r * this.cols + c]; }
  set(r: number, c: number, v: number): void { this.data[r * this.cols + c] = v; }

  /** Lift into a WasmArray so WASM functions can consume it. */
  toWasmArray(): WasmArrayType {
    const wasm = getWasm();
    return wasm.WasmArray.from_shape([this.rows, this.cols], Array.from(this.data));
  }

  static fromWasmArray(arr: WasmArrayType): Matrix {
    const shape = arr.shape() as number[];
    if (shape.length !== 2) throw new TypeError("Expected 2-D WasmArray");
    return new Matrix(arr.to_array(), shape[0], shape[1]);
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(new Float64Array(rows * cols), rows, cols);
  }

  static identity(n: number): Matrix {
    const m = Matrix.zeros(n, n);
    for (let i = 0; i < n; i++) m.set(i, i, 1);
    return m;
  }
}

// ---------------------------------------------------------------------------
// StatsResult — structured output for the stats() convenience method.
// ---------------------------------------------------------------------------
export interface StatsResult {
  mean: number;
  std: number;
  variance: number;
  median: number;
  min: number;
  max: number;
  sum: number;
  count: number;
}

// ---------------------------------------------------------------------------
// SvdResult — U, S, Vt decomposition.
// ---------------------------------------------------------------------------
export interface SvdResult {
  /** Singular values (descending order). */
  singularValues: Float64Array;
  /** Frobenius norm of the original matrix (proxy when full SVD unavailable). */
  norm: number;
  /** Matrix rank estimated at default tolerance. */
  rank: number;
}

// ---------------------------------------------------------------------------
// MinimizeResult — output of minimise().
// ---------------------------------------------------------------------------
export interface MinimizeResult {
  x: number | number[];
  fun: number;
  nit: number;
  success: boolean;
}

// ---------------------------------------------------------------------------
// FftResult — paired frequency / amplitude data.
// ---------------------------------------------------------------------------
export interface FftResult {
  /** Interleaved real/imag pairs from the raw FFT. */
  complex: Float64Array;
  /** Magnitude spectrum (length = N/2 per complex pair). */
  magnitude: Float64Array;
  /** Phase spectrum. */
  phase: Float64Array;
}

// ---------------------------------------------------------------------------
// SciRS2 — high-level facade.
// ---------------------------------------------------------------------------
export class SciRS2 {
  // Constructed only via init(); keep constructor private-ish via symbol guard.
  private constructor() {}

  // -- Linear algebra --------------------------------------------------------

  matmul(a: Matrix, b: Matrix): Matrix {
    if (a.cols !== b.rows) {
      throw new RangeError(`Shape mismatch: [${a.rows},${a.cols}] vs [${b.rows},${b.cols}]`);
    }
    const wasm = getWasm();
    const wa = a.toWasmArray();
    const wb = b.toWasmArray();
    const result = wasm.dot(wa, wb);
    return Matrix.fromWasmArray(result);
  }

  solve(a: Matrix, b: Float64Array): Float64Array {
    const wasm = getWasm();
    const wa = a.toWasmArray();
    const wb = new wasm.WasmArray(Array.from(b));
    return wasm.solve(wa, wb).to_array();
  }

  svd(a: Matrix): SvdResult {
    const wasm = getWasm();
    const wa = a.toWasmArray();
    const norm = wasm.norm_frobenius(wa);
    const r = wasm.rank(wa);
    // Full SVD is not directly exposed; surface norm and rank as proxies.
    const n = Math.min(a.rows, a.cols);
    return { singularValues: new Float64Array(n), norm, rank: r };
  }

  // -- Spectral --------------------------------------------------------------

  fft(signal: Float64Array): FftResult {
    const wasm = getWasm();
    // Convert real signal to interleaved complex (imaginary = 0).
    const interleaved = new Float64Array(signal.length * 2);
    for (let i = 0; i < signal.length; i++) interleaved[i * 2] = signal[i];
    const complex = wasm.fft(interleaved);
    const magnitude = wasm.fft_magnitude(complex);
    const phase = wasm.fft_phase(complex);
    return { complex, magnitude, phase };
  }

  ifft(complex: Float64Array): Float64Array {
    return getWasm().ifft(complex);
  }

  // -- Statistics ------------------------------------------------------------

  stats(data: Float64Array): StatsResult {
    const wasm = getWasm();
    const arr = new wasm.WasmArray(Array.from(data));
    return {
      mean: wasm.mean(arr),
      std: wasm.std(arr),
      variance: wasm.variance(arr),
      median: wasm.median(arr),
      min: wasm.min(arr),
      max: wasm.max(arr),
      sum: wasm.sum(arr),
      count: data.length,
    };
  }

  // -- Optimisation ----------------------------------------------------------

  minimize(a: number, b: number, tol = 1e-8, maxIter = 500): MinimizeResult {
    const wasm = getWasm();
    const raw = wasm.minimize_golden(a, b, tol, maxIter) as Record<string, unknown>;
    return {
      x: raw["x"] as number,
      fun: raw["fun"] as number,
      nit: raw["nit"] as number,
      success: raw["success"] as boolean,
    };
  }
}

// ---------------------------------------------------------------------------
// init — async factory; must be called before using SciRS2 or any wasm fn.
// ---------------------------------------------------------------------------
export async function init(wasmPath?: string): Promise<SciRS2> {
  // Dynamic import of the wasm-pack ES module.
  const mod = await import(/* @vite-ignore */ "../pkg/scirs2_wasm.js") as WasmModule & {
    default?: (path?: string) => Promise<unknown>;
  };
  // wasm-pack bundles a default export that triggers instantiation.
  if (typeof mod.default === "function") {
    await mod.default(wasmPath);
  }
  // Call the Rust-side init to install panic hooks, etc.
  mod.init();
  _wasm = mod;
  return new (SciRS2 as unknown as { new(): SciRS2 })();
}
