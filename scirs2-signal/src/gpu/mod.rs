// GPU-accelerated (CPU fallback) batched signal processing
//
// Note: "GPU" here means batch-optimised CPU implementations backed by OxiFFT
// (via scirs2-fft). No CUDA/ROCm dependencies are required; the module is named
// `gpu` to reflect its intended use-case as a drop-in for GPU batch workloads.
//
// # Sub-modules
//
// | Module               | Description                                          |
// |----------------------|------------------------------------------------------|
// | `batched_stft`       | Batched Short-Time Fourier Transform (STFT)          |
// | `batched_welch`      | Batched Welch power-spectral-density estimator       |
// | `matched_filter_bank`| FFT-based matched filter bank with peak detection    |
// | `fast_wavelet`       | FFT-convolution DWT/IDWT with Haar, Db4, Db8         |

pub mod batched_stft;
pub mod batched_welch;
pub mod fast_wavelet;
pub mod matched_filter_bank;

// ---------------------------------------------------------------------------
// Flat re-exports – BatchWindowType is shared between the two STFT/Welch mods
// ---------------------------------------------------------------------------
pub use batched_stft::{BatchWindowType, BatchedStftConfig, BatchedStftEngine};
pub use batched_welch::{batched_welch_psd, BatchedWelchConfig, WelchScaling};
pub use fast_wavelet::{fast_dwt, fast_dwt_batch, fast_idwt, FastDwtConfig, FastWaveletType};
pub use matched_filter_bank::{Detection, MatchedFilterBank, MatchedFilterConfig};
