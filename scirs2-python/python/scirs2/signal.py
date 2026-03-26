"""Signal processing functions.

Provides signal processing routines backed by the SciRS2 Rust implementation.
The API mirrors ``scipy.signal`` for easy migration.

Functions
---------
convolve        : Convolve two N-dimensional arrays
correlate       : Cross-correlate two N-dimensional arrays
fftconvolve     : Convolve two arrays using the FFT method
butter          : Butterworth digital filter design
cheby1          : Chebyshev type I digital filter design
sosfilt         : Filter data along one dimension using second-order sections
filtfilt        : Zero-phase digital filter
stft            : Short-time Fourier transform
istft           : Inverse short-time Fourier transform
spectrogram     : Compute a spectrogram with consecutive Fourier transforms
welch           : Power spectral density by Welch's method
find_peaks      : Find peaks in a 1D array
"""

from .scirs2 import (  # noqa: F401
    convolve_py as convolve,
    correlate_py as correlate,
    stft_py as stft,
    istft_py as istft,
    find_peaks_py as find_peaks,
)

__all__ = [
    "convolve",
    "correlate",
    "stft",
    "istft",
    "find_peaks",
]
