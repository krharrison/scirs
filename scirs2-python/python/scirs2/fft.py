"""Fast Fourier Transform functions.

Provides FFT routines backed by the OxiFFT-powered SciRS2 Rust implementation.
The API mirrors ``scipy.fft`` and ``numpy.fft`` for easy migration.

Functions
---------
fft         : One-dimensional discrete Fourier transform
ifft        : Inverse one-dimensional discrete Fourier transform
fft2        : Two-dimensional discrete Fourier transform
ifft2       : Inverse two-dimensional discrete Fourier transform
rfft        : Real input FFT (returns only positive frequencies)
irfft       : Inverse real FFT
fftfreq     : DFT sample frequencies
rfftfreq    : DFT sample frequencies for real input
"""

from .scirs2 import (  # noqa: F401
    fft_py as fft,
    ifft_py as ifft,
    rfft_py as rfft,
    irfft_py as irfft,
    fftfreq_py as fftfreq,
    rfftfreq_py as rfftfreq,
)

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fftfreq",
    "rfftfreq",
]
