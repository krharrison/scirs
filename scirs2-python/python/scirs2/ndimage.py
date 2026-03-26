"""N-dimensional image processing.

Provides image processing routines backed by the SciRS2 Rust implementation.
The API mirrors ``scipy.ndimage`` for easy migration.

Functions
---------
gaussian_filter     : Multi-dimensional Gaussian filter
median_filter       : Multi-dimensional median filter
sobel               : Sobel edge detection
label               : Label connected components
binary_dilation     : Binary dilation
binary_erosion      : Binary erosion
binary_opening      : Binary opening (erosion then dilation)
binary_closing      : Binary closing (dilation then erosion)
zoom                : Zoom/rescale an array
rotate              : Rotate an array
"""

from .scirs2 import (  # noqa: F401
    gaussian_filter_py as gaussian_filter,
    median_filter_py as median_filter,
    sobel_py as sobel,
)

__all__ = [
    "gaussian_filter",
    "median_filter",
    "sobel",
]
