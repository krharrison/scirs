"""Statistical functions and distributions.

Provides statistical distributions, hypothesis tests, and descriptive
statistics backed by the SciRS2 Rust implementation.  The API mirrors
``scipy.stats`` for easy migration.

Classes
-------
Normal      : Normal (Gaussian) distribution
Beta        : Beta distribution
Gamma       : Gamma distribution
Binomial    : Binomial distribution
Poisson     : Poisson distribution

Functions
---------
mean        : Arithmetic mean
std         : Standard deviation
var         : Variance
median      : Median
ttest_1samp : One-sample t-test
ttest_ind   : Independent two-sample t-test
ttest_rel   : Paired t-test
chi2_test   : Chi-squared test
ks_test     : Kolmogorov-Smirnov test
pearsonr    : Pearson correlation coefficient
spearmanr   : Spearman rank correlation
"""

from .scirs2 import (  # noqa: F401
    mean_py as mean,
    std_py as std,
    var_py as var,
    median_py as median,
)

__all__ = [
    "mean",
    "std",
    "var",
    "median",
]
