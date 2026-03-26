"""Clustering algorithms.

Provides clustering algorithms backed by the SciRS2 Rust implementation.
The API mirrors ``sklearn.cluster`` for easy migration.

Classes
-------
KMeans          : K-Means clustering
DBSCAN          : Density-Based Spatial Clustering of Applications with Noise
AgglomerativeClustering : Hierarchical agglomerative clustering

Functions
---------
silhouette_score        : Silhouette score for cluster quality
davies_bouldin_score    : Davies-Bouldin index for cluster quality
calinski_harabasz_score : Calinski-Harabasz score for cluster quality
"""

from .scirs2 import (  # noqa: F401
    KMeans,
    silhouette_score_py as silhouette_score,
    davies_bouldin_score_py as davies_bouldin_score,
    calinski_harabasz_score_py as calinski_harabasz_score,
)

__all__ = [
    "KMeans",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
]
