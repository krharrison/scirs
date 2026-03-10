# scirs2-metrics TODO

## Status: v0.3.1 Released (March 9, 2026)

## v0.3.1 Completed

### Classification Metrics
- [x] `accuracy_score` with sample weighting
- [x] `precision_score`, `recall_score`, `f1_score`, `fbeta_score`
- [x] `precision_recall_fscore_support` (all in one)
- [x] Matthews correlation coefficient (MCC)
- [x] Balanced accuracy, Cohen's kappa
- [x] `roc_curve`, `roc_auc_score`
- [x] `average_precision_score`, `precision_recall_curve`
- [x] `confusion_matrix`, `classification_report`
- [x] `log_loss`, `brier_score_loss`
- [x] `hinge_loss`, `hamming_loss`, `jaccard_score`
- [x] Multi-class: micro/macro/weighted/samples averaging
- [x] `cohen_kappa_score`, `matthews_corrcoef`
- [x] Optimal threshold: `g_means_score`, `find_optimal_threshold`
- [x] Label binarization utilities

### Regression Metrics
- [x] `mean_squared_error` (MSE), `root_mean_squared_error` (RMSE)
- [x] `mean_absolute_error` (MAE), `median_absolute_error`
- [x] `mean_absolute_percentage_error` (MAPE), symmetric MAPE
- [x] `r2_score`, `explained_variance_score`, adjusted R²
- [x] `mean_squared_log_error` (MSLE)
- [x] `mean_tweedie_deviance`, Poisson/Gamma deviance
- [x] Huber loss, quantile (pinball) loss
- [x] Max error, relative absolute/squared error
- [x] `regression_advanced`: interval score, coverage probability, Winkler score

### Clustering Metrics
- [x] `silhouette_score`, `silhouette_samples`
- [x] `calinski_harabasz_score` (variance ratio)
- [x] `davies_bouldin_score`
- [x] Dunn index, gap statistic
- [x] `adjusted_rand_index` (ARI)
- [x] `normalized_mutual_info_score`, `adjusted_mutual_info_score`
- [x] `homogeneity_completeness_v_measure`
- [x] `fowlkes_mallows_score`
- [x] Contingency matrix, pair confusion matrix
- [x] Cluster stability, consensus scoring

### Ranking and Information Retrieval
- [x] `ndcg_score` at k, DCG
- [x] `mean_average_precision` (MAP), MAP@k
- [x] `mrr_score` (MRR)
- [x] `precision_at_k`, `recall_at_k`
- [x] Kendall's tau, Spearman's rank correlation
- [x] Label ranking average precision (LRAP)
- [x] `ir_metrics` module: comprehensive IR evaluation

### Object Detection Metrics (New in v0.3.1)
- [x] `iou_score` for axis-aligned bounding boxes
- [x] `average_precision` at IoU threshold
- [x] `compute_map` (mAP) with configurable IoU thresholds
- [x] Non-Maximum Suppression (NMS) utilities
- [x] PASCAL VOC-style evaluation
- [x] COCO-style mAP@[0.5:0.95] evaluation
- [x] Per-class AP breakdown

### Generative Model Evaluation (New in v0.3.1)
- [x] Fréchet Inception Distance (FID)
- [x] Inception Score (IS)
- [x] Precision and Recall for generative models
- [x] Maximum Mean Discrepancy (MMD)
- [x] Kernel-based evaluation metrics

### Segmentation Metrics (New in v0.3.1)
- [x] Pixel accuracy, mean pixel accuracy
- [x] Per-class IoU, mean IoU (mIoU)
- [x] Dice coefficient, Jaccard index
- [x] Boundary F-measure
- [x] Panoptic Quality (PQ)

### Fairness and Bias Detection
- [x] `demographic_parity` (difference and ratio)
- [x] `equalized_odds` difference
- [x] `equal_opportunity` difference
- [x] `disparate_impact` ratio
- [x] Consistency score across subgroups
- [x] Slice analysis for subgroup performance
- [x] Intersectional fairness measures
- [x] Robustness testing: performance invariance, sensitivity

### Streaming Metrics
- [x] Memory-efficient online evaluation
- [x] `streaming/optimization/patterns/batching.rs` - batch accumulator
- [x] `streaming/optimization/patterns/buffering.rs` - ring-buffer streaming
- [x] `streaming/optimization/patterns/partitioning.rs` - keyed/group metrics
- [x] `streaming/optimization/patterns/windowing.rs` - sliding/tumbling windows
- [x] Online Welford variance for streaming statistics

### Evaluation Framework
- [x] K-fold cross-validation, stratified K-fold, leave-one-out
- [x] Time series cross-validation
- [x] `cross_val_score`, `cross_validate`
- [x] Learning curve, validation curve generation
- [x] `grid_search_cv`, `randomized_search_cv`
- [x] Bootstrap confidence intervals
- [x] McNemar's test, Friedman test, Wilcoxon signed-rank test

### Bayesian Evaluation
- [x] Bayes factor computation
- [x] BIC, AIC, WAIC, LOO-CV, DIC
- [x] Posterior predictive checks
- [x] Bayesian model averaging
- [x] Credible intervals and HPD intervals

### Hardware Acceleration
- [x] SIMD-accelerated computations (SSE2, AVX2, AVX-512)
- [x] Automatic hardware capability detection
- [x] Configurable acceleration settings
- [x] Parallel Rayon-based batch evaluation

### Visualization
- [x] ROC curve, precision-recall curve, calibration curve
- [x] Confusion matrix heatmap (normalized and unnormalized)
- [x] Learning and validation curves
- [x] Histogram, scatter, bar chart, heatmap
- [x] Plotters backend (PNG/SVG)
- [x] Plotly backend (interactive HTML)
- [x] Dashboard server (HTTP, Chart.js, RESTful API)
- [x] Export: JSON, CSV, HTML

### Neural Integration (`neural_common` feature)
- [x] `NeuralMetricAdapter` for `scirs2-neural` trainer callbacks
- [x] `MetricsCallback` for per-epoch metric collection
- [x] Training history visualization

### Optimization Integration (`optim_integration` feature)
- [x] `MetricScheduler` (reduce-on-plateau)
- [x] `HyperParameterTuner` with random and grid search
- [x] `MetricBasedReduceOnPlateau` optimizer wrapper

## v0.4.0 Roadmap

### Distributional Metrics
- [ ] Wasserstein distance (Earth Mover's distance) — exact and approximate
- [ ] Sinkhorn divergence for regularized optimal transport
- [ ] Energy distance between empirical distributions
- [ ] Kernel Stein Discrepancy (KSD) for goodness-of-fit
- [ ] Total variation distance

### Uncertainty Calibration Metrics
- [ ] Expected Calibration Error (ECE)
- [ ] Maximum Calibration Error (MCE)
- [ ] Reliability diagram generation
- [ ] Temperature scaling calibration diagnostic
- [ ] Adaptive Calibration Error (ACE)
- [ ] Conformal prediction coverage metrics

### Active Learning Utility
- [ ] Margin sampling score
- [ ] Entropy-based uncertainty score
- [ ] Query-by-committee disagreement
- [ ] Core-set selection metrics
- [ ] Expected model change

### Expanded Detection Metrics
- [ ] 3D IoU for point cloud bounding boxes
- [ ] Rotated bounding box IoU
- [ ] Tracking metrics: MOTA, MOTP, IDF1
- [ ] Keypoint metrics: OKS (Object Keypoint Similarity), PCK

### Time Series Metrics
- [ ] Dynamic Time Warping (DTW) distance as metric
- [ ] Forecast skill scores (Brier skill score, CRPS)
- [ ] Directional accuracy (hit rate) for forecasts
- [ ] Diebold-Mariano test for forecast comparison

### Documentation and Examples
- [ ] Comprehensive cookbook with domain-specific metric selection guides
- [ ] Integration examples with popular ML frameworks
- [ ] Benchmark report vs `sklearn.metrics`

## Known Issues

- `plotly_backend` feature generates HTML that requires a network connection to load Chart.js from CDN; add offline/bundled mode.
- `dashboard_server` requires `tokio` runtime; document how to integrate with existing async applications.
- SIMD acceleration is only available on x86/x86_64 targets; ARM NEON support is planned for v0.4.0.
- The `generative` module (FID, IS) requires pre-computed feature vectors; it does not include the Inception network — document this clearly.
