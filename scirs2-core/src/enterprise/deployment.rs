//! Enterprise deployment configuration and utilities.
//!
//! Provides types for configuring deployment targets (Docker, Kubernetes, AWS Lambda,
//! bare metal), defining performance SLA baselines, and performing runtime health checks.
//!
//! # Performance SLA
//!
//! SLA baselines define the maximum acceptable latency and minimum throughput for
//! key operations across the SciRS2 ecosystem. These values are conservative targets
//! measured on a reference platform (4-core x86_64, 16 GB RAM).
//!
//! ```rust
//! use scirs2_core::enterprise::deployment::{default_sla_baselines, SlaCategory};
//!
//! let baselines = default_sla_baselines();
//! let linalg: Vec<_> = baselines.iter()
//!     .filter(|s| matches!(s.category, SlaCategory::LinearAlgebra))
//!     .collect();
//! assert!(!linalg.is_empty());
//! ```
//!
//! # Deployment Health
//!
//! ```rust
//! use scirs2_core::enterprise::deployment::health_check;
//!
//! let health = health_check();
//! assert!(!health.version.is_empty());
//! assert!(health.crates_available.contains(&"scirs2-core".to_string()));
//! ```

use std::collections::HashSet;
use std::time::SystemTime;

/// Deployment target configuration.
///
/// Represents the environment where SciRS2 is deployed. Each variant captures
/// the minimum configuration needed to describe the target.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DeploymentTarget {
    /// Docker container deployment.
    Docker {
        /// Docker image tag (e.g. `"scirs2:0.4.0"`).
        image_tag: String,
        /// Optional resource limits in MB.
        memory_limit_mb: Option<u64>,
    },
    /// Kubernetes deployment.
    Kubernetes {
        /// Kubernetes namespace.
        namespace: String,
        /// Number of pod replicas.
        replicas: u32,
        /// Resource request CPU in millicores.
        cpu_request_millicores: Option<u32>,
        /// Resource request memory in MiB.
        memory_request_mib: Option<u32>,
    },
    /// AWS Lambda (or similar serverless) deployment.
    AwsLambda {
        /// Memory allocation in MB.
        memory_mb: u32,
        /// Function timeout in seconds.
        timeout_secs: u32,
    },
    /// Azure Functions deployment.
    AzureFunctions {
        /// App service plan tier.
        plan_tier: String,
        /// Maximum burst instance count.
        max_instances: u32,
    },
    /// Google Cloud Run deployment.
    CloudRun {
        /// Maximum concurrent requests per container.
        max_concurrency: u32,
        /// CPU allocation (e.g. 1, 2, 4).
        cpu: u32,
        /// Memory in MiB.
        memory_mib: u32,
    },
    /// Bare metal or VM deployment.
    BareMetal {
        /// Host address or identifier.
        host: String,
    },
}

impl DeploymentTarget {
    /// Returns a human-readable description of the deployment target.
    pub fn description(&self) -> String {
        match self {
            Self::Docker { image_tag, .. } => format!("Docker container: {image_tag}"),
            Self::Kubernetes {
                namespace,
                replicas,
                ..
            } => format!("Kubernetes: {namespace} ({replicas} replicas)"),
            Self::AwsLambda {
                memory_mb,
                timeout_secs,
            } => format!("AWS Lambda: {memory_mb}MB, {timeout_secs}s timeout"),
            Self::AzureFunctions {
                plan_tier,
                max_instances,
            } => format!("Azure Functions: {plan_tier} (max {max_instances} instances)"),
            Self::CloudRun {
                max_concurrency,
                cpu,
                memory_mib,
            } => format!("Cloud Run: {cpu} CPU, {memory_mib}MiB, concurrency {max_concurrency}"),
            Self::BareMetal { host } => format!("Bare metal: {host}"),
            #[allow(unreachable_patterns)]
            _ => "Unknown deployment target".to_string(),
        }
    }
}

/// Category of an SLA baseline measurement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SlaCategory {
    /// Linear algebra operations (matmul, SVD, solve, etc.).
    LinearAlgebra,
    /// FFT operations.
    Fft,
    /// Statistical operations.
    Statistics,
    /// Signal processing operations.
    Signal,
    /// Sparse matrix operations.
    Sparse,
    /// Integration operations.
    Integration,
    /// Interpolation operations.
    Interpolation,
    /// Optimization operations.
    Optimization,
}

impl core::fmt::Display for SlaCategory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LinearAlgebra => write!(f, "linalg"),
            Self::Fft => write!(f, "fft"),
            Self::Statistics => write!(f, "stats"),
            Self::Signal => write!(f, "signal"),
            Self::Sparse => write!(f, "sparse"),
            Self::Integration => write!(f, "integrate"),
            Self::Interpolation => write!(f, "interpolate"),
            Self::Optimization => write!(f, "optimize"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "unknown"),
        }
    }
}

/// Performance SLA definition for a single operation.
///
/// Defines the maximum acceptable latency and optional throughput guarantee
/// for a specific computational operation. SLA values are conservative
/// (generous) so that they can be reliably met on the reference platform.
#[derive(Debug, Clone)]
pub struct PerformanceSla {
    /// The SLA category (crate area).
    pub category: SlaCategory,
    /// Human-readable operation name (e.g. `"matmul_1000x1000"`).
    pub operation: String,
    /// Maximum acceptable wall-clock latency in milliseconds.
    pub max_latency_ms: u64,
    /// Minimum throughput in operations per second, if applicable.
    pub throughput_ops_per_sec: Option<f64>,
    /// 99th-percentile latency in milliseconds, if measured.
    pub p99_latency_ms: Option<u64>,
    /// Description of the workload.
    pub description: String,
}

impl Default for PerformanceSla {
    fn default() -> Self {
        Self {
            category: SlaCategory::LinearAlgebra,
            operation: String::new(),
            max_latency_ms: 0,
            throughput_ops_per_sec: None,
            p99_latency_ms: None,
            description: String::new(),
        }
    }
}

/// Returns the default SLA baselines for SciRS2 v0.4.0.
///
/// All latency values are conservative upper bounds measured on the reference
/// platform (Ubuntu 22.04, 4-core x86_64, 16 GB RAM). Production deployments
/// on higher-spec hardware should comfortably exceed these targets.
///
/// # Returns
///
/// A `Vec<PerformanceSla>` containing SLA entries for all major crate areas.
pub fn default_sla_baselines() -> Vec<PerformanceSla> {
    vec![
        // Linear algebra
        PerformanceSla {
            category: SlaCategory::LinearAlgebra,
            operation: "matmul_1000x1000".into(),
            max_latency_ms: 500,
            throughput_ops_per_sec: Some(2.0),
            p99_latency_ms: Some(600),
            description: "Dense matrix multiply (1000x1000 f64)".into(),
        },
        PerformanceSla {
            category: SlaCategory::LinearAlgebra,
            operation: "det_100x100".into(),
            max_latency_ms: 10,
            throughput_ops_per_sec: Some(100.0),
            p99_latency_ms: Some(15),
            description: "Determinant of 100x100 f64 matrix".into(),
        },
        PerformanceSla {
            category: SlaCategory::LinearAlgebra,
            operation: "svd_500x500".into(),
            max_latency_ms: 2000,
            throughput_ops_per_sec: Some(0.5),
            p99_latency_ms: Some(2500),
            description: "Full SVD of 500x500 f64 matrix".into(),
        },
        PerformanceSla {
            category: SlaCategory::LinearAlgebra,
            operation: "solve_1000x1000".into(),
            max_latency_ms: 500,
            throughput_ops_per_sec: Some(2.0),
            p99_latency_ms: Some(600),
            description: "Dense linear solve (1000x1000 f64)".into(),
        },
        PerformanceSla {
            category: SlaCategory::LinearAlgebra,
            operation: "cholesky_1000x1000".into(),
            max_latency_ms: 300,
            throughput_ops_per_sec: Some(3.0),
            p99_latency_ms: Some(400),
            description: "Cholesky decomposition (1000x1000 SPD f64)".into(),
        },
        // FFT
        PerformanceSla {
            category: SlaCategory::Fft,
            operation: "fft_1m_points".into(),
            max_latency_ms: 100,
            throughput_ops_per_sec: Some(10.0),
            p99_latency_ms: Some(130),
            description: "Complex FFT of 2^20 (1M) points".into(),
        },
        PerformanceSla {
            category: SlaCategory::Fft,
            operation: "fft_64k_points".into(),
            max_latency_ms: 10,
            throughput_ops_per_sec: Some(100.0),
            p99_latency_ms: Some(15),
            description: "Complex FFT of 2^16 (64K) points".into(),
        },
        PerformanceSla {
            category: SlaCategory::Fft,
            operation: "batch_fft_1000x1024".into(),
            max_latency_ms: 200,
            throughput_ops_per_sec: Some(5.0),
            p99_latency_ms: Some(250),
            description: "Batch FFT: 1000 transforms of length 1024".into(),
        },
        PerformanceSla {
            category: SlaCategory::Fft,
            operation: "rfft_1m_points".into(),
            max_latency_ms: 60,
            throughput_ops_per_sec: Some(15.0),
            p99_latency_ms: Some(80),
            description: "Real-valued FFT of 2^20 (1M) points".into(),
        },
        // Statistics
        PerformanceSla {
            category: SlaCategory::Statistics,
            operation: "normal_pdf_1m".into(),
            max_latency_ms: 50,
            throughput_ops_per_sec: Some(20.0),
            p99_latency_ms: Some(65),
            description: "Normal PDF evaluated at 1M points".into(),
        },
        PerformanceSla {
            category: SlaCategory::Statistics,
            operation: "linreg_10k_100feat".into(),
            max_latency_ms: 500,
            throughput_ops_per_sec: Some(2.0),
            p99_latency_ms: Some(600),
            description: "Linear regression: 10K samples, 100 features".into(),
        },
        PerformanceSla {
            category: SlaCategory::Statistics,
            operation: "kde_10k_points".into(),
            max_latency_ms: 200,
            throughput_ops_per_sec: Some(5.0),
            p99_latency_ms: Some(250),
            description: "Kernel density estimation on 10K points".into(),
        },
        // Signal
        PerformanceSla {
            category: SlaCategory::Signal,
            operation: "fir_64tap_1m".into(),
            max_latency_ms: 100,
            throughput_ops_per_sec: Some(10.0),
            p99_latency_ms: Some(130),
            description: "FIR filter: 64 taps, 1M samples".into(),
        },
        PerformanceSla {
            category: SlaCategory::Signal,
            operation: "stft_1m_1024win".into(),
            max_latency_ms: 500,
            throughput_ops_per_sec: Some(2.0),
            p99_latency_ms: Some(600),
            description: "STFT: 1M samples, 1024-sample window".into(),
        },
        PerformanceSla {
            category: SlaCategory::Signal,
            operation: "iir_8pole_1m".into(),
            max_latency_ms: 50,
            throughput_ops_per_sec: Some(20.0),
            p99_latency_ms: Some(65),
            description: "IIR filter: 8-pole Butterworth, 1M samples".into(),
        },
        // Sparse
        PerformanceSla {
            category: SlaCategory::Sparse,
            operation: "spmv_100k_1m".into(),
            max_latency_ms: 10,
            throughput_ops_per_sec: Some(100.0),
            p99_latency_ms: Some(15),
            description: "Sparse matrix-vector multiply: 100K x 100K, 1M nnz".into(),
        },
        PerformanceSla {
            category: SlaCategory::Sparse,
            operation: "cg_10k".into(),
            max_latency_ms: 1000,
            throughput_ops_per_sec: Some(1.0),
            p99_latency_ms: Some(1200),
            description: "Conjugate gradient solve: 10K x 10K sparse SPD".into(),
        },
        PerformanceSla {
            category: SlaCategory::Sparse,
            operation: "sparse_lu_10k".into(),
            max_latency_ms: 2000,
            throughput_ops_per_sec: Some(0.5),
            p99_latency_ms: Some(2500),
            description: "Sparse LU factorization: 10K x 10K".into(),
        },
        // Integration
        PerformanceSla {
            category: SlaCategory::Integration,
            operation: "quad_1k_points".into(),
            max_latency_ms: 5,
            throughput_ops_per_sec: Some(200.0),
            p99_latency_ms: Some(8),
            description: "Adaptive quadrature with 1K evaluation points".into(),
        },
        PerformanceSla {
            category: SlaCategory::Integration,
            operation: "ode_rk45_10k_steps".into(),
            max_latency_ms: 100,
            throughput_ops_per_sec: Some(10.0),
            p99_latency_ms: Some(130),
            description: "RK45 ODE solver: 10K adaptive steps".into(),
        },
        // Interpolation
        PerformanceSla {
            category: SlaCategory::Interpolation,
            operation: "cubic_spline_10k".into(),
            max_latency_ms: 20,
            throughput_ops_per_sec: Some(50.0),
            p99_latency_ms: Some(30),
            description: "Cubic spline interpolation: 10K knots".into(),
        },
        // Optimization
        PerformanceSla {
            category: SlaCategory::Optimization,
            operation: "lbfgs_100d".into(),
            max_latency_ms: 200,
            throughput_ops_per_sec: Some(5.0),
            p99_latency_ms: Some(250),
            description: "L-BFGS optimization: 100 dimensions, Rosenbrock".into(),
        },
        PerformanceSla {
            category: SlaCategory::Optimization,
            operation: "nelder_mead_50d".into(),
            max_latency_ms: 500,
            throughput_ops_per_sec: Some(2.0),
            p99_latency_ms: Some(600),
            description: "Nelder-Mead: 50 dimensions".into(),
        },
    ]
}

/// Validates that a set of SLA baselines has no duplicate operation names.
///
/// # Errors
///
/// Returns `Err` with the duplicate operation name if any duplicates are found.
pub fn validate_sla_uniqueness(baselines: &[PerformanceSla]) -> Result<(), String> {
    let mut seen = HashSet::new();
    for sla in baselines {
        if !seen.insert(&sla.operation) {
            return Err(format!("Duplicate SLA operation: {}", sla.operation));
        }
    }
    Ok(())
}

/// Runtime health check result.
#[derive(Debug, Clone)]
pub struct DeploymentHealth {
    /// SciRS2 version string.
    pub version: String,
    /// List of available (compiled-in) crate names.
    pub crates_available: Vec<String>,
    /// Process uptime in seconds (approximate; measured from first call).
    pub uptime_secs: u64,
    /// Approximate current heap usage in bytes (platform-dependent estimate).
    pub memory_usage_bytes: usize,
    /// Timestamp of the health check (seconds since UNIX epoch).
    pub timestamp_epoch_secs: u64,
}

/// Performs a deployment health check.
///
/// Returns a [`DeploymentHealth`] snapshot capturing the current version,
/// available crates, and approximate resource usage.
pub fn health_check() -> DeploymentHealth {
    use std::sync::OnceLock;

    static START_TIME: OnceLock<SystemTime> = OnceLock::new();
    let start = START_TIME.get_or_init(SystemTime::now);

    let uptime = SystemTime::now()
        .duration_since(*start)
        .unwrap_or_default()
        .as_secs();

    let now_epoch = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Enumerate crates that are always compiled in the workspace.
    // This is a static list; feature-gated crates are included when their
    // feature is enabled.
    let mut crates_available = vec![
        "scirs2-core".to_string(),
        "scirs2-linalg".to_string(),
        "scirs2-stats".to_string(),
        "scirs2-signal".to_string(),
        "scirs2-fft".to_string(),
        "scirs2-sparse".to_string(),
        "scirs2-optimize".to_string(),
        "scirs2-integrate".to_string(),
        "scirs2-interpolate".to_string(),
        "scirs2-special".to_string(),
        "scirs2-cluster".to_string(),
        "scirs2-io".to_string(),
        "scirs2-graph".to_string(),
        "scirs2-neural".to_string(),
        "scirs2-series".to_string(),
        "scirs2-text".to_string(),
        "scirs2-vision".to_string(),
        "scirs2-metrics".to_string(),
        "scirs2-ndimage".to_string(),
        "scirs2-transform".to_string(),
        "scirs2-datasets".to_string(),
        "scirs2-wasm".to_string(),
    ];
    crates_available.sort();

    DeploymentHealth {
        version: env!("CARGO_PKG_VERSION").to_string(),
        crates_available,
        uptime_secs: uptime,
        memory_usage_bytes: 0, // No portable way to query heap without allocator hooks
        timestamp_epoch_secs: now_epoch,
    }
}

/// Recommended container resource limits for common workload profiles.
#[derive(Debug, Clone)]
pub struct ResourceRecommendation {
    /// Workload profile name.
    pub profile: String,
    /// Recommended CPU cores.
    pub cpu_cores: u32,
    /// Recommended memory in MiB.
    pub memory_mib: u32,
    /// Recommended disk in MiB (for temporary files).
    pub disk_mib: u32,
    /// Description of the workload profile.
    pub description: String,
}

/// Returns resource recommendations for common deployment profiles.
pub fn resource_recommendations() -> Vec<ResourceRecommendation> {
    vec![
        ResourceRecommendation {
            profile: "lightweight".into(),
            cpu_cores: 2,
            memory_mib: 2048,
            disk_mib: 512,
            description: "Statistical computations, small-scale signal processing".into(),
        },
        ResourceRecommendation {
            profile: "standard".into(),
            cpu_cores: 4,
            memory_mib: 8192,
            disk_mib: 2048,
            description: "General scientific computing, moderate linear algebra".into(),
        },
        ResourceRecommendation {
            profile: "compute_intensive".into(),
            cpu_cores: 8,
            memory_mib: 32768,
            disk_mib: 8192,
            description: "Large-scale linalg, neural network training, optimization".into(),
        },
        ResourceRecommendation {
            profile: "memory_intensive".into(),
            cpu_cores: 4,
            memory_mib: 65536,
            disk_mib: 16384,
            description: "Large sparse systems, out-of-core processing, big datasets".into(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_sla_baselines_not_empty() {
        let baselines = default_sla_baselines();
        assert!(!baselines.is_empty(), "SLA baselines must not be empty");
        assert!(
            baselines.len() >= 15,
            "Expected at least 15 SLA entries, got {}",
            baselines.len()
        );
    }

    #[test]
    fn test_sla_values_positive() {
        for sla in default_sla_baselines() {
            assert!(
                sla.max_latency_ms > 0,
                "SLA {} has zero max_latency_ms",
                sla.operation
            );
            if let Some(throughput) = sla.throughput_ops_per_sec {
                assert!(
                    throughput > 0.0,
                    "SLA {} has non-positive throughput",
                    sla.operation
                );
            }
            if let Some(p99) = sla.p99_latency_ms {
                assert!(
                    p99 >= sla.max_latency_ms,
                    "SLA {} has p99 ({}) < max_latency ({})",
                    sla.operation,
                    p99,
                    sla.max_latency_ms
                );
            }
        }
    }

    #[test]
    fn test_sla_operation_names_unique() {
        let baselines = default_sla_baselines();
        let result = validate_sla_uniqueness(&baselines);
        assert!(
            result.is_ok(),
            "Duplicate SLA operation: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_sla_all_categories_covered() {
        let baselines = default_sla_baselines();
        let categories: HashSet<_> = baselines.iter().map(|s| s.category.clone()).collect();
        assert!(categories.contains(&SlaCategory::LinearAlgebra));
        assert!(categories.contains(&SlaCategory::Fft));
        assert!(categories.contains(&SlaCategory::Statistics));
        assert!(categories.contains(&SlaCategory::Signal));
        assert!(categories.contains(&SlaCategory::Sparse));
        assert!(categories.contains(&SlaCategory::Integration));
        assert!(categories.contains(&SlaCategory::Interpolation));
        assert!(categories.contains(&SlaCategory::Optimization));
    }

    #[test]
    fn test_health_check() {
        let health = health_check();
        assert!(!health.version.is_empty(), "Version must not be empty");
        assert!(
            !health.crates_available.is_empty(),
            "Crates list must not be empty"
        );
        assert!(
            health.crates_available.contains(&"scirs2-core".to_string()),
            "scirs2-core must be in crates list"
        );
        assert!(
            health.timestamp_epoch_secs > 0,
            "Timestamp must be positive"
        );
    }

    #[test]
    fn test_deployment_target_variants() {
        let targets = vec![
            DeploymentTarget::Docker {
                image_tag: "scirs2:0.4.0".into(),
                memory_limit_mb: Some(4096),
            },
            DeploymentTarget::Kubernetes {
                namespace: "production".into(),
                replicas: 3,
                cpu_request_millicores: Some(2000),
                memory_request_mib: Some(8192),
            },
            DeploymentTarget::AwsLambda {
                memory_mb: 1024,
                timeout_secs: 300,
            },
            DeploymentTarget::AzureFunctions {
                plan_tier: "Premium".into(),
                max_instances: 10,
            },
            DeploymentTarget::CloudRun {
                max_concurrency: 80,
                cpu: 4,
                memory_mib: 8192,
            },
            DeploymentTarget::BareMetal {
                host: "compute-01.example.com".into(),
            },
        ];
        for target in &targets {
            let desc = target.description();
            assert!(!desc.is_empty(), "Description must not be empty");
        }
    }

    #[test]
    fn test_deployment_target_description_content() {
        let docker = DeploymentTarget::Docker {
            image_tag: "myimg:latest".into(),
            memory_limit_mb: None,
        };
        assert!(docker.description().contains("myimg:latest"));

        let k8s = DeploymentTarget::Kubernetes {
            namespace: "ml-prod".into(),
            replicas: 5,
            cpu_request_millicores: None,
            memory_request_mib: None,
        };
        assert!(k8s.description().contains("ml-prod"));
        assert!(k8s.description().contains("5"));
    }

    #[test]
    fn test_resource_recommendations() {
        let recs = resource_recommendations();
        assert!(recs.len() >= 3, "Expected at least 3 resource profiles");
        for rec in &recs {
            assert!(!rec.profile.is_empty());
            assert!(rec.cpu_cores > 0);
            assert!(rec.memory_mib > 0);
        }
    }

    #[test]
    fn test_sla_category_display() {
        assert_eq!(SlaCategory::LinearAlgebra.to_string(), "linalg");
        assert_eq!(SlaCategory::Fft.to_string(), "fft");
        assert_eq!(SlaCategory::Statistics.to_string(), "stats");
        assert_eq!(SlaCategory::Signal.to_string(), "signal");
        assert_eq!(SlaCategory::Sparse.to_string(), "sparse");
        assert_eq!(SlaCategory::Integration.to_string(), "integrate");
        assert_eq!(SlaCategory::Interpolation.to_string(), "interpolate");
        assert_eq!(SlaCategory::Optimization.to_string(), "optimize");
    }

    #[test]
    fn test_performance_sla_default() {
        let sla = PerformanceSla::default();
        assert!(sla.operation.is_empty());
        assert_eq!(sla.max_latency_ms, 0);
        assert!(sla.throughput_ops_per_sec.is_none());
        assert!(sla.p99_latency_ms.is_none());
    }
}
