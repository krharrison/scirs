//! Configuration structures for streaming optimization
//!
//! This module provides comprehensive configuration options for:
//! - Streaming metrics configuration
//! - Window adaptation strategies
//! - Drift detection methods
//! - Anomaly detection algorithms
//! - Alert configurations
//! - Performance monitoring settings

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Base window size
    pub base_window_size: usize,
    /// Maximum window size
    pub max_window_size: usize,
    /// Minimum window size
    pub min_window_size: usize,
    /// Drift detection sensitivity
    pub drift_sensitivity: f64,
    /// Warning threshold for drift
    pub warning_threshold: f64,
    /// Drift threshold for adaptation
    pub drift_threshold: f64,
    /// Enable adaptive windowing
    pub adaptive_windowing: bool,
    /// Window adaptation strategy
    pub adaptation_strategy: WindowAdaptationStrategy,
    /// Enable concept drift detection
    pub enable_drift_detection: bool,
    /// Drift detection methods
    pub drift_detection_methods: Vec<DriftDetectionMethod>,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection algorithm
    pub anomaly_algorithm: AnomalyDetectionAlgorithm,
    /// Performance monitoring interval
    pub monitoring_interval: Duration,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert configuration
    pub alert_config: AlertConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            base_window_size: 1000,
            max_window_size: 10000,
            min_window_size: 100,
            drift_sensitivity: 0.05,
            warning_threshold: 0.95,
            drift_threshold: 0.99,
            adaptive_windowing: true,
            adaptation_strategy: WindowAdaptationStrategy::DriftBased,
            enable_drift_detection: true,
            drift_detection_methods: vec![
                DriftDetectionMethod::Adwin { delta: 0.002 },
                DriftDetectionMethod::Ddm { alpha: 0.05, beta: 0.0 },
            ],
            enable_anomaly_detection: true,
            anomaly_algorithm: AnomalyDetectionAlgorithm::IsolationForest {
                n_trees: 100,
                subsample_size: 256,
            },
            monitoring_interval: Duration::from_secs(10),
            enable_alerts: true,
            alert_config: AlertConfig::default(),
        }
    }
}

/// Window adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAdaptationStrategy {
    /// Fixed window size
    Fixed,
    /// Exponential decay-based adaptation
    ExponentialDecay { decay_rate: f64 },
    /// Performance-based adaptation
    PerformanceBased { target_accuracy: f64 },
    /// Drift-based adaptation
    DriftBased,
    /// Hybrid approach combining multiple strategies
    Hybrid {
        strategies: Vec<WindowAdaptationStrategy>,
        weights: Vec<f64>,
    },
    /// Machine learning-based adaptation
    MLBased { model_type: String },
}

/// Drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    /// ADWIN (Adaptive Windowing) algorithm
    Adwin {
        /// Confidence delta parameter
        delta: f64,
    },
    /// DDM (Drift Detection Method)
    Ddm {
        /// Warning level parameter
        alpha: f64,
        /// Drift level parameter
        beta: f64,
    },
    /// Page-Hinkley test
    PageHinkley {
        /// Minimum number of instances
        min_instances: usize,
        /// Detection threshold
        threshold: f64,
        /// Alpha parameter
        alpha: f64,
    },
    /// KSWIN (Kolmogorov-Smirnov Windowing)
    Kswin {
        /// Window size
        window_size: usize,
        /// Statistical significance level
        stat_size: usize,
    },
    /// HDDM_A (Hoeffding's bounds based Drift Detection Method - Average)
    HddmA {
        /// Drift confidence
        drift_confidence: f64,
        /// Warning confidence
        warning_confidence: f64,
    },
    /// HDDM_W (Hoeffding's bounds based Drift Detection Method - Weighted)
    HddmW {
        /// Drift confidence
        drift_confidence: f64,
        /// Warning confidence
        warning_confidence: f64,
        /// Lambda parameter for exponential decay
        lambda: f64,
    },
    /// STEPD (Statistical Test of Equal Proportions)
    Stepd {
        /// Window size
        window_size: usize,
        /// Alpha parameter
        alpha: f64,
    },
    /// Custom drift detection method
    Custom {
        /// Method name
        name: String,
        /// Method parameters
        parameters: std::collections::HashMap<String, f64>,
    },
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Isolation Forest
    IsolationForest {
        /// Number of trees
        n_trees: usize,
        /// Subsample size
        subsample_size: usize,
    },
    /// One-Class SVM
    OneClassSvm {
        /// Nu parameter
        nu: f64,
        /// Gamma parameter
        gamma: f64,
    },
    /// Local Outlier Factor
    LocalOutlierFactor {
        /// Number of neighbors
        n_neighbors: usize,
        /// Contamination ratio
        contamination: f64,
    },
    /// Elliptic Envelope
    EllipticEnvelope {
        /// Contamination ratio
        contamination: f64,
    },
    /// Statistical outlier detection
    Statistical {
        /// Z-score threshold
        z_threshold: f64,
        /// Use modified Z-score
        modified: bool,
    },
    /// Robust covariance-based detection
    RobustCovariance {
        /// Support fraction
        support_fraction: f64,
    },
    /// Custom anomaly detection algorithm
    Custom {
        /// Algorithm name
        name: String,
        /// Algorithm parameters
        parameters: std::collections::HashMap<String, f64>,
    },
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable drift alerts
    pub drift_alerts: bool,
    /// Enable anomaly alerts
    pub anomaly_alerts: bool,
    /// Enable performance alerts
    pub performance_alerts: bool,
    /// Minimum alert severity to report
    pub min_severity: AlertSeverity,
    /// Alert cooldown period
    pub cooldown_period: Duration,
    /// Maximum alerts per time window
    pub max_alerts_per_window: usize,
    /// Alert aggregation window
    pub aggregation_window: Duration,
    /// Email notifications
    pub email_notifications: bool,
    /// Webhook notifications
    pub webhook_notifications: bool,
    /// Webhook URL
    pub webhook_url: Option<String>,
    /// Custom alert handlers
    pub custom_handlers: Vec<String>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            drift_alerts: true,
            anomaly_alerts: true,
            performance_alerts: true,
            min_severity: AlertSeverity::Warning,
            cooldown_period: Duration::from_secs(300), // 5 minutes
            max_alerts_per_window: 10,
            aggregation_window: Duration::from_secs(3600), // 1 hour
            email_notifications: false,
            webhook_notifications: false,
            webhook_url: None,
            custom_handlers: vec![],
        }
    }
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
}

/// Window adaptation trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationTrigger {
    /// Time-based trigger
    Time(Duration),
    /// Sample count-based trigger
    SampleCount(usize),
    /// Performance-based trigger
    Performance { 
        accuracy_threshold: f64,
        latency_threshold: Duration,
    },
    /// Drift-based trigger
    Drift { confidence: f64 },
    /// Manual trigger
    Manual,
    /// Combined triggers
    Combined(Vec<AdaptationTrigger>),
}

/// Ensemble aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleAggregation {
    /// Simple average
    Average,
    /// Weighted average
    WeightedAverage(Vec<f64>),
    /// Maximum value
    Maximum,
    /// Minimum value
    Minimum,
    /// Median value
    Median,
    /// Majority voting
    MajorityVoting,
    /// Soft voting with probabilities
    SoftVoting,
    /// Custom aggregation function
    Custom(String),
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Enable latency monitoring
    pub monitor_latency: bool,
    /// Enable throughput monitoring
    pub monitor_throughput: bool,
    /// Enable memory usage monitoring
    pub monitor_memory: bool,
    /// Enable accuracy monitoring
    pub monitor_accuracy: bool,
    /// Monitoring sampling rate
    pub sampling_rate: f64,
    /// Performance baseline update interval
    pub baseline_update_interval: Duration,
    /// Performance degradation threshold
    pub degradation_threshold: f64,
    /// Alert on performance degradation
    pub alert_on_degradation: bool,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            monitor_latency: true,
            monitor_throughput: true,
            monitor_memory: true,
            monitor_accuracy: true,
            sampling_rate: 1.0,
            baseline_update_interval: Duration::from_secs(3600), // 1 hour
            degradation_threshold: 0.1, // 10% degradation
            alert_on_degradation: true,
        }
    }
}

/// Buffering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Maximum buffer size
    pub max_size: usize,
    /// Buffer eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Persistence configuration
    pub persistence: PersistenceConfig,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_size: 100000,
            eviction_policy: EvictionPolicy::Lru,
            enable_compression: false,
            compression_algorithm: CompressionAlgorithm::Gzip,
            persistence: PersistenceConfig::default(),
        }
    }
}

/// Buffer eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In, First Out
    Fifo,
    /// Random eviction
    Random,
    /// Time-based eviction
    TimeBased(Duration),
    /// Size-based eviction
    SizeBased(usize),
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// Zstandard compression
    Zstd,
    /// Snappy compression
    Snappy,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Enable persistence
    pub enabled: bool,
    /// Persistence backend
    pub backend: PersistenceBackend,
    /// Sync interval
    pub sync_interval: Duration,
    /// Compression for persistent data
    pub compress_data: bool,
    /// Retention policy
    pub retention_policy: RetentionPolicy,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: PersistenceBackend::File {
                path: {
                    let mut p = std::env::temp_dir();
                    p.push("streaming_data");
                    p.to_string_lossy().into_owned()
                },
            },
            sync_interval: Duration::from_secs(60),
            compress_data: true,
            retention_policy: RetentionPolicy::TimeBasedRetention(Duration::from_secs(86400 * 7)), // 7 days
        }
    }
}

/// Persistence backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceBackend {
    /// File-based persistence
    File { path: String },
    /// Database persistence
    Database { 
        connection_string: String,
        table_name: String,
    },
    /// Redis persistence
    Redis { 
        host: String,
        port: u16,
        database: usize,
    },
    /// Custom persistence backend
    Custom { 
        backend_type: String,
        config: std::collections::HashMap<String, String>,
    },
}

/// Data retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionPolicy {
    /// Keep all data
    KeepAll,
    /// Time-based retention
    TimeBasedRetention(Duration),
    /// Size-based retention
    SizeBasedRetention(usize),
    /// Count-based retention
    CountBasedRetention(usize),
    /// Custom retention policy
    Custom {
        policy_name: String,
        parameters: std::collections::HashMap<String, String>,
    },
}

/// Batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Enable batching
    pub enabled: bool,
    /// Batch size
    pub batch_size: usize,
    /// Batch timeout
    pub batch_timeout: Duration,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batching strategy
    pub strategy: BatchingStrategy,
    /// Enable adaptive batching
    pub adaptive_batching: bool,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
            max_batch_size: 1000,
            strategy: BatchingStrategy::SizeBased,
            adaptive_batching: true,
        }
    }
}

/// Batching strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Size-based batching
    SizeBased,
    /// Time-based batching
    TimeBased,
    /// Hybrid batching (size and time)
    Hybrid,
    /// Load-based adaptive batching
    LoadBased,
    /// Latency-optimized batching
    LatencyOptimized,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.base_window_size, 1000);
        assert!(config.adaptive_windowing);
        assert!(config.enable_drift_detection);
        assert!(config.enable_anomaly_detection);
    }

    #[test]
    fn test_alert_config_default() {
        let config = AlertConfig::default();
        assert!(config.drift_alerts);
        assert!(config.anomaly_alerts);
        assert_eq!(config.min_severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_window_adaptation_strategy() {
        let strategy = WindowAdaptationStrategy::ExponentialDecay { decay_rate: 0.95 };
        match strategy {
            WindowAdaptationStrategy::ExponentialDecay { decay_rate } => {
                assert_eq!(decay_rate, 0.95);
            }
            _ => panic!("Unexpected strategy type"),
        }
    }

    #[test]
    fn test_drift_detection_method() {
        let method = DriftDetectionMethod::Adwin { delta: 0.002 };
        match method {
            DriftDetectionMethod::Adwin { delta } => {
                assert_eq!(delta, 0.002);
            }
            _ => panic!("Unexpected method type"),
        }
    }

    #[test]
    fn test_anomaly_detection_algorithm() {
        let algorithm = AnomalyDetectionAlgorithm::IsolationForest {
            n_trees: 100,
            subsample_size: 256,
        };
        
        match algorithm {
            AnomalyDetectionAlgorithm::IsolationForest { n_trees, subsample_size } => {
                assert_eq!(n_trees, 100);
                assert_eq!(subsample_size, 256);
            }
            _ => panic!("Unexpected algorithm type"),
        }
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::Error);
        assert!(AlertSeverity::Error > AlertSeverity::Warning);
        assert!(AlertSeverity::Warning > AlertSeverity::Info);
    }

    #[test]
    fn test_buffer_config_default() {
        let config = BufferConfig::default();
        assert_eq!(config.max_size, 100000);
        assert!(matches!(config.eviction_policy, EvictionPolicy::Lru));
    }

    #[test]
    fn test_performance_monitoring_config() {
        let config = PerformanceMonitoringConfig::default();
        assert!(config.monitor_latency);
        assert!(config.monitor_throughput);
        assert_eq!(config.sampling_rate, 1.0);
    }

    #[test]
    fn test_batching_config_default() {
        let config = BatchingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.batch_size, 100);
        assert!(config.adaptive_batching);
    }
}