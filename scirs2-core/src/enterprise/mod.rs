//! Enterprise deployment, performance SLA, and support channel utilities.
//!
//! This module provides configuration types and utilities for deploying SciRS2
//! in enterprise environments, defining performance SLA baselines, and
//! accessing support channel information.
//!
//! # Modules
//!
//! - [`deployment`]: Deployment target configuration, health checks, and environment detection
//! - [`support`]: Support channel configuration and contact information
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::enterprise::{deployment, support};
//!
//! // Get default SLA baselines
//! let baselines = deployment::default_sla_baselines();
//! assert!(!baselines.is_empty());
//!
//! // Get support channel info
//! let config = support::support_info();
//! assert!(!config.github_issues_url.is_empty());
//! ```

pub mod deployment;
pub mod support;

// Re-export key types at module level
pub use deployment::{
    default_sla_baselines, health_check, DeploymentHealth, DeploymentTarget, PerformanceSla,
    SlaCategory,
};
pub use support::{support_info, SupportConfig, SupportTier};
