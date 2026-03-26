//! Support channel configuration and contact information.
//!
//! Provides types describing how users and enterprise customers can obtain
//! support for SciRS2, including issue trackers, documentation portals,
//! email contacts, and response-time expectations.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::enterprise::support::{support_info, SupportTier};
//!
//! let config = support_info();
//! assert_eq!(config.tier, SupportTier::Community);
//! assert!(!config.github_issues_url.is_empty());
//! ```

/// Support tier level.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SupportTier {
    /// Community support via GitHub issues and discussions.
    Community,
    /// Standard commercial support with guaranteed response times.
    Standard,
    /// Premium support with dedicated engineering contact.
    Premium,
    /// Enterprise support with SLA-backed response and resolution times.
    Enterprise,
}

impl core::fmt::Display for SupportTier {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Community => write!(f, "Community"),
            Self::Standard => write!(f, "Standard"),
            Self::Premium => write!(f, "Premium"),
            Self::Enterprise => write!(f, "Enterprise"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "Unknown"),
        }
    }
}

/// Support channel configuration.
///
/// Contains all the information needed for users and customers to reach
/// the SciRS2 support team.
#[derive(Debug, Clone)]
pub struct SupportConfig {
    /// GitHub issues URL for bug reports and feature requests.
    pub github_issues_url: &'static str,
    /// GitHub discussions URL for Q&A.
    pub github_discussions_url: &'static str,
    /// Documentation website URL.
    pub documentation_url: &'static str,
    /// API reference URL.
    pub api_reference_url: &'static str,
    /// Support email address.
    pub email: &'static str,
    /// Target response time in hours for the current tier.
    pub response_time_hours: u32,
    /// Target resolution time in hours for the current tier.
    pub resolution_time_hours: Option<u32>,
    /// Current support tier.
    pub tier: SupportTier,
    /// Security vulnerability reporting email.
    pub security_email: &'static str,
    /// Changelog / release notes URL.
    pub changelog_url: &'static str,
}

impl Default for SupportConfig {
    fn default() -> Self {
        Self {
            github_issues_url: "https://github.com/cool-japan/scirs/issues",
            github_discussions_url: "https://github.com/cool-japan/scirs/discussions",
            documentation_url: "https://cool-japan.github.io/scirs/",
            api_reference_url: "https://docs.rs/scirs2-core/latest/scirs2_core/",
            email: "support@cooljapan.dev",
            response_time_hours: 48,
            resolution_time_hours: None,
            tier: SupportTier::Community,
            security_email: "security@cooljapan.dev",
            changelog_url: "https://github.com/cool-japan/scirs/blob/master/CHANGELOG.md",
        }
    }
}

impl SupportConfig {
    /// Returns a support configuration for the given tier.
    pub fn for_tier(tier: SupportTier) -> Self {
        match tier {
            SupportTier::Community => Self::default(),
            SupportTier::Standard => Self {
                response_time_hours: 24,
                resolution_time_hours: Some(72),
                tier: SupportTier::Standard,
                ..Self::default()
            },
            SupportTier::Premium => Self {
                response_time_hours: 8,
                resolution_time_hours: Some(48),
                tier: SupportTier::Premium,
                ..Self::default()
            },
            SupportTier::Enterprise => Self {
                response_time_hours: 4,
                resolution_time_hours: Some(24),
                tier: SupportTier::Enterprise,
                ..Self::default()
            },
            #[allow(unreachable_patterns)]
            _ => Self::default(),
        }
    }

    /// Returns a formatted summary of the support configuration.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("SciRS2 Support Configuration ({})", self.tier));
        lines.push(format!("  Issues:        {}", self.github_issues_url));
        lines.push(format!("  Discussions:   {}", self.github_discussions_url));
        lines.push(format!("  Documentation: {}", self.documentation_url));
        lines.push(format!("  API Reference: {}", self.api_reference_url));
        lines.push(format!("  Email:         {}", self.email));
        lines.push(format!("  Security:      {}", self.security_email));
        lines.push(format!(
            "  Response time: {} hours",
            self.response_time_hours
        ));
        if let Some(resolution) = self.resolution_time_hours {
            lines.push(format!("  Resolution:    {} hours", resolution));
        }
        lines.push(format!("  Changelog:     {}", self.changelog_url));
        lines.join("\n")
    }
}

/// Returns the default (community) support configuration.
pub fn support_info() -> SupportConfig {
    SupportConfig::default()
}

/// Escalation path for support issues.
#[derive(Debug, Clone)]
pub struct EscalationPath {
    /// Steps in the escalation path, from least to most urgent.
    pub steps: Vec<EscalationStep>,
}

/// A single step in the escalation path.
#[derive(Debug, Clone)]
pub struct EscalationStep {
    /// Step number (1-indexed).
    pub step: u32,
    /// Description of the action.
    pub action: String,
    /// Channel to use (e.g. "GitHub Issues", "Email", "Phone").
    pub channel: String,
    /// Expected wait time in hours before escalating to the next step.
    pub wait_hours: u32,
}

/// Returns the default escalation path for support issues.
pub fn default_escalation_path() -> EscalationPath {
    EscalationPath {
        steps: vec![
            EscalationStep {
                step: 1,
                action: "Search existing GitHub issues and documentation".into(),
                channel: "Self-service".into(),
                wait_hours: 0,
            },
            EscalationStep {
                step: 2,
                action: "Open a new GitHub issue with reproduction steps".into(),
                channel: "GitHub Issues".into(),
                wait_hours: 48,
            },
            EscalationStep {
                step: 3,
                action: "Post in GitHub Discussions for community help".into(),
                channel: "GitHub Discussions".into(),
                wait_hours: 24,
            },
            EscalationStep {
                step: 4,
                action: "Email support team with issue link".into(),
                channel: "Email".into(),
                wait_hours: 24,
            },
            EscalationStep {
                step: 5,
                action: "For security issues, email security team directly".into(),
                channel: "Security Email".into(),
                wait_hours: 0,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_config_default() {
        let config = SupportConfig::default();
        assert!(!config.github_issues_url.is_empty());
        assert!(!config.documentation_url.is_empty());
        assert!(!config.email.is_empty());
        assert!(config.response_time_hours > 0);
        assert_eq!(config.tier, SupportTier::Community);
    }

    #[test]
    fn test_support_info_returns_community() {
        let config = support_info();
        assert_eq!(config.tier, SupportTier::Community);
    }

    #[test]
    fn test_support_tiers_response_times() {
        let community = SupportConfig::for_tier(SupportTier::Community);
        let standard = SupportConfig::for_tier(SupportTier::Standard);
        let premium = SupportConfig::for_tier(SupportTier::Premium);
        let enterprise = SupportConfig::for_tier(SupportTier::Enterprise);

        assert!(community.response_time_hours >= standard.response_time_hours);
        assert!(standard.response_time_hours >= premium.response_time_hours);
        assert!(premium.response_time_hours >= enterprise.response_time_hours);
    }

    #[test]
    fn test_support_tier_display() {
        assert_eq!(SupportTier::Community.to_string(), "Community");
        assert_eq!(SupportTier::Standard.to_string(), "Standard");
        assert_eq!(SupportTier::Premium.to_string(), "Premium");
        assert_eq!(SupportTier::Enterprise.to_string(), "Enterprise");
    }

    #[test]
    fn test_support_config_summary() {
        let config = SupportConfig::for_tier(SupportTier::Premium);
        let summary = config.summary();
        assert!(summary.contains("Premium"));
        assert!(summary.contains("8 hours"));
        assert!(summary.contains("48 hours"));
        assert!(summary.contains("github.com"));
    }

    #[test]
    fn test_escalation_path() {
        let path = default_escalation_path();
        assert!(
            path.steps.len() >= 4,
            "Expected at least 4 escalation steps"
        );
        for (i, step) in path.steps.iter().enumerate() {
            assert_eq!(step.step as usize, i + 1);
            assert!(!step.action.is_empty());
            assert!(!step.channel.is_empty());
        }
    }

    #[test]
    fn test_enterprise_tier_has_resolution_time() {
        let config = SupportConfig::for_tier(SupportTier::Enterprise);
        assert!(
            config.resolution_time_hours.is_some(),
            "Enterprise tier must have resolution time"
        );
    }

    #[test]
    fn test_security_email_present() {
        let config = support_info();
        assert!(
            config.security_email.contains('@'),
            "Security email must be a valid email"
        );
    }
}
