//! Quantization-aware training utilities
//!
//! Re-exports from the main quantization module for convenience.

/// Configuration for quantization-aware training
#[derive(Debug, Clone)]
pub struct QuantizationAwareConfig {
    /// Whether to enable quantization-aware training
    pub enabled: bool,
    /// Bit width for quantization
    pub bit_width: u8,
    /// Warmup steps before quantization kicks in
    pub warmup_steps: usize,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// EMA factor for updating quantization parameters
    pub ema_factor: f32,
}

impl Default for QuantizationAwareConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bit_width: 8,
            warmup_steps: 1000,
            symmetric: true,
            ema_factor: 0.01,
        }
    }
}

impl QuantizationAwareConfig {
    /// Create a configuration with quantization enabled
    pub fn enabled(bit_width: u8) -> Self {
        Self {
            enabled: true,
            bit_width,
            ..Default::default()
        }
    }

    /// Set asymmetric quantization mode
    pub fn with_asymmetric(mut self) -> Self {
        self.symmetric = false;
        self
    }

    /// Set warmup steps
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Convert to the main quantization module's config
    pub fn to_quantization_config(&self) -> crate::quantization::QuantizationConfig {
        crate::quantization::QuantizationConfig {
            bits: self.bit_width,
            signed: true,
            scheme: if self.symmetric {
                crate::quantization::QuantizationScheme::Symmetric
            } else {
                crate::quantization::QuantizationScheme::Asymmetric
            },
            calibration_size: 1000,
            mode: crate::quantization::QuantizationMode::QAT,
            per_channel: false,
            range_clipping: 0.999,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_config_default() {
        let config = QuantizationAwareConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.bit_width, 8);
        assert_eq!(config.warmup_steps, 1000);
        assert!(config.symmetric);
    }

    #[test]
    fn test_qat_config_enabled() {
        let config = QuantizationAwareConfig::enabled(4);
        assert!(config.enabled);
        assert_eq!(config.bit_width, 4);
    }

    #[test]
    fn test_qat_config_builder() {
        let config = QuantizationAwareConfig::enabled(8)
            .with_asymmetric()
            .with_warmup_steps(500);
        assert!(config.enabled);
        assert!(!config.symmetric);
        assert_eq!(config.warmup_steps, 500);
    }

    #[test]
    fn test_to_quantization_config() {
        let config = QuantizationAwareConfig::enabled(8);
        let qconfig = config.to_quantization_config();
        assert_eq!(qconfig.bits, 8);
        assert_eq!(
            qconfig.scheme,
            crate::quantization::QuantizationScheme::Symmetric
        );
        assert_eq!(qconfig.mode, crate::quantization::QuantizationMode::QAT);

        let asym_config = QuantizationAwareConfig::enabled(4).with_asymmetric();
        let asym_qconfig = asym_config.to_quantization_config();
        assert_eq!(
            asym_qconfig.scheme,
            crate::quantization::QuantizationScheme::Asymmetric
        );
    }
}
