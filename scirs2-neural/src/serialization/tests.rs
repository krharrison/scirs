//! Tests for legacy model serialization (Sequential JSON format).
//!
//! Requires `legacy_serialization` feature. For modern SafeTensors-based
//! serialization tests (ResNet, BERT), see `model_serializer.rs`.

// Legacy serialization tests are in model_serializer.rs under #[cfg(test)].
// The legacy JSON serialization for Sequential models is deprecated.
// New code should use ModelSerializer trait with SafeTensors format.
