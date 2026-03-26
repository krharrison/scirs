//! In-process MQTT-style topic-based message broker.
//!
//! Provides publish/subscribe semantics with full MQTT wildcard support:
//!
//! - `+` matches a single topic level
//! - `#` (last segment) matches all remaining levels
//!
//! No network, no external crates — everything runs in-process.
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_io::mqtt_broker::{BrokerConfig, BrokerHandle, MqttMessage};
//!
//! let handle = BrokerHandle::new(BrokerConfig::default());
//! handle.subscribe("sensor-monitor", "sensor/+/temp", Box::new(|msg| {
//!     println!("Received on {}: {:?}", msg.topic, msg.payload);
//! }));
//! handle.publish(MqttMessage::new("sensor/1/temp", b"23.5".to_vec()));
//! ```

pub mod broker;
pub mod types;

pub use broker::{BrokerHandle, InProcessBroker, MessageHandler};
pub use types::{BrokerConfig, MqttMessage, QosLevel};
