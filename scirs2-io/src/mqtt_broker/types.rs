//! MQTT-style broker types: QoS levels, messages, and broker configuration.

use std::time::SystemTime;

/// Quality-of-Service level for message delivery.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QosLevel {
    /// Fire-and-forget: message is delivered at most once.
    AtMostOnce,
    /// Message is delivered at least once (retried until acknowledged).
    AtLeastOnce,
    /// Message is delivered exactly once (simplified guarantee).
    ExactlyOnce,
}

/// An MQTT-style message carrying a topic, payload, and delivery metadata.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MqttMessage {
    /// The topic this message was published to.
    pub topic: String,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
    /// Quality-of-service level.
    pub qos: QosLevel,
    /// Whether the broker should retain this message for new subscribers.
    pub retained: bool,
    /// Time the message was created.
    pub timestamp: SystemTime,
}

impl MqttMessage {
    /// Create a new message with default QoS (AtMostOnce) and not retained.
    pub fn new(topic: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            topic: topic.into(),
            payload,
            qos: QosLevel::AtMostOnce,
            retained: false,
            timestamp: SystemTime::now(),
        }
    }

    /// Builder: set QoS level.
    pub fn with_qos(mut self, qos: QosLevel) -> Self {
        self.qos = qos;
        self
    }

    /// Builder: mark as retained.
    pub fn retained(mut self) -> Self {
        self.retained = true;
        self
    }
}

/// Configuration for an `InProcessBroker`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BrokerConfig {
    /// Maximum number of messages held in the delivery queue at once.
    pub max_queue_depth: usize,
    /// Whether the broker stores the last retained message per topic.
    pub retain_messages: bool,
    /// Whether `+` and `#` wildcards are supported in subscription patterns.
    pub wildcard_enabled: bool,
}

impl Default for BrokerConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 1000,
            retain_messages: true,
            wildcard_enabled: true,
        }
    }
}
