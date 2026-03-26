//! In-process MQTT-style topic broker.
//!
//! Provides `InProcessBroker` (single-threaded, owned) and `BrokerHandle`
//! (thread-safe shared handle backed by `Arc<Mutex<InProcessBroker>>`).
//!
//! Topic matching rules:
//! - `+` matches exactly one topic level: `a/+/c` matches `a/x/c` but not `a/x/y/c`.
//! - `#` must appear as the last segment and matches all remaining levels:
//!   `a/#` matches `a/b`, `a/b/c`, `a`.
//! - Exact patterns match only exact topics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::types::{BrokerConfig, MqttMessage, QosLevel};

/// A callback invoked when a matching message is published.
pub type MessageHandler = Box<dyn Fn(&MqttMessage) + Send + 'static>;

/// A subscription: a (client_id, handler) pair for a topic pattern.
struct Subscription {
    client_id: String,
    handler: MessageHandler,
}

/// In-process, synchronous topic broker.
///
/// Dispatches published messages to all registered subscribers whose pattern
/// matches the message topic.  Supports MQTT wildcard semantics (`+` and `#`).
pub struct InProcessBroker {
    config: BrokerConfig,
    /// Map from topic pattern → list of subscriptions.
    subscriptions: HashMap<String, Vec<Subscription>>,
    /// Retained messages: last value published per topic with `retained = true`.
    retained: HashMap<String, MqttMessage>,
    /// Pending outbound messages (for QoS tracking; simple ring buffer approach).
    message_queue: Vec<MqttMessage>,
}

impl InProcessBroker {
    /// Create a new broker with the given configuration.
    pub fn new(config: BrokerConfig) -> Self {
        Self {
            config,
            subscriptions: HashMap::new(),
            retained: HashMap::new(),
            message_queue: Vec::new(),
        }
    }

    // ── Subscribe / Unsubscribe ────────────────────────────────────────────

    /// Subscribe `client_id` to `pattern`, calling `handler` on each match.
    ///
    /// If the broker has a retained message for a topic matching `pattern`,
    /// `handler` is called immediately with that retained message.
    pub fn subscribe(&mut self, client_id: &str, pattern: &str, handler: MessageHandler) {
        // Deliver any currently-retained messages matching the pattern.
        if self.config.retain_messages {
            let matching: Vec<MqttMessage> = self
                .retained
                .iter()
                .filter(|(topic, _)| Self::matches_pattern(pattern, topic))
                .map(|(_, msg)| msg.clone())
                .collect();
            for msg in matching {
                handler(&msg);
            }
        }

        let subs = self.subscriptions.entry(pattern.to_owned()).or_default();
        subs.push(Subscription {
            client_id: client_id.to_owned(),
            handler,
        });
    }

    /// Remove all subscriptions for `client_id` under `pattern`.
    pub fn unsubscribe(&mut self, client_id: &str, pattern: &str) {
        if let Some(subs) = self.subscriptions.get_mut(pattern) {
            subs.retain(|s| s.client_id != client_id);
            if subs.is_empty() {
                self.subscriptions.remove(pattern);
            }
        }
    }

    // ── Publish ────────────────────────────────────────────────────────────

    /// Publish a message, dispatching it to all matching subscribers.
    ///
    /// Returns the number of handlers invoked.
    pub fn publish(&mut self, message: MqttMessage) -> usize {
        // Store/update retention.
        if message.retained && self.config.retain_messages {
            self.retained.insert(message.topic.clone(), message.clone());
        }

        // Enqueue (cap at max_queue_depth).
        if self.message_queue.len() < self.config.max_queue_depth {
            self.message_queue.push(message.clone());
        }

        // Dispatch to matching subscribers.
        let mut delivered = 0usize;

        // Collect matching (pattern, handler indices) to avoid borrow issues.
        let matching_patterns: Vec<String> = self
            .subscriptions
            .keys()
            .filter(|p| Self::matches_pattern(p, &message.topic))
            .cloned()
            .collect();

        for pattern in matching_patterns {
            if let Some(subs) = self.subscriptions.get(&pattern) {
                for sub in subs {
                    (sub.handler)(&message);
                    delivered += 1;
                }
            }
        }

        delivered
    }

    /// Convenience: publish a raw payload to `topic` with default QoS.
    pub fn publish_simple(&mut self, topic: &str, payload: Vec<u8>) -> usize {
        self.publish(MqttMessage::new(topic, payload))
    }

    // ── Retained messages ──────────────────────────────────────────────────

    /// Retrieve the retained message for an exact `topic`, if any.
    pub fn get_retained(&self, topic: &str) -> Option<&MqttMessage> {
        self.retained.get(topic)
    }

    /// Drain the internal message queue and return all queued messages.
    pub fn drain_queue(&mut self) -> Vec<MqttMessage> {
        std::mem::take(&mut self.message_queue)
    }

    /// Current queue depth.
    pub fn queue_depth(&self) -> usize {
        self.message_queue.len()
    }

    // ── Topic matching ─────────────────────────────────────────────────────

    /// Returns `true` if `topic` matches `pattern` under MQTT wildcard rules.
    ///
    /// - `+` matches exactly one level.
    /// - `#` (only valid as the last segment) matches zero or more levels.
    /// - Otherwise, exact character match per segment.
    pub fn matches_pattern(pattern: &str, topic: &str) -> bool {
        let p_parts: Vec<&str> = pattern.split('/').collect();
        let t_parts: Vec<&str> = topic.split('/').collect();

        matches_parts(&p_parts, &t_parts)
    }
}

/// Recursive helper for segment-by-segment MQTT wildcard matching.
fn matches_parts(pattern: &[&str], topic: &[&str]) -> bool {
    match (pattern.first(), topic.first()) {
        (None, None) => true,
        (Some(&"#"), _) => true, // # matches anything remaining (including empty)
        (None, Some(_)) => false, // pattern exhausted but topic has more
        (Some(_), None) => false, // topic exhausted but pattern has more (no #)
        (Some(&"+"), Some(_)) => matches_parts(&pattern[1..], &topic[1..]),
        (Some(&p), Some(&t)) => p == t && matches_parts(&pattern[1..], &topic[1..]),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BrokerHandle
// ─────────────────────────────────────────────────────────────────────────────

/// A thread-safe shared handle to an `InProcessBroker`.
#[derive(Clone)]
pub struct BrokerHandle(Arc<Mutex<InProcessBroker>>);

impl BrokerHandle {
    /// Create a new handle wrapping a fresh `InProcessBroker`.
    pub fn new(config: BrokerConfig) -> Self {
        Self(Arc::new(Mutex::new(InProcessBroker::new(config))))
    }

    /// Subscribe to a pattern.  The handler is called under the broker lock,
    /// so it must not attempt to re-enter the broker.
    pub fn subscribe(&self, client_id: &str, pattern: &str, handler: MessageHandler) {
        if let Ok(mut b) = self.0.lock() {
            b.subscribe(client_id, pattern, handler);
        }
    }

    /// Unsubscribe a client from a pattern.
    pub fn unsubscribe(&self, client_id: &str, pattern: &str) {
        if let Ok(mut b) = self.0.lock() {
            b.unsubscribe(client_id, pattern);
        }
    }

    /// Publish a message. Returns the delivery count, or 0 on lock failure.
    pub fn publish(&self, message: MqttMessage) -> usize {
        self.0.lock().map(|mut b| b.publish(message)).unwrap_or(0)
    }

    /// Convenience: publish a raw payload.
    pub fn publish_simple(&self, topic: &str, payload: Vec<u8>) -> usize {
        self.publish(MqttMessage::new(topic, payload))
    }

    /// Get a clone of the retained message for `topic`.
    pub fn get_retained(&self, topic: &str) -> Option<MqttMessage> {
        self.0
            .lock()
            .ok()
            .and_then(|b| b.get_retained(topic).cloned())
    }

    /// Queue depth.
    pub fn queue_depth(&self) -> usize {
        self.0.lock().map(|b| b.queue_depth()).unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn default_broker() -> InProcessBroker {
        InProcessBroker::new(BrokerConfig::default())
    }

    // ── Topic matching ────────────────────────────────────────────────────

    #[test]
    fn test_exact_match() {
        assert!(InProcessBroker::matches_pattern(
            "sensor/1/temp",
            "sensor/1/temp"
        ));
        assert!(!InProcessBroker::matches_pattern(
            "sensor/1/temp",
            "sensor/2/temp"
        ));
    }

    #[test]
    fn test_plus_wildcard_single_level() {
        assert!(InProcessBroker::matches_pattern(
            "sensor/+/temp",
            "sensor/1/temp"
        ));
        assert!(InProcessBroker::matches_pattern(
            "sensor/+/temp",
            "sensor/abc/temp"
        ));
        assert!(!InProcessBroker::matches_pattern(
            "sensor/+/temp",
            "sensor/1/2/temp"
        ));
    }

    #[test]
    fn test_hash_wildcard_multi_level() {
        assert!(InProcessBroker::matches_pattern("sensor/#", "sensor/1"));
        assert!(InProcessBroker::matches_pattern(
            "sensor/#",
            "sensor/1/temp"
        ));
        assert!(InProcessBroker::matches_pattern("sensor/#", "sensor/a/b/c"));
        assert!(!InProcessBroker::matches_pattern("sensor/#", "other/1"));
        // # at root matches everything
        assert!(InProcessBroker::matches_pattern("#", "any/topic/here"));
    }

    // ── Publish / Subscribe ───────────────────────────────────────────────

    #[test]
    fn test_publish_handler_called_once() {
        let mut broker = default_broker();
        let counter = Arc::new(Mutex::new(0usize));
        let c = Arc::clone(&counter);
        broker.subscribe(
            "client1",
            "a/b",
            Box::new(move |_| {
                *c.lock().unwrap_or_else(|e| e.into_inner()) += 1;
            }),
        );
        let delivered = broker.publish(MqttMessage::new("a/b", b"hello".to_vec()));
        assert_eq!(delivered, 1);
        assert_eq!(*counter.lock().unwrap_or_else(|e| e.into_inner()), 1);
    }

    #[test]
    fn test_no_match_not_called() {
        let mut broker = default_broker();
        let counter = Arc::new(Mutex::new(0usize));
        let c = Arc::clone(&counter);
        broker.subscribe(
            "c",
            "x/y",
            Box::new(move |_| {
                *c.lock().unwrap_or_else(|e| e.into_inner()) += 1;
            }),
        );
        let delivered = broker.publish(MqttMessage::new("a/b", b"nope".to_vec()));
        assert_eq!(delivered, 0);
        assert_eq!(*counter.lock().unwrap_or_else(|e| e.into_inner()), 0);
    }

    #[test]
    fn test_retained_message_stored() {
        let mut broker = default_broker();
        let msg = MqttMessage::new("t/temp", b"42".to_vec()).retained();
        broker.publish(msg);
        let retained = broker.get_retained("t/temp");
        assert!(retained.is_some());
        assert_eq!(retained.unwrap().payload, b"42");
    }

    #[test]
    fn test_get_retained_returns_correct_message() {
        let mut broker = default_broker();
        broker.publish(MqttMessage::new("a", b"first".to_vec()).retained());
        broker.publish(MqttMessage::new("a", b"second".to_vec()).retained());
        let retained = broker.get_retained("a").expect("should exist");
        assert_eq!(retained.payload, b"second");
    }

    #[test]
    fn test_unsubscribe_not_called() {
        let mut broker = default_broker();
        let counter = Arc::new(Mutex::new(0usize));
        let c = Arc::clone(&counter);
        broker.subscribe(
            "c1",
            "data/#",
            Box::new(move |_| {
                *c.lock().unwrap_or_else(|e| e.into_inner()) += 1;
            }),
        );
        broker.unsubscribe("c1", "data/#");
        broker.publish(MqttMessage::new("data/sensor", b"x".to_vec()));
        assert_eq!(*counter.lock().unwrap_or_else(|e| e.into_inner()), 0);
    }

    #[test]
    fn test_multiple_subscribers_same_topic() {
        let mut broker = default_broker();
        let counter = Arc::new(Mutex::new(0usize));
        for _ in 0..3 {
            let c = Arc::clone(&counter);
            broker.subscribe(
                "cx",
                "t",
                Box::new(move |_| {
                    *c.lock().unwrap_or_else(|e| e.into_inner()) += 1;
                }),
            );
        }
        broker.publish(MqttMessage::new("t", b"data".to_vec()));
        assert_eq!(*counter.lock().unwrap_or_else(|e| e.into_inner()), 3);
    }

    #[test]
    fn test_broker_handle_thread_safe_publish() {
        let handle = BrokerHandle::new(BrokerConfig::default());
        let counter = Arc::new(Mutex::new(0usize));
        let c = Arc::clone(&counter);
        handle.subscribe(
            "h",
            "test/+",
            Box::new(move |_| {
                *c.lock().unwrap_or_else(|e| e.into_inner()) += 1;
            }),
        );
        handle.publish(MqttMessage::new("test/1", b"x".to_vec()));
        assert_eq!(*counter.lock().unwrap_or_else(|e| e.into_inner()), 1);
    }

    #[test]
    fn test_qos_level_ordering() {
        assert!(QosLevel::AtMostOnce < QosLevel::AtLeastOnce);
        assert!(QosLevel::AtLeastOnce < QosLevel::ExactlyOnce);
    }

    #[test]
    fn test_broker_config_default() {
        let cfg = BrokerConfig::default();
        assert_eq!(cfg.max_queue_depth, 1000);
        assert!(cfg.retain_messages);
        assert!(cfg.wildcard_enabled);
    }
}
