//! Structured log file parsing
//!
//! Provides parsers for common log formats:
//! - Apache/Nginx combined/common log format
//! - Structured JSON log lines
//! - RFC 5424 syslog format
//!
//! Plus a `LogAggregator` for filtering, bucketing, and summarising log streams.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{IoError, Result};

// ──────────────────────────────────────────────────────────────────────────────
// Core types
// ──────────────────────────────────────────────────────────────────────────────

/// Severity level for a log entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Trace-level (most verbose) messages.
    Trace,
    /// Debug-level messages.
    Debug,
    /// Informational messages.
    Info,
    /// Notice-level messages (slightly more severe than Info).
    Notice,
    /// Warning conditions.
    Warning,
    /// Error conditions.
    Error,
    /// Critical conditions.
    Critical,
    /// Action must be taken immediately.
    Alert,
    /// System is unusable.
    Emergency,
    /// Unrecognised severity string.
    Unknown(String),
}

impl LogLevel {
    fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "TRACE" | "TRC" => LogLevel::Trace,
            "DEBUG" | "DBG" | "7" => LogLevel::Debug,
            "INFO" | "INFORMATION" | "6" => LogLevel::Info,
            "NOTICE" | "5" => LogLevel::Notice,
            "WARN" | "WARNING" | "4" => LogLevel::Warning,
            "ERROR" | "ERR" | "3" => LogLevel::Error,
            "CRIT" | "CRITICAL" | "2" => LogLevel::Critical,
            "ALERT" | "1" => LogLevel::Alert,
            "EMERG" | "EMERGENCY" | "0" => LogLevel::Emergency,
            other => LogLevel::Unknown(other.to_string()),
        }
    }

    /// Display as a canonical uppercase string.
    pub fn as_str(&self) -> &str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Notice => "NOTICE",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
            LogLevel::Critical => "CRITICAL",
            LogLevel::Alert => "ALERT",
            LogLevel::Emergency => "EMERGENCY",
            LogLevel::Unknown(s) => s.as_str(),
        }
    }
}

/// A structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Parsed timestamp, Unix millis (0 if not available).
    pub timestamp_ms: i64,
    /// Severity level.
    pub level: LogLevel,
    /// Primary log message.
    pub message: String,
    /// Optional hostname / source identifier.
    pub host: Option<String>,
    /// Optional process/application identifier.
    pub app: Option<String>,
    /// Optional process ID.
    pub pid: Option<u32>,
    /// Optional structured fields extracted from the entry.
    pub fields: HashMap<String, String>,
    /// The raw log line (before parsing).
    pub raw: String,
}

impl LogEntry {
    fn new_raw(raw: impl Into<String>) -> Self {
        LogEntry {
            timestamp_ms: 0,
            level: LogLevel::Unknown(String::new()),
            message: String::new(),
            host: None,
            app: None,
            pid: None,
            fields: HashMap::new(),
            raw: raw.into(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Timestamp parsing
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a timestamp string in multiple common formats.
///
/// Supported formats (examples):
/// - Unix millis: `"1711234567890"`
/// - Unix secs (decimal): `"1711234567.123"`
/// - ISO 8601 / RFC 3339: `"2024-03-23T14:22:47Z"`, `"2024-03-23T14:22:47.123+05:30"`
/// - Apache CLF: `"23/Mar/2024:14:22:47 +0000"`
/// - Syslog: `"Mar 23 14:22:47"`
///
/// Returns Unix milliseconds, or an error if none of the formats match.
pub fn parse_timestamp(s: &str) -> Result<i64> {
    let trimmed = s.trim();

    // 1. Pure integer (Unix millis or seconds)
    if let Ok(n) = trimmed.parse::<i64>() {
        // Heuristic: 13-digit → already millis; fewer → seconds
        return if n.abs() > 9_999_999_999 {
            Ok(n)
        } else {
            Ok(n * 1000)
        };
    }

    // 2. Float (Unix secs with sub-second fraction)
    if let Ok(f) = trimmed.parse::<f64>() {
        return Ok((f * 1000.0) as i64);
    }

    // 3. ISO 8601 / RFC 3339
    if let Some(ms) = parse_iso8601(trimmed) {
        return Ok(ms);
    }

    // 4. Apache CLF: "23/Mar/2024:14:22:47 +0000"
    if let Some(ms) = parse_clf_timestamp(trimmed) {
        return Ok(ms);
    }

    // 5. Syslog: "Mar 23 14:22:47"
    if let Some(ms) = parse_syslog_timestamp(trimmed) {
        return Ok(ms);
    }

    Err(IoError::ParseError(format!(
        "unrecognised timestamp format: '{}'",
        s
    )))
}

/// Parse ISO 8601 / RFC 3339 timestamps.
fn parse_iso8601(s: &str) -> Option<i64> {
    // Minimum: "YYYY-MM-DDTHH:MM:SS"
    if s.len() < 19 {
        return None;
    }
    let year: i64 = s[0..4].parse().ok()?;
    let month: i64 = s[5..7].parse().ok()?;
    let day: i64 = s[8..10].parse().ok()?;
    if s.as_bytes().get(10) != Some(&b'T') && s.as_bytes().get(10) != Some(&b' ') {
        return None;
    }
    let hour: i64 = s[11..13].parse().ok()?;
    let min: i64 = s[14..16].parse().ok()?;
    let sec: i64 = s[17..19].parse().ok()?;

    // Optional fractional seconds
    let mut frac_ms: i64 = 0;
    let mut rest = &s[19..];
    if rest.starts_with('.') {
        let end = rest[1..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len() - 1);
        let frac_str = &rest[1..end + 1];
        // Normalise to millis (3 digits)
        frac_ms = match frac_str.len() {
            0 => 0,
            1 => frac_str.parse::<i64>().unwrap_or(0) * 100,
            2 => frac_str.parse::<i64>().unwrap_or(0) * 10,
            _ => frac_str[..3].parse::<i64>().unwrap_or(0),
        };
        rest = &rest[end + 1..];
    }

    // Timezone offset (ignored for simplicity — treat as UTC)
    let _tz = rest; // "Z", "+HH:MM", "-HH:MM"

    let days_since_epoch = days_from_ymd(year, month, day)?;
    let secs = days_since_epoch * 86400 + hour * 3600 + min * 60 + sec;
    Some(secs * 1000 + frac_ms)
}

fn days_from_ymd(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // Zeller-like algorithm to get days since Unix epoch (1970-01-01)
    let m = if month <= 2 { month + 12 } else { month };
    let y = if month <= 2 { year - 1 } else { year };
    let jdn = day + (153 * m - 457) / 5 + 365 * y + y / 4 - y / 100 + y / 400 + 1721119;
    let unix_epoch_jdn: i64 = 2440588; // JDN of 1970-01-01
    Some(jdn - unix_epoch_jdn)
}

const MONTHS: &[&str] = &[
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn month_from_abbr(s: &str) -> Option<i64> {
    MONTHS
        .iter()
        .position(|m| m.eq_ignore_ascii_case(s))
        .map(|i| i as i64 + 1)
}

/// Parse Apache CLF timestamp: `"23/Mar/2024:14:22:47 +0000"`.
fn parse_clf_timestamp(s: &str) -> Option<i64> {
    // Format: DD/Mon/YYYY:HH:MM:SS ±HHMM
    let parts: Vec<&str> = s.splitn(4, '/').collect();
    if parts.len() < 3 {
        return None;
    }
    let day: i64 = parts[0].parse().ok()?;
    let month = month_from_abbr(parts[1])?;
    let rest = parts[2]; // "2024:14:22:47 +0000"
    let colon = rest.find(':')?;
    let year: i64 = rest[..colon].parse().ok()?;
    let time = &rest[colon + 1..]; // "14:22:47 +0000"
    let time_parts: Vec<&str> = time.splitn(4, ':').collect();
    if time_parts.len() < 3 {
        return None;
    }
    let hour: i64 = time_parts[0].parse().ok()?;
    let min: i64 = time_parts[1].parse().ok()?;
    let sec_str = time_parts[2].split_whitespace().next()?;
    let sec: i64 = sec_str.parse().ok()?;

    let days = days_from_ymd(year, month, day)?;
    Some((days * 86400 + hour * 3600 + min * 60 + sec) * 1000)
}

/// Parse syslog timestamp: `"Mar 23 14:22:47"` (no year — use 1970).
fn parse_syslog_timestamp(s: &str) -> Option<i64> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let month = month_from_abbr(parts[0])?;
    let day: i64 = parts[1].parse().ok()?;
    let time_parts: Vec<&str> = parts[2].split(':').collect();
    if time_parts.len() < 3 {
        return None;
    }
    let hour: i64 = time_parts[0].parse().ok()?;
    let min: i64 = time_parts[1].parse().ok()?;
    let sec: i64 = time_parts[2].parse().ok()?;

    let days = days_from_ymd(1970, month, day)?;
    Some((days * 86400 + hour * 3600 + min * 60 + sec) * 1000)
}

// ──────────────────────────────────────────────────────────────────────────────
// CommonLogParser (Apache / Nginx combined log format)
// ──────────────────────────────────────────────────────────────────────────────

/// Parser for Apache/Nginx Common Log / Combined Log Format.
///
/// CLF line format:
/// ```text
/// 127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326
/// ```
///
/// Combined log adds: `"http://www.example.com/start.html" "Mozilla/5.0 ..."`
#[derive(Debug, Default)]
pub struct CommonLogParser;

impl CommonLogParser {
    /// Create a new `CommonLogParser`.
    pub fn new() -> Self {
        CommonLogParser
    }

    /// Parse a single CLF / combined log line.
    pub fn parse(&self, line: &str) -> Result<LogEntry> {
        let mut entry = LogEntry::new_raw(line);
        entry.level = LogLevel::Info;

        let mut rest = line.trim();

        // 1. Remote host
        let (remote_host, r) = split_field(rest);
        entry
            .fields
            .insert("remote_host".into(), remote_host.into());
        rest = r.trim_start();

        // 2. Ident (usually "-")
        let (ident, r) = split_field(rest);
        if ident != "-" {
            entry.fields.insert("ident".into(), ident.into());
        }
        rest = r.trim_start();

        // 3. Auth user
        let (auth_user, r) = split_field(rest);
        if auth_user != "-" {
            entry.fields.insert("auth_user".into(), auth_user.into());
        }
        rest = r.trim_start();

        // 4. Timestamp in brackets
        if rest.starts_with('[') {
            if let Some(end) = rest.find(']') {
                let ts_str = &rest[1..end];
                entry.timestamp_ms = parse_timestamp(ts_str).unwrap_or(0);
                rest = rest[end + 1..].trim_start();
            }
        }

        // 5. Request line in quotes
        if rest.starts_with('"') {
            let (req, r) = extract_quoted(rest);
            entry.message = req.to_string();
            let parts: Vec<&str> = req.splitn(3, ' ').collect();
            if parts.len() >= 2 {
                entry.fields.insert("method".into(), parts[0].into());
                entry.fields.insert("path".into(), parts[1].into());
            }
            if parts.len() == 3 {
                entry.fields.insert("protocol".into(), parts[2].into());
            }
            rest = r.trim_start();
        }

        // 6. Status code
        let (status, r) = split_field(rest);
        entry.fields.insert("status".into(), status.into());
        if let Ok(code) = status.parse::<u16>() {
            if code >= 400 {
                entry.level = LogLevel::Warning;
            }
            if code >= 500 {
                entry.level = LogLevel::Error;
            }
        }
        rest = r.trim_start();

        // 7. Bytes
        let (bytes, r) = split_field(rest);
        if bytes != "-" {
            entry.fields.insert("bytes".into(), bytes.into());
        }
        rest = r.trim_start();

        // 8. Optional referrer (combined log)
        if rest.starts_with('"') {
            let (referer, r) = extract_quoted(rest);
            if referer != "-" {
                entry.fields.insert("referer".into(), referer.to_string());
            }
            rest = r.trim_start();
        }

        // 9. Optional user-agent
        if rest.starts_with('"') {
            let (ua, _) = extract_quoted(rest);
            if ua != "-" {
                entry.fields.insert("user_agent".into(), ua.to_string());
            }
        }

        Ok(entry)
    }

    /// Parse all lines from a reader.
    pub fn parse_reader<R: BufRead>(&self, reader: R) -> Vec<Result<LogEntry>> {
        reader
            .lines()
            .filter_map(|lr| match lr {
                Err(e) => Some(Err(IoError::Io(e))),
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(self.parse(&trimmed))
                    }
                }
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// JsonLogParser
// ──────────────────────────────────────────────────────────────────────────────

/// Field name mappings for the JSON log parser.
#[derive(Debug, Clone)]
pub struct JsonLogFieldMap {
    /// Field key that holds the timestamp.
    pub timestamp: String,
    /// Field key that holds the severity level.
    pub level: String,
    /// Field key that holds the message.
    pub message: String,
    /// Field key that holds the hostname.
    pub host: String,
}

impl Default for JsonLogFieldMap {
    fn default() -> Self {
        JsonLogFieldMap {
            timestamp: "timestamp".into(),
            level: "level".into(),
            message: "message".into(),
            host: "host".into(),
        }
    }
}

/// Parser for structured JSON log lines (e.g., logfmt / Winston / Bunyan).
#[derive(Debug, Default)]
pub struct JsonLogParser {
    /// Mapping from JSON keys to canonical `LogEntry` fields.
    pub field_map: JsonLogFieldMap,
}

impl JsonLogParser {
    /// Create a new `JsonLogParser` with default field mappings.
    pub fn new() -> Self {
        JsonLogParser::default()
    }

    /// Create a `JsonLogParser` with a custom field name map.
    pub fn with_field_map(field_map: JsonLogFieldMap) -> Self {
        JsonLogParser { field_map }
    }

    /// Parse a single JSON log line into a `LogEntry`.
    pub fn parse(&self, line: &str) -> Result<LogEntry> {
        let mut entry = LogEntry::new_raw(line);

        // We parse the flat JSON object manually to avoid a serde dependency
        let inner = line
            .trim()
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .ok_or_else(|| IoError::ParseError("JSON log: not a JSON object".into()))?;

        let fields = parse_json_flat_fields(inner);

        // Map well-known fields
        if let Some(ts_str) = fields.get(&self.field_map.timestamp) {
            entry.timestamp_ms = parse_timestamp(ts_str).unwrap_or(0);
        }
        if let Some(lv) = fields.get(&self.field_map.level) {
            entry.level = LogLevel::from_str(lv);
        }
        if let Some(msg) = fields.get(&self.field_map.message) {
            entry.message = msg.clone();
        }
        if let Some(host) = fields.get(&self.field_map.host) {
            entry.host = Some(host.clone());
        }

        // All remaining fields go into entry.fields
        for (k, v) in &fields {
            entry.fields.insert(k.clone(), v.clone());
        }

        Ok(entry)
    }

    /// Parse all non-empty lines.
    pub fn parse_reader<R: BufRead>(&self, reader: R) -> Vec<Result<LogEntry>> {
        reader
            .lines()
            .filter_map(|lr| match lr {
                Err(e) => Some(Err(IoError::Io(e))),
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(self.parse(&trimmed))
                    }
                }
            })
            .collect()
    }
}

/// Very simple flat-JSON field extractor — handles string and number values.
fn parse_json_flat_fields(inner: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut rest = inner.trim();

    while !rest.is_empty() {
        rest = rest.trim_start_matches(',').trim();
        if rest.is_empty() {
            break;
        }
        // Read key
        let (key, after_key) = match parse_json_string_simple(rest) {
            Some(v) => v,
            None => break,
        };
        rest = after_key.trim_start();
        if !rest.starts_with(':') {
            break;
        }
        rest = rest[1..].trim_start();

        // Read value as string representation
        let (val_str, after_val) = extract_json_value_str(rest);
        map.insert(key, val_str);
        rest = after_val.trim_start();
    }
    map
}

/// Returns (unescaped_string, remaining_slice) for a JSON string starting at `s`.
fn parse_json_string_simple(s: &str) -> Option<(String, &str)> {
    if !s.starts_with('"') {
        return None;
    }
    let mut result = String::new();
    let mut chars = s[1..].char_indices();
    loop {
        match chars.next() {
            None => return None,
            Some((i, '"')) => return Some((result, &s[i + 2..])),
            Some((_, '\\')) => match chars.next() {
                Some((_, 'n')) => result.push('\n'),
                Some((_, 'r')) => result.push('\r'),
                Some((_, 't')) => result.push('\t'),
                Some((_, '"')) => result.push('"'),
                Some((_, '\\')) => result.push('\\'),
                Some((_, c)) => result.push(c),
                None => return None,
            },
            Some((_, c)) => result.push(c),
        }
    }
}

/// Extract a JSON value and return it as a String plus the remaining slice.
fn extract_json_value_str(s: &str) -> (String, &str) {
    if s.starts_with('"') {
        if let Some((val, rest)) = parse_json_string_simple(s) {
            return (val, rest);
        }
    }
    if s.starts_with('{') || s.starts_with('[') {
        // nested object/array — find matching bracket
        let open = s.as_bytes()[0];
        let close = if open == b'{' { b'}' } else { b']' };
        let mut depth = 0usize;
        let mut in_str = false;
        let mut escape = false;
        for (i, &b) in s.as_bytes().iter().enumerate() {
            if escape {
                escape = false;
                continue;
            }
            if b == b'\\' && in_str {
                escape = true;
                continue;
            }
            if b == b'"' {
                in_str = !in_str;
                continue;
            }
            if in_str {
                continue;
            }
            if b == open {
                depth += 1;
            } else if b == close {
                depth -= 1;
                if depth == 0 {
                    return (s[..=i].to_string(), &s[i + 1..]);
                }
            }
        }
    }
    // Primitive: read until comma, `}`, `]`
    let end = s.find([',', '}', ']']).unwrap_or(s.len());
    (s[..end].trim().to_string(), &s[end..])
}

// ──────────────────────────────────────────────────────────────────────────────
// SyslogParser (RFC 5424)
// ──────────────────────────────────────────────────────────────────────────────

/// RFC 5424 syslog parser.
///
/// Full syslog line format:
/// ```text
/// <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID STRUCTURED-DATA MSG
/// ```
/// Also handles legacy BSD syslog (RFC 3164).
#[derive(Debug, Default)]
pub struct SyslogParser;

impl SyslogParser {
    /// Create a new `SyslogParser`.
    pub fn new() -> Self {
        SyslogParser
    }

    /// Parse a single syslog line.
    pub fn parse(&self, line: &str) -> Result<LogEntry> {
        let mut entry = LogEntry::new_raw(line);
        let rest = line.trim();

        if rest.starts_with('<') {
            // Parse <PRI>
            if let Some(close) = rest.find('>') {
                let pri_str = &rest[1..close];
                if let Ok(pri) = pri_str.parse::<u8>() {
                    let severity = pri & 0x07;
                    entry.level = LogLevel::from_str(&severity.to_string());
                    let _facility = pri >> 3;
                    entry
                        .fields
                        .insert("facility".into(), (_facility).to_string());
                }
                let after_pri = &rest[close + 1..];
                // Check for RFC 5424 version digit
                let payload = if after_pri.starts_with(|c: char| c.is_ascii_digit()) {
                    // skip version
                    after_pri[1..].trim_start()
                } else {
                    after_pri.trim_start()
                };
                self.parse_syslog_payload(payload, &mut entry);
            }
        } else {
            // Fallback: treat as BSD syslog without priority
            self.parse_syslog_payload(rest, &mut entry);
        }

        Ok(entry)
    }

    fn parse_syslog_payload(&self, payload: &str, entry: &mut LogEntry) {
        let mut parts = payload.splitn(7, ' ');

        // TIMESTAMP
        if let Some(ts_str) = parts.next() {
            if ts_str != "-" {
                entry.timestamp_ms = parse_timestamp(ts_str).unwrap_or(0);
            }
        }

        // HOSTNAME
        if let Some(host) = parts.next() {
            if host != "-" {
                entry.host = Some(host.to_string());
            }
        }

        // APP-NAME
        if let Some(app) = parts.next() {
            if app != "-" {
                entry.app = Some(app.to_string());
            }
        }

        // PROCID
        if let Some(pid_str) = parts.next() {
            if pid_str != "-" {
                entry.pid = pid_str.parse::<u32>().ok();
            }
        }

        // MSGID
        if let Some(msgid) = parts.next() {
            if msgid != "-" {
                entry.fields.insert("msgid".into(), msgid.to_string());
            }
        }

        // STRUCTURED-DATA (skip for now)
        parts.next();

        // MESSAGE
        if let Some(msg) = parts.next() {
            // Strip optional BOM
            entry.message = msg.trim_start_matches('\u{FEFF}').to_string();
        }
    }

    /// Parse all lines from a reader.
    pub fn parse_reader<R: BufRead>(&self, reader: R) -> Vec<Result<LogEntry>> {
        reader
            .lines()
            .filter_map(|lr| match lr {
                Err(e) => Some(Err(IoError::Io(e))),
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(self.parse(&trimmed))
                    }
                }
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LogAggregator
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for log aggregation.
#[derive(Debug, Clone, Default)]
pub struct AggregationConfig {
    /// Keep only entries at or above this severity.
    pub min_level: Option<LogLevel>,
    /// Keep only entries containing this substring in their message.
    pub message_contains: Option<String>,
    /// Keep only entries from this host.
    pub host_filter: Option<String>,
    /// Time bucket width for time-based aggregation (seconds). 0 = no bucketing.
    pub bucket_secs: u64,
}

/// Aggregated summary over a set of log entries.
#[derive(Debug, Default, Clone)]
pub struct LogSummary {
    /// Total number of entries processed.
    pub total: usize,
    /// Count per severity level.
    pub by_level: HashMap<String, usize>,
    /// Count per host.
    pub by_host: HashMap<String, usize>,
    /// Count per time bucket (bucket_start_ms → count).
    pub by_bucket: HashMap<i64, usize>,
    /// Earliest timestamp seen (Unix ms).
    pub earliest_ms: i64,
    /// Latest timestamp seen (Unix ms).
    pub latest_ms: i64,
    /// Entries retained after filtering.
    pub retained: Vec<LogEntry>,
}

/// Aggregate and filter log entries.
pub struct LogAggregator {
    config: AggregationConfig,
}

impl LogAggregator {
    /// Create a new `LogAggregator` with the given aggregation configuration.
    pub fn new(config: AggregationConfig) -> Self {
        LogAggregator { config }
    }

    /// Process a slice of already-parsed `LogEntry` items.
    pub fn aggregate(&self, entries: &[LogEntry]) -> LogSummary {
        let mut summary = LogSummary {
            earliest_ms: i64::MAX,
            latest_ms: i64::MIN,
            ..Default::default()
        };

        for entry in entries {
            // Level filter
            if let Some(ref min_level) = self.config.min_level {
                if &entry.level < min_level {
                    continue;
                }
            }

            // Message filter
            if let Some(ref substr) = self.config.message_contains {
                if !entry.message.contains(substr.as_str()) {
                    continue;
                }
            }

            // Host filter
            if let Some(ref host_filter) = self.config.host_filter {
                if entry.host.as_deref() != Some(host_filter.as_str()) {
                    continue;
                }
            }

            // Passed all filters
            summary.total += 1;

            *summary
                .by_level
                .entry(entry.level.as_str().to_string())
                .or_insert(0) += 1;

            if let Some(ref host) = entry.host {
                *summary.by_host.entry(host.clone()).or_insert(0) += 1;
            }

            if entry.timestamp_ms > 0 {
                if entry.timestamp_ms < summary.earliest_ms {
                    summary.earliest_ms = entry.timestamp_ms;
                }
                if entry.timestamp_ms > summary.latest_ms {
                    summary.latest_ms = entry.timestamp_ms;
                }

                if self.config.bucket_secs > 0 {
                    let bucket_ms = (entry.timestamp_ms / 1000 / self.config.bucket_secs as i64)
                        * self.config.bucket_secs as i64
                        * 1000;
                    *summary.by_bucket.entry(bucket_ms).or_insert(0) += 1;
                }
            }

            summary.retained.push(entry.clone());
        }

        if summary.earliest_ms == i64::MAX {
            summary.earliest_ms = 0;
        }
        if summary.latest_ms == i64::MIN {
            summary.latest_ms = 0;
        }

        summary
    }

    /// Parse and aggregate an Apache CLF log file.
    pub fn aggregate_clf_file<P: AsRef<Path>>(&self, path: P) -> Result<LogSummary> {
        let file = std::fs::File::open(path.as_ref()).map_err(IoError::Io)?;
        let reader = BufReader::new(file);
        let parser = CommonLogParser::new();
        let entries: Vec<LogEntry> = parser
            .parse_reader(reader)
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        Ok(self.aggregate(&entries))
    }

    /// Parse and aggregate a JSON log file.
    pub fn aggregate_json_file<P: AsRef<Path>>(&self, path: P) -> Result<LogSummary> {
        let file = std::fs::File::open(path.as_ref()).map_err(IoError::Io)?;
        let reader = BufReader::new(file);
        let parser = JsonLogParser::new();
        let entries: Vec<LogEntry> = parser
            .parse_reader(reader)
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        Ok(self.aggregate(&entries))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Parsing utilities
// ──────────────────────────────────────────────────────────────────────────────

/// Split off the first whitespace-delimited token.
fn split_field(s: &str) -> (&str, &str) {
    if let Some(pos) = s.find(char::is_whitespace) {
        (&s[..pos], &s[pos..])
    } else {
        (s, "")
    }
}

/// Extract a quoted string starting at `s`; returns (inner, rest).
fn extract_quoted(s: &str) -> (&str, &str) {
    if !s.starts_with('"') {
        return split_field(s);
    }
    // Find closing unescaped quote
    let inner = &s[1..];
    let mut prev_backslash = false;
    for (i, c) in inner.char_indices() {
        if prev_backslash {
            prev_backslash = false;
            continue;
        }
        if c == '\\' {
            prev_backslash = true;
            continue;
        }
        if c == '"' {
            return (&inner[..i], &inner[i + 1..]);
        }
    }
    (inner, "")
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_iso8601_timestamp() {
        let ms = parse_timestamp("2024-03-23T14:22:47Z").expect("parse ISO 8601");
        assert!(ms > 0);
    }

    #[test]
    fn test_parse_unix_millis_timestamp() {
        let ms = parse_timestamp("1711234567890").expect("parse unix millis");
        assert_eq!(ms, 1711234567890);
    }

    #[test]
    fn test_parse_unix_secs_timestamp() {
        let ms = parse_timestamp("1711234567").expect("parse unix secs");
        assert_eq!(ms, 1711234567000);
    }

    #[test]
    fn test_clf_parser() {
        let line =
            r#"127.0.0.1 - frank [23/Mar/2024:14:22:47 +0000] "GET /index.html HTTP/1.1" 200 1234"#;
        let parser = CommonLogParser::new();
        let entry = parser.parse(line).expect("parse CLF line");
        assert_eq!(entry.fields.get("method").map(|s| s.as_str()), Some("GET"));
        assert_eq!(entry.fields.get("status").map(|s| s.as_str()), Some("200"));
    }

    #[test]
    fn test_clf_parser_500_level() {
        let line = r#"10.0.0.1 - - [23/Mar/2024:14:22:47 +0000] "POST /api HTTP/1.1" 500 0"#;
        let parser = CommonLogParser::new();
        let entry = parser.parse(line).expect("parse error CLF line");
        assert_eq!(entry.level, LogLevel::Error);
    }

    #[test]
    fn test_json_log_parser() {
        let line = r#"{"timestamp":"2024-03-23T14:22:47Z","level":"ERROR","message":"disk full","host":"srv01"}"#;
        let parser = JsonLogParser::new();
        let entry = parser.parse(line).expect("parse JSON log");
        assert_eq!(entry.level, LogLevel::Error);
        assert_eq!(entry.message, "disk full");
        assert_eq!(entry.host.as_deref(), Some("srv01"));
    }

    #[test]
    fn test_syslog_parser_rfc5424() {
        let line = "<34>1 2024-03-23T14:22:47Z myhost myapp 1234 ID47 - An application event";
        let parser = SyslogParser::new();
        let entry = parser.parse(line).expect("parse syslog");
        assert_eq!(entry.host.as_deref(), Some("myhost"));
        assert_eq!(entry.app.as_deref(), Some("myapp"));
        assert_eq!(entry.pid, Some(1234));
        assert_eq!(entry.message, "An application event");
    }

    #[test]
    fn test_log_aggregator_level_filter() {
        let entries = vec![
            {
                let mut e = LogEntry::new_raw("a");
                e.level = LogLevel::Debug;
                e.timestamp_ms = 1000;
                e
            },
            {
                let mut e = LogEntry::new_raw("b");
                e.level = LogLevel::Error;
                e.timestamp_ms = 2000;
                e
            },
        ];
        let config = AggregationConfig {
            min_level: Some(LogLevel::Warning),
            ..Default::default()
        };
        let agg = LogAggregator::new(config);
        let summary = agg.aggregate(&entries);
        assert_eq!(summary.total, 1);
        assert_eq!(summary.retained[0].level, LogLevel::Error);
    }

    #[test]
    fn test_log_aggregator_bucket() {
        let config = AggregationConfig {
            bucket_secs: 60,
            ..Default::default()
        };
        let agg = LogAggregator::new(config);
        let entries: Vec<LogEntry> = (0..5)
            .map(|i| {
                let mut e = LogEntry::new_raw(format!("msg {i}"));
                e.level = LogLevel::Info;
                e.timestamp_ms = 1_700_000_000_000 + i * 1000;
                e
            })
            .collect();
        let summary = agg.aggregate(&entries);
        assert_eq!(summary.total, 5);
        // All 5 timestamps fall in the same 60-second bucket
        assert_eq!(summary.by_bucket.values().sum::<usize>(), 5);
    }
}
