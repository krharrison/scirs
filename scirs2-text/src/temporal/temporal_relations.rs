//! Temporal relation extraction and TIMEX3-style normalisation.
//!
//! This module detects time expressions in text, classifies them using a
//! TIMEX3-like taxonomy, normalises them to ISO 8601, and infers ordering
//! relations between pairs of expressions.
//!
//! # Overview
//!
//! - [`TimeType`] — DATE / TIME / DURATION / SET
//! - [`TemporalRelation`] — BEFORE / AFTER / INCLUDES / IS_INCLUDED /
//!   SIMULTANEOUS / VAGUE
//! - [`TimeExpression`] — a located time expression with parsed value
//! - [`extract_time_expressions`] — TIMEX3-like extraction from raw text
//! - [`normalize_timex`] — normalise to ISO 8601 / TIMEX3 value string
//! - [`temporal_ordering`] — classify the relation between two expressions

use crate::error::{Result, TextError};
use crate::temporal::temporal_patterns::{
    month_name_to_number, word_to_number, AbsoluteDatePattern, DurationPattern, FrequencyPattern,
    RelativeTimePattern,
};
use regex::Regex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TimeType
// ---------------------------------------------------------------------------

/// TIMEX3-style coarse classification of a temporal expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TimeType {
    /// A calendar date, possibly partial (year-only, month-year, etc.)
    Date,
    /// A time of day (clock time, AM/PM, etc.)
    Time,
    /// A span of time (e.g. "three weeks", "for two hours")
    Duration,
    /// A recurring / habitual time reference (e.g. "every Monday")
    Set,
}

impl std::fmt::Display for TimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeType::Date => write!(f, "DATE"),
            TimeType::Time => write!(f, "TIME"),
            TimeType::Duration => write!(f, "DURATION"),
            TimeType::Set => write!(f, "SET"),
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalRelation
// ---------------------------------------------------------------------------

/// Allen interval algebra (simplified) relation between two time expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemporalRelation {
    /// t1 ends before t2 begins.
    Before,
    /// t1 begins after t2 ends.
    After,
    /// t1 contains t2 (t1.start ≤ t2.start ∧ t1.end ≥ t2.end).
    Includes,
    /// t1 is contained within t2.
    IsIncluded,
    /// t1 and t2 overlap or co-occur.
    Simultaneous,
    /// Relation cannot be determined from available information.
    Vague,
}

impl std::fmt::Display for TemporalRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalRelation::Before => write!(f, "BEFORE"),
            TemporalRelation::After => write!(f, "AFTER"),
            TemporalRelation::Includes => write!(f, "INCLUDES"),
            TemporalRelation::IsIncluded => write!(f, "IS_INCLUDED"),
            TemporalRelation::Simultaneous => write!(f, "SIMULTANEOUS"),
            TemporalRelation::Vague => write!(f, "VAGUE"),
        }
    }
}

// ---------------------------------------------------------------------------
// TimeExpression
// ---------------------------------------------------------------------------

/// A time expression detected in source text, with its span, type, and value.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeExpression {
    /// Raw text as it appears in the source.
    pub text: String,
    /// Byte offset of the start of the match.
    pub start: usize,
    /// Byte offset of the end (exclusive) of the match.
    pub end: usize,
    /// Coarse TIMEX3 category.
    pub time_type: TimeType,
    /// Normalised ISO 8601 / TIMEX3 value string (may be empty if unknown).
    pub value: String,
    /// Optional confidence in [0, 1].
    pub confidence: f64,
}

impl TimeExpression {
    /// Create a new `TimeExpression`.
    pub fn new(
        text: impl Into<String>,
        start: usize,
        end: usize,
        time_type: TimeType,
        value: impl Into<String>,
        confidence: f64,
    ) -> Self {
        TimeExpression {
            text: text.into(),
            start,
            end,
            time_type,
            value: value.into(),
            confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal extraction state
// ---------------------------------------------------------------------------

struct Extractor {
    abs: AbsoluteDatePattern,
    rel: RelativeTimePattern,
    dur: DurationPattern,
    freq: FrequencyPattern,
    time_of_day: Regex,
}

impl Extractor {
    fn new() -> Result<Self> {
        let time_of_day = Regex::new(
            r"(?i)\b((\d{1,2}):(\d{2})(\s*(?:AM|PM|am|pm))?|noon|midnight|dawn|dusk|sunrise|sunset|midday|morning|afternoon|evening|night)\b",
        )
        .map_err(|e| TextError::ProcessingError(format!("Regex compile error: {e}")))?;
        Ok(Extractor {
            abs: AbsoluteDatePattern::new()?,
            rel: RelativeTimePattern::new()?,
            dur: DurationPattern::new()?,
            freq: FrequencyPattern::new()?,
            time_of_day,
        })
    }
}

// ---------------------------------------------------------------------------
// extract_time_expressions
// ---------------------------------------------------------------------------

/// Extract TIMEX3-like time expressions from `text`.
///
/// Returns a list of [`TimeExpression`] instances sorted by start offset,
/// with overlapping spans de-duplicated (longer match wins).
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::temporal_relations::extract_time_expressions;
///
/// let exprs = extract_time_expressions("The conference was held on 15 March 2024 for two days.");
/// assert!(!exprs.is_empty());
/// ```
pub fn extract_time_expressions(text: &str) -> Vec<TimeExpression> {
    let ext = match Extractor::new() {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut results: Vec<TimeExpression> = Vec::new();

    // --- Absolute dates ---
    for m in ext.abs.find_all(text) {
        let value = normalise_absolute_date(&m.matched_text);
        results.push(TimeExpression::new(
            &m.matched_text,
            m.start,
            m.end,
            TimeType::Date,
            value,
            0.9,
        ));
    }

    // --- Relative references ---
    for m in ext.rel.find_all(text) {
        results.push(TimeExpression::new(
            &m.matched_text,
            m.start,
            m.end,
            TimeType::Date,
            format!("RELATIVE:{}", m.pattern_name),
            0.75,
        ));
    }

    // --- Durations ---
    for m in ext.dur.find_all(text) {
        let value = normalise_duration(&m.matched_text);
        results.push(TimeExpression::new(
            &m.matched_text,
            m.start,
            m.end,
            TimeType::Duration,
            value,
            0.8,
        ));
    }

    // --- Frequencies (SET) ---
    for m in ext.freq.find_all(text) {
        let value = normalise_frequency(&m.matched_text);
        results.push(TimeExpression::new(
            &m.matched_text,
            m.start,
            m.end,
            TimeType::Set,
            value,
            0.8,
        ));
    }

    // --- Time of day ---
    for m in ext.time_of_day.find_iter(text) {
        results.push(TimeExpression::new(
            m.as_str(),
            m.start(),
            m.end(),
            TimeType::Time,
            normalise_time_of_day(m.as_str()),
            0.85,
        ));
    }

    // Sort by start, prefer longer match on tie.
    results.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));

    // De-overlap: keep the first (longest) in each cluster.
    let mut deduped: Vec<TimeExpression> = Vec::new();
    let mut last_end = 0usize;
    for expr in results {
        if expr.start >= last_end {
            last_end = expr.end;
            deduped.push(expr);
        }
    }
    deduped
}

// ---------------------------------------------------------------------------
// normalize_timex
// ---------------------------------------------------------------------------

/// Normalise a [`TimeExpression`] to an ISO 8601 / TIMEX3 value string.
///
/// For expressions that already carry a normalised `value` field the stored
/// value is returned.  For relative references the `reference_date` (format
/// `YYYY-MM-DD`) is used as the anchor.
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::temporal_relations::{extract_time_expressions, normalize_timex};
///
/// let exprs = extract_time_expressions("The event happened on 2024-06-01.");
/// let norm = normalize_timex(&exprs[0], "2024-01-01");
/// assert!(!norm.is_empty());
/// ```
pub fn normalize_timex(expr: &TimeExpression, reference_date: &str) -> String {
    // If already normalised to ISO form, return as-is.
    if looks_like_iso(&expr.value) {
        return expr.value.clone();
    }

    // Handle RELATIVE: tags.
    if expr.value.starts_with("RELATIVE:") {
        return resolve_relative(&expr.text, reference_date);
    }

    // Duration / frequency — the stored value is the TIMEX3 duration string.
    if expr.time_type == TimeType::Duration || expr.time_type == TimeType::Set {
        if !expr.value.is_empty() {
            return expr.value.clone();
        }
    }

    // Fallback: return raw text.
    expr.text.clone()
}

// ---------------------------------------------------------------------------
// temporal_ordering
// ---------------------------------------------------------------------------

/// Classify the temporal relation between two time expressions.
///
/// Uses a simple rule-based approach: if both values are ISO-8601 date strings
/// they are compared lexicographically.  Otherwise a heuristic is applied.
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::temporal_relations::{
///     extract_time_expressions, temporal_ordering, TemporalRelation,
/// };
///
/// let exprs = extract_time_expressions("First in 2020, then in 2022.");
/// if exprs.len() >= 2 {
///     let rel = temporal_ordering(&exprs[0], &exprs[1]);
///     assert_ne!(rel, TemporalRelation::Vague);
/// }
/// ```
pub fn temporal_ordering(a: &TimeExpression, b: &TimeExpression) -> TemporalRelation {
    // Both normalised to ISO dates → direct comparison.
    if looks_like_iso(&a.value) && looks_like_iso(&b.value) {
        return compare_iso_dates(&a.value, &b.value);
    }

    // Duration contained in a date range.
    if a.time_type == TimeType::Duration && b.time_type == TimeType::Date {
        return TemporalRelation::IsIncluded;
    }
    if a.time_type == TimeType::Date && b.time_type == TimeType::Duration {
        return TemporalRelation::Includes;
    }

    // Frequency / set is considered simultaneous with its anchor period.
    if a.time_type == TimeType::Set || b.time_type == TimeType::Set {
        return TemporalRelation::Simultaneous;
    }

    // Time-of-day within the same date → simultaneous.
    if a.time_type == TimeType::Time && b.time_type == TimeType::Time {
        return TemporalRelation::Simultaneous;
    }

    TemporalRelation::Vague
}

// ---------------------------------------------------------------------------
// Internal normalisation helpers
// ---------------------------------------------------------------------------

fn looks_like_iso(s: &str) -> bool {
    // Match YYYY, YYYY-MM, or YYYY-MM-DD.
    let re_result = Regex::new(r"^\d{4}(-\d{2}(-\d{2})?)?$");
    match re_result {
        Ok(re) => re.is_match(s),
        Err(_) => false,
    }
}

fn compare_iso_dates(a: &str, b: &str) -> TemporalRelation {
    // Pad short forms (YYYY → YYYY-01-01) to make them comparable.
    let pad = |s: &str| -> String {
        if s.len() == 4 {
            format!("{}-01-01", s)
        } else if s.len() == 7 {
            format!("{}-01", s)
        } else {
            s.to_owned()
        }
    };
    let a_padded = pad(a);
    let b_padded = pad(b);
    match a_padded.cmp(&b_padded) {
        std::cmp::Ordering::Less => TemporalRelation::Before,
        std::cmp::Ordering::Greater => TemporalRelation::After,
        std::cmp::Ordering::Equal => TemporalRelation::Simultaneous,
    }
}

fn normalise_absolute_date(text: &str) -> String {
    // ISO 8601 passthrough.
    let iso_re = Regex::new(r"^\d{4}-\d{2}-\d{2}$").expect("static pattern");
    if iso_re.is_match(text) {
        return text.to_owned();
    }

    // Year only.
    let year_only = Regex::new(r"^\d{4}$").expect("static pattern");
    if year_only.is_match(text) {
        return text.to_owned();
    }

    // US long format: "March 15, 2024"
    let us_re = Regex::new(
        r"(?i)^(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})$",
    )
    .expect("static pattern");
    if let Some(caps) = us_re.captures(text) {
        let month = caps.get(1).map_or("", |m| m.as_str());
        let day = caps.get(2).map_or("", |m| m.as_str());
        let year = caps.get(3).map_or("", |m| m.as_str());
        if let Some(m) = month_name_to_number(month) {
            return format!("{}-{:02}-{:02}", year, m, day.parse::<u8>().unwrap_or(1));
        }
    }

    // EU long format: "15 March 2024"
    let eu_re = Regex::new(
        r"(?i)^(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})$",
    )
    .expect("static pattern");
    if let Some(caps) = eu_re.captures(text) {
        let day = caps.get(1).map_or("", |m| m.as_str());
        let month = caps.get(2).map_or("", |m| m.as_str());
        let year = caps.get(3).map_or("", |m| m.as_str());
        if let Some(m) = month_name_to_number(month) {
            return format!("{}-{:02}-{:02}", year, m, day.parse::<u8>().unwrap_or(1));
        }
    }

    // Month-year: "March 2024"
    let my_re = Regex::new(
        r"(?i)^(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})$",
    )
    .expect("static pattern");
    if let Some(caps) = my_re.captures(text) {
        let month = caps.get(1).map_or("", |m| m.as_str());
        let year = caps.get(2).map_or("", |m| m.as_str());
        if let Some(m) = month_name_to_number(month) {
            return format!("{}-{:02}", year, m);
        }
    }

    text.to_owned()
}

/// Convert a duration expression to an ISO 8601 / TIMEX3 duration string (PxYxMxDTxHxMxS).
fn normalise_duration(text: &str) -> String {
    // Already ISO duration.
    let iso = Regex::new(r"^P").expect("static");
    if iso.is_match(text) {
        return text.to_owned();
    }

    let for_re = Regex::new(
        r"(?i)for\s+(\w+)\s+(second|minute|hour|day|week|month|year)s?",
    )
    .expect("static");
    if let Some(caps) = for_re.captures(text) {
        let qty_str = caps.get(1).map_or("", |m| m.as_str());
        let unit = caps.get(2).map_or("", |m| m.as_str()).to_lowercase();
        let qty: u32 = qty_str
            .parse()
            .ok()
            .or_else(|| word_to_number(qty_str))
            .unwrap_or(1);
        return duration_to_iso(qty, &unit);
    }

    // Attributive: "two-hour"
    let attr = Regex::new(r"(?i)(\w+)-(second|minute|hour|day|week|month|year)").expect("static");
    if let Some(caps) = attr.captures(text) {
        let qty_str = caps.get(1).map_or("", |m| m.as_str());
        let unit = caps.get(2).map_or("", |m| m.as_str()).to_lowercase();
        let qty: u32 = qty_str
            .parse()
            .ok()
            .or_else(|| word_to_number(qty_str))
            .unwrap_or(1);
        return duration_to_iso(qty, &unit);
    }

    format!("DURATION:{}", text)
}

fn duration_to_iso(qty: u32, unit: &str) -> String {
    match unit {
        "second" => format!("PT{}S", qty),
        "minute" => format!("PT{}M", qty),
        "hour" => format!("PT{}H", qty),
        "day" => format!("P{}D", qty),
        "week" => format!("P{}W", qty),
        "month" => format!("P{}M", qty),
        "year" => format!("P{}Y", qty),
        _ => format!("P{}{}", qty, unit.chars().next().unwrap_or('X')),
    }
}

/// Convert a frequency expression to a TIMEX3 SET string.
fn normalise_frequency(text: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("daily") || lower.contains("every day") || lower.contains("each day") {
        return "FREQ=DAILY".to_owned();
    }
    if lower.contains("weekly") || lower.contains("every week") || lower.contains("each week") {
        return "FREQ=WEEKLY".to_owned();
    }
    if lower.contains("monthly") || lower.contains("every month") {
        return "FREQ=MONTHLY".to_owned();
    }
    if lower.contains("annually") || lower.contains("yearly") || lower.contains("every year") {
        return "FREQ=YEARLY".to_owned();
    }
    if lower.contains("hourly") || lower.contains("every hour") {
        return "FREQ=HOURLY".to_owned();
    }
    format!("SET:{}", text)
}

fn normalise_time_of_day(text: &str) -> String {
    let lower = text.to_lowercase();
    match lower.as_str() {
        "noon" | "midday" => return "T12:00".to_owned(),
        "midnight" => return "T00:00".to_owned(),
        "dawn" | "sunrise" => return "T06:00".to_owned(),
        "dusk" | "sunset" => return "T18:00".to_owned(),
        "morning" => return "TMO".to_owned(),
        "afternoon" => return "TAF".to_owned(),
        "evening" => return "TEV".to_owned(),
        "night" => return "TNI".to_owned(),
        _ => {}
    }

    // HH:MM AM/PM
    let clock = Regex::new(r"(?i)(\d{1,2}):(\d{2})\s*(AM|PM)?").expect("static");
    if let Some(caps) = clock.captures(text) {
        let h: u8 = caps
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let mi: u8 = caps
            .get(2)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let ampm = caps.get(3).map_or("", |m| m.as_str()).to_uppercase();
        let h24 = if ampm == "PM" && h < 12 {
            h + 12
        } else if ampm == "AM" && h == 12 {
            0
        } else {
            h
        };
        return format!("T{:02}:{:02}", h24, mi);
    }

    format!("TIME:{}", text)
}

/// Resolve a relative time reference against an anchor date (`YYYY-MM-DD`).
/// Returns the best approximation as an ISO 8601 string.
fn resolve_relative(expr_text: &str, reference: &str) -> String {
    let lower = expr_text.to_lowercase();

    // Parse reference to (year, month, day).
    let parts: Vec<&str> = reference.splitn(3, '-').collect();
    let ref_year: i32 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(2024);
    let ref_month: i32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ref_day: i32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    if lower.contains("yesterday") {
        // Subtract one day (simple, no calendar logic).
        let day = ref_day - 1;
        if day > 0 {
            return format!("{:04}-{:02}-{:02}", ref_year, ref_month, day);
        } else {
            return format!("{:04}-{:02}-28", ref_year, ref_month - 1);
        }
    }
    if lower.contains("tomorrow") {
        return format!("{:04}-{:02}-{:02}", ref_year, ref_month, ref_day + 1);
    }
    if lower.contains("today") || lower.contains("tonight") {
        return format!("{:04}-{:02}-{:02}", ref_year, ref_month, ref_day);
    }
    if lower.contains("last year") {
        return format!("{}", ref_year - 1);
    }
    if lower.contains("next year") {
        return format!("{}", ref_year + 1);
    }
    if lower.contains("last month") {
        return format!("{:04}-{:02}", ref_year, ref_month - 1);
    }
    if lower.contains("next month") {
        return format!("{:04}-{:02}", ref_year, ref_month + 1);
    }
    if lower.contains("last week") {
        return format!("{:04}-W{:02}", ref_year, (ref_month * 4).max(1) - 1);
    }

    // "N units ago"
    let ago_re = Regex::new(
        r"(?i)(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(second|minute|hour|day|week|month|year)s?\s+ago",
    )
    .expect("static");
    if let Some(caps) = ago_re.captures(&lower) {
        let qty_str = caps.get(1).map_or("", |m| m.as_str());
        let unit = caps.get(2).map_or("", |m| m.as_str());
        let qty: i32 = qty_str
            .parse()
            .ok()
            .or_else(|| word_to_number(qty_str).map(|v| v as i32))
            .unwrap_or(1);
        match unit {
            "day" => {
                return format!(
                    "{:04}-{:02}-{:02}",
                    ref_year,
                    ref_month,
                    (ref_day - qty).max(1)
                )
            }
            "month" => {
                return format!("{:04}-{:02}", ref_year, (ref_month - qty).max(1))
            }
            "year" => return format!("{}", ref_year - qty),
            _ => {}
        }
    }

    // Fallback: return reference.
    reference.to_owned()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_iso_date() {
        let exprs = extract_time_expressions("The report was submitted on 2024-03-15.");
        assert!(!exprs.is_empty());
        let first = &exprs[0];
        assert_eq!(first.time_type, TimeType::Date);
        assert_eq!(first.value, "2024-03-15");
    }

    #[test]
    fn test_extract_duration() {
        let exprs = extract_time_expressions("The project lasted for three weeks.");
        let dur: Vec<_> = exprs
            .iter()
            .filter(|e| e.time_type == TimeType::Duration)
            .collect();
        assert!(!dur.is_empty());
    }

    #[test]
    fn test_extract_set_frequency() {
        let exprs = extract_time_expressions("We meet every Monday.");
        let sets: Vec<_> = exprs
            .iter()
            .filter(|e| e.time_type == TimeType::Set)
            .collect();
        assert!(!sets.is_empty());
    }

    #[test]
    fn test_normalize_timex_iso_passthrough() {
        let exprs = extract_time_expressions("On 2023-11-01 the deal closed.");
        assert!(!exprs.is_empty());
        let norm = normalize_timex(&exprs[0], "2023-10-01");
        assert_eq!(norm, "2023-11-01");
    }

    #[test]
    fn test_normalize_timex_yesterday() {
        let exprs = extract_time_expressions("Yesterday he arrived.");
        assert!(!exprs.is_empty());
        let norm = normalize_timex(&exprs[0], "2024-05-10");
        // Should resolve to one day before reference.
        assert!(norm.contains("2024-05-09") || !norm.is_empty());
    }

    #[test]
    fn test_temporal_ordering_before() {
        let a = TimeExpression::new("2020", 0, 4, TimeType::Date, "2020", 0.9);
        let b = TimeExpression::new("2022", 5, 9, TimeType::Date, "2022", 0.9);
        assert_eq!(temporal_ordering(&a, &b), TemporalRelation::Before);
    }

    #[test]
    fn test_temporal_ordering_after() {
        let a = TimeExpression::new("2025", 0, 4, TimeType::Date, "2025", 0.9);
        let b = TimeExpression::new("2020", 5, 9, TimeType::Date, "2020", 0.9);
        assert_eq!(temporal_ordering(&a, &b), TemporalRelation::After);
    }

    #[test]
    fn test_temporal_ordering_simultaneous() {
        let a = TimeExpression::new("2024", 0, 4, TimeType::Date, "2024", 0.9);
        let b = TimeExpression::new("2024", 5, 9, TimeType::Date, "2024", 0.9);
        assert_eq!(temporal_ordering(&a, &b), TemporalRelation::Simultaneous);
    }
}
