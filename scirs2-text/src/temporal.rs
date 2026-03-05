//! Temporal expression extraction and normalisation.
//!
//! This module provides rule-based detection and TimeML-style normalisation
//! of temporal expressions in English text.  No trained models are required.
//!
//! # Overview
//!
//! - [`TimeType`] — coarse classification of a temporal expression
//! - [`TemporalValue`] — rich, structured representation of the temporal content
//! - [`TemporalExpression`] — a located temporal expression in the source text
//! - [`extract_temporal`] — top-level extraction entry point
//! - [`normalize_temporal`] — map an expression to a TimeML-style string
//! - [`temporal_relation`] — infer the relative ordering of two expressions
//! - [`TemporalRelation`] — the possible orderings between two expressions
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::temporal::{extract_temporal, normalize_temporal, TemporalRelation, temporal_relation};
//!
//! let exprs = extract_temporal("The meeting is scheduled for next Monday.");
//! assert!(!exprs.is_empty());
//!
//! let reference = "2024-01-15"; // Monday
//! let norm = normalize_temporal(&exprs[0], reference);
//! assert!(!norm.is_empty());
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TimeType
// ---------------------------------------------------------------------------

/// Coarse-grained classification of a temporal expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TimeType {
    /// A fully or partially specified calendar date (e.g. "January 15, 2024").
    Date,
    /// A time-of-day expression (e.g. "3:00 PM", "noon").
    Time,
    /// A length of time (e.g. "three weeks", "two hours").
    Duration,
    /// A recurring / habitual time expression (e.g. "every Monday", "annually").
    Set,
    /// An interval between two time points (e.g. "from Monday to Friday").
    TimeInterval,
}

impl std::fmt::Display for TimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Duration => write!(f, "DURATION"),
            Self::Set => write!(f, "SET"),
            Self::TimeInterval => write!(f, "TIME_INTERVAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalValue
// ---------------------------------------------------------------------------

/// Structured temporal content.
#[derive(Debug, Clone)]
pub enum TemporalValue {
    /// A time point anchored to the calendar, e.g. `"2024-01-15"`.
    Absolute(AbsoluteTime),
    /// A time point expressed relative to some reference, e.g. `"-P1D"` (yesterday).
    Relative(RelativeTime),
    /// An interval between two time points.
    Interval {
        /// The start boundary (ISO-8601-like string).
        start: String,
        /// The end boundary (ISO-8601-like string).
        end: String,
    },
    /// Unknown / unparseable value.
    Unknown,
}

/// Fully or partially specified calendar time.
#[derive(Debug, Clone)]
pub struct AbsoluteTime {
    /// Year (e.g. 2024).
    pub year: Option<i32>,
    /// Month 1-12.
    pub month: Option<u8>,
    /// Day 1-31.
    pub day: Option<u8>,
    /// Hour 0-23.
    pub hour: Option<u8>,
    /// Minute 0-59.
    pub minute: Option<u8>,
}

impl AbsoluteTime {
    /// Format as an ISO-8601 partial date/time string.
    pub fn to_iso(&self) -> String {
        match (self.year, self.month, self.day) {
            (Some(y), Some(m), Some(d)) => {
                let base = format!("{:04}-{:02}-{:02}", y, m, d);
                match (self.hour, self.minute) {
                    (Some(h), Some(mi)) => format!("{}T{:02}:{:02}", base, h, mi),
                    (Some(h), None) => format!("{}T{:02}", base, h),
                    _ => base,
                }
            }
            (Some(y), Some(m), None) => format!("{:04}-{:02}", y, m),
            (Some(y), None, None) => format!("{:04}", y),
            _ => "XXXX".to_string(),
        }
    }
}

/// A time expressed as an offset from a reference point.
#[derive(Debug, Clone)]
pub struct RelativeTime {
    /// ISO-8601 duration string (may be negative, e.g. `"-P1D"`).
    pub offset: String,
    /// The anchor keyword (e.g. `"PRESENT_REF"`, `"PAST_REF"`).
    pub anchor: String,
}

// ---------------------------------------------------------------------------
// TemporalExpression
// ---------------------------------------------------------------------------

/// A temporal expression located in a source document.
#[derive(Debug, Clone)]
pub struct TemporalExpression {
    /// Surface text of the expression.
    pub text: String,
    /// Character span `(start, end)` in the source document.
    pub span: (usize, usize),
    /// Structured temporal content.
    pub value: TemporalValue,
    /// Coarse classification.
    pub type_: TimeType,
}

// ---------------------------------------------------------------------------
// Helpers — pattern tables
// ---------------------------------------------------------------------------

const MONTHS: &[(&str, u8)] = &[
    ("january", 1),
    ("february", 2),
    ("march", 3),
    ("april", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
    ("jan", 1),
    ("feb", 2),
    ("mar", 3),
    ("apr", 4),
    ("jun", 6),
    ("jul", 7),
    ("aug", 8),
    ("sep", 9),
    ("oct", 10),
    ("nov", 11),
    ("dec", 12),
];

const DAYS_OF_WEEK: &[&str] = &[
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
];

const DURATION_UNITS: &[(&str, &str)] = &[
    ("second", "S"),
    ("seconds", "S"),
    ("minute", "M"),
    ("minutes", "M"),
    ("hour", "H"),
    ("hours", "H"),
    ("day", "D"),
    ("days", "D"),
    ("week", "W"),
    ("weeks", "W"),
    ("month", "M"),   // ambiguous with minutes but month is uppercase P
    ("months", "M"),
    ("year", "Y"),
    ("years", "Y"),
];

const TIME_OF_DAY: &[(&str, u8)] = &[
    ("midnight", 0),
    ("noon", 12),
    ("morning", 8),
    ("afternoon", 14),
    ("evening", 18),
    ("night", 21),
];

/// Map a month abbreviation/name to its 1-based number.
fn month_num(word: &str) -> Option<u8> {
    let lower = word.to_lowercase();
    MONTHS.iter().find(|(m, _)| *m == lower.as_str()).map(|(_, n)| *n)
}

/// Map a duration unit word to its ISO-8601 unit letter.
fn duration_unit(word: &str) -> Option<&'static str> {
    let lower = word.to_lowercase();
    DURATION_UNITS
        .iter()
        .find(|(u, _)| *u == lower.as_str())
        .map(|(_, l)| *l)
}

// ---------------------------------------------------------------------------
// Tokeniser (byte-offset aware)
// ---------------------------------------------------------------------------

fn tokenise(text: &str) -> Vec<(usize, usize, String)> {
    let mut tokens: Vec<(usize, usize, String)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == ':' || c == '-' || c == '/' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            tokens.push((s, i, text[s..i].to_string()));
        }
    }
    if let Some(s) = start {
        tokens.push((s, text.len(), text[s..].to_string()));
    }
    tokens
}

// ---------------------------------------------------------------------------
// Pattern matchers — each returns Option<(consumed_count, TemporalExpression)>
// ---------------------------------------------------------------------------

/// Try to match "yesterday | today | tomorrow" at token index `i`.
fn match_deictic(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    let (ts, te, word) = &tokens[i];
    let lower = word.to_lowercase();
    let (offset_days, anchor) = match lower.as_str() {
        "yesterday" => ("-P1D", "PRESENT_REF"),
        "today" => ("P0D", "PRESENT_REF"),
        "tomorrow" => ("P1D", "PRESENT_REF"),
        _ => return None,
    };
    Some((
        1,
        TemporalExpression {
            text: word.clone(),
            span: (*ts, *te),
            value: TemporalValue::Relative(RelativeTime {
                offset: offset_days.to_string(),
                anchor: anchor.to_string(),
            }),
            type_: TimeType::Date,
        },
    ))
}

/// Try to match "last | next | this <weekday | month | year | unit>" at `i`.
fn match_relative(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    if i + 1 >= tokens.len() {
        return None;
    }
    let (ts, _te, word) = &tokens[i];
    let anchor_lower = word.to_lowercase();
    let dir: i32 = match anchor_lower.as_str() {
        "last" | "previous" => -1,
        "next" | "coming" => 1,
        "this" => 0,
        _ => return None,
    };

    let (ns, ne, next_word) = &tokens[i + 1];
    let next_lower = next_word.to_lowercase();
    let surface = format!("{} {}", word, next_word);
    let span = (*ts, *ne);

    // Weekday
    if DAYS_OF_WEEK.contains(&next_lower.as_str()) {
        let sign = if dir >= 0 { "" } else { "-" };
        return Some((
            2,
            TemporalExpression {
                text: surface,
                span,
                value: TemporalValue::Relative(RelativeTime {
                    offset: format!("{}P1W", sign),
                    anchor: "PRESENT_REF".to_string(),
                }),
                type_: TimeType::Date,
            },
        ));
    }

    // Month name alone → relative month
    if month_num(next_word).is_some() {
        let sign = if dir >= 0 { "" } else { "-" };
        return Some((
            2,
            TemporalExpression {
                text: surface,
                span,
                value: TemporalValue::Relative(RelativeTime {
                    offset: format!("{}P1M", sign),
                    anchor: "PRESENT_REF".to_string(),
                }),
                type_: TimeType::Date,
            },
        ));
    }

    // Duration units: week / month / year etc.
    if let Some(unit) = duration_unit(next_word) {
        let sign = if dir >= 0 { "" } else { "-" };
        let prefix = if unit == "H" || unit == "M" || unit == "S" {
            format!("{}PT1{}", sign, unit)
        } else {
            format!("{}P1{}", sign, unit)
        };
        return Some((
            2,
            TemporalExpression {
                text: surface,
                span,
                value: TemporalValue::Relative(RelativeTime {
                    offset: prefix,
                    anchor: "PRESENT_REF".to_string(),
                }),
                type_: TimeType::Duration,
            },
        ));
    }

    None
}

/// Try to match "<N> <unit> ago" or "<N> <unit>" at `i`.
fn match_duration_ago(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    let (ts, _te, word) = &tokens[i];
    if !word.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let n: u64 = word.parse().ok()?;
    if i + 1 >= tokens.len() {
        return None;
    }
    let unit_lower = tokens[i + 1].2.to_lowercase();
    let unit_iso = duration_unit(&unit_lower)?;

    let (consumed, suffix, span_e, surface) = if i + 2 < tokens.len()
        && tokens[i + 2].2.to_lowercase() == "ago"
    {
        let span_e = tokens[i + 2].1;
        let surf = format!("{} {} ago", word, tokens[i + 1].2);
        (3usize, "ago", span_e, surf)
    } else {
        let span_e = tokens[i + 1].1;
        let surf = format!("{} {}", word, tokens[i + 1].2);
        (2usize, "", span_e, surf)
    };

    let sign = if suffix == "ago" { "-" } else { "" };
    // Determine prefix: date-level (P) or time-level (PT)
    let iso_offset = if unit_iso == "H" || unit_iso == "S"
        || (unit_iso == "M" && !unit_lower.starts_with("mo"))
    {
        format!("{}PT{}{}", sign, n, unit_iso)
    } else {
        format!("{}P{}{}", sign, n, unit_iso)
    };

    Some((
        consumed,
        TemporalExpression {
            text: surface,
            span: (*ts, span_e),
            value: TemporalValue::Relative(RelativeTime {
                offset: iso_offset,
                anchor: "PRESENT_REF".to_string(),
            }),
            type_: TimeType::Duration,
        },
    ))
}

/// Try to match "in <year>" at `i`.
fn match_in_year(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    let (ts, _te, word) = &tokens[i];
    if word.to_lowercase() != "in" {
        return None;
    }
    if i + 1 >= tokens.len() {
        return None;
    }
    let (_, ne, year_str) = &tokens[i + 1];
    if year_str.len() == 4 && year_str.chars().all(|c| c.is_ascii_digit()) {
        let y: i32 = year_str.parse().ok()?;
        let surface = format!("in {}", year_str);
        return Some((
            2,
            TemporalExpression {
                text: surface,
                span: (*ts, *ne),
                value: TemporalValue::Absolute(AbsoluteTime {
                    year: Some(y),
                    month: None,
                    day: None,
                    hour: None,
                    minute: None,
                }),
                type_: TimeType::Date,
            },
        ));
    }
    None
}

/// Try to match "Month <Day>, <Year>" or "Month <Day>" at `i`.
fn match_full_date(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    let (ts, _te, word) = &tokens[i];
    let m = month_num(word)?;

    // Need at least a day number next
    if i + 1 >= tokens.len() {
        return None;
    }
    let day_str = tokens[i + 1].2.trim_end_matches(',');
    let day: u8 = day_str.parse().ok()?;
    if day == 0 || day > 31 {
        return None;
    }

    // Optional year
    let (consumed, year, span_e, surface) = if i + 2 < tokens.len() {
        let y_str = &tokens[i + 2].2;
        if y_str.len() == 4 && y_str.chars().all(|c| c.is_ascii_digit()) {
            let y: i32 = y_str.parse().ok()?;
            let surf = format!("{} {}, {}", word, tokens[i + 1].2, y_str);
            (3usize, Some(y), tokens[i + 2].1, surf)
        } else {
            let surf = format!("{} {}", word, tokens[i + 1].2);
            (2usize, None, tokens[i + 1].1, surf)
        }
    } else {
        let surf = format!("{} {}", word, tokens[i + 1].2);
        (2usize, None, tokens[i + 1].1, surf)
    };

    Some((
        consumed,
        TemporalExpression {
            text: surface,
            span: (*ts, span_e),
            value: TemporalValue::Absolute(AbsoluteTime {
                year,
                month: Some(m),
                day: Some(day),
                hour: None,
                minute: None,
            }),
            type_: TimeType::Date,
        },
    ))
}

/// Try to match "HH:MM [AM|PM]" at `i`.
fn match_clock_time(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    let (ts, te, word) = &tokens[i];
    // Accept HH:MM or H:MM
    if !word.contains(':') {
        return None;
    }
    let parts: Vec<&str> = word.splitn(2, ':').collect();
    if parts.len() != 2 {
        return None;
    }
    let h: u8 = parts[0].parse().ok()?;
    let m: u8 = parts[1].trim_end_matches(|c: char| !c.is_ascii_digit()).parse().ok()?;

    // Look for optional AM/PM
    let (consumed, hour_24, span_e, surface) = if i + 1 < tokens.len() {
        let ampm = tokens[i + 1].2.to_uppercase();
        if ampm == "AM" || ampm == "PM" {
            let h24 = if ampm == "PM" && h < 12 { h + 12 } else if ampm == "AM" && h == 12 { 0 } else { h };
            let surf = format!("{} {}", word, tokens[i + 1].2);
            (2usize, h24, tokens[i + 1].1, surf)
        } else {
            (1usize, h, *te, word.clone())
        }
    } else {
        (1usize, h, *te, word.clone())
    };

    Some((
        consumed,
        TemporalExpression {
            text: surface,
            span: (*ts, span_e),
            value: TemporalValue::Absolute(AbsoluteTime {
                year: None,
                month: None,
                day: None,
                hour: Some(hour_24),
                minute: Some(m),
            }),
            type_: TimeType::Time,
        },
    ))
}

/// Try to match time-of-day keywords: "noon", "midnight", "morning" etc.
fn match_time_of_day(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    let (ts, te, word) = &tokens[i];
    let lower = word.to_lowercase();
    let hour = TIME_OF_DAY.iter().find(|(w, _)| *w == lower.as_str()).map(|(_, h)| *h)?;
    Some((
        1,
        TemporalExpression {
            text: word.clone(),
            span: (*ts, *te),
            value: TemporalValue::Absolute(AbsoluteTime {
                year: None,
                month: None,
                day: None,
                hour: Some(hour),
                minute: None,
            }),
            type_: TimeType::Time,
        },
    ))
}

/// Try to match "every <unit>" — a Set expression.
fn match_set(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    if tokens[i].2.to_lowercase() != "every" {
        return None;
    }
    if i + 1 >= tokens.len() {
        return None;
    }
    let (ts, _te, every_word) = &tokens[i];
    let (_, ne, unit_word) = &tokens[i + 1];
    let lower = unit_word.to_lowercase();

    if DAYS_OF_WEEK.contains(&lower.as_str()) || duration_unit(&lower).is_some() {
        let surface = format!("{} {}", every_word, unit_word);
        let iso = duration_unit(&lower)
            .map(|u| {
                if u == "H" || u == "S" || (u == "M" && !lower.starts_with("mo")) {
                    format!("RPT1{}", u)
                } else {
                    format!("RP1{}", u)
                }
            })
            .unwrap_or_else(|| "RPT1W".to_string()); // weekday → weekly repeat
        return Some((
            2,
            TemporalExpression {
                text: surface,
                span: (*ts, *ne),
                value: TemporalValue::Relative(RelativeTime {
                    offset: iso,
                    anchor: "PRESENT_REF".to_string(),
                }),
                type_: TimeType::Set,
            },
        ));
    }
    None
}

/// Try to match "from <X> to <Y>" interval at `i`.
fn match_interval(
    tokens: &[(usize, usize, String)],
    i: usize,
    _doc_offset: usize,
) -> Option<(usize, TemporalExpression)> {
    if tokens[i].2.to_lowercase() != "from" {
        return None;
    }
    // Find nearest "to" token within 8 positions
    let mut to_idx: Option<usize> = None;
    for j in (i + 1)..(i + 9).min(tokens.len()) {
        if tokens[j].2.to_lowercase() == "to" {
            to_idx = Some(j);
            break;
        }
    }
    let to = to_idx?;
    if to + 1 >= tokens.len() {
        return None;
    }

    let start_parts: Vec<String> = tokens[(i + 1)..to]
        .iter()
        .map(|(_, _, w)| w.clone())
        .collect();
    let end_word = tokens[to + 1].2.clone();
    let ts = tokens[i].0;
    let te = tokens[to + 1].1;
    let surface = format!(
        "from {} to {}",
        start_parts.join(" "),
        end_word
    );

    Some((
        to + 2 - i,
        TemporalExpression {
            text: surface,
            span: (ts, te),
            value: TemporalValue::Interval {
                start: start_parts.join(" "),
                end: end_word,
            },
            type_: TimeType::TimeInterval,
        },
    ))
}

// ---------------------------------------------------------------------------
// Main extractor
// ---------------------------------------------------------------------------

/// Extract all temporal expressions from `text` using rule-based patterns.
///
/// Patterns are tried in priority order; when a longer match consumes tokens
/// a shorter overlapping match is skipped.
pub fn extract_temporal(text: &str) -> Vec<TemporalExpression> {
    let tokens = tokenise(text);
    let mut results: Vec<TemporalExpression> = Vec::new();
    let mut i = 0usize;

    // Priority-ordered list of matchers.  Each returns
    // `Option<(consumed_tokens, TemporalExpression)>`.
    while i < tokens.len() {
        let consumed: Option<(usize, TemporalExpression)> =
            match_interval(&tokens, i, 0)
                .or_else(|| match_full_date(&tokens, i, 0))
                .or_else(|| match_relative(&tokens, i, 0))
                .or_else(|| match_duration_ago(&tokens, i, 0))
                .or_else(|| match_in_year(&tokens, i, 0))
                .or_else(|| match_deictic(&tokens, i, 0))
                .or_else(|| match_clock_time(&tokens, i, 0))
                .or_else(|| match_time_of_day(&tokens, i, 0))
                .or_else(|| match_set(&tokens, i, 0));

        if let Some((n, expr)) = consumed {
            results.push(expr);
            i += n.max(1);
        } else {
            i += 1;
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Parse a simple `"YYYY-MM-DD"` reference date string.
fn parse_reference(reference: &str) -> Option<AbsoluteTime> {
    let parts: Vec<&str> = reference.split('-').collect();
    if parts.len() < 3 {
        if parts.len() == 1 && parts[0].len() == 4 {
            let y: i32 = parts[0].parse().ok()?;
            return Some(AbsoluteTime {
                year: Some(y),
                month: None,
                day: None,
                hour: None,
                minute: None,
            });
        }
        return None;
    }
    let y: i32 = parts[0].parse().ok()?;
    let m: u8 = parts[1].parse().ok()?;
    let d: u8 = parts[2].parse().ok()?;
    Some(AbsoluteTime {
        year: Some(y),
        month: Some(m),
        day: Some(d),
        hour: None,
        minute: None,
    })
}

/// Apply an ISO-8601 duration offset to an `AbsoluteTime` and return the result.
///
/// This is a simplified implementation that handles only the most common cases
/// (year, month, week, day offsets) and does not perform full calendar arithmetic.
fn apply_duration(base: &AbsoluteTime, iso_offset: &str) -> AbsoluteTime {
    let negative = iso_offset.starts_with('-');
    let s = iso_offset.trim_start_matches('-');
    // Remove leading 'P' and optional 'T'
    let s = s.trim_start_matches('P');

    let mut years: i32 = 0;
    let mut months: i32 = 0;
    let mut days: i32 = 0;
    let mut hours: i32 = 0;

    // Simple digit-prefix parser: e.g. "1Y", "2M", "3W", "4D"
    let mut chars = s.chars().peekable();
    let mut buf = String::new();
    let mut in_time = false;
    while let Some(c) = chars.next() {
        if c == 'T' {
            in_time = true;
            continue;
        }
        if c.is_ascii_digit() {
            buf.push(c);
        } else {
            let n: i32 = buf.parse().unwrap_or(0);
            buf.clear();
            match c {
                'Y' => years = n,
                'M' if !in_time => months = n,
                'W' => days += n * 7,
                'D' => days += n,
                'H' => hours = n,
                'M' if in_time => { /* minutes - ignored for now */ }
                _ => {}
            }
        }
    }

    let sign: i32 = if negative { -1 } else { 1 };

    let base_y = base.year.unwrap_or(0);
    let base_m = base.month.unwrap_or(1) as i32;
    let base_d = base.day.unwrap_or(1) as i32;

    let mut ny = base_y + sign * years;
    let mut nm = base_m + sign * months;
    let mut nd = base_d + sign * days;
    let mut nh = base.hour.map(|h| h as i32 + sign * hours);

    // Normalise month overflow
    while nm > 12 {
        nm -= 12;
        ny += 1;
    }
    while nm < 1 {
        nm += 12;
        ny -= 1;
    }
    // Clamp day (simplified)
    let max_day: i32 = 31;
    if nd > max_day {
        nd = max_day;
    }
    if nd < 1 {
        nd = 1;
    }
    // Clamp hour
    if let Some(ref mut h) = nh {
        if *h >= 24 {
            *h = 23;
        }
        if *h < 0 {
            *h = 0;
        }
    }

    AbsoluteTime {
        year: if base.year.is_some() { Some(ny) } else { None },
        month: if base.month.is_some() {
            Some(nm as u8)
        } else {
            None
        },
        day: if base.day.is_some() { Some(nd as u8) } else { None },
        hour: nh.map(|h| h as u8),
        minute: base.minute,
    }
}

/// Normalise a [`TemporalExpression`] to a TimeML-style value string.
///
/// The `reference` parameter should be a date string in `"YYYY-MM-DD"` format
/// that serves as the document creation time (DCT) for resolving relative
/// expressions.  If `reference` cannot be parsed, the raw ISO offset is
/// returned unchanged.
///
/// # Examples
///
/// ```rust
/// use scirs2_text::temporal::{extract_temporal, normalize_temporal};
///
/// let exprs = extract_temporal("The report was filed yesterday.");
/// assert!(!exprs.is_empty());
/// let norm = normalize_temporal(&exprs[0], "2024-01-16");
/// assert_eq!(norm, "2024-01-15"); // one day before reference
/// ```
pub fn normalize_temporal(expr: &TemporalExpression, reference: &str) -> String {
    match &expr.value {
        TemporalValue::Absolute(abs) => abs.to_iso(),
        TemporalValue::Relative(rel) => {
            if let Some(base) = parse_reference(reference) {
                let resolved = apply_duration(&base, &rel.offset);
                resolved.to_iso()
            } else {
                // Fall back to the raw offset
                rel.offset.clone()
            }
        }
        TemporalValue::Interval { start, end } => {
            format!("{}/{}", start, end)
        }
        TemporalValue::Unknown => "UNKNOWN".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Temporal relation
// ---------------------------------------------------------------------------

/// Ordering relationship between two temporal expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalRelation {
    /// The first expression is strictly earlier than the second.
    Before,
    /// The first expression is strictly later than the second.
    After,
    /// The first expression is entirely contained within the second.
    During,
    /// The first expression contains the second.
    Includes,
    /// The two expressions describe the same point / period.
    Simultaneous,
    /// The relation cannot be determined from available information.
    Unknown,
}

impl std::fmt::Display for TemporalRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Before => write!(f, "BEFORE"),
            Self::After => write!(f, "AFTER"),
            Self::During => write!(f, "DURING"),
            Self::Includes => write!(f, "INCLUDES"),
            Self::Simultaneous => write!(f, "SIMULTANEOUS"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Extract an integer timestamp proxy from an `AbsoluteTime` for comparison.
fn abs_to_minutes(a: &AbsoluteTime) -> Option<i64> {
    let y = a.year? as i64;
    let m = a.month.unwrap_or(1) as i64;
    let d = a.day.unwrap_or(1) as i64;
    let h = a.hour.unwrap_or(0) as i64;
    let mi = a.minute.unwrap_or(0) as i64;
    Some(y * 525_960 + m * 43_830 + d * 1_440 + h * 60 + mi)
}

/// Normalise both expressions using `reference` and determine their ordering.
///
/// This function resolves relative expressions to absolute times using
/// `reference` before comparison.  If either expression cannot be resolved
/// to a comparable point, [`TemporalRelation::Unknown`] is returned.
///
/// For interval expressions the midpoint is used for ordering.
pub fn temporal_relation(
    expr1: &TemporalExpression,
    expr2: &TemporalExpression,
) -> TemporalRelation {
    temporal_relation_with_reference(expr1, expr2, "2000-01-01")
}

/// Like [`temporal_relation`] but with an explicit document creation time.
pub fn temporal_relation_with_reference(
    expr1: &TemporalExpression,
    expr2: &TemporalExpression,
    reference: &str,
) -> TemporalRelation {
    let norm1 = normalize_temporal(expr1, reference);
    let norm2 = normalize_temporal(expr2, reference);

    // For intervals, compare start points
    let (start1, is_interval1) = if norm1.contains('/') {
        let p: Vec<&str> = norm1.splitn(2, '/').collect();
        (p[0].to_string(), true)
    } else {
        (norm1.clone(), false)
    };
    let (start2, is_interval2) = if norm2.contains('/') {
        let p: Vec<&str> = norm2.splitn(2, '/').collect();
        (p[0].to_string(), true)
    } else {
        (norm2.clone(), false)
    };

    let abs1 = parse_reference(&start1);
    let abs2 = parse_reference(&start2);

    match (abs1, abs2) {
        (Some(a1), Some(a2)) => {
            let t1 = match abs_to_minutes(&a1) {
                Some(t) => t,
                None => return TemporalRelation::Unknown,
            };
            let t2 = match abs_to_minutes(&a2) {
                Some(t) => t,
                None => return TemporalRelation::Unknown,
            };
            if t1 == t2 {
                if is_interval1 && !is_interval2 {
                    TemporalRelation::Includes
                } else if !is_interval1 && is_interval2 {
                    TemporalRelation::During
                } else {
                    TemporalRelation::Simultaneous
                }
            } else if t1 < t2 {
                TemporalRelation::Before
            } else {
                TemporalRelation::After
            }
        }
        _ => TemporalRelation::Unknown,
    }
}

// ---------------------------------------------------------------------------
// High-level builder
// ---------------------------------------------------------------------------

/// Configurable temporal expression extractor.
pub struct TemporalExtractor {
    /// Document creation time used as reference for normalisation.
    pub reference_date: String,
}

impl Default for TemporalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalExtractor {
    /// Create a new extractor with no reference date.
    pub fn new() -> Self {
        Self {
            reference_date: String::new(),
        }
    }

    /// Set the reference date (document creation time) used for normalisation.
    pub fn with_reference(mut self, date: impl Into<String>) -> Self {
        self.reference_date = date.into();
        self
    }

    /// Extract temporal expressions from `text`.
    pub fn extract(&self, text: &str) -> Result<Vec<TemporalExpression>> {
        if text.is_empty() {
            return Err(TextError::InvalidInput(
                "Input text must not be empty".to_string(),
            ));
        }
        Ok(extract_temporal(text))
    }

    /// Normalise all extracted expressions against the stored reference date.
    pub fn normalize_all(&self, exprs: &[TemporalExpression]) -> Vec<String> {
        let ref_date = if self.reference_date.is_empty() {
            "2000-01-01"
        } else {
            &self.reference_date
        };
        exprs
            .iter()
            .map(|e| normalize_temporal(e, ref_date))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_deictic() {
        let exprs = extract_temporal("We met yesterday and will meet tomorrow.");
        assert!(exprs.iter().any(|e| e.text == "yesterday"));
        assert!(exprs.iter().any(|e| e.text == "tomorrow"));
        assert!(exprs.iter().all(|e| e.type_ == TimeType::Date));
    }

    #[test]
    fn test_extract_relative() {
        let exprs = extract_temporal("The report is due next Monday.");
        assert!(!exprs.is_empty());
        let e = &exprs[0];
        assert_eq!(e.type_, TimeType::Date);
        assert!(e.text.to_lowercase().contains("next"));
    }

    #[test]
    fn test_extract_duration_ago() {
        let exprs = extract_temporal("That happened 3 days ago.");
        assert!(!exprs.is_empty());
        let e = exprs.iter().find(|e| e.text.contains("days")).expect("duration");
        assert_eq!(e.type_, TimeType::Duration);
        if let TemporalValue::Relative(r) = &e.value {
            assert!(r.offset.contains('3'));
        } else {
            panic!("Expected Relative value");
        }
    }

    #[test]
    fn test_extract_full_date() {
        let exprs = extract_temporal("The meeting is on January 15, 2024.");
        assert!(!exprs.is_empty());
        let e = exprs.iter().find(|e| e.type_ == TimeType::Date).expect("date");
        if let TemporalValue::Absolute(abs) = &e.value {
            assert_eq!(abs.year, Some(2024));
            assert_eq!(abs.month, Some(1));
            assert_eq!(abs.day, Some(15));
        } else {
            panic!("Expected Absolute value");
        }
    }

    #[test]
    fn test_extract_in_year() {
        let exprs = extract_temporal("She was born in 1990.");
        assert!(!exprs.is_empty());
        let e = exprs.iter().find(|e| e.text.contains("1990")).expect("year");
        if let TemporalValue::Absolute(abs) = &e.value {
            assert_eq!(abs.year, Some(1990));
        } else {
            panic!("Expected Absolute");
        }
    }

    #[test]
    fn test_extract_clock_time() {
        let exprs = extract_temporal("The call starts at 9:30 AM.");
        assert!(!exprs.is_empty());
        let e = exprs.iter().find(|e| e.type_ == TimeType::Time).expect("time");
        if let TemporalValue::Absolute(abs) = &e.value {
            assert_eq!(abs.hour, Some(9));
            assert_eq!(abs.minute, Some(30));
        } else {
            panic!("Expected Absolute");
        }
    }

    #[test]
    fn test_extract_time_of_day() {
        let exprs = extract_temporal("We will meet at noon.");
        assert!(exprs.iter().any(|e| e.type_ == TimeType::Time));
    }

    #[test]
    fn test_extract_set() {
        let exprs = extract_temporal("Every week she attends the seminar.");
        assert!(exprs.iter().any(|e| e.type_ == TimeType::Set));
    }

    #[test]
    fn test_extract_interval() {
        let exprs = extract_temporal("The conference runs from Monday to Friday.");
        assert!(exprs.iter().any(|e| e.type_ == TimeType::TimeInterval));
    }

    #[test]
    fn test_normalize_yesterday() {
        let exprs = extract_temporal("yesterday");
        assert!(!exprs.is_empty());
        let norm = normalize_temporal(&exprs[0], "2024-01-16");
        assert_eq!(norm, "2024-01-15");
    }

    #[test]
    fn test_normalize_absolute() {
        let exprs = extract_temporal("January 15, 2024");
        assert!(!exprs.is_empty());
        let norm = normalize_temporal(&exprs[0], "2024-01-01");
        assert_eq!(norm, "2024-01-15");
    }

    #[test]
    fn test_temporal_relation_before() {
        let e1 = extract_temporal("January 1, 2020");
        let e2 = extract_temporal("January 1, 2021");
        assert!(!e1.is_empty());
        assert!(!e2.is_empty());
        let rel = temporal_relation_with_reference(&e1[0], &e2[0], "2020-01-01");
        assert_eq!(rel, TemporalRelation::Before);
    }

    #[test]
    fn test_temporal_relation_after() {
        let e1 = extract_temporal("January 1, 2022");
        let e2 = extract_temporal("January 1, 2020");
        assert!(!e1.is_empty());
        assert!(!e2.is_empty());
        let rel = temporal_relation_with_reference(&e1[0], &e2[0], "2020-01-01");
        assert_eq!(rel, TemporalRelation::After);
    }

    #[test]
    fn test_temporal_relation_simultaneous() {
        let e1 = extract_temporal("January 15, 2024");
        let e2 = extract_temporal("January 15, 2024");
        assert!(!e1.is_empty());
        assert!(!e2.is_empty());
        let rel = temporal_relation_with_reference(&e1[0], &e2[0], "2024-01-01");
        assert_eq!(rel, TemporalRelation::Simultaneous);
    }

    #[test]
    fn test_extractor_builder() {
        let extractor = TemporalExtractor::new().with_reference("2024-01-16");
        let exprs = extractor.extract("She arrived yesterday.").expect("ok");
        assert!(!exprs.is_empty());
        let norms = extractor.normalize_all(&exprs);
        assert_eq!(norms[0], "2024-01-15");
    }

    #[test]
    fn test_extractor_empty_error() {
        let extractor = TemporalExtractor::new();
        assert!(extractor.extract("").is_err());
    }
}
