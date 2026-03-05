//! Common temporal expression patterns for English text.
//!
//! This module provides compiled regex patterns for detecting temporal expressions:
//! calendar dates, times of day, durations, relative references, and recurring
//! frequency expressions.  All patterns are compiled once via `lazy_static` and
//! reused across calls.
//!
//! # Overview
//!
//! - [`AbsoluteDatePattern`] — fully specified calendar dates
//! - [`RelativeTimePattern`] — "yesterday", "last week", "in 2024"
//! - [`DurationPattern`] — "for 3 hours", "during the meeting"
//! - [`FrequencyPattern`] — "every day", "twice weekly"
//! - [`AnchorPattern`] — "before X", "after Y", "while Z"
//! - [`all_patterns`] — concatenated list of every pattern name
//! - [`PatternMatch`] — a single pattern hit with span and label

use crate::error::{Result, TextError};
use regex::Regex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PatternMatch
// ---------------------------------------------------------------------------

/// A single hit from a temporal pattern search.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternMatch {
    /// Human-readable name of the pattern that matched.
    pub pattern_name: String,
    /// Matched text slice.
    pub matched_text: String,
    /// Byte offset of the start of the match within the source string.
    pub start: usize,
    /// Byte offset of the end (exclusive) of the match.
    pub end: usize,
    /// Optional normalised / canonical form (may be empty).
    pub canonical: String,
}

impl PatternMatch {
    /// Create a new pattern match.
    pub fn new(
        pattern_name: impl Into<String>,
        matched_text: impl Into<String>,
        start: usize,
        end: usize,
        canonical: impl Into<String>,
    ) -> Self {
        PatternMatch {
            pattern_name: pattern_name.into(),
            matched_text: matched_text.into(),
            start,
            end,
            canonical: canonical.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern category structs
// ---------------------------------------------------------------------------

/// Compiled patterns for absolute calendar dates.
pub struct AbsoluteDatePattern {
    patterns: Vec<(&'static str, Regex)>,
}

impl AbsoluteDatePattern {
    /// Build and compile all absolute date patterns.
    pub fn new() -> Result<Self> {
        let raw: Vec<(&'static str, &'static str)> = vec![
            // ISO 8601: 2024-01-15
            ("iso_date", r"(?i)\b(\d{4})-(\d{2})-(\d{2})\b"),
            // US: January 15, 2024  /  Jan 15, 2024
            (
                "us_long",
                r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b",
            ),
            // Reverse: 15 January 2024
            (
                "eu_long",
                r"(?i)\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})\b",
            ),
            // Month-year: January 2024 / Jan 2024
            (
                "month_year",
                r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})\b",
            ),
            // Slash date: 01/15/2024  or  1/15/24
            ("slash_date", r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"),
            // Dot date: 15.01.2024
            ("dot_date", r"\b(\d{1,2})\.(\d{2})\.(\d{4})\b"),
            // Year only: in 2024 / of 1999
            ("year_only", r"(?i)\b(in|of|since|until|by|from)\s+(\d{4})\b"),
            // Decade: the 1990s, the '80s
            ("decade", r"(?i)\b(the\s+)?(\d{2,4}0)s\b"),
            // Season + year: Summer 2024
            (
                "season_year",
                r"(?i)\b(Spring|Summer|Autumn|Fall|Winter)\s+(\d{4})\b",
            ),
            // Quarter: Q1 2024 / first quarter of 2024
            (
                "quarter",
                r"(?i)\b(Q[1-4]|[Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth)\s+(quarter)\s+(of\s+)?(\d{4})\b",
            ),
        ];
        let mut patterns = Vec::with_capacity(raw.len());
        for (name, pat) in raw {
            let re = Regex::new(pat)
                .map_err(|e| TextError::ProcessingError(format!("Regex compile error for {name}: {e}")))?;
            patterns.push((name, re));
        }
        Ok(AbsoluteDatePattern { patterns })
    }

    /// Search `text` and return all absolute date matches.
    pub fn find_all<'a>(&'a self, text: &'a str) -> Vec<PatternMatch> {
        let mut out = Vec::new();
        for (name, re) in &self.patterns {
            for m in re.find_iter(text) {
                out.push(PatternMatch::new(
                    *name,
                    m.as_str(),
                    m.start(),
                    m.end(),
                    "",
                ));
            }
        }
        out
    }
}

impl Default for AbsoluteDatePattern {
    fn default() -> Self {
        AbsoluteDatePattern::new().expect("AbsoluteDatePattern default patterns must compile")
    }
}

// ---------------------------------------------------------------------------
// RelativeTimePattern
// ---------------------------------------------------------------------------

/// Compiled patterns for relative temporal references.
pub struct RelativeTimePattern {
    patterns: Vec<(&'static str, Regex)>,
}

impl RelativeTimePattern {
    /// Build and compile all relative time patterns.
    pub fn new() -> Result<Self> {
        let raw: Vec<(&'static str, &'static str)> = vec![
            // Simple deixis
            ("today", r"(?i)\b(today|tonight|this morning|this afternoon|this evening)\b"),
            ("yesterday", r"(?i)\b(yesterday|the day before yesterday)\b"),
            ("tomorrow", r"(?i)\b(tomorrow|the day after tomorrow)\b"),
            // Last/next + unit
            (
                "last_next_unit",
                r"(?i)\b(last|next|this|coming|past|previous)\s+(second|minute|hour|day|week|month|year|decade|century|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|November|December)\b",
            ),
            // N units ago / in N units
            (
                "n_units_ago",
                r"(?i)\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(second|minute|hour|day|week|month|year|decade)s?\s+ago\b",
            ),
            (
                "in_n_units",
                r"(?i)\bin\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(second|minute|hour|day|week|month|year|decade)s?\b",
            ),
            // recently / soon / lately / formerly / previously
            ("vague_recent", r"(?i)\b(recently|lately|just now|not long ago|a while ago|some time ago)\b"),
            ("vague_future", r"(?i)\b(soon|shortly|before long|in the near future|any day now)\b"),
            ("formerly", r"(?i)\b(formerly|previously|at one point|once|at that time|back then)\b"),
            // at the time
            ("at_the_time", r"(?i)\bat\s+(the\s+)?(time|moment|point)\b"),
            // early/mid/late + period
            (
                "early_mid_late",
                r"(?i)\b(early|mid|late|mid-)\s*(\d{4}s?|January|February|March|April|May|June|July|August|September|October|November|December|spring|summer|autumn|fall|winter)\b",
            ),
            // The Xth century
            (
                "century",
                r"(?i)\b(the\s+)?(\d{1,2})(st|nd|rd|th)\s+century\b",
            ),
        ];
        let mut patterns = Vec::with_capacity(raw.len());
        for (name, pat) in raw {
            let re = Regex::new(pat)
                .map_err(|e| TextError::ProcessingError(format!("Regex compile error for {name}: {e}")))?;
            patterns.push((name, re));
        }
        Ok(RelativeTimePattern { patterns })
    }

    /// Search `text` and return all relative time matches.
    pub fn find_all<'a>(&'a self, text: &'a str) -> Vec<PatternMatch> {
        let mut out = Vec::new();
        for (name, re) in &self.patterns {
            for m in re.find_iter(text) {
                out.push(PatternMatch::new(*name, m.as_str(), m.start(), m.end(), ""));
            }
        }
        out
    }
}

impl Default for RelativeTimePattern {
    fn default() -> Self {
        RelativeTimePattern::new().expect("RelativeTimePattern default patterns must compile")
    }
}

// ---------------------------------------------------------------------------
// DurationPattern
// ---------------------------------------------------------------------------

/// Compiled patterns for duration expressions.
pub struct DurationPattern {
    patterns: Vec<(&'static str, Regex)>,
}

impl DurationPattern {
    /// Build and compile all duration patterns.
    pub fn new() -> Result<Self> {
        let raw: Vec<(&'static str, &'static str)> = vec![
            // for N unit(s)
            (
                "for_n_units",
                r"(?i)\bfor\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(second|minute|hour|day|week|month|year|decade)s?\b",
            ),
            // ISO 8601 duration: P3Y2M1DT4H5M6S
            ("iso_duration", r"\bP(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?:\d+H)?(?:\d+M)?(?:\d+S)?)?\b"),
            // during the <noun phrase>  -- noun phrase approximation
            (
                "during_the",
                r"(?i)\bduring\s+(the\s+)?[A-Za-z\s]{2,30}?\b(?=\s|,|\.|$)",
            ),
            // over the past N unit(s)
            (
                "over_the_past",
                r"(?i)\bover\s+the\s+(past|last|next|coming)\s+(\d+\s+)?(second|minute|hour|day|week|month|year|decade|few|several|many)s?\b",
            ),
            // throughout the <period>
            (
                "throughout",
                r"(?i)\bthroughout\s+(the\s+)?(day|week|month|year|decade|century|period|morning|afternoon|evening|night|meeting|conference|session)\b",
            ),
            // spanning N units
            (
                "spanning",
                r"(?i)\bspanning\s+(\d+)\s+(second|minute|hour|day|week|month|year|decade)s?\b",
            ),
            // <N>-<unit> (attributive): a three-hour meeting, a two-day conference
            (
                "attributive",
                r"(?i)\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)-(second|minute|hour|day|week|month|year)\b",
            ),
            // from X to Y  (duration interval header; refined in temporal_relations)
            ("from_to", r"(?i)\bfrom\s+.{1,40}?\s+to\s+.{1,40}?\b"),
            // between X and Y
            (
                "between_and",
                r"(?i)\bbetween\s+.{1,40}?\s+and\s+.{1,40}?\b",
            ),
        ];
        let mut patterns = Vec::with_capacity(raw.len());
        for (name, pat) in raw {
            let re = Regex::new(pat)
                .map_err(|e| TextError::ProcessingError(format!("Regex compile error for {name}: {e}")))?;
            patterns.push((name, re));
        }
        Ok(DurationPattern { patterns })
    }

    /// Search `text` and return all duration matches.
    pub fn find_all<'a>(&'a self, text: &'a str) -> Vec<PatternMatch> {
        let mut out = Vec::new();
        for (name, re) in &self.patterns {
            for m in re.find_iter(text) {
                out.push(PatternMatch::new(*name, m.as_str(), m.start(), m.end(), ""));
            }
        }
        out
    }
}

impl Default for DurationPattern {
    fn default() -> Self {
        DurationPattern::new().expect("DurationPattern default patterns must compile")
    }
}

// ---------------------------------------------------------------------------
// FrequencyPattern
// ---------------------------------------------------------------------------

/// Compiled patterns for frequency / recurrence expressions.
pub struct FrequencyPattern {
    patterns: Vec<(&'static str, Regex)>,
}

impl FrequencyPattern {
    /// Build and compile all frequency patterns.
    pub fn new() -> Result<Self> {
        let raw: Vec<(&'static str, &'static str)> = vec![
            // every <unit>
            (
                "every_unit",
                r"(?i)\bevery\s+(second|minute|hour|day|week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|morning|afternoon|evening|night)\b",
            ),
            // once/twice/N times a <unit>
            (
                "n_times_a_unit",
                r"(?i)\b(once|twice|\d+\s+times?)\s+(a|per|each)\s+(second|minute|hour|day|week|month|year)\b",
            ),
            // daily / weekly / monthly / yearly / annually / hourly
            (
                "adverb_frequency",
                r"(?i)\b(daily|weekly|bi-weekly|biweekly|fortnightly|monthly|bi-monthly|bimonthly|quarterly|annually|yearly|hourly|minutely|secondly)\b",
            ),
            // every N units
            (
                "every_n_units",
                r"(?i)\bevery\s+(\d+|other|other)\s+(second|minute|hour|day|week|month|year)s?\b",
            ),
            // on a <unit> basis
            (
                "on_a_basis",
                r"(?i)\bon\s+a\s+(daily|weekly|monthly|quarterly|yearly|annual|regular|recurring)\s+basis\b",
            ),
            // recurring  / periodically / regularly
            (
                "recurrence_adverb",
                r"(?i)\b(recurring|periodically|regularly|repeatedly|intermittently|sporadically|consistently|routinely)\b",
            ),
            // each <unit>
            (
                "each_unit",
                r"(?i)\beach\s+(second|minute|hour|day|week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|morning|afternoon|evening|night)\b",
            ),
        ];
        let mut patterns = Vec::with_capacity(raw.len());
        for (name, pat) in raw {
            let re = Regex::new(pat)
                .map_err(|e| TextError::ProcessingError(format!("Regex compile error for {name}: {e}")))?;
            patterns.push((name, re));
        }
        Ok(FrequencyPattern { patterns })
    }

    /// Search `text` and return all frequency matches.
    pub fn find_all<'a>(&'a self, text: &'a str) -> Vec<PatternMatch> {
        let mut out = Vec::new();
        for (name, re) in &self.patterns {
            for m in re.find_iter(text) {
                out.push(PatternMatch::new(*name, m.as_str(), m.start(), m.end(), ""));
            }
        }
        out
    }
}

impl Default for FrequencyPattern {
    fn default() -> Self {
        FrequencyPattern::new().expect("FrequencyPattern default patterns must compile")
    }
}

// ---------------------------------------------------------------------------
// AnchorPattern  (before X / after Y / while Z)
// ---------------------------------------------------------------------------

/// Compiled patterns for relational anchor expressions.
pub struct AnchorPattern {
    patterns: Vec<(&'static str, Regex)>,
}

impl AnchorPattern {
    /// Build and compile all anchor patterns.
    pub fn new() -> Result<Self> {
        let raw: Vec<(&'static str, &'static str)> = vec![
            // before / after <NP or clause>
            (
                "before_event",
                r"(?i)\bbefore\s+(?:the\s+)?([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
            (
                "after_event",
                r"(?i)\bafter\s+(?:the\s+)?([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
            // while / during / as / when
            (
                "while_clause",
                r"(?i)\b(while|during|as|when|whenever|at the time)\s+([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
            // prior to / subsequent to
            (
                "prior_subsequent",
                r"(?i)\b(prior|subsequent)\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
            // immediately before/after
            (
                "immediately",
                r"(?i)\bimmediately\s+(before|after|following|preceding)\s+(?:the\s+)?([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
            // at the same time as
            (
                "at_same_time",
                r"(?i)\bat\s+the\s+same\s+time\s+as\s+([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
            // simultaneously with
            (
                "simultaneously",
                r"(?i)\bsimultaneously\s+(with\s+)?([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
            // by the time
            (
                "by_the_time",
                r"(?i)\bby\s+the\s+time\s+([A-Za-z][A-Za-z\s]{2,40}?)\b(?=\s|,|\.|;|$)",
            ),
        ];
        let mut patterns = Vec::with_capacity(raw.len());
        for (name, pat) in raw {
            let re = Regex::new(pat)
                .map_err(|e| TextError::ProcessingError(format!("Regex compile error for {name}: {e}")))?;
            patterns.push((name, re));
        }
        Ok(AnchorPattern { patterns })
    }

    /// Search `text` and return all anchor matches.
    pub fn find_all<'a>(&'a self, text: &'a str) -> Vec<PatternMatch> {
        let mut out = Vec::new();
        for (name, re) in &self.patterns {
            for m in re.find_iter(text) {
                out.push(PatternMatch::new(*name, m.as_str(), m.start(), m.end(), ""));
            }
        }
        out
    }
}

impl Default for AnchorPattern {
    fn default() -> Self {
        AnchorPattern::new().expect("AnchorPattern default patterns must compile")
    }
}

// ---------------------------------------------------------------------------
// TemporalPatternBank  (unified access)
// ---------------------------------------------------------------------------

/// A unified bank of all temporal pattern families.
///
/// Construct once and reuse for efficiency; all regex compilations happen
/// in `new()`.
pub struct TemporalPatternBank {
    /// Absolute calendar date patterns.
    pub absolute: AbsoluteDatePattern,
    /// Relative temporal reference patterns.
    pub relative: RelativeTimePattern,
    /// Duration patterns.
    pub duration: DurationPattern,
    /// Frequency / recurrence patterns.
    pub frequency: FrequencyPattern,
    /// Relational anchor patterns.
    pub anchor: AnchorPattern,
}

impl TemporalPatternBank {
    /// Compile all pattern families.
    pub fn new() -> Result<Self> {
        Ok(TemporalPatternBank {
            absolute: AbsoluteDatePattern::new()?,
            relative: RelativeTimePattern::new()?,
            duration: DurationPattern::new()?,
            frequency: FrequencyPattern::new()?,
            anchor: AnchorPattern::new()?,
        })
    }

    /// Run all pattern families against `text` and return a combined,
    /// de-overlapped list sorted by start offset.
    pub fn find_all(&self, text: &str) -> Vec<PatternMatch> {
        let mut hits: Vec<PatternMatch> = Vec::new();
        hits.extend(self.absolute.find_all(text));
        hits.extend(self.relative.find_all(text));
        hits.extend(self.duration.find_all(text));
        hits.extend(self.frequency.find_all(text));
        hits.extend(self.anchor.find_all(text));

        // Sort by start, then prefer longer match on ties.
        hits.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| b.end.cmp(&a.end))
        });

        // Remove overlapping matches (keep the earlier / longer one).
        let mut deduped: Vec<PatternMatch> = Vec::new();
        let mut last_end = 0usize;
        for hit in hits {
            if hit.start >= last_end {
                last_end = hit.end;
                deduped.push(hit);
            }
        }
        deduped
    }
}

impl Default for TemporalPatternBank {
    fn default() -> Self {
        TemporalPatternBank::new().expect("TemporalPatternBank default patterns must compile")
    }
}

// ---------------------------------------------------------------------------
// Public convenience helpers
// ---------------------------------------------------------------------------

/// Return all temporal pattern hits in `text`, sorted by start offset.
///
/// This is a convenience wrapper around [`TemporalPatternBank::find_all`].
pub fn all_patterns(text: &str) -> Vec<PatternMatch> {
    TemporalPatternBank::default().find_all(text)
}

/// Normalise a cardinal word to the corresponding integer, e.g. "three" → 3.
/// Returns `None` for unrecognised words.
pub fn word_to_number(s: &str) -> Option<u32> {
    let table: HashMap<&str, u32> = [
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
        ("eleven", 11),
        ("twelve", 12),
        ("a", 1),
        ("an", 1),
    ]
    .iter()
    .copied()
    .collect();
    let lower = s.to_lowercase();
    table.get(lower.as_str()).copied()
}

/// Convert a month name (full or abbreviated) to its 1-based numeric index.
/// Returns `None` for unrecognised names.
pub fn month_name_to_number(month: &str) -> Option<u8> {
    let table: HashMap<&str, u8> = [
        ("january", 1),
        ("jan", 1),
        ("february", 2),
        ("feb", 2),
        ("march", 3),
        ("mar", 3),
        ("april", 4),
        ("apr", 4),
        ("may", 5),
        ("june", 6),
        ("jun", 6),
        ("july", 7),
        ("jul", 7),
        ("august", 8),
        ("aug", 8),
        ("september", 9),
        ("sep", 9),
        ("sept", 9),
        ("october", 10),
        ("oct", 10),
        ("november", 11),
        ("nov", 11),
        ("december", 12),
        ("dec", 12),
    ]
    .iter()
    .copied()
    .collect();
    let lower = month.to_lowercase();
    table.get(lower.as_str()).copied()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absolute_date_iso() {
        let bank = AbsoluteDatePattern::new().expect("compile");
        let hits = bank.find_all("The meeting on 2024-03-15 was productive.");
        assert!(!hits.is_empty(), "expected at least one absolute date hit");
        assert_eq!(hits[0].matched_text, "2024-03-15");
    }

    #[test]
    fn test_relative_yesterday() {
        let bank = RelativeTimePattern::new().expect("compile");
        let hits = bank.find_all("I saw him yesterday morning.");
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_duration_for_n_units() {
        let bank = DurationPattern::new().expect("compile");
        let hits = bank.find_all("The conference lasted for three days.");
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_frequency_every_day() {
        let bank = FrequencyPattern::new().expect("compile");
        let hits = bank.find_all("She jogs every day at 6 AM.");
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_anchor_before_event() {
        let bank = AnchorPattern::new().expect("compile");
        let hits = bank.find_all("He arrived before the meeting started.");
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_bank_deduplication() {
        let bank = TemporalPatternBank::new().expect("compile");
        let hits = bank.find_all("Yesterday I attended a meeting for two hours.");
        // Should not have overlapping spans.
        for window in hits.windows(2) {
            assert!(window[1].start >= window[0].end, "overlapping spans detected");
        }
    }

    #[test]
    fn test_word_to_number() {
        assert_eq!(word_to_number("three"), Some(3));
        assert_eq!(word_to_number("twelve"), Some(12));
        assert_eq!(word_to_number("a"), Some(1));
        assert_eq!(word_to_number("forty"), None);
    }

    #[test]
    fn test_month_name_to_number() {
        assert_eq!(month_name_to_number("January"), Some(1));
        assert_eq!(month_name_to_number("dec"), Some(12));
        assert_eq!(month_name_to_number("month"), None);
    }
}
