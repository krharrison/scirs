//! Event extraction from natural language text.
//!
//! This module provides rule-based event detection aligned with the ACE 2005
//! event taxonomy, argument identification, and temporal grounding of events.
//!
//! # Overview
//!
//! - [`EventType`] — ACE-style event type hierarchy
//! - [`ArgumentRole`] — semantic role labels for event arguments
//! - [`Argument`] — a single event argument with text, span, and role
//! - [`Event`] — a complete extracted event instance
//! - [`EventExtractor`] — configurable event extractor with pattern tables
//! - [`extract_events`] — convenience top-level function
//! - [`event_patterns`] — enumerate built-in ACE-style patterns
//! - [`argument_detection`] — heuristic argument detection

use crate::error::{Result, TextError};
use crate::temporal::temporal_relations::{extract_time_expressions, TimeExpression};
use regex::Regex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// EventType
// ---------------------------------------------------------------------------

/// ACE 2005-inspired event type taxonomy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    // --- Life ---
    /// Birth of a person.
    BeBorn,
    /// Marriage of persons.
    Marry,
    /// Divorce of persons.
    Divorce,
    /// Injury to a person.
    Injure,
    /// Death of a person.
    Die,

    // --- Movement ---
    /// Transportation / movement event.
    Transport,

    // --- Transaction ---
    /// Transfer of ownership.
    TransferOwnership,
    /// Transfer of money.
    TransferMoney,

    // --- Business ---
    /// Start of an organisation or position.
    StartOrg,
    /// Merge of organisations.
    MergeOrg,
    /// Declaration of bankruptcy.
    DeclareBankruptcy,
    /// End / dissolution of an organisation.
    EndOrg,

    // --- Conflict ---
    /// Armed attack or assault.
    Attack,
    /// Non-violent demonstration.
    Demonstrate,

    // --- Contact ---
    /// Meeting or gathering.
    Meet,
    /// Phone, email, or other communication.
    PhoneWrite,

    // --- Personnel ---
    /// Start of a job or role.
    StartPosition,
    /// End of a job or role.
    EndPosition,
    /// Nomination to a position.
    Nominate,
    /// Election to a position.
    Elect,

    // --- Justice ---
    /// Arrest or detention.
    ArrestJail,
    /// Legal charge filing.
    ChargeIndict,
    /// Court judgement.
    Trial,
    /// Sentencing.
    Sentence,
    /// Conviction.
    Convict,
    /// Acquittal.
    Acquit,
    /// Appeal.
    Appeal,
    /// Pardon or release.
    Pardon,
    /// Execution.
    Execute,

    // --- Generic ---
    /// User-defined or unclassified event.
    Custom(String),
}

impl EventType {
    /// Return a human-readable label.
    pub fn label(&self) -> &str {
        match self {
            EventType::BeBorn => "Life:BeBorn",
            EventType::Marry => "Life:Marry",
            EventType::Divorce => "Life:Divorce",
            EventType::Injure => "Life:Injure",
            EventType::Die => "Life:Die",
            EventType::Transport => "Movement:Transport",
            EventType::TransferOwnership => "Transaction:TransferOwnership",
            EventType::TransferMoney => "Transaction:TransferMoney",
            EventType::StartOrg => "Business:StartOrg",
            EventType::MergeOrg => "Business:MergeOrg",
            EventType::DeclareBankruptcy => "Business:DeclareBankruptcy",
            EventType::EndOrg => "Business:EndOrg",
            EventType::Attack => "Conflict:Attack",
            EventType::Demonstrate => "Conflict:Demonstrate",
            EventType::Meet => "Contact:Meet",
            EventType::PhoneWrite => "Contact:PhoneWrite",
            EventType::StartPosition => "Personnel:StartPosition",
            EventType::EndPosition => "Personnel:EndPosition",
            EventType::Nominate => "Personnel:Nominate",
            EventType::Elect => "Personnel:Elect",
            EventType::ArrestJail => "Justice:ArrestJail",
            EventType::ChargeIndict => "Justice:ChargeIndict",
            EventType::Trial => "Justice:Trial",
            EventType::Sentence => "Justice:Sentence",
            EventType::Convict => "Justice:Convict",
            EventType::Acquit => "Justice:Acquit",
            EventType::Appeal => "Justice:Appeal",
            EventType::Pardon => "Justice:Pardon",
            EventType::Execute => "Justice:Execute",
            EventType::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// ArgumentRole
// ---------------------------------------------------------------------------

/// Semantic role of an event argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArgumentRole {
    /// The agent initiating the event.
    Agent,
    /// The entity undergoing or affected by the event.
    Patient,
    /// Destination of movement or transfer.
    Destination,
    /// Origin of movement or transfer.
    Origin,
    /// Instrument used in the event.
    Instrument,
    /// Location where the event takes place.
    Place,
    /// Temporal anchor of the event.
    Time,
    /// Cause or reason.
    Cause,
    /// The object being transferred / owned.
    Artifact,
    /// Amount of money.
    Money,
    /// A person who receives something.
    Recipient,
    /// A person who gives something.
    Giver,
    /// An organisation involved.
    Organisation,
    /// Generic participant.
    Participant,
    /// User-defined role.
    Custom(String),
}

impl std::fmt::Display for ArgumentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ArgumentRole::Agent => "AGENT",
            ArgumentRole::Patient => "PATIENT",
            ArgumentRole::Destination => "DESTINATION",
            ArgumentRole::Origin => "ORIGIN",
            ArgumentRole::Instrument => "INSTRUMENT",
            ArgumentRole::Place => "PLACE",
            ArgumentRole::Time => "TIME",
            ArgumentRole::Cause => "CAUSE",
            ArgumentRole::Artifact => "ARTIFACT",
            ArgumentRole::Money => "MONEY",
            ArgumentRole::Recipient => "RECIPIENT",
            ArgumentRole::Giver => "GIVER",
            ArgumentRole::Organisation => "ORGANISATION",
            ArgumentRole::Participant => "PARTICIPANT",
            ArgumentRole::Custom(s) => s.as_str(),
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// Argument
// ---------------------------------------------------------------------------

/// A single extracted event argument.
#[derive(Debug, Clone, PartialEq)]
pub struct Argument {
    /// Raw text of the argument span.
    pub text: String,
    /// Byte start in the source string.
    pub start: usize,
    /// Byte end (exclusive) in the source string.
    pub end: usize,
    /// Semantic role.
    pub role: ArgumentRole,
    /// Confidence in [0, 1].
    pub confidence: f64,
}

impl Argument {
    /// Create a new argument.
    pub fn new(
        text: impl Into<String>,
        start: usize,
        end: usize,
        role: ArgumentRole,
        confidence: f64,
    ) -> Self {
        Argument {
            text: text.into(),
            start,
            end,
            role,
            confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A single extracted event instance.
#[derive(Debug, Clone)]
pub struct Event {
    /// ACE-style type of the event.
    pub event_type: EventType,
    /// The trigger word or phrase that signals the event.
    pub trigger_word: String,
    /// Byte start of the trigger in the source string.
    pub trigger_start: usize,
    /// Byte end (exclusive) of the trigger.
    pub trigger_end: usize,
    /// Extracted arguments with their roles.
    pub arguments: Vec<Argument>,
    /// Temporal expressions associated with this event.
    pub time_expressions: Vec<TimeExpression>,
    /// The sentence (or nearby context) containing the event.
    pub context: String,
    /// Extraction confidence in [0, 1].
    pub confidence: f64,
}

impl Event {
    /// Create a new event with no arguments or time expressions.
    pub fn new(
        event_type: EventType,
        trigger_word: impl Into<String>,
        trigger_start: usize,
        trigger_end: usize,
        context: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Event {
            event_type,
            trigger_word: trigger_word.into(),
            trigger_start,
            trigger_end,
            arguments: Vec::new(),
            time_expressions: Vec::new(),
            context: context.into(),
            confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// EventPattern  (internal)
// ---------------------------------------------------------------------------

/// An internal pattern entry in the event extractor.
struct EventPattern {
    event_type: EventType,
    trigger_re: Regex,
}

// ---------------------------------------------------------------------------
// EventExtractor
// ---------------------------------------------------------------------------

/// Rule-based event extractor.
///
/// Compile once with [`EventExtractor::default_english`] and reuse across
/// many texts.
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::event_extraction::EventExtractor;
///
/// let extractor = EventExtractor::default_english().expect("build extractor");
/// let events = extractor.extract("Police arrested the suspect yesterday in New York.");
/// assert!(!events.is_empty());
/// ```
pub struct EventExtractor {
    patterns: Vec<EventPattern>,
    sentence_boundary: Regex,
}

impl EventExtractor {
    /// Create an extractor with a custom set of patterns.
    ///
    /// `patterns` is a list of `(EventType, trigger_regex_pattern)` pairs.
    pub fn new(patterns: Vec<(EventType, &str)>) -> Result<Self> {
        let mut compiled = Vec::with_capacity(patterns.len());
        for (etype, pat) in patterns {
            let re = Regex::new(pat).map_err(|e| {
                TextError::ProcessingError(format!(
                    "Event pattern compile error for {}: {}",
                    etype, e
                ))
            })?;
            compiled.push(EventPattern {
                event_type: etype,
                trigger_re: re,
            });
        }
        let sentence_boundary = Regex::new(r"[.!?]+\s+")
            .map_err(|e| TextError::ProcessingError(format!("Regex error: {e}")))?;
        Ok(EventExtractor {
            patterns: compiled,
            sentence_boundary,
        })
    }

    /// Build an extractor pre-loaded with ACE-style English trigger patterns.
    pub fn default_english() -> Result<Self> {
        let raw = event_patterns();
        let pairs: Vec<(EventType, &str)> = raw
            .iter()
            .map(|(et, pat)| (et.clone(), pat.as_str()))
            .collect();
        EventExtractor::new(pairs)
    }

    /// Extract events from `text`, optionally scoped to sentence-level context.
    pub fn extract(&self, text: &str) -> Vec<Event> {
        let sentences = self.split_sentences(text);
        let mut events: Vec<Event> = Vec::new();

        for (sent, sent_start) in &sentences {
            for pattern in &self.patterns {
                for m in pattern.trigger_re.find_iter(sent) {
                    let trigger = m.as_str().to_owned();
                    let abs_start = sent_start + m.start();
                    let abs_end = sent_start + m.end();

                    let time_exprs = extract_time_expressions(sent);
                    let arguments = argument_detection(sent, &pattern.event_type);

                    let mut event = Event::new(
                        pattern.event_type.clone(),
                        trigger,
                        abs_start,
                        abs_end,
                        sent.trim(),
                        0.7,
                    );
                    event.time_expressions = time_exprs;
                    event.arguments = arguments;
                    events.push(event);
                }
            }
        }
        events
    }

    /// Split `text` into sentences, returning `(sentence, byte_offset)` pairs.
    fn split_sentences<'a>(&'a self, text: &'a str) -> Vec<(String, usize)> {
        let mut sentences: Vec<(String, usize)> = Vec::new();
        let mut last = 0usize;
        for m in self.sentence_boundary.find_iter(text) {
            let end = m.end();
            let slice = &text[last..end];
            if !slice.trim().is_empty() {
                sentences.push((slice.to_owned(), last));
            }
            last = end;
        }
        if last < text.len() {
            let tail = &text[last..];
            if !tail.trim().is_empty() {
                sentences.push((tail.to_owned(), last));
            }
        }
        if sentences.is_empty() {
            sentences.push((text.to_owned(), 0));
        }
        sentences
    }
}

// ---------------------------------------------------------------------------
// event_patterns
// ---------------------------------------------------------------------------

/// Return the built-in ACE-style trigger pattern table.
///
/// Each entry is `(EventType, regex_pattern_string)`.
pub fn event_patterns() -> Vec<(EventType, String)> {
    vec![
        // Life events
        (
            EventType::BeBorn,
            r"(?i)\b(born|birth|nativity|came into (the )?world)\b".to_owned(),
        ),
        (
            EventType::Marry,
            r"(?i)\b(married|wed(ded)?|nuptials?|wedding|matrimony)\b".to_owned(),
        ),
        (
            EventType::Divorce,
            r"(?i)\b(divorced|separated|split|dissolution of marriage)\b".to_owned(),
        ),
        (
            EventType::Injure,
            r"(?i)\b(injur(ed|y)|wounded|hurt|maimed|crippled|harmed)\b".to_owned(),
        ),
        (
            EventType::Die,
            r"(?i)\b(died?|killed|deceased|fatally|death|murdered|slain|perished|passed away)\b"
                .to_owned(),
        ),
        // Movement
        (
            EventType::Transport,
            r"(?i)\b(transported?|mov(ed|ing)|travel(led|ing)?|fled|evacuated?|deported?|migrated?|arrived?|departed?|left|returned)\b".to_owned(),
        ),
        // Transaction
        (
            EventType::TransferOwnership,
            r"(?i)\b(sold|bought|purchased|acquired|transferred ownership|deed(ed)?|conveyed|donated|bequeathed)\b".to_owned(),
        ),
        (
            EventType::TransferMoney,
            r"(?i)\b(paid|transferred|remitted?|wired|disbursed?|funded|invested|loaned|financed)\b".to_owned(),
        ),
        // Business
        (
            EventType::StartOrg,
            r"(?i)\b(founded|established|launched|incorporated|set up|opened|created|started)\b".to_owned(),
        ),
        (
            EventType::MergeOrg,
            r"(?i)\b(merged?|acquisition|acquired|consolidated?|absorbed?|amalgamated?)\b".to_owned(),
        ),
        (
            EventType::DeclareBankruptcy,
            r"(?i)\b(bankrupt(cy)?|insolven(t|cy)|filed for (chapter|bankruptcy)|liquidat(ed|ion))\b".to_owned(),
        ),
        (
            EventType::EndOrg,
            r"(?i)\b(dissolved?|shut down|closed down|disbanded?|ceased operations?|wound up)\b".to_owned(),
        ),
        // Conflict
        (
            EventType::Attack,
            r"(?i)\b(attack(ed)?|bomb(ed|ing)?|shot|fired on|assaulted?|struck|raided?|shelled?|invaded?|ambushed?|terroris(ed|ts?))\b".to_owned(),
        ),
        (
            EventType::Demonstrate,
            r"(?i)\b(demonstrated?|protested?|rallied?|march(ed|ing)?|picketed?|riots?|rioted)\b".to_owned(),
        ),
        // Contact
        (
            EventType::Meet,
            r"(?i)\b(met|meeting|conference|summit|convened?|gathered?|assembled?|caucus(ed)?|conferred?)\b".to_owned(),
        ),
        (
            EventType::PhoneWrite,
            r"(?i)\b(called?|phoned?|emailed?|texted?|messaged?|wired?|telegrammed?|contacted?|communicated?)\b".to_owned(),
        ),
        // Personnel
        (
            EventType::StartPosition,
            r"(?i)\b(appointed|elected|hired|promoted|assumed office|took office|sworn in|inaugurated|named (as|to)?)\b".to_owned(),
        ),
        (
            EventType::EndPosition,
            r"(?i)\b(resigned?|retired?|fired?|dismissed?|stepped down|left office|impeached?|removed from)\b".to_owned(),
        ),
        (
            EventType::Nominate,
            r"(?i)\b(nominated?|put forward|endorsed?|recommended?|proposed? (as|for))\b".to_owned(),
        ),
        (
            EventType::Elect,
            r"(?i)\b(elected?|voted (in|for)|won (the )?election|re-elected?|chosen)\b".to_owned(),
        ),
        // Justice
        (
            EventType::ArrestJail,
            r"(?i)\b(arrested?|detained?|jailed?|imprisoned?|taken into custody|apprehended?|nabbed?|captured?)\b".to_owned(),
        ),
        (
            EventType::ChargeIndict,
            r"(?i)\b(charged?|indicted?|accused?|prosecuted?|brought charges?|filed charges?)\b".to_owned(),
        ),
        (
            EventType::Trial,
            r"(?i)\b(tried?|on trial|court hearing|courtroom|testified?|hearing)\b".to_owned(),
        ),
        (
            EventType::Sentence,
            r"(?i)\b(sentenced?|given (a )?sentence|jail term|prison term|years? in prison)\b".to_owned(),
        ),
        (
            EventType::Convict,
            r"(?i)\b(convicted?|found guilty|guilty verdict|pronounced guilty)\b".to_owned(),
        ),
        (
            EventType::Acquit,
            r"(?i)\b(acquitted?|found not guilty|cleared?|exonerated?|not guilty verdict)\b".to_owned(),
        ),
        (
            EventType::Appeal,
            r"(?i)\b(appealed?|filed an appeal|challenging (the )?verdict|overturned?)\b".to_owned(),
        ),
        (
            EventType::Pardon,
            r"(?i)\b(pardoned?|commuted?|granted clemency|released?|freed?|paroled?)\b".to_owned(),
        ),
        (
            EventType::Execute,
            r"(?i)\b(executed?|put to death|capital punishment|death penalty|lethal injection|hanged)\b".to_owned(),
        ),
    ]
}

// ---------------------------------------------------------------------------
// argument_detection
// ---------------------------------------------------------------------------

/// Heuristic argument detection for a given sentence and event type.
///
/// Uses syntactic heuristics (named entity cues, prepositions, etc.) to
/// identify likely agents, patients, places, and times.
pub fn argument_detection(sentence: &str, event_type: &EventType) -> Vec<Argument> {
    let mut args: Vec<Argument> = Vec::new();

    // --- Time arguments (from TIMEX extraction) ---
    let time_exprs = extract_time_expressions(sentence);
    for texpr in time_exprs {
        args.push(Argument::new(
            &texpr.text,
            texpr.start,
            texpr.end,
            ArgumentRole::Time,
            0.8,
        ));
    }

    // --- Place arguments ---
    let place_cues = build_place_regex();
    if let Ok(re) = place_cues {
        for m in re.find_iter(sentence) {
            // Capture the following noun phrase as a place argument.
            let remainder = &sentence[m.end()..];
            if let Some(np) = extract_np(remainder) {
                let np_start = m.end();
                let np_end = np_start + np.len();
                args.push(Argument::new(np, np_start, np_end, ArgumentRole::Place, 0.65));
            }
        }
    }

    // --- Agent / Patient heuristics per event type ---
    match event_type {
        EventType::Attack | EventType::ArrestJail | EventType::Die => {
            detect_agent_patient(sentence, &mut args);
        }
        EventType::TransferMoney | EventType::TransferOwnership => {
            detect_giver_recipient(sentence, &mut args);
        }
        EventType::Meet | EventType::Demonstrate => {
            detect_participants(sentence, &mut args);
        }
        _ => {
            detect_agent_patient(sentence, &mut args);
        }
    }

    args
}

// ---------------------------------------------------------------------------
// Internal argument detection helpers
// ---------------------------------------------------------------------------

fn build_place_regex() -> Result<Regex> {
    Regex::new(r"(?i)\b(in|at|near|from|to|towards?|outside|inside|within)\s+")
        .map_err(|e| TextError::ProcessingError(format!("place regex: {e}")))
}

/// Extract a simple noun phrase (up to 4 tokens) from the beginning of `text`.
fn extract_np(text: &str) -> Option<&str> {
    let text = text.trim_start();
    // Skip common determiners/articles if present.
    let stripped = text
        .trim_start_matches(|c: char| c.is_lowercase() && c.is_alphabetic())
        .trim_start();

    // Capture up to 4 capitalised or lowercase words.
    let re = Regex::new(r"^([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3}|[a-z]+(?:\s+[a-z]+){0,2})")
        .ok()?;
    let m = re.find(stripped)?;
    if m.end() == 0 {
        return None;
    }
    // Return a slice of the original text with correct offset.
    let offset = text.len() - stripped.len();
    let end = offset + m.end();
    if end <= text.len() {
        Some(&text[..end.min(text.len())])
    } else {
        None
    }
}

fn detect_agent_patient(sentence: &str, args: &mut Vec<Argument>) {
    // Subject heuristic: first proper noun (capital letter word) before the trigger.
    let subject_re = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b").expect("static");
    let mut first = true;
    for m in subject_re.find_iter(sentence) {
        let word = m.as_str();
        // Skip common sentence-starters and titles.
        if matches!(
            word,
            "The" | "A" | "An" | "This" | "That" | "He" | "She" | "It" | "They" | "We" | "I"
        ) {
            continue;
        }
        if first {
            args.push(Argument::new(
                word,
                m.start(),
                m.end(),
                ArgumentRole::Agent,
                0.55,
            ));
            first = false;
        } else {
            args.push(Argument::new(
                word,
                m.start(),
                m.end(),
                ArgumentRole::Patient,
                0.5,
            ));
            break;
        }
    }
}

fn detect_giver_recipient(sentence: &str, args: &mut Vec<Argument>) {
    // "from X" → Giver, "to Y" → Recipient
    let from_re =
        Regex::new(r"(?i)\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)").expect("static");
    let to_re = Regex::new(r"(?i)\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)").expect("static");

    for caps in from_re.captures_iter(sentence) {
        if let Some(m) = caps.get(1) {
            args.push(Argument::new(
                m.as_str(),
                m.start(),
                m.end(),
                ArgumentRole::Giver,
                0.65,
            ));
        }
    }
    for caps in to_re.captures_iter(sentence) {
        if let Some(m) = caps.get(1) {
            args.push(Argument::new(
                m.as_str(),
                m.start(),
                m.end(),
                ArgumentRole::Recipient,
                0.65,
            ));
        }
    }

    // Money amounts
    let money_re = Regex::new(r"(?i)\$[\d,]+(\.\d{2})?|\d+\s*(million|billion|thousand)?\s*(dollars?|euros?|pounds?)").expect("static");
    for m in money_re.find_iter(sentence) {
        args.push(Argument::new(
            m.as_str(),
            m.start(),
            m.end(),
            ArgumentRole::Money,
            0.8,
        ));
    }
}

fn detect_participants(sentence: &str, args: &mut Vec<Argument>) {
    // Collect all proper noun spans as participants.
    let prop_re = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b").expect("static");
    for m in prop_re.find_iter(sentence) {
        let word = m.as_str();
        if matches!(
            word,
            "The" | "A" | "An" | "This" | "That" | "He" | "She" | "It" | "They" | "We" | "I"
        ) {
            continue;
        }
        args.push(Argument::new(
            word,
            m.start(),
            m.end(),
            ArgumentRole::Participant,
            0.5,
        ));
    }
}

// ---------------------------------------------------------------------------
// Top-level convenience function
// ---------------------------------------------------------------------------

/// Extract events from `text` using the default English pattern table.
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::event_extraction::extract_events;
///
/// let events = extract_events("Police arrested the suspect yesterday in New York.");
/// assert!(!events.is_empty());
/// ```
pub fn extract_events(text: &str) -> Vec<Event> {
    match EventExtractor::default_english() {
        Ok(extractor) => extractor.extract(text),
        Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_arrest_event() {
        let events = extract_events("Police arrested the suspect yesterday in New York.");
        assert!(!events.is_empty());
        let first = &events[0];
        assert_eq!(first.event_type, EventType::ArrestJail);
    }

    #[test]
    fn test_extract_attack_event() {
        let events = extract_events("The army attacked the city at dawn on 2024-06-01.");
        let attack_events: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == EventType::Attack)
            .collect();
        assert!(!attack_events.is_empty());
    }

    #[test]
    fn test_extract_die_event() {
        let events = extract_events("Three soldiers died in the explosion last Monday.");
        let die_events: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == EventType::Die)
            .collect();
        assert!(!die_events.is_empty());
    }

    #[test]
    fn test_time_argument_populated() {
        let events = extract_events("The CEO was arrested on 2023-11-05.");
        assert!(!events.is_empty());
        let has_time_arg = events[0]
            .arguments
            .iter()
            .any(|a| a.role == ArgumentRole::Time);
        assert!(has_time_arg, "expected a time argument");
    }

    #[test]
    fn test_event_patterns_non_empty() {
        let pats = event_patterns();
        assert!(!pats.is_empty());
    }

    #[test]
    fn test_argument_detection_place() {
        let args =
            argument_detection("The troops were deployed in Baghdad last week.", &EventType::Transport);
        let place_args: Vec<_> = args
            .iter()
            .filter(|a| a.role == ArgumentRole::Place)
            .collect();
        assert!(!place_args.is_empty(), "expected at least one place argument");
    }

    #[test]
    fn test_default_english_extractor_builds() {
        let ext = EventExtractor::default_english();
        assert!(ext.is_ok(), "EventExtractor::default_english should succeed");
    }
}
