//! Language detection module
//!
//! Provides multiple strategies for detecting the language of a text sample:
//!
//! - **N-gram profile comparison**: Compares character n-gram frequency
//!   profiles against reference profiles for known languages.
//! - **Common-word frequency analysis**: Counts occurrences of high-frequency
//!   function words unique to each language.
//! - **Unicode script detection**: Uses Unicode block analysis for scripts
//!   like CJK, Cyrillic, Arabic, Devanagari, etc.
//! - **Combined detection**: Merges evidence from all strategies.
//!
//! The unified entry point is [`detect_language`].

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// ISO 639-1 language codes supported by the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectedLanguage {
    /// English
    En,
    /// Spanish
    Es,
    /// French
    Fr,
    /// German
    De,
    /// Italian
    It,
    /// Portuguese
    Pt,
    /// Dutch
    Nl,
    /// Russian
    Ru,
    /// Chinese (Mandarin)
    Zh,
    /// Japanese
    Ja,
    /// Korean
    Ko,
    /// Arabic
    Ar,
    /// Hindi
    Hi,
    /// Turkish
    Tr,
    /// Swedish
    Sv,
    /// Polish
    Pl,
    /// Unknown / unrecognised
    Unknown,
}

impl DetectedLanguage {
    /// Return the ISO 639-1 code as a string.
    pub fn iso_code(&self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Es => "es",
            Self::Fr => "fr",
            Self::De => "de",
            Self::It => "it",
            Self::Pt => "pt",
            Self::Nl => "nl",
            Self::Ru => "ru",
            Self::Zh => "zh",
            Self::Ja => "ja",
            Self::Ko => "ko",
            Self::Ar => "ar",
            Self::Hi => "hi",
            Self::Tr => "tr",
            Self::Sv => "sv",
            Self::Pl => "pl",
            Self::Unknown => "und",
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::En => "English",
            Self::Es => "Spanish",
            Self::Fr => "French",
            Self::De => "German",
            Self::It => "Italian",
            Self::Pt => "Portuguese",
            Self::Nl => "Dutch",
            Self::Ru => "Russian",
            Self::Zh => "Chinese",
            Self::Ja => "Japanese",
            Self::Ko => "Korean",
            Self::Ar => "Arabic",
            Self::Hi => "Hindi",
            Self::Tr => "Turkish",
            Self::Sv => "Swedish",
            Self::Pl => "Polish",
            Self::Unknown => "Unknown",
        }
    }
}

/// Result of language detection.
#[derive(Debug, Clone)]
pub struct LanguageDetectionOutput {
    /// The most likely language.
    pub language: DetectedLanguage,
    /// Confidence in [0, 1].
    pub confidence: f64,
    /// Alternative candidates ranked by confidence (descending).
    pub alternatives: Vec<(DetectedLanguage, f64)>,
}

/// Detection strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionStrategy {
    /// Character n-gram profile comparison.
    Ngram,
    /// Common-word frequency analysis.
    WordFrequency,
    /// Unicode script analysis.
    UnicodeScript,
    /// Combined evidence from all strategies.
    Combined,
}

// ---------------------------------------------------------------------------
// Unified API
// ---------------------------------------------------------------------------

/// Detect the language of `text` using the combined strategy by default.
///
/// Returns a [`LanguageDetectionOutput`] containing the best guess, its
/// confidence, and a list of alternatives.
///
/// # Errors
///
/// Returns an error only if `text` is empty or too short for meaningful
/// detection (fewer than 3 characters). For very short texts the confidence
/// will simply be low.
pub fn detect_language(text: &str) -> Result<LanguageDetectionOutput> {
    detect_language_with_strategy(text, DetectionStrategy::Combined)
}

/// Detect language using a specific strategy.
pub fn detect_language_with_strategy(
    text: &str,
    strategy: DetectionStrategy,
) -> Result<LanguageDetectionOutput> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(TextError::InvalidInput(
            "Cannot detect language of empty text".to_string(),
        ));
    }

    match strategy {
        DetectionStrategy::Ngram => detect_by_ngram(trimmed),
        DetectionStrategy::WordFrequency => detect_by_word_frequency(trimmed),
        DetectionStrategy::UnicodeScript => detect_by_unicode_script(trimmed),
        DetectionStrategy::Combined => detect_combined(trimmed),
    }
}

// ---------------------------------------------------------------------------
// N-gram profile comparison
// ---------------------------------------------------------------------------

fn detect_by_ngram(text: &str) -> Result<LanguageDetectionOutput> {
    let text_profile = build_ngram_profile(text, 3);
    if text_profile.is_empty() {
        return Ok(unknown_result());
    }

    let reference_profiles = reference_ngram_profiles();
    let mut scores: Vec<(DetectedLanguage, f64)> = Vec::new();

    for (lang, ref_profile) in &reference_profiles {
        let similarity = profile_similarity(&text_profile, ref_profile);
        scores.push((*lang, similarity));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if scores.is_empty() {
        return Ok(unknown_result());
    }

    let best = scores[0];
    let confidence = best.1.clamp(0.0, 1.0);

    Ok(LanguageDetectionOutput {
        language: best.0,
        confidence,
        alternatives: scores.into_iter().skip(1).collect(),
    })
}

/// Build a frequency profile of character n-grams.
fn build_ngram_profile(text: &str, n: usize) -> HashMap<String, f64> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut counts: HashMap<String, f64> = HashMap::new();

    if chars.len() < n {
        return counts;
    }

    for window in chars.windows(n) {
        let gram: String = window.iter().collect();
        *counts.entry(gram).or_insert(0.0) += 1.0;
    }

    // Normalise.
    let total: f64 = counts.values().sum();
    if total > 0.0 {
        for v in counts.values_mut() {
            *v /= total;
        }
    }

    counts
}

/// Cosine similarity between two n-gram profiles.
fn profile_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (gram, &va) in a {
        norm_a += va * va;
        if let Some(&vb) = b.get(gram) {
            dot += va * vb;
        }
    }
    for &vb in b.values() {
        norm_b += vb * vb;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Reference n-gram profiles for known languages.
fn reference_ngram_profiles() -> HashMap<DetectedLanguage, HashMap<String, f64>> {
    let mut profiles = HashMap::new();

    // English trigrams.
    profiles.insert(
        DetectedLanguage::En,
        build_ref_profile(&[
            ("the", 50.0),
            ("and", 30.0),
            ("ing", 25.0),
            ("tion", 20.0),
            ("her", 18.0),
            ("ent", 17.0),
            ("ion", 16.0),
            ("tio", 16.0),
            ("for", 15.0),
            ("ate", 14.0),
            ("hat", 13.0),
            ("tha", 13.0),
            ("ere", 12.0),
            ("his", 12.0),
            ("hin", 11.0),
            ("ter", 11.0),
            ("was", 10.0),
            ("all", 10.0),
            ("ith", 9.0),
            ("ver", 9.0),
        ]),
    );

    // Spanish trigrams.
    profiles.insert(
        DetectedLanguage::Es,
        build_ref_profile(&[
            ("que", 45.0),
            ("ent", 30.0),
            ("los", 28.0),
            ("ion", 25.0),
            ("aci", 22.0),
            ("cion", 20.0),
            ("del", 19.0),
            ("las", 18.0),
            ("con", 17.0),
            ("est", 16.0),
            ("por", 15.0),
            ("nte", 14.0),
            ("ado", 13.0),
            ("una", 13.0),
            ("tra", 12.0),
            ("par", 11.0),
            ("com", 10.0),
            ("ero", 10.0),
            ("ien", 9.0),
            ("sta", 9.0),
        ]),
    );

    // French trigrams.
    profiles.insert(
        DetectedLanguage::Fr,
        build_ref_profile(&[
            ("les", 45.0),
            ("ent", 35.0),
            ("que", 30.0),
            ("des", 28.0),
            ("ion", 25.0),
            ("ait", 22.0),
            ("ous", 20.0),
            ("est", 18.0),
            ("une", 17.0),
            ("ant", 16.0),
            ("par", 15.0),
            ("eur", 14.0),
            ("sur", 13.0),
            ("tre", 12.0),
            ("eme", 11.0),
            ("dan", 10.0),
            ("pas", 10.0),
            ("tio", 9.0),
            ("pou", 9.0),
            ("ais", 8.0),
        ]),
    );

    // German trigrams.
    profiles.insert(
        DetectedLanguage::De,
        build_ref_profile(&[
            ("ein", 45.0),
            ("ich", 40.0),
            ("der", 35.0),
            ("die", 33.0),
            ("und", 30.0),
            ("den", 25.0),
            ("sch", 23.0),
            ("cht", 20.0),
            ("ung", 18.0),
            ("gen", 17.0),
            ("ber", 16.0),
            ("ver", 15.0),
            ("auf", 14.0),
            ("eit", 13.0),
            ("ach", 12.0),
            ("mit", 11.0),
            ("aus", 10.0),
            ("ine", 10.0),
            ("das", 9.0),
            ("ent", 8.0),
        ]),
    );

    // Italian trigrams.
    profiles.insert(
        DetectedLanguage::It,
        build_ref_profile(&[
            ("che", 45.0),
            ("ell", 30.0),
            ("per", 28.0),
            ("del", 25.0),
            ("ato", 22.0),
            ("ion", 20.0),
            ("ent", 18.0),
            ("con", 17.0),
            ("lla", 16.0),
            ("azi", 15.0),
            ("tta", 14.0),
            ("gli", 13.0),
            ("sta", 12.0),
            ("nte", 11.0),
            ("one", 10.0),
            ("ere", 10.0),
            ("tto", 9.0),
            ("ato", 9.0),
            ("ment", 8.0),
            ("pre", 8.0),
        ]),
    );

    // Portuguese trigrams.
    profiles.insert(
        DetectedLanguage::Pt,
        build_ref_profile(&[
            ("que", 45.0),
            ("ent", 30.0),
            ("nte", 25.0),
            ("ado", 22.0),
            ("ica", 20.0),
            ("est", 18.0),
            ("dos", 17.0),
            ("con", 16.0),
            ("par", 15.0),
            ("men", 14.0),
            ("com", 13.0),
            ("aco", 12.0),
            ("tra", 11.0),
            ("ida", 10.0),
            ("pro", 10.0),
            ("uma", 9.0),
            ("mos", 9.0),
            ("oes", 8.0),
            ("ter", 8.0),
            ("ais", 7.0),
        ]),
    );

    // Dutch trigrams.
    profiles.insert(
        DetectedLanguage::Nl,
        build_ref_profile(&[
            ("een", 45.0),
            ("van", 40.0),
            ("het", 35.0),
            ("aar", 28.0),
            ("ing", 25.0),
            ("oor", 22.0),
            ("ver", 20.0),
            ("den", 18.0),
            ("ijk", 16.0),
            ("ond", 15.0),
            ("ent", 14.0),
            ("erd", 13.0),
            ("sch", 12.0),
            ("ter", 11.0),
            ("and", 10.0),
            ("ede", 10.0),
            ("aat", 9.0),
            ("met", 9.0),
            ("nde", 8.0),
            ("dat", 8.0),
        ]),
    );

    // Turkish trigrams.
    profiles.insert(
        DetectedLanguage::Tr,
        build_ref_profile(&[
            ("lar", 45.0),
            ("bir", 40.0),
            ("ler", 35.0),
            ("eri", 30.0),
            ("ara", 25.0),
            ("ini", 22.0),
            ("rin", 20.0),
            ("yor", 18.0),
            ("ile", 16.0),
            ("dir", 15.0),
            ("dan", 14.0),
            ("rak", 13.0),
            ("len", 12.0),
            ("ası", 11.0),
            ("lik", 10.0),
            ("olu", 10.0),
            ("ind", 9.0),
            ("yan", 9.0),
            ("ama", 8.0),
            ("aki", 8.0),
        ]),
    );

    profiles
}

fn build_ref_profile(data: &[(&str, f64)]) -> HashMap<String, f64> {
    let total: f64 = data.iter().map(|(_, f)| f).sum();
    let mut profile = HashMap::new();
    for (gram, freq) in data {
        profile.insert(gram.to_string(), freq / total);
    }
    profile
}

// ---------------------------------------------------------------------------
// Common-word frequency analysis
// ---------------------------------------------------------------------------

fn detect_by_word_frequency(text: &str) -> Result<LanguageDetectionOutput> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.is_empty() {
        return Ok(unknown_result());
    }

    let word_lists = common_word_lists();
    let mut scores: Vec<(DetectedLanguage, f64)> = Vec::new();

    for (lang, common_words) in &word_lists {
        let matches = words.iter().filter(|w| common_words.contains(*w)).count();
        let ratio = matches as f64 / words.len() as f64;
        scores.push((*lang, ratio));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if scores.is_empty() || scores[0].1 < 0.01 {
        return Ok(unknown_result());
    }

    let best = scores[0];
    let confidence = (best.1 * 2.5).clamp(0.0, 1.0);

    Ok(LanguageDetectionOutput {
        language: best.0,
        confidence,
        alternatives: scores.into_iter().skip(1).collect(),
    })
}

fn common_word_lists() -> HashMap<DetectedLanguage, Vec<&'static str>> {
    let mut lists = HashMap::new();

    lists.insert(
        DetectedLanguage::En,
        vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when",
        ],
    );

    lists.insert(
        DetectedLanguage::Es,
        vec![
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
            "para", "con", "no", "una", "su", "al", "es", "lo", "como", "pero", "sus", "le", "ya",
            "o", "este", "ha", "si", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
            "ser", "tambien", "me", "hasta", "hay", "donde", "quien",
        ],
    );

    lists.insert(
        DetectedLanguage::Fr,
        vec![
            "de", "la", "le", "et", "les", "des", "en", "un", "du", "une", "que", "est", "dans",
            "qui", "par", "pour", "au", "il", "sur", "pas", "plus", "ce", "ne", "se", "avec",
            "mais", "on", "son", "tout", "je", "nous", "vous", "elle", "ou", "bien", "ces", "sont",
            "sans", "comme", "peut", "fait", "aux", "entre", "deux",
        ],
    );

    lists.insert(
        DetectedLanguage::De,
        vec![
            "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf",
            "nicht", "ein", "ist", "dem", "eine", "auch", "es", "an", "als", "nach", "wie", "aber",
            "vor", "hat", "nur", "oder", "ich", "bei", "noch", "unter", "bis", "kann", "wird",
            "so", "wenn", "sie", "sehr", "wir", "uber", "schon", "dann",
        ],
    );

    lists.insert(
        DetectedLanguage::It,
        vec![
            "di", "che", "il", "la", "in", "un", "per", "del", "non", "una", "con", "sono", "gli",
            "le", "si", "da", "al", "lo", "ha", "come", "ma", "anche", "io", "suo", "dei", "nel",
            "alla", "piu", "questo", "era", "essere", "tutto", "fra", "stato", "ancora", "dove",
            "hanno", "ogni", "alle", "nella",
        ],
    );

    lists.insert(
        DetectedLanguage::Pt,
        vec![
            "de", "que", "o", "a", "do", "da", "em", "para", "com", "um", "uma", "os", "no", "se",
            "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele", "das", "seu", "sua",
            "ou", "quando", "muito", "nos", "ja", "eu", "tambem", "so", "pelo", "pela", "ate",
            "isso", "ela", "entre", "depois", "sem", "mesmo",
        ],
    );

    lists.insert(
        DetectedLanguage::Nl,
        vec![
            "de", "het", "een", "van", "en", "in", "is", "dat", "op", "te", "zijn", "voor", "met",
            "die", "niet", "aan", "er", "maar", "om", "ook", "als", "dan", "bij", "nog", "uit",
            "kan", "al", "wel", "zo", "was", "worden", "tot", "naar", "heeft", "over", "meer",
            "hun", "dit", "door", "onder", "heel", "deze", "dus",
        ],
    );

    lists.insert(
        DetectedLanguage::Tr,
        vec![
            "bir", "bu", "da", "ve", "ile", "olan", "icin", "var", "ama", "den", "daha", "gibi",
            "sonra", "kadar", "olarak", "hem", "her", "ya", "mi", "ne", "ben", "sen", "biz", "siz",
            "o", "onlar", "ise", "ancak", "yok", "cok",
        ],
    );

    lists
}

// ---------------------------------------------------------------------------
// Unicode script detection
// ---------------------------------------------------------------------------

fn detect_by_unicode_script(text: &str) -> Result<LanguageDetectionOutput> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return Ok(unknown_result());
    }

    let total = chars.len() as f64;
    let mut script_counts: HashMap<&str, usize> = HashMap::new();

    for &ch in &chars {
        let script = classify_char(ch);
        *script_counts.entry(script).or_insert(0) += 1;
    }

    // Map scripts to languages.
    let mut lang_scores: HashMap<DetectedLanguage, f64> = HashMap::new();

    if let Some(&count) = script_counts.get("cjk") {
        // Distinguish Chinese, Japanese, Korean by auxiliary scripts.
        let hiragana = *script_counts.get("hiragana").unwrap_or(&0) as f64;
        let katakana = *script_counts.get("katakana").unwrap_or(&0) as f64;
        let hangul = *script_counts.get("hangul").unwrap_or(&0) as f64;

        if hiragana + katakana > hangul {
            *lang_scores.entry(DetectedLanguage::Ja).or_insert(0.0) +=
                (count as f64 + hiragana + katakana) / total;
        } else if hangul > 0.0 {
            *lang_scores.entry(DetectedLanguage::Ko).or_insert(0.0) +=
                (count as f64 + hangul) / total;
        } else {
            *lang_scores.entry(DetectedLanguage::Zh).or_insert(0.0) += count as f64 / total;
        }
    }

    if let Some(&count) = script_counts.get("hiragana") {
        *lang_scores.entry(DetectedLanguage::Ja).or_insert(0.0) += count as f64 / total;
    }
    if let Some(&count) = script_counts.get("katakana") {
        *lang_scores.entry(DetectedLanguage::Ja).or_insert(0.0) += count as f64 / total;
    }
    if let Some(&count) = script_counts.get("hangul") {
        *lang_scores.entry(DetectedLanguage::Ko).or_insert(0.0) += count as f64 / total;
    }
    if let Some(&count) = script_counts.get("cyrillic") {
        *lang_scores.entry(DetectedLanguage::Ru).or_insert(0.0) += count as f64 / total;
    }
    if let Some(&count) = script_counts.get("arabic") {
        *lang_scores.entry(DetectedLanguage::Ar).or_insert(0.0) += count as f64 / total;
    }
    if let Some(&count) = script_counts.get("devanagari") {
        *lang_scores.entry(DetectedLanguage::Hi).or_insert(0.0) += count as f64 / total;
    }

    // Latin script -> delegate to n-gram for Latin-script languages.
    if let Some(&count) = script_counts.get("latin") {
        let latin_ratio = count as f64 / total;
        if latin_ratio > 0.5 {
            // Fall back to ngram for Latin languages.
            return detect_by_ngram(text);
        }
    }

    let mut scores: Vec<(DetectedLanguage, f64)> = lang_scores.into_iter().collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if scores.is_empty() {
        return Ok(unknown_result());
    }

    let best = scores[0];
    let confidence = best.1.clamp(0.0, 1.0);

    Ok(LanguageDetectionOutput {
        language: best.0,
        confidence,
        alternatives: scores.into_iter().skip(1).collect(),
    })
}

/// Classify a character into a script category.
fn classify_char(ch: char) -> &'static str {
    let code = ch as u32;
    match code {
        // Basic Latin / Latin Extended
        0x0041..=0x024F => "latin",
        // Latin Extended Additional
        0x1E00..=0x1EFF => "latin",
        // Cyrillic
        0x0400..=0x052F => "cyrillic",
        // Arabic
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => "arabic",
        // Devanagari
        0x0900..=0x097F => "devanagari",
        // CJK Unified Ideographs
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF => "cjk",
        // Hiragana
        0x3040..=0x309F => "hiragana",
        // Katakana
        0x30A0..=0x30FF | 0x31F0..=0x31FF => "katakana",
        // Hangul
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => "hangul",
        // Thai
        0x0E00..=0x0E7F => "thai",
        // Greek
        0x0370..=0x03FF => "greek",
        // Hebrew
        0x0590..=0x05FF => "hebrew",
        _ => "other",
    }
}

// ---------------------------------------------------------------------------
// Combined detection
// ---------------------------------------------------------------------------

fn detect_combined(text: &str) -> Result<LanguageDetectionOutput> {
    // 1. Try Unicode script first (fast, decisive for non-Latin scripts).
    let script_result = detect_by_unicode_script(text)?;
    if script_result.language != DetectedLanguage::Unknown && script_result.confidence > 0.6 {
        return Ok(script_result);
    }

    // 2. For Latin-script text, combine n-gram and word frequency.
    let ngram_result = detect_by_ngram(text)?;
    let word_result = detect_by_word_frequency(text)?;

    // Merge scores (weighted average).
    let mut combined: HashMap<DetectedLanguage, f64> = HashMap::new();
    let ngram_weight = 0.55;
    let word_weight = 0.45;

    // Add n-gram scores.
    *combined.entry(ngram_result.language).or_insert(0.0) += ngram_weight * ngram_result.confidence;
    for (lang, score) in &ngram_result.alternatives {
        *combined.entry(*lang).or_insert(0.0) += ngram_weight * score;
    }

    // Add word frequency scores.
    *combined.entry(word_result.language).or_insert(0.0) += word_weight * word_result.confidence;
    for (lang, score) in &word_result.alternatives {
        *combined.entry(*lang).or_insert(0.0) += word_weight * score;
    }

    let mut scores: Vec<(DetectedLanguage, f64)> = combined.into_iter().collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if scores.is_empty() {
        return Ok(unknown_result());
    }

    let best = scores[0];
    let confidence = best.1.clamp(0.0, 1.0);

    Ok(LanguageDetectionOutput {
        language: best.0,
        confidence,
        alternatives: scores.into_iter().skip(1).collect(),
    })
}

fn unknown_result() -> LanguageDetectionOutput {
    LanguageDetectionOutput {
        language: DetectedLanguage::Unknown,
        confidence: 0.0,
        alternatives: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- English detection ----

    #[test]
    fn test_detect_english() {
        let result = detect_language(
            "The quick brown fox jumps over the lazy dog. This is a test of the language detection system.",
        )
        .expect("Should succeed");
        assert_eq!(result.language, DetectedLanguage::En);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_detect_english_short() {
        let result = detect_language("Hello world, how are you today?").expect("Should succeed");
        assert_eq!(result.language, DetectedLanguage::En);
    }

    #[test]
    fn test_detect_english_ngram_strategy() {
        let result = detect_language_with_strategy(
            "The weather is wonderful and everything looks beautiful in the morning light.",
            DetectionStrategy::Ngram,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::En);
    }

    #[test]
    fn test_detect_english_word_frequency() {
        let result = detect_language_with_strategy(
            "This is a test of the word frequency detection method.",
            DetectionStrategy::WordFrequency,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::En);
    }

    #[test]
    fn test_english_has_alternatives() {
        let result = detect_language(
            "The system provides comprehensive analysis and detailed reporting for all users.",
        )
        .expect("ok");
        assert!(!result.alternatives.is_empty());
    }

    // ---- Spanish detection ----

    #[test]
    fn test_detect_spanish() {
        let result = detect_language(
            "El gato se sienta en la alfombra. Esta es una prueba del sistema de deteccion de idioma.",
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Es);
    }

    #[test]
    fn test_detect_spanish_ngram() {
        let result = detect_language_with_strategy(
            "Los estudiantes que asistieron a la conferencia disfrutaron de las presentaciones.",
            DetectionStrategy::Ngram,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Es);
    }

    #[test]
    fn test_detect_spanish_word_frequency() {
        let result = detect_language_with_strategy(
            "Para los que no saben, el libro es una de las mejores novelas del siglo.",
            DetectionStrategy::WordFrequency,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Es);
    }

    #[test]
    fn test_detect_spanish_combined() {
        let result = detect_language_with_strategy(
            "La empresa ha contratado a nuevos empleados para el departamento de marketing.",
            DetectionStrategy::Combined,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Es);
    }

    #[test]
    fn test_spanish_confidence_range() {
        let result =
            detect_language("Buenos dias, como estas? Espero que todo vaya bien con la familia.")
                .expect("ok");
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    // ---- French detection ----

    #[test]
    fn test_detect_french() {
        let result = detect_language(
            "Le chat est assis sur le tapis. Les enfants jouent dans le jardin avec leurs amis.",
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Fr);
    }

    #[test]
    fn test_detect_french_ngram() {
        let result = detect_language_with_strategy(
            "Les resultats des elections ont ete publies dans les journaux ce matin.",
            DetectionStrategy::Ngram,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Fr);
    }

    #[test]
    fn test_detect_french_word() {
        let result = detect_language_with_strategy(
            "Je ne suis pas sur que nous puissions terminer ce projet dans les delais prevus.",
            DetectionStrategy::WordFrequency,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Fr);
    }

    #[test]
    fn test_french_confidence() {
        let result = detect_language("Bonjour, comment allez-vous? Je suis content de vous voir.")
            .expect("ok");
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_detect_french_combined() {
        let result = detect_language(
            "Les entreprises francaises investissent dans les nouvelles technologies pour une meilleure productivite.",
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Fr);
    }

    // ---- German detection ----

    #[test]
    fn test_detect_german() {
        let result = detect_language(
            "Die Katze sitzt auf der Matte. Die Kinder spielen im Garten mit ihren Freunden.",
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::De);
    }

    #[test]
    fn test_detect_german_word() {
        let result = detect_language_with_strategy(
            "Ich bin nicht sicher, ob wir dieses Projekt noch rechtzeitig fertigstellen werden.",
            DetectionStrategy::WordFrequency,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::De);
    }

    #[test]
    fn test_detect_german_ngram() {
        let result = detect_language_with_strategy(
            "Die Ergebnisse der Untersuchung wurden gestern veroffentlicht und haben grosse Aufmerksamkeit erregt.",
            DetectionStrategy::Ngram,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::De);
    }

    #[test]
    fn test_german_confidence() {
        let result = detect_language("Guten Tag, wie geht es Ihnen? Ich hoffe, es geht Ihnen gut.")
            .expect("ok");
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_detect_german_combined() {
        let result = detect_language(
            "Die Wissenschaftler haben einen wichtigen Durchbruch in der Forschung erzielt.",
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::De);
    }

    // ---- CJK / non-Latin script detection ----

    #[test]
    fn test_detect_chinese() {
        let result =
            detect_language("今天天气很好，我们去公园散步吧。这是一个美丽的城市。").expect("ok");
        assert_eq!(result.language, DetectedLanguage::Zh);
    }

    #[test]
    fn test_detect_japanese() {
        let result =
            detect_language("今日はとてもいい天気です。公園で散歩しましょう。").expect("ok");
        assert_eq!(result.language, DetectedLanguage::Ja);
    }

    #[test]
    fn test_detect_korean() {
        let result =
            detect_language("오늘 날씨가 정말 좋습니다. 공원에서 산책합시다.").expect("ok");
        assert_eq!(result.language, DetectedLanguage::Ko);
    }

    #[test]
    fn test_detect_russian() {
        let result = detect_language("Сегодня прекрасная погода. Давайте пойдем гулять в парк.")
            .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Ru);
    }

    #[test]
    fn test_detect_arabic() {
        let result = detect_language("الطقس جميل اليوم. دعونا نذهب للمشي في الحديقة.").expect("ok");
        assert_eq!(result.language, DetectedLanguage::Ar);
    }

    // ---- Edge cases ----

    #[test]
    fn test_empty_text_error() {
        let result = detect_language("");
        assert!(result.is_err());
    }

    #[test]
    fn test_whitespace_only_error() {
        let result = detect_language("   \t\n  ");
        assert!(result.is_err());
    }

    #[test]
    fn test_very_short_text() {
        // Very short text may have low confidence but should not error.
        let result = detect_language("Hi").expect("ok");
        // Confidence may be low but it should return something.
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_iso_code_round_trip() {
        let lang = DetectedLanguage::En;
        assert_eq!(lang.iso_code(), "en");
        assert_eq!(lang.name(), "English");
    }

    #[test]
    fn test_unknown_iso_code() {
        let lang = DetectedLanguage::Unknown;
        assert_eq!(lang.iso_code(), "und");
    }

    // ---- Unicode script strategy tests ----

    #[test]
    fn test_unicode_script_cjk() {
        let result =
            detect_language_with_strategy("这是一个测试。", DetectionStrategy::UnicodeScript)
                .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Zh);
    }

    #[test]
    fn test_unicode_script_cyrillic() {
        let result = detect_language_with_strategy(
            "Привет мир, как дела?",
            DetectionStrategy::UnicodeScript,
        )
        .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Ru);
    }

    #[test]
    fn test_unicode_script_arabic() {
        let result =
            detect_language_with_strategy("مرحبا بالعالم", DetectionStrategy::UnicodeScript)
                .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Ar);
    }

    #[test]
    fn test_unicode_script_devanagari() {
        let result =
            detect_language_with_strategy("नमस्ते दुनिया, कैसे हो?", DetectionStrategy::UnicodeScript)
                .expect("ok");
        assert_eq!(result.language, DetectedLanguage::Hi);
    }

    #[test]
    fn test_unicode_script_latin_falls_back() {
        // Latin text should fall back to n-gram detection within unicode_script strategy.
        let result = detect_language_with_strategy(
            "The quick brown fox jumps over the lazy dog.",
            DetectionStrategy::UnicodeScript,
        )
        .expect("ok");
        // Should still detect English.
        assert_eq!(result.language, DetectedLanguage::En);
    }
}
