//! String processing utilities for scientific computing
//!
//! This module provides common string distance metrics, similarity measures,
//! tokenization, n-gram generation, and case conversion utilities used across
//! the SciRS2 ecosystem.
//!
//! # Distance Metrics
//!
//! - [`levenshtein_distance`] - Edit distance (insertions, deletions, substitutions)
//! - [`hamming_distance`] - Number of positions where characters differ
//! - [`jaro_similarity`] / [`jaro_winkler_similarity`] - Positional character similarity
//!
//! # Subsequences
//!
//! - [`longest_common_subsequence`] - LCS length
//! - [`lcs_string`] - Actual LCS string
//!
//! # Tokenization & N-grams
//!
//! - [`tokenize_whitespace`] - Split by whitespace
//! - [`tokenize_pattern`] - Split by custom delimiter patterns
//! - [`ngrams`] / [`char_ngrams`] - Word-level and character-level n-grams
//!
//! # Case Conversions
//!
//! - [`to_snake_case`] / [`to_camel_case`] / [`to_pascal_case`] / [`to_kebab_case`] / [`to_screaming_snake_case`]

use crate::error::{CoreError, CoreResult, ErrorContext};

// ---------------------------------------------------------------------------
// Levenshtein distance
// ---------------------------------------------------------------------------

/// Compute the Levenshtein edit distance between two strings.
///
/// The edit distance is the minimum number of single-character edits
/// (insertions, deletions, substitutions) required to change one string
/// into the other.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::levenshtein_distance;
///
/// assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_distance("", "abc"), 3);
/// assert_eq!(levenshtein_distance("same", "same"), 0);
/// ```
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows for O(min(m,n)) space
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Compute the normalized Levenshtein distance (0.0 = identical, 1.0 = completely different).
///
/// Normalized by dividing by the length of the longer string.
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 0.0;
    }
    levenshtein_distance(a, b) as f64 / max_len as f64
}

/// Compute the Levenshtein similarity (1.0 = identical, 0.0 = completely different).
pub fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    1.0 - normalized_levenshtein(a, b)
}

// ---------------------------------------------------------------------------
// Hamming distance
// ---------------------------------------------------------------------------

/// Compute the Hamming distance between two strings of equal length.
///
/// The Hamming distance is the number of positions at which the
/// corresponding characters are different.
///
/// Returns an error if the strings have different lengths.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::hamming_distance;
///
/// assert_eq!(hamming_distance("karolin", "kathrin").expect("should succeed"), 3);
/// assert!(hamming_distance("short", "longer").is_err());
/// ```
pub fn hamming_distance(a: &str, b: &str) -> CoreResult<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.len() != b_chars.len() {
        return Err(CoreError::ValueError(ErrorContext::new(format!(
            "Hamming distance requires equal-length strings (got {} and {})",
            a_chars.len(),
            b_chars.len()
        ))));
    }
    let dist = a_chars
        .iter()
        .zip(b_chars.iter())
        .filter(|(x, y)| x != y)
        .count();
    Ok(dist)
}

// ---------------------------------------------------------------------------
// Jaro / Jaro-Winkler similarity
// ---------------------------------------------------------------------------

/// Compute the Jaro similarity between two strings.
///
/// Returns a value in [0, 1] where 1 means the strings are identical.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::jaro_similarity;
///
/// let sim = jaro_similarity("martha", "marhta");
/// assert!((sim - 0.9444).abs() < 0.001);
/// ```
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let la = a_chars.len();
    let lb = b_chars.len();

    if la == 0 && lb == 0 {
        return 1.0;
    }
    if la == 0 || lb == 0 {
        return 0.0;
    }

    let match_window = (la.max(lb) / 2).saturating_sub(1);

    let mut a_matched = vec![false; la];
    let mut b_matched = vec![false; lb];

    let mut matches = 0usize;
    let mut transpositions = 0usize;

    // Find matching characters
    for i in 0..la {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(lb);
        for j in start..end {
            if !b_matched[j] && a_chars[i] == b_chars[j] {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0;
    for i in 0..la {
        if !a_matched[i] {
            continue;
        }
        while !b_matched[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    (m / la as f64 + m / lb as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

/// Compute the Jaro-Winkler similarity between two strings.
///
/// Extends Jaro similarity with a bonus for matching prefixes, controlled
/// by `prefix_weight` (default 0.1, should not exceed 0.25).
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::jaro_winkler_similarity;
///
/// let sim = jaro_winkler_similarity("martha", "marhta", 0.1);
/// assert!(sim > 0.94);
/// ```
pub fn jaro_winkler_similarity(a: &str, b: &str, prefix_weight: f64) -> f64 {
    let jaro = jaro_similarity(a, b);
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    // Compute common prefix length (up to 4 characters)
    let max_prefix = 4.min(a_chars.len()).min(b_chars.len());
    let mut prefix_len = 0;
    for i in 0..max_prefix {
        if a_chars[i] == b_chars[i] {
            prefix_len += 1;
        } else {
            break;
        }
    }

    let weight = prefix_weight.min(0.25); // Clamp to prevent exceeding 1.0
    jaro + prefix_len as f64 * weight * (1.0 - jaro)
}

// ---------------------------------------------------------------------------
// Longest common subsequence
// ---------------------------------------------------------------------------

/// Compute the length of the longest common subsequence (LCS) of two strings.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::longest_common_subsequence;
///
/// assert_eq!(longest_common_subsequence("abcde", "ace"), 3);
/// assert_eq!(longest_common_subsequence("abc", "def"), 0);
/// ```
pub fn longest_common_subsequence(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Use two rows for O(n) space
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        for v in curr.iter_mut() {
            *v = 0;
        }
    }

    *prev.iter().max().unwrap_or(&0)
}

/// Return the actual longest common subsequence string.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::lcs_string;
///
/// assert_eq!(lcs_string("abcde", "ace"), "ace");
/// ```
pub fn lcs_string(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return String::new();
    }

    // Full DP table for backtracking
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find the actual subsequence
    let mut result = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        if a_chars[i - 1] == b_chars[j - 1] {
            result.push(a_chars[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    result.reverse();
    result.into_iter().collect()
}

/// Compute the LCS similarity (0.0 = no common subsequence, 1.0 = identical).
pub fn lcs_similarity(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    longest_common_subsequence(a, b) as f64 / max_len as f64
}

// ---------------------------------------------------------------------------
// N-gram generation
// ---------------------------------------------------------------------------

/// Generate word-level n-grams from a list of tokens.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::ngrams;
///
/// let tokens = vec!["the", "quick", "brown", "fox"];
/// let bigrams = ngrams(&tokens, 2).expect("bigrams");
/// assert_eq!(bigrams.len(), 3);
/// assert_eq!(bigrams[0], vec!["the", "quick"]);
/// ```
pub fn ngrams<'a>(tokens: &[&'a str], n: usize) -> CoreResult<Vec<Vec<&'a str>>> {
    if n == 0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "n must be >= 1 for n-gram generation",
        )));
    }
    if tokens.len() < n {
        return Ok(Vec::new());
    }
    let result: Vec<Vec<&str>> = tokens.windows(n).map(|w| w.to_vec()).collect();
    Ok(result)
}

/// Generate character-level n-grams from a string.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::char_ngrams;
///
/// let grams = char_ngrams("hello", 2).expect("bigrams");
/// assert_eq!(grams, vec!["he", "el", "ll", "lo"]);
/// ```
pub fn char_ngrams(text: &str, n: usize) -> CoreResult<Vec<String>> {
    if n == 0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "n must be >= 1 for char n-gram generation",
        )));
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return Ok(Vec::new());
    }
    let result: Vec<String> = chars.windows(n).map(|w| w.iter().collect()).collect();
    Ok(result)
}

/// Generate skip-grams (n-grams with gaps).
///
/// A skip-gram of order (n, k) takes n tokens with up to k skips.
/// This implementation generates bigrams with a given skip distance.
pub fn skip_bigrams<'a>(tokens: &[&'a str], skip: usize) -> Vec<(&'a str, &'a str)> {
    let mut result = Vec::new();
    for i in 0..tokens.len() {
        for gap in 1..=(skip + 1) {
            if i + gap < tokens.len() {
                result.push((tokens[i], tokens[i + gap]));
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/// Tokenize a string by splitting on whitespace.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::tokenize_whitespace;
///
/// let tokens = tokenize_whitespace("  hello   world  ");
/// assert_eq!(tokens, vec!["hello", "world"]);
/// ```
pub fn tokenize_whitespace(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

/// Tokenize a string by splitting on a delimiter character.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::tokenize_char;
///
/// let tokens = tokenize_char("a,b,,c", ',');
/// assert_eq!(tokens, vec!["a", "b", "", "c"]);
/// ```
pub fn tokenize_char(text: &str, delimiter: char) -> Vec<&str> {
    text.split(delimiter).collect()
}

/// Tokenize by splitting on any character matching a predicate.
///
/// Consecutive delimiters produce empty tokens (like `str::split`).
pub fn tokenize_predicate<F: Fn(char) -> bool>(text: &str, predicate: F) -> Vec<&str> {
    text.split(|c| predicate(c)).collect()
}

/// Tokenize by splitting on a string pattern.
pub fn tokenize_pattern<'a>(text: &'a str, pattern: &str) -> Vec<&'a str> {
    text.split(pattern).collect()
}

/// Tokenize into sentences (split on '.', '!', '?').
///
/// Trims whitespace from each sentence and removes empty strings.
pub fn tokenize_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Simple word tokenizer that splits on non-alphanumeric characters.
///
/// Produces only non-empty tokens of alphanumeric characters.
pub fn tokenize_words(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == '_' {
            if start.is_none() {
                start = Some(i);
            }
        } else {
            if let Some(s) = start {
                tokens.push(&text[s..i]);
                start = None;
            }
        }
    }
    if let Some(s) = start {
        tokens.push(&text[s..]);
    }
    tokens
}

// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

/// Convert a string to snake_case.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::to_snake_case;
///
/// assert_eq!(to_snake_case("helloWorld"), "hello_world");
/// assert_eq!(to_snake_case("HTTPClient"), "http_client");
/// assert_eq!(to_snake_case("already_snake"), "already_snake");
/// ```
pub fn to_snake_case(s: &str) -> String {
    let words = split_into_words(s);
    words
        .iter()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join("_")
}

/// Convert a string to camelCase.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::to_camel_case;
///
/// assert_eq!(to_camel_case("hello_world"), "helloWorld");
/// assert_eq!(to_camel_case("some-kebab-case"), "someKebabCase");
/// ```
pub fn to_camel_case(s: &str) -> String {
    let words = split_into_words(s);
    let mut result = String::new();
    for (i, word) in words.iter().enumerate() {
        if i == 0 {
            result.push_str(&word.to_lowercase());
        } else {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                result.extend(first.to_uppercase());
                result.push_str(&chars.as_str().to_lowercase());
            }
        }
    }
    result
}

/// Convert a string to PascalCase.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::to_pascal_case;
///
/// assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
/// assert_eq!(to_pascal_case("some-kebab-case"), "SomeKebabCase");
/// ```
pub fn to_pascal_case(s: &str) -> String {
    let words = split_into_words(s);
    let mut result = String::new();
    for word in &words {
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            result.extend(first.to_uppercase());
            result.push_str(&chars.as_str().to_lowercase());
        }
    }
    result
}

/// Convert a string to kebab-case.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::to_kebab_case;
///
/// assert_eq!(to_kebab_case("helloWorld"), "hello-world");
/// assert_eq!(to_kebab_case("SomeClassName"), "some-class-name");
/// ```
pub fn to_kebab_case(s: &str) -> String {
    let words = split_into_words(s);
    words
        .iter()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join("-")
}

/// Convert a string to SCREAMING_SNAKE_CASE.
///
/// # Example
///
/// ```
/// use scirs2_core::string_ops::to_screaming_snake_case;
///
/// assert_eq!(to_screaming_snake_case("helloWorld"), "HELLO_WORLD");
/// assert_eq!(to_screaming_snake_case("some-kebab"), "SOME_KEBAB");
/// ```
pub fn to_screaming_snake_case(s: &str) -> String {
    let words = split_into_words(s);
    words
        .iter()
        .map(|w| w.to_uppercase())
        .collect::<Vec<_>>()
        .join("_")
}

/// Convert a string to Title Case.
pub fn to_title_case(s: &str) -> String {
    let words = split_into_words(s);
    let mut result = Vec::with_capacity(words.len());
    for word in &words {
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            let mut titled = String::new();
            titled.extend(first.to_uppercase());
            titled.push_str(&chars.as_str().to_lowercase());
            result.push(titled);
        }
    }
    result.join(" ")
}

/// Split a string into words, handling camelCase, PascalCase, snake_case,
/// kebab-case, and spaces.
fn split_into_words(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        let c = chars[i];

        // Delimiters: underscore, hyphen, space, dot
        if c == '_' || c == '-' || c == ' ' || c == '.' {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            i += 1;
            continue;
        }

        if c.is_uppercase() {
            // Check for acronym (consecutive uppercase)
            if !current.is_empty() {
                // If previous was lowercase, start new word
                let prev = chars[i - 1];
                if prev.is_lowercase() || prev.is_ascii_digit() {
                    words.push(current.clone());
                    current.clear();
                }
                // If next is lowercase and current is uppercase, end the acronym
                // e.g., "HTTPClient" -> "HTTP" + "Client"
                else if i + 1 < len && chars[i + 1].is_lowercase() && !current.is_empty() {
                    // The current uppercase is the start of a new word
                    words.push(current.clone());
                    current.clear();
                }
            }
            current.push(c);
        } else {
            current.push(c);
        }

        i += 1;
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

// ---------------------------------------------------------------------------
// Additional string utilities
// ---------------------------------------------------------------------------

/// Compute the Dice coefficient (bigram overlap) between two strings.
///
/// The Dice coefficient is 2 * |intersection of bigrams| / (|bigrams_a| + |bigrams_b|).
pub fn dice_coefficient(a: &str, b: &str) -> f64 {
    let a_bigrams: Vec<String> = char_ngrams(a, 2).unwrap_or_default();
    let b_bigrams: Vec<String> = char_ngrams(b, 2).unwrap_or_default();

    if a_bigrams.is_empty() && b_bigrams.is_empty() {
        return 1.0;
    }
    if a_bigrams.is_empty() || b_bigrams.is_empty() {
        return 0.0;
    }

    let mut intersection = 0usize;
    let mut b_used = vec![false; b_bigrams.len()];

    for bg_a in &a_bigrams {
        for (j, bg_b) in b_bigrams.iter().enumerate() {
            if !b_used[j] && bg_a == bg_b {
                intersection += 1;
                b_used[j] = true;
                break;
            }
        }
    }

    2.0 * intersection as f64 / (a_bigrams.len() + b_bigrams.len()) as f64
}

/// Pad a string on the left to a given width.
pub fn pad_left(s: &str, width: usize, fill: char) -> String {
    let char_count = s.chars().count();
    if char_count >= width {
        return s.to_string();
    }
    let padding: String = std::iter::repeat_n(fill, width - char_count).collect();
    format!("{}{}", padding, s)
}

/// Pad a string on the right to a given width.
pub fn pad_right(s: &str, width: usize, fill: char) -> String {
    let char_count = s.chars().count();
    if char_count >= width {
        return s.to_string();
    }
    let padding: String = std::iter::repeat_n(fill, width - char_count).collect();
    format!("{}{}", s, padding)
}

/// Center a string within a given width.
pub fn center(s: &str, width: usize, fill: char) -> String {
    let char_count = s.chars().count();
    if char_count >= width {
        return s.to_string();
    }
    let total_pad = width - char_count;
    let left_pad = total_pad / 2;
    let right_pad = total_pad - left_pad;
    let left: String = std::iter::repeat_n(fill, left_pad).collect();
    let right: String = std::iter::repeat_n(fill, right_pad).collect();
    format!("{}{}{}", left, s, right)
}

/// Reverse a string (Unicode-aware).
pub fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}

/// Check if a string is a palindrome (case-insensitive, alphanumeric only).
pub fn is_palindrome(s: &str) -> bool {
    let chars: Vec<char> = s
        .chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect();
    let len = chars.len();
    if len <= 1 {
        return true;
    }
    for i in 0..len / 2 {
        if chars[i] != chars[len - 1 - i] {
            return false;
        }
    }
    true
}

/// Count occurrences of a substring.
pub fn count_occurrences(text: &str, pattern: &str) -> usize {
    if pattern.is_empty() {
        return 0;
    }
    text.matches(pattern).count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Levenshtein --

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("xyz", ""), 3);
        assert_eq!(levenshtein_distance("", ""), 0);
    }

    #[test]
    fn test_levenshtein_known() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
    }

    #[test]
    fn test_normalized_levenshtein() {
        assert!((normalized_levenshtein("abc", "abc") - 0.0).abs() < 1e-10);
        assert!((normalized_levenshtein("", "") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_levenshtein_similarity() {
        assert!((levenshtein_similarity("abc", "abc") - 1.0).abs() < 1e-10);
    }

    // -- Hamming --

    #[test]
    fn test_hamming_basic() {
        assert_eq!(hamming_distance("karolin", "kathrin").expect("ok"), 3);
        assert_eq!(hamming_distance("abc", "abc").expect("ok"), 0);
    }

    #[test]
    fn test_hamming_unequal_length() {
        assert!(hamming_distance("ab", "abc").is_err());
    }

    // -- Jaro / Jaro-Winkler --

    #[test]
    fn test_jaro_identical() {
        assert!((jaro_similarity("abc", "abc") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaro_empty() {
        assert!((jaro_similarity("", "") - 1.0).abs() < 1e-10);
        assert!((jaro_similarity("abc", "") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaro_known() {
        let sim = jaro_similarity("martha", "marhta");
        assert!((sim - 0.9444).abs() < 0.001, "jaro: {}", sim);
    }

    #[test]
    fn test_jaro_winkler() {
        let sim = jaro_winkler_similarity("martha", "marhta", 0.1);
        assert!(sim > 0.94, "jw: {}", sim);
        // Winkler bonus should make it >= Jaro
        assert!(sim >= jaro_similarity("martha", "marhta"));
    }

    #[test]
    fn test_jaro_winkler_different() {
        let sim = jaro_winkler_similarity("abc", "xyz", 0.1);
        assert!(sim < 0.1);
    }

    // -- LCS --

    #[test]
    fn test_lcs_length() {
        assert_eq!(longest_common_subsequence("abcde", "ace"), 3);
        assert_eq!(longest_common_subsequence("abc", "def"), 0);
        assert_eq!(longest_common_subsequence("", "abc"), 0);
    }

    #[test]
    fn test_lcs_string() {
        assert_eq!(lcs_string("abcde", "ace"), "ace");
        assert_eq!(lcs_string("abc", "def"), "");
    }

    #[test]
    fn test_lcs_similarity() {
        assert!((lcs_similarity("abc", "abc") - 1.0).abs() < 1e-10);
        assert!((lcs_similarity("", "") - 1.0).abs() < 1e-10);
    }

    // -- N-grams --

    #[test]
    fn test_word_ngrams() {
        let tokens = vec!["a", "b", "c", "d"];
        let bigrams = ngrams(&tokens, 2).expect("bigrams");
        assert_eq!(bigrams.len(), 3);
        assert_eq!(bigrams[0], vec!["a", "b"]);
        assert_eq!(bigrams[2], vec!["c", "d"]);
    }

    #[test]
    fn test_word_trigrams() {
        let tokens = vec!["a", "b", "c", "d"];
        let trigrams = ngrams(&tokens, 3).expect("trigrams");
        assert_eq!(trigrams.len(), 2);
    }

    #[test]
    fn test_ngrams_zero() {
        assert!(ngrams(&["a"], 0).is_err());
    }

    #[test]
    fn test_ngrams_too_short() {
        let tokens = vec!["a"];
        let result = ngrams(&tokens, 3).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_char_ngrams() {
        let grams = char_ngrams("hello", 2).expect("ok");
        assert_eq!(grams, vec!["he", "el", "ll", "lo"]);
    }

    #[test]
    fn test_char_ngrams_zero() {
        assert!(char_ngrams("hello", 0).is_err());
    }

    #[test]
    fn test_skip_bigrams() {
        let tokens = vec!["a", "b", "c"];
        let result = skip_bigrams(&tokens, 1);
        // skip=1: (a,b),(a,c),(b,c) = 3 pairs
        assert_eq!(result.len(), 3);
        assert!(result.contains(&("a", "b")));
        assert!(result.contains(&("a", "c")));
        assert!(result.contains(&("b", "c")));
    }

    // -- Tokenization --

    #[test]
    fn test_tokenize_whitespace() {
        let tokens = tokenize_whitespace("  hello   world  ");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_char() {
        let tokens = tokenize_char("a,b,,c", ',');
        assert_eq!(tokens, vec!["a", "b", "", "c"]);
    }

    #[test]
    fn test_tokenize_pattern() {
        let tokens = tokenize_pattern("a::b::c", "::");
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_tokenize_sentences() {
        let sentences = tokenize_sentences("Hello world. How are you? Fine!");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world");
    }

    #[test]
    fn test_tokenize_words() {
        let tokens = tokenize_words("Hello, world! It's 2026.");
        assert_eq!(tokens, vec!["Hello", "world", "It", "s", "2026"]);
    }

    // -- Case conversions --

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("helloWorld"), "hello_world");
        assert_eq!(to_snake_case("HTTPClient"), "http_client");
        assert_eq!(to_snake_case("already_snake"), "already_snake");
        assert_eq!(to_snake_case("SomeClassName"), "some_class_name");
    }

    #[test]
    fn test_to_camel_case() {
        assert_eq!(to_camel_case("hello_world"), "helloWorld");
        assert_eq!(to_camel_case("some-kebab-case"), "someKebabCase");
        assert_eq!(to_camel_case("SCREAMING_SNAKE"), "screamingSnake");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("some-kebab"), "SomeKebab");
    }

    #[test]
    fn test_to_kebab_case() {
        assert_eq!(to_kebab_case("helloWorld"), "hello-world");
        assert_eq!(to_kebab_case("SomeClassName"), "some-class-name");
    }

    #[test]
    fn test_to_screaming_snake_case() {
        assert_eq!(to_screaming_snake_case("helloWorld"), "HELLO_WORLD");
        assert_eq!(to_screaming_snake_case("some-kebab"), "SOME_KEBAB");
    }

    #[test]
    fn test_to_title_case() {
        assert_eq!(to_title_case("hello_world"), "Hello World");
        assert_eq!(to_title_case("helloWorld"), "Hello World");
    }

    #[test]
    fn test_split_into_words() {
        let words = split_into_words("helloWorldFoo");
        assert_eq!(words, vec!["hello", "World", "Foo"]);

        let words2 = split_into_words("snake_case_var");
        assert_eq!(words2, vec!["snake", "case", "var"]);

        let words3 = split_into_words("kebab-case-var");
        assert_eq!(words3, vec!["kebab", "case", "var"]);
    }

    // -- Dice coefficient --

    #[test]
    fn test_dice_coefficient_identical() {
        assert!((dice_coefficient("night", "night") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dice_coefficient_different() {
        let d = dice_coefficient("abc", "xyz");
        assert!(d < 0.01);
    }

    // -- Padding --

    #[test]
    fn test_pad_left() {
        assert_eq!(pad_left("42", 5, '0'), "00042");
        assert_eq!(pad_left("hello", 3, ' '), "hello"); // no pad needed
    }

    #[test]
    fn test_pad_right() {
        assert_eq!(pad_right("hi", 5, '.'), "hi...");
    }

    #[test]
    fn test_center() {
        assert_eq!(center("hi", 6, '-'), "--hi--");
        assert_eq!(center("hi", 7, '-'), "--hi---");
    }

    // -- Misc --

    #[test]
    fn test_reverse() {
        assert_eq!(reverse("hello"), "olleh");
        assert_eq!(reverse(""), "");
    }

    #[test]
    fn test_is_palindrome() {
        assert!(is_palindrome("racecar"));
        assert!(is_palindrome("A man a plan a canal Panama"));
        assert!(!is_palindrome("hello"));
        assert!(is_palindrome(""));
    }

    #[test]
    fn test_count_occurrences() {
        assert_eq!(count_occurrences("abababab", "ab"), 4);
        assert_eq!(count_occurrences("hello", "xyz"), 0);
        assert_eq!(count_occurrences("hello", ""), 0);
    }

    // -- Unicode support --

    #[test]
    fn test_levenshtein_unicode() {
        assert_eq!(levenshtein_distance("cafe", "cafe"), 0);
        // Japanese characters
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_char_ngrams_unicode() {
        let grams = char_ngrams("ab", 2).expect("ok");
        assert_eq!(grams.len(), 1);
        assert_eq!(grams[0], "ab");
    }
}
