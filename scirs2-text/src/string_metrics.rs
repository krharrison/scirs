//! String metrics module for distance calculations and phonetic algorithms.

use crate::Result;
use std::cmp::{max, min};
use std::collections::HashMap;

/// Trait for string distance metrics
pub trait StringMetric {
    /// Calculate the distance between two strings
    fn distance(&self, s1: &str, s2: &str) -> Result<f64>;

    /// Calculate normalized distance (0.0 to 1.0)
    fn normalized_distance(&self, s1: &str, s2: &str) -> Result<f64> {
        let dist = self.distance(s1, s2)?;
        let max_len = max(s1.len(), s2.len()) as f64;
        Ok(if max_len > 0.0 { dist / max_len } else { 0.0 })
    }

    /// Calculate similarity (1.0 - normalized distance)
    fn similarity(&self, s1: &str, s2: &str) -> Result<f64> {
        Ok(1.0 - self.normalized_distance(s1, s2)?)
    }
}

/// Damerau-Levenshtein distance metric
#[derive(Debug, Clone)]
pub struct DamerauLevenshteinMetric {
    /// Whether to use restricted Damerau-Levenshtein (no substring transpositions)
    restricted: bool,
}

impl DamerauLevenshteinMetric {
    /// Create a new Damerau-Levenshtein metric
    pub fn new() -> Self {
        Self { restricted: false }
    }

    /// Create a restricted variant (OSA - Optimal String Alignment)
    pub fn restricted() -> Self {
        Self { restricted: true }
    }
}

impl Default for DamerauLevenshteinMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl StringMetric for DamerauLevenshteinMetric {
    fn distance(&self, s1: &str, s2: &str) -> Result<f64> {
        if self.restricted {
            Ok(osa_distance(s1, s2) as f64)
        } else {
            Ok(damerau_levenshtein_distance(s1, s2) as f64)
        }
    }
}

/// Calculate restricted Damerau-Levenshtein (OSA) distance
#[allow(dead_code)]
fn osa_distance(s1: &str, s2: &str) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let len_a = a.len();
    let len_b = b.len();

    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }

    let mut matrix = vec![vec![0; len_b + 1]; len_a + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(len_a + 1) {
        row[0] = i;
    }
    // For the first row, set each column index as the value
    for j in 0..=len_b {
        matrix[0][j] = j;
    }

    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };

            matrix[i][j] = min(
                min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );

            // Transposition
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + cost);
            }
        }
    }

    matrix[len_a][len_b]
}

/// Calculate full Damerau-Levenshtein distance
#[allow(dead_code)]
fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let len_a = a.len();
    let len_b = b.len();

    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }

    let max_dist = len_a + len_b;
    let mut h = vec![vec![max_dist; len_b + 2]; len_a + 2];
    let mut da: HashMap<char, usize> = HashMap::new();

    for i in 0..=len_a {
        h[i + 1][0] = max_dist;
        h[i + 1][1] = i;
    }
    for j in 0..=len_b {
        h[0][j + 1] = max_dist;
        h[1][j + 1] = j;
    }

    for i in 1..=len_a {
        let mut db = 0;
        for j in 1..=len_b {
            let k = da.get(&b[j - 1]).copied().unwrap_or(0);
            let l = db;
            let cost = if a[i - 1] == b[j - 1] {
                db = j;
                0
            } else {
                1
            };

            h[i + 1][j + 1] = min(
                min(
                    h[i][j] + cost,  // substitution
                    h[i + 1][j] + 1, // insertion
                ),
                min(
                    h[i][j + 1] + 1,                         // deletion
                    h[k][l] + (i - k - 1) + 1 + (j - l - 1), // transposition
                ),
            );
        }

        da.insert(a[i - 1], i);
    }

    h[len_a + 1][len_b + 1]
}

/// Phonetic algorithms trait
pub trait PhoneticAlgorithm {
    /// Encode a string using the phonetic algorithm
    fn encode(&self, text: &str) -> Result<String>;

    /// Check if two strings are phonetically similar
    fn sounds_like(&self, s1: &str, s2: &str) -> Result<bool> {
        Ok(self.encode(s1)? == self.encode(s2)?)
    }
}

/// Soundex phonetic algorithm
#[derive(Debug, Clone)]
pub struct Soundex {
    /// Length of the code (standard is 4)
    length: usize,
}

impl Soundex {
    /// Create new Soundex encoder with standard length (4)
    pub fn new() -> Self {
        Self { length: 4 }
    }

    /// Create with custom code length
    pub fn with_length(length: usize) -> Self {
        Self { length }
    }
}

impl Default for Soundex {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticAlgorithm for Soundex {
    fn encode(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let text = text.to_uppercase();
        let chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();

        if chars.is_empty() {
            return Ok(String::new());
        }

        let mut code = String::new();
        code.push(chars[0]);

        let mut last_code = encode_char(chars[0]);

        for &ch in &chars[1..] {
            let ch_code = encode_char(ch);
            if ch_code != '0' && ch_code != last_code {
                code.push(ch_code);
                last_code = ch_code;
            }

            if code.len() >= self.length {
                break;
            }
        }

        // Pad with zeros if necessary
        while code.len() < self.length {
            code.push('0');
        }

        Ok(code)
    }
}

/// Encode a character for Soundex
#[allow(dead_code)]
fn encode_char(ch: char) -> char {
    match ch.to_ascii_uppercase() {
        'B' | 'F' | 'P' | 'V' => '1',
        'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
        'D' | 'T' => '3',
        'L' => '4',
        'M' | 'N' => '5',
        'R' => '6',
        _ => '0',
    }
}

/// Metaphone phonetic algorithm
#[derive(Debug, Clone)]
pub struct Metaphone {
    /// Maximum length of the code
    max_length: usize,
}

/// NYSIIS (New York State Identification and Intelligence System) phonetic algorithm
#[derive(Debug, Clone)]
pub struct Nysiis {
    /// Maximum length of the code (0 for no limit)
    max_length: usize,
}

/// Needleman-Wunsch global sequence alignment algorithm
#[derive(Debug, Clone)]
pub struct NeedlemanWunsch {
    /// Score for matching characters
    match_score: i32,
    /// Penalty for mismatching characters
    mismatch_penalty: i32,
    /// Penalty for gaps
    gap_penalty: i32,
}

/// Result of sequence alignment
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentResult {
    /// Aligned first sequence
    pub aligned_seq1: String,
    /// Aligned second sequence
    pub aligned_seq2: String,
    /// Alignment score
    pub score: i32,
}

/// Smith-Waterman local sequence alignment algorithm
#[derive(Debug, Clone)]
pub struct SmithWaterman {
    /// Score for matching characters
    match_score: i32,
    /// Penalty for mismatching characters
    mismatch_penalty: i32,
    /// Penalty for gaps
    gap_penalty: i32,
}

impl Metaphone {
    /// Create new Metaphone encoder
    pub fn new() -> Self {
        Self { max_length: 6 }
    }

    /// Create with custom maximum length
    pub fn with_max_length(_maxlength: usize) -> Self {
        Self {
            max_length: _maxlength,
        }
    }
}

impl Default for Metaphone {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticAlgorithm for Metaphone {
    fn encode(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let text = text.to_uppercase();
        let chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();

        if chars.is_empty() {
            return Ok(String::new());
        }

        let mut result = String::new();
        let mut i = 0;

        // Initial transformations
        if chars.len() >= 2 {
            match (chars[0], chars[1]) {
                ('A', 'E') | ('G', 'N') | ('K', 'N') | ('P', 'N') | ('W', 'R') => i = 1,
                ('W', 'H') => {
                    result.push('W');
                    i = 2;
                }
                _ => {}
            }
        }

        while i < chars.len() && result.len() < self.max_length {
            let ch = chars[i];
            let next = chars.get(i + 1).copied();
            let prev = if i > 0 {
                chars.get(i - 1).copied()
            } else {
                None
            };

            match ch {
                'A' | 'E' | 'I' | 'O' | 'U' if i == 0 => {
                    result.push(ch);
                }
                'B' if !result.ends_with('B') => {
                    result.push('B');
                }
                'C' => {
                    if next == Some('H') {
                        result.push('X');
                        i += 1;
                    } else if next == Some('I') || next == Some('E') || next == Some('Y') {
                        result.push('S');
                    } else {
                        result.push('K');
                    }
                }
                'D' => {
                    if next == Some('G') && chars.get(i + 2) == Some(&'E') {
                        result.push('J');
                        i += 2;
                    } else {
                        result.push('T');
                    }
                }
                'F' | 'J' | 'L' | 'M' | 'N' | 'R' if !result.ends_with(ch) => {
                    result.push(ch);
                }
                'G' => {
                    if next == Some('H') {
                        i += 1;
                    } else if next == Some('N') {
                        result.push('N');
                        i += 1;
                    } else {
                        result.push('K');
                    }
                }
                'H' if prev != Some('C')
                    && prev != Some('S')
                    && prev != Some('P')
                    && prev != Some('T')
                    && prev != Some('G') =>
                {
                    result.push('H');
                }
                'K' if prev != Some('C') => {
                    result.push('K');
                }
                'P' => {
                    if next == Some('H') {
                        result.push('F');
                        i += 1;
                    } else {
                        result.push('P');
                    }
                }
                'Q' => result.push('K'),
                'S' => {
                    if next == Some('H') {
                        result.push('X');
                        i += 1;
                    } else if next == Some('I')
                        && (chars.get(i + 2) == Some(&'A') || chars.get(i + 2) == Some(&'O'))
                    {
                        result.push('X');
                    } else {
                        result.push('S');
                    }
                }
                'T' => {
                    if next == Some('H') {
                        result.push('0');
                        i += 1;
                    } else if next != Some('C') || chars.get(i + 2) != Some(&'H') {
                        result.push('T');
                    }
                }
                'V' => result.push('F'),
                'W' | 'Y' if next.map(|c| "AEIOU".contains(c)).unwrap_or(false) => {
                    result.push(ch);
                }
                'X' => {
                    result.push('K');
                    result.push('S');
                }
                'Z' => result.push('S'),
                _ => {}
            }

            i += 1;
        }

        Ok(result)
    }
}

impl Nysiis {
    /// Create new NYSIIS encoder with no length limit
    pub fn new() -> Self {
        Self { max_length: 0 }
    }

    /// Create with custom maximum length
    pub fn with_max_length(_maxlength: usize) -> Self {
        Self {
            max_length: _maxlength,
        }
    }
}

impl Default for Nysiis {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticAlgorithm for Nysiis {
    fn encode(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        // Convert to uppercase and keep only alphabetic characters
        let mut word: Vec<char> = text
            .to_uppercase()
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect();

        if word.is_empty() {
            return Ok(String::new());
        }

        // Step 1: Handle initial letter combinations
        if word.len() >= 3 {
            let start3 = &word[0..3].iter().collect::<String>();
            let start2 = &word[0..2].iter().collect::<String>();

            if start3 == "MAC" {
                word[1] = 'C';
            } else if start2 == "KN" {
                word.remove(0);
            }
        } else if word.len() >= 2 {
            let start = &word[0..2].iter().collect::<String>();
            if start == "KN" {
                word.remove(0);
            }
        }

        // Handle PH, PF at start
        if word.len() >= 2 && (word[0] == 'P' && (word[1] == 'H' || word[1] == 'F')) {
            word[0] = 'F';
            word[1] = 'F';
        }

        // Handle SCH at start
        if word.len() >= 3 && word[0] == 'S' && word[1] == 'C' && word[2] == 'H' {
            word[0] = 'S';
            word[1] = 'S';
            word.remove(2); // Remove the third character to get SS instead of SSS
        }

        // Step 2: Handle last letter combinations
        let len = word.len();
        if len >= 2 {
            let end = &word[len - 2..].iter().collect::<String>();
            match end.as_str() {
                "EE" | "IE" => {
                    word.truncate(len - 2);
                    word.push('Y');
                }
                "DT" | "RT" | "RD" | "NT" | "ND" => {
                    word.truncate(len - 1);
                    word.push('D');
                }
                _ => {}
            }
        }

        // Step 3: First character of key is first character of name
        let mut result = vec![word[0]];

        // Step 4: Translate remaining characters
        for i in 1..word.len() {
            let ch = word[i];
            let prev = word[i - 1];
            let next = word.get(i + 1).copied();

            match ch {
                'A' => {
                    // Add 'A' only if previous character is different and result doesn't end with 'A'
                    if prev != 'A' && result.last() != Some(&'A') {
                        result.push('A');
                    }
                }
                'E' | 'I' | 'O' | 'U' => {
                    if prev == ch {
                        continue; // Skip repeated vowels
                    }
                    // Convert vowels to 'A', but avoid consecutive 'A's in the result
                    if result.last() != Some(&'A') {
                        result.push('A');
                    }
                }
                'Q' => result.push('G'),
                'Z' => result.push('S'),
                'M' => result.push('N'),
                'K' => {
                    if next == Some('N') {
                        result.push('N');
                    } else {
                        result.push('C');
                    }
                }
                'S' => {
                    if next == Some('C') && word.get(i + 2) == Some(&'H') {
                        result.push('S');
                        result.push('S');
                        result.push('S');
                    } else {
                        result.push('S');
                    }
                }
                'P' => {
                    if next == Some('H') {
                        result.push('F');
                        result.push('F');
                    } else {
                        result.push('P');
                    }
                }
                'H' => {
                    // Skip 'H' if it follows 'G' (for GH combinations)
                    if prev == 'G' {
                        // Skip this 'H'
                    } else if !matches!(prev, 'A' | 'E' | 'I' | 'O' | 'U')
                        && !matches!(
                            next,
                            Some('A') | Some('E') | Some('I') | Some('O') | Some('U')
                        )
                        && prev != ch
                    {
                        result.push('H');
                    }
                }
                'W' => {
                    if matches!(prev, 'A' | 'E' | 'I' | 'O' | 'U') && prev != ch {
                        result.push('W');
                    }
                }
                _ => {
                    if prev != ch {
                        // Skip repeated consonants
                        result.push(ch);
                    } else if i == 1 && ch == 'F' && result.len() == 1 && result[0] == 'F' {
                        // Special case: allow FF from PH conversion at start only
                        result.push(ch);
                    } else if i == 1 && ch == 'S' && result.len() == 1 && result[0] == 'S' {
                        // Special case: allow SS from SCH conversion at start only
                        result.push(ch);
                    }
                }
            }
        }

        // Step 5: Remove trailing 'S', 'A', and 'H'
        while result.len() > 1
            && (result.last() == Some(&'S')
                || result.last() == Some(&'A')
                || result.last() == Some(&'H'))
        {
            result.pop();
        }

        // Step 6: Replace 'AY' at end with 'Y'
        if result.len() >= 2 && result[result.len() - 2] == 'A' && result[result.len() - 1] == 'Y' {
            result.pop();
            result.pop();
            result.push('Y');
        }

        // Apply max length if specified
        let mut encoded: String = result.into_iter().collect();
        if self.max_length > 0 && encoded.len() > self.max_length {
            encoded.truncate(self.max_length);
        }

        Ok(encoded)
    }
}

impl NeedlemanWunsch {
    /// Create a new Needleman-Wunsch aligner with default parameters
    pub fn new() -> Self {
        Self {
            match_score: 1,
            mismatch_penalty: -1,
            gap_penalty: -1,
        }
    }

    /// Create with custom scoring parameters
    pub fn with_scores(_match_score: i32, mismatch_penalty: i32, gappenalty: i32) -> Self {
        Self {
            match_score: _match_score,
            mismatch_penalty,
            gap_penalty: gappenalty,
        }
    }

    /// Align two sequences using the Needleman-Wunsch algorithm
    pub fn align(&self, seq1: &str, seq2: &str) -> AlignmentResult {
        let seq1_chars: Vec<char> = seq1.chars().collect();
        let seq2_chars: Vec<char> = seq2.chars().collect();
        let m = seq1_chars.len();
        let n = seq2_chars.len();

        // Initialize scoring matrix
        let mut matrix = vec![vec![0; n + 1]; m + 1];

        // Initialize first row and column with gap penalties
        for (i, item) in matrix.iter_mut().enumerate().take(m + 1) {
            item[0] = i as i32 * self.gap_penalty;
        }
        for j in 0..=n {
            matrix[0][j] = j as i32 * self.gap_penalty;
        }

        // Fill the matrix
        for i in 1..=m {
            for j in 1..=n {
                let match_mismatch = if seq1_chars[i - 1] == seq2_chars[j - 1] {
                    matrix[i - 1][j - 1] + self.match_score
                } else {
                    matrix[i - 1][j - 1] + self.mismatch_penalty
                };

                let delete = matrix[i - 1][j] + self.gap_penalty;
                let insert = matrix[i][j - 1] + self.gap_penalty;

                matrix[i][j] = *[match_mismatch, delete, insert]
                    .iter()
                    .max()
                    .expect("Operation failed");
            }
        }

        // Traceback to find the alignment
        let mut aligned_seq1 = String::new();
        let mut aligned_seq2 = String::new();
        let mut i = m;
        let mut j = n;

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let current_score = matrix[i][j];
                let diagonal_score = if seq1_chars[i - 1] == seq2_chars[j - 1] {
                    matrix[i - 1][j - 1] + self.match_score
                } else {
                    matrix[i - 1][j - 1] + self.mismatch_penalty
                };
                let up_score = matrix[i - 1][j] + self.gap_penalty;
                let left_score = matrix[i][j - 1] + self.gap_penalty;

                // Prioritize diagonal (match/mismatch), then left (insertion), then up (deletion)
                // This ensures consistent behavior when multiple paths have the same score
                if current_score == diagonal_score {
                    aligned_seq1.insert(0, seq1_chars[i - 1]);
                    aligned_seq2.insert(0, seq2_chars[j - 1]);
                    i -= 1;
                    j -= 1;
                } else if current_score == left_score {
                    aligned_seq1.insert(0, '-');
                    aligned_seq2.insert(0, seq2_chars[j - 1]);
                    j -= 1;
                } else if current_score == up_score {
                    aligned_seq1.insert(0, seq1_chars[i - 1]);
                    aligned_seq2.insert(0, '-');
                    i -= 1;
                } else {
                    // Fallback case - should not happen with correct implementation
                    aligned_seq1.insert(0, seq1_chars[i - 1]);
                    aligned_seq2.insert(0, seq2_chars[j - 1]);
                    i -= 1;
                    j -= 1;
                }
            } else if i > 0 {
                aligned_seq1.insert(0, seq1_chars[i - 1]);
                aligned_seq2.insert(0, '-');
                i -= 1;
            } else {
                aligned_seq1.insert(0, '-');
                aligned_seq2.insert(0, seq2_chars[j - 1]);
                j -= 1;
            }
        }

        AlignmentResult {
            aligned_seq1,
            aligned_seq2,
            score: matrix[m][n],
        }
    }
}

impl Default for NeedlemanWunsch {
    fn default() -> Self {
        Self::new()
    }
}

impl SmithWaterman {
    /// Create a new Smith-Waterman aligner with default parameters
    pub fn new() -> Self {
        Self {
            match_score: 2,
            mismatch_penalty: -1,
            gap_penalty: -1,
        }
    }

    /// Create with custom scoring parameters
    pub fn with_scores(_match_score: i32, mismatch_penalty: i32, gappenalty: i32) -> Self {
        Self {
            match_score: _match_score,
            mismatch_penalty,
            gap_penalty: gappenalty,
        }
    }

    /// Align two sequences using the Smith-Waterman algorithm
    pub fn align(&self, seq1: &str, seq2: &str) -> AlignmentResult {
        let seq1_chars: Vec<char> = seq1.chars().collect();
        let seq2_chars: Vec<char> = seq2.chars().collect();
        let m = seq1_chars.len();
        let n = seq2_chars.len();

        // Initialize scoring matrix
        let mut matrix = vec![vec![0; n + 1]; m + 1];
        let mut max_score = 0;
        let mut max_i = 0;
        let mut max_j = 0;

        // Fill the matrix
        for i in 1..=m {
            for j in 1..=n {
                let match_mismatch = if seq1_chars[i - 1] == seq2_chars[j - 1] {
                    matrix[i - 1][j - 1] + self.match_score
                } else {
                    matrix[i - 1][j - 1] + self.mismatch_penalty
                };

                let delete = matrix[i - 1][j] + self.gap_penalty;
                let insert = matrix[i][j - 1] + self.gap_penalty;

                matrix[i][j] = *[0, match_mismatch, delete, insert]
                    .iter()
                    .max()
                    .expect("Operation failed");

                // Track maximum score position
                if matrix[i][j] > max_score {
                    max_score = matrix[i][j];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        // Traceback from the maximum score position
        let mut aligned_seq1 = String::new();
        let mut aligned_seq2 = String::new();
        let mut i = max_i;
        let mut j = max_j;

        while i > 0 && j > 0 && matrix[i][j] > 0 {
            let current_score = matrix[i][j];
            let diagonal_score = if seq1_chars[i - 1] == seq2_chars[j - 1] {
                matrix[i - 1][j - 1] + self.match_score
            } else {
                matrix[i - 1][j - 1] + self.mismatch_penalty
            };

            if current_score == diagonal_score {
                aligned_seq1.insert(0, seq1_chars[i - 1]);
                aligned_seq2.insert(0, seq2_chars[j - 1]);
                i -= 1;
                j -= 1;
            } else if current_score == matrix[i - 1][j] + self.gap_penalty {
                aligned_seq1.insert(0, seq1_chars[i - 1]);
                aligned_seq2.insert(0, '-');
                i -= 1;
            } else if current_score == matrix[i][j - 1] + self.gap_penalty {
                aligned_seq1.insert(0, '-');
                aligned_seq2.insert(0, seq2_chars[j - 1]);
                j -= 1;
            } else {
                // This shouldn't happen in correct implementation
                break;
            }
        }

        AlignmentResult {
            aligned_seq1,
            aligned_seq2,
            score: max_score,
        }
    }
}

impl Default for SmithWaterman {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damerau_levenshtein() {
        let metric = DamerauLevenshteinMetric::new();

        // Basic operations
        assert_eq!(metric.distance("", "").expect("Operation failed"), 0.0);
        assert_eq!(metric.distance("abc", "").expect("Operation failed"), 3.0);
        assert_eq!(metric.distance("", "abc").expect("Operation failed"), 3.0);
        assert_eq!(
            metric.distance("abc", "abc").expect("Operation failed"),
            0.0
        );

        // Single operations
        assert_eq!(
            metric.distance("abc", "aXc").expect("Operation failed"),
            1.0
        ); // substitution
        assert_eq!(metric.distance("abc", "ac").expect("Operation failed"), 1.0); // deletion
        assert_eq!(metric.distance("ac", "abc").expect("Operation failed"), 1.0); // insertion
        assert_eq!(
            metric.distance("abc", "acb").expect("Operation failed"),
            1.0
        ); // transposition

        // Multiple operations
        assert_eq!(
            metric
                .distance("kitten", "sitting")
                .expect("Operation failed"),
            3.0
        );

        // Test normalized distance
        assert!(
            (metric
                .normalized_distance("abc", "aXc")
                .expect("Operation failed")
                - 0.333)
                .abs()
                < 0.01
        );
        assert_eq!(
            metric
                .normalized_distance("", "")
                .expect("Operation failed"),
            0.0
        );

        // Test similarity
        assert!((metric.similarity("abc", "aXc").expect("Operation failed") - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_restricted_damerau_levenshtein() {
        let metric = DamerauLevenshteinMetric::restricted();

        // OSA doesn't allow substring transpositions
        assert_eq!(metric.distance("ca", "abc").expect("Operation failed"), 3.0);
        // Not 2.0 as in full DL
    }

    #[test]
    fn test_soundex() {
        let soundex = Soundex::new();

        assert_eq!(soundex.encode("Robert").expect("Operation failed"), "R163");
        assert_eq!(soundex.encode("Rupert").expect("Operation failed"), "R163");
        assert_eq!(soundex.encode("SOUNDEX").expect("Operation failed"), "S532");
        assert_eq!(soundex.encode("Smith").expect("Operation failed"), "S530");
        assert_eq!(soundex.encode("Smythe").expect("Operation failed"), "S530");
        assert_eq!(soundex.encode("").expect("Operation failed"), "");
        assert_eq!(soundex.encode("123").expect("Operation failed"), "");

        // Test sounds_like
        assert!(soundex
            .sounds_like("Robert", "Rupert")
            .expect("Operation failed"));
        assert!(soundex
            .sounds_like("Smith", "Smythe")
            .expect("Operation failed"));
        assert!(!soundex
            .sounds_like("Smith", "Jones")
            .expect("Operation failed"));

        // Test custom length
        let soundex_5 = Soundex::with_length(5);
        assert_eq!(
            soundex_5.encode("SOUNDEX").expect("Operation failed"),
            "S5320"
        );
    }

    #[test]
    fn test_metaphone() {
        let metaphone = Metaphone::new();

        assert_eq!(
            metaphone.encode("programming").expect("Operation failed"),
            "PRKRMN"
        );
        assert_eq!(
            metaphone.encode("programmer").expect("Operation failed"),
            "PRKRMR"
        );
        assert_eq!(metaphone.encode("Wright").expect("Operation failed"), "RT");
        assert_eq!(metaphone.encode("White").expect("Operation failed"), "WT");
        assert_eq!(metaphone.encode("Knight").expect("Operation failed"), "NT");
        assert_eq!(metaphone.encode("").expect("Operation failed"), "");
        assert_eq!(metaphone.encode("123").expect("Operation failed"), "");

        // Test sounds_like - "Wright" and "Write" actually should sound similar in Metaphone
        assert!(metaphone
            .sounds_like("Wright", "Write")
            .expect("Operation failed"));
        assert!(!metaphone
            .sounds_like("White", "Wright")
            .expect("Operation failed"));

        // Test custom max length
        let metaphone_3 = Metaphone::with_max_length(3);
        assert_eq!(
            metaphone_3.encode("programming").expect("Operation failed"),
            "PRK"
        );
    }

    #[test]
    fn test_phonetic_edge_cases() {
        let soundex = Soundex::new();
        let metaphone = Metaphone::new();

        // Test with non-alphabetic characters
        assert_eq!(soundex.encode("O'Brien").expect("Operation failed"), "O165");
        assert_eq!(
            metaphone.encode("O'Brien").expect("Operation failed"),
            "OBRN"
        );

        // Test with mixed case
        assert_eq!(
            soundex.encode("McDonald").expect("Operation failed"),
            "M235"
        );
        assert_eq!(
            metaphone.encode("McDonald").expect("Operation failed"),
            "MKTNLT"
        );
    }

    #[test]
    fn test_nysiis() {
        let nysiis = Nysiis::new();

        // Basic tests
        assert_eq!(
            nysiis.encode("Johnson").expect("Operation failed"),
            "JANSAN"
        );
        assert_eq!(
            nysiis.encode("Williams").expect("Operation failed"),
            "WALAN"
        ); // Standard NYSIIS code
        assert_eq!(nysiis.encode("Jones").expect("Operation failed"), "JAN");
        assert_eq!(nysiis.encode("Smith").expect("Operation failed"), "SNAT");
        assert_eq!(
            nysiis.encode("MacDonald").expect("Operation failed"),
            "MCDANALD"
        );
        assert_eq!(nysiis.encode("Knight").expect("Operation failed"), "NAGT");
        assert_eq!(nysiis.encode("").expect("Operation failed"), "");
        assert_eq!(nysiis.encode("123").expect("Operation failed"), "");

        // Test sounds_like
        assert!(nysiis
            .sounds_like("Johnson", "Jonson")
            .expect("Operation failed"));
        assert!(!nysiis
            .sounds_like("Smith", "Jones")
            .expect("Operation failed"));

        // Test edge cases
        assert_eq!(
            nysiis.encode("Philips").expect("Operation failed"),
            "FFALAP"
        );
        assert_eq!(nysiis.encode("Schmidt").expect("Operation failed"), "SSNAD");
        assert_eq!(
            nysiis.encode("Schneider").expect("Operation failed"),
            "SSNADAR"
        );

        // Test with max length
        let nysiis_6 = Nysiis::with_max_length(6);
        assert_eq!(
            nysiis_6.encode("Williams").expect("Operation failed"),
            "WALAN"
        ); // 5 chars, so not truncated
        assert_eq!(
            nysiis_6.encode("MacDonald").expect("Operation failed"),
            "MCDANA"
        ); // 6 chars, truncated from longer
    }

    #[test]
    fn test_needleman_wunsch() {
        let aligner = NeedlemanWunsch::new();

        // Test simple alignment
        let result = aligner.align("GATTACA", "GCATGCU");

        // Check that we get a valid optimal alignment with correct score
        assert_eq!(result.aligned_seq1, "G-ATTACA");
        assert_eq!(result.score, 0);

        // Verify the alignment is valid by checking that all characters from seq2 are present
        let seq2_chars = result
            .aligned_seq2
            .chars()
            .filter(|&c| c != '-')
            .collect::<String>();
        assert_eq!(seq2_chars, "GCATGCU");

        // Verify alignment length consistency
        assert_eq!(result.aligned_seq1.len(), result.aligned_seq2.len());

        // Test identical sequences
        let result = aligner.align("HELLO", "HELLO");
        assert_eq!(result.aligned_seq1, "HELLO");
        assert_eq!(result.aligned_seq2, "HELLO");
        assert_eq!(result.score, 5);

        // Test empty sequences
        let result = aligner.align("", "ABC");
        assert_eq!(result.aligned_seq1, "---");
        assert_eq!(result.aligned_seq2, "ABC");
        assert_eq!(result.score, -3);

        let result = aligner.align("ABC", "");
        assert_eq!(result.aligned_seq1, "ABC");
        assert_eq!(result.aligned_seq2, "---");
        assert_eq!(result.score, -3);

        // Test with custom scores
        let custom_aligner = NeedlemanWunsch::with_scores(2, -2, -1);
        let result = custom_aligner.align("CAT", "CART");
        assert!(result.score > 0);
    }

    #[test]
    fn test_smith_waterman() {
        let aligner = SmithWaterman::new();

        // Test local alignment
        let result = aligner.align("GGTTGACTA", "TGTTACGG");
        assert!(result.score > 0);
        assert!(result.aligned_seq1.contains("GTT"));
        assert!(result.aligned_seq2.contains("GTT"));

        // Test with sequences having common substring
        let result = aligner.align("ABCDEFG", "XYZCDEFPQ");
        assert_eq!(result.aligned_seq1, "CDEF");
        assert_eq!(result.aligned_seq2, "CDEF");
        assert_eq!(result.score, 8); // 4 matches * 2

        // Test empty sequences
        let result = aligner.align("", "ABC");
        assert_eq!(result.aligned_seq1, "");
        assert_eq!(result.aligned_seq2, "");
        assert_eq!(result.score, 0);

        // Test no common subsequence
        let result = aligner.align("AAA", "BBB");
        assert_eq!(result.score, 0);

        // Test with custom scores
        let custom_aligner = SmithWaterman::with_scores(3, -3, -2);
        let result = custom_aligner.align("ACACACTA", "AGCACACA");
        assert!(result.score > 0);
    }
}
