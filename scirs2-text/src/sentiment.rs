//! Sentiment analysis functionality
//!
//! This module provides comprehensive sentiment analysis capabilities:
//!
//! - **Lexicon-based**: Score text using word-level sentiment dictionaries
//! - **Rule-based (VADER-inspired)**: Handle negation, intensifiers, but-clauses,
//!   capitalization emphasis, and punctuation heuristics
//! - **Naive Bayes classifier**: Train a probabilistic sentiment classifier from labeled data
//! - **Aspect-based**: Extract sentiment for specific aspects/entities in text
//! - **Document aggregation**: Aggregate sentiment across multiple texts/paragraphs
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::sentiment::{LexiconSentimentAnalyzer, VaderSentimentAnalyzer, Sentiment};
//!
//! // Basic lexicon-based analysis
//! let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();
//! let result = analyzer.analyze("I love this product!").unwrap();
//! assert_eq!(result.sentiment, Sentiment::Positive);
//!
//! // VADER-style analysis with intensifiers and negation
//! let vader = VaderSentimentAnalyzer::new();
//! let result = vader.analyze("This movie is not just good, it is ABSOLUTELY amazing!").unwrap();
//! assert_eq!(result.sentiment, Sentiment::Positive);
//! ```

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use std::collections::HashMap;

/// Sentiment polarity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sentiment {
    /// Positive sentiment
    Positive,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment
    Neutral,
}

impl std::fmt::Display for Sentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sentiment::Positive => write!(f, "Positive"),
            Sentiment::Negative => write!(f, "Negative"),
            Sentiment::Neutral => write!(f, "Neutral"),
        }
    }
}

impl Sentiment {
    /// Convert sentiment to a numerical score
    pub fn to_score(&self) -> f64 {
        match self {
            Sentiment::Positive => 1.0,
            Sentiment::Neutral => 0.0,
            Sentiment::Negative => -1.0,
        }
    }

    /// Convert a numerical score to sentiment
    pub fn from_score(score: f64) -> Self {
        if score > 0.05 {
            Sentiment::Positive
        } else if score < -0.05 {
            Sentiment::Negative
        } else {
            Sentiment::Neutral
        }
    }
}

/// Result of sentiment analysis
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// The overall sentiment
    pub sentiment: Sentiment,
    /// The raw sentiment score
    pub score: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Breakdown of positive and negative word counts
    pub word_counts: SentimentWordCounts,
}

/// Word counts for sentiment analysis
#[derive(Debug, Clone, Default)]
pub struct SentimentWordCounts {
    /// Number of positive words
    pub positive_words: usize,
    /// Number of negative words
    pub negative_words: usize,
    /// Number of neutral words
    pub neutral_words: usize,
    /// Total number of words analyzed
    pub total_words: usize,
}

// ─── Sentiment Lexicon ───────────────────────────────────────────────────────

/// A sentiment lexicon mapping words to sentiment scores
#[derive(Debug, Clone)]
pub struct SentimentLexicon {
    /// Word to sentiment score mapping
    lexicon: HashMap<String, f64>,
    /// Default score for unknown words
    default_score: f64,
}

impl SentimentLexicon {
    /// Create a new sentiment lexicon
    pub fn new() -> Self {
        Self {
            lexicon: HashMap::new(),
            default_score: 0.0,
        }
    }

    /// Create a basic sentiment lexicon with common words
    pub fn with_basiclexicon() -> Self {
        let mut lexicon = HashMap::new();

        // Positive words (AFINN-style scores)
        let positive_words = [
            ("good", 1.0),
            ("great", 2.0),
            ("excellent", 3.0),
            ("amazing", 3.0),
            ("wonderful", 2.5),
            ("fantastic", 2.5),
            ("love", 2.0),
            ("like", 1.0),
            ("happy", 2.0),
            ("joy", 2.0),
            ("pleased", 1.5),
            ("satisfied", 1.0),
            ("positive", 1.0),
            ("perfect", 3.0),
            ("best", 2.5),
            ("awesome", 2.5),
            ("beautiful", 2.0),
            ("brilliant", 2.5),
            ("superb", 2.5),
            ("nice", 1.0),
            ("outstanding", 3.0),
            ("exceptional", 3.0),
            ("remarkable", 2.0),
            ("delightful", 2.5),
            ("impressive", 2.0),
            ("enjoy", 1.5),
            ("recommend", 1.5),
            ("better", 1.0),
            ("superior", 2.0),
            ("exciting", 2.0),
        ];

        // Negative words
        let negative_words = [
            ("bad", -1.0),
            ("terrible", -2.5),
            ("awful", -2.5),
            ("horrible", -3.0),
            ("hate", -2.5),
            ("dislike", -1.5),
            ("sad", -2.0),
            ("unhappy", -2.0),
            ("disappointed", -2.0),
            ("negative", -1.0),
            ("worst", -3.0),
            ("poor", -1.5),
            ("disgusting", -3.0),
            ("ugly", -2.0),
            ("nasty", -2.5),
            ("stupid", -2.0),
            ("pathetic", -2.5),
            ("failure", -2.0),
            ("fail", -2.0),
            ("sucks", -2.0),
            ("boring", -1.5),
            ("mediocre", -1.0),
            ("inferior", -2.0),
            ("lousy", -2.0),
            ("dreadful", -2.5),
            ("annoying", -1.5),
            ("frustrating", -2.0),
            ("disappointing", -2.0),
            ("terrible", -2.5),
            ("useless", -2.0),
        ];

        for (word, score) in &positive_words {
            lexicon.insert(word.to_string(), *score);
        }

        for (word, score) in &negative_words {
            lexicon.insert(word.to_string(), *score);
        }

        Self {
            lexicon,
            default_score: 0.0,
        }
    }

    /// Add a word to the lexicon
    pub fn add_word(&mut self, word: String, score: f64) {
        self.lexicon.insert(word.to_lowercase(), score);
    }

    /// Get the sentiment score for a word
    pub fn get_score(&self, word: &str) -> f64 {
        self.lexicon
            .get(&word.to_lowercase())
            .copied()
            .unwrap_or(self.default_score)
    }

    /// Check if a word is in the lexicon
    pub fn contains(&self, word: &str) -> bool {
        self.lexicon.contains_key(&word.to_lowercase())
    }

    /// Get the size of the lexicon
    pub fn len(&self) -> usize {
        self.lexicon.len()
    }

    /// Check if the lexicon is empty
    pub fn is_empty(&self) -> bool {
        self.lexicon.is_empty()
    }

    /// Get all words and their scores
    pub fn entries(&self) -> &HashMap<String, f64> {
        &self.lexicon
    }
}

impl Default for SentimentLexicon {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Lexicon-based Sentiment Analyzer ────────────────────────────────────────

/// Lexicon-based sentiment analyzer
pub struct LexiconSentimentAnalyzer {
    /// The sentiment lexicon
    lexicon: SentimentLexicon,
    /// The tokenizer to use
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Negation words that reverse sentiment
    negation_words: Vec<String>,
    /// Window size for negation detection
    negation_window: usize,
}

impl LexiconSentimentAnalyzer {
    /// Create a new lexicon-based sentiment analyzer
    pub fn new(lexicon: SentimentLexicon) -> Self {
        let negation_words = vec![
            "not".to_string(),
            "no".to_string(),
            "never".to_string(),
            "neither".to_string(),
            "nobody".to_string(),
            "nothing".to_string(),
            "nowhere".to_string(),
            "n't".to_string(),
            "cannot".to_string(),
            "without".to_string(),
        ];

        Self {
            lexicon,
            tokenizer: Box::new(WordTokenizer::default()),
            negation_words,
            negation_window: 3,
        }
    }

    /// Create an analyzer with a basic lexicon
    pub fn with_basiclexicon() -> Self {
        Self::new(SentimentLexicon::with_basiclexicon())
    }

    /// Set a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Analyze the sentiment of a text
    pub fn analyze(&self, text: &str) -> Result<SentimentResult> {
        let tokens = self.tokenizer.tokenize(text)?;

        if tokens.is_empty() {
            return Ok(SentimentResult {
                sentiment: Sentiment::Neutral,
                score: 0.0,
                confidence: 0.0,
                word_counts: SentimentWordCounts {
                    positive_words: 0,
                    negative_words: 0,
                    neutral_words: 0,
                    total_words: 0,
                },
            });
        }

        let mut total_score = 0.0;
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;

        // Analyze each token
        for (i, token) in tokens.iter().enumerate() {
            let token_lower = token.to_lowercase();
            let mut score = self.lexicon.get_score(&token_lower);

            // Check for negation
            if score != 0.0 {
                for j in 1..=self.negation_window.min(i) {
                    let prev_token = &tokens[i - j].to_lowercase();
                    if self.negation_words.contains(prev_token) {
                        score *= -1.0;
                        break;
                    }
                }
            }

            total_score += score;

            if score > 0.0 {
                positive_count += 1;
            } else if score < 0.0 {
                negative_count += 1;
            } else {
                neutral_count += 1;
            }
        }

        let total_words = tokens.len();
        let sentiment = Sentiment::from_score(total_score);

        // Calculate confidence based on the proportion of sentiment-bearing words
        let sentiment_words = positive_count + negative_count;
        let confidence = if total_words > 0 {
            (sentiment_words as f64 / total_words as f64).min(1.0)
        } else {
            0.0
        };

        Ok(SentimentResult {
            sentiment,
            score: total_score,
            confidence,
            word_counts: SentimentWordCounts {
                positive_words: positive_count,
                negative_words: negative_count,
                neutral_words: neutral_count,
                total_words,
            },
        })
    }

    /// Analyze sentiment for multiple texts
    pub fn analyze_batch(&self, texts: &[&str]) -> Result<Vec<SentimentResult>> {
        texts.iter().map(|&text| self.analyze(text)).collect()
    }
}

// ─── Rule-based Sentiment (intensifiers, diminishers) ────────────────────────

/// Rule-based sentiment modifications
#[derive(Debug, Clone)]
pub struct SentimentRules {
    /// Intensifier words that increase sentiment magnitude
    intensifiers: HashMap<String, f64>,
    /// Diminisher words that decrease sentiment magnitude
    diminishers: HashMap<String, f64>,
}

impl Default for SentimentRules {
    fn default() -> Self {
        let mut intensifiers = HashMap::new();
        intensifiers.insert("very".to_string(), 1.5);
        intensifiers.insert("extremely".to_string(), 2.0);
        intensifiers.insert("incredibly".to_string(), 2.0);
        intensifiers.insert("really".to_string(), 1.3);
        intensifiers.insert("so".to_string(), 1.3);
        intensifiers.insert("absolutely".to_string(), 2.0);
        intensifiers.insert("truly".to_string(), 1.5);
        intensifiers.insert("totally".to_string(), 1.5);
        intensifiers.insert("utterly".to_string(), 1.8);
        intensifiers.insert("remarkably".to_string(), 1.5);

        let mut diminishers = HashMap::new();
        diminishers.insert("somewhat".to_string(), 0.5);
        diminishers.insert("slightly".to_string(), 0.5);
        diminishers.insert("barely".to_string(), 0.3);
        diminishers.insert("hardly".to_string(), 0.3);
        diminishers.insert("a little".to_string(), 0.5);
        diminishers.insert("kind of".to_string(), 0.5);
        diminishers.insert("sort of".to_string(), 0.5);
        diminishers.insert("marginally".to_string(), 0.4);

        Self {
            intensifiers,
            diminishers,
        }
    }
}

impl SentimentRules {
    /// Apply rules to modify a sentiment score
    pub fn apply(&self, tokens: &[String], basescores: &[f64]) -> Vec<f64> {
        let mut modified_scores = basescores.to_vec();

        for (i, score) in modified_scores.iter_mut().enumerate() {
            if *score == 0.0 {
                continue;
            }

            // Check for intensifiers/diminishers in the preceding words
            for j in 1..=2.min(i) {
                let prev_token = &tokens[i - j].to_lowercase();

                if let Some(&multiplier) = self.intensifiers.get(prev_token) {
                    *score *= multiplier;
                    break;
                } else if let Some(&multiplier) = self.diminishers.get(prev_token) {
                    *score *= multiplier;
                    break;
                }
            }
        }

        modified_scores
    }
}

/// Advanced rule-based sentiment analyzer
pub struct RuleBasedSentimentAnalyzer {
    /// The base analyzer
    base_analyzer: LexiconSentimentAnalyzer,
    /// Sentiment modification rules
    rules: SentimentRules,
}

impl RuleBasedSentimentAnalyzer {
    /// Create a new rule-based sentiment analyzer
    pub fn new(lexicon: SentimentLexicon) -> Self {
        Self {
            base_analyzer: LexiconSentimentAnalyzer::new(lexicon),
            rules: SentimentRules::default(),
        }
    }

    /// Create an analyzer with a basic lexicon
    pub fn with_basiclexicon() -> Self {
        Self::new(SentimentLexicon::with_basiclexicon())
    }

    /// Analyze sentiment with rule modifications
    pub fn analyze(&self, text: &str) -> Result<SentimentResult> {
        let tokens = self.base_analyzer.tokenizer.tokenize(text)?;

        if tokens.is_empty() {
            return self.base_analyzer.analyze(text);
        }

        // Get base scores for each token
        let basescores: Vec<f64> = tokens
            .iter()
            .map(|token| self.base_analyzer.lexicon.get_score(token))
            .collect();

        // Apply rules to modify scores
        let modified_scores = self.rules.apply(&tokens, &basescores);

        // Calculate final sentiment
        let total_score: f64 = modified_scores.iter().sum();
        let sentiment = Sentiment::from_score(total_score);

        // Count sentiment words
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;

        for &score in &modified_scores {
            if score > 0.0 {
                positive_count += 1;
            } else if score < 0.0 {
                negative_count += 1;
            } else {
                neutral_count += 1;
            }
        }

        let total_words = tokens.len();
        let sentiment_words = positive_count + negative_count;
        let confidence = if total_words > 0 {
            (sentiment_words as f64 / total_words as f64).min(1.0)
        } else {
            0.0
        };

        Ok(SentimentResult {
            sentiment,
            score: total_score,
            confidence,
            word_counts: SentimentWordCounts {
                positive_words: positive_count,
                negative_words: negative_count,
                neutral_words: neutral_count,
                total_words,
            },
        })
    }
}

// ─── VADER-Inspired Sentiment Analyzer ───────────────────────────────────────

/// VADER (Valence Aware Dictionary and sEntiment Reasoner) inspired sentiment result
#[derive(Debug, Clone)]
pub struct VaderResult {
    /// Positive proportion (0-1)
    pub positive: f64,
    /// Negative proportion (0-1)
    pub negative: f64,
    /// Neutral proportion (0-1)
    pub neutral: f64,
    /// Compound score (-1 to +1), normalized using the formula from the VADER paper
    pub compound: f64,
    /// Overall sentiment label
    pub sentiment: Sentiment,
}

/// VADER-inspired sentiment analyzer
///
/// Implements key heuristics from VADER (Hutto & Gilbert, 2014):
/// - Negation handling (flips sentiment polarity)
/// - Intensifier words (boost magnitude)
/// - But-clause handling (weight sentiment after "but" more heavily)
/// - ALL CAPS emphasis (boosts sentiment of capitalized words)
/// - Exclamation marks (slight positive boost)
/// - Question marks at end (reduce sentiment)
pub struct VaderSentimentAnalyzer {
    /// Sentiment lexicon
    lexicon: SentimentLexicon,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Negation words
    negation_words: Vec<String>,
    /// Intensifier multipliers
    intensifiers: HashMap<String, f64>,
    /// Diminisher multipliers
    diminishers: HashMap<String, f64>,
    /// But-clause weight (how much more weight to give sentiment after "but")
    but_weight: f64,
    /// Caps emphasis multiplier
    caps_multiplier: f64,
    /// Exclamation boost per mark (up to 4)
    exclamation_boost: f64,
    /// Question reduction factor
    question_reduction: f64,
}

impl VaderSentimentAnalyzer {
    /// Create a new VADER-style analyzer with default settings
    pub fn new() -> Self {
        let mut intensifiers = HashMap::new();
        intensifiers.insert("very".to_string(), 0.293);
        intensifiers.insert("extremely".to_string(), 0.293);
        intensifiers.insert("absolutely".to_string(), 0.293);
        intensifiers.insert("incredibly".to_string(), 0.293);
        intensifiers.insert("really".to_string(), 0.18);
        intensifiers.insert("so".to_string(), 0.18);
        intensifiers.insert("truly".to_string(), 0.18);
        intensifiers.insert("totally".to_string(), 0.18);
        intensifiers.insert("quite".to_string(), 0.1);

        let mut diminishers = HashMap::new();
        diminishers.insert("somewhat".to_string(), -0.1);
        diminishers.insert("barely".to_string(), -0.2);
        diminishers.insert("hardly".to_string(), -0.2);
        diminishers.insert("slightly".to_string(), -0.1);
        diminishers.insert("kind of".to_string(), -0.1);
        diminishers.insert("sort of".to_string(), -0.1);

        let negation_words = vec![
            "not".to_string(),
            "no".to_string(),
            "never".to_string(),
            "neither".to_string(),
            "nobody".to_string(),
            "nothing".to_string(),
            "nowhere".to_string(),
            "cannot".to_string(),
            "without".to_string(),
            "don't".to_string(),
            "doesn't".to_string(),
            "didn't".to_string(),
            "isn't".to_string(),
            "wasn't".to_string(),
            "won't".to_string(),
            "wouldn't".to_string(),
            "shouldn't".to_string(),
            "couldn't".to_string(),
            "aren't".to_string(),
            "weren't".to_string(),
        ];

        Self {
            lexicon: SentimentLexicon::with_basiclexicon(),
            tokenizer: Box::new(WordTokenizer::default()),
            negation_words,
            intensifiers,
            diminishers,
            but_weight: 0.5,
            caps_multiplier: 0.733,
            exclamation_boost: 0.292,
            question_reduction: 0.18,
        }
    }

    /// Create with a custom lexicon
    pub fn with_lexicon(mut self, lexicon: SentimentLexicon) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Analyze text and return VADER-style compound scores
    pub fn analyze(&self, text: &str) -> Result<VaderResult> {
        let tokens = self.tokenizer.tokenize(text)?;

        if tokens.is_empty() {
            return Ok(VaderResult {
                positive: 0.0,
                negative: 0.0,
                neutral: 1.0,
                compound: 0.0,
                sentiment: Sentiment::Neutral,
            });
        }

        // Get raw sentiment scores for each token
        let mut sentiments: Vec<f64> = Vec::with_capacity(tokens.len());

        for (i, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            let mut score = self.lexicon.get_score(&lower);

            if score == 0.0 {
                sentiments.push(0.0);
                continue;
            }

            // Check for ALL CAPS emphasis (token must be > 1 char and all uppercase)
            if token.len() > 1 && token.chars().all(|c| c.is_uppercase()) {
                if score > 0.0 {
                    score += self.caps_multiplier;
                } else {
                    score -= self.caps_multiplier;
                }
            }

            // Check preceding words for intensifiers/diminishers
            for j in 1..=3.min(i) {
                let prev = tokens[i - j].to_lowercase();
                if let Some(&boost) = self.intensifiers.get(&prev) {
                    if score > 0.0 {
                        score += boost;
                    } else {
                        score -= boost;
                    }
                    break;
                } else if let Some(&reduce) = self.diminishers.get(&prev) {
                    if score > 0.0 {
                        score += reduce; // reduce is negative
                    } else {
                        score -= reduce;
                    }
                    break;
                }
            }

            // Check for negation in preceding words
            let mut negated = false;
            for j in 1..=3.min(i) {
                let prev = tokens[i - j].to_lowercase();
                if self.negation_words.contains(&prev) {
                    negated = true;
                    break;
                }
            }

            if negated {
                score *= -0.74; // VADER uses a constant negation multiplier
            }

            sentiments.push(score);
        }

        // But-clause handling: weight sentiment after "but" more heavily
        let mut but_idx = None;
        for (i, token) in tokens.iter().enumerate() {
            if token.to_lowercase() == "but" || token.to_lowercase() == "however" {
                but_idx = Some(i);
            }
        }

        if let Some(idx) = but_idx {
            // Reduce weight of sentiment before "but", increase after
            for (i, score) in sentiments.iter_mut().enumerate() {
                if i < idx {
                    *score *= 1.0 - self.but_weight;
                } else if i > idx {
                    *score *= 1.0 + self.but_weight;
                }
            }
        }

        // Sum all sentiments
        let mut sum_scores: f64 = sentiments.iter().sum();

        // Exclamation mark boost (count in original text, up to 4)
        let excl_count = text.chars().filter(|&c| c == '!').count().min(4);
        if excl_count > 0 {
            sum_scores += excl_count as f64 * self.exclamation_boost * sum_scores.signum();
        }

        // Question mark at end reduces sentiment
        if text.trim_end().ends_with('?') {
            sum_scores *= 1.0 - self.question_reduction;
        }

        // Compute compound score using VADER's normalization
        let compound = self.normalize(sum_scores);

        // Compute positive, negative, neutral proportions
        let mut pos_sum = 0.0;
        let mut neg_sum = 0.0;
        let mut neu_count = 0.0;

        for &s in &sentiments {
            if s > 0.0 {
                pos_sum += s;
            } else if s < 0.0 {
                neg_sum += s;
            } else {
                neu_count += 1.0;
            }
        }

        let total = pos_sum + neg_sum.abs() + neu_count;
        let (positive, negative, neutral) = if total > 0.0 {
            (
                (pos_sum / total).abs(),
                (neg_sum / total).abs(),
                neu_count / total,
            )
        } else {
            (0.0, 0.0, 1.0)
        };

        let sentiment = if compound >= 0.05 {
            Sentiment::Positive
        } else if compound <= -0.05 {
            Sentiment::Negative
        } else {
            Sentiment::Neutral
        };

        Ok(VaderResult {
            positive,
            negative,
            neutral,
            compound,
            sentiment,
        })
    }

    /// Normalize the sum of sentiments using VADER's formula
    fn normalize(&self, score: f64) -> f64 {
        let alpha = 15.0; // Approximation parameter
        score / (score * score + alpha).sqrt()
    }

    /// Analyze multiple texts
    pub fn analyze_batch(&self, texts: &[&str]) -> Result<Vec<VaderResult>> {
        texts.iter().map(|&text| self.analyze(text)).collect()
    }
}

impl Default for VaderSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Naive Bayes Sentiment Classifier ────────────────────────────────────────

/// Naive Bayes classifier for sentiment analysis
///
/// Implements multinomial Naive Bayes with Laplace smoothing.
/// Can be trained on labeled (text, sentiment) pairs and used to
/// predict sentiment for new text.
pub struct NaiveBayesSentiment {
    /// Word counts per class: class_label -> (word -> count)
    word_counts: HashMap<String, HashMap<String, f64>>,
    /// Total word count per class
    class_word_totals: HashMap<String, f64>,
    /// Document count per class
    class_doc_counts: HashMap<String, usize>,
    /// Total documents
    total_docs: usize,
    /// Vocabulary of all known words
    vocabulary: HashMap<String, usize>,
    /// Laplace smoothing factor
    alpha: f64,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl std::fmt::Debug for NaiveBayesSentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NaiveBayesSentiment")
            .field("total_docs", &self.total_docs)
            .field("vocabulary_size", &self.vocabulary.len())
            .field("alpha", &self.alpha)
            .field("classes", &self.class_doc_counts.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl NaiveBayesSentiment {
    /// Create a new Naive Bayes sentiment classifier
    pub fn new() -> Self {
        Self {
            word_counts: HashMap::new(),
            class_word_totals: HashMap::new(),
            class_doc_counts: HashMap::new(),
            total_docs: 0,
            vocabulary: HashMap::new(),
            alpha: 1.0,
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }

    /// Set the Laplace smoothing parameter
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Train the classifier on labeled examples
    ///
    /// # Arguments
    /// * `texts` - Training text samples
    /// * `labels` - Corresponding labels (e.g., "positive", "negative", "neutral")
    pub fn train(&mut self, texts: &[&str], labels: &[&str]) -> Result<()> {
        if texts.len() != labels.len() {
            return Err(TextError::InvalidInput(
                "texts and labels must have the same length".into(),
            ));
        }

        if texts.is_empty() {
            return Err(TextError::InvalidInput("No training data provided".into()));
        }

        for (text, &label) in texts.iter().zip(labels.iter()) {
            let tokens = self.tokenizer.tokenize(text)?;

            // Update class document count
            *self.class_doc_counts.entry(label.to_string()).or_insert(0) += 1;
            self.total_docs += 1;

            // Update word counts for this class
            let class_words = self.word_counts.entry(label.to_string()).or_default();

            for token in &tokens {
                let lower = token.to_lowercase();
                *class_words.entry(lower.clone()).or_insert(0.0) += 1.0;
                *self
                    .class_word_totals
                    .entry(label.to_string())
                    .or_insert(0.0) += 1.0;

                // Add to vocabulary
                let vocab_len = self.vocabulary.len();
                self.vocabulary.entry(lower).or_insert(vocab_len);
            }
        }

        Ok(())
    }

    /// Predict the class label for a text
    pub fn predict(&self, text: &str) -> Result<String> {
        let (label, _) = self.predict_with_score(text)?;
        Ok(label)
    }

    /// Predict the class label with log-probability scores
    pub fn predict_with_score(&self, text: &str) -> Result<(String, f64)> {
        if self.total_docs == 0 {
            return Err(TextError::ModelNotFitted(
                "Classifier not trained. Call train() first".into(),
            ));
        }

        let tokens = self.tokenizer.tokenize(text)?;
        let vocab_size = self.vocabulary.len() as f64;

        let mut best_label = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for (label, &doc_count) in &self.class_doc_counts {
            // Log prior: P(class)
            let log_prior = (doc_count as f64 / self.total_docs as f64).ln();

            // Log likelihood: sum of log P(word|class) for each word
            let class_words = self.word_counts.get(label);
            let class_total = self.class_word_totals.get(label).copied().unwrap_or(0.0);

            let mut log_likelihood = 0.0;

            for token in &tokens {
                let lower = token.to_lowercase();
                let word_count = class_words
                    .and_then(|wc| wc.get(&lower))
                    .copied()
                    .unwrap_or(0.0);

                // Laplace smoothing: P(word|class) = (count + alpha) / (total + alpha * vocab_size)
                let prob = (word_count + self.alpha) / (class_total + self.alpha * vocab_size);
                log_likelihood += prob.ln();
            }

            let score = log_prior + log_likelihood;
            if score > best_score {
                best_score = score;
                best_label = label.clone();
            }
        }

        Ok((best_label, best_score))
    }

    /// Predict probabilities for all classes
    pub fn predict_proba(&self, text: &str) -> Result<HashMap<String, f64>> {
        if self.total_docs == 0 {
            return Err(TextError::ModelNotFitted("Classifier not trained".into()));
        }

        let tokens = self.tokenizer.tokenize(text)?;
        let vocab_size = self.vocabulary.len() as f64;

        let mut log_scores: Vec<(String, f64)> = Vec::new();

        for (label, &doc_count) in &self.class_doc_counts {
            let log_prior = (doc_count as f64 / self.total_docs as f64).ln();

            let class_words = self.word_counts.get(label);
            let class_total = self.class_word_totals.get(label).copied().unwrap_or(0.0);

            let mut log_likelihood = 0.0;
            for token in &tokens {
                let lower = token.to_lowercase();
                let word_count = class_words
                    .and_then(|wc| wc.get(&lower))
                    .copied()
                    .unwrap_or(0.0);

                let prob = (word_count + self.alpha) / (class_total + self.alpha * vocab_size);
                log_likelihood += prob.ln();
            }

            log_scores.push((label.clone(), log_prior + log_likelihood));
        }

        // Convert log-scores to probabilities using log-sum-exp trick
        let max_score = log_scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);

        let sum_exp: f64 = log_scores.iter().map(|(_, s)| (s - max_score).exp()).sum();

        let mut probas = HashMap::new();
        for (label, score) in &log_scores {
            let prob = (score - max_score).exp() / sum_exp;
            probas.insert(label.clone(), prob);
        }

        Ok(probas)
    }

    /// Get the classes this classifier knows about
    pub fn classes(&self) -> Vec<String> {
        self.class_doc_counts.keys().cloned().collect()
    }
}

impl Default for NaiveBayesSentiment {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Aspect-Based Sentiment ─────────────────────────────────────────────────

/// An aspect with its associated sentiment
#[derive(Debug, Clone)]
pub struct AspectSentiment {
    /// The aspect/entity name
    pub aspect: String,
    /// The sentiment for this aspect
    pub sentiment: Sentiment,
    /// The sentiment score
    pub score: f64,
    /// The relevant text snippet
    pub context: String,
}

/// Aspect-based sentiment analyzer
///
/// Extracts sentiment for specific aspects mentioned in text.
/// Uses a window-based approach: for each aspect mention found,
/// computes sentiment from surrounding words.
pub struct AspectSentimentAnalyzer {
    /// The sentiment lexicon
    lexicon: SentimentLexicon,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Window size around aspect for sentiment extraction
    context_window: usize,
    /// Negation words
    negation_words: Vec<String>,
}

impl AspectSentimentAnalyzer {
    /// Create a new aspect-based sentiment analyzer
    pub fn new() -> Self {
        Self {
            lexicon: SentimentLexicon::with_basiclexicon(),
            tokenizer: Box::new(WordTokenizer::default()),
            context_window: 5,
            negation_words: vec![
                "not".to_string(),
                "no".to_string(),
                "never".to_string(),
                "n't".to_string(),
                "without".to_string(),
            ],
        }
    }

    /// Set a custom lexicon
    pub fn with_lexicon(mut self, lexicon: SentimentLexicon) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Set the context window size
    pub fn with_context_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }

    /// Extract sentiment for specific aspects in text
    ///
    /// # Arguments
    /// * `text` - The text to analyze
    /// * `aspects` - List of aspect keywords to look for
    pub fn analyze(&self, text: &str, aspects: &[&str]) -> Result<Vec<AspectSentiment>> {
        let tokens = self.tokenizer.tokenize(text)?;
        let lower_tokens: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();

        let mut results = Vec::new();

        for &aspect in aspects {
            let aspect_lower = aspect.to_lowercase();
            let aspect_tokens: Vec<String> =
                aspect_lower.split_whitespace().map(String::from).collect();

            // Find all positions where the aspect occurs
            for pos in 0..lower_tokens.len() {
                // Check if the aspect (possibly multi-word) starts at this position
                let aspect_matches = if aspect_tokens.len() == 1 {
                    lower_tokens[pos] == aspect_tokens[0]
                } else {
                    pos + aspect_tokens.len() <= lower_tokens.len()
                        && aspect_tokens
                            .iter()
                            .enumerate()
                            .all(|(j, at)| lower_tokens[pos + j] == *at)
                };

                if !aspect_matches {
                    continue;
                }

                // Extract sentiment from surrounding context
                // Respect discourse boundaries: "but", "however", "although", "yet"
                let discourse_markers = [
                    "but",
                    "however",
                    "although",
                    "yet",
                    "though",
                    "nevertheless",
                ];
                let mut start = pos.saturating_sub(self.context_window);
                let end = (pos + aspect_tokens.len() + self.context_window).min(lower_tokens.len());

                // Adjust start to not cross discourse markers before the aspect
                // Find the last discourse marker before pos and set start after it
                let initial_start = start;
                if let Some(last_marker_idx) = (initial_start..pos)
                    .rev()
                    .find(|&i| discourse_markers.contains(&lower_tokens[i].as_str()))
                {
                    start = last_marker_idx + 1;
                }
                // Adjust end to not cross discourse markers after the aspect
                let mut effective_end = end;
                for i in (pos + aspect_tokens.len())..end {
                    if discourse_markers.contains(&lower_tokens[i].as_str()) {
                        effective_end = i;
                        break;
                    }
                }

                let mut score = 0.0;
                let mut is_negated = false;

                for i in start..effective_end {
                    // Skip the aspect tokens themselves
                    if i >= pos && i < pos + aspect_tokens.len() {
                        continue;
                    }

                    let token = &lower_tokens[i];

                    // Track negation
                    if self.negation_words.contains(token) {
                        is_negated = true;
                        continue;
                    }

                    let word_score = self.lexicon.get_score(token);
                    if word_score != 0.0 {
                        if is_negated {
                            score -= word_score;
                            is_negated = false;
                        } else {
                            score += word_score;
                        }
                    }
                }

                // Build context string
                let context_tokens = &tokens[start..end];
                let context = context_tokens.join(" ");

                results.push(AspectSentiment {
                    aspect: aspect.to_string(),
                    sentiment: Sentiment::from_score(score),
                    score,
                    context,
                });
            }
        }

        Ok(results)
    }
}

impl Default for AspectSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Document Sentiment Aggregation ──────────────────────────────────────────

/// Result of aggregating sentiment across multiple texts
#[derive(Debug, Clone)]
pub struct AggregatedSentiment {
    /// Mean sentiment score across all texts
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_score: f64,
    /// Overall sentiment based on mean score
    pub overall_sentiment: Sentiment,
    /// Proportion of positive texts
    pub positive_ratio: f64,
    /// Proportion of negative texts
    pub negative_ratio: f64,
    /// Proportion of neutral texts
    pub neutral_ratio: f64,
    /// Number of texts analyzed
    pub count: usize,
    /// Individual results
    pub results: Vec<SentimentResult>,
}

/// Aggregate sentiment analysis results across multiple texts/documents
pub fn aggregate_sentiment(results: &[SentimentResult]) -> AggregatedSentiment {
    if results.is_empty() {
        return AggregatedSentiment {
            mean_score: 0.0,
            std_score: 0.0,
            overall_sentiment: Sentiment::Neutral,
            positive_ratio: 0.0,
            negative_ratio: 0.0,
            neutral_ratio: 0.0,
            count: 0,
            results: Vec::new(),
        };
    }

    let n = results.len() as f64;

    // Calculate mean score
    let sum: f64 = results.iter().map(|r| r.score).sum();
    let mean_score = sum / n;

    // Calculate standard deviation
    let variance: f64 = results
        .iter()
        .map(|r| (r.score - mean_score).powi(2))
        .sum::<f64>()
        / n;
    let std_score = variance.sqrt();

    // Count sentiments
    let mut pos = 0;
    let mut neg = 0;
    let mut neu = 0;
    for r in results {
        match r.sentiment {
            Sentiment::Positive => pos += 1,
            Sentiment::Negative => neg += 1,
            Sentiment::Neutral => neu += 1,
        }
    }

    AggregatedSentiment {
        mean_score,
        std_score,
        overall_sentiment: Sentiment::from_score(mean_score),
        positive_ratio: pos as f64 / n,
        negative_ratio: neg as f64 / n,
        neutral_ratio: neu as f64 / n,
        count: results.len(),
        results: results.to_vec(),
    }
}

/// Analyze and aggregate sentiment for a batch of texts
pub fn analyze_and_aggregate(texts: &[&str]) -> Result<AggregatedSentiment> {
    let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();
    let results = analyzer.analyze_batch(texts)?;
    Ok(aggregate_sentiment(&results))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Lexicon Tests ───────────────────────────────────────────────

    #[test]
    fn test_sentimentlexicon() {
        let mut lexicon = SentimentLexicon::new();
        lexicon.add_word("happy".to_string(), 2.0);
        lexicon.add_word("sad".to_string(), -2.0);

        assert_eq!(lexicon.get_score("happy"), 2.0);
        assert_eq!(lexicon.get_score("sad"), -2.0);
        assert_eq!(lexicon.get_score("unknown"), 0.0);
    }

    #[test]
    fn test_basic_sentiment_analysis() {
        let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();

        let positive_result = analyzer
            .analyze("This is a wonderful day!")
            .expect("Operation failed");
        assert_eq!(positive_result.sentiment, Sentiment::Positive);
        assert!(positive_result.score > 0.0);

        let negative_result = analyzer
            .analyze("This is terrible and awful")
            .expect("Operation failed");
        assert_eq!(negative_result.sentiment, Sentiment::Negative);
        assert!(negative_result.score < 0.0);

        let neutral_result = analyzer
            .analyze("This is a book")
            .expect("Operation failed");
        assert_eq!(neutral_result.sentiment, Sentiment::Neutral);
    }

    #[test]
    fn test_negation_handling() {
        let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();

        let negated_result = analyzer
            .analyze("This is not good")
            .expect("Operation failed");
        assert_eq!(negated_result.sentiment, Sentiment::Negative);
        assert!(negated_result.score < 0.0);
    }

    #[test]
    fn test_rule_based_sentiment() {
        let analyzer = RuleBasedSentimentAnalyzer::with_basiclexicon();

        let intensified_result = analyzer
            .analyze("This is very good")
            .expect("Operation failed");
        let normal_result = analyzer.analyze("This is good").expect("Operation failed");

        assert!(intensified_result.score > normal_result.score);
    }

    #[test]
    fn test_sentiment_batch_analysis() {
        let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();
        let texts = vec!["I love this", "I hate this", "This is okay"];

        let results = analyzer.analyze_batch(&texts).expect("Operation failed");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].sentiment, Sentiment::Positive);
        assert_eq!(results[1].sentiment, Sentiment::Negative);
    }

    // ─── VADER Tests ─────────────────────────────────────────────────

    #[test]
    fn test_vader_positive() {
        let vader = VaderSentimentAnalyzer::new();
        let result = vader
            .analyze("This movie is amazing and wonderful")
            .expect("analyze");
        assert_eq!(result.sentiment, Sentiment::Positive);
        assert!(result.compound > 0.0);
    }

    #[test]
    fn test_vader_negative() {
        let vader = VaderSentimentAnalyzer::new();
        let result = vader
            .analyze("This movie is terrible and awful")
            .expect("analyze");
        assert_eq!(result.sentiment, Sentiment::Negative);
        assert!(result.compound < 0.0);
    }

    #[test]
    fn test_vader_neutral() {
        let vader = VaderSentimentAnalyzer::new();
        let result = vader.analyze("The sky is blue").expect("analyze");
        assert_eq!(result.sentiment, Sentiment::Neutral);
    }

    #[test]
    fn test_vader_negation() {
        let vader = VaderSentimentAnalyzer::new();
        let result = vader.analyze("This is not good at all").expect("analyze");
        assert!(result.compound < 0.0, "Negated positive should be negative");
    }

    #[test]
    fn test_vader_intensifier() {
        let vader = VaderSentimentAnalyzer::new();
        let base = vader.analyze("This is good").expect("analyze");
        let intensified = vader.analyze("This is very good").expect("analyze");
        assert!(
            intensified.compound > base.compound,
            "Intensified should score higher: {} vs {}",
            intensified.compound,
            base.compound
        );
    }

    #[test]
    fn test_vader_but_clause() {
        let vader = VaderSentimentAnalyzer::new();
        let result = vader
            .analyze("The food was good but the service was terrible")
            .expect("analyze");
        // After "but" has more weight, so service's negative should dominate
        assert!(result.compound < 0.0);
    }

    #[test]
    fn test_vader_caps_emphasis() {
        let vader = VaderSentimentAnalyzer::new();
        let normal = vader.analyze("This is good").expect("analyze");
        let caps = vader.analyze("This is GOOD").expect("analyze");
        assert!(
            caps.compound >= normal.compound,
            "CAPS should score higher or equal"
        );
    }

    #[test]
    fn test_vader_batch() {
        let vader = VaderSentimentAnalyzer::new();
        let texts = vec!["I love this!", "I hate this!"];
        let results = vader.analyze_batch(&texts).expect("batch");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].sentiment, Sentiment::Positive);
        assert_eq!(results[1].sentiment, Sentiment::Negative);
    }

    #[test]
    fn test_vader_compound_range() {
        let vader = VaderSentimentAnalyzer::new();
        let result = vader
            .analyze("This is the most absolutely amazing incredible thing ever!!!")
            .expect("analyze");
        assert!(result.compound >= -1.0 && result.compound <= 1.0);
    }

    // ─── Naive Bayes Tests ───────────────────────────────────────────

    #[test]
    fn test_naive_bayes_train_predict() {
        let mut clf = NaiveBayesSentiment::new();

        let texts = vec![
            "I love this product it is amazing",
            "Great quality excellent experience",
            "Wonderful service very happy",
            "This is terrible and awful",
            "Horrible experience very bad",
            "Worst product I have ever bought",
        ];
        let labels = vec![
            "positive", "positive", "positive", "negative", "negative", "negative",
        ];

        clf.train(&texts, &labels).expect("training failed");

        // Positive prediction
        let pred = clf.predict("This is amazing and great").expect("predict");
        assert_eq!(pred, "positive");

        // Negative prediction
        let pred = clf
            .predict("This is terrible and horrible")
            .expect("predict");
        assert_eq!(pred, "negative");
    }

    #[test]
    fn test_naive_bayes_predict_proba() {
        let mut clf = NaiveBayesSentiment::new();

        let texts = vec![
            "good great excellent",
            "good wonderful amazing",
            "bad terrible awful",
            "bad horrible disgusting",
        ];
        let labels = vec!["positive", "positive", "negative", "negative"];

        clf.train(&texts, &labels).expect("training failed");

        let probas = clf.predict_proba("good excellent").expect("predict_proba");
        assert!(probas.contains_key("positive"));
        assert!(probas.contains_key("negative"));

        // Positive should have higher probability
        let pos_prob = probas.get("positive").copied().unwrap_or(0.0);
        let neg_prob = probas.get("negative").copied().unwrap_or(0.0);
        assert!(pos_prob > neg_prob);

        // Probabilities should sum to ~1
        let total: f64 = probas.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_naive_bayes_not_trained() {
        let clf = NaiveBayesSentiment::new();
        let result = clf.predict("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_naive_bayes_classes() {
        let mut clf = NaiveBayesSentiment::new();
        let texts = vec!["a", "b", "c"];
        let labels = vec!["pos", "neg", "pos"];
        clf.train(&texts, &labels).expect("train");

        let classes = clf.classes();
        assert_eq!(classes.len(), 2);
    }

    // ─── Aspect-Based Sentiment Tests ────────────────────────────────

    #[test]
    fn test_aspect_sentiment_basic() {
        let analyzer = AspectSentimentAnalyzer::new();

        let results = analyzer
            .analyze(
                "The food was excellent but the service was terrible",
                &["food", "service"],
            )
            .expect("analyze");

        assert_eq!(results.len(), 2);

        let food_result = results.iter().find(|r| r.aspect == "food");
        assert!(food_result.is_some());
        let food = food_result.expect("food aspect");
        assert_eq!(food.sentiment, Sentiment::Positive);

        let service_result = results.iter().find(|r| r.aspect == "service");
        assert!(service_result.is_some());
        let service = service_result.expect("service aspect");
        assert_eq!(service.sentiment, Sentiment::Negative);
    }

    #[test]
    fn test_aspect_sentiment_negation() {
        let analyzer = AspectSentimentAnalyzer::new();

        let results = analyzer
            .analyze("The price was not good", &["price"])
            .expect("analyze");

        assert!(!results.is_empty());
        // "not good" should flip to negative
        assert_eq!(results[0].sentiment, Sentiment::Negative);
    }

    #[test]
    fn test_aspect_sentiment_no_match() {
        let analyzer = AspectSentimentAnalyzer::new();
        let results = analyzer
            .analyze("The sky is blue", &["food", "service"])
            .expect("analyze");
        assert!(results.is_empty());
    }

    #[test]
    fn test_aspect_with_custom_window() {
        let analyzer = AspectSentimentAnalyzer::new().with_context_window(2);
        let results = analyzer
            .analyze("The food here is really great and beautiful", &["food"])
            .expect("analyze");
        assert!(!results.is_empty());
    }

    // ─── Aggregation Tests ───────────────────────────────────────────

    #[test]
    fn test_aggregate_sentiment() {
        let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();
        let results = analyzer
            .analyze_batch(&["I love this", "I love this too", "This is terrible"])
            .expect("batch");

        let agg = aggregate_sentiment(&results);
        assert_eq!(agg.count, 3);
        assert!(agg.mean_score > 0.0); // 2 positive, 1 negative
        assert!(agg.positive_ratio > 0.5);
        assert!(agg.std_score > 0.0);
    }

    #[test]
    fn test_aggregate_empty() {
        let agg = aggregate_sentiment(&[]);
        assert_eq!(agg.count, 0);
        assert_eq!(agg.overall_sentiment, Sentiment::Neutral);
    }

    #[test]
    fn test_analyze_and_aggregate() {
        let texts = vec!["I love this product", "It is amazing", "Very good quality"];
        let agg = analyze_and_aggregate(&texts).expect("aggregate");
        assert_eq!(agg.count, 3);
        assert!(agg.mean_score > 0.0);
        assert_eq!(agg.overall_sentiment, Sentiment::Positive);
    }

    #[test]
    fn test_sentiment_display() {
        assert_eq!(format!("{}", Sentiment::Positive), "Positive");
        assert_eq!(format!("{}", Sentiment::Negative), "Negative");
        assert_eq!(format!("{}", Sentiment::Neutral), "Neutral");
    }

    #[test]
    fn test_sentiment_from_score_thresholds() {
        assert_eq!(Sentiment::from_score(0.1), Sentiment::Positive);
        assert_eq!(Sentiment::from_score(-0.1), Sentiment::Negative);
        assert_eq!(Sentiment::from_score(0.0), Sentiment::Neutral);
        assert_eq!(Sentiment::from_score(0.03), Sentiment::Neutral);
    }
}
