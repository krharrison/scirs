//! Python bindings for scirs2-text
//!
//! This module provides Python bindings for text processing operations,
//! including tokenization, vectorization, sentiment analysis, stemming,
//! string similarity metrics, and text cleaning.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// NumPy types for Python array interface
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// Direct imports from scirs2-text
use scirs2_text::{
    // Cleansing functions
    cleansing::{
        expand_contractions, normalize_unicode, normalize_whitespace, remove_accents,
        replace_emails, replace_urls, strip_html_tags,
    },
    // Sentiment
    sentiment::{LexiconSentimentAnalyzer, Sentiment},
    // Stemming
    stemming::{LancasterStemmer, PorterStemmer, SnowballStemmer, Stemmer},
    // Tokenization
    tokenize::{
        CharacterTokenizer, NgramTokenizer, RegexTokenizer, SentenceTokenizer, Tokenizer,
        WhitespaceTokenizer, WordTokenizer,
    },
    // Vectorization
    vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer},
};

// ========================================
// TOKENIZATION
// ========================================

/// Word tokenizer
#[pyclass(name = "WordTokenizer")]
pub struct PyWordTokenizer {
    inner: WordTokenizer,
}

#[pymethods]
impl PyWordTokenizer {
    #[new]
    #[pyo3(signature = (lowercase=true))]
    fn new(lowercase: bool) -> Self {
        Self {
            inner: WordTokenizer::new(lowercase),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner
            .tokenize(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))
    }

    fn tokenize_batch(&self, texts: &Bound<'_, PyList>) -> PyResult<Vec<Vec<String>>> {
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let text_strs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        self.inner
            .tokenize_batch(&text_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Batch tokenization failed: {}", e)))
    }
}

/// Sentence tokenizer
#[pyclass(name = "SentenceTokenizer")]
pub struct PySentenceTokenizer {
    inner: SentenceTokenizer,
}

#[pymethods]
impl PySentenceTokenizer {
    #[new]
    fn new() -> Self {
        Self {
            inner: SentenceTokenizer::new(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner
            .tokenize(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))
    }
}

/// Character tokenizer
#[pyclass(name = "CharacterTokenizer")]
pub struct PyCharacterTokenizer {
    inner: CharacterTokenizer,
}

#[pymethods]
impl PyCharacterTokenizer {
    #[new]
    #[pyo3(signature = (use_grapheme_clusters=true))]
    fn new(use_grapheme_clusters: bool) -> Self {
        Self {
            inner: CharacterTokenizer::new(use_grapheme_clusters),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner
            .tokenize(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))
    }
}

/// N-gram tokenizer
#[pyclass(name = "NgramTokenizer")]
pub struct PyNgramTokenizer {
    inner: NgramTokenizer,
}

#[pymethods]
impl PyNgramTokenizer {
    #[new]
    #[pyo3(signature = (n=2))]
    fn new(n: usize) -> PyResult<Self> {
        let tokenizer = NgramTokenizer::new(n).map_err(|e| {
            PyRuntimeError::new_err(format!("NgramTokenizer creation failed: {}", e))
        })?;
        Ok(Self { inner: tokenizer })
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner
            .tokenize(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))
    }
}

/// Whitespace tokenizer
#[pyclass(name = "WhitespaceTokenizer")]
pub struct PyWhitespaceTokenizer {
    inner: WhitespaceTokenizer,
}

#[pymethods]
impl PyWhitespaceTokenizer {
    #[new]
    fn new() -> Self {
        Self {
            inner: WhitespaceTokenizer::new(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner
            .tokenize(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))
    }
}

/// Regex tokenizer
#[pyclass(name = "RegexTokenizer")]
pub struct PyRegexTokenizer {
    inner: RegexTokenizer,
}

#[pymethods]
impl PyRegexTokenizer {
    #[new]
    #[pyo3(signature = (pattern, gaps=false))]
    fn new(pattern: &str, gaps: bool) -> PyResult<Self> {
        let tokenizer = RegexTokenizer::new(pattern, gaps).map_err(|e| {
            PyRuntimeError::new_err(format!("RegexTokenizer creation failed: {}", e))
        })?;
        Ok(Self { inner: tokenizer })
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner
            .tokenize(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))
    }
}

// ========================================
// VECTORIZATION
// ========================================

/// Count vectorizer (bag-of-words)
#[pyclass(name = "CountVectorizer")]
pub struct PyCountVectorizer {
    inner: CountVectorizer,
}

#[pymethods]
impl PyCountVectorizer {
    #[new]
    #[pyo3(signature = (binary=false))]
    fn new(binary: bool) -> Self {
        Self {
            inner: CountVectorizer::new(binary),
        }
    }

    fn fit(&mut self, texts: &Bound<'_, PyList>) -> PyResult<()> {
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let text_strs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        self.inner
            .fit(&text_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(&self, py: Python, text: &str) -> PyResult<Py<PyArray1<f64>>> {
        let result = self
            .inner
            .transform(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn transform_batch(
        &self,
        py: Python,
        texts: &Bound<'_, PyList>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let text_strs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        let result = self
            .inner
            .transform_batch(&text_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Batch transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        texts: &Bound<'_, PyList>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let text_strs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        let result = self
            .inner
            .fit_transform(&text_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn vocabulary_size(&self) -> usize {
        self.inner.vocabulary_size()
    }

    fn get_feature_names(&self) -> Vec<String> {
        let vocab = self.inner.vocabulary();
        let mut features: Vec<(usize, String)> = vocab
            .token_to_index()
            .iter()
            .map(|(token, &idx)| (idx, token.clone()))
            .collect();
        features.sort_by_key(|(idx, _)| *idx);
        features.into_iter().map(|(_, token)| token).collect()
    }
}

/// TF-IDF vectorizer
#[pyclass(name = "TfidfVectorizer")]
pub struct PyTfidfVectorizer {
    inner: TfidfVectorizer,
}

#[pymethods]
impl PyTfidfVectorizer {
    #[new]
    #[pyo3(signature = (lowercase=true, norm=true, norm_type=None))]
    fn new(lowercase: bool, norm: bool, norm_type: Option<String>) -> Self {
        Self {
            inner: TfidfVectorizer::new(lowercase, norm, norm_type),
        }
    }

    fn fit(&mut self, texts: &Bound<'_, PyList>) -> PyResult<()> {
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let text_strs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        self.inner
            .fit(&text_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(&self, py: Python, text: &str) -> PyResult<Py<PyArray1<f64>>> {
        let result = self
            .inner
            .transform(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn transform_batch(
        &self,
        py: Python,
        texts: &Bound<'_, PyList>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let text_strs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        let result = self
            .inner
            .transform_batch(&text_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Batch transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        texts: &Bound<'_, PyList>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let text_strs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        let result = self
            .inner
            .fit_transform(&text_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn vocabulary_size(&self) -> usize {
        self.inner.vocabulary_size()
    }

    fn get_feature_names(&self) -> Vec<String> {
        let vocab = self.inner.vocabulary();
        let mut features: Vec<(usize, String)> = vocab
            .token_to_index()
            .iter()
            .map(|(token, &idx)| (idx, token.clone()))
            .collect();
        features.sort_by_key(|(idx, _)| *idx);
        features.into_iter().map(|(_, token)| token).collect()
    }
}

// ========================================
// SENTIMENT ANALYSIS
// ========================================

/// Convert Sentiment enum to string for Python
fn sentiment_to_string(sentiment: &Sentiment) -> String {
    match sentiment {
        Sentiment::Positive => "positive".to_string(),
        Sentiment::Negative => "negative".to_string(),
        Sentiment::Neutral => "neutral".to_string(),
    }
}

/// Lexicon-based sentiment analyzer
#[pyclass(name = "LexiconSentimentAnalyzer")]
pub struct PyLexiconSentimentAnalyzer {
    inner: LexiconSentimentAnalyzer,
}

#[pymethods]
impl PyLexiconSentimentAnalyzer {
    #[new]
    fn new() -> Self {
        Self {
            inner: LexiconSentimentAnalyzer::with_basiclexicon(),
        }
    }

    fn analyze(&self, py: Python, text: &str) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .analyze(text)
            .map_err(|e| PyRuntimeError::new_err(format!("Sentiment analysis failed: {}", e)))?;

        // Convert to Python dict
        let dict = PyDict::new(py);
        dict.set_item("sentiment", sentiment_to_string(&result.sentiment))?;
        dict.set_item("score", result.score)?;
        dict.set_item("confidence", result.confidence)?;

        let word_counts = PyDict::new(py);
        word_counts.set_item("positive_words", result.word_counts.positive_words)?;
        word_counts.set_item("negative_words", result.word_counts.negative_words)?;
        word_counts.set_item("neutral_words", result.word_counts.neutral_words)?;
        word_counts.set_item("total_words", result.word_counts.total_words)?;
        dict.set_item("word_counts", word_counts)?;

        Ok(dict.into())
    }
}

// ========================================
// STEMMING
// ========================================

/// Porter stemmer
#[pyclass(name = "PorterStemmer")]
pub struct PyPorterStemmer {
    inner: PorterStemmer,
}

#[pymethods]
impl PyPorterStemmer {
    #[new]
    fn new() -> Self {
        Self {
            inner: PorterStemmer::new(),
        }
    }

    fn stem(&self, word: &str) -> PyResult<String> {
        self.inner
            .stem(word)
            .map_err(|e| PyRuntimeError::new_err(format!("Stemming failed: {}", e)))
    }

    fn stem_batch(&self, words: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
        let words_owned: Vec<String> = words
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let word_strs: Vec<&str> = words_owned.iter().map(|s| s.as_str()).collect();
        self.inner
            .stem_batch(&word_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Batch stemming failed: {}", e)))
    }
}

/// Snowball stemmer
#[pyclass(name = "SnowballStemmer")]
pub struct PySnowballStemmer {
    inner: SnowballStemmer,
}

#[pymethods]
impl PySnowballStemmer {
    #[new]
    #[pyo3(signature = (language="english"))]
    fn new(language: &str) -> PyResult<Self> {
        let stemmer = SnowballStemmer::new(language).map_err(|e| {
            PyRuntimeError::new_err(format!("SnowballStemmer creation failed: {}", e))
        })?;
        Ok(Self { inner: stemmer })
    }

    fn stem(&self, word: &str) -> PyResult<String> {
        self.inner
            .stem(word)
            .map_err(|e| PyRuntimeError::new_err(format!("Stemming failed: {}", e)))
    }

    fn stem_batch(&self, words: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
        let words_owned: Vec<String> = words
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let word_strs: Vec<&str> = words_owned.iter().map(|s| s.as_str()).collect();
        self.inner
            .stem_batch(&word_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Batch stemming failed: {}", e)))
    }
}

/// Lancaster stemmer
#[pyclass(name = "LancasterStemmer")]
pub struct PyLancasterStemmer {
    inner: LancasterStemmer,
}

#[pymethods]
impl PyLancasterStemmer {
    #[new]
    fn new() -> Self {
        Self {
            inner: LancasterStemmer::new(),
        }
    }

    fn stem(&self, word: &str) -> PyResult<String> {
        self.inner
            .stem(word)
            .map_err(|e| PyRuntimeError::new_err(format!("Stemming failed: {}", e)))
    }

    fn stem_batch(&self, words: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
        let words_owned: Vec<String> = words
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        let word_strs: Vec<&str> = words_owned.iter().map(|s| s.as_str()).collect();
        self.inner
            .stem_batch(&word_strs)
            .map_err(|e| PyRuntimeError::new_err(format!("Batch stemming failed: {}", e)))
    }
}

// ========================================
// STRING SIMILARITY METRICS
// ========================================

/// Levenshtein distance
#[pyfunction]
fn levenshtein_distance_py(s1: &str, s2: &str) -> usize {
    scirs2_text::distance::levenshtein_distance(s1, s2)
}

/// Cosine similarity between two vectors
#[pyfunction]
fn cosine_similarity_py(
    vec1: &Bound<'_, PyArray1<f64>>,
    vec2: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let v1_binding = vec1.readonly();
    let v2_binding = vec2.readonly();
    let v1_view = v1_binding.as_array();
    let v2_view = v2_binding.as_array();

    scirs2_text::distance::cosine_similarity(v1_view, v2_view)
        .map_err(|e| PyRuntimeError::new_err(format!("Similarity calculation failed: {}", e)))
}

/// Jaccard similarity between two token sets
#[pyfunction]
fn jaccard_similarity_py(s1: &str, s2: &str) -> PyResult<f64> {
    scirs2_text::distance::jaccard_similarity(s1, s2, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Similarity calculation failed: {}", e)))
}

// ========================================
// TEXT CLEANING
// ========================================

/// Strip HTML tags
#[pyfunction]
fn strip_html_tags_py(text: &str) -> String {
    strip_html_tags(text)
}

/// Replace URLs with a replacement string
#[pyfunction]
#[pyo3(signature = (text, replacement="<URL>"))]
fn replace_urls_py(text: &str, replacement: &str) -> String {
    replace_urls(text, replacement)
}

/// Replace emails with a replacement string
#[pyfunction]
#[pyo3(signature = (text, replacement="<EMAIL>"))]
fn replace_emails_py(text: &str, replacement: &str) -> String {
    replace_emails(text, replacement)
}

/// Expand contractions
#[pyfunction]
fn expand_contractions_py(text: &str) -> String {
    expand_contractions(text)
}

/// Normalize Unicode
#[pyfunction]
fn normalize_unicode_py(text: &str) -> PyResult<String> {
    normalize_unicode(text)
        .map_err(|e| PyRuntimeError::new_err(format!("Unicode normalization failed: {}", e)))
}

/// Normalize whitespace
#[pyfunction]
fn normalize_whitespace_py(text: &str) -> String {
    normalize_whitespace(text)
}

/// Remove accents
#[pyfunction]
fn remove_accents_py(text: &str) -> String {
    remove_accents(text)
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tokenization
    m.add_class::<PyWordTokenizer>()?;
    m.add_class::<PySentenceTokenizer>()?;
    m.add_class::<PyCharacterTokenizer>()?;
    m.add_class::<PyNgramTokenizer>()?;
    m.add_class::<PyWhitespaceTokenizer>()?;
    m.add_class::<PyRegexTokenizer>()?;

    // Vectorization
    m.add_class::<PyCountVectorizer>()?;
    m.add_class::<PyTfidfVectorizer>()?;

    // Sentiment analysis
    m.add_class::<PyLexiconSentimentAnalyzer>()?;

    // Stemming
    m.add_class::<PyPorterStemmer>()?;
    m.add_class::<PySnowballStemmer>()?;
    m.add_class::<PyLancasterStemmer>()?;

    // String similarity metrics
    m.add_function(wrap_pyfunction!(levenshtein_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_py, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity_py, m)?)?;

    // Text cleaning
    m.add_function(wrap_pyfunction!(strip_html_tags_py, m)?)?;
    m.add_function(wrap_pyfunction!(replace_urls_py, m)?)?;
    m.add_function(wrap_pyfunction!(replace_emails_py, m)?)?;
    m.add_function(wrap_pyfunction!(expand_contractions_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_unicode_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_whitespace_py, m)?)?;
    m.add_function(wrap_pyfunction!(remove_accents_py, m)?)?;

    Ok(())
}
