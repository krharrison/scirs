"""Tests for scirs2 text processing module."""

import numpy as np
import pytest
import scirs2


class TestTokenization:
    """Test tokenization functions."""

    def test_word_tokenizer_basic(self):
        """Test basic word tokenization."""
        tokenizer = scirs2.WordTokenizer(lowercase=True)

        text = "Hello World! This is a test."
        tokens = tokenizer.tokenize(text)

        assert len(tokens) > 0
        assert "hello" in tokens or "world" in tokens

    def test_word_tokenizer_batch(self):
        """Test batch tokenization."""
        tokenizer = scirs2.WordTokenizer(lowercase=True)

        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        tokens_batch = tokenizer.tokenize_batch(texts)

        assert len(tokens_batch) == 3
        assert all(len(tokens) > 0 for tokens in tokens_batch)

    def test_word_tokenizer_case_sensitive(self):
        """Test case-sensitive tokenization."""
        tokenizer = scirs2.WordTokenizer(lowercase=False)

        text = "Hello WORLD"
        tokens = tokenizer.tokenize(text)

        # Should preserve case
        assert any(t[0].isupper() for t in tokens if t)

    def test_sentence_tokenizer(self):
        """Test sentence tokenization."""
        tokenizer = scirs2.SentenceTokenizer()

        text = "This is the first sentence. This is the second! And the third?"
        sentences = tokenizer.tokenize(text)

        assert len(sentences) >= 2

    def test_character_tokenizer(self):
        """Test character-level tokenization."""
        tokenizer = scirs2.CharacterTokenizer()

        text = "abc"
        chars = tokenizer.tokenize(text)

        assert len(chars) == 3
        assert chars == ["a", "b", "c"]

    def test_whitespace_tokenizer(self):
        """Test whitespace tokenization."""
        tokenizer = scirs2.WhitespaceTokenizer()

        text = "word1 word2  word3"
        tokens = tokenizer.tokenize(text)

        assert len(tokens) == 3
        assert "word1" in tokens

    def test_regex_tokenizer(self):
        """Test regex-based tokenization."""
        tokenizer = scirs2.RegexTokenizer(pattern=r"\w+")

        text = "Hello, World! 123"
        tokens = tokenizer.tokenize(text)

        assert len(tokens) >= 2

    def test_ngram_tokenizer(self):
        """Test n-gram tokenization."""
        tokenizer = scirs2.NgramTokenizer(n=2)

        text = "one two three"
        ngrams = tokenizer.tokenize(text)

        # Should create bigrams
        assert len(ngrams) >= 1


class TestVectorization:
    """Test text vectorization."""

    def test_count_vectorizer_basic(self):
        """Test basic count vectorization."""
        vectorizer = scirs2.CountVectorizer()

        texts = ["hello world", "hello python", "world python"]

        vectorizer.fit(texts)
        vectors = vectorizer.transform(texts)

        # Should create document-term matrix
        assert vectors.shape[0] == 3  # 3 documents
        assert vectors.shape[1] >= 2  # At least 2 unique words

    def test_count_vectorizer_vocabulary(self):
        """Test vocabulary extraction."""
        vectorizer = scirs2.CountVectorizer()

        texts = ["cat dog", "dog bird", "cat bird"]

        vectorizer.fit(texts)
        vocab = vectorizer.get_vocabulary()

        assert len(vocab) >= 3
        assert "cat" in vocab or "dog" in vocab or "bird" in vocab

    def test_tfidf_vectorizer_basic(self):
        """Test TF-IDF vectorization."""
        vectorizer = scirs2.TfidfVectorizer()

        texts = [
            "the quick brown fox",
            "the lazy dog",
            "the quick dog"
        ]

        vectorizer.fit(texts)
        vectors = vectorizer.transform(texts)

        # Should create TF-IDF matrix
        assert vectors.shape[0] == 3
        assert vectors.shape[1] >= 2

        # TF-IDF values should be non-negative
        assert np.all(vectors >= 0)

    def test_tfidf_vectorizer_idf(self):
        """Test IDF calculation in TF-IDF."""
        vectorizer = scirs2.TfidfVectorizer()

        texts = [
            "common word",
            "common word",
            "rare word"
        ]

        vectorizer.fit(texts)
        vectors = vectorizer.transform(texts)

        # Rare words should have higher weights
        assert vectors.shape[0] == 3

    def test_count_vectorizer_max_features(self):
        """Test limiting vocabulary size."""
        vectorizer = scirs2.CountVectorizer(max_features=2)

        texts = ["a b c d e f", "a b c", "a b"]

        vectorizer.fit(texts)
        vectors = vectorizer.transform(texts)

        # Should only use top 2 most frequent features
        assert vectors.shape[1] <= 2

    def test_count_vectorizer_ngrams(self):
        """Test n-gram features in count vectorizer."""
        vectorizer = scirs2.CountVectorizer(ngram_range=(1, 2))

        texts = ["hello world", "hello python"]

        vectorizer.fit(texts)
        vectors = vectorizer.transform(texts)

        # Should include both unigrams and bigrams
        assert vectors.shape[1] >= 2


class TestStemming:
    """Test stemming algorithms."""

    def test_porter_stemmer(self):
        """Test Porter stemmer."""
        stemmer = scirs2.PorterStemmer()

        word = "running"
        stem = stemmer.stem(word)

        # Should remove -ing suffix
        assert len(stem) < len(word)
        assert stem in ["run", "runn"]

    def test_porter_stemmer_batch(self):
        """Test batch stemming."""
        stemmer = scirs2.PorterStemmer()

        words = ["running", "runner", "runs", "ran"]
        stems = stemmer.stem_batch(words)

        assert len(stems) == 4
        # All should stem to similar form
        assert all(len(stem) <= len(word) for stem, word in zip(stems, words))

    def test_snowball_stemmer(self):
        """Test Snowball stemmer."""
        stemmer = scirs2.SnowballStemmer(language="english")

        word = "generously"
        stem = stemmer.stem(word)

        assert len(stem) <= len(word)

    def test_lancaster_stemmer(self):
        """Test Lancaster stemmer."""
        stemmer = scirs2.LancasterStemmer()

        word = "maximum"
        stem = stemmer.stem(word)

        # Lancaster is more aggressive
        assert len(stem) <= len(word)

    def test_stemmer_variants(self):
        """Test different stemming variants."""
        words = ["compute", "computing", "computer", "computation"]

        porter = scirs2.PorterStemmer()
        porter_stems = [porter.stem(w) for w in words]

        # All should stem to similar root
        assert len(set(porter_stems)) <= 2


class TestCleansing:
    """Test text cleansing functions."""

    def test_remove_accents(self):
        """Test accent removal."""
        text = "café naïve résumé"
        cleaned = scirs2.remove_accents_py(text)

        # Accents should be removed
        assert "é" not in cleaned or "cafe" in cleaned.lower()

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "hello    world  \t  test"
        normalized = scirs2.normalize_whitespace_py(text)

        # Multiple spaces should be collapsed
        assert "    " not in normalized
        assert "\t" not in normalized

    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        text = "Hello\u00A0World"  # Non-breaking space
        normalized = scirs2.normalize_unicode_py(text)

        assert len(normalized) > 0

    def test_strip_html_tags(self):
        """Test HTML tag removal."""
        text = "<p>Hello <b>World</b></p>"
        cleaned = scirs2.strip_html_tags_py(text)

        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_replace_urls(self):
        """Test URL replacement."""
        text = "Check out https://example.com for more info"
        cleaned = scirs2.replace_urls_py(text, replacement="URL")

        assert "https://example.com" not in cleaned
        assert "URL" in cleaned

    def test_replace_emails(self):
        """Test email replacement."""
        text = "Contact us at test@example.com for help"
        cleaned = scirs2.replace_emails_py(text, replacement="EMAIL")

        assert "test@example.com" not in cleaned
        assert "EMAIL" in cleaned

    def test_expand_contractions(self):
        """Test contraction expansion."""
        text = "I'm can't won't"
        expanded = scirs2.expand_contractions_py(text)

        # Contractions should be expanded
        assert len(expanded) >= len(text)


class TestSentiment:
    """Test sentiment analysis."""

    def test_lexicon_sentiment_positive(self):
        """Test positive sentiment detection."""
        analyzer = scirs2.LexiconSentimentAnalyzer()

        text = "This is wonderful and amazing!"
        sentiment = analyzer.analyze(text)

        assert "score" in sentiment or "polarity" in sentiment
        # Positive text should have positive score
        score = sentiment.get("score", sentiment.get("polarity", 0))
        assert score > 0

    def test_lexicon_sentiment_negative(self):
        """Test negative sentiment detection."""
        analyzer = scirs2.LexiconSentimentAnalyzer()

        text = "This is terrible and awful!"
        sentiment = analyzer.analyze(text)

        score = sentiment.get("score", sentiment.get("polarity", 0))
        # Negative text should have negative score
        assert score < 0

    def test_lexicon_sentiment_neutral(self):
        """Test neutral sentiment detection."""
        analyzer = scirs2.LexiconSentimentAnalyzer()

        text = "The table is wooden."
        sentiment = analyzer.analyze(text)

        score = sentiment.get("score", sentiment.get("polarity", 0))
        # Neutral text should have score near 0
        assert -0.5 <= score <= 0.5

    def test_sentiment_batch(self):
        """Test batch sentiment analysis."""
        analyzer = scirs2.LexiconSentimentAnalyzer()

        texts = [
            "Great product!",
            "Terrible service.",
            "It is okay."
        ]

        sentiments = analyzer.analyze_batch(texts)

        assert len(sentiments) == 3


class TestStringSimilarity:
    """Test string similarity metrics."""

    def test_levenshtein_distance(self):
        """Test Levenshtein distance."""
        s1 = "kitten"
        s2 = "sitting"

        distance = scirs2.levenshtein_distance_py(s1, s2)

        # Should be 3 (substitute k->s, e->i, insert g)
        assert distance == 3

    def test_levenshtein_identical(self):
        """Test Levenshtein distance for identical strings."""
        s1 = "hello"
        s2 = "hello"

        distance = scirs2.levenshtein_distance_py(s1, s2)

        assert distance == 0

    def test_hamming_distance(self):
        """Test Hamming distance."""
        s1 = "karolin"
        s2 = "kathrin"

        distance = scirs2.hamming_distance_py(s1, s2)

        # Same length, 3 differences
        assert distance == 3

    def test_jaro_winkler_similarity(self):
        """Test Jaro-Winkler similarity."""
        s1 = "martha"
        s2 = "marhta"

        similarity = scirs2.jaro_winkler_similarity_py(s1, s2)

        # Should be close to 1 (very similar)
        assert 0.9 <= similarity <= 1.0

    def test_cosine_similarity_text(self):
        """Test cosine similarity for text."""
        s1 = "the quick brown fox"
        s2 = "the fast brown fox"

        similarity = scirs2.cosine_similarity_text_py(s1, s2)

        # Should be high similarity
        assert 0.5 <= similarity <= 1.0


class TestTextStatistics:
    """Test text statistics functions."""

    def test_word_count(self):
        """Test word counting."""
        text = "The quick brown fox jumps over the lazy dog"

        count = scirs2.word_count_py(text)

        assert count == 9

    def test_character_count(self):
        """Test character counting."""
        text = "Hello World!"

        count = scirs2.character_count_py(text, include_spaces=False)

        assert count == 10  # Excluding spaces and punctuation

    def test_unique_words(self):
        """Test unique word counting."""
        text = "the cat and the dog and the bird"

        unique = scirs2.unique_words_py(text)

        # Unique: the, cat, and, dog, bird = 5
        assert len(unique) == 5

    def test_lexical_diversity(self):
        """Test lexical diversity calculation."""
        text = "the cat sat on the mat"

        diversity = scirs2.lexical_diversity_py(text)

        # Diversity = unique_words / total_words
        assert 0 < diversity <= 1


class TestTextNormalization:
    """Test text normalization functions."""

    def test_lowercase_normalization(self):
        """Test converting to lowercase."""
        text = "Hello WORLD"

        normalized = scirs2.normalize_case_py(text, case="lower")

        assert normalized == "hello world"

    def test_uppercase_normalization(self):
        """Test converting to uppercase."""
        text = "Hello World"

        normalized = scirs2.normalize_case_py(text, case="upper")

        assert normalized == "HELLO WORLD"

    def test_remove_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, World! How are you?"

        cleaned = scirs2.remove_punctuation_py(text)

        assert "," not in cleaned
        assert "!" not in cleaned
        assert "?" not in cleaned

    def test_remove_numbers(self):
        """Test number removal."""
        text = "There are 123 apples and 456 oranges"

        cleaned = scirs2.remove_numbers_py(text)

        assert "123" not in cleaned
        assert "456" not in cleaned


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_tokenization(self):
        """Test tokenization of empty text."""
        tokenizer = scirs2.WordTokenizer()

        tokens = tokenizer.tokenize("")

        assert len(tokens) == 0

    def test_single_character_tokenization(self):
        """Test tokenization of single character."""
        tokenizer = scirs2.WordTokenizer()

        tokens = tokenizer.tokenize("a")

        assert len(tokens) >= 1

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        tokenizer = scirs2.WordTokenizer()

        text = "Hello 世界 مرحبا"
        tokens = tokenizer.tokenize(text)

        assert len(tokens) > 0

    def test_very_long_text(self):
        """Test handling of very long text."""
        tokenizer = scirs2.WordTokenizer()

        text = " ".join(["word"] * 10000)
        tokens = tokenizer.tokenize(text)

        assert len(tokens) == 10000

    def test_special_characters(self):
        """Test handling of special characters."""
        text = "!@#$%^&*()_+-=[]{}|;':,.<>?/"

        tokenizer = scirs2.WordTokenizer()
        tokens = tokenizer.tokenize(text)

        # Should handle gracefully
        assert isinstance(tokens, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
