# scirs2-text TODO

## Status: v0.3.4 Released (March 18, 2026)

## v0.3.3 Completed

### Core Tokenization
- [x] `WordTokenizer` - Unicode-aware word tokenization, configurable lowercase
- [x] `SentenceTokenizer` - Rule-based sentence boundary detection
- [x] `CharTokenizer` - Character and Unicode grapheme cluster tokenization
- [x] `NgramTokenizer` - N-grams with fixed n and range support
- [x] `RegexTokenizer` - Pattern-based and gap-based tokenization
- [x] `WhitespaceTokenizer` - Simple whitespace splitting
- [x] `BpeTokenizer` - Byte Pair Encoding with vocabulary learning and save/load
- [x] WordPiece tokenizer (BERT-style subword)
- [x] `Tokenizer` trait for interchangeable backends

### Text Preprocessing
- [x] `BasicNormalizer` - Unicode normalization, case folding, accent removal
- [x] `BasicTextCleaner` - HTML/XML stripping, URL/email normalization, stopwords
- [x] Contraction expansion
- [x] Number normalization (dates, currencies, percentages, ordinals)
- [x] `TextPreprocessor` - Composable normalizer + cleaner pipeline

### Stemming and Lemmatization
- [x] `PorterStemmer` - Classic Porter algorithm
- [x] `SnowballStemmer` - Snowball algorithm (English)
- [x] `LancasterStemmer` - Aggressive Lancaster stemming
- [x] `SimpleLemmatizer` - Dictionary-based lemmatization with morphological analysis
- [x] `Stemmer` trait for interchangeable backends

### Text Vectorization
- [x] `CountVectorizer` - Bag-of-words with N-gram support and vocabulary management
- [x] `TfidfVectorizer` - TF-IDF with smoothing, sublinear TF, L1/L2 normalization
- [x] `BinaryVectorizer` - Binary occurrence representation
- [x] `EnhancedCountVectorizer` - Max features, min/max document frequency
- [x] `EnhancedTfidfVectorizer` - All enhanced options + advanced IDF weighting
- [x] Sparse matrix output for memory efficiency
- [x] Vocabulary persistence (save/load)

### Word Embeddings
- [x] `Word2Vec` - Skip-gram and CBOW with negative sampling
- [x] Configurable: vector size, window, min_count, iterations, negative samples
- [x] `most_similar()` cosine similarity lookup
- [x] Binary and text format save/load
- [x] GloVe vector loading
- [x] `FastText` (pure Rust subword embeddings with character n-grams)

### Sequence Labelling
- [x] `CrfTagger` - CRF with Viterbi decoding and custom feature functions
- [x] `HmmTagger` - HMM for POS tagging (forward-backward, Viterbi)
- [x] Feature engineering utilities for NER, POS, chunking

### Named Entity Recognition (NER)
- [x] Rule-based NER with regex patterns
- [x] Dictionary/gazetteer-based NER
- [x] CRF-based NER with feature engineering
- [x] Standard types: PER, ORG, LOC, DATE, TIME, MONEY, PERCENT
- [x] Entity span detection with start/end offsets

### Advanced NLP (New in v0.3.1)
- [x] `coreference` - Mention detection and coreference clustering
- [x] `dependency` - Arc-factored dependency graph construction
- [x] `discourse` - Discourse analysis and RST primitives
- [x] `event_extraction` - Event trigger and argument extraction
- [x] `question_answering` - Extractive span detection
- [x] `knowledge_graph` - Entity-relation-entity triple extraction
- [x] `semantic_parsing` - Logical form generation
- [x] `temporal` - Date/time expression normalization (TIMEX3-style)
- [x] `grammar` - Rule-based grammar error detection
- [x] `annotation` - Annotation layer management

### Topic Modeling
- [x] `LatentDirichletAllocation` (LDA) - variational inference
- [x] Coherence metrics: CV, UMass, UCI
- [x] NMF-based topic modeling
- [x] `TopicModel` trait

### Summarization
- [x] Extractive: TextRank, centroid-based, keyword-based sentence scoring
- [x] `abstractive_summary.rs` - Abstractive summarization primitives

### Sentiment Analysis
- [x] `LexiconSentimentAnalyzer` - VADER-style with negation and intensifiers
- [x] Rule-based sentiment with modifier handling
- [x] Compound sentiment score
- [x] ML-based classifier adapter

### Text Classification
- [x] Feature extraction pipeline (bag-of-words, TF-IDF, n-gram combos)
- [x] `MultinomialNaiveBayes` (text-optimized with Laplace smoothing)
- [x] Classification dataset handling and evaluation
- [x] `text_classification.rs` - Classification workflow

### String Metrics and Phonetics
- [x] `LevenshteinMetric` (basic edit distance)
- [x] `DamerauLevenshteinMetric` - with transpositions, restricted/unrestricted modes
- [x] Jaro-Winkler similarity
- [x] `WeightedLevenshtein` - per-operation and per-character-pair costs
- [x] `WeightedDamerauLevenshtein` - with weighted transpositions
- [x] Cosine similarity, Jaccard similarity (set-based and n-gram)
- [x] `Soundex` phonetic encoding
- [x] `Metaphone` phonetic algorithm
- [x] `NYSIIS` phonetic algorithm
- [x] `advanced_distance.rs` - Word Mover's Distance, Soft Cosine, Conceptual Similarity

### Language Models
- [x] N-gram language model with Kneser-Ney smoothing
- [x] Character-level language model
- [x] Perplexity computation
- [x] `language_models` module

### Text Statistics and Readability
- [x] Flesch Reading Ease, Flesch-Kincaid Grade Level
- [x] Gunning Fog Index, SMOG Index, Coleman-Liau Index
- [x] Lexical diversity, type-token ratio
- [x] Word count, sentence count, average sentence length
- [x] `ReadabilityMetrics` struct with all common formulas

### Performance and Infrastructure
- [x] Rayon-based parallel tokenization and vectorization
- [x] `simd_ops.rs` - SIMD-accelerated string operations and distance computation
- [x] Memory-mapped corpus for large-scale processing
- [x] Sparse matrix output from all vectorizers
- [x] `parallel.rs` - Parallel corpus processing utilities
- [x] `information_theory` - Entropy, mutual information, KL divergence for text
- [x] `multilingual_ext.rs` - Language detection and multilingual utilities

### Testing and Quality
- [x] 160+ unit tests
- [x] 8 doctest examples
- [x] Zero-warning builds
- [x] All public APIs documented

## v0.4.0 Roadmap

### Transformer Tokenizers
- [ ] SentencePiece tokenizer (Unigram LM-based, used by T5/LLaMA)
- [ ] BERT/RoBERTa tokenizer (WordPiece with special tokens: [CLS], [SEP], [MASK])
- [ ] GPT-2/GPT-4 tokenizer (BPE with byte-level encoding)
- [ ] LLaMA tokenizer (SentencePiece + BPE hybrid)
- [ ] Tokenizer serialization compatible with HuggingFace `tokenizers` JSON format
- [ ] Batch tokenization with padding and truncation

### Sentence Embeddings
- [ ] Sentence-BERT-style aggregation (mean pooling of token embeddings)
- [ ] Universal Sentence Encoder-style (transformer + pooling)
- [ ] Contrastive sentence representation learning (SimCSE-style)
- [ ] Semantic similarity via sentence embeddings
- [ ] Cross-lingual sentence embeddings

### Multilingual Models
- [ ] Language-agnostic tokenization (Unicode-based, no language assumptions)
- [ ] Multilingual vocabulary (shared BPE across 50+ languages)
- [ ] Cross-lingual NER transfer
- [ ] Transliteration utilities for CJK and Cyrillic scripts

### Enhanced Topic Modeling
- [ ] Online LDA for streaming corpora
- [ ] Hierarchical Dirichlet Process (HDP) for automatic topic number selection
- [ ] Correlated Topic Model (CTM) with logistic-normal prior
- [ ] Dynamic Topic Model (DTM) for temporal analysis

### Neural NLP Integration
- [ ] Bridge to `scirs2-neural` for transformer-based NLP
- [ ] Attention visualization for transformer token attribution
- [ ] BERT-style fine-tuning API for classification and NER
- [ ] Named entity recognition via neural sequence labeler

### Evaluation and Benchmarks
- [ ] CoNLL-2003 NER evaluation protocol (span-level F1)
- [ ] BLEU, ROUGE, METEOR for generation/summarization
- [ ] STS benchmark integration (semantic textual similarity)
- [ ] Perplexity benchmarks on standard corpora (PTB, WikiText)

## Known Issues

- The `MultinomialNaiveBayes` import was previously duplicated in `text_classification.rs`; resolved in v0.3.1.
- LDA coherence computation uses the corpus vocabulary; very small corpora may produce unreliable scores — document minimum corpus size recommendations.
- `abstractive_summary.rs` provides primitives only; full abstractive summarization requires a neural sequence-to-sequence model from `scirs2-neural`.
- Word2Vec training convergence depends heavily on `min_count` and corpus size; add validation warnings for very small corpora.
- FastText character n-gram support may increase memory significantly for large vocabulary sizes; document memory tradeoffs.
