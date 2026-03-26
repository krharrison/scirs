# NLP & Text (scirs2-text)

`scirs2-text` provides natural language processing primitives including tokenization,
word/sentence embeddings, topic models, and evaluation metrics for NER and text generation.

## Tokenization

### Byte-Pair Encoding (BPE)

```rust,ignore
use scirs2_text::tokenization::{BPETokenizer, ByteLevelBPE};

// Standard BPE
let tokenizer = BPETokenizer::from_vocab("vocab.json", "merges.txt")?;
let tokens = tokenizer.encode("Hello, world!")?;
let text = tokenizer.decode(&tokens)?;

// Byte-level BPE (GPT-2 style)
let tokenizer = ByteLevelBPE::from_vocab("vocab.json", "merges.txt")?;
let tokens = tokenizer.encode("Scientific computing in Rust")?;
```

### LLaMA Tokenizer

SentencePiece-compatible tokenizer:

```rust,ignore
use scirs2_text::tokenization::llama::LlamaTokenizer;

let tokenizer = LlamaTokenizer::from_model("tokenizer.model")?;
let tokens = tokenizer.encode("The quick brown fox")?;
let text = tokenizer.decode(&tokens)?;
```

### Multilingual BPE

```rust,ignore
use scirs2_text::tokenization::multilingual_bpe::MultilingualBPE;

let tokenizer = MultilingualBPE::new(vocab_size, languages)?;
tokenizer.train(&multilingual_corpus)?;
```

### HuggingFace JSON Format

```rust,ignore
use scirs2_text::tokenization::hf_json::HFTokenizer;

// Load tokenizer from HuggingFace JSON format
let tokenizer = HFTokenizer::from_file("tokenizer.json")?;
let encoding = tokenizer.encode("Input text", true)?;  // add_special_tokens=true
```

## Embeddings

### Word2Vec and GloVe

```rust,ignore
use scirs2_text::embeddings::{Word2Vec, GloVe};

// Train Word2Vec
let model = Word2Vec::new(embed_dim, window_size, min_count)?;
model.train(&corpus, num_epochs)?;

// Similarity and analogy
let similarity = model.similarity("king", "queen")?;
let result = model.analogy("king", "man", "woman")?;  // -> "queen"

// Load pre-trained GloVe
let glove = GloVe::load("glove.6B.300d.txt")?;
let vector = glove.get("science")?;
```

### Sentence Embeddings

```rust,ignore
use scirs2_text::embeddings::sentence::SentenceBERT;

let model = SentenceBERT::new(config)?;
let embedding = model.encode("This is a sentence.")?;
let similarity = model.cosine_similarity(
    "Machine learning is great.",
    "Deep learning is powerful."
)?;
```

## Topic Models

### Latent Dirichlet Allocation (LDA)

```rust,ignore
use scirs2_text::lda::{LDA, LDAOptions};

let options = LDAOptions {
    num_topics: 10,
    alpha: 0.1,
    beta: 0.01,
    num_iterations: 1000,
};
let model = LDA::new(options)?;
model.fit(&document_term_matrix)?;

let topics = model.top_words(10)?;  // top 10 words per topic
let doc_topics = model.transform(&new_document)?;
```

### Correlated Topic Model (CTM)

```rust,ignore
use scirs2_text::ctm::{CTM, CTMOptions};

let model = CTM::new(CTMOptions {
    num_topics: 20,
    num_iterations: 500,
})?;
model.fit(&dtm)?;
let correlation_matrix = model.topic_correlations()?;
```

### Dynamic Topic Model (DTM)

```rust,ignore
use scirs2_text::dtm::{DTM, DTMOptions};

let model = DTM::new(DTMOptions {
    num_topics: 10,
    num_time_slices: 12,
})?;
model.fit(&time_sliced_dtm)?;
let evolution = model.topic_evolution(topic_id)?;
```

## Evaluation

### NER Metrics

```rust,ignore
use scirs2_text::evaluation::ner::{ner_precision_recall_f1, NERScheme};

let metrics = ner_precision_recall_f1(
    &predicted_tags, &gold_tags, NERScheme::IOB2
)?;
println!("F1: {:.4}", metrics.f1);
println!("Per-entity: {:?}", metrics.per_entity);
```

### Text Generation Metrics

```rust,ignore
use scirs2_text::evaluation::{bleu, rouge};

let bleu_score = bleu(&reference, &hypothesis, 4)?;  // BLEU-4
let rouge_scores = rouge(&reference, &hypothesis)?;
println!("ROUGE-L: {:.4}", rouge_scores.rouge_l);
```
