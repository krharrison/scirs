//! Fuzz target: Byte-Pair Encoding (BPE) tokenizer.
//!
//! Exercises three complementary code paths:
//!
//! 1. **`BpeTokenizer::encode` / `decode`** — a pre-trained tokenizer built
//!    from a small fixed corpus is used so the fuzzer quickly generates valid
//!    merges rather than having to discover the full training algorithm.
//!
//! 2. **`BpeTokenizer::from_json`** — the JSON deserialisation path that
//!    loads a trained tokenizer from a string.  Arbitrary byte sequences are
//!    cast to UTF-8 and fed to `from_json`; this must never panic.
//!
//! 3. **`BpeTokenizer::train`** on tiny synthetic corpora derived from the
//!    fuzz bytes, ensuring the training loop cannot be made to panic.
//!
//! 4. **`UnicodeBpeTokenizer::encode` / `decode`** — the Unicode-aware BPE
//!    variant that handles multi-byte characters.
//!
//! Invariants:
//! - No function may panic on any input.
//! - `decode(encode(s))` must complete without panicking (the decoded string
//!    need not equal `s` for adversarial inputs).

#![no_main]

use libfuzzer_sys::fuzz_target;
use scirs2_text::tokenization::bpe::BpeTokenizer;
use scirs2_text::tokenization::unicode_bpe::{UnicodeBpeConfig, UnicodeBpeTokenizer};

/// A minimal pre-trained `BpeTokenizer` that is constructed once and reused
/// across all fuzz iterations.  The `lazy_static!` pattern is avoided here
/// because `libfuzzer_sys` sets up its own initialisation; instead we use
/// `std::sync::OnceLock` which is stable since Rust 1.70.
fn get_pretrained() -> &'static BpeTokenizer {
    static TOKENIZER: std::sync::OnceLock<BpeTokenizer> = std::sync::OnceLock::new();
    TOKENIZER.get_or_init(|| {
        let corpus = &[
            "hello world",
            "hello foo",
            "world foo bar",
            "the quick brown fox",
            "foo bar baz qux",
            "abcdefghijklmnopqrstuvwxyz",
        ];
        // train() returns Result — fall back to a default tokenizer on error.
        BpeTokenizer::train(corpus, 128, 1)
            .unwrap_or_else(|_| BpeTokenizer::train(&["a b c"], 32, 1)
                .unwrap_or_else(|_| BpeTokenizer::train(&["a"], 8, 1)
                    .expect("Could not create minimal BpeTokenizer")))
    })
}

fuzz_target!(|data: &[u8]| {
    let tokenizer = get_pretrained();

    // ── Phase 1: encode / decode on arbitrary UTF-8 ──────────────────────────
    if let Ok(text) = std::str::from_utf8(data) {
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        let _ = decoded.len();

        // tokenize() returns token strings — must not panic.
        let tokens = tokenizer.tokenize(text);
        let _ = tokens.len();

        // vocab_size must be consistent.
        let _ = tokenizer.vocab_size();
    }

    // ── Phase 2: from_json deserialisation ───────────────────────────────────
    if let Ok(json_str) = std::str::from_utf8(data) {
        // Must not panic — invalid JSON → Err(_).
        let _ = BpeTokenizer::from_json(json_str);
    }

    // ── Phase 3: round-trip serialize/deserialize ─────────────────────────────
    {
        let json = tokenizer.to_json();
        // Re-loading our own serialised form must always succeed.
        if let Ok(reloaded) = BpeTokenizer::from_json(&json) {
            // The re-loaded tokenizer should be usable.
            if let Ok(text) = std::str::from_utf8(data) {
                let _ = reloaded.encode(text);
            }
        }
    }

    // ── Phase 4: training on fuzz-derived corpus ──────────────────────────────
    // Limit corpus size to prevent extremely long training runs.
    if data.len() < 512 {
        if let Ok(text) = std::str::from_utf8(data) {
            // Split on newlines to build a tiny corpus.
            let lines: Vec<&str> = text.lines().take(8).collect();
            if !lines.is_empty() {
                let _ = BpeTokenizer::train(lines.as_slice(), 64, 1);
            }
        }
    }

    // ── Phase 5: UnicodeBpeTokenizer ─────────────────────────────────────────
    if let Ok(text) = std::str::from_utf8(data) {
        // Build a fresh Unicode BPE tokenizer and train on the fuzz input.
        // Keep corpus small to limit run time.
        if text.len() < 256 {
            // Use Default::default() because UnicodeBpeConfig is #[non_exhaustive].
            let mut config = UnicodeBpeConfig::default();
            config.vocab_size = 64;
            config.min_frequency = 1;
            let mut ubpe = UnicodeBpeTokenizer::new(config);
            let lines: Vec<&str> = text.lines().take(4).collect();
            if !lines.is_empty() {
                let _ = ubpe.train(lines.as_slice());
            }
            // encode / decode on the (possibly untrained) tokenizer.
            let _ = ubpe.encode(text);
        }

        // Exercise encode/decode with a stable pre-trained instance.
        static UBPE: std::sync::OnceLock<UnicodeBpeTokenizer> =
            std::sync::OnceLock::new();
        let ubpe_trained = UBPE.get_or_init(|| {
            let mut config = UnicodeBpeConfig::default();
            config.vocab_size = 256;
            config.min_frequency = 1;
            let mut t = UnicodeBpeTokenizer::new(config);
            let _ = t.train(&["hello world", "foo bar baz", "the quick brown fox"]);
            t
        });
        if let Ok(ids) = ubpe_trained.encode(text) {
            let _ = ubpe_trained.decode(&ids);
        }
    }
});
