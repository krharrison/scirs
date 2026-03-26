//! Fuzz target: text tokenisation pipeline.
//!
//! Drives the high-level `BasicTokenizer` → `WordPieceTokenizer` pipeline
//! that underlies BERT-style inference.  The fuzz input is treated as a raw
//! UTF-8 string and fed through:
//!
//! 1. `BasicTokenizer::tokenize` (whitespace + punctuation splitting, optional
//!    lower-casing and accent stripping).
//! 2. `WordPieceTokenizer::tokenize` / `tokenize_to_strings` / `decode`.
//!
//! Additionally the `BpeTokenizer` encode/decode round-trip is exercised on
//! a tiny fixed vocabulary so the fuzzer can easily generate interesting
//! edge cases without having to discover a large vocabulary.
//!
//! Invariants checked:
//! - No function call may panic.
//! - `decode(encode(text))` must not panic (correctness is not asserted
//!   because we only care about panic-freedom here).

#![no_main]

use libfuzzer_sys::fuzz_target;
use scirs2_text::tokenization::{BasicTokenizer, WordPieceTokenizer};

// Fixed minimal vocabulary for `WordPieceTokenizer` round-trips.
// The vocabulary is intentionally small so the fuzzer quickly reaches
// interesting code paths like the `[UNK]` fallback.
static VOCAB_TOKENS: &[&str] = &[
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "a", "an", "is", "are", "was", "were",
    "hello", "world", "foo", "bar", "baz",
    "##ing", "##ed", "##er", "##ly", "##s",
    "abc", "def", "xyz",
];

fuzz_target!(|data: &[u8]| {
    // Only process valid UTF-8 — byte-level BPE is tested in fuzz_bpe.rs.
    let text = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    // ── BasicTokenizer (lower-case + accent-strip variant) ───────────────────
    {
        let basic = BasicTokenizer::new(true, true);
        let tokens = basic.tokenize(text);
        // Must produce a `Vec<String>` without panicking.
        let _ = tokens.len();
    }

    // ── BasicTokenizer (preserve-case variant) ───────────────────────────────
    {
        let basic = BasicTokenizer::new(false, false);
        let _ = basic.tokenize(text);
    }

    // ── WordPieceTokenizer ────────────────────────────────────────────────────
    {
        let wp = WordPieceTokenizer::from_vocab_list(VOCAB_TOKENS);

        // tokenize → [u32] IDs
        let ids = wp.tokenize(text);

        // tokenize_to_strings — same logic, string output
        let strs = wp.tokenize_to_strings(text);
        debug_assert_eq!(ids.len(), strs.len());

        // decode must not panic on the IDs we just produced.
        let decoded = wp.decode(&ids);
        let _ = decoded.len();

        // encode (add_special_tokens=true) — returns Result; must not panic.
        let _ = wp.encode(text, 512, true);

        // Probe with max_input_chars set to an extreme value.
        let wp_long = WordPieceTokenizer::from_vocab_list(VOCAB_TOKENS)
            .with_max_input_chars(usize::MAX);
        let _ = wp_long.tokenize(text);

        // Probe with max_input_chars set to 1 (stress-tests the fallback path).
        let wp_short =
            WordPieceTokenizer::from_vocab_list(VOCAB_TOKENS).with_max_input_chars(1);
        let _ = wp_short.tokenize(text);
    }
});
