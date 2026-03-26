//! Fuzz target: WordPiece tokenizer edge cases.
//!
//! While `fuzz_tokenizer.rs` exercises the normal `BasicTokenizer` →
//! `WordPieceTokenizer` pipeline, this target focuses specifically on
//! `WordPieceTokenizer` with a wide range of vocabulary configurations:
//!
//! 1. **Empty vocabulary** — the tokenizer must fall back to `[UNK]` cleanly.
//! 2. **Single-character vocabulary** — each character is its own token.
//! 3. **`##`-prefixed continuation tokens** — the greedy algorithm's main
//!    code path.
//! 4. **Very long words** — stress-tests the `max_input_chars` limit.
//! 5. **`encode` method** — a higher-level wrapper that adds `[CLS]`/`[SEP]`.
//!
//! The fuzz bytes are interpreted as UTF-8; non-UTF-8 inputs are silently
//! skipped because WordPiece operates on Unicode strings.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scirs2_text::tokenization::{BasicTokenizer, WordPieceTokenizer};

fuzz_target!(|data: &[u8]| {
    let text = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    // ── Vocabulary variant 1: only special tokens ─────────────────────────────
    {
        let wp = WordPieceTokenizer::from_vocab_list(&["[PAD]", "[UNK]"]);
        let ids = wp.tokenize(text);
        let _ = wp.decode(&ids);
        let _ = wp.tokenize_to_strings(text);
    }

    // ── Vocabulary variant 2: ASCII alphabet + continuation tokens ────────────
    {
        let mut vocab: Vec<String> = vec!["[PAD]".into(), "[UNK]".into()];
        for c in 'a'..='z' {
            vocab.push(c.to_string());
            vocab.push(format!("##{c}"));
        }
        let wp = WordPieceTokenizer::from_vocab_list(
            &vocab.iter().map(String::as_str).collect::<Vec<_>>(),
        );
        let ids = wp.tokenize(text);
        let decoded = wp.decode(&ids);
        let _ = decoded.len();
    }

    // ── Vocabulary variant 3: full ASCII printable ────────────────────────────
    {
        let mut vocab: Vec<String> = vec!["[UNK]".into()];
        for b in 0x20u8..=0x7eu8 {
            vocab.push((b as char).to_string());
        }
        let wp = WordPieceTokenizer::from_vocab_list(
            &vocab.iter().map(String::as_str).collect::<Vec<_>>(),
        );
        let _ = wp.tokenize(text);
    }

    // ── max_input_chars boundary ──────────────────────────────────────────────
    for max_chars in [0usize, 1, 2, 10, usize::MAX] {
        let wp = WordPieceTokenizer::from_vocab_list(&["[UNK]", "a", "##b"])
            .with_max_input_chars(max_chars);
        let _ = wp.tokenize(text);
    }

    // ── encode() (adds [CLS] / [SEP]) ─────────────────────────────────────────
    {
        let wp = WordPieceTokenizer::from_vocab_list(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world",
        ]);
        // encode(text, max_length, add_special_tokens) → Result<(Vec<u32>, Vec<u8>)>
        let _ = wp.encode(text, 128, true);
        let _ = wp.encode(text, 128, false);
    }

    // ── BasicTokenizer variations then WordPiece ──────────────────────────────
    for (lc, sa) in [(true, true), (true, false), (false, true), (false, false)] {
        let basic = BasicTokenizer::new(lc, sa);
        let words = basic.tokenize(text);
        let wp = WordPieceTokenizer::from_vocab_list(&["[UNK]", "the", "##ing"]);
        for word in &words {
            let ids = wp.tokenize(word);
            let _ = wp.decode(&ids);
        }
    }
});
