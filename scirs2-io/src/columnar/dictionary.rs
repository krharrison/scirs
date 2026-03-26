//! Dictionary encoding for columnar data.
//!
//! Replaces repeated string (or generic) values with small integer codes,
//! achieving significant compression for low-cardinality columns.
//!
//! ## Design
//!
//! - Vocabulary is built from training data sorted by descending frequency.
//! - Codes are `u16`; code `u16::MAX` (65535) is reserved as the "null" sentinel.
//! - Maximum vocabulary size is `u16::MAX - 1` = 65534 entries.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_io::columnar::dictionary::{DictionaryEncoder, UnknownPolicy};
//!
//! let data = vec!["red", "blue", "red", "green", "red", "blue"];
//! let enc: DictionaryEncoder<String> =
//!     DictionaryEncoder::fit(
//!         &data.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
//!         UnknownPolicy::Error,
//!     ).expect("fit failed");
//!
//! let codes = enc.encode(
//!     &data.iter().map(|s| s.to_string()).collect::<Vec<_>>()
//! ).expect("encode failed");
//! assert_eq!(codes.len(), 6);
//! ```

use std::collections::HashMap;
use std::hash::Hash;

use crate::error::IoError;

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// How to handle values encountered during encoding that are not in the
/// vocabulary built by [`DictionaryEncoder::fit`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownPolicy {
    /// Return an [`IoError`] when an unknown value is encountered.
    Error,
    /// Map unknown values to `u16::MAX` (the null sentinel code).
    UseNullCode,
    /// Dynamically extend the vocabulary; errors if capacity is exhausted.
    AddToVocab,
}

// ---------------------------------------------------------------------------
// DictionaryEncoder
// ---------------------------------------------------------------------------

/// Dictionary encoder for a generic value type `T`.
///
/// `T` must be `Eq + Hash + Clone + Ord`; the `Ord` bound is used only for
/// tie-breaking when two values have the same frequency.
#[derive(Debug, Clone)]
pub struct DictionaryEncoder<T: Eq + Hash + Clone> {
    /// Vocabulary: `vocab[code]` → value.  Sorted by descending frequency.
    pub vocab: Vec<T>,
    /// Reverse map: value → code.
    code_map: HashMap<T, u16>,
    /// Policy for values not present in the vocabulary.
    pub unknown_policy: UnknownPolicy,
}

impl<T: Eq + Hash + Clone + Ord> DictionaryEncoder<T> {
    /// Create an empty encoder with the given unknown policy.
    pub fn new(unknown_policy: UnknownPolicy) -> Self {
        Self {
            vocab: Vec::new(),
            code_map: HashMap::new(),
            unknown_policy,
        }
    }

    /// Build a `DictionaryEncoder` by scanning `data` and counting frequencies.
    ///
    /// The vocabulary is sorted by descending frequency (ties broken by
    /// value ordering).  Codes `0..vocab.len()` are assigned in that order.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of unique values exceeds `u16::MAX - 1`
    /// (= 65534).
    pub fn fit(data: &[T], unknown_policy: UnknownPolicy) -> IoResult<Self> {
        let mut freq: HashMap<T, u64> = HashMap::new();
        for item in data {
            *freq.entry(item.clone()).or_insert(0) += 1;
        }

        const MAX_VOCAB: usize = u16::MAX as usize - 1; // 65534
        if freq.len() > MAX_VOCAB {
            return Err(IoError::FormatError(format!(
                "dictionary encoding: too many unique values ({} > {MAX_VOCAB})",
                freq.len()
            )));
        }

        // Sort by descending frequency; break ties by ascending value order
        let mut entries: Vec<(T, u64)> = freq.into_iter().collect();
        entries.sort_by(|(va, fa), (vb, fb)| fb.cmp(fa).then_with(|| va.cmp(vb)));

        let mut vocab = Vec::with_capacity(entries.len());
        let mut code_map: HashMap<T, u16> = HashMap::with_capacity(entries.len());

        for (code, (value, _freq)) in entries.into_iter().enumerate() {
            vocab.push(value.clone());
            code_map.insert(value, code as u16);
        }

        Ok(Self {
            vocab,
            code_map,
            unknown_policy,
        })
    }

    /// Encode a slice of values to codes.
    ///
    /// Behaviour for unknown values is governed by `self.unknown_policy`.
    pub fn encode(&self, data: &[T]) -> IoResult<Vec<u16>>
    where
        T: std::fmt::Debug,
    {
        let mut codes = Vec::with_capacity(data.len());
        for item in data {
            let code = self.encode_single(item)?;
            codes.push(code);
        }
        Ok(codes)
    }

    /// Decode a slice of codes back to values.
    ///
    /// Returns an error if any code equals `u16::MAX` (null sentinel) or is
    /// out of bounds.
    pub fn decode(&self, codes: &[u16]) -> IoResult<Vec<T>> {
        let mut out = Vec::with_capacity(codes.len());
        for &code in codes {
            out.push(self.decode_single(code)?.clone());
        }
        Ok(out)
    }

    /// Return the current vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode a single value to its code.
    pub fn encode_single(&self, value: &T) -> IoResult<u16>
    where
        T: std::fmt::Debug,
    {
        match self.code_map.get(value) {
            Some(&code) => Ok(code),
            None => match self.unknown_policy {
                UnknownPolicy::Error => Err(IoError::FormatError(format!(
                    "dictionary encoding: unknown value {:?}",
                    value
                ))),
                UnknownPolicy::UseNullCode => Ok(u16::MAX),
                UnknownPolicy::AddToVocab => Err(IoError::FormatError(
                    "dictionary encoding: AddToVocab requires a mutable encoder; \
                         use encode_single_mut instead"
                        .to_string(),
                )),
            },
        }
    }

    /// Encode a single value, potentially adding it to the vocabulary
    /// (only meaningful when `unknown_policy == AddToVocab`).
    pub fn encode_single_mut(&mut self, value: T) -> IoResult<u16>
    where
        T: std::fmt::Debug,
    {
        if let Some(&code) = self.code_map.get(&value) {
            return Ok(code);
        }

        match self.unknown_policy {
            UnknownPolicy::Error => Err(IoError::FormatError(format!(
                "dictionary encoding: unknown value {:?}",
                value
            ))),
            UnknownPolicy::UseNullCode => Ok(u16::MAX),
            UnknownPolicy::AddToVocab => {
                const MAX_VOCAB: usize = u16::MAX as usize - 1;
                if self.vocab.len() >= MAX_VOCAB {
                    return Err(IoError::FormatError(
                        "dictionary encoding: vocabulary capacity exhausted".to_string(),
                    ));
                }
                let code = self.vocab.len() as u16;
                self.vocab.push(value.clone());
                self.code_map.insert(value, code);
                Ok(code)
            }
        }
    }

    /// Decode a single code to a reference to the corresponding value.
    pub fn decode_single(&self, code: u16) -> IoResult<&T> {
        if code == u16::MAX {
            return Err(IoError::FormatError(
                "dictionary encoding: null sentinel code (u16::MAX) encountered during decode"
                    .to_string(),
            ));
        }
        self.vocab.get(code as usize).ok_or_else(|| {
            IoError::FormatError(format!(
                "dictionary encoding: code {} out of range (vocab size {})",
                code,
                self.vocab.len()
            ))
        })
    }
}

/// Convenience type alias for string dictionary encoding.
pub type StringDictionaryEncoder = DictionaryEncoder<String>;

// ---------------------------------------------------------------------------
// IoResult alias (for convenience)
// ---------------------------------------------------------------------------

type IoResult<T> = crate::error::Result<T>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict_encoder_fit_all_unique() {
        let data: Vec<String> = (0..100).map(|i| format!("val_{i}")).collect();
        let enc = DictionaryEncoder::fit(&data, UnknownPolicy::Error).expect("fit failed");
        assert_eq!(enc.vocab_size(), 100);
    }

    #[test]
    fn test_dict_encoder_encode_decode_roundtrip() {
        let data: Vec<String> = vec!["a", "b", "a", "c", "b", "a"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let enc = DictionaryEncoder::fit(&data, UnknownPolicy::Error).expect("fit failed");

        // "a" has freq 3 → code 0; "b" freq 2 → code 1; "c" freq 1 → code 2
        assert_eq!(enc.vocab[0], "a");

        let codes = enc.encode(&data).expect("encode failed");
        assert_eq!(codes.len(), 6);

        let decoded = enc.decode(&codes).expect("decode failed");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_dict_encoder_unknown_error_policy() {
        let train: Vec<String> = vec!["x", "y"].into_iter().map(|s| s.to_string()).collect();
        let enc = DictionaryEncoder::fit(&train, UnknownPolicy::Error).expect("fit failed");

        let unknown = vec!["z".to_string()];
        let result = enc.encode(&unknown);
        assert!(result.is_err(), "expected error for unknown value");
    }

    #[test]
    fn test_dict_encoder_null_code_policy() {
        let train: Vec<String> = vec!["a".to_string()];
        let enc = DictionaryEncoder::fit(&train, UnknownPolicy::UseNullCode).expect("fit failed");

        let data = vec!["a".to_string(), "unknown".to_string()];
        let codes = enc.encode(&data).expect("encode failed");
        assert_eq!(codes[0], 0);
        assert_eq!(codes[1], u16::MAX);
    }

    #[test]
    fn test_dict_encoder_add_to_vocab_policy() {
        let train: Vec<String> = vec!["a".to_string()];
        let mut enc =
            DictionaryEncoder::fit(&train, UnknownPolicy::AddToVocab).expect("fit failed");

        let code = enc
            .encode_single_mut("b".to_string())
            .expect("encode failed");
        assert_eq!(enc.vocab_size(), 2);
        assert_eq!(enc.vocab[code as usize], "b");
    }

    #[test]
    fn test_dict_encoder_decode_null_sentinel_error() {
        let train: Vec<String> = vec!["a".to_string()];
        let enc = DictionaryEncoder::fit(&train, UnknownPolicy::Error).expect("fit failed");

        let result = enc.decode(&[u16::MAX]);
        assert!(result.is_err(), "decoding null sentinel should error");
    }

    #[test]
    fn test_dict_encoder_too_many_unique() {
        // Build a dataset with exactly 65535 unique strings (exceeds limit)
        let data: Vec<String> = (0u32..65535).map(|i| format!("v{i}")).collect();
        let result = DictionaryEncoder::fit(&data, UnknownPolicy::Error);
        assert!(result.is_err(), "should error on > 65534 unique values");
    }
}
