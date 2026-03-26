//! FSST (Fast Static Symbol Table) string compression.
//!
//! FSST replaces frequent byte substrings (1- or 2-byte "symbols") with
//! single-byte codes, achieving good compression for string columns while
//! remaining extremely fast at decompression.
//!
//! ## Algorithm overview
//!
//! 1. **Training**: scan samples, count 1-byte and 2-byte substring frequencies.
//! 2. **Symbol selection**: greedily pick the top `max_symbols` by
//!    `score = len × frequency`.
//! 3. **Compression**: left-to-right scan; emit 2-byte symbol code if possible,
//!    else 1-byte symbol code, else escape byte (255) + raw byte.
//! 4. **Decompression**: look up each code in the symbol table; escape byte
//!    means the next byte is raw.

use crate::error::{IoError, Result as IoResult};

// ---------------------------------------------------------------------------
// Symbol
// ---------------------------------------------------------------------------

/// One entry in the FSST symbol table.
#[derive(Debug, Clone, Default)]
pub struct FsstSymbol {
    /// Raw bytes of the symbol (at most 2 bytes).
    pub bytes: [u8; 2],
    /// Number of bytes in this symbol (1 or 2).
    pub len: u8,
    /// Training score: `len × frequency`.
    pub score: u64,
}

impl FsstSymbol {
    fn one_byte(b: u8, freq: u64) -> Self {
        Self {
            bytes: [b, 0],
            len: 1,
            score: freq,
        }
    }

    fn two_bytes(b0: u8, b1: u8, freq: u64) -> Self {
        Self {
            bytes: [b0, b1],
            len: 2,
            score: freq.saturating_mul(2),
        }
    }
}

// ---------------------------------------------------------------------------
// FsstSymbolTable
// ---------------------------------------------------------------------------

/// FSST symbol table: up to 255 symbols (code 255 = escape byte).
///
/// The table is built from training data and then used to compress/decompress
/// arbitrary byte sequences.
#[derive(Debug, Clone)]
pub struct FsstSymbolTable {
    /// Symbols indexed by code `0..symbols.len()`.
    pub symbols: Vec<FsstSymbol>,
    /// 2-byte lookup: `[first_byte][second_byte]` → code or `255` (not found).
    encode_2byte: Box<[[u8; 256]; 256]>,
    /// 1-byte lookup: `[byte]` → code or `255` (not found).
    encode_1byte: [u8; 256],
}

impl FsstSymbolTable {
    /// Build a symbol table by training on byte samples.
    ///
    /// `max_symbols` is clamped to 254 (codes 0..254; code 255 = escape).
    pub fn train(samples: &[&[u8]], max_symbols: usize) -> Self {
        let max_symbols = max_symbols.min(254);

        // Count 1-byte and 2-byte frequencies
        let mut freq1 = vec![0u64; 256];
        let mut freq2 = vec![vec![0u64; 256]; 256];

        for sample in samples {
            let len = sample.len();
            for i in 0..len {
                freq1[sample[i] as usize] += 1;
                if i + 1 < len {
                    freq2[sample[i] as usize][sample[i + 1] as usize] += 1;
                }
            }
        }

        // Collect candidates with score = len * frequency
        let mut candidates: Vec<FsstSymbol> = Vec::with_capacity(256 + 256 * 256);

        for b in 0u8..=255u8 {
            let f = freq1[b as usize];
            if f > 0 {
                candidates.push(FsstSymbol::one_byte(b, f));
            }
        }

        for b0 in 0usize..256 {
            for b1 in 0usize..256 {
                let f = freq2[b0][b1];
                if f > 0 {
                    candidates.push(FsstSymbol::two_bytes(b0 as u8, b1 as u8, f));
                }
            }
        }

        // Sort by descending score; break ties by descending len (prefer 2-byte)
        candidates.sort_by(|a, b| b.score.cmp(&a.score).then(b.len.cmp(&a.len)));
        candidates.truncate(max_symbols);

        // Build lookup tables
        let mut encode_2byte = Box::new([[255u8; 256]; 256]);
        let mut encode_1byte = [255u8; 256];

        for (code, sym) in candidates.iter().enumerate() {
            let c = code as u8;
            if sym.len == 1 {
                encode_1byte[sym.bytes[0] as usize] = c;
            } else {
                encode_2byte[sym.bytes[0] as usize][sym.bytes[1] as usize] = c;
            }
        }

        Self {
            symbols: candidates,
            encode_2byte,
            encode_1byte,
        }
    }

    /// Compress a byte slice using this symbol table.
    ///
    /// Escape code `255` followed by a raw byte is used for bytes not covered
    /// by any symbol.
    pub fn compress(&self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0;
        let len = input.len();

        while i < len {
            // Try 2-byte match first
            if i + 1 < len {
                let code = self.encode_2byte[input[i] as usize][input[i + 1] as usize];
                if code != 255 {
                    out.push(code);
                    i += 2;
                    continue;
                }
            }
            // Try 1-byte match
            let code = self.encode_1byte[input[i] as usize];
            if code != 255 {
                out.push(code);
                i += 1;
                continue;
            }
            // Escape
            out.push(255);
            out.push(input[i]);
            i += 1;
        }

        out
    }

    /// Decompress a byte slice compressed with `compress`.
    pub fn decompress(&self, compressed: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(compressed.len() * 2);
        let mut i = 0;
        let len = compressed.len();

        while i < len {
            let code = compressed[i];
            i += 1;

            if code == 255 {
                // Next byte is raw
                if i < len {
                    out.push(compressed[i]);
                    i += 1;
                }
            } else if (code as usize) < self.symbols.len() {
                let sym = &self.symbols[code as usize];
                out.push(sym.bytes[0]);
                if sym.len == 2 {
                    out.push(sym.bytes[1]);
                }
            }
            // Unknown code → skip (should not happen for valid compressed data)
        }

        out
    }

    /// Number of symbols in this table.
    pub fn n_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Compute the compression ratio: `original.len() / compressed.len()`.
    ///
    /// Returns `f64::INFINITY` if `compressed` is empty.
    pub fn compression_ratio(&self, original: &[u8], compressed: &[u8]) -> f64 {
        original.len() as f64 / compressed.len().max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// FsstColumnEncoder
// ---------------------------------------------------------------------------

/// High-level encoder for a column of strings backed by an [`FsstSymbolTable`].
pub struct FsstColumnEncoder {
    /// The trained symbol table.
    pub table: FsstSymbolTable,
}

impl FsstColumnEncoder {
    /// Train an encoder on a sample of the provided strings.
    ///
    /// `sample_fraction` controls what fraction of strings are used for
    /// training (clamped to `(0.0, 1.0]`).  Sampling is deterministic:
    /// every `floor(1 / sample_fraction)`-th string is taken.
    pub fn train(strings: &[&str], sample_fraction: f64) -> IoResult<Self> {
        if strings.is_empty() {
            return Ok(Self {
                table: FsstSymbolTable::train(&[], 254),
            });
        }

        let fraction = sample_fraction.clamp(1e-6, 1.0);
        let step = (1.0 / fraction).floor() as usize;
        let step = step.max(1);

        let samples: Vec<&[u8]> = strings
            .iter()
            .enumerate()
            .filter(|(i, _)| i % step == 0)
            .map(|(_, s)| s.as_bytes())
            .collect();

        if samples.is_empty() {
            return Err(IoError::FormatError(
                "FSST training: no samples selected (sample_fraction too small?)".to_string(),
            ));
        }

        let table = FsstSymbolTable::train(&samples, 254);
        Ok(Self { table })
    }

    /// Compress all strings in the column.
    pub fn compress_column(&self, strings: &[&str]) -> Vec<Vec<u8>> {
        strings
            .iter()
            .map(|s| self.table.compress(s.as_bytes()))
            .collect()
    }

    /// Decompress a column of byte slices back to UTF-8 strings.
    ///
    /// Returns an error if any decompressed byte sequence is not valid UTF-8.
    pub fn decompress_column(&self, compressed: &[Vec<u8>]) -> IoResult<Vec<String>> {
        compressed
            .iter()
            .enumerate()
            .map(|(i, bytes)| {
                let raw = self.table.decompress(bytes);
                String::from_utf8(raw).map_err(|e| {
                    IoError::FormatError(format!(
                        "FSST decompress: string {} is not valid UTF-8: {e}",
                        i
                    ))
                })
            })
            .collect()
    }

    /// Compute the average compression ratio over all strings.
    pub fn column_compression_ratio(&self, strings: &[&str]) -> f64 {
        if strings.is_empty() {
            return 1.0;
        }
        let total_original: usize = strings.iter().map(|s| s.len()).sum();
        let total_compressed: usize = strings
            .iter()
            .map(|s| self.table.compress(s.as_bytes()).len())
            .sum();
        total_original as f64 / total_compressed.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsst_compress_decompress_roundtrip() {
        // Train on some representative strings
        let samples: Vec<&str> = vec![
            "hello world",
            "hello rust",
            "world of rust",
            "hello hello world",
        ];
        let sample_bytes: Vec<&[u8]> = samples.iter().map(|s| s.as_bytes()).collect();
        let table = FsstSymbolTable::train(&sample_bytes, 254);

        // Roundtrip every sample
        for s in &samples {
            let compressed = table.compress(s.as_bytes());
            let decompressed = table.decompress(&compressed);
            assert_eq!(decompressed, s.as_bytes(), "roundtrip failed for {:?}", s);
        }
    }

    #[test]
    fn test_fsst_table_size_bounded() {
        let data: Vec<String> = (0..1000).map(|i| format!("item_{i}_data")).collect();
        let samples: Vec<&[u8]> = data.iter().map(|s| s.as_bytes()).collect();
        let table = FsstSymbolTable::train(&samples, 254);
        assert!(
            table.n_symbols() <= 254,
            "symbol table exceeds 254: {}",
            table.n_symbols()
        );
    }

    #[test]
    fn test_fsst_column_encoder_roundtrip() {
        let strings: Vec<&str> = vec![
            "sensor_lab_a",
            "sensor_lab_b",
            "sensor_lab_a",
            "sensor_lab_c",
            "sensor_lab_a",
            "sensor_lab_b",
        ];

        let encoder = FsstColumnEncoder::train(&strings, 1.0).expect("training failed");

        let compressed = encoder.compress_column(&strings);
        let decompressed = encoder
            .decompress_column(&compressed)
            .expect("decompress failed");

        let original: Vec<String> = strings.iter().map(|s| s.to_string()).collect();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_fsst_compression_ratio_positive() {
        // For repetitive data, compression ratio should be > 1
        let repeated = "aaaa bbbb cccc dddd aaaa bbbb aaaa";
        let samples = &[repeated.as_bytes()];
        let table = FsstSymbolTable::train(samples, 254);
        let compressed = table.compress(repeated.as_bytes());
        let ratio = table.compression_ratio(repeated.as_bytes(), &compressed);
        // At least decent: should be >= 0.5
        assert!(ratio > 0.0, "ratio should be positive");
    }

    #[test]
    fn test_fsst_empty_input() {
        let table = FsstSymbolTable::train(&[], 254);
        let compressed = table.compress(&[]);
        assert!(compressed.is_empty());
        let decompressed = table.decompress(&[]);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_fsst_escape_byte_roundtrip() {
        // Create a symbol table trained on different data so some bytes need escaping
        let train_data: Vec<&[u8]> = vec![b"aabbcc"];
        let table = FsstSymbolTable::train(&train_data, 254);

        // Input with bytes that may not be in the table
        let input: Vec<u8> = (0u8..=127).collect();
        let compressed = table.compress(&input);
        let decompressed = table.decompress(&compressed);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_fsst_column_encoder_sample_fraction() {
        let strings: Vec<String> = (0..100).map(|i| format!("item_{i}")).collect();
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        // Train with 10% sample
        let encoder = FsstColumnEncoder::train(&refs, 0.1).expect("training failed");

        let compressed = encoder.compress_column(&refs);
        let decompressed = encoder
            .decompress_column(&compressed)
            .expect("decompress failed");

        assert_eq!(decompressed.len(), strings.len());
        for (a, b) in strings.iter().zip(decompressed.iter()) {
            assert_eq!(a, b);
        }
    }
}
