//! HuggingFace tokenizers JSON serialization format compatibility.
//!
//! Provides (de)serialization for the canonical `tokenizers` JSON format used
//! by the HuggingFace `tokenizers` library so that vocabularies and BPE merge
//! lists produced by SciRS2 tokenizers can be loaded by HF tooling and vice
//! versa.
//!
//! # Format outline
//!
//! ```json
//! {
//!   "version": "1.0",
//!   "truncation": null,
//!   "padding": null,
//!   "added_tokens": [...],
//!   "normalizer": {"type": "BertNormalizer", "lowercase": true, ...},
//!   "pre_tokenizer": {"type": "BertPreTokenizer"},
//!   "post_processor": {"type": "TemplateProcessing", ...},
//!   "decoder": {"type": "WordPiece", "prefix": "##"},
//!   "model": {"type": "WordPiece", "vocab": {...}, "unk_token": "[UNK]", ...}
//! }
//! ```

use crate::error::{Result, TextError};
use crate::tokenizers::{BertTokenizer, RobertaTokenizer};
use std::collections::HashMap;
use std::fs;

// ─── Public data types ────────────────────────────────────────────────────────

/// Configuration for the normalizer component.
#[derive(Debug, Clone, PartialEq)]
pub struct HfNormalizerConfig {
    /// Whether to lowercase text.
    pub lowercase: bool,
    /// Whether to strip accents/combining marks.
    pub strip_accents: bool,
    /// Whether to add spaces around CJK characters.
    pub handle_chinese_chars: bool,
}

impl Default for HfNormalizerConfig {
    fn default() -> Self {
        HfNormalizerConfig {
            lowercase: true,
            strip_accents: true,
            handle_chinese_chars: true,
        }
    }
}

/// An "added token" entry in the HuggingFace JSON format.
#[derive(Debug, Clone, PartialEq)]
pub struct HfAddedToken {
    /// Numeric token ID.
    pub id: u32,
    /// Token string (e.g. `"[CLS]"`, `"<s>"`).
    pub content: String,
    /// Whether the token is a special / control token.
    pub special: bool,
    /// Whether to strip whitespace on the left.
    pub lstrip: bool,
    /// Whether to strip whitespace on the right.
    pub rstrip: bool,
    /// Whether the token is a single-word token.
    pub single_word: bool,
    /// Whether the token is normalised before matching.
    pub normalized: bool,
}

impl HfAddedToken {
    /// Construct a basic special token entry.
    pub fn special(id: u32, content: impl Into<String>) -> Self {
        HfAddedToken {
            id,
            content: content.into(),
            special: true,
            lstrip: false,
            rstrip: false,
            single_word: false,
            normalized: false,
        }
    }
}

/// Represents a tokenizer serialized in the HuggingFace `tokenizers` JSON format.
///
/// Supports `WordPiece` (BERT) and `BPE` (RoBERTa / GPT-2) model types.
#[derive(Debug, Clone)]
pub struct HfTokenizerJson {
    /// Format version — always `"1.0"`.
    pub version: String,
    /// Model type: `"WordPiece"`, `"BPE"`, or `"Unigram"`.
    pub model_type: String,
    /// `token → id` mapping (applies to both WordPiece vocab and BPE vocab).
    pub vocab: HashMap<String, u32>,
    /// BPE merge rules in priority order (empty for WordPiece).
    pub merges: Vec<(String, String)>,
    /// Special tokens keyed by their role (e.g. `"unk_token"`, `"cls_token"`).
    pub special_tokens: HashMap<String, u32>,
    /// Optional normalizer configuration.
    pub normalizer: Option<HfNormalizerConfig>,
    /// Pre-tokenizer type string (e.g. `"BertPreTokenizer"`, `"ByteLevel"`).
    pub pre_tokenizer: Option<String>,
    /// Added token entries (special tokens with metadata).
    pub added_tokens: Vec<HfAddedToken>,
    /// Continuation prefix used by WordPiece decoder (default `"##"`).
    pub wordpiece_prefix: String,
    /// Unknown token string.
    pub unk_token: String,
}

impl HfTokenizerJson {
    // ── Construction ──────────────────────────────────────────────────────

    /// Create a `WordPiece`-type JSON wrapper with default settings.
    pub fn new_wordpiece(vocab: HashMap<String, u32>) -> Self {
        HfTokenizerJson {
            version: "1.0".to_string(),
            model_type: "WordPiece".to_string(),
            vocab,
            merges: Vec::new(),
            special_tokens: HashMap::new(),
            normalizer: Some(HfNormalizerConfig::default()),
            pre_tokenizer: Some("BertPreTokenizer".to_string()),
            added_tokens: Vec::new(),
            wordpiece_prefix: "##".to_string(),
            unk_token: "[UNK]".to_string(),
        }
    }

    /// Create a `BPE`-type JSON wrapper with default settings.
    pub fn new_bpe(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        HfTokenizerJson {
            version: "1.0".to_string(),
            model_type: "BPE".to_string(),
            vocab,
            merges,
            special_tokens: HashMap::new(),
            normalizer: None,
            pre_tokenizer: Some("ByteLevel".to_string()),
            added_tokens: Vec::new(),
            wordpiece_prefix: "##".to_string(),
            unk_token: "<unk>".to_string(),
        }
    }

    // ── Conversion from SciRS2 tokenizers ────────────────────────────────

    /// Build an `HfTokenizerJson` from a [`BertTokenizer`].
    ///
    /// The vocabulary is extracted via the public `vocab()` method; special
    /// tokens are collected from well-known BERT token strings.
    pub fn from_bert(tokenizer: &BertTokenizer) -> Self {
        let vocab = tokenizer.vocab().clone();

        let bert_specials = [
            ("[PAD]", "pad_token"),
            ("[UNK]", "unk_token"),
            ("[CLS]", "cls_token"),
            ("[SEP]", "sep_token"),
            ("[MASK]", "mask_token"),
        ];

        let mut special_tokens = HashMap::new();
        let mut added_tokens: Vec<HfAddedToken> = Vec::new();

        for (token, role) in &bert_specials {
            if let Some(&id) = vocab.get(*token) {
                special_tokens.insert(role.to_string(), id);
                added_tokens.push(HfAddedToken::special(id, *token));
            }
        }

        let unk_token = vocab
            .get("[UNK]")
            .map(|_| "[UNK]".to_string())
            .unwrap_or_default();

        let normalizer = Some(HfNormalizerConfig {
            lowercase: tokenizer.lowercase(),
            strip_accents: true,
            handle_chinese_chars: true,
        });

        HfTokenizerJson {
            version: "1.0".to_string(),
            model_type: "WordPiece".to_string(),
            vocab,
            merges: Vec::new(),
            special_tokens,
            normalizer,
            pre_tokenizer: Some("BertPreTokenizer".to_string()),
            added_tokens,
            wordpiece_prefix: "##".to_string(),
            unk_token,
        }
    }

    /// Build an `HfTokenizerJson` from a [`RobertaTokenizer`].
    pub fn from_roberta(tokenizer: &RobertaTokenizer) -> Self {
        let vocab = tokenizer.vocab().clone();
        let merges = tokenizer.merges().to_vec();

        let roberta_specials = [
            ("<s>", "bos_token"),
            ("</s>", "eos_token"),
            ("<pad>", "pad_token"),
            ("<unk>", "unk_token"),
            ("<mask>", "mask_token"),
        ];

        let mut special_tokens = HashMap::new();
        let mut added_tokens: Vec<HfAddedToken> = Vec::new();

        for (token, role) in &roberta_specials {
            if let Some(&id) = vocab.get(*token) {
                special_tokens.insert(role.to_string(), id);
                added_tokens.push(HfAddedToken::special(id, *token));
            }
        }

        let unk_token = "<unk>".to_string();

        HfTokenizerJson {
            version: "1.0".to_string(),
            model_type: "BPE".to_string(),
            vocab,
            merges,
            special_tokens,
            normalizer: None,
            pre_tokenizer: Some("ByteLevel".to_string()),
            added_tokens,
            wordpiece_prefix: "##".to_string(),
            unk_token,
        }
    }

    /// Reconstruct a [`BertTokenizer`] from this JSON description.
    ///
    /// Only applicable when `model_type == "WordPiece"`.
    pub fn to_bert_tokenizer(&self) -> Result<BertTokenizer> {
        if self.model_type != "WordPiece" {
            return Err(TextError::InvalidInput(format!(
                "Cannot create BertTokenizer from model type '{}'",
                self.model_type
            )));
        }
        let lowercase = self
            .normalizer
            .as_ref()
            .map(|n| n.lowercase)
            .unwrap_or(true);
        Ok(BertTokenizer::new(self.vocab.clone(), lowercase))
    }

    // ── JSON (de)serialization ────────────────────────────────────────────

    /// Serialize to a HuggingFace-compatible JSON string.
    ///
    /// The output can be loaded by the HuggingFace `tokenizers` Python library
    /// via `tokenizers.Tokenizer.from_file(path)` after writing to disk.
    pub fn to_json(&self) -> Result<String> {
        let mut obj = serde_json_obj();

        // --- version ---
        obj.insert("version".to_string(), json_string(&self.version));

        // --- truncation / padding ---
        obj.insert("truncation".to_string(), "null".to_string());
        obj.insert("padding".to_string(), "null".to_string());

        // --- added_tokens ---
        let added_tokens_json = self
            .added_tokens
            .iter()
            .map(|t| {
                format!(
                    "{{\"id\":{},\"content\":{},\"single_word\":{},\"lstrip\":{},\
                     \"rstrip\":{},\"normalized\":{},\"special\":{}}}",
                    t.id,
                    json_string(&t.content),
                    t.single_word,
                    t.lstrip,
                    t.rstrip,
                    t.normalized,
                    t.special,
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        obj.insert(
            "added_tokens".to_string(),
            format!("[{}]", added_tokens_json),
        );

        // --- normalizer ---
        let normalizer_json = match &self.normalizer {
            None => "null".to_string(),
            Some(n) => {
                format!(
                    "{{\"type\":\"BertNormalizer\",\"clean_text\":true,\
                     \"handle_chinese_chars\":{},\"strip_accents\":{},\"lowercase\":{}}}",
                    n.handle_chinese_chars, n.strip_accents, n.lowercase,
                )
            }
        };
        obj.insert("normalizer".to_string(), normalizer_json);

        // --- pre_tokenizer ---
        let pre_tok_json = match &self.pre_tokenizer {
            None => "null".to_string(),
            Some(name) if name == "BertPreTokenizer" => {
                "{\"type\":\"BertPreTokenizer\"}".to_string()
            }
            Some(name) if name == "ByteLevel" => {
                "{\"type\":\"ByteLevel\",\"add_prefix_space\":false}".to_string()
            }
            Some(name) => format!("{{\"type\":{}}}", json_string(name)),
        };
        obj.insert("pre_tokenizer".to_string(), pre_tok_json);

        // --- post_processor ---
        let post_proc_json = if self.model_type == "WordPiece" {
            if let (Some(&cls_id), Some(&sep_id)) = (
                self.special_tokens.get("cls_token"),
                self.special_tokens.get("sep_token"),
            ) {
                format!(
                    "{{\"type\":\"TemplateProcessing\",\
                     \"single\":\"[CLS]:0 $A:0 [SEP]:0\",\
                     \"pair\":\"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1\",\
                     \"special_tokens\":{{\
                     \"[CLS]\":{{\"id\":{},\"ids\":[{}],\"tokens\":[\"[CLS]\"]}},\
                     \"[SEP]\":{{\"id\":{},\"ids\":[{}],\"tokens\":[\"[SEP]\"]}}\
                     }}}}",
                    cls_id, cls_id, sep_id, sep_id,
                )
            } else {
                "null".to_string()
            }
        } else {
            "null".to_string()
        };
        obj.insert("post_processor".to_string(), post_proc_json);

        // --- decoder ---
        let decoder_json = if self.model_type == "WordPiece" {
            format!(
                "{{\"type\":\"WordPiece\",\"prefix\":{},\"cleanup\":true}}",
                json_string(&self.wordpiece_prefix)
            )
        } else {
            "{\"type\":\"ByteLevel\",\"add_prefix_space\":false}".to_string()
        };
        obj.insert("decoder".to_string(), decoder_json);

        // --- model ---
        let model_json = match self.model_type.as_str() {
            "WordPiece" => {
                let vocab_entries = build_vocab_json(&self.vocab);
                format!(
                    "{{\"type\":\"WordPiece\",\"unk_token\":{},\"continuing_subword_prefix\":{},\
                     \"max_input_chars_per_word\":100,\"vocab\":{{{}}}}}",
                    json_string(&self.unk_token),
                    json_string(&self.wordpiece_prefix),
                    vocab_entries,
                )
            }
            "BPE" => {
                let vocab_entries = build_vocab_json(&self.vocab);
                let merges_entries = self
                    .merges
                    .iter()
                    .map(|(a, b)| json_string(&format!("{} {}", a, b)))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    "{{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":{},\
                     \"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,\
                     \"fuse_unk\":false,\"vocab\":{{{}}},\"merges\":[{}]}}",
                    json_string(&self.unk_token),
                    vocab_entries,
                    merges_entries,
                )
            }
            other => {
                return Err(TextError::InvalidInput(format!(
                    "Unsupported model type for JSON serialization: {}",
                    other
                )))
            }
        };
        obj.insert("model".to_string(), model_json);

        // Assemble top-level object in canonical field order.
        let fields = [
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        ];

        let body = fields
            .iter()
            .filter_map(|k| obj.get(*k).map(|v| format!("\"{}\":{}", k, v)))
            .collect::<Vec<_>>()
            .join(",");

        Ok(format!("{{{}}}", body))
    }

    /// Parse an `HfTokenizerJson` from a HuggingFace-compatible JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        // Use a minimal hand-rolled parser to avoid depending on serde_json
        // at the public API boundary (serde-support is feature-gated).
        // We delegate to the serde_json feature when available.
        parse_hf_json(json)
    }

    /// Save to a file at `path`.
    pub fn save(&self, path: &str) -> Result<()> {
        let json = self.to_json()?;
        fs::write(path, json).map_err(|e| TextError::IoError(e.to_string()))
    }

    /// Load from a file at `path`.
    pub fn load(path: &str) -> Result<Self> {
        let contents = fs::read_to_string(path).map_err(|e| TextError::IoError(e.to_string()))?;
        Self::from_json(&contents)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Escape a string for JSON output.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Build a sorted `"token":id` JSON object body from a vocabulary map.
fn build_vocab_json(vocab: &HashMap<String, u32>) -> String {
    let mut entries: Vec<(&String, &u32)> = vocab.iter().collect();
    entries.sort_by_key(|(_, &id)| id);
    entries
        .iter()
        .map(|(k, v)| format!("{}:{}", json_string(k), v))
        .collect::<Vec<_>>()
        .join(",")
}

/// Placeholder helper — returns a new empty ordered map.
fn serde_json_obj() -> HashMap<String, String> {
    HashMap::new()
}

// ─── JSON parser ──────────────────────────────────────────────────────────────

/// Parse an HF tokenizers JSON string into an [`HfTokenizerJson`].
///
/// Uses a lightweight recursive-descent parser that handles the subset of JSON
/// required by the HF tokenizers format.
fn parse_hf_json(json: &str) -> Result<HfTokenizerJson> {
    let root = JsonValue::parse(json)
        .ok_or_else(|| TextError::InvalidInput("Failed to parse JSON".to_string()))?;

    if root.as_obj().is_none() {
        return Err(TextError::InvalidInput(
            "Root must be a JSON object".to_string(),
        ));
    }

    // --- model ---
    let model = root
        .get("model")
        .ok_or_else(|| TextError::InvalidInput("Missing 'model' field".to_string()))?;

    let model_type = model
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("WordPiece")
        .to_string();

    // vocabulary
    let vocab: HashMap<String, u32> = model
        .get("vocab")
        .and_then(|v| v.as_obj())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_u32().map(|id| (k.clone(), id)))
                .collect()
        })
        .unwrap_or_default();

    // BPE merges
    let merges: Vec<(String, String)> = model
        .get("merges")
        .and_then(|m| m.as_arr())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let s = item.as_str()?;
                    let mut parts = s.splitn(2, ' ');
                    let a = parts.next()?.to_string();
                    let b = parts.next()?.to_string();
                    Some((a, b))
                })
                .collect()
        })
        .unwrap_or_default();

    let unk_token = model
        .get("unk_token")
        .and_then(|t| t.as_str())
        .unwrap_or("[UNK]")
        .to_string();

    let wordpiece_prefix = model
        .get("continuing_subword_prefix")
        .and_then(|p| p.as_str())
        .unwrap_or("##")
        .to_string();

    // --- normalizer ---
    let normalizer = root.get("normalizer").and_then(|n| {
        n.as_obj()?;
        Some(HfNormalizerConfig {
            lowercase: n
                .get("lowercase")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            strip_accents: n
                .get("strip_accents")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            handle_chinese_chars: n
                .get("handle_chinese_chars")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    });

    // --- pre_tokenizer ---
    let pre_tokenizer = root
        .get("pre_tokenizer")
        .and_then(|pt| pt.get("type"))
        .and_then(|t| t.as_str())
        .map(|s| s.to_string());

    // --- added_tokens ---
    let added_tokens: Vec<HfAddedToken> = root
        .get("added_tokens")
        .and_then(|at| at.as_arr())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    item.as_obj()?;
                    let id = item.get("id")?.as_u32()?;
                    let content = item.get("content")?.as_str()?.to_string();
                    let special = item
                        .get("special")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let lstrip = item
                        .get("lstrip")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let rstrip = item
                        .get("rstrip")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let single_word = item
                        .get("single_word")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let normalized = item
                        .get("normalized")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    Some(HfAddedToken {
                        id,
                        content,
                        special,
                        lstrip,
                        rstrip,
                        single_word,
                        normalized,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // --- special_tokens from vocab ---
    let mut special_tokens: HashMap<String, u32> = HashMap::new();
    if model_type == "WordPiece" {
        let roles = [
            ("[PAD]", "pad_token"),
            ("[UNK]", "unk_token"),
            ("[CLS]", "cls_token"),
            ("[SEP]", "sep_token"),
            ("[MASK]", "mask_token"),
        ];
        for (tok, role) in &roles {
            if let Some(&id) = vocab.get(*tok) {
                special_tokens.insert(role.to_string(), id);
            }
        }
    } else {
        let roles = [
            ("<s>", "bos_token"),
            ("</s>", "eos_token"),
            ("<pad>", "pad_token"),
            ("<unk>", "unk_token"),
            ("<mask>", "mask_token"),
        ];
        for (tok, role) in &roles {
            if let Some(&id) = vocab.get(*tok) {
                special_tokens.insert(role.to_string(), id);
            }
        }
    }

    Ok(HfTokenizerJson {
        version: root
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0")
            .to_string(),
        model_type,
        vocab,
        merges,
        special_tokens,
        normalizer,
        pre_tokenizer,
        added_tokens,
        wordpiece_prefix,
        unk_token,
    })
}

// ─── Minimal JSON value tree ──────────────────────────────────────────────────

/// Minimal JSON value representation for parsing HF tokenizer JSON.
#[derive(Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn as_str(&self) -> Option<&str> {
        if let JsonValue::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    fn as_bool(&self) -> Option<bool> {
        if let JsonValue::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    fn as_u32(&self) -> Option<u32> {
        if let JsonValue::Number(n) = self {
            Some(*n as u32)
        } else {
            None
        }
    }

    fn as_obj(&self) -> Option<&[(String, JsonValue)]> {
        if let JsonValue::Object(fields) = self {
            Some(fields.as_slice())
        } else {
            None
        }
    }

    /// Look up a field by name in an object `JsonValue`.
    fn get(&self, key: &str) -> Option<&JsonValue> {
        if let JsonValue::Object(fields) = self {
            fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }

    fn as_arr(&self) -> Option<&[JsonValue]> {
        if let JsonValue::Array(items) = self {
            Some(items.as_slice())
        } else {
            None
        }
    }

    /// Parse a JSON string into a `JsonValue`.
    fn parse(s: &str) -> Option<Self> {
        let mut p = Parser {
            src: s.as_bytes(),
            pos: 0,
        };
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos == p.src.len() {
            Some(v)
        } else {
            None
        }
    }
}

// ─── Recursive-descent JSON parser ───────────────────────────────────────────

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        self.pos += 1;
        b
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, byte: u8) -> Option<()> {
        self.skip_ws();
        if self.peek()? == byte {
            self.pos += 1;
            Some(())
        } else {
            None
        }
    }

    fn parse_value(&mut self) -> Option<JsonValue> {
        self.skip_ws();
        match self.peek()? {
            b'"' => self.parse_string().map(JsonValue::Str),
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b't' => {
                self.expect_literal(b"true")?;
                Some(JsonValue::Bool(true))
            }
            b'f' => {
                self.expect_literal(b"false")?;
                Some(JsonValue::Bool(false))
            }
            b'n' => {
                self.expect_literal(b"null")?;
                Some(JsonValue::Null)
            }
            b'-' | b'0'..=b'9' => self.parse_number().map(JsonValue::Number),
            _ => None,
        }
    }

    fn expect_literal(&mut self, lit: &[u8]) -> Option<()> {
        let end = self.pos + lit.len();
        if self.src.get(self.pos..end)? == lit {
            self.pos = end;
            Some(())
        } else {
            None
        }
    }

    fn parse_string(&mut self) -> Option<String> {
        self.skip_ws();
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            match self.advance()? {
                b'"' => break,
                b'\\' => {
                    match self.advance()? {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'b' => s.push('\x08'),
                        b'f' => s.push('\x0C'),
                        b'u' => {
                            // 4-hex-digit unicode escape
                            let mut code: u32 = 0;
                            for _ in 0..4 {
                                let h = self.advance()?;
                                let digit = match h {
                                    b'0'..=b'9' => h - b'0',
                                    b'a'..=b'f' => h - b'a' + 10,
                                    b'A'..=b'F' => h - b'A' + 10,
                                    _ => return None,
                                };
                                code = (code << 4) | digit as u32;
                            }
                            s.push(char::from_u32(code)?);
                        }
                        _ => return None,
                    }
                }
                byte => {
                    // Build char from potentially multi-byte UTF-8.
                    let start = self.pos - 1;
                    // Collect remaining continuation bytes.
                    let leading = byte;
                    let extra = if leading < 0x80 {
                        0
                    } else if leading < 0xE0 {
                        1
                    } else if leading < 0xF0 {
                        2
                    } else {
                        3
                    };
                    for _ in 0..extra {
                        self.advance()?;
                    }
                    let slice = &self.src[start..self.pos];
                    let ch = std::str::from_utf8(slice).ok()?.chars().next()?;
                    s.push(ch);
                }
            }
        }
        Some(s)
    }

    fn parse_number(&mut self) -> Option<f64> {
        let start = self.pos;
        // Consume optional sign.
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        // Integer part.
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        // Fraction.
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        // Exponent.
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let slice = std::str::from_utf8(&self.src[start..self.pos]).ok()?;
        slice.parse::<f64>().ok()
    }

    fn parse_object(&mut self) -> Option<JsonValue> {
        self.expect(b'{')?;
        let mut fields: Vec<(String, JsonValue)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Some(JsonValue::Object(fields));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            fields.push((key, val));
            self.skip_ws();
            match self.peek()? {
                b',' => {
                    self.pos += 1;
                }
                b'}' => {
                    self.pos += 1;
                    break;
                }
                _ => return None,
            }
        }
        Some(JsonValue::Object(fields))
    }

    fn parse_array(&mut self) -> Option<JsonValue> {
        self.expect(b'[')?;
        let mut items: Vec<JsonValue> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Some(JsonValue::Array(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws();
            match self.peek()? {
                b',' => {
                    self.pos += 1;
                }
                b']' => {
                    self.pos += 1;
                    break;
                }
                _ => return None,
            }
        }
        Some(JsonValue::Array(items))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_bert_vocab() -> HashMap<String, u32> {
        let pairs = [
            ("[PAD]", 0u32),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MASK]", 4),
            ("hello", 5),
            ("world", 6),
            ("##ing", 7),
        ];
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn small_bpe_vocab() -> HashMap<String, u32> {
        let pairs = [
            ("<s>", 0u32),
            ("<pad>", 1),
            ("</s>", 2),
            ("<unk>", 3),
            ("he", 4),
            ("llo", 5),
            ("hello", 6),
            ("<mask>", 50264),
        ];
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_hf_json_wordpiece_roundtrip() {
        let vocab = small_bert_vocab();
        let hf = HfTokenizerJson::new_wordpiece(vocab.clone());

        let json = hf.to_json().expect("serialize");
        let parsed = HfTokenizerJson::from_json(&json).expect("deserialize");

        assert_eq!(parsed.model_type, "WordPiece");
        assert_eq!(parsed.vocab.len(), vocab.len());
        for (k, v) in &vocab {
            assert_eq!(parsed.vocab.get(k), Some(v), "mismatch for token {}", k);
        }
    }

    #[test]
    fn test_hf_json_from_bert() {
        let vocab = small_bert_vocab();
        let tokenizer = BertTokenizer::new(vocab.clone(), true);
        let hf = HfTokenizerJson::from_bert(&tokenizer);

        assert_eq!(hf.model_type, "WordPiece");
        assert_eq!(hf.version, "1.0");
        assert!(hf.vocab.contains_key("[CLS]"));
        assert!(hf.vocab.contains_key("[SEP]"));
    }

    #[test]
    fn test_hf_json_special_tokens() {
        let vocab = small_bert_vocab();
        let tokenizer = BertTokenizer::new(vocab, true);
        let hf = HfTokenizerJson::from_bert(&tokenizer);

        assert!(hf.special_tokens.contains_key("cls_token"));
        assert!(hf.special_tokens.contains_key("sep_token"));
        assert!(hf.special_tokens.contains_key("pad_token"));
        assert!(hf.special_tokens.contains_key("unk_token"));
        assert!(hf.special_tokens.contains_key("mask_token"));

        // added_tokens must list them
        let contents: Vec<&str> = hf.added_tokens.iter().map(|t| t.content.as_str()).collect();
        assert!(contents.contains(&"[CLS]"));
        assert!(contents.contains(&"[SEP]"));
    }

    #[test]
    fn test_hf_json_bpe_merges() {
        let vocab = small_bpe_vocab();
        let merges = vec![("he".to_string(), "llo".to_string())];
        let hf = HfTokenizerJson::new_bpe(vocab.clone(), merges.clone());

        let json = hf.to_json().expect("serialize");
        let parsed = HfTokenizerJson::from_json(&json).expect("deserialize");

        assert_eq!(parsed.model_type, "BPE");
        assert_eq!(parsed.merges.len(), 1);
        assert_eq!(parsed.merges[0], ("he".to_string(), "llo".to_string()));
    }

    #[test]
    fn test_hf_json_to_bert_tokenizer() {
        let vocab = small_bert_vocab();
        let hf = HfTokenizerJson::new_wordpiece(vocab.clone());
        let tokenizer = hf.to_bert_tokenizer().expect("reconstruction");
        assert_eq!(tokenizer.vocab_size(), vocab.len());
    }

    #[test]
    fn test_hf_json_wordpiece_prefix_preserved() {
        let vocab = small_bert_vocab();
        let mut hf = HfTokenizerJson::new_wordpiece(vocab);
        hf.wordpiece_prefix = "@@".to_string();

        let json = hf.to_json().expect("serialize");
        let parsed = HfTokenizerJson::from_json(&json).expect("deserialize");
        assert_eq!(parsed.wordpiece_prefix, "@@");
    }

    #[test]
    fn test_hf_json_normalizer_roundtrip() {
        let vocab = small_bert_vocab();
        let mut hf = HfTokenizerJson::new_wordpiece(vocab);
        hf.normalizer = Some(HfNormalizerConfig {
            lowercase: false,
            strip_accents: true,
            handle_chinese_chars: false,
        });

        let json = hf.to_json().expect("serialize");
        let parsed = HfTokenizerJson::from_json(&json).expect("deserialize");
        let norm = parsed.normalizer.expect("normalizer present");
        assert!(!norm.lowercase);
        assert!(norm.strip_accents);
        assert!(!norm.handle_chinese_chars);
    }

    #[test]
    fn test_hf_json_empty_merges_bpe() {
        let vocab = small_bpe_vocab();
        let hf = HfTokenizerJson::new_bpe(vocab, vec![]);
        let json = hf.to_json().expect("serialize");
        let parsed = HfTokenizerJson::from_json(&json).expect("deserialize");
        assert!(parsed.merges.is_empty());
    }

    #[test]
    fn test_hf_json_save_and_load() {
        let vocab = small_bert_vocab();
        let hf = HfTokenizerJson::new_wordpiece(vocab.clone());

        let tmp = std::env::temp_dir().join("test_hf_tokenizer.json");
        let path = tmp.to_str().expect("valid path");

        hf.save(path).expect("save");
        let loaded = HfTokenizerJson::load(path).expect("load");

        assert_eq!(loaded.model_type, "WordPiece");
        assert_eq!(loaded.vocab.len(), vocab.len());

        let _ = std::fs::remove_file(path);
    }
}
