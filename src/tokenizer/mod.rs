/// Tokenizer module.
///
/// Implements a BERT-compatible tokenizer pipeline:
/// 1. Pre-tokenization (lowercasing, whitespace/punctuation splitting)
/// 2. WordPiece subword tokenization
/// 3. Encoding (adding special tokens, truncation, padding)

pub mod vocab;
pub mod wordpiece;
pub mod pre_tokenize;
pub mod encode;

pub use encode::{Encoding, Tokenizer};
