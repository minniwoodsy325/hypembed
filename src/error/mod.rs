/// Error types for the HypEmbed library.
///
/// All public functions return `Result<T, HypEmbedError>`.
/// Errors carry context to help diagnose issues.

use thiserror::Error;

/// The main error type for HypEmbed.
#[derive(Error, Debug)]
pub enum HypEmbedError {
    /// Tensor operation error (shape mismatch, invalid index, etc.)
    #[error("Tensor error: {0}")]
    Tensor(String),

    /// Tokenizer error (vocab loading, encoding, etc.)
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Model error (weight loading, config parsing, forward pass)
    #[error("Model error: {0}")]
    Model(String),

    /// IO error (file not found, read failure)
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// JSON parsing error
    #[error("JSON error: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, HypEmbedError>;
