/// Model module.
///
/// Implements a BERT-like encoder architecture:
/// - Model configuration parsing
/// - SafeTensors weight loading
/// - Token/position/segment embeddings
/// - Multi-head self-attention
/// - Feed-forward network
/// - Encoder layer stacking
/// - Pooling strategies

pub mod config;
pub mod safetensors;
pub mod weights;
pub mod embedding;
pub mod attention;
pub mod ff;
pub mod encoder;
pub mod pool;
