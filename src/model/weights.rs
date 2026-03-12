/// Weight loading and management.
///
/// Loads model weights from a SafeTensors file, maps them to the correct
/// layer structure, and validates shapes against the model configuration.

use std::path::Path;
use crate::error::{HypEmbedError, Result};
use crate::tensor::Tensor;
use crate::model::config::ModelConfig;
use crate::model::safetensors::SafeTensorsFile;

/// Weights for a single attention layer.
#[derive(Debug)]
pub struct AttentionWeights {
    pub query_weight: Tensor,
    pub query_bias: Tensor,
    pub key_weight: Tensor,
    pub key_bias: Tensor,
    pub value_weight: Tensor,
    pub value_bias: Tensor,
    pub output_weight: Tensor,
    pub output_bias: Tensor,
    pub output_ln_weight: Tensor,
    pub output_ln_bias: Tensor,
}

/// Weights for a single feed-forward layer.
#[derive(Debug)]
pub struct FeedForwardWeights {
    pub intermediate_weight: Tensor,
    pub intermediate_bias: Tensor,
    pub output_weight: Tensor,
    pub output_bias: Tensor,
    pub output_ln_weight: Tensor,
    pub output_ln_bias: Tensor,
}

/// Weights for a single encoder layer.
#[derive(Debug)]
pub struct EncoderLayerWeights {
    pub attention: AttentionWeights,
    pub ff: FeedForwardWeights,
}

/// All model weights.
#[derive(Debug)]
pub struct ModelWeights {
    pub word_embeddings: Tensor,
    pub position_embeddings: Tensor,
    pub token_type_embeddings: Tensor,
    pub embedding_ln_weight: Tensor,
    pub embedding_ln_bias: Tensor,
    pub layers: Vec<EncoderLayerWeights>,
}

impl ModelWeights {
    /// Load weights from a SafeTensors file.
    ///
    /// Maps HuggingFace BERT naming convention to our weight structure.
    pub fn load<P: AsRef<Path>>(path: P, config: &ModelConfig) -> Result<Self> {
        let st = SafeTensorsFile::load(path)?;

        // Embedding weights
        let word_embeddings = st.get_tensor("embeddings.word_embeddings.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.word_embeddings.weight"))?;
        let position_embeddings = st.get_tensor("embeddings.position_embeddings.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.position_embeddings.weight"))?;
        let token_type_embeddings = st.get_tensor("embeddings.token_type_embeddings.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.token_type_embeddings.weight"))?;
        let embedding_ln_weight = st.get_tensor("embeddings.LayerNorm.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.LayerNorm.weight"))?;
        let embedding_ln_bias = st.get_tensor("embeddings.LayerNorm.bias")
            .or_else(|_| st.get_tensor("bert.embeddings.LayerNorm.bias"))?;

        // Validate embedding shapes
        Self::validate_shape(&word_embeddings, &[config.vocab_size, config.hidden_size], "word_embeddings")?;
        Self::validate_shape(&position_embeddings, &[config.max_position_embeddings, config.hidden_size], "position_embeddings")?;
        Self::validate_shape(&token_type_embeddings, &[config.type_vocab_size, config.hidden_size], "token_type_embeddings")?;

        // Load each encoder layer
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Self::load_layer(&st, config, i)?;
            layers.push(layer);
        }

        Ok(ModelWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_ln_weight,
            embedding_ln_bias,
            layers,
        })
    }

    fn load_layer(st: &SafeTensorsFile, config: &ModelConfig, layer_idx: usize) -> Result<EncoderLayerWeights> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;

        // Try both naming conventions: with and without "bert." prefix
        let get = |short_name: &str| -> Result<Tensor> {
            let full = format!("encoder.layer.{}.{}", layer_idx, short_name);
            let bert_full = format!("bert.encoder.layer.{}.{}", layer_idx, short_name);
            st.get_tensor(&full).or_else(|_| st.get_tensor(&bert_full))
        };

        let attention = AttentionWeights {
            query_weight: get("attention.self.query.weight")?,
            query_bias: get("attention.self.query.bias")?,
            key_weight: get("attention.self.key.weight")?,
            key_bias: get("attention.self.key.bias")?,
            value_weight: get("attention.self.value.weight")?,
            value_bias: get("attention.self.value.bias")?,
            output_weight: get("attention.output.dense.weight")?,
            output_bias: get("attention.output.dense.bias")?,
            output_ln_weight: get("attention.output.LayerNorm.weight")?,
            output_ln_bias: get("attention.output.LayerNorm.bias")?,
        };

        // Validate attention weight shapes
        Self::validate_shape(&attention.query_weight, &[h, h], "query_weight")?;
        Self::validate_shape(&attention.query_bias, &[h], "query_bias")?;

        let ff = FeedForwardWeights {
            intermediate_weight: get("intermediate.dense.weight")?,
            intermediate_bias: get("intermediate.dense.bias")?,
            output_weight: get("output.dense.weight")?,
            output_bias: get("output.dense.bias")?,
            output_ln_weight: get("output.LayerNorm.weight")?,
            output_ln_bias: get("output.LayerNorm.bias")?,
        };

        // Validate FF weight shapes
        Self::validate_shape(&ff.intermediate_weight, &[inter, h], "intermediate_weight")?;
        Self::validate_shape(&ff.output_weight, &[h, inter], "output_weight")?;

        Ok(EncoderLayerWeights { attention, ff })
    }

    fn validate_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
        if tensor.shape().dims() != expected {
            return Err(HypEmbedError::Model(format!(
                "Shape mismatch for '{}': expected {:?}, got {}",
                name, expected, tensor.shape()
            )));
        }
        Ok(())
    }
}
