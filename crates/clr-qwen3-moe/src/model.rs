use clr_core::{RuntimeError, Tensor, TensorShape, TensorView};

use crate::{
    Qwen3MoeBlock, Qwen3MoeBlockOutput, Qwen3MoeBlockWeightsSpec, Qwen3MoeConfig,
    block::{linear, rms_norm},
};

/// Unvalidated weights for the complete tiny Qwen3-MoE decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeModelWeightsSpec {
    /// Token embedding table `[vocabulary, hidden]`.
    pub token_embeddings: Tensor,
    /// Sparse decoder-block weights in layer order.
    pub blocks: Vec<Qwen3MoeBlockWeightsSpec>,
    /// Final RMS normalization weight `[hidden]`.
    pub final_norm: Tensor,
    /// Language-model head weight `[vocabulary, hidden]`.
    pub language_model_head: Tensor,
}

/// Numerical outputs from the complete tiny decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeModelOutput {
    /// Embedding output followed by every decoder layer hidden state.
    pub hidden_states: Vec<Tensor>,
    /// Per-layer numerical checkpoints and expert selections.
    pub block_outputs: Vec<Qwen3MoeBlockOutput>,
    /// Final normalized hidden state.
    pub final_norm: Tensor,
    /// Language-model logits `[sequence, vocabulary]`.
    pub logits: Tensor,
}

/// Correctness-first complete tiny Qwen3-MoE decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeModel {
    config: Qwen3MoeConfig,
    token_embeddings: Tensor,
    blocks: Vec<Qwen3MoeBlock>,
    final_norm: Tensor,
    language_model_head: Tensor,
}

impl Qwen3MoeModel {
    /// Validates weights and creates a complete tiny decoder.
    ///
    /// # Errors
    ///
    /// Returns a structured shape error when embedding, layer, normalization,
    /// or language-model-head weights do not match `config`.
    pub fn new(
        config: Qwen3MoeConfig,
        weights: Qwen3MoeModelWeightsSpec,
    ) -> Result<Self, RuntimeError> {
        let vocabulary = config.model().vocabulary_size();
        let hidden = config.model().hidden_size();
        require_shape(
            &weights.token_embeddings,
            &[vocabulary, hidden],
            "token embedding weight",
        )?;
        require_shape(&weights.final_norm, &[hidden], "final norm weight")?;
        require_shape(
            &weights.language_model_head,
            &[vocabulary, hidden],
            "language model head weight",
        )?;
        if weights.blocks.len() != config.model().layer_count() {
            return Err(RuntimeError::TensorDataLengthMismatch {
                expected: config.model().layer_count(),
                actual: weights.blocks.len(),
            });
        }
        let blocks = weights
            .blocks
            .into_iter()
            .map(|block_weights| Qwen3MoeBlock::new(config, block_weights))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            config,
            token_embeddings: weights.token_embeddings,
            blocks,
            final_norm: weights.final_norm,
            language_model_head: weights.language_model_head,
        })
    }

    /// Runs the complete tiny decoder for one non-empty token-ID sequence.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidShape`] for an empty sequence,
    /// [`RuntimeError::IndexOutOfBounds`] for an invalid token ID, or a
    /// structured numerical/shape error from a decoder stage.
    pub fn forward(&self, token_ids: &[usize]) -> Result<Qwen3MoeModelOutput, RuntimeError> {
        let embeddings = embedding_lookup(
            self.token_embeddings.view(),
            token_ids,
            self.config.model().vocabulary_size(),
        )?;
        let mut hidden_states = vec![embeddings.clone()];
        let mut block_outputs = Vec::with_capacity(self.blocks.len());
        let mut current = embeddings;
        for block in &self.blocks {
            let output = block.forward(current.view())?;
            current = output.block_output.clone();
            hidden_states.push(current.clone());
            block_outputs.push(output);
        }
        let final_norm = rms_norm(
            current.view(),
            self.final_norm.view(),
            self.config.rms_norm_epsilon(),
        )?;
        let logits = linear(
            final_norm.view(),
            self.language_model_head.view(),
            "language model head",
        )?;

        Ok(Qwen3MoeModelOutput {
            hidden_states,
            block_outputs,
            final_norm,
            logits,
        })
    }
}

pub(crate) fn embedding_lookup(
    embedding_weight: TensorView<'_>,
    token_ids: &[usize],
    vocabulary_size: usize,
) -> Result<Tensor, RuntimeError> {
    if token_ids.is_empty() {
        return Err(RuntimeError::InvalidShape {
            reason: "token ID sequence must not be empty",
        });
    }
    let hidden = embedding_weight.shape().dimensions()[1];
    let mut output = Vec::with_capacity(token_ids.len() * hidden);
    for token_id in token_ids {
        if *token_id >= vocabulary_size {
            return Err(RuntimeError::IndexOutOfBounds {
                index: *token_id,
                length: vocabulary_size,
            });
        }
        let start = token_id * hidden;
        output.extend_from_slice(&embedding_weight.data()[start..start + hidden]);
    }
    Tensor::new(TensorShape::new([token_ids.len(), hidden]), output)
}

fn require_shape(
    tensor: &Tensor,
    expected: &[usize],
    operation: &'static str,
) -> Result<(), RuntimeError> {
    if tensor.shape().dimensions() != expected {
        return Err(RuntimeError::ShapeMismatch {
            operation,
            expected: expected.into(),
            actual: tensor.shape().dimensions().into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
