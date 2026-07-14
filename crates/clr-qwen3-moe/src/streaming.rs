use std::fmt;

use clr_core::{DataType, RuntimeError, Tensor, TensorView, ops::elementwise_add};
use clr_storage::{ByteOrder, ExpertKey, ExpertStore, StorageError};

use crate::{
    Qwen3MoeBlockOutput, Qwen3MoeConfig, Qwen3MoeModelOutput,
    block::{
        attention_with_weights, combine_routed_experts, expert_mlp, linear, rms_norm, route_tokens,
    },
    model::embedding_lookup,
};

/// Byte ranges for one packed expert payload in gate/up/down order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedExpertLayout {
    pub gate_offset: usize,
    pub gate_length: usize,
    pub up_offset: usize,
    pub up_length: usize,
    pub down_offset: usize,
    pub down_length: usize,
    pub data_type: DataType,
    pub byte_order: ByteOrder,
    pub total_byte_length: usize,
}

impl PackedExpertLayout {
    #[must_use]
    pub fn for_config(config: Qwen3MoeConfig) -> Self {
        let hidden = config.model().hidden_size();
        let intermediate = config.moe_intermediate_size();
        let matrix_bytes = hidden * intermediate * DataType::F32.byte_width();
        Self {
            gate_offset: 0,
            gate_length: matrix_bytes,
            up_offset: matrix_bytes,
            up_length: matrix_bytes,
            down_offset: 2 * matrix_bytes,
            down_length: matrix_bytes,
            data_type: DataType::F32,
            byte_order: ByteOrder::Little,
            total_byte_length: 3 * matrix_bytes,
        }
    }
}

/// Dense and router weights retained for one streaming sparse block.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingBlockWeightsSpec {
    pub input_norm: Tensor,
    pub query_projection: Tensor,
    pub key_projection: Tensor,
    pub value_projection: Tensor,
    pub output_projection: Tensor,
    pub query_norm: Tensor,
    pub key_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub router: Tensor,
}

/// Resident non-expert weights for the streaming tiny model.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingModelWeightsSpec {
    pub token_embeddings: Tensor,
    pub blocks: Vec<StreamingBlockWeightsSpec>,
    pub final_norm: Tensor,
    pub language_model_head: Tensor,
}

/// Errors from storage-aware model execution.
#[derive(Debug)]
pub enum StreamingModelError {
    Runtime(RuntimeError),
    Storage(StorageError),
    InvalidExpertPayload {
        key: ExpertKey,
        reason: &'static str,
    },
}

impl From<RuntimeError> for StreamingModelError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<StorageError> for StreamingModelError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}

impl fmt::Display for StreamingModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => write!(formatter, "runtime error: {error}"),
            Self::Storage(error) => write!(formatter, "storage error: {error}"),
            Self::InvalidExpertPayload { key, reason } => {
                write!(formatter, "invalid packed expert {key:?}: {reason}")
            }
        }
    }
}

impl std::error::Error for StreamingModelError {}

/// Tiny Qwen3-MoE path with dense/router weights resident and experts on demand.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingQwen3MoeModel {
    config: Qwen3MoeConfig,
    weights: StreamingModelWeightsSpec,
    layout: PackedExpertLayout,
}

struct DecodedExpert {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

impl StreamingQwen3MoeModel {
    #[must_use]
    pub fn new(config: Qwen3MoeConfig, weights: StreamingModelWeightsSpec) -> Self {
        Self {
            config,
            weights,
            layout: PackedExpertLayout::for_config(config),
        }
    }

    /// Runs the model while holding each expert lease through its computation.
    ///
    /// # Errors
    ///
    /// Returns a runtime, storage, or malformed-payload error before producing
    /// output from invalid expert bytes.
    pub fn forward(
        &self,
        token_ids: &[usize],
        store: &mut ExpertStore,
    ) -> Result<Qwen3MoeModelOutput, StreamingModelError> {
        let mut current = embedding_lookup(
            self.weights.token_embeddings.view(),
            token_ids,
            self.config.model().vocabulary_size(),
        )?;
        let mut hidden_states = vec![current.clone()];
        let mut block_outputs = Vec::with_capacity(self.weights.blocks.len());
        for (layer_index, weights) in self.weights.blocks.iter().enumerate() {
            let output = self.forward_block(layer_index, current.view(), weights, store)?;
            current = output.block_output.clone();
            hidden_states.push(current.clone());
            block_outputs.push(output);
        }
        let final_norm = rms_norm(
            current.view(),
            self.weights.final_norm.view(),
            self.config.rms_norm_epsilon(),
        )?;
        let logits = linear(
            final_norm.view(),
            self.weights.language_model_head.view(),
            "streaming language model head",
        )?;
        Ok(Qwen3MoeModelOutput {
            hidden_states,
            block_outputs,
            final_norm,
            logits,
        })
    }

    fn forward_block(
        &self,
        layer_index: usize,
        hidden: TensorView<'_>,
        weights: &StreamingBlockWeightsSpec,
        store: &mut ExpertStore,
    ) -> Result<Qwen3MoeBlockOutput, StreamingModelError> {
        let input_norm = rms_norm(
            hidden,
            weights.input_norm.view(),
            self.config.rms_norm_epsilon(),
        )?;
        let attention_output = attention_with_weights(
            input_norm.view(),
            self.config,
            weights.query_projection.view(),
            weights.key_projection.view(),
            weights.value_projection.view(),
            weights.output_projection.view(),
            weights.query_norm.view(),
            weights.key_norm.view(),
        )?;
        let after_attention = elementwise_add(hidden, attention_output.view())?;
        let post_attention_norm = rms_norm(
            after_attention.view(),
            weights.post_attention_norm.view(),
            self.config.rms_norm_epsilon(),
        )?;
        let router = route_tokens(
            post_attention_norm.view(),
            weights.router.view(),
            self.config,
        )?;
        let hidden_size = self.config.model().hidden_size();
        let intermediate = self.config.moe_intermediate_size();
        let moe_output = combine_routed_experts(
            post_attention_norm.view(),
            &router,
            self.config,
            |expert_id, occurrences| -> Result<Vec<Vec<f32>>, StreamingModelError> {
                let key = ExpertKey {
                    layer_index: u32::try_from(layer_index).unwrap_or(u32::MAX),
                    expert_id: clr_storage::ExpertId(u32::try_from(expert_id).unwrap_or(u32::MAX)),
                };
                let lease = store.load(key)?;
                let decoded = decode_payload(key, lease.bytes(), self.layout)?;
                Ok(occurrences
                    .iter()
                    .map(|(token, _)| {
                        let input = &post_attention_norm.data()
                            [token * hidden_size..(token + 1) * hidden_size];
                        expert_mlp(
                            input,
                            &decoded.gate,
                            &decoded.up,
                            &decoded.down,
                            hidden_size,
                            intermediate,
                        )
                    })
                    .collect())
            },
        )?;
        let block_output = elementwise_add(after_attention.view(), moe_output.view())?;
        Ok(Qwen3MoeBlockOutput {
            input_norm,
            attention_output,
            post_attention_norm,
            router_logits: router.logits,
            routing_weights: router.weights,
            selected_experts: router.selected_experts,
            moe_output,
            block_output,
        })
    }
}

fn decode_payload(
    key: ExpertKey,
    bytes: &[u8],
    layout: PackedExpertLayout,
) -> Result<DecodedExpert, StreamingModelError> {
    if layout.data_type != DataType::F32
        || layout.byte_order != ByteOrder::Little
        || bytes.len() != layout.total_byte_length
    {
        return Err(StreamingModelError::InvalidExpertPayload {
            key,
            reason: "layout or total byte length mismatch",
        });
    }
    Ok(DecodedExpert {
        gate: decode_f32(&bytes[layout.gate_offset..layout.gate_offset + layout.gate_length]),
        up: decode_f32(&bytes[layout.up_offset..layout.up_offset + layout.up_length]),
        down: decode_f32(&bytes[layout.down_offset..layout.down_offset + layout.down_length]),
    })
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32 payload")))
        .collect()
}

#[cfg(test)]
mod tests;
