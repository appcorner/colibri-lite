use std::fmt;

use clr_core::{DataType, RuntimeError, Tensor, TensorView, ops::elementwise_add};
use clr_storage::{ByteOrder, ExpertKey, ExpertLoadObservation, ExpertStore, StorageError};

#[cfg(all(test, feature = "full-model-validation"))]
use crate::block::{ExpertMlpTrace, expert_mlp_trace};
use crate::{
    Qwen3MoeBlockOutput, Qwen3MoeConfig, Qwen3MoeModelOutput,
    block::{combine_routed_experts, expert_mlp, linear, pre_router_with_weights, rms_norm},
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
    pub(crate) config: Qwen3MoeConfig,
    pub(crate) weights: StreamingModelWeightsSpec,
    pub(crate) layout: PackedExpertLayout,
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
        let pre_router = pre_router_with_weights(
            hidden,
            weights.input_norm.view(),
            weights.query_projection.view(),
            weights.key_projection.view(),
            weights.value_projection.view(),
            weights.output_projection.view(),
            weights.query_norm.view(),
            weights.key_norm.view(),
            weights.post_attention_norm.view(),
            weights.router.view(),
            self.config,
        )?;
        let moe_output = streaming_routed_experts(
            pre_router.post_attention_norm.view(),
            &pre_router.router,
            self.config,
            layer_index,
            store,
            self.layout,
        )?;
        let block_output = elementwise_add(pre_router.residual_output.view(), moe_output.view())?;
        Ok(Qwen3MoeBlockOutput {
            input_norm: pre_router.input_norm,
            attention_output: pre_router.attention_output,
            post_attention_norm: pre_router.post_attention_norm,
            router_logits: pre_router.router.logits,
            routing_weights: pre_router.router.weights,
            selected_experts: pre_router.router.selected_experts,
            moe_output,
            block_output,
        })
    }
}

pub(crate) fn streaming_routed_experts(
    hidden_states: TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: Qwen3MoeConfig,
    layer_index: usize,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
) -> Result<Tensor, StreamingModelError> {
    streaming_routed_experts_with_observer(
        hidden_states,
        router,
        config,
        layer_index,
        store,
        layout,
        |_, _, _, _| {},
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn streaming_routed_experts_with_observer<F>(
    hidden_states: TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: Qwen3MoeConfig,
    layer_index: usize,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
    mut observe: F,
) -> Result<Tensor, StreamingModelError>
where
    F: FnMut(usize, usize, usize, &[f32]),
{
    let hidden_size = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    combine_routed_experts(
        hidden_states,
        router,
        config,
        |expert_id, occurrences| -> Result<Vec<Vec<f32>>, StreamingModelError> {
            let key = ExpertKey {
                layer_index: u32::try_from(layer_index).unwrap_or(u32::MAX),
                expert_id: clr_storage::ExpertId(u32::try_from(expert_id).unwrap_or(u32::MAX)),
            };
            let lease = store.load(key)?;
            let decoded = decode_payload(key, lease.bytes(), layout)?;
            let mut outputs = Vec::with_capacity(occurrences.len());
            for &(token, position) in occurrences {
                let input = &hidden_states.data()[token * hidden_size..(token + 1) * hidden_size];
                let output = expert_mlp(
                    input,
                    &decoded.gate,
                    &decoded.up,
                    &decoded.down,
                    hidden_size,
                    intermediate,
                );
                observe(expert_id, token, position, &output);
                outputs.push(output);
            }
            Ok(outputs)
        },
    )
}

#[cfg(feature = "full-model-validation")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn streaming_routed_experts_with_request_observer<R, F>(
    hidden_states: TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: Qwen3MoeConfig,
    layer_index: usize,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
    mut request_observer: R,
    mut observe: F,
) -> Result<Tensor, StreamingModelError>
where
    R: FnMut(usize, usize, usize, usize, usize, ExpertLoadObservation),
    F: FnMut(usize, usize, usize, &[f32]),
{
    let hidden_size = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    combine_routed_experts(
        hidden_states,
        router,
        config,
        |expert_id, occurrences| -> Result<Vec<Vec<f32>>, StreamingModelError> {
            let key = ExpertKey {
                layer_index: u32::try_from(layer_index).unwrap_or(u32::MAX),
                expert_id: clr_storage::ExpertId(u32::try_from(expert_id).unwrap_or(u32::MAX)),
            };
            let mut observation = None;
            let lease = store.load_with_observer(key, |value| observation = Some(value))?;
            let observation = observation.expect("expert load observer called exactly once");
            let decoded = decode_payload(key, lease.bytes(), layout)?;
            let mut outputs = Vec::with_capacity(occurrences.len());
            for &(token, position) in occurrences {
                let top_k = config.experts_per_token();
                let rank = router.selected_experts[token * top_k..(token + 1) * top_k]
                    .iter()
                    .position(|&selected| selected == expert_id)
                    .expect("occurrence expert has a selected rank");
                request_observer(layer_index, expert_id, token, position, rank, observation);
                let input = &hidden_states.data()[token * hidden_size..(token + 1) * hidden_size];
                let output = expert_mlp(
                    input,
                    &decoded.gate,
                    &decoded.up,
                    &decoded.down,
                    hidden_size,
                    intermediate,
                );
                observe(expert_id, token, position, &output);
                outputs.push(output);
            }
            Ok(outputs)
        },
    )
}

#[cfg(all(test, feature = "full-model-validation"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn streaming_routed_experts_with_trace_observer<P, F>(
    hidden_states: TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: Qwen3MoeConfig,
    layer_index: usize,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
    mut trace_requested: P,
    mut observe: F,
) -> Result<Tensor, StreamingModelError>
where
    P: FnMut(usize, usize, usize) -> bool,
    F: FnMut(usize, usize, usize, &[f32], &ExpertMlpTrace),
{
    let hidden_size = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    combine_routed_experts(
        hidden_states,
        router,
        config,
        |expert_id, occurrences| -> Result<Vec<Vec<f32>>, StreamingModelError> {
            let key = ExpertKey {
                layer_index: u32::try_from(layer_index).unwrap_or(u32::MAX),
                expert_id: clr_storage::ExpertId(u32::try_from(expert_id).unwrap_or(u32::MAX)),
            };
            let lease = store.load(key)?;
            let decoded = decode_payload(key, lease.bytes(), layout)?;
            let mut outputs = Vec::with_capacity(occurrences.len());
            for &(token, position) in occurrences {
                let input = &hidden_states.data()[token * hidden_size..(token + 1) * hidden_size];
                let output = expert_mlp(
                    input,
                    &decoded.gate,
                    &decoded.up,
                    &decoded.down,
                    hidden_size,
                    intermediate,
                );
                if trace_requested(expert_id, token, position) {
                    let trace = expert_mlp_trace(
                        input,
                        &decoded.gate,
                        &decoded.up,
                        &decoded.down,
                        hidden_size,
                        intermediate,
                    );
                    observe(expert_id, token, position, &output, &trace);
                }
                outputs.push(output);
            }
            Ok(outputs)
        },
    )
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
