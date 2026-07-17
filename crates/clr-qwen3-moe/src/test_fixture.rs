use clr_core::{DataType, ModelConfig, ModelConfigSpec, Tensor, TensorShape};

use crate::{
    Qwen3MoeBlockWeightsSpec, Qwen3MoeConfig, Qwen3MoeConfigSpec, Qwen3MoeModel,
    Qwen3MoeModelWeightsSpec,
};

const WEIGHTS_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/reference/fixtures/tiny-qwen3-moe/weights.safetensors"
));
const CHECKPOINT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/reference/fixtures/tiny-qwen3-moe/checkpoints.safetensors"
));

#[allow(dead_code)]
pub(crate) mod frozen_config {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../python/reference/fixtures/tiny-qwen3-moe/rust-config.rs"
    ));
}

pub(crate) fn config() -> Qwen3MoeConfig {
    let model = ModelConfig::new(ModelConfigSpec {
        vocabulary_size: frozen_config::VOCABULARY_SIZE,
        hidden_size: frozen_config::HIDDEN_SIZE,
        layer_count: frozen_config::LAYER_COUNT,
        attention_head_count: frozen_config::ATTENTION_HEAD_COUNT,
        key_value_head_count: frozen_config::KEY_VALUE_HEAD_COUNT,
        head_dimension: frozen_config::HEAD_DIMENSION,
        intermediate_size: frozen_config::INTERMEDIATE_SIZE,
        max_sequence_length: frozen_config::MAX_SEQUENCE_LENGTH,
        data_type: DataType::F32,
    })
    .expect("valid frozen generic config");
    Qwen3MoeConfig::new(Qwen3MoeConfigSpec {
        model,
        rms_norm_epsilon: frozen_config::RMS_NORM_EPSILON,
        rope_theta: frozen_config::ROPE_THETA,
        expert_count: frozen_config::EXPERT_COUNT,
        experts_per_token: frozen_config::EXPERTS_PER_TOKEN,
        moe_intermediate_size: frozen_config::MOE_INTERMEDIATE_SIZE,
        normalize_topk_probabilities: frozen_config::NORMALIZE_TOPK_PROBABILITIES,
    })
    .expect("valid frozen Qwen config")
}

pub(crate) fn block_weights(layer: usize) -> Qwen3MoeBlockWeightsSpec {
    Qwen3MoeBlockWeightsSpec {
        input_norm: layer_weight(layer, "input_norm"),
        query_projection: layer_weight(layer, "query_projection"),
        key_projection: layer_weight(layer, "key_projection"),
        value_projection: layer_weight(layer, "value_projection"),
        output_projection: layer_weight(layer, "output_projection"),
        query_norm: layer_weight(layer, "query_norm"),
        key_norm: layer_weight(layer, "key_norm"),
        post_attention_norm: layer_weight(layer, "post_attention_norm"),
        router: layer_weight(layer, "router"),
        expert_gate_up: layer_weight(layer, "expert_gate_up"),
        expert_down: layer_weight(layer, "expert_down"),
    }
}

pub(crate) fn token_embeddings() -> Tensor {
    read_f32(WEIGHTS_BYTES, 6_616, 10_712, &[64, 16])
}

pub(crate) fn final_norm_weight() -> Tensor {
    read_f32(WEIGHTS_BYTES, 54_552, 54_616, &[16])
}

pub(crate) fn language_model_head() -> Tensor {
    read_f32(WEIGHTS_BYTES, 2_520, 6_616, &[64, 16])
}

pub(crate) fn token_ids() -> Vec<usize> {
    read_i64(CHECKPOINT_BYTES, 2_272, 2_304)
}

/// Builds the frozen deterministic tiny Qwen3-MoE model bundled for
/// correctness checks and the M3 token-ID CLI.
///
/// # Errors
///
/// Returns a structured validation error if the versioned fixture no longer
/// satisfies the model contract.
pub fn frozen_tiny_model() -> Result<Qwen3MoeModel, clr_core::RuntimeError> {
    Qwen3MoeModel::new(
        config(),
        Qwen3MoeModelWeightsSpec {
            token_embeddings: token_embeddings(),
            blocks: vec![block_weights(0), block_weights(1)],
            final_norm: final_norm_weight(),
            language_model_head: language_model_head(),
        },
    )
}

/// Returns the frozen prompt token IDs used by the tiny correctness oracle.
#[must_use]
pub fn frozen_tiny_prompt() -> Vec<usize> {
    token_ids()
}

#[cfg(test)]
pub(crate) fn hidden_state(index: usize) -> Tensor {
    let (start, end) = match index {
        0 => (3_712, 3_968),
        1 => (3_968, 4_224),
        2 => (4_224, 4_480),
        _ => panic!("unknown hidden-state checkpoint {index}"),
    };
    read_f32(CHECKPOINT_BYTES, start, end, &[4, 16])
}

#[cfg(test)]
pub(crate) fn block_checkpoint(layer: usize, name: &str) -> Tensor {
    let (start, end, shape): (usize, usize, &[usize]) = match (layer, name) {
        (0, "attention_output") => (4_480, 4_736, &[4, 16]),
        (0, "block_output") => (4_736, 4_992, &[4, 16]),
        (0, "input_norm") => (4_992, 5_248, &[4, 16]),
        (0, "moe_output") => (5_376, 5_632, &[4, 16]),
        (0, "post_attention_norm") => (5_632, 5_888, &[4, 16]),
        (0, "router_logits") => (6_144, 6_208, &[4, 4]),
        (0, "routing_weights") => (6_208, 6_240, &[4, 2]),
        (1, "attention_output") => (6_240, 6_496, &[4, 16]),
        (1, "block_output") => (6_496, 6_752, &[4, 16]),
        (1, "input_norm") => (6_752, 7_008, &[4, 16]),
        (1, "moe_output") => (7_136, 7_392, &[4, 16]),
        (1, "post_attention_norm") => (7_392, 7_648, &[4, 16]),
        (1, "router_logits") => (7_904, 7_968, &[4, 4]),
        (1, "routing_weights") => (7_968, 8_000, &[4, 2]),
        _ => panic!("unknown layer {layer} checkpoint: {name}"),
    };
    read_f32(CHECKPOINT_BYTES, start, end, shape)
}

#[cfg(test)]
pub(crate) fn final_norm_checkpoint() -> Tensor {
    read_f32(CHECKPOINT_BYTES, 3_456, 3_712, &[4, 16])
}

#[cfg(test)]
pub(crate) fn final_logits() -> Tensor {
    read_f32(CHECKPOINT_BYTES, 2_432, 3_456, &[4, 64])
}

#[cfg(test)]
pub(crate) fn selected_experts(layer: usize) -> Vec<usize> {
    let (start, end) = match layer {
        0 => (2_304, 2_368),
        1 => (2_368, 2_432),
        _ => panic!("unknown selected-expert layer {layer}"),
    };
    read_i64(CHECKPOINT_BYTES, start, end)
}

fn layer_weight(layer: usize, name: &str) -> Tensor {
    let (start, end, shape): (usize, usize, &[usize]) = match (layer, name) {
        (0, "input_norm") => (10_712, 10_776, &[16]),
        (0, "expert_down") => (10_776, 16_920, &[4, 16, 24]),
        (0, "expert_gate_up") => (16_920, 29_208, &[4, 48, 16]),
        (0, "router") => (29_208, 29_464, &[4, 16]),
        (0, "post_attention_norm") => (29_464, 29_528, &[16]),
        (0, "key_norm") => (29_528, 29_544, &[4]),
        (0, "key_projection") => (29_544, 30_056, &[8, 16]),
        (0, "output_projection") => (30_056, 31_080, &[16, 16]),
        (0, "query_norm") => (31_080, 31_096, &[4]),
        (0, "query_projection") => (31_096, 32_120, &[16, 16]),
        (0, "value_projection") => (32_120, 32_632, &[8, 16]),
        (1, "input_norm") => (32_632, 32_696, &[16]),
        (1, "expert_down") => (32_696, 38_840, &[4, 16, 24]),
        (1, "expert_gate_up") => (38_840, 51_128, &[4, 48, 16]),
        (1, "router") => (51_128, 51_384, &[4, 16]),
        (1, "post_attention_norm") => (51_384, 51_448, &[16]),
        (1, "key_norm") => (51_448, 51_464, &[4]),
        (1, "key_projection") => (51_464, 51_976, &[8, 16]),
        (1, "output_projection") => (51_976, 53_000, &[16, 16]),
        (1, "query_norm") => (53_000, 53_016, &[4]),
        (1, "query_projection") => (53_016, 54_040, &[16, 16]),
        (1, "value_projection") => (54_040, 54_552, &[8, 16]),
        _ => panic!("unknown layer {layer} fixture weight: {name}"),
    };
    read_f32(WEIGHTS_BYTES, start, end, shape)
}

fn read_f32(bytes: &[u8], start: usize, end: usize, shape: &[usize]) -> Tensor {
    let data = bytes[start..end]
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte f32")))
        .collect();
    Tensor::new(TensorShape::new(shape.to_vec()), data).expect("valid frozen f32 tensor")
}

fn read_i64(bytes: &[u8], start: usize, end: usize) -> Vec<usize> {
    bytes[start..end]
        .chunks_exact(8)
        .map(|value| {
            usize::try_from(i64::from_le_bytes(
                value.try_into().expect("eight-byte integer"),
            ))
            .expect("non-negative fixture integer")
        })
        .collect()
}
