use clr_core::{DataType, ModelConfig, ModelConfigSpec, Tensor, TensorShape};

use super::*;
use crate::Qwen3MoeConfigSpec;

const WEIGHTS_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/reference/fixtures/tiny-qwen3-moe/weights.safetensors"
));
const CHECKPOINT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/reference/fixtures/tiny-qwen3-moe/checkpoints.safetensors"
));

#[allow(dead_code)]
mod frozen_config {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../python/reference/fixtures/tiny-qwen3-moe/rust-config.rs"
    ));
}

#[derive(Debug, PartialEq)]
enum OracleMismatch {
    Shape {
        stage: &'static str,
        expected: Box<[usize]>,
        actual: Box<[usize]>,
    },
    Value {
        stage: &'static str,
        index: usize,
        expected: f32,
        actual: f32,
        tolerance: f32,
    },
}

impl OracleMismatch {
    const fn stage(&self) -> &'static str {
        match self {
            Self::Shape { stage, .. } | Self::Value { stage, .. } => stage,
        }
    }
}

fn fixture_config() -> Qwen3MoeConfig {
    config_with_theta_and_normalization(frozen_config::ROPE_THETA, false)
}

fn config_with_theta_and_normalization(
    rope_theta: f32,
    normalize_topk_probabilities: bool,
) -> Qwen3MoeConfig {
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
        rope_theta,
        expert_count: frozen_config::EXPERT_COUNT,
        experts_per_token: frozen_config::EXPERTS_PER_TOKEN,
        moe_intermediate_size: frozen_config::MOE_INTERMEDIATE_SIZE,
        normalize_topk_probabilities,
    })
    .expect("valid frozen Qwen config")
}

fn layer_zero_weights() -> Qwen3MoeBlockWeightsSpec {
    Qwen3MoeBlockWeightsSpec {
        input_norm: weight("input_norm"),
        query_projection: weight("query_projection"),
        key_projection: weight("key_projection"),
        value_projection: weight("value_projection"),
        output_projection: weight("output_projection"),
        query_norm: weight("query_norm"),
        key_norm: weight("key_norm"),
        post_attention_norm: weight("post_attention_norm"),
        router: weight("router"),
        expert_gate_up: weight("expert_gate_up"),
        expert_down: weight("expert_down"),
    }
}

fn weight(name: &str) -> Tensor {
    let (start, end, shape): (usize, usize, &[usize]) = match name {
        "input_norm" => (10_712, 10_776, &[16]),
        "expert_down" => (10_776, 16_920, &[4, 16, 24]),
        "expert_gate_up" => (16_920, 29_208, &[4, 48, 16]),
        "router" => (29_208, 29_464, &[4, 16]),
        "post_attention_norm" => (29_464, 29_528, &[16]),
        "key_norm" => (29_528, 29_544, &[4]),
        "key_projection" => (29_544, 30_056, &[8, 16]),
        "output_projection" => (30_056, 31_080, &[16, 16]),
        "query_norm" => (31_080, 31_096, &[4]),
        "query_projection" => (31_096, 32_120, &[16, 16]),
        "value_projection" => (32_120, 32_632, &[8, 16]),
        _ => panic!("unknown layer-zero fixture weight: {name}"),
    };
    read_f32_tensor(WEIGHTS_BYTES, start, end, shape)
}

fn checkpoint_matrix(name: &str) -> Tensor {
    let (start, end, shape): (usize, usize, &[usize]) = match name {
        "hidden_state_0" => (3_712, 3_968, &[4, 16]),
        "attention_output" => (4_480, 4_736, &[4, 16]),
        "block_output" => (4_736, 4_992, &[4, 16]),
        "input_norm" => (4_992, 5_248, &[4, 16]),
        "moe_output" => (5_376, 5_632, &[4, 16]),
        "post_attention_norm" => (5_632, 5_888, &[4, 16]),
        "router_logits" => (6_144, 6_208, &[4, 4]),
        "routing_weights" => (6_208, 6_240, &[4, 2]),
        _ => panic!("unknown layer-zero checkpoint: {name}"),
    };
    read_f32_tensor(CHECKPOINT_BYTES, start, end, shape)
}

fn rope_checkpoint(name: &str) -> Tensor {
    let (start, end, shape): (usize, usize, &[usize]) = match name {
        "key_rope" => (5_248, 5_376, &[1, 2, 4, 4]),
        "query_rope" => (5_888, 6_144, &[1, 4, 4, 4]),
        _ => panic!("unknown RoPE checkpoint: {name}"),
    };
    read_f32_tensor(CHECKPOINT_BYTES, start, end, shape)
}

fn selected_experts() -> Vec<usize> {
    CHECKPOINT_BYTES[2_304..2_368]
        .chunks_exact(8)
        .map(|bytes| {
            usize::try_from(i64::from_le_bytes(
                bytes.try_into().expect("eight-byte expert ID"),
            ))
            .expect("non-negative expert ID")
        })
        .collect()
}

fn read_f32_tensor(bytes: &[u8], start: usize, end: usize, shape: &[usize]) -> Tensor {
    let data = bytes[start..end]
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte f32")))
        .collect();
    Tensor::new(TensorShape::new(shape.to_vec()), data).expect("valid frozen tensor")
}

fn compare_stage(
    stage: &'static str,
    actual: &Tensor,
    expected: &Tensor,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Result<(), OracleMismatch> {
    if actual.shape() != expected.shape() {
        return Err(OracleMismatch::Shape {
            stage,
            expected: expected.shape().dimensions().into(),
            actual: actual.shape().dimensions().into(),
        });
    }
    for (index, (actual_value, expected_value)) in
        actual.data().iter().zip(expected.data()).enumerate()
    {
        let tolerance = absolute_tolerance + relative_tolerance * expected_value.abs();
        if (actual_value - expected_value).abs() > tolerance {
            return Err(OracleMismatch::Value {
                stage,
                index,
                expected: *expected_value,
                actual: *actual_value,
                tolerance,
            });
        }
    }
    Ok(())
}

fn assert_stage(
    stage: &'static str,
    actual: &Tensor,
    expected: &Tensor,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) {
    if let Err(mismatch) = compare_stage(
        stage,
        actual,
        expected,
        absolute_tolerance,
        relative_tolerance,
    ) {
        panic!("first oracle mismatch at stage '{stage}': {mismatch:?}");
    }
}

fn oracle_head_layout(input: &Tensor) -> Tensor {
    let [sequence_length, head_count, head_dimension] =
        <[usize; 3]>::try_from(input.shape().dimensions()).expect("rank-three head tensor");
    let mut data = Vec::with_capacity(input.data().len());
    for head in 0..head_count {
        for position in 0..sequence_length {
            let start = (position * head_count + head) * head_dimension;
            data.extend_from_slice(&input.data()[start..start + head_dimension]);
        }
    }
    Tensor::new(
        TensorShape::new([1, head_count, sequence_length, head_dimension]),
        data,
    )
    .expect("valid oracle head layout")
}

#[test]
fn rms_norm_matches_frozen_oracle() {
    let weights = layer_zero_weights();
    let actual = rms_norm(
        checkpoint_matrix("hidden_state_0").view(),
        weights.input_norm.view(),
        fixture_config().rms_norm_epsilon(),
    )
    .expect("RMSNorm succeeds");

    assert_stage(
        "layer_0.input_norm",
        &actual,
        &checkpoint_matrix("input_norm"),
        1.0e-6,
        1.0e-5,
    );
}

#[test]
fn rope_reads_theta_from_config_and_changes_with_theta() {
    let query = Tensor::new(
        TensorShape::new([2, 1, 4]),
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    )
    .expect("valid query");
    let key = query.clone();
    let fixture_theta = config_with_theta_and_normalization(10_000.0, false);
    let alternate_theta = config_with_theta_and_normalization(1_000_000.0, false);

    let (fixture_output, _) =
        apply_rotary_embeddings(query.view(), key.view(), fixture_theta).expect("fixture RoPE");
    let (alternate_output, _) =
        apply_rotary_embeddings(query.view(), key.view(), alternate_theta).expect("alternate RoPE");

    assert_ne!(fixture_output.data(), alternate_output.data());
    assert_eq!(fixture_theta.rope_theta().to_bits(), 10_000.0_f32.to_bits());
}

#[test]
fn rope_matches_frozen_oracle() {
    let config = fixture_config();
    let weights = layer_zero_weights();
    let hidden = checkpoint_matrix("input_norm");
    let sequence_length = hidden.shape().dimensions()[0];
    let head_dimension = config.head_dimension();
    let query =
        linear(hidden.view(), weights.query_projection.view(), "query").expect("query projection");
    let key = linear(hidden.view(), weights.key_projection.view(), "key").expect("key projection");
    let query = Tensor::new(
        TensorShape::new([
            sequence_length,
            config.model().attention_head_count(),
            head_dimension,
        ]),
        query.into_data(),
    )
    .expect("query heads");
    let key = Tensor::new(
        TensorShape::new([
            sequence_length,
            config.model().key_value_head_count(),
            head_dimension,
        ]),
        key.into_data(),
    )
    .expect("key heads");
    let query = rms_norm(
        query.view(),
        weights.query_norm.view(),
        config.rms_norm_epsilon(),
    )
    .expect("query norm");
    let key = rms_norm(
        key.view(),
        weights.key_norm.view(),
        config.rms_norm_epsilon(),
    )
    .expect("key norm");
    let (query, key) = apply_rotary_embeddings(query.view(), key.view(), config).expect("RoPE");

    assert_stage(
        "layer_0.query_rope",
        &oracle_head_layout(&query),
        &rope_checkpoint("query_rope"),
        1.0e-6,
        1.0e-5,
    );
    assert_stage(
        "layer_0.key_rope",
        &oracle_head_layout(&key),
        &rope_checkpoint("key_rope"),
        1.0e-6,
        1.0e-5,
    );
}

#[test]
fn causal_grouped_query_attention_matches_frozen_oracle() {
    let weights = layer_zero_weights();
    let actual = attention(
        checkpoint_matrix("input_norm").view(),
        fixture_config(),
        &weights,
    )
    .expect("attention succeeds");

    assert_stage(
        "layer_0.attention_output",
        &actual,
        &checkpoint_matrix("attention_output"),
        1.0e-6,
        1.0e-5,
    );
}

#[test]
fn router_matches_oracle_and_has_deterministic_ties() {
    let config = fixture_config();
    let weights = layer_zero_weights();
    let hidden = checkpoint_matrix("post_attention_norm");
    let actual = route_tokens(hidden.view(), weights.router.view(), config).expect("router");

    assert_stage(
        "layer_0.router_logits",
        &actual.logits,
        &checkpoint_matrix("router_logits"),
        1.0e-7,
        1.0e-6,
    );
    assert_stage(
        "layer_0.routing_weights",
        &actual.weights,
        &checkpoint_matrix("routing_weights"),
        1.0e-6,
        1.0e-5,
    );
    assert_eq!(actual.selected_experts, selected_experts());

    let zero_hidden =
        Tensor::new(TensorShape::new([1, 16]), vec![0.0; 16]).expect("zero hidden state");
    let zero_router = Tensor::new(TensorShape::new([4, 16]), vec![0.0; 64]).expect("zero router");
    let tied = route_tokens(zero_hidden.view(), zero_router.view(), config).expect("tied router");
    assert_eq!(tied.selected_experts, [0, 1]);
    assert_eq!(tied.weights.data(), [0.25, 0.25]);
}

#[test]
fn router_can_renormalize_selected_probabilities() {
    let config = config_with_theta_and_normalization(frozen_config::ROPE_THETA, true);
    let hidden = Tensor::new(TensorShape::new([1, 16]), vec![0.0; 16]).expect("zero hidden");
    let router = Tensor::new(TensorShape::new([4, 16]), vec![0.0; 64]).expect("zero router");

    let output = route_tokens(hidden.view(), router.view(), config).expect("normalized router");

    assert_eq!(output.selected_experts, [0, 1]);
    assert_eq!(output.weights.data(), [0.5, 0.5]);
}

fn route_explicit_logits(logits: [f32; 4]) -> RouterOutput {
    let config = fixture_config();
    let mut hidden = vec![0.0; 16];
    hidden[0] = 1.0;
    let hidden = Tensor::new(TensorShape::new([1, 16]), hidden).expect("hidden");
    let mut weights = vec![0.0; 4 * 16];
    for (expert, logit) in logits.into_iter().enumerate() {
        weights[expert * 16] = logit;
    }
    let router = Tensor::new(TensorShape::new([4, 16]), weights).expect("router");
    route_tokens(hidden.view(), router.view(), config).expect("route explicit logits")
}

#[test]
fn router_orders_strict_scores_from_highest_to_lowest() {
    assert_eq!(
        route_explicit_logits([1.0, 4.0, 2.0, 3.0]).selected_experts,
        [1, 3]
    );
}

#[test]
fn router_orders_all_equal_scores_by_lower_expert_id() {
    assert_eq!(route_explicit_logits([2.0; 4]).selected_experts, [0, 1]);
}

#[test]
fn router_orders_ties_inside_selected_set_by_lower_expert_id() {
    assert_eq!(
        route_explicit_logits([4.0, 4.0, 2.0, 1.0]).selected_experts,
        [0, 1]
    );
}

#[test]
fn router_orders_top_k_boundary_ties_by_lower_expert_id() {
    assert_eq!(
        route_explicit_logits([4.0, 3.0, 3.0, 1.0]).selected_experts,
        [0, 1]
    );
}

#[test]
fn router_tie_order_is_repeatable() {
    let expected = route_explicit_logits([4.0, 3.0, 3.0, 1.0]).selected_experts;
    for _ in 0..32 {
        assert_eq!(
            route_explicit_logits([4.0, 3.0, 3.0, 1.0]).selected_experts,
            expected
        );
    }
}

#[test]
fn gated_experts_match_frozen_oracle() {
    let config = fixture_config();
    let weights = layer_zero_weights();
    let router = RouterOutput {
        logits: checkpoint_matrix("router_logits"),
        weights: checkpoint_matrix("routing_weights"),
        selected_experts: selected_experts(),
    };
    let actual = routed_experts(
        checkpoint_matrix("post_attention_norm").view(),
        weights.expert_gate_up.view(),
        weights.expert_down.view(),
        &router,
        config,
    )
    .expect("routed experts succeed");

    assert_stage(
        "layer_0.moe_output",
        &actual,
        &checkpoint_matrix("moe_output"),
        1.0e-6,
        1.0e-5,
    );
}

#[test]
fn full_sparse_block_matches_every_frozen_stage() {
    let block =
        Qwen3MoeBlock::new(fixture_config(), layer_zero_weights()).expect("valid frozen block");
    let output = block
        .forward(checkpoint_matrix("hidden_state_0").view())
        .expect("block forward succeeds");

    let stages = [
        ("layer_0.input_norm", &output.input_norm, "input_norm"),
        (
            "layer_0.attention_output",
            &output.attention_output,
            "attention_output",
        ),
        (
            "layer_0.post_attention_norm",
            &output.post_attention_norm,
            "post_attention_norm",
        ),
        (
            "layer_0.router_logits",
            &output.router_logits,
            "router_logits",
        ),
        (
            "layer_0.routing_weights",
            &output.routing_weights,
            "routing_weights",
        ),
        ("layer_0.moe_output", &output.moe_output, "moe_output"),
        ("layer_0.block_output", &output.block_output, "block_output"),
    ];
    for (stage, actual, checkpoint_name) in stages {
        let (absolute, relative) = if stage.ends_with("router_logits") {
            (1.0e-7, 1.0e-6)
        } else {
            (1.0e-6, 1.0e-5)
        };
        assert_stage(
            stage,
            actual,
            &checkpoint_matrix(checkpoint_name),
            absolute,
            relative,
        );
    }
    assert_eq!(output.selected_experts, selected_experts());
}

#[test]
fn diagnostics_name_the_first_mismatching_stage() {
    let expected = Tensor::new(TensorShape::new([1]), vec![1.0]).expect("expected");
    let matching = expected.clone();
    let mismatching = Tensor::new(TensorShape::new([1]), vec![2.0]).expect("mismatch");
    let stages = [
        ("input_norm", &matching),
        ("attention", &mismatching),
        ("moe", &mismatching),
    ];

    let first = stages
        .into_iter()
        .find_map(|(stage, actual)| compare_stage(stage, actual, &expected, 0.0, 0.0).err())
        .expect("one stage must mismatch");

    assert_eq!(first.stage(), "attention");
}
