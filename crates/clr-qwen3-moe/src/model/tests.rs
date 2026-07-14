use super::*;
use crate::test_fixture;

fn fixture_model() -> Qwen3MoeModel {
    Qwen3MoeModel::new(
        test_fixture::config(),
        Qwen3MoeModelWeightsSpec {
            token_embeddings: test_fixture::token_embeddings(),
            blocks: vec![
                test_fixture::block_weights(0),
                test_fixture::block_weights(1),
            ],
            final_norm: test_fixture::final_norm_weight(),
            language_model_head: test_fixture::language_model_head(),
        },
    )
    .expect("valid frozen model")
}

fn assert_stage(stage: &str, actual: &Tensor, expected: &Tensor) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch at {stage}"
    );
    for (index, (actual_value, expected_value)) in
        actual.data().iter().zip(expected.data()).enumerate()
    {
        let tolerance = 1.0e-6 + 1.0e-5 * expected_value.abs();
        assert!(
            (actual_value - expected_value).abs() <= tolerance,
            "first oracle mismatch at {stage}[{index}]: expected {expected_value}, got {actual_value}, tolerance {tolerance}"
        );
    }
}

#[test]
fn embedding_lookup_matches_frozen_oracle() {
    let actual = embedding_lookup(
        test_fixture::token_embeddings().view(),
        &test_fixture::token_ids(),
        test_fixture::frozen_config::VOCABULARY_SIZE,
    )
    .expect("embedding lookup succeeds");

    assert_stage("hidden_state_0", &actual, &test_fixture::hidden_state(0));
}

#[test]
fn embedding_lookup_rejects_empty_and_out_of_range_ids() {
    let embeddings = test_fixture::token_embeddings();

    assert_eq!(
        embedding_lookup(
            embeddings.view(),
            &[],
            test_fixture::frozen_config::VOCABULARY_SIZE,
        ),
        Err(RuntimeError::InvalidShape {
            reason: "token ID sequence must not be empty",
        })
    );
    assert_eq!(
        embedding_lookup(
            embeddings.view(),
            &[test_fixture::frozen_config::VOCABULARY_SIZE],
            test_fixture::frozen_config::VOCABULARY_SIZE,
        ),
        Err(RuntimeError::IndexOutOfBounds {
            index: test_fixture::frozen_config::VOCABULARY_SIZE,
            length: test_fixture::frozen_config::VOCABULARY_SIZE,
        })
    );
}

#[test]
fn complete_tiny_decoder_matches_hidden_states_experts_and_logits() {
    let model = fixture_model();
    let output = model
        .forward(&test_fixture::token_ids())
        .expect("model forward succeeds");

    assert_eq!(output.hidden_states.len(), 3);
    assert_eq!(output.block_outputs.len(), 2);
    for (layer, block_output) in output.block_outputs.iter().enumerate() {
        let stages = [
            ("input_norm", &block_output.input_norm),
            ("attention_output", &block_output.attention_output),
            ("post_attention_norm", &block_output.post_attention_norm),
            ("router_logits", &block_output.router_logits),
            ("routing_weights", &block_output.routing_weights),
            ("moe_output", &block_output.moe_output),
            ("block_output", &block_output.block_output),
        ];
        for (stage, actual) in stages {
            let qualified_stage = match (layer, stage) {
                (0, "input_norm") => "layer_0.input_norm",
                (0, "attention_output") => "layer_0.attention_output",
                (0, "post_attention_norm") => "layer_0.post_attention_norm",
                (0, "router_logits") => "layer_0.router_logits",
                (0, "routing_weights") => "layer_0.routing_weights",
                (0, "moe_output") => "layer_0.moe_output",
                (0, "block_output") => "layer_0.block_output",
                (1, "input_norm") => "layer_1.input_norm",
                (1, "attention_output") => "layer_1.attention_output",
                (1, "post_attention_norm") => "layer_1.post_attention_norm",
                (1, "router_logits") => "layer_1.router_logits",
                (1, "routing_weights") => "layer_1.routing_weights",
                (1, "moe_output") => "layer_1.moe_output",
                (1, "block_output") => "layer_1.block_output",
                _ => unreachable!(),
            };
            assert_stage(
                qualified_stage,
                actual,
                &test_fixture::block_checkpoint(layer, stage),
            );
        }
    }
    for (index, hidden_state) in output.hidden_states.iter().enumerate() {
        let (stage, expected) = match index {
            0 => ("hidden_state_0", test_fixture::hidden_state(0)),
            1 => ("hidden_state_1", test_fixture::hidden_state(1)),
            2 => (
                "layer_1.block_output",
                test_fixture::block_checkpoint(1, "block_output"),
            ),
            _ => unreachable!(),
        };
        assert_stage(stage, hidden_state, &expected);
    }
    for (layer, block_output) in output.block_outputs.iter().enumerate() {
        assert_eq!(
            block_output.selected_experts,
            test_fixture::selected_experts(layer),
            "expert IDs differ at layer {layer}"
        );
    }
    assert_stage(
        "final_norm",
        &output.final_norm,
        &test_fixture::final_norm_checkpoint(),
    );
    assert_stage(
        "hidden_state_2",
        &output.final_norm,
        &test_fixture::hidden_state(2),
    );
    assert_stage(
        "final_logits",
        &output.logits,
        &test_fixture::final_logits(),
    );
}

#[test]
fn repeated_full_model_runs_are_identical() {
    let model = fixture_model();
    let token_ids = test_fixture::token_ids();

    let first = model.forward(&token_ids).expect("first run succeeds");
    let second = model.forward(&token_ids).expect("second run succeeds");

    assert_eq!(first, second);
}

#[test]
fn model_rejects_wrong_layer_count() {
    let error = Qwen3MoeModel::new(
        test_fixture::config(),
        Qwen3MoeModelWeightsSpec {
            token_embeddings: test_fixture::token_embeddings(),
            blocks: vec![test_fixture::block_weights(0)],
            final_norm: test_fixture::final_norm_weight(),
            language_model_head: test_fixture::language_model_head(),
        },
    );

    assert!(matches!(
        error,
        Err(RuntimeError::TensorDataLengthMismatch {
            expected: 2,
            actual: 1,
        })
    ));
}
