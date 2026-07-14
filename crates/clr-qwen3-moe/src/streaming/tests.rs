use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use clr_core::{DataType, Tensor, TensorShape};
use clr_storage::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ArtifactReader, ByteOrder, ExpertId,
    ExpertRegistration, TensorLocation, TensorMetadata, sha256_digest,
};

use super::*;
use crate::{GenerationSession, Qwen3MoeModel, Qwen3MoeModelWeightsSpec, test_fixture};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    resident: Qwen3MoeModel,
    streaming: StreamingQwen3MoeModel,
    store: ExpertStore,
    payloads: Vec<(ExpertKey, Vec<u8>)>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove streaming fixture directory");
    }
}

fn fixture(budget: usize) -> Fixture {
    let config = test_fixture::config();
    let resident_blocks = vec![
        test_fixture::block_weights(0),
        test_fixture::block_weights(1),
    ];
    let resident = Qwen3MoeModel::new(
        config,
        Qwen3MoeModelWeightsSpec {
            token_embeddings: test_fixture::token_embeddings(),
            blocks: resident_blocks.clone(),
            final_norm: test_fixture::final_norm_weight(),
            language_model_head: test_fixture::language_model_head(),
        },
    )
    .expect("resident fixture model");
    let streaming_blocks = resident_blocks.iter().map(dense_weights).collect();
    let streaming = StreamingQwen3MoeModel::new(
        config,
        StreamingModelWeightsSpec {
            token_embeddings: test_fixture::token_embeddings(),
            blocks: streaming_blocks,
            final_norm: test_fixture::final_norm_weight(),
            language_model_head: test_fixture::language_model_head(),
        },
    );

    let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "colibri-streaming-test-{}-{id}",
        std::process::id()
    ));
    fs::create_dir(&root).expect("create streaming fixture directory");
    let mut shard = Vec::new();
    let mut metadata = Vec::new();
    let mut registrations = Vec::new();
    let mut payloads = Vec::new();
    for (layer, block) in resident_blocks.iter().enumerate() {
        for expert in 0..config.expert_count() {
            let key = ExpertKey {
                layer_index: u32::try_from(layer).expect("layer index"),
                expert_id: ExpertId(u32::try_from(expert).expect("expert ID")),
            };
            let payload = pack_expert(block, expert, config);
            let offset = u64::try_from(shard.len()).expect("shard offset");
            shard.extend_from_slice(&payload);
            let name = format!("layer.{layer}.expert.{expert}");
            metadata.push(TensorMetadata {
                name: name.clone(),
                shape: TensorShape::new([payload.len() / 4]),
                data_type: DataType::F32,
                location: TensorLocation {
                    path: "experts.shard".into(),
                    offset,
                    length: u64::try_from(payload.len()).expect("payload length"),
                },
                sha256: sha256_digest(&payload),
            });
            registrations.push(ExpertRegistration {
                key,
                tensor_name: name,
            });
            payloads.push((key, payload));
        }
    }
    fs::write(root.join("experts.shard"), shard).expect("write expert shard");
    let manifest = ArtifactManifest::new(ARTIFACT_FORMAT_VERSION, ByteOrder::Little, metadata)
        .expect("streaming manifest");
    let reader = ArtifactReader::open(&root, manifest).expect("streaming reader");
    let store = ExpertStore::new(reader, registrations, budget).expect("streaming store");
    Fixture {
        root,
        resident,
        streaming,
        store,
        payloads,
    }
}

fn dense_weights(block: &crate::Qwen3MoeBlockWeightsSpec) -> StreamingBlockWeightsSpec {
    StreamingBlockWeightsSpec {
        input_norm: block.input_norm.clone(),
        query_projection: block.query_projection.clone(),
        key_projection: block.key_projection.clone(),
        value_projection: block.value_projection.clone(),
        output_projection: block.output_projection.clone(),
        query_norm: block.query_norm.clone(),
        key_norm: block.key_norm.clone(),
        post_attention_norm: block.post_attention_norm.clone(),
        router: block.router.clone(),
    }
}

fn pack_expert(
    block: &crate::Qwen3MoeBlockWeightsSpec,
    expert: usize,
    config: Qwen3MoeConfig,
) -> Vec<u8> {
    let hidden = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    let matrix_values = hidden * intermediate;
    let gate_up_start = expert * 2 * matrix_values;
    let down_start = expert * matrix_values;
    let values = block.expert_gate_up.data()[gate_up_start..gate_up_start + 2 * matrix_values]
        .iter()
        .chain(&block.expert_down.data()[down_start..down_start + matrix_values]);
    values.flat_map(|value| value.to_le_bytes()).collect()
}

fn assert_tensor(stage: &str, actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape(), "shape at {stage}");
    for (index, (actual, expected)) in actual.data().iter().zip(expected.data()).enumerate() {
        let tolerance = 1.0e-6 + 1.0e-5 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "streaming mismatch at {stage}[{index}]: {actual} vs {expected}"
        );
    }
}

#[test]
fn packed_payload_round_trips_every_f32_exactly() {
    let config = test_fixture::config();
    let mut fixture = fixture(8 * PackedExpertLayout::for_config(config).total_byte_length);
    for (key, expected) in &fixture.payloads {
        let lease = fixture.store.load(*key).expect("load packed expert");
        assert_eq!(lease.bytes(), expected);
        for (actual, expected) in lease.bytes().chunks_exact(4).zip(expected.chunks_exact(4)) {
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn streaming_matches_resident_with_forced_eviction() {
    let config = test_fixture::config();
    let payload = PackedExpertLayout::for_config(config).total_byte_length;
    let mut fixture = fixture(2 * payload);
    let token_ids = test_fixture::token_ids();
    let resident = fixture
        .resident
        .forward(&token_ids)
        .expect("resident forward");
    let streaming = fixture
        .streaming
        .forward(&token_ids, &mut fixture.store)
        .expect("streaming forward");

    for (layer, (actual, expected)) in streaming
        .block_outputs
        .iter()
        .zip(&resident.block_outputs)
        .enumerate()
    {
        assert_eq!(actual.selected_experts, expected.selected_experts);
        assert_tensor("input norm", &actual.input_norm, &expected.input_norm);
        assert_tensor(
            "attention output",
            &actual.attention_output,
            &expected.attention_output,
        );
        assert_tensor(
            "post attention norm",
            &actual.post_attention_norm,
            &expected.post_attention_norm,
        );
        assert_tensor(
            "router logits",
            &actual.router_logits,
            &expected.router_logits,
        );
        assert_tensor(
            "routing weights",
            &actual.routing_weights,
            &expected.routing_weights,
        );
        assert_tensor("moe output", &actual.moe_output, &expected.moe_output);
        assert_tensor("block output", &actual.block_output, &expected.block_output);
        assert_eq!(
            actual.selected_experts,
            test_fixture::selected_experts(layer)
        );
    }
    assert_tensor("final logits", &streaming.logits, &resident.logits);
    let metrics = fixture.store.metrics();
    assert_eq!(metrics.loads, 8);
    assert_eq!(metrics.misses, 8);
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.evictions, 6);
    assert_eq!(metrics.resident_bytes, 2 * payload);
    assert_eq!(metrics.peak_resident_bytes, 2 * payload);
    assert_eq!(
        metrics.bytes_read,
        u64::try_from(8 * payload).expect("bytes read")
    );
}

#[test]
fn streaming_prefill_matches_resident_logits_and_experts() {
    let config = test_fixture::config();
    let payload = PackedExpertLayout::for_config(config).total_byte_length;
    let mut fixture = fixture(2 * payload);
    let token_ids = test_fixture::token_ids();
    let expected = fixture
        .resident
        .forward(&token_ids)
        .expect("resident forward");
    let mut session = GenerationSession::streaming(
        &fixture.streaming,
        &mut fixture.store,
        config.model().max_sequence_length(),
        0,
    )
    .expect("streaming session");

    let actual = session.prefill(&token_ids).expect("streaming prefill");

    assert_tensor("prefill logits", &actual.logits, &expected.logits);
    assert_eq!(
        actual.selected_experts,
        expected
            .block_outputs
            .iter()
            .map(|block| block.selected_experts.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(session.cache().len(), token_ids.len());
    assert_eq!(session.sequence(), token_ids);
}

#[test]
fn streaming_cached_decode_matches_resident_cached_decode() {
    let config = test_fixture::config();
    let payload = PackedExpertLayout::for_config(config).total_byte_length;
    let mut fixture = fixture(2 * payload);
    let token_ids = test_fixture::token_ids();
    let capacity = config.model().max_sequence_length();
    let mut resident =
        GenerationSession::resident(&fixture.resident, capacity, 42).expect("resident session");
    let mut streaming =
        GenerationSession::streaming(&fixture.streaming, &mut fixture.store, capacity, 42)
            .expect("streaming session");
    resident.prefill(&token_ids).expect("resident prefill");
    streaming.prefill(&token_ids).expect("streaming prefill");

    for _ in 0..4 {
        assert_eq!(
            streaming.decode_greedy().expect("streaming decode"),
            resident.decode_greedy().expect("resident decode")
        );
    }
    assert_eq!(streaming.sequence(), resident.sequence());
    assert_eq!(streaming.cache().len(), resident.cache().len());
}

#[test]
fn oversize_and_corrupt_payloads_fail_before_computation() {
    let config = test_fixture::config();
    let payload = PackedExpertLayout::for_config(config).total_byte_length;
    let mut oversize = fixture(payload - 1);
    assert!(matches!(
        oversize
            .streaming
            .forward(&test_fixture::token_ids(), &mut oversize.store),
        Err(StreamingModelError::Storage(
            StorageError::ExpertExceedsBudget { .. }
        ))
    ));

    let mut corrupt = fixture(2 * payload);
    let path = corrupt.root.join("experts.shard");
    let mut bytes = fs::read(&path).expect("read shard");
    bytes[0] ^= 0xff;
    fs::write(&path, bytes).expect("corrupt shard");
    assert!(matches!(
        corrupt
            .streaming
            .forward(&test_fixture::token_ids(), &mut corrupt.store),
        Err(StreamingModelError::Storage(
            StorageError::HashMismatch { .. }
        ))
    ));

    let mut truncated = fixture(2 * payload);
    fs::write(truncated.root.join("experts.shard"), [0_u8; 4]).expect("truncate shard");
    assert!(matches!(
        truncated
            .streaming
            .forward(&test_fixture::token_ids(), &mut truncated.store),
        Err(StreamingModelError::Storage(
            StorageError::TruncatedTensor { .. }
        ))
    ));
}
