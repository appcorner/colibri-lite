use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use clr_storage::{
    ArtifactReader, DEFAULT_CONVERSION_CHUNK_BYTES, DenseSourceShard, ExpertStore, Sha256Hasher,
};

use super::*;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
const SOURCE_PROJECTION_BYTES: u64 = 768 * 2_048 * 2;
const ARTIFACT_PROJECTION_BYTES: u64 = SOURCE_PROJECTION_BYTES * 2;
const EXPERT_PAYLOAD_BYTES: u64 = ARTIFACT_PROJECTION_BYTES * 3;

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "colibri-expert-converter-{label}-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create test directory");
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

fn projection_name(layer: usize, expert: usize, projection: ExpertProjectionKind) -> String {
    let projection = match projection {
        ExpertProjectionKind::Gate => "gate_proj",
        ExpertProjectionKind::Up => "up_proj",
        ExpertProjectionKind::Down => "down_proj",
    };
    format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight")
}

fn source_projection(
    layer: usize,
    expert: usize,
    projection: ExpertProjectionKind,
    offset: u64,
) -> Qwen3MoeExpertSourceProjection {
    let shape = match projection {
        ExpertProjectionKind::Gate | ExpertProjectionKind::Up => [768, 2_048],
        ExpertProjectionKind::Down => [2_048, 768],
    };
    Qwen3MoeExpertSourceProjection {
        layer_index: layer,
        expert_index: expert,
        projection,
        metadata: Qwen3MoeTensorMetadata::new(
            projection_name(layer, expert, projection),
            DataType::BF16,
            TensorShape::new(shape),
            0,
        ),
        offset,
        length: SOURCE_PROJECTION_BYTES,
    }
}

fn metadata_only_spec() -> Qwen3MoeExpertConversionSpec {
    let mut projections = Vec::new();
    let mut offset = 0;
    for (layer, expert) in [(0, 0), (0, 127), (47, 127)] {
        for kind in [
            ExpertProjectionKind::Gate,
            ExpertProjectionKind::Up,
            ExpertProjectionKind::Down,
        ] {
            projections.push(source_projection(layer, expert, kind, offset));
            offset += SOURCE_PROJECTION_BYTES;
        }
    }
    Qwen3MoeExpertConversionSpec {
        scope: Qwen3MoeExpertConversionScope::VerticalSlice,
        source_shards: vec![DenseSourceShard {
            path: "unused.safetensors".into(),
            byte_length: offset,
            sha256: [0; 32],
        }],
        projections,
        output_directory: "unused-output".into(),
        available_space_bytes: u64::MAX,
        source_chunk_bytes: DEFAULT_CONVERSION_CHUNK_BYTES,
    }
}

fn real_shape_fixture(root: &std::path::Path) -> Qwen3MoeExpertConversionSpec {
    let mut spec = metadata_only_spec();
    let source_path = root.join("source.safetensors");
    let file = File::create(&source_path).expect("create source shard");
    let mut writer = BufWriter::new(file);
    let mut hasher = Sha256Hasher::new();
    let mut chunk = vec![0_u8; DEFAULT_CONVERSION_CHUNK_BYTES];
    for (projection_index, projection) in spec.projections.iter().enumerate() {
        let word = 0x3f00_u16 + u16::try_from(projection_index).unwrap();
        for bytes in chunk.chunks_exact_mut(2) {
            bytes.copy_from_slice(&word.to_le_bytes());
        }
        let mut remaining = projection.length;
        while remaining != 0 {
            let length = usize::try_from(remaining.min(chunk.len() as u64)).unwrap();
            writer
                .write_all(&chunk[..length])
                .expect("write source bytes");
            hasher.update(&chunk[..length]);
            remaining -= length as u64;
        }
    }
    writer.flush().expect("flush source shard");
    writer.get_ref().sync_all().expect("sync source shard");
    drop(writer);
    let byte_length = fs::metadata(&source_path).unwrap().len();
    spec.source_shards[0] = DenseSourceShard {
        path: source_path,
        byte_length,
        sha256: hasher.finalize(),
    };
    spec.output_directory = root.join("artifact");
    spec.available_space_bytes = 1_000_000_000;
    spec
}

#[test]
fn validates_projection_order_identity_shape_inventory_and_duplicates() {
    let valid = metadata_only_spec();

    let mut wrong_order = valid.clone();
    wrong_order.projections.swap(0, 1);
    assert!(matches!(
        validate_spec(&wrong_order),
        Err(Qwen3MoeExpertConversionError::InvalidProjectionOrder { .. })
    ));

    let mut wrong_identity = valid.clone();
    wrong_identity.projections[0].expert_index = 1;
    assert!(matches!(
        validate_spec(&wrong_identity),
        Err(Qwen3MoeExpertConversionError::ProjectionIdentityMismatch { .. })
    ));

    let mut wrong_layer = valid.clone();
    wrong_layer.projections[0].layer_index = 1;
    assert!(matches!(
        validate_spec(&wrong_layer),
        Err(Qwen3MoeExpertConversionError::ProjectionIdentityMismatch { .. })
    ));

    let mut wrong_shape = valid.clone();
    wrong_shape.projections[0].metadata = Qwen3MoeTensorMetadata::new(
        projection_name(0, 0, ExpertProjectionKind::Gate),
        DataType::BF16,
        TensorShape::new([769, 2_048]),
        0,
    );
    assert!(matches!(
        validate_spec(&wrong_shape),
        Err(Qwen3MoeExpertConversionError::Inventory(
            Qwen3MoeTensorInventoryError::ShapeMismatch { .. }
        ))
    ));

    let mut wrong_shard = valid.clone();
    wrong_shard.projections[0].metadata = Qwen3MoeTensorMetadata::new(
        projection_name(0, 0, ExpertProjectionKind::Gate),
        DataType::BF16,
        TensorShape::new([768, 2_048]),
        1,
    );
    assert!(matches!(
        validate_spec(&wrong_shard),
        Err(Qwen3MoeExpertConversionError::Inventory(
            Qwen3MoeTensorInventoryError::ShardIndexOutOfRange { .. }
        ))
    ));

    let mut invalid_offset = valid.clone();
    invalid_offset.projections[0].offset = u64::MAX;
    assert!(matches!(
        validate_spec(&invalid_offset),
        Err(Qwen3MoeExpertConversionError::InvalidSourceRange { .. })
    ));

    let mut duplicate = valid.clone();
    duplicate
        .projections
        .splice(3..6, valid.projections[0..3].iter().cloned());
    assert!(matches!(
        validate_spec(&duplicate),
        Err(Qwen3MoeExpertConversionError::DuplicateExpert { .. })
    ));

    let mut incomplete = valid.clone();
    incomplete.scope = Qwen3MoeExpertConversionScope::Complete;
    assert!(matches!(
        validate_spec(&incomplete),
        Err(Qwen3MoeExpertConversionError::IncompleteExpertInventory { .. })
    ));
}

#[test]
fn vertical_slice_requires_two_layers_and_complete_projection_group() {
    let mut one_layer = metadata_only_spec();
    one_layer.projections.truncate(6);
    assert!(matches!(
        validate_spec(&one_layer),
        Err(Qwen3MoeExpertConversionError::InsufficientVerticalSliceLayers { actual: 1 })
    ));

    let mut incomplete = metadata_only_spec();
    incomplete.projections.pop();
    assert!(matches!(
        validate_spec(&incomplete),
        Err(Qwen3MoeExpertConversionError::IncompleteExpertInventory { .. })
    ));

    let mut invalid_chunk = metadata_only_spec();
    invalid_chunk.source_chunk_bytes = 0;
    assert!(matches!(
        validate_spec(&invalid_chunk),
        Err(Qwen3MoeExpertConversionError::InvalidChunkSize { actual: 0 })
    ));

    let mut insufficient_space = metadata_only_spec();
    insufficient_space.available_space_bytes = 0;
    assert!(matches!(
        convert_pinned_qwen3_moe_experts(&insufficient_space),
        Err(Qwen3MoeExpertConversionError::InsufficientDiskSpace { available: 0, .. })
    ));
}

#[test]
fn real_shape_slice_round_trips_repeats_and_reads_one_logical_payload() {
    let directory = TestDirectory::new("round-trip");
    let first_spec = real_shape_fixture(&directory.0);
    let mut second_spec = first_spec.clone();
    second_spec.output_directory = directory.0.join("artifact-repeat");

    let first = convert_pinned_qwen3_moe_experts(&first_spec).expect("first conversion");
    let second = convert_pinned_qwen3_moe_experts(&second_spec).expect("repeat conversion");

    assert_eq!(first.manifest_sha256, second.manifest_sha256);
    assert_eq!(first.manifest.shards(), second.manifest.shards());
    assert_eq!(first.manifest.experts(), second.manifest.experts());
    assert_eq!(first.artifact_bytes_written, EXPERT_PAYLOAD_BYTES * 3);
    assert_eq!(first.peak_buffer_bytes, DEFAULT_CONVERSION_CHUNK_BYTES * 5);
    let records = first.manifest.experts();
    assert_eq!(
        (records[0].key.layer_index, records[0].key.expert_id.0),
        (0, 0)
    );
    assert_eq!(records[0].payload_offset, 0);
    assert_eq!(records[1].payload_offset, EXPERT_PAYLOAD_BYTES);
    assert_eq!(records[2].payload_offset, 0);
    assert_eq!(records[0].gate.offset, 0);
    assert_eq!(records[0].up.offset, ARTIFACT_PROJECTION_BYTES);
    assert_eq!(records[0].down.offset, ARTIFACT_PROJECTION_BYTES * 2);
    assert_eq!(records[0].payload_length, EXPERT_PAYLOAD_BYTES);

    let reader = ArtifactReader::open(
        &first_spec.output_directory,
        first.manifest.runtime_manifest().clone(),
    )
    .expect("open runtime artifact");
    let mut store = ExpertStore::new(
        reader,
        first.manifest.registrations().to_vec(),
        usize::try_from(EXPERT_PAYLOAD_BYTES).unwrap(),
    )
    .expect("create existing expert store");
    let key = expert_key(0, 127);
    let lease = store.load(key).expect("load one logical expert");
    assert_eq!(
        lease.bytes().len(),
        usize::try_from(EXPERT_PAYLOAD_BYTES).unwrap()
    );
    assert_eq!(store.metrics().bytes_read, EXPERT_PAYLOAD_BYTES);
    assert_eq!(
        fs::metadata(first_spec.output_directory.join(shard_path(0)))
            .unwrap()
            .len(),
        EXPERT_PAYLOAD_BYTES * 2
    );
}

#[test]
fn rejects_corruption_truncation_and_cleans_incomplete_shards() {
    let directory = TestDirectory::new("failures");
    let fixture = real_shape_fixture(&directory.0);

    let mut corruption = fixture.clone();
    corruption.output_directory = directory.0.join("corrupt-output");
    corruption.source_shards[0].sha256 = [0; 32];
    assert!(matches!(
        convert_pinned_qwen3_moe_experts(&corruption),
        Err(Qwen3MoeExpertConversionError::SourceShardHashMismatch { .. })
    ));
    assert!(!corruption.output_directory.exists());

    let mut truncation = fixture.clone();
    truncation.output_directory = directory.0.join("truncated-output");
    truncation.source_shards[0].byte_length += 1;
    assert!(matches!(
        convert_pinned_qwen3_moe_experts(&truncation),
        Err(Qwen3MoeExpertConversionError::SourceShardLengthMismatch { .. })
    ));
    assert!(!truncation.output_directory.exists());

    let mut incomplete = fixture;
    incomplete.output_directory = directory.0.join("incomplete-output");
    assert!(matches!(
        io::convert(&incomplete, Some(0)),
        Err(Qwen3MoeExpertConversionError::InjectedIncompleteOutput)
    ));
    assert!(!incomplete.output_directory.exists());
}
