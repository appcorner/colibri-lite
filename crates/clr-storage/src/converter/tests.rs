use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::*;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "colibri-dense-converter-{label}-{}-{id}",
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

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn fixture(root: &Path) -> DenseConversionSpec {
    let first = bf16_bytes(&[0x3f80, 0x4000, 0x4040, 0x4080]);
    let second = bf16_bytes(&[0x0000, 0x8000, 0x7f80]);
    let mut shard = vec![0x55; 7];
    let first_offset = shard.len() as u64;
    shard.extend_from_slice(&first);
    shard.extend_from_slice(&[0xaa; 5]);
    let second_offset = shard.len() as u64;
    shard.extend_from_slice(&second);
    let shard_path = root.join("source.safetensors");
    fs::write(&shard_path, &shard).expect("write source fixture");

    DenseConversionSpec {
        model_id: "test/model".to_owned(),
        model_revision: "0123456789abcdef".to_owned(),
        source_shards: vec![DenseSourceShard {
            path: shard_path,
            byte_length: shard.len() as u64,
            sha256: hash_bytes(&shard),
        }],
        tensors: vec![
            DenseSourceTensor {
                name: "z.second".to_owned(),
                shape: TensorShape::new([3]),
                data_type: DataType::BF16,
                shard_index: 0,
                offset: second_offset,
                length: second.len() as u64,
            },
            DenseSourceTensor {
                name: "a.first".to_owned(),
                shape: TensorShape::new([4]),
                data_type: DataType::BF16,
                shard_index: 0,
                offset: first_offset,
                length: first.len() as u64,
            },
        ],
        output_directory: root.join("output"),
        available_space_bytes: 1_000_000,
        source_chunk_bytes: 4,
    }
}

#[test]
fn bf16_decode_preserves_normal_zero_subnormal_infinity_and_nan_bits() {
    let cases = [
        (0x3f80, 0x3f80_0000),
        (0xc020, 0xc020_0000),
        (0x0000, 0x0000_0000),
        (0x8000, 0x8000_0000),
        (0x0001, 0x0001_0000),
        (0x8001, 0x8001_0000),
        (0x7f80, 0x7f80_0000),
        (0xff80, 0xff80_0000),
        (0x7fc1, 0x7fc1_0000),
        (0xffc1, 0xffc1_0000),
    ];

    for (source, expected) in cases {
        assert_eq!(decode_bf16(source).to_bits(), expected);
    }
}

#[test]
fn real_pinned_norm_samples_decode_to_exact_f32_bits() {
    let samples = [
        (0x401d, 0x401d_0000),
        (0x402b, 0x402b_0000),
        (0x4037, 0x4037_0000),
        (0x3f7c, 0x3f7c_0000),
        (0x3f94, 0x3f94_0000),
        (0x3f1d, 0x3f1d_0000),
        (0x3fe3, 0x3fe3_0000),
        (0x3ffb, 0x3ffb_0000),
        (0x3ff1, 0x3ff1_0000),
    ];

    for (source, expected) in samples {
        assert_eq!(decode_bf16(source).to_bits(), expected);
    }
}

#[test]
fn chunked_conversion_round_trips_and_is_reproducible() {
    let directory = TestDirectory::new("reproducible");
    let first_spec = fixture(&directory.0);
    let mut second_spec = first_spec.clone();
    second_spec.output_directory = directory.0.join("second-output");

    let first = convert_dense_bf16_to_f32(&first_spec).expect("first conversion");
    let second = convert_dense_bf16_to_f32(&second_spec).expect("second conversion");

    assert_eq!(first.payload_sha256, second.payload_sha256);
    assert_eq!(first.manifest_sha256, second.manifest_sha256);
    assert_eq!(
        fs::read(&first.payload_path).unwrap(),
        fs::read(&second.payload_path).unwrap()
    );
    assert_eq!(
        fs::read(&first.manifest_path).unwrap(),
        fs::read(&second.manifest_path).unwrap()
    );
    assert_eq!(first.artifact_manifest.tensors()[0].name, "a.first");
    assert_eq!(first.artifact_manifest.tensors()[1].name, "z.second");
    assert_eq!(first.artifact_bytes_written, 28);
    assert_eq!(first.source_payload_bytes_read, 28);
    assert_eq!(first.peak_buffer_bytes, 20);
    assert!(
        fs::read_to_string(first.manifest_path)
            .unwrap()
            .contains("\"model_revision\": \"0123456789abcdef\"")
    );
}

#[test]
fn rejects_wrong_hash_and_truncated_shard_before_output() {
    let directory = TestDirectory::new("integrity");
    let mut wrong_hash = fixture(&directory.0);
    wrong_hash.source_shards[0].sha256 = [0; 32];
    assert!(matches!(
        convert_dense_bf16_to_f32(&wrong_hash),
        Err(DenseConversionError::SourceShardHashMismatch { .. })
    ));
    assert!(!wrong_hash.output_directory.exists());

    let mut truncated = fixture(&directory.0);
    truncated.source_shards[0].byte_length += 1;
    assert!(matches!(
        convert_dense_bf16_to_f32(&truncated),
        Err(DenseConversionError::SourceShardLengthMismatch { .. })
    ));
    assert!(!truncated.output_directory.exists());
}

#[test]
fn rejects_wrong_dtype_shape_and_source_range() {
    let directory = TestDirectory::new("metadata");
    let mut wrong_dtype = fixture(&directory.0);
    wrong_dtype.tensors[0].data_type = DataType::F32;
    assert!(matches!(
        convert_dense_bf16_to_f32(&wrong_dtype),
        Err(DenseConversionError::SourceDataTypeMismatch { .. })
    ));

    let mut wrong_shape = fixture(&directory.0);
    wrong_shape.tensors[0].shape = TensorShape::new([4]);
    assert!(matches!(
        convert_dense_bf16_to_f32(&wrong_shape),
        Err(DenseConversionError::SourceTensorLengthMismatch { .. })
    ));

    let mut invalid_range = fixture(&directory.0);
    invalid_range.tensors[0].offset = u64::MAX;
    assert!(matches!(
        convert_dense_bf16_to_f32(&invalid_range),
        Err(DenseConversionError::InvalidSourceRange { .. })
    ));
}

#[test]
fn preflight_rejects_insufficient_space_before_output() {
    let directory = TestDirectory::new("preflight");
    let mut spec = fixture(&directory.0);
    spec.available_space_bytes = 0;

    assert!(matches!(
        convert_dense_bf16_to_f32(&spec),
        Err(DenseConversionError::InsufficientDiskSpace { .. })
    ));
    assert!(!spec.output_directory.exists());

    let mut overflowing_buffers = fixture(&directory.0);
    overflowing_buffers.source_chunk_bytes = usize::MAX - 1;
    assert!(matches!(
        convert_dense_bf16_to_f32(&overflowing_buffers),
        Err(DenseConversionError::ArithmeticOverflow {
            operation: "conversion working buffers"
        })
    ));
}

#[test]
fn injected_incomplete_commit_removes_temporary_and_final_outputs() {
    let directory = TestDirectory::new("cleanup");
    let spec = fixture(&directory.0);

    assert!(matches!(
        convert_impl(&spec, true),
        Err(DenseConversionError::InjectedIncompleteOutput)
    ));
    assert!(spec.output_directory.exists());
    assert!(
        fs::read_dir(&spec.output_directory)
            .expect("read output directory")
            .next()
            .is_none()
    );
}
