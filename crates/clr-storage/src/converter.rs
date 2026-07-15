use std::{
    collections::HashSet,
    fmt,
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use clr_core::{DataType, TensorShape};

use crate::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ByteOrder, Sha256Hasher, StorageError,
    TensorLocation, TensorMetadata,
};

/// Default source chunk size used by the unoptimized correctness converter.
pub const DEFAULT_CONVERSION_CHUNK_BYTES: usize = 64 * 1_024;

const DENSE_ARTIFACT_MANIFEST_VERSION: u32 = 1;
const PAYLOAD_FILE_NAME: &str = "dense-f32.bin";
const MANIFEST_FILE_NAME: &str = "dense-manifest-v1.json";
const TEMP_PAYLOAD_FILE_NAME: &str = ".dense-f32.bin.incomplete";
const TEMP_MANIFEST_FILE_NAME: &str = ".dense-manifest-v1.json.incomplete";

/// One complete source shard that must be verified before payload decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseSourceShard {
    /// Local path to the complete pinned source shard.
    pub path: PathBuf,
    /// Expected complete file length from pinned provenance.
    pub byte_length: u64,
    /// Expected SHA-256 of the complete source shard.
    pub sha256: [u8; 32],
}

/// One contiguous source tensor range selected for dense conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseSourceTensor {
    /// Canonical source tensor name.
    pub name: String,
    /// Validated source tensor shape in source axis order.
    pub shape: TensorShape,
    /// Source storage dtype, which must be BF16 for this converter.
    pub data_type: DataType,
    /// Zero-based index into the conversion's source-shard list.
    pub shard_index: usize,
    /// Absolute byte offset from the start of the source shard.
    pub offset: u64,
    /// Exact contiguous source payload length.
    pub length: u64,
}

/// Inputs for one deterministic dense BF16-to-F32 conversion transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseConversionSpec {
    /// Immutable upstream model identifier.
    pub model_id: String,
    /// Immutable upstream model revision.
    pub model_revision: String,
    /// Complete source shards needed by selected tensors.
    pub source_shards: Vec<DenseSourceShard>,
    /// Selected non-overlapping logical tensors.
    pub tensors: Vec<DenseSourceTensor>,
    /// Directory that receives the payload and manifest atomically.
    pub output_directory: PathBuf,
    /// Free bytes observed by the caller for preflight.
    pub available_space_bytes: u64,
    /// Even, non-zero BF16 source chunk size.
    pub source_chunk_bytes: usize,
}

/// Evidence returned after a committed, exact-round-trip dense conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseConversionSummary {
    /// Validated runtime artifact manifest.
    pub artifact_manifest: ArtifactManifest,
    /// Final shared physical payload path.
    pub payload_path: PathBuf,
    /// Final deterministic JSON manifest path.
    pub manifest_path: PathBuf,
    /// Complete physical payload SHA-256.
    pub payload_sha256: [u8; 32],
    /// Deterministic JSON manifest SHA-256.
    pub manifest_sha256: [u8; 32],
    /// Complete source bytes read while verifying shard hashes.
    pub source_verification_bytes_read: u64,
    /// BF16 tensor bytes read by conversion and exact verification.
    pub source_payload_bytes_read: u64,
    /// F32 payload bytes written.
    pub artifact_bytes_written: u64,
    /// Conservative free-space requirement checked before output creation.
    pub preflight_required_bytes: u64,
    /// Maximum explicitly allocated source/output working buffers.
    pub peak_buffer_bytes: usize,
}

/// Structured failures from dense source validation and conversion.
#[derive(Debug)]
pub enum DenseConversionError {
    /// Checked size arithmetic overflowed.
    ArithmeticOverflow { operation: &'static str },
    /// Source chunk size is zero or not BF16-aligned.
    InvalidChunkSize { actual: usize },
    /// A source tensor name occurs more than once.
    DuplicateTensor { name: String },
    /// A selected source tensor is not BF16.
    SourceDataTypeMismatch { tensor: String, actual: DataType },
    /// Shape-derived source bytes differ from the selected range length.
    SourceTensorLengthMismatch {
        tensor: String,
        expected: u64,
        actual: u64,
    },
    /// A tensor references a missing source shard.
    SourceShardOutOfRange {
        tensor: String,
        shard_index: usize,
        shard_count: usize,
    },
    /// A tensor byte range exceeds its declared source shard.
    InvalidSourceRange {
        tensor: String,
        offset: u64,
        length: u64,
        shard_length: u64,
    },
    /// A local source shard length differs from pinned provenance.
    SourceShardLengthMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    /// A complete local source shard hash differs from pinned provenance.
    SourceShardHashMismatch {
        path: PathBuf,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// Preflight found insufficient free space for output and manifest.
    InsufficientDiskSpace { required: u64, available: u64 },
    /// A final output path already exists.
    OutputExists { path: PathBuf },
    /// Streaming verification found an F32 byte mismatch.
    RoundTripMismatch { tensor: String, element: u64 },
    /// Existing artifact contract validation failed.
    Artifact(StorageError),
    /// A filesystem operation failed.
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    #[cfg(test)]
    InjectedIncompleteOutput,
}

impl fmt::Display for DenseConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow { operation } => {
                write!(
                    formatter,
                    "arithmetic overflow while calculating {operation}"
                )
            }
            Self::InvalidChunkSize { actual } => {
                write!(formatter, "invalid BF16 source chunk size {actual}")
            }
            Self::DuplicateTensor { name } => write!(formatter, "duplicate tensor '{name}'"),
            Self::SourceDataTypeMismatch { tensor, actual } => {
                write!(formatter, "tensor '{tensor}' must be BF16, got {actual}")
            }
            Self::SourceTensorLengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor '{tensor}' source length mismatch: expected {expected}, got {actual}"
            ),
            Self::SourceShardOutOfRange {
                tensor,
                shard_index,
                shard_count,
            } => write!(
                formatter,
                "tensor '{tensor}' uses shard {shard_index}, outside 0..{shard_count}"
            ),
            Self::InvalidSourceRange {
                tensor,
                offset,
                length,
                shard_length,
            } => write!(
                formatter,
                "tensor '{tensor}' range {offset}+{length} exceeds shard length {shard_length}"
            ),
            Self::SourceShardLengthMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "source shard '{}' length mismatch: expected {expected}, got {actual}",
                path.display()
            ),
            Self::SourceShardHashMismatch { path, .. } => {
                write!(
                    formatter,
                    "source shard '{}' SHA-256 mismatch",
                    path.display()
                )
            }
            Self::InsufficientDiskSpace {
                required,
                available,
            } => write!(
                formatter,
                "dense conversion requires {required} free bytes, only {available} available"
            ),
            Self::OutputExists { path } => {
                write!(formatter, "output '{}' already exists", path.display())
            }
            Self::RoundTripMismatch { tensor, element } => write!(
                formatter,
                "tensor '{tensor}' differs after F32 round trip at element {element}"
            ),
            Self::Artifact(error) => write!(formatter, "invalid dense artifact: {error}"),
            Self::Io {
                action,
                path,
                source,
            } => write!(
                formatter,
                "failed to {action} '{}': {source}",
                path.display()
            ),
            #[cfg(test)]
            Self::InjectedIncompleteOutput => write!(formatter, "injected incomplete output"),
        }
    }
}

impl std::error::Error for DenseConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<StorageError> for DenseConversionError {
    fn from(error: StorageError) -> Self {
        Self::Artifact(error)
    }
}

/// Converts BF16 bits to exactly representable F32 bits.
#[must_use]
pub const fn decode_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

/// Calculates the conservative output-space requirement without touching files.
///
/// # Errors
///
/// Returns a structured arithmetic error if shape-derived sizes overflow.
pub fn dense_conversion_preflight_bytes(
    tensors: &[DenseSourceTensor],
) -> Result<u64, DenseConversionError> {
    let payload_bytes = tensors.iter().try_fold(0_u64, |total, tensor| {
        let bytes = tensor
            .shape
            .byte_count(DataType::F32)
            .map_err(|_| overflow("dense artifact byte count"))?;
        let bytes = u64::try_from(bytes).map_err(|_| overflow("dense artifact byte count"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| overflow("complete dense artifact byte count"))
    })?;
    let names = tensors.iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(tensor.name.len() as u64)
            .ok_or_else(|| overflow("dense manifest name bytes"))
    })?;
    let manifest_reserve = (tensors.len() as u64)
        .checked_mul(512)
        .and_then(|bytes| bytes.checked_add(names))
        .and_then(|bytes| bytes.checked_add(4_096))
        .ok_or_else(|| overflow("dense manifest reserve"))?;
    payload_bytes
        .checked_add(manifest_reserve)
        .ok_or_else(|| overflow("dense conversion preflight bytes"))
}

/// Verifies complete source shards, converts selected logical tensors in
/// bounded chunks, verifies exact F32 bytes, and atomically commits the output.
///
/// # Errors
///
/// Returns a structured validation, integrity, preflight, round-trip, artifact,
/// or filesystem error. Incomplete temporary and final outputs are removed.
pub fn convert_dense_bf16_to_f32(
    spec: &DenseConversionSpec,
) -> Result<DenseConversionSummary, DenseConversionError> {
    convert_impl(spec, false)
}

fn convert_impl(
    spec: &DenseConversionSpec,
    fail_before_manifest_rename: bool,
) -> Result<DenseConversionSummary, DenseConversionError> {
    let validated = validate_spec(spec)?;
    verify_source_shards(spec, &validated.used_shards)?;
    fs::create_dir_all(&spec.output_directory).map_err(|source| {
        io_error(
            "create dense output directory",
            spec.output_directory.clone(),
            source,
        )
    })?;

    let mut outputs = OutputTransaction::new(&spec.output_directory)?;
    let payload = write_payload(spec, &validated.tensors, &outputs.temp_payload)?;
    verify_round_trip(spec, &validated.tensors, &outputs.temp_payload)?;
    let artifact_manifest =
        ArtifactManifest::new(ARTIFACT_FORMAT_VERSION, ByteOrder::Little, payload.tensors)?;
    let manifest_json = serialize_manifest(
        spec,
        &validated.tensors,
        &artifact_manifest,
        payload.sha256,
        payload.byte_length,
    );
    let manifest_sha256 = hash_bytes(manifest_json.as_bytes());
    write_manifest(&outputs.temp_manifest, manifest_json.as_bytes())?;

    fs::rename(&outputs.temp_payload, &outputs.final_payload).map_err(|source| {
        io_error(
            "commit dense payload",
            outputs.final_payload.clone(),
            source,
        )
    })?;
    outputs.payload_committed = true;
    if fail_before_manifest_rename {
        #[cfg(test)]
        return Err(DenseConversionError::InjectedIncompleteOutput);
        #[cfg(not(test))]
        unreachable!();
    }
    fs::rename(&outputs.temp_manifest, &outputs.final_manifest).map_err(|source| {
        io_error(
            "commit dense manifest",
            outputs.final_manifest.clone(),
            source,
        )
    })?;
    outputs.manifest_committed = true;

    let summary = DenseConversionSummary {
        artifact_manifest,
        payload_path: outputs.final_payload.clone(),
        manifest_path: outputs.final_manifest.clone(),
        payload_sha256: payload.sha256,
        manifest_sha256,
        source_verification_bytes_read: validated.verification_bytes,
        source_payload_bytes_read: validated
            .source_payload_bytes
            .checked_mul(2)
            .ok_or_else(|| overflow("source conversion and verification bytes"))?,
        artifact_bytes_written: payload.byte_length,
        preflight_required_bytes: validated.preflight_required_bytes,
        peak_buffer_bytes: spec
            .source_chunk_bytes
            .checked_mul(5)
            .ok_or_else(|| overflow("conversion working buffers"))?,
    };
    outputs.complete = true;
    Ok(summary)
}

#[derive(Debug)]
struct ValidatedSpec {
    tensors: Vec<DenseSourceTensor>,
    used_shards: Vec<bool>,
    verification_bytes: u64,
    source_payload_bytes: u64,
    preflight_required_bytes: u64,
}

struct ConvertedPayload {
    tensors: Vec<TensorMetadata>,
    sha256: [u8; 32],
    byte_length: u64,
}

fn write_payload(
    spec: &DenseConversionSpec,
    tensors: &[DenseSourceTensor],
    path: &Path,
) -> Result<ConvertedPayload, DenseConversionError> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io_error("create temporary dense payload", path.to_owned(), source))?;
    let mut writer = BufWriter::new(file);
    let mut source_buffer = vec![0_u8; spec.source_chunk_bytes];
    let mut output_buffer = vec![0_u8; spec.source_chunk_bytes * 2];
    let mut payload_hasher = Sha256Hasher::new();
    let mut artifact_tensors = Vec::with_capacity(tensors.len());
    let mut artifact_offset = 0_u64;
    for source_tensor in tensors {
        let (metadata, length) = write_tensor(
            spec,
            source_tensor,
            artifact_offset,
            &mut writer,
            &mut source_buffer,
            &mut output_buffer,
            &mut payload_hasher,
            path,
        )?;
        artifact_tensors.push(metadata);
        artifact_offset = artifact_offset
            .checked_add(length)
            .ok_or_else(|| overflow("dense artifact offset"))?;
    }
    writer
        .flush()
        .map_err(|source| io_error("flush temporary dense payload", path.to_owned(), source))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|source| io_error("sync temporary dense payload", path.to_owned(), source))?;
    Ok(ConvertedPayload {
        tensors: artifact_tensors,
        sha256: payload_hasher.finalize(),
        byte_length: artifact_offset,
    })
}

#[allow(clippy::too_many_arguments)] // Keeps one tensor's streaming state explicit.
fn write_tensor(
    spec: &DenseConversionSpec,
    source_tensor: &DenseSourceTensor,
    artifact_offset: u64,
    writer: &mut impl Write,
    source_buffer: &mut [u8],
    output_buffer: &mut [u8],
    payload_hasher: &mut Sha256Hasher,
    payload_path: &Path,
) -> Result<(TensorMetadata, u64), DenseConversionError> {
    let mut source = open_source_tensor(spec, source_tensor)?;
    let mut remaining = source_tensor.length;
    let mut tensor_hasher = Sha256Hasher::new();
    while remaining != 0 {
        let read_length = usize::try_from(remaining.min(spec.source_chunk_bytes as u64))
            .map_err(|_| overflow("source chunk length"))?;
        source
            .read_exact(&mut source_buffer[..read_length])
            .map_err(|source| {
                io_error(
                    "read BF16 source tensor",
                    spec.source_shards[source_tensor.shard_index].path.clone(),
                    source,
                )
            })?;
        let write_length = decode_chunk(
            &source_buffer[..read_length],
            &mut output_buffer[..read_length * 2],
        );
        writer
            .write_all(&output_buffer[..write_length])
            .map_err(|source| {
                io_error(
                    "write temporary dense payload",
                    payload_path.to_owned(),
                    source,
                )
            })?;
        tensor_hasher.update(&output_buffer[..write_length]);
        payload_hasher.update(&output_buffer[..write_length]);
        remaining -= read_length as u64;
    }
    let artifact_length = source_tensor
        .length
        .checked_mul(2)
        .ok_or_else(|| overflow("converted tensor byte length"))?;
    Ok((
        TensorMetadata {
            name: source_tensor.name.clone(),
            shape: source_tensor.shape.clone(),
            data_type: DataType::F32,
            location: TensorLocation {
                path: PAYLOAD_FILE_NAME.into(),
                offset: artifact_offset,
                length: artifact_length,
            },
            sha256: tensor_hasher.finalize(),
        },
        artifact_length,
    ))
}

fn write_manifest(path: &Path, bytes: &[u8]) -> Result<(), DenseConversionError> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io_error("create temporary dense manifest", path.to_owned(), source))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .and_then(|()| writer.flush())
        .map_err(|source| io_error("write temporary dense manifest", path.to_owned(), source))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|source| io_error("sync temporary dense manifest", path.to_owned(), source))
}

fn validate_spec(spec: &DenseConversionSpec) -> Result<ValidatedSpec, DenseConversionError> {
    if spec.source_chunk_bytes == 0 || spec.source_chunk_bytes % 2 != 0 {
        return Err(DenseConversionError::InvalidChunkSize {
            actual: spec.source_chunk_bytes,
        });
    }
    spec.source_chunk_bytes
        .checked_mul(5)
        .ok_or_else(|| overflow("conversion working buffers"))?;
    let preflight_required_bytes = dense_conversion_preflight_bytes(&spec.tensors)?;
    if preflight_required_bytes > spec.available_space_bytes {
        return Err(DenseConversionError::InsufficientDiskSpace {
            required: preflight_required_bytes,
            available: spec.available_space_bytes,
        });
    }
    let mut tensors = spec.tensors.clone();
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let mut names = HashSet::with_capacity(tensors.len());
    let mut used_shards = vec![false; spec.source_shards.len()];
    let mut source_payload_bytes = 0_u64;
    for tensor in &tensors {
        if !names.insert(tensor.name.as_str()) {
            return Err(DenseConversionError::DuplicateTensor {
                name: tensor.name.clone(),
            });
        }
        if tensor.data_type != DataType::BF16 {
            return Err(DenseConversionError::SourceDataTypeMismatch {
                tensor: tensor.name.clone(),
                actual: tensor.data_type,
            });
        }
        let expected = tensor
            .shape
            .byte_count(DataType::BF16)
            .map_err(|_| overflow("BF16 source tensor byte count"))?;
        let expected =
            u64::try_from(expected).map_err(|_| overflow("BF16 source tensor byte count"))?;
        if expected != tensor.length {
            return Err(DenseConversionError::SourceTensorLengthMismatch {
                tensor: tensor.name.clone(),
                expected,
                actual: tensor.length,
            });
        }
        let Some(shard) = spec.source_shards.get(tensor.shard_index) else {
            return Err(DenseConversionError::SourceShardOutOfRange {
                tensor: tensor.name.clone(),
                shard_index: tensor.shard_index,
                shard_count: spec.source_shards.len(),
            });
        };
        let Some(end) = tensor.offset.checked_add(tensor.length) else {
            return Err(DenseConversionError::InvalidSourceRange {
                tensor: tensor.name.clone(),
                offset: tensor.offset,
                length: tensor.length,
                shard_length: shard.byte_length,
            });
        };
        if end > shard.byte_length {
            return Err(DenseConversionError::InvalidSourceRange {
                tensor: tensor.name.clone(),
                offset: tensor.offset,
                length: tensor.length,
                shard_length: shard.byte_length,
            });
        }
        used_shards[tensor.shard_index] = true;
        source_payload_bytes = source_payload_bytes
            .checked_add(tensor.length)
            .ok_or_else(|| overflow("selected source payload bytes"))?;
    }
    let verification_bytes = spec
        .source_shards
        .iter()
        .zip(&used_shards)
        .filter(|(_, used)| **used)
        .try_fold(0_u64, |total, (shard, _)| {
            total
                .checked_add(shard.byte_length)
                .ok_or_else(|| overflow("source verification bytes"))
        })?;
    Ok(ValidatedSpec {
        tensors,
        used_shards,
        verification_bytes,
        source_payload_bytes,
        preflight_required_bytes,
    })
}

fn verify_source_shards(
    spec: &DenseConversionSpec,
    used_shards: &[bool],
) -> Result<(), DenseConversionError> {
    let mut buffer = vec![0_u8; spec.source_chunk_bytes];
    for (shard, used) in spec.source_shards.iter().zip(used_shards) {
        if !used {
            continue;
        }
        let actual_length = fs::metadata(&shard.path)
            .map_err(|source| io_error("read source shard metadata", shard.path.clone(), source))?
            .len();
        if actual_length != shard.byte_length {
            return Err(DenseConversionError::SourceShardLengthMismatch {
                path: shard.path.clone(),
                expected: shard.byte_length,
                actual: actual_length,
            });
        }
        let file = File::open(&shard.path)
            .map_err(|source| io_error("open source shard", shard.path.clone(), source))?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256Hasher::new();
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|source| io_error("hash source shard", shard.path.clone(), source))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = hasher.finalize();
        if actual != shard.sha256 {
            return Err(DenseConversionError::SourceShardHashMismatch {
                path: shard.path.clone(),
                expected: shard.sha256,
                actual,
            });
        }
    }
    Ok(())
}

fn open_source_tensor(
    spec: &DenseConversionSpec,
    tensor: &DenseSourceTensor,
) -> Result<File, DenseConversionError> {
    let path = &spec.source_shards[tensor.shard_index].path;
    let mut file = File::open(path)
        .map_err(|source| io_error("open verified source shard", path.clone(), source))?;
    file.seek(SeekFrom::Start(tensor.offset))
        .map_err(|source| io_error("seek verified source tensor", path.clone(), source))?;
    Ok(file)
}

fn verify_round_trip(
    spec: &DenseConversionSpec,
    tensors: &[DenseSourceTensor],
    payload_path: &Path,
) -> Result<(), DenseConversionError> {
    let mut artifact = File::open(payload_path).map_err(|source| {
        io_error(
            "open temporary dense payload",
            payload_path.to_owned(),
            source,
        )
    })?;
    let mut source_buffer = vec![0_u8; spec.source_chunk_bytes];
    let mut expected_buffer = vec![0_u8; spec.source_chunk_bytes * 2];
    let mut actual_buffer = vec![0_u8; spec.source_chunk_bytes * 2];
    let mut artifact_offset = 0_u64;
    for tensor in tensors {
        let mut source = open_source_tensor(spec, tensor)?;
        artifact
            .seek(SeekFrom::Start(artifact_offset))
            .map_err(|source| {
                io_error(
                    "seek temporary dense payload",
                    payload_path.to_owned(),
                    source,
                )
            })?;
        let mut remaining = tensor.length;
        let mut element = 0_u64;
        while remaining != 0 {
            let read_length = usize::try_from(remaining.min(spec.source_chunk_bytes as u64))
                .map_err(|_| overflow("verification chunk length"))?;
            source
                .read_exact(&mut source_buffer[..read_length])
                .map_err(|source| {
                    io_error(
                        "read source for round-trip verification",
                        spec.source_shards[tensor.shard_index].path.clone(),
                        source,
                    )
                })?;
            let artifact_length = decode_chunk(
                &source_buffer[..read_length],
                &mut expected_buffer[..read_length * 2],
            );
            artifact
                .read_exact(&mut actual_buffer[..artifact_length])
                .map_err(|source| {
                    io_error(
                        "read artifact for round-trip verification",
                        payload_path.to_owned(),
                        source,
                    )
                })?;
            if let Some(index) = expected_buffer[..artifact_length]
                .chunks_exact(4)
                .zip(actual_buffer[..artifact_length].chunks_exact(4))
                .position(|(expected, actual)| expected != actual)
            {
                return Err(DenseConversionError::RoundTripMismatch {
                    tensor: tensor.name.clone(),
                    element: element + index as u64,
                });
            }
            element += (read_length / 2) as u64;
            remaining -= read_length as u64;
        }
        let artifact_length = tensor
            .length
            .checked_mul(2)
            .ok_or_else(|| overflow("round-trip tensor byte length"))?;
        artifact_offset = artifact_offset
            .checked_add(artifact_length)
            .ok_or_else(|| overflow("round-trip artifact offset"))?;
    }
    Ok(())
}

fn decode_chunk(source: &[u8], output: &mut [u8]) -> usize {
    for (source, output) in source.chunks_exact(2).zip(output.chunks_exact_mut(4)) {
        let value = decode_bf16(u16::from_le_bytes([source[0], source[1]]));
        output.copy_from_slice(&value.to_bits().to_le_bytes());
    }
    source.len() * 2
}

fn serialize_manifest(
    spec: &DenseConversionSpec,
    source_tensors: &[DenseSourceTensor],
    artifact: &ArtifactManifest,
    payload_sha256: [u8; 32],
    payload_bytes: u64,
) -> String {
    let mut output = String::new();
    writeln!(&mut output, "{{").expect("write string");
    writeln!(
        &mut output,
        "  \"format_version\": {DENSE_ARTIFACT_MANIFEST_VERSION},"
    )
    .expect("write string");
    write!(&mut output, "  \"model_id\": ").expect("write string");
    write_json_string(&mut output, &spec.model_id);
    writeln!(&mut output, ",").expect("write string");
    write!(&mut output, "  \"model_revision\": ").expect("write string");
    write_json_string(&mut output, &spec.model_revision);
    writeln!(&mut output, ",").expect("write string");
    writeln!(&mut output, "  \"source_dtype\": \"BF16\",").expect("write string");
    writeln!(&mut output, "  \"artifact_dtype\": \"F32\",").expect("write string");
    writeln!(&mut output, "  \"endianness\": \"little\",").expect("write string");
    writeln!(&mut output, "  \"artifact\": {{").expect("write string");
    writeln!(&mut output, "    \"path\": \"{PAYLOAD_FILE_NAME}\",").expect("write string");
    writeln!(&mut output, "    \"byte_length\": {payload_bytes},").expect("write string");
    writeln!(&mut output, "    \"sha256\": \"{}\"", hex(&payload_sha256)).expect("write string");
    writeln!(&mut output, "  }},").expect("write string");
    writeln!(&mut output, "  \"tensors\": [").expect("write string");
    for (index, (source, tensor)) in source_tensors.iter().zip(artifact.tensors()).enumerate() {
        writeln!(&mut output, "    {{").expect("write string");
        write!(&mut output, "      \"name\": ").expect("write string");
        write_json_string(&mut output, &tensor.name);
        writeln!(&mut output, ",").expect("write string");
        writeln!(
            &mut output,
            "      \"shape\": {},",
            shape_json(&tensor.shape)
        )
        .expect("write string");
        writeln!(&mut output, "      \"offset\": {},", tensor.location.offset)
            .expect("write string");
        writeln!(
            &mut output,
            "      \"byte_length\": {},",
            tensor.location.length
        )
        .expect("write string");
        writeln!(
            &mut output,
            "      \"sha256\": \"{}\",",
            hex(&tensor.sha256)
        )
        .expect("write string");
        writeln!(
            &mut output,
            "      \"source_shard_index\": {},",
            source.shard_index
        )
        .expect("write string");
        writeln!(&mut output, "      \"source_offset\": {},", source.offset).expect("write string");
        writeln!(
            &mut output,
            "      \"source_byte_length\": {}",
            source.length
        )
        .expect("write string");
        let comma = if index + 1 == source_tensors.len() {
            ""
        } else {
            ","
        };
        writeln!(&mut output, "    }}{comma}").expect("write string");
    }
    writeln!(&mut output, "  ]").expect("write string");
    writeln!(&mut output, "}}").expect("write string");
    output
}

fn write_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                write!(output, "\\u{:04x}", u32::from(character)).expect("write string");
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

fn shape_json(shape: &TensorShape) -> String {
    format!(
        "[{}]",
        shape
            .dimensions()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn hex(hash: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in hash {
        write!(&mut output, "{byte:02x}").expect("write string");
    }
    output
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

struct OutputTransaction {
    temp_payload: PathBuf,
    temp_manifest: PathBuf,
    final_payload: PathBuf,
    final_manifest: PathBuf,
    payload_committed: bool,
    manifest_committed: bool,
    complete: bool,
}

impl OutputTransaction {
    fn new(root: &Path) -> Result<Self, DenseConversionError> {
        let transaction = Self {
            temp_payload: root.join(TEMP_PAYLOAD_FILE_NAME),
            temp_manifest: root.join(TEMP_MANIFEST_FILE_NAME),
            final_payload: root.join(PAYLOAD_FILE_NAME),
            final_manifest: root.join(MANIFEST_FILE_NAME),
            payload_committed: false,
            manifest_committed: false,
            complete: false,
        };
        for final_path in [&transaction.final_payload, &transaction.final_manifest] {
            if final_path.exists() {
                return Err(DenseConversionError::OutputExists {
                    path: final_path.clone(),
                });
            }
        }
        remove_if_exists(&transaction.temp_payload)?;
        remove_if_exists(&transaction.temp_manifest)?;
        Ok(transaction)
    }
}

impl Drop for OutputTransaction {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let _ = fs::remove_file(&self.temp_payload);
        let _ = fs::remove_file(&self.temp_manifest);
        if self.payload_committed {
            let _ = fs::remove_file(&self.final_payload);
        }
        if self.manifest_committed {
            let _ = fs::remove_file(&self.final_manifest);
        }
    }
}

fn remove_if_exists(path: &Path) -> Result<(), DenseConversionError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io_error(
            "remove incomplete dense output",
            path.to_owned(),
            source,
        )),
    }
}

const fn overflow(operation: &'static str) -> DenseConversionError {
    DenseConversionError::ArithmeticOverflow { operation }
}

fn io_error(action: &'static str, path: PathBuf, source: io::Error) -> DenseConversionError {
    DenseConversionError::Io {
        action,
        path,
        source,
    }
}

#[cfg(test)]
mod tests;
