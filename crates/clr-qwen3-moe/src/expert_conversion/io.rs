use std::{
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use clr_core::{DataType, TensorShape};
use clr_storage::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ByteOrder, ExpertRegistration, Sha256Hasher,
    TensorLocation, TensorMetadata, decode_bf16,
};

use super::{
    ExpertProjectionArtifactRange, PINNED_QWEN3_30B_A3B_MODEL_ID, PINNED_QWEN3_30B_A3B_REVISION,
    Qwen3MoeExpertArtifactManifest, Qwen3MoeExpertArtifactRecord, Qwen3MoeExpertConversionError,
    Qwen3MoeExpertConversionSpec, Qwen3MoeExpertConversionSummary, Qwen3MoeExpertShardRecord,
    ValidatedExpert, ValidatedSpec, expert_key, logical_name, overflow, projection_ranges,
    shard_path, validate_spec,
};

const EXPERT_MANIFEST_VERSION: u32 = 1;
const MANIFEST_FILE_NAME: &str = "expert-manifest-v1.json";
const TEMP_MANIFEST_FILE_NAME: &str = ".expert-manifest-v1.json.incomplete";

pub(super) fn convert(
    spec: &Qwen3MoeExpertConversionSpec,
    fail_after_shard: Option<usize>,
) -> Result<Qwen3MoeExpertConversionSummary, Qwen3MoeExpertConversionError> {
    let validated = validate_spec(spec)?;
    let (placeholder_shards, placeholder_experts) = placeholder_records(&validated);
    let exact_manifest_bytes =
        manifest_json(&placeholder_shards, &placeholder_experts).len() as u64;
    let preflight_required_bytes = validated
        .artifact_bytes
        .checked_add(exact_manifest_bytes)
        .ok_or_else(|| overflow("expert preflight bytes"))?;
    if preflight_required_bytes > spec.available_space_bytes {
        return Err(Qwen3MoeExpertConversionError::InsufficientDiskSpace {
            required: preflight_required_bytes,
            available: spec.available_space_bytes,
        });
    }
    verify_source_shards(spec, &validated)?;
    let mut transaction = OutputTransaction::new(&spec.output_directory)?;
    let mut runtime_tensors = Vec::with_capacity(validated.experts.len());
    let mut registrations = Vec::with_capacity(validated.experts.len());
    let mut expert_records = Vec::with_capacity(validated.experts.len());
    let mut shard_records = Vec::with_capacity(validated.output_shard_lengths.len());

    for (shard_id, expected_length) in &validated.output_shard_lengths {
        let experts: Vec<_> = validated
            .experts
            .iter()
            .filter(|expert| expert.shard_id == *shard_id)
            .collect();
        let relative_path = shard_path(*shard_id);
        let final_path = spec.output_directory.join(&relative_path);
        let temp_path = spec
            .output_directory
            .join(format!(".{}.incomplete", relative_path.display()));
        let output = write_layer_shard(spec, &validated, &experts, &relative_path, &temp_path)?;
        if output.byte_length != *expected_length {
            return Err(overflow("final expert layer shard length"));
        }
        fs::rename(&temp_path, &final_path)
            .map_err(|source| io_error("commit expert layer shard", final_path.clone(), source))?;
        transaction.track(final_path);
        runtime_tensors.extend(output.runtime_tensors);
        registrations.extend(output.registrations);
        expert_records.extend(output.expert_records);
        shard_records.push(Qwen3MoeExpertShardRecord {
            shard_id: *shard_id,
            path: relative_path,
            byte_length: output.byte_length,
            sha256: output.sha256,
        });
        if fail_after_shard == Some(*shard_id) {
            #[cfg(test)]
            return Err(Qwen3MoeExpertConversionError::InjectedIncompleteOutput);
            #[cfg(not(test))]
            unreachable!();
        }
    }

    let runtime_manifest =
        ArtifactManifest::new(ARTIFACT_FORMAT_VERSION, ByteOrder::Little, runtime_tensors)?;
    let manifest = Qwen3MoeExpertArtifactManifest {
        runtime_manifest,
        registrations,
        shards: shard_records,
        experts: expert_records,
    };
    let manifest_bytes = manifest_json(manifest.shards(), manifest.experts()).into_bytes();
    if manifest_bytes.len() as u64 != exact_manifest_bytes {
        return Err(overflow("deterministic expert manifest length"));
    }
    let manifest_sha256 = hash_bytes(&manifest_bytes);
    let temp_manifest = spec.output_directory.join(TEMP_MANIFEST_FILE_NAME);
    let final_manifest = spec.output_directory.join(MANIFEST_FILE_NAME);
    write_synced(&temp_manifest, &manifest_bytes, "write expert manifest")?;
    fs::rename(&temp_manifest, &final_manifest)
        .map_err(|source| io_error("commit expert manifest", final_manifest.clone(), source))?;
    transaction.track(final_manifest.clone());

    let summary = Qwen3MoeExpertConversionSummary {
        manifest,
        manifest_path: final_manifest,
        manifest_sha256,
        source_verification_bytes_read: validated.source_verification_bytes,
        source_payload_bytes_read: validated
            .source_payload_bytes
            .checked_mul(2)
            .ok_or_else(|| overflow("expert conversion and verification bytes"))?,
        artifact_bytes_written: validated.artifact_bytes,
        preflight_required_bytes,
        peak_buffer_bytes: validated.peak_buffer_bytes,
    };
    transaction.complete = true;
    Ok(summary)
}

struct LayerOutput {
    runtime_tensors: Vec<TensorMetadata>,
    registrations: Vec<ExpertRegistration>,
    expert_records: Vec<Qwen3MoeExpertArtifactRecord>,
    byte_length: u64,
    sha256: [u8; 32],
}

fn write_layer_shard(
    spec: &Qwen3MoeExpertConversionSpec,
    validated: &ValidatedSpec,
    experts: &[&ValidatedExpert],
    relative_path: &Path,
    temp_path: &Path,
) -> Result<LayerOutput, Qwen3MoeExpertConversionError> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp_path)
        .map_err(|source| {
            io_error(
                "create temporary expert shard",
                temp_path.to_owned(),
                source,
            )
        })?;
    let mut writer = BufWriter::new(file);
    let mut source_buffer = vec![0_u8; spec.source_chunk_bytes];
    let mut output_buffer = vec![0_u8; spec.source_chunk_bytes * 2];
    let mut shard_hasher = Sha256Hasher::new();
    let mut runtime_tensors = Vec::with_capacity(experts.len());
    let mut registrations = Vec::with_capacity(experts.len());
    let mut expert_records = Vec::with_capacity(experts.len());
    let ranges = projection_ranges(validated.layout, validated.config);

    for expert in experts {
        let mut expert_hasher = Sha256Hasher::new();
        for projection in &expert.projections {
            write_projection(
                spec,
                projection,
                &mut writer,
                &mut source_buffer,
                &mut output_buffer,
                &mut expert_hasher,
                &mut shard_hasher,
                temp_path,
            )?;
        }
        let hash = expert_hasher.finalize();
        let name = logical_name(expert.layer_index, expert.expert_index);
        let key = expert_key(expert.layer_index, expert.expert_index);
        let payload_length = validated.layout.total_byte_length as u64;
        runtime_tensors.push(TensorMetadata {
            name: name.clone(),
            shape: TensorShape::new([validated.layout.total_byte_length / 4]),
            data_type: DataType::F32,
            location: TensorLocation {
                path: relative_path.to_owned(),
                offset: expert.payload_offset,
                length: payload_length,
            },
            sha256: hash,
        });
        registrations.push(ExpertRegistration {
            key,
            tensor_name: name,
        });
        expert_records.push(expert_record(expert, ranges.clone(), hash, payload_length));
    }
    writer
        .flush()
        .map_err(|source| io_error("flush temporary expert shard", temp_path.to_owned(), source))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|source| io_error("sync temporary expert shard", temp_path.to_owned(), source))?;
    let byte_length = writer
        .get_ref()
        .metadata()
        .map_err(|source| {
            io_error(
                "read temporary expert shard metadata",
                temp_path.to_owned(),
                source,
            )
        })?
        .len();
    drop(writer);
    verify_layer_round_trip(spec, experts, temp_path)?;
    Ok(LayerOutput {
        runtime_tensors,
        registrations,
        expert_records,
        byte_length,
        sha256: shard_hasher.finalize(),
    })
}

#[allow(clippy::too_many_arguments)] // Makes both logical and physical hashes explicit.
fn write_projection(
    spec: &Qwen3MoeExpertConversionSpec,
    projection: &super::Qwen3MoeExpertSourceProjection,
    writer: &mut impl Write,
    source_buffer: &mut [u8],
    output_buffer: &mut [u8],
    expert_hasher: &mut Sha256Hasher,
    shard_hasher: &mut Sha256Hasher,
    temp_path: &Path,
) -> Result<(), Qwen3MoeExpertConversionError> {
    let source_path = &spec.source_shards[projection.metadata.shard_index()].path;
    let mut source = File::open(source_path)
        .map_err(|error| io_error("open expert source shard", source_path.clone(), error))?;
    source
        .seek(SeekFrom::Start(projection.offset))
        .map_err(|error| io_error("seek expert source projection", source_path.clone(), error))?;
    let mut remaining = projection.length;
    while remaining != 0 {
        let read_length = usize::try_from(remaining.min(spec.source_chunk_bytes as u64))
            .map_err(|_| overflow("expert source chunk length"))?;
        source
            .read_exact(&mut source_buffer[..read_length])
            .map_err(|error| {
                io_error("read expert source projection", source_path.clone(), error)
            })?;
        let write_length = decode_chunk(
            &source_buffer[..read_length],
            &mut output_buffer[..read_length * 2],
        );
        writer
            .write_all(&output_buffer[..write_length])
            .map_err(|error| {
                io_error("write temporary expert shard", temp_path.to_owned(), error)
            })?;
        expert_hasher.update(&output_buffer[..write_length]);
        shard_hasher.update(&output_buffer[..write_length]);
        remaining -= read_length as u64;
    }
    Ok(())
}

fn verify_layer_round_trip(
    spec: &Qwen3MoeExpertConversionSpec,
    experts: &[&ValidatedExpert],
    artifact_path: &Path,
) -> Result<(), Qwen3MoeExpertConversionError> {
    let mut artifact = File::open(artifact_path).map_err(|source| {
        io_error(
            "open temporary expert shard",
            artifact_path.to_owned(),
            source,
        )
    })?;
    let mut source_buffer = vec![0_u8; spec.source_chunk_bytes];
    let mut expected_buffer = vec![0_u8; spec.source_chunk_bytes * 2];
    let mut actual_buffer = vec![0_u8; spec.source_chunk_bytes * 2];
    for expert in experts {
        artifact
            .seek(SeekFrom::Start(expert.payload_offset))
            .map_err(|source| {
                io_error(
                    "seek temporary expert payload",
                    artifact_path.to_owned(),
                    source,
                )
            })?;
        let mut element = 0_u64;
        for projection in &expert.projections {
            let source_path = &spec.source_shards[projection.metadata.shard_index()].path;
            let mut source = File::open(source_path).map_err(|error| {
                io_error(
                    "open expert verification source",
                    source_path.clone(),
                    error,
                )
            })?;
            source
                .seek(SeekFrom::Start(projection.offset))
                .map_err(|error| {
                    io_error(
                        "seek expert verification source",
                        source_path.clone(),
                        error,
                    )
                })?;
            let mut remaining = projection.length;
            while remaining != 0 {
                let read_length = usize::try_from(remaining.min(spec.source_chunk_bytes as u64))
                    .map_err(|_| overflow("expert verification chunk length"))?;
                source
                    .read_exact(&mut source_buffer[..read_length])
                    .map_err(|error| {
                        io_error(
                            "read expert verification source",
                            source_path.clone(),
                            error,
                        )
                    })?;
                let artifact_length = decode_chunk(
                    &source_buffer[..read_length],
                    &mut expected_buffer[..read_length * 2],
                );
                artifact
                    .read_exact(&mut actual_buffer[..artifact_length])
                    .map_err(|error| {
                        io_error(
                            "read expert verification artifact",
                            artifact_path.to_owned(),
                            error,
                        )
                    })?;
                if let Some(index) = expected_buffer[..artifact_length]
                    .chunks_exact(4)
                    .zip(actual_buffer[..artifact_length].chunks_exact(4))
                    .position(|(expected, actual)| expected != actual)
                {
                    return Err(Qwen3MoeExpertConversionError::RoundTripMismatch {
                        layer: expert.layer_index,
                        expert: expert.expert_index,
                        element: element + index as u64,
                    });
                }
                element += (read_length / 2) as u64;
                remaining -= read_length as u64;
            }
        }
    }
    Ok(())
}

fn verify_source_shards(
    spec: &Qwen3MoeExpertConversionSpec,
    validated: &ValidatedSpec,
) -> Result<(), Qwen3MoeExpertConversionError> {
    let mut buffer = vec![0_u8; spec.source_chunk_bytes];
    for (shard, used) in spec.source_shards.iter().zip(&validated.used_source_shards) {
        if !used {
            continue;
        }
        let actual_length = fs::metadata(&shard.path)
            .map_err(|source| io_error("read expert source metadata", shard.path.clone(), source))?
            .len();
        if actual_length != shard.byte_length {
            return Err(Qwen3MoeExpertConversionError::SourceShardLengthMismatch {
                path: shard.path.clone(),
            });
        }
        let file = File::open(&shard.path)
            .map_err(|source| io_error("open expert source shard", shard.path.clone(), source))?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256Hasher::new();
        loop {
            let read = reader.read(&mut buffer).map_err(|source| {
                io_error("hash expert source shard", shard.path.clone(), source)
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        if hasher.finalize() != shard.sha256 {
            return Err(Qwen3MoeExpertConversionError::SourceShardHashMismatch {
                path: shard.path.clone(),
            });
        }
    }
    Ok(())
}

fn placeholder_records(
    validated: &ValidatedSpec,
) -> (
    Vec<Qwen3MoeExpertShardRecord>,
    Vec<Qwen3MoeExpertArtifactRecord>,
) {
    let shards = validated
        .output_shard_lengths
        .iter()
        .map(|(shard_id, byte_length)| Qwen3MoeExpertShardRecord {
            shard_id: *shard_id,
            path: shard_path(*shard_id),
            byte_length: *byte_length,
            sha256: [0; 32],
        })
        .collect();
    let ranges = projection_ranges(validated.layout, validated.config);
    let experts = validated
        .experts
        .iter()
        .map(|expert| {
            expert_record(
                expert,
                ranges.clone(),
                [0; 32],
                validated.layout.total_byte_length as u64,
            )
        })
        .collect();
    (shards, experts)
}

fn expert_record(
    expert: &ValidatedExpert,
    [gate, up, down]: [ExpertProjectionArtifactRange; 3],
    sha256: [u8; 32],
    payload_length: u64,
) -> Qwen3MoeExpertArtifactRecord {
    Qwen3MoeExpertArtifactRecord {
        key: expert_key(expert.layer_index, expert.expert_index),
        shard_id: expert.shard_id,
        payload_offset: expert.payload_offset,
        payload_length,
        gate,
        up,
        down,
        source_data_type: DataType::BF16,
        artifact_data_type: DataType::F32,
        sha256,
    }
}

fn manifest_json(
    shards: &[Qwen3MoeExpertShardRecord],
    experts: &[Qwen3MoeExpertArtifactRecord],
) -> String {
    let mut output = String::new();
    writeln!(&mut output, "{{").expect("write string");
    writeln!(
        &mut output,
        "  \"format_version\": {EXPERT_MANIFEST_VERSION},"
    )
    .expect("write string");
    writeln!(
        &mut output,
        "  \"model_id\": \"{PINNED_QWEN3_30B_A3B_MODEL_ID}\","
    )
    .expect("write string");
    writeln!(
        &mut output,
        "  \"model_revision\": \"{PINNED_QWEN3_30B_A3B_REVISION}\","
    )
    .expect("write string");
    writeln!(&mut output, "  \"source_dtype\": \"BF16\",").expect("write string");
    writeln!(&mut output, "  \"artifact_dtype\": \"F32\",").expect("write string");
    writeln!(&mut output, "  \"endianness\": \"little\",").expect("write string");
    writeln!(
        &mut output,
        "  \"shard_policy\": \"one container per selected layer\","
    )
    .expect("write string");
    writeln!(&mut output, "  \"shards\": [").expect("write string");
    for (index, shard) in shards.iter().enumerate() {
        let comma = comma(index, shards.len());
        writeln!(
            &mut output,
            "    {{\"shard_id\": {}, \"path\": \"{}\", \"byte_length\": {}, \"sha256\": \"{}\"}}{comma}",
            shard.shard_id,
            shard.path.display(),
            shard.byte_length,
            hex(&shard.sha256)
        )
        .expect("write string");
    }
    writeln!(&mut output, "  ],").expect("write string");
    writeln!(&mut output, "  \"experts\": [").expect("write string");
    for (index, expert) in experts.iter().enumerate() {
        let comma = comma(index, experts.len());
        writeln!(&mut output, "    {{").expect("write string");
        writeln!(&mut output, "      \"layer\": {},", expert.key.layer_index)
            .expect("write string");
        writeln!(&mut output, "      \"expert\": {},", expert.key.expert_id.0)
            .expect("write string");
        writeln!(&mut output, "      \"shard_id\": {},", expert.shard_id).expect("write string");
        writeln!(
            &mut output,
            "      \"payload_offset\": {},",
            expert.payload_offset
        )
        .expect("write string");
        writeln!(
            &mut output,
            "      \"payload_length\": {},",
            expert.payload_length
        )
        .expect("write string");
        writeln!(&mut output, "      \"source_dtype\": \"BF16\",").expect("write string");
        writeln!(&mut output, "      \"artifact_dtype\": \"F32\",").expect("write string");
        write_range(&mut output, "gate", &expert.gate, true);
        write_range(&mut output, "up", &expert.up, true);
        write_range(&mut output, "down", &expert.down, true);
        writeln!(&mut output, "      \"sha256\": \"{}\"", hex(&expert.sha256))
            .expect("write string");
        writeln!(&mut output, "    }}{comma}").expect("write string");
    }
    writeln!(&mut output, "  ]").expect("write string");
    writeln!(&mut output, "}}").expect("write string");
    output
}

fn write_range(
    output: &mut String,
    name: &str,
    range: &ExpertProjectionArtifactRange,
    trailing_comma: bool,
) {
    let comma = if trailing_comma { "," } else { "" };
    writeln!(
        output,
        "      \"{name}\": {{\"offset\": {}, \"length\": {}, \"shape\": {}}}{comma}",
        range.offset,
        range.length,
        shape_json(&range.shape)
    )
    .expect("write string");
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

fn decode_chunk(source: &[u8], output: &mut [u8]) -> usize {
    for (source, output) in source.chunks_exact(2).zip(output.chunks_exact_mut(4)) {
        let value = decode_bf16(u16::from_le_bytes([source[0], source[1]]));
        output.copy_from_slice(&value.to_bits().to_le_bytes());
    }
    source.len() * 2
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn hex(hash: &[u8; 32]) -> String {
    hash.iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("write string");
            output
        })
}

const fn comma(index: usize, length: usize) -> &'static str {
    if index + 1 == length { "" } else { "," }
}

fn write_synced(
    path: &Path,
    bytes: &[u8],
    action: &'static str,
) -> Result<(), Qwen3MoeExpertConversionError> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io_error(action, path.to_owned(), source))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .and_then(|()| writer.flush())
        .map_err(|source| io_error(action, path.to_owned(), source))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|source| io_error(action, path.to_owned(), source))
}

struct OutputTransaction {
    root: PathBuf,
    paths: Vec<PathBuf>,
    complete: bool,
}

impl OutputTransaction {
    fn new(root: &Path) -> Result<Self, Qwen3MoeExpertConversionError> {
        if root.exists() {
            return Err(Qwen3MoeExpertConversionError::OutputExists {
                path: root.to_owned(),
            });
        }
        fs::create_dir(root).map_err(|source| {
            io_error("create expert output directory", root.to_owned(), source)
        })?;
        Ok(Self {
            root: root.to_owned(),
            paths: Vec::new(),
            complete: false,
        })
    }

    fn track(&mut self, path: PathBuf) {
        self.paths.push(path);
    }
}

impl Drop for OutputTransaction {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        for path in self.paths.iter().rev() {
            let _ = fs::remove_file(path);
        }
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let _ = fs::remove_dir(&self.root);
    }
}

fn io_error(
    action: &'static str,
    path: PathBuf,
    source: std::io::Error,
) -> Qwen3MoeExpertConversionError {
    Qwen3MoeExpertConversionError::Io {
        action,
        path,
        source,
    }
}
