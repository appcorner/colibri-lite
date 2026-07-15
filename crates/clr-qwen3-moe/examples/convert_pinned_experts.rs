use std::{collections::HashSet, env, fmt::Write as _, fs, path::PathBuf, process};

use clr_core::{DataType, TensorShape};
use clr_qwen3_moe::{
    ExpertProjectionKind, PINNED_QWEN3_30B_A3B_MODEL_ID, PINNED_QWEN3_30B_A3B_REVISION,
    Qwen3MoeExpertConversionScope, Qwen3MoeExpertConversionSpec, Qwen3MoeExpertSourceProjection,
    Qwen3MoeTensorMetadata, convert_pinned_qwen3_moe_experts,
};
use clr_storage::{
    ArtifactReader, DEFAULT_CONVERSION_CHUNK_BYTES, DenseSourceShard, ExpertId, ExpertKey,
    ExpertStore, Sha256Hasher,
};

fn main() {
    match run() {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(2);
        }
    }
}

fn run() -> Result<String, String> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    if arguments.len() != 6 {
        return Err(
            "usage: convert_pinned_experts <vertical-slice|complete> <plan> <source-root> <output-directory> <available-bytes> <layer:expert,...|->"
                .to_owned(),
        );
    }
    let scope = match arguments[0].as_str() {
        "vertical-slice" => Qwen3MoeExpertConversionScope::VerticalSlice,
        "complete" => Qwen3MoeExpertConversionScope::Complete,
        value => return Err(format!("invalid conversion scope '{value}'")),
    };
    let selection = parse_selection(&arguments[5])?;
    if scope == Qwen3MoeExpertConversionScope::Complete && !selection.is_empty() {
        return Err("complete conversion selection must be '-'".to_owned());
    }
    let plan_path = PathBuf::from(&arguments[1]);
    let source_root = PathBuf::from(&arguments[2]);
    let output_directory = PathBuf::from(&arguments[3]);
    let available_space_bytes = arguments[4]
        .parse::<u64>()
        .map_err(|_| format!("invalid available byte count '{}'", arguments[4]))?;
    let plan = fs::read_to_string(&plan_path)
        .map_err(|error| format!("failed to read '{}': {error}", plan_path.display()))?;
    let (source_shards, all_projections) = parse_plan(&plan, &source_root)?;
    let projections = if selection.is_empty() {
        all_projections
    } else {
        all_projections
            .into_iter()
            .filter(|projection| {
                selection.contains(&(projection.layer_index, projection.expert_index))
            })
            .collect()
    };
    let summary = convert_pinned_qwen3_moe_experts(&Qwen3MoeExpertConversionSpec {
        scope,
        source_shards,
        projections,
        output_directory: output_directory.clone(),
        available_space_bytes,
        source_chunk_bytes: DEFAULT_CONVERSION_CHUNK_BYTES,
    })
    .map_err(|error| error.to_string())?;

    let probe = if scope == Qwen3MoeExpertConversionScope::Complete {
        (23_usize, 64_usize)
    } else {
        *selection
            .iter()
            .min()
            .ok_or("vertical selection is empty")?
    };
    let key = ExpertKey {
        layer_index: u32::try_from(probe.0).map_err(|_| "probe layer overflow")?,
        expert_id: ExpertId(u32::try_from(probe.1).map_err(|_| "probe expert overflow")?),
    };
    let record = summary
        .manifest
        .experts()
        .iter()
        .find(|record| record.key == key)
        .ok_or("probe expert missing from converted manifest")?;
    let budget = usize::try_from(record.payload_length).map_err(|_| "probe budget overflow")?;
    let reader = ArtifactReader::open(
        &output_directory,
        summary.manifest.runtime_manifest().clone(),
    )
    .map_err(|error| error.to_string())?;
    let mut store = ExpertStore::new(reader, summary.manifest.registrations().to_vec(), budget)
        .map_err(|error| error.to_string())?;
    let lease = store.load(key).map_err(|error| error.to_string())?;
    let loaded_bytes = lease.bytes().len();
    drop(lease);
    let random_access_bytes = store.metrics().bytes_read;
    let mut shard_set_hasher = Sha256Hasher::new();
    for shard in summary.manifest.shards() {
        shard_set_hasher.update(&shard.sha256);
    }

    Ok(format!(
        "logical_experts={}\nsource_tensors={}\nartifact_shards={}\nsource_verification_bytes={}\nsource_payload_bytes={}\ntotal_source_bytes={}\nartifact_bytes={}\npreflight_bytes={}\npeak_buffer_bytes={}\nrandom_access_bytes={}\nloaded_bytes={}\nmanifest_sha256={}\nshard_set_sha256={}\nmanifest={}",
        summary.manifest.experts().len(),
        summary.manifest.experts().len() * 3,
        summary.manifest.shards().len(),
        summary.source_verification_bytes_read,
        summary.source_payload_bytes_read,
        summary.source_verification_bytes_read + summary.source_payload_bytes_read,
        summary.artifact_bytes_written,
        summary.preflight_required_bytes,
        summary.peak_buffer_bytes,
        random_access_bytes,
        loaded_bytes,
        hex(&summary.manifest_sha256),
        hex(&shard_set_hasher.finalize()),
        summary.manifest_path.display()
    ))
}

fn parse_plan(
    plan: &str,
    source_root: &std::path::Path,
) -> Result<(Vec<DenseSourceShard>, Vec<Qwen3MoeExpertSourceProjection>), String> {
    let mut format_version = None;
    let mut model_id = None;
    let mut revision = None;
    let mut shards = Vec::new();
    let mut projections = Vec::new();
    for (line_index, line) in plan.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        match fields.as_slice() {
            ["format_version", value] if format_version.is_none() => format_version = Some(*value),
            ["model_id", value] if model_id.is_none() => model_id = Some(*value),
            ["revision", value] if revision.is_none() => revision = Some(*value),
            ["shard", index, path, length, hash] => {
                let parsed_index = parse_usize(index, line_index)?;
                if parsed_index != shards.len() {
                    return Err(format!(
                        "line {}: shard index is not sequential",
                        line_index + 1
                    ));
                }
                shards.push(DenseSourceShard {
                    path: source_root.join(path),
                    byte_length: parse_u64(length, line_index)?,
                    sha256: parse_hash(hash, line_index)?,
                });
            }
            [
                "projection",
                layer,
                expert,
                kind,
                name,
                shard,
                offset,
                length,
                dimensions,
            ] => {
                let projection = match *kind {
                    "gate" => ExpertProjectionKind::Gate,
                    "up" => ExpertProjectionKind::Up,
                    "down" => ExpertProjectionKind::Down,
                    value => {
                        return Err(format!(
                            "line {}: invalid projection '{value}'",
                            line_index + 1
                        ));
                    }
                };
                let dimensions = dimensions
                    .split(',')
                    .map(|value| parse_usize(value, line_index))
                    .collect::<Result<Vec<_>, _>>()?;
                let shard_index = parse_usize(shard, line_index)?;
                projections.push(Qwen3MoeExpertSourceProjection {
                    layer_index: parse_usize(layer, line_index)?,
                    expert_index: parse_usize(expert, line_index)?,
                    projection,
                    metadata: Qwen3MoeTensorMetadata::new(
                        *name,
                        DataType::BF16,
                        TensorShape::new(dimensions),
                        shard_index,
                    ),
                    offset: parse_u64(offset, line_index)?,
                    length: parse_u64(length, line_index)?,
                });
            }
            _ => return Err(format!("line {}: invalid plan record", line_index + 1)),
        }
    }
    if format_version != Some("1") {
        return Err("source plan format_version must be 1".to_owned());
    }
    if model_id != Some(PINNED_QWEN3_30B_A3B_MODEL_ID) {
        return Err("source plan model_id does not match pinned model".to_owned());
    }
    if revision != Some(PINNED_QWEN3_30B_A3B_REVISION) {
        return Err("source plan revision does not match pinned revision".to_owned());
    }
    Ok((shards, projections))
}

fn parse_selection(value: &str) -> Result<HashSet<(usize, usize)>, String> {
    if value == "-" {
        return Ok(HashSet::new());
    }
    value
        .split(',')
        .map(|item| {
            let (layer, expert) = item
                .split_once(':')
                .ok_or_else(|| format!("invalid expert selection '{item}'"))?;
            Ok((
                layer
                    .parse()
                    .map_err(|_| format!("invalid layer '{layer}'"))?,
                expert
                    .parse()
                    .map_err(|_| format!("invalid expert '{expert}'"))?,
            ))
        })
        .collect()
}

fn parse_usize(value: &str, line_index: usize) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("line {}: invalid integer '{value}'", line_index + 1))
}

fn parse_u64(value: &str, line_index: usize) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("line {}: invalid integer '{value}'", line_index + 1))
}

fn parse_hash(value: &str, line_index: usize) -> Result<[u8; 32], String> {
    if value.len() != 64 {
        return Err(format!(
            "line {}: SHA-256 must have 64 hex digits",
            line_index + 1
        ));
    }
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| format!("line {}: invalid SHA-256", line_index + 1))?;
    }
    Ok(output)
}

fn hex(hash: &[u8; 32]) -> String {
    hash.iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("write string");
            output
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pinned_projection_plan_and_selection() {
        let plan = format!(
            "format_version\t1\nmodel_id\t{PINNED_QWEN3_30B_A3B_MODEL_ID}\nrevision\t{PINNED_QWEN3_30B_A3B_REVISION}\nshard\t0\tmodel.safetensors\t8\t{}\nprojection\t0\t0\tgate\tmodel.layers.0.mlp.experts.0.gate_proj.weight\t0\t0\t3145728\t768,2048\n",
            "00".repeat(32)
        );
        let (shards, projections) =
            parse_plan(&plan, std::path::Path::new("source")).expect("valid expert plan");

        assert_eq!(shards.len(), 1);
        assert_eq!(projections.len(), 1);
        assert_eq!(projections[0].projection, ExpertProjectionKind::Gate);
        assert!(parse_selection("0:0,47:127").unwrap().contains(&(47, 127)));
    }
}
