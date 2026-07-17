use std::{env, fmt::Write as _, fs, path::PathBuf, process};

use clr_core::{DataType, TensorShape};
use clr_qwen3_moe::{
    PINNED_QWEN3_30B_A3B_MODEL_ID, PINNED_QWEN3_30B_A3B_REVISION, Qwen3MoeDenseConversionScope,
    Qwen3MoeDenseConversionSpec, Qwen3MoeDenseSourceTensor, Qwen3MoeTensorMetadata,
    convert_pinned_qwen3_moe_dense_tensors,
};
use clr_storage::{DEFAULT_CONVERSION_CHUNK_BYTES, DenseSourceShard};

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
    if arguments.len() != 5 {
        return Err(
            "usage: convert_pinned_dense <vertical-slice|complete> <plan> <source-root> <output-directory> <available-bytes>"
                .to_owned(),
        );
    }
    let scope = match arguments[0].as_str() {
        "vertical-slice" => Qwen3MoeDenseConversionScope::VerticalSlice,
        "complete" => Qwen3MoeDenseConversionScope::Complete,
        value => return Err(format!("invalid conversion scope '{value}'")),
    };
    let plan_path = PathBuf::from(&arguments[1]);
    let source_root = PathBuf::from(&arguments[2]);
    let output_directory = PathBuf::from(&arguments[3]);
    let available_space_bytes = arguments[4]
        .parse::<u64>()
        .map_err(|_| format!("invalid available byte count '{}'", arguments[4]))?;
    let plan = fs::read_to_string(&plan_path)
        .map_err(|error| format!("failed to read '{}': {error}", plan_path.display()))?;
    let (source_shards, tensors) = parse_plan(&plan, &source_root)?;
    let summary = convert_pinned_qwen3_moe_dense_tensors(&Qwen3MoeDenseConversionSpec {
        scope,
        source_shards,
        tensors,
        output_directory,
        available_space_bytes,
        source_chunk_bytes: DEFAULT_CONVERSION_CHUNK_BYTES,
    })
    .map_err(|error| error.to_string())?;

    Ok(format!(
        "tensors={}\nsource_verification_bytes={}\nsource_payload_bytes={}\nartifact_bytes={}\npreflight_bytes={}\npeak_buffer_bytes={}\npayload_sha256={}\nmanifest_sha256={}\npayload={}\nmanifest={}",
        summary.artifact_manifest.tensors().len(),
        summary.source_verification_bytes_read,
        summary.source_payload_bytes_read,
        summary.artifact_bytes_written,
        summary.preflight_required_bytes,
        summary.peak_buffer_bytes,
        hex(&summary.payload_sha256),
        hex(&summary.manifest_sha256),
        summary.payload_path.display(),
        summary.manifest_path.display()
    ))
}

fn parse_plan(
    plan: &str,
    source_root: &std::path::Path,
) -> Result<(Vec<DenseSourceShard>, Vec<Qwen3MoeDenseSourceTensor>), String> {
    let mut format_version = None;
    let mut model_id = None;
    let mut revision = None;
    let mut shards = Vec::new();
    let mut tensors = Vec::new();
    for (line_index, line) in plan.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        match fields.as_slice() {
            ["format_version", value] if format_version.is_none() => {
                format_version = Some(*value);
            }
            ["model_id", value] if model_id.is_none() => model_id = Some(*value),
            ["revision", value] if revision.is_none() => revision = Some(*value),
            ["shard", index, path, length, hash] => {
                let parsed_index = parse_usize(index, line_index)?;
                if parsed_index != shards.len() {
                    return Err(format!(
                        "line {}: shard index {parsed_index} is not sequential",
                        line_index + 1
                    ));
                }
                shards.push(DenseSourceShard {
                    path: source_root.join(path),
                    byte_length: parse_u64(length, line_index)?,
                    sha256: parse_hash(hash, line_index)?,
                });
            }
            ["tensor", name, shard, offset, length, dimensions] => {
                let dimensions = dimensions
                    .split(',')
                    .map(|value| parse_usize(value, line_index))
                    .collect::<Result<Vec<_>, _>>()?;
                tensors.push(Qwen3MoeDenseSourceTensor {
                    metadata: Qwen3MoeTensorMetadata::new(
                        *name,
                        DataType::BF16,
                        TensorShape::new(dimensions),
                        parse_usize(shard, line_index)?,
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
        return Err("source plan model_id does not match the pinned model".to_owned());
    }
    if revision != Some(PINNED_QWEN3_30B_A3B_REVISION) {
        return Err("source plan revision does not match the pinned revision".to_owned());
    }
    Ok((shards, tensors))
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
            write!(&mut output, "{byte:02x}").expect("write to string");
            output
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_versioned_pinned_plan() {
        let plan = format!(
            "format_version\t1\nmodel_id\t{PINNED_QWEN3_30B_A3B_MODEL_ID}\nrevision\t{PINNED_QWEN3_30B_A3B_REVISION}\nshard\t0\tmodel.safetensors\t8\t{}\ntensor\tmodel.norm.weight\t0\t0\t4096\t2048\n",
            "00".repeat(32)
        );

        let (shards, tensors) =
            parse_plan(&plan, std::path::Path::new("source")).expect("valid source plan");

        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].path, PathBuf::from("source/model.safetensors"));
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].metadata.shape().dimensions(), [2_048]);
    }
}
