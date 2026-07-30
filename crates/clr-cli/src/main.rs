use clr_core::runtime_info;
use clr_qwen3_moe::{GenerationSession, frozen_tiny_model};
use serde_json::{Value, json};
use std::path::Path;

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match execute(&arguments) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    }
}

fn execute(arguments: &[String]) -> Result<String, String> {
    if arguments.is_empty() {
        let info = runtime_info();
        return Ok(format!(
            "{} {}\nhost: {}-{}\nstatus: bootstrap ready",
            info.name, info.version, info.architecture, info.operating_system
        ));
    }
    match arguments[0].as_str() {
        "generate" => {
            let options = GenerateOptions::parse(&arguments[1..])?;
            generate(&options)
        }
        "doctor" => compose_doctor(&ProfileOptions::parse(
            &arguments[1..],
            ProfileKind::Doctor,
        )?),
        "profile-model" => {
            compose_model_profile(&ProfileOptions::parse(&arguments[1..], ProfileKind::Model)?)
        }
        _ => Err(format!("unknown command '{}'", arguments[0])),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileKind {
    Doctor,
    Model,
}

#[derive(Debug, PartialEq)]
struct ProfileOptions {
    inputs: Vec<(String, String)>,
    output: String,
    created_at: String,
    runtime_commit: String,
    rust_version: Option<String>,
}

impl ProfileOptions {
    fn parse(arguments: &[String], kind: ProfileKind) -> Result<Self, String> {
        let expected_inputs: &[&str] = match kind {
            ProfileKind::Doctor => &["--cpu-ram", "--storage", "--memory-gpu"],
            ProfileKind::Model => &["--reference", "--quality", "--release"],
        };
        let mut values = std::collections::BTreeMap::new();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            if values.insert(flag.clone(), value.clone()).is_some() {
                return Err(format!("duplicate option {flag}"));
            }
            index += 2;
        }
        let required = |flag: &str| {
            values
                .get(flag)
                .cloned()
                .ok_or_else(|| format!("missing required {flag}"))
        };
        let mut inputs = Vec::new();
        for flag in expected_inputs {
            inputs.push(((*flag).to_string(), required(flag)?));
        }
        let rust_version = match kind {
            ProfileKind::Doctor => Some(required("--rust-version")?),
            ProfileKind::Model => None,
        };
        let mut allowed: Vec<&str> = expected_inputs
            .iter()
            .copied()
            .chain(["--output", "--created-at", "--runtime-commit"])
            .collect();
        if kind == ProfileKind::Doctor {
            allowed.push("--rust-version");
        }
        if let Some(flag) = values.keys().find(|flag| !allowed.contains(&flag.as_str())) {
            return Err(format!("unknown option {flag}"));
        }
        Ok(Self {
            inputs,
            output: required("--output")?,
            created_at: required("--created-at")?,
            runtime_commit: required("--runtime-commit")?,
            rust_version,
        })
    }

    fn input(&self, flag: &str) -> Result<&str, String> {
        self.inputs
            .iter()
            .find_map(|(name, value)| (name == flag).then_some(value.as_str()))
            .ok_or_else(|| format!("missing input {flag}"))
    }
}

#[derive(Debug, PartialEq)]
struct GenerateOptions {
    tokens: Vec<usize>,
    max_new_tokens: usize,
    temperature: Option<f32>,
    seed: u64,
}

impl GenerateOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut tokens = None;
        let mut max_new_tokens = None;
        let mut temperature = None;
        let mut seed = None;
        let mut index = 0;
        while index < arguments.len() {
            let flag = arguments[index].as_str();
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag {
                "--tokens" if tokens.is_none() => tokens = Some(parse_tokens(value)?),
                "--max-new-tokens" if max_new_tokens.is_none() => {
                    max_new_tokens = Some(parse_value(value, flag)?);
                }
                "--temperature" if temperature.is_none() => {
                    temperature = Some(parse_value(value, flag)?);
                }
                "--seed" if seed.is_none() => seed = Some(parse_value(value, flag)?),
                "--tokens" | "--max-new-tokens" | "--temperature" | "--seed" => {
                    return Err(format!("duplicate option {flag}"));
                }
                _ => return Err(format!("unknown option {flag}")),
            }
            index += 2;
        }
        Ok(Self {
            tokens: tokens.ok_or_else(|| "missing required --tokens".to_string())?,
            max_new_tokens: max_new_tokens
                .ok_or_else(|| "missing required --max-new-tokens".to_string())?,
            temperature,
            seed: seed.unwrap_or(0),
        })
    }
}

fn parse_tokens(value: &str) -> Result<Vec<usize>, String> {
    if value.is_empty() {
        return Err("--tokens must contain at least one token ID".to_string());
    }
    value
        .split(',')
        .map(|token| {
            token
                .parse::<usize>()
                .map_err(|_| format!("invalid token ID '{token}'"))
        })
        .collect()
}

fn parse_value<T>(value: &str, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("invalid value '{value}' for {flag}"))
}

fn generate(options: &GenerateOptions) -> Result<String, String> {
    let capacity = options
        .tokens
        .len()
        .checked_add(options.max_new_tokens)
        .ok_or_else(|| "requested context length overflowed".to_string())?;
    let model = frozen_tiny_model().map_err(|error| error.to_string())?;
    let mut session = GenerationSession::resident(&model, capacity, options.seed)
        .map_err(|error| error.to_string())?;
    session
        .prefill(&options.tokens)
        .map_err(|error| error.to_string())?;
    let mut generated = Vec::with_capacity(options.max_new_tokens);
    for _ in 0..options.max_new_tokens {
        let token = match options.temperature {
            Some(temperature) => session
                .decode_temperature(temperature)
                .map_err(|error| error.to_string())?,
            None => session.decode_greedy().map_err(|error| error.to_string())?,
        };
        generated.push(token);
    }
    Ok(format!(
        "generated: {}\nsequence: {}\nkv-cache: {} bytes, {}/{} tokens",
        join_tokens(&generated),
        join_tokens(session.sequence()),
        session.cache().byte_size(),
        session.cache().len(),
        session.cache().capacity()
    ))
}

fn join_tokens(tokens: &[usize]) -> String {
    tokens
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn compose_doctor(options: &ProfileOptions) -> Result<String, String> {
    let cpu = read_json(options.input("--cpu-ram")?)?;
    let storage = read_json(options.input("--storage")?)?;
    let memory = read_json(options.input("--memory-gpu")?)?;
    let cpu_distribution = distribution(&cpu, "/results/cpu_kernel_gflops")?;
    let ram_distribution = distribution(&cpu, "/results/ram_copy_gib_per_second")?;
    let sequential_distribution =
        distribution(&storage, "/results/sequential_read/warm_throughput")?;
    let random_distribution = distribution(
        &storage,
        "/results/expert_sized_random_read/warm_throughput",
    )?;
    let cpu_semantics = value(&cpu, "/measurement_semantics/cpu_kernel")?;
    let ram_semantics = value(&cpu, "/measurement_semantics/ram_copy")?;
    let storage_path = string(&storage, "/storage_target/path")?;
    let logical_cores = number(&cpu, "/host/logical_core_count")?;
    let document = json!({
        "schema": "colibri-lite-hardware-profile-v1",
        "schema_version": 1,
        "profile_id": format!("{}-doctor", string(&memory, "/profile_id")?),
        "created_at": options.created_at,
        "runtime": {"name": "colibri-lite-rs", "commit": options.runtime_commit, "rust_version": options.rust_version.as_deref().unwrap_or_default(), "target": "x86_64-pc-windows-msvc", "build_profile": "release"},
        "host": {
            "operating_system": {"name": string(&cpu, "/host/os_name")?, "version": string(&cpu, "/host/os_version")?},
            "cpu": {"model": string(&cpu, "/host/cpu_model")?, "logical_core_count": logical_cores},
            "ram": {"total_bytes": number(&memory, "/ram/total_physical_bytes")?, "safe_budget_bytes": number(&memory, "/ram/usable_budget_bytes")?, "safety_reserve_bytes": number(&memory, "/ram/safety_reserve_bytes")?}
        },
        "measurements": {
            "cpu_kernels": [{"kernel_id": string(cpu_semantics, "/kernel_id")?, "benchmark": measured("GFLOP/s", "not_applicable", number(cpu_semantics, "/matrix_rows")? * number(cpu_semantics, "/matrix_columns")? * 4, number(cpu_semantics, "/sample_count")?, &cpu_distribution)}],
            "ram_bandwidth": measured("GiB/s", "not_applicable", number(ram_semantics, "/payload_bytes")?, number(ram_semantics, "/sample_count")?, &ram_distribution),
            "storage": [{"target_id": "m6.1-03-primary", "path": storage_path, "sequential_read": measured("GiB/s", "warm", number(&storage, "/results/sequential_read/payload_bytes")?, number(&storage, "/results/sequential_read/repetitions")?, &sequential_distribution), "expert_sized_random_read": measured("MiB/s", "unknown", number(&storage, "/results/expert_sized_random_read/payload_bytes")?, number(&storage, "/results/expert_sized_random_read/repetitions_per_cache_state")?, &random_distribution)}],
            "backends": backends(&memory)?
        },
        "recommendations": {"ram_budget_bytes": number(&memory, "/recommendations/ram_budget_bytes")?, "vram_budget_bytes": 0, "confidence": "partial"},
        "limitations": [
            "Composed from the explicit M6.1-02, M6.1-03, and M6.1-04 evidence inputs.",
            "Storage first-touch readings do not claim cold-device semantics; the schema exposes the likely-warm throughput distribution.",
            "No colibri GPU backend is usable before M6.3 review, so VRAM and transfer budgets remain zero."
        ]
    });
    write_json(&options.output, &document)?;
    Ok(format!("wrote {}", options.output))
}

fn compose_model_profile(options: &ProfileOptions) -> Result<String, String> {
    let reference = read_json(options.input("--reference")?)?;
    let quality = read_json(options.input("--quality")?)?;
    let release = read_json(options.input("--release")?)?;
    let dimensions = value(&release, "/canonical_artifact/dimensions")?;
    let artifact = value(&release, "/canonical_artifact")?;
    let expert_bytes = number(artifact, "/experts/bytes")?;
    let layers = number(dimensions, "/layers")?;
    let experts = number(dimensions, "/experts")?;
    let per_expert_bytes = 18_874_368_u64;
    let per_layer_bytes = per_expert_bytes * experts;
    let per_layer_payload_bytes = vec![
        per_layer_bytes;
        usize::try_from(layers)
            .map_err(|_| "layer count does not fit usize")?
    ];
    let document = json!({
        "schema": "colibri-lite-model-profile-v1",
        "schema_version": 1,
        "profile_id": "qwen3-30b-a3b-f32-reference-v1",
        "created_at": options.created_at,
        "runtime": {"name": "colibri-lite-rs", "commit": options.runtime_commit, "target": "x86_64-pc-windows-msvc"},
        "model": {
            "id": string(&reference, "/model/model_id")?, "revision": string(&reference, "/model/revision")?, "architecture": string(&reference, "/model/architecture")?,
            "layers": layers, "hidden_size": number(dimensions, "/hidden_size")?, "vocabulary_size": number(dimensions, "/vocabulary_size")?,
            "routing": {"expert_count": experts, "experts_per_token": number(dimensions, "/experts_per_token")?}
        },
        "artifact": {
            "format": string(artifact, "/format")?, "format_version": number(artifact, "/version")?, "root_manifest_sha256": string(artifact, "/root_manifest_sha256")?, "storage_dtype": string(artifact, "/dtype/storage")?,
            "dense": {"bytes": number(artifact, "/dense/bytes")?, "tensor_count": number(artifact, "/dense/tensor_count")?},
            "experts": {"bytes": expert_bytes, "tensor_count": number(artifact, "/experts/source_tensor_count")?, "logical_expert_count": number(artifact, "/experts/logical_expert_count")?, "shard_count": number(artifact, "/experts/shard_count")?, "per_layer_payload_bytes": per_layer_payload_bytes}
        },
        "execution": {
            "kv_cache": {"bytes_per_token": 196_608, "context_lengths": [128, 512, 2048, 8192]},
            "precision_candidates": [{"candidate_id": "reference-f32-v1", "status": "accepted_reference", "weight_dtype": "F32", "evidence": "reference-f32-v1 frozen manifest"}],
            "estimated_bytes_per_routed_token": per_expert_bytes * layers * number(dimensions, "/experts_per_token")?
        },
        "quality_reference": {
            "reference_id": "reference-f32-v1",
            "manifest_sha256": integrity_hash(&quality, "reference_f32_manifest")?,
            "bilingual_fixture_manifest_sha256": file_sha256(options.input("--quality")?)?
        },
        "limitations": [
            "This profile describes the frozen F32 reference artifact only.",
            "No quantized format or GPU backend is authorized by this profile.",
            "Estimated routed-token bytes are logical expert payload demand before cache reuse."
        ]
    });
    write_json(&options.output, &document)?;
    Ok(format!("wrote {}", options.output))
}

fn read_json(path: &str) -> Result<Value, String> {
    let source =
        std::fs::read_to_string(path).map_err(|error| format!("failed to read {path}: {error}"))?;
    serde_json::from_str(&source).map_err(|error| format!("invalid JSON in {path}: {error}"))
}
fn write_json(path: &str, document: &Value) -> Result<(), String> {
    if let Some(parent) = Path::new(path)
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let output = serde_json::to_string_pretty(document).map_err(|error| error.to_string())?;
    std::fs::write(path, format!("{output}\n"))
        .map_err(|error| format!("failed to write {path}: {error}"))
}
fn value<'a>(root: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    root.pointer(pointer)
        .ok_or_else(|| format!("missing {pointer}"))
}
fn string<'a>(root: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value(root, pointer)?
        .as_str()
        .ok_or_else(|| format!("{pointer} must be a string"))
}
fn number(root: &Value, pointer: &str) -> Result<u64, String> {
    value(root, pointer)?
        .as_u64()
        .ok_or_else(|| format!("{pointer} must be an unsigned integer"))
}
fn distribution(root: &Value, pointer: &str) -> Result<Value, String> {
    let source = value(root, pointer)?;
    let sample_count = source
        .pointer("/samples")
        .and_then(Value::as_array)
        .map(|values| values.len() as u64)
        .ok_or_else(|| format!("{pointer}/samples missing"))?;
    Ok(
        json!({"sample_count": sample_count, "median": value(source, "/median")?, "p10": value(source, "/p10")?, "p90": value(source, "/p90")?}),
    )
}
fn measured(
    unit: &str,
    cache_state: &str,
    payload_bytes: u64,
    repetitions: u64,
    distribution: &Value,
) -> Value {
    json!({"status": "measured", "unit": unit, "cache_state": cache_state, "payload_bytes": payload_bytes, "repetitions": repetitions, "distribution": distribution})
}
fn backends(memory: &Value) -> Result<Vec<Value>, String> {
    value(memory, "/backends")?.as_array().ok_or_else(|| "/backends must be an array".to_string())?.iter().map(|backend| {
        Ok(json!({"backend_id": string(backend, "/backend_id")?, "availability": string(backend, "/availability")?, "device_name": "", "vram": {"total_bytes": 0, "safe_budget_bytes": 0}, "compute": {"status":"not_run","unit":"GFLOP/s","cache_state":"not_applicable","payload_bytes":0,"repetitions":0,"reason":"No usable colibri backend in M6.1."}, "host_to_device": value(backend, "/host_to_device")?, "device_to_host": value(backend, "/device_to_host")?}))
    }).collect()
}
fn integrity_hash<'a>(quality: &'a Value, role: &str) -> Result<&'a str, String> {
    value(quality, "/integrity_records")?
        .as_array()
        .and_then(|records| {
            records
                .iter()
                .find(|record| record.get("role").and_then(Value::as_str) == Some(role))
        })
        .and_then(|record| record.get("sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing integrity hash for {role}"))
}
fn file_sha256(path: &str) -> Result<String, String> {
    let output = std::process::Command::new("certutil")
        .args(["-hashfile", path, "SHA256"])
        .output()
        .map_err(|error| format!("failed to run certutil: {error}"))?;
    if !output.status.success() {
        return Err("certutil SHA256 failed".to_string());
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| {
            let hash = line.replace(' ', "");
            (hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
                .then(|| hash.to_ascii_lowercase())
        })
        .ok_or_else(|| "certutil did not emit a SHA256 value".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn no_arguments_preserves_bootstrap_smoke_output() {
        let output = execute(&[]).expect("bootstrap output");
        assert!(output.contains("colibri-lite-rs"));
        assert!(output.contains("status: bootstrap ready"));
    }

    #[test]
    fn generate_accepts_token_ids_and_emits_cache_accounting() {
        let output = execute(&arguments(&[
            "generate",
            "--tokens",
            "1,7,3,12",
            "--max-new-tokens",
            "2",
        ]))
        .expect("greedy generation");

        assert!(output.contains("generated: 10"));
        assert!(output.contains("sequence: 1,7,3,12"));
        assert!(output.contains("6/6 tokens"));
    }

    #[test]
    fn generate_rejects_missing_invalid_and_duplicate_options() {
        assert_eq!(
            execute(&arguments(&["generate", "--max-new-tokens", "1"])),
            Err("missing required --tokens".to_string())
        );
        assert_eq!(
            execute(&arguments(&[
                "generate",
                "--tokens",
                "1,nope",
                "--max-new-tokens",
                "1",
            ])),
            Err("invalid token ID 'nope'".to_string())
        );
        assert_eq!(
            execute(&arguments(&[
                "generate",
                "--tokens",
                "1",
                "--tokens",
                "2",
                "--max-new-tokens",
                "1",
            ])),
            Err("duplicate option --tokens".to_string())
        );
    }

    #[test]
    fn profile_commands_require_explicit_reproducibility_inputs() {
        assert_eq!(
            ProfileOptions::parse(
                &arguments(&[
                    "--cpu-ram",
                    "cpu.json",
                    "--storage",
                    "storage.json",
                    "--memory-gpu",
                    "gpu.json",
                    "--output",
                    "doctor.json",
                    "--created-at",
                    "2026-07-30T00:00:00Z",
                    "--runtime-commit",
                    "abcdef0",
                ]),
                ProfileKind::Doctor,
            ),
            Err("missing required --rust-version".to_string())
        );
        assert_eq!(
            ProfileOptions::parse(
                &arguments(&[
                    "--reference",
                    "reference.json",
                    "--quality",
                    "quality.json",
                    "--release",
                    "release.json",
                    "--output",
                    "model.json",
                    "--created-at",
                    "2026-07-30T00:00:00Z",
                    "--runtime-commit",
                    "abcdef0",
                    "--rust-version",
                    "value",
                ]),
                ProfileKind::Model,
            ),
            Err("unknown option --rust-version".to_string())
        );
    }

    #[test]
    fn profile_json_helpers_release_windows_file_handles() {
        let unique = format!(
            "clr-cli-profile-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir(&directory).expect("temporary directory");
        let input = directory.join("input.json");
        let renamed_input = directory.join("input-renamed.json");
        let output = directory.join("output.json");
        let renamed_output = directory.join("output-renamed.json");
        std::fs::write(&input, "{\"value\": 1}\n").expect("input");

        let parsed = read_json(input.to_str().expect("UTF-8 path")).expect("read JSON");
        assert_eq!(number(&parsed, "/value"), Ok(1));
        std::fs::rename(&input, &renamed_input).expect("input handle released");
        write_json(output.to_str().expect("UTF-8 path"), &json!({"ok": true})).expect("write JSON");
        std::fs::rename(&output, &renamed_output).expect("output handle released");

        std::fs::remove_dir_all(&directory).expect("temporary directory cleanup");
    }
}
