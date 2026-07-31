use clr_core::{
    AnalyticalCostModel, BudgetAdmission, CandidateResourceRequirements, MeasuredRate,
    PlacementCapabilities, PlacementTier, PlannerBudgets, PlannerRejectionCode, PlannerWorkload,
    ProfileMeasurementStatus, TokenWork, admit_candidate, runtime_info,
};
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
        "plan" => plan(&PlanOptions::parse(&arguments[1..])?),
        _ => Err(format!("unknown command '{}'", arguments[0])),
    }
}

#[derive(Debug, PartialEq)]
struct PlanOptions {
    hardware_profile: String,
    model_profile: String,
    ram_budget_bytes: u64,
    vram_budget_bytes: u64,
    context_tokens: u64,
    prefill_tokens: u64,
    decode_tokens: u64,
    compute_gflop_per_token: f64,
    request_id: String,
    result_id: String,
    output: String,
}

impl PlanOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
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
        let allowed = [
            "--hardware-profile",
            "--model-profile",
            "--ram-budget-bytes",
            "--vram-budget-bytes",
            "--context-tokens",
            "--prefill-tokens",
            "--decode-tokens",
            "--compute-gflop-per-token",
            "--request-id",
            "--result-id",
            "--output",
        ];
        if let Some(flag) = values.keys().find(|flag| !allowed.contains(&flag.as_str())) {
            return Err(format!("unknown option {flag}"));
        }
        let compute_gflop_per_token: f64 = parse_value(
            &required("--compute-gflop-per-token")?,
            "--compute-gflop-per-token",
        )?;
        if !compute_gflop_per_token.is_finite() || compute_gflop_per_token <= 0.0 {
            return Err(
                "--compute-gflop-per-token must be finite and greater than zero".to_string(),
            );
        }
        Ok(Self {
            hardware_profile: required("--hardware-profile")?,
            model_profile: required("--model-profile")?,
            ram_budget_bytes: parse_value(&required("--ram-budget-bytes")?, "--ram-budget-bytes")?,
            vram_budget_bytes: parse_value(
                &required("--vram-budget-bytes")?,
                "--vram-budget-bytes",
            )?,
            context_tokens: parse_value(&required("--context-tokens")?, "--context-tokens")?,
            prefill_tokens: parse_value(&required("--prefill-tokens")?, "--prefill-tokens")?,
            decode_tokens: parse_value(&required("--decode-tokens")?, "--decode-tokens")?,
            compute_gflop_per_token,
            request_id: required("--request-id")?,
            result_id: required("--result-id")?,
            output: required("--output")?,
        })
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

fn plan(options: &PlanOptions) -> Result<String, String> {
    let hardware = read_json(&options.hardware_profile)?;
    let model = read_json(&options.model_profile)?;
    let hardware_hash = file_sha256(&options.hardware_profile)?;
    let model_hash = file_sha256(&options.model_profile)?;
    let document = plan_document(options, &hardware, &model, &hardware_hash, &model_hash)?;
    write_json(&options.output, &document)?;
    Ok(format!("wrote {}", options.output))
}

#[allow(clippy::too_many_lines)] // Keeps profile validation and emitted provenance at one CLI boundary.
fn plan_document(
    options: &PlanOptions,
    hardware: &Value,
    model: &Value,
    hardware_hash: &str,
    model_hash: &str,
) -> Result<Value, String> {
    require_schema(hardware, "colibri-lite-hardware-profile-v1")?;
    require_schema(model, "colibri-lite-model-profile-v1")?;
    let hardware_profile_id = string(hardware, "/profile_id")?;
    let model_profile_id = string(model, "/profile_id")?;
    let cpu_benchmark = value(hardware, "/measurements/cpu_kernels/0/benchmark")?;
    let ram_benchmark = value(hardware, "/measurements/ram_bandwidth")?;
    let storage_benchmark = value(hardware, "/measurements/storage/0/expert_sized_random_read")?;
    let compute = measured_rate(cpu_benchmark, "measurements.cpu_kernels[0].benchmark")?;
    let ram = measured_rate(ram_benchmark, "measurements.ram_bandwidth")?;
    let storage = measured_rate(
        storage_benchmark,
        "measurements.storage[0].expert_sized_random_read",
    )?;
    let capabilities = PlacementCapabilities::new(
        "cpu",
        profile_status(cpu_benchmark)?,
        ProfileMeasurementStatus::Unavailable,
        profile_status(ram_benchmark)?,
        ProfileMeasurementStatus::Unavailable,
        profile_status(storage_benchmark)?,
        ProfileMeasurementStatus::NotRun,
    )
    .map_err(|error| error.to_string())?;
    let cost_model = AnalyticalCostModel::new(compute, ram, storage);
    let workload = PlannerWorkload::new(options.prefill_tokens, options.decode_tokens)
        .map_err(|error| error.to_string())?;
    let budgets = PlannerBudgets::new(
        options.ram_budget_bytes,
        options.vram_budget_bytes,
        options.context_tokens,
    );
    let dense_bytes = number(model, "/artifact/dense/bytes")?;
    let expert_bytes = number(model, "/artifact/experts/bytes")?;
    let kv_bytes_per_token = number(model, "/execution/kv_cache/bytes_per_token")?;
    let routed_bytes_per_token = number(model, "/execution/estimated_bytes_per_routed_token")?;
    let max_context_tokens = value(model, "/execution/kv_cache/context_lengths")?
        .as_array()
        .ok_or_else(|| "/execution/kv_cache/context_lengths must be an array".to_string())?
        .iter()
        .filter_map(Value::as_u64)
        .max()
        .ok_or_else(|| {
            "/execution/kv_cache/context_lengths must contain a token length".to_string()
        })?;
    let precision_candidate_id = accepted_reference_precision(model)?;
    let quality_reference = string(model, "/quality_reference/reference_id")?;
    let kv_bytes = kv_bytes_per_token
        .checked_mul(options.context_tokens)
        .ok_or_else(|| "KV-cache byte requirement overflowed".to_string())?;
    let resident_base_bytes = dense_bytes
        .checked_add(kv_bytes)
        .ok_or_else(|| "resident RAM requirement overflowed".to_string())?;
    let request = json!({
        "schema": "colibri-lite-planner-request-v1",
        "schema_version": 1,
        "request_id": options.request_id,
        "hardware_profile": {"profile_id": hardware_profile_id, "document_sha256": hardware_hash},
        "model_profile": {"profile_id": model_profile_id, "document_sha256": model_hash},
        "workload": {"workload_id": "explicit-cli-workload-v1", "prefill_tokens": options.prefill_tokens, "decode_tokens": options.decode_tokens, "context_tokens": options.context_tokens},
        "budgets": {"ram_budget_bytes": options.ram_budget_bytes, "vram_budget_bytes": options.vram_budget_bytes}
    });
    let mut feasible = Vec::new();
    let mut rejections = Vec::new();
    for candidate in capabilities.enumerate_supported() {
        let experts_in_ram = candidate.expert_location() == PlacementTier::Ram;
        let expert_cache_bytes = if experts_in_ram { expert_bytes } else { 0 };
        let ram_bytes = resident_base_bytes
            .checked_add(expert_cache_bytes)
            .ok_or_else(|| "candidate RAM requirement overflowed".to_string())?;
        let disk_bytes_per_token = if candidate.expert_location() == PlacementTier::Ssd {
            routed_bytes_per_token
        } else {
            0
        };
        let startup_storage_bytes = dense_bytes
            .checked_add(expert_cache_bytes)
            .ok_or_else(|| "candidate startup storage requirement overflowed".to_string())?;
        let work = TokenWork::new(
            options.compute_gflop_per_token,
            routed_bytes_per_token,
            disk_bytes_per_token,
        )
        .map_err(|error| error.to_string())?;
        let estimate = cost_model
            .estimate(workload, work, work, startup_storage_bytes)
            .map_err(|error| error.to_string())?;
        let requirements = CandidateResourceRequirements::new(ram_bytes, 0, max_context_tokens);
        match admit_candidate(&candidate, requirements, budgets) {
            BudgetAdmission::Admitted => feasible.push(json!({
                "plan_id": candidate.plan_id(),
                "placement": {"backend_id": candidate.backend_id(), "dense_location": tier_name(candidate.dense_location()), "expert_location": tier_name(candidate.expert_location())},
                "precision_candidate_id": precision_candidate_id,
                "resources": {"ram_bytes": ram_bytes, "vram_bytes": 0, "expert_cache_bytes": expert_cache_bytes, "max_context_tokens": max_context_tokens, "disk_bytes_per_token": disk_bytes_per_token, "startup_seconds": estimate.startup_seconds()},
                "estimate": {"status": "available", "method": "analytical-v1", "confidence": "measured_inputs", "prefill_tokens_per_second": 1.0 / estimate.prefill_per_token().total_seconds(), "decode_tokens_per_second": estimate.decode_tokens_per_second(), "measurement_references": [
                    {"profile_kind": "hardware", "profile_id": hardware_profile_id, "document_sha256": hardware_hash, "measurement_id": "measurements.cpu_kernels[0].benchmark"},
                    {"profile_kind": "hardware", "profile_id": hardware_profile_id, "document_sha256": hardware_hash, "measurement_id": "measurements.ram_bandwidth"},
                    {"profile_kind": "hardware", "profile_id": hardware_profile_id, "document_sha256": hardware_hash, "measurement_id": "measurements.storage[0].expert_sized_random_read"}
                ]},
                "quality_risk": {"level": "none", "reference_id": quality_reference, "evidence": "accepted_reference precision candidate"}
            })),
            BudgetAdmission::Rejected(items) => {
                rejections.extend(items.iter().map(rejection_value));
            }
        }
    }
    rank_candidate_plans(&mut feasible)?;
    let ranking: Vec<Value> = feasible
        .iter()
        .map(|candidate| value(candidate, "/plan_id").cloned())
        .collect::<Result<_, _>>()?;
    Ok(json!({
        "schema": "colibri-lite-planner-result-v1",
        "schema_version": 1,
        "result_id": options.result_id,
        "request": request,
        "candidate_plans": feasible,
        "rejections": rejections,
        "ranking": ranking
    }))
}

fn require_schema(document: &Value, expected: &str) -> Result<(), String> {
    if string(document, "/schema")? != expected || number(document, "/schema_version")? != 1 {
        return Err(format!("expected {expected} schema version 1"));
    }
    Ok(())
}

fn profile_status(benchmark: &Value) -> Result<ProfileMeasurementStatus, String> {
    match string(benchmark, "/status")? {
        "measured" => Ok(ProfileMeasurementStatus::Measured),
        "unavailable" => Ok(ProfileMeasurementStatus::Unavailable),
        "not_run" => Ok(ProfileMeasurementStatus::NotRun),
        status => Err(format!("unsupported profile measurement status '{status}'")),
    }
}

fn measured_rate(benchmark: &Value, measurement_id: &str) -> Result<MeasuredRate, String> {
    if profile_status(benchmark)? != ProfileMeasurementStatus::Measured {
        return Err(format!("{measurement_id} is not measured"));
    }
    let median = decimal_number(benchmark, "/distribution/median")?;
    MeasuredRate::new(median, measurement_id).map_err(|error| error.to_string())
}

fn accepted_reference_precision(model: &Value) -> Result<&str, String> {
    value(model, "/execution/precision_candidates")?
        .as_array()
        .and_then(|candidates| {
            candidates.iter().find_map(|candidate| {
                (candidate.pointer("/status").and_then(Value::as_str) == Some("accepted_reference"))
                    .then(|| candidate.pointer("/candidate_id").and_then(Value::as_str))
                    .flatten()
            })
        })
        .ok_or_else(|| "missing accepted reference precision candidate".to_string())
}

fn tier_name(tier: PlacementTier) -> &'static str {
    match tier {
        PlacementTier::Ram => "ram",
        PlacementTier::Vram => "vram",
        PlacementTier::Ssd => "ssd",
    }
}

fn rejection_value(rejection: &clr_core::PlannerRejection) -> Value {
    let (code, reason, unit) = match rejection.code() {
        PlannerRejectionCode::RamBudgetExceeded => (
            "ram_budget_exceeded",
            "required RAM exceeds the requested RAM budget",
            "bytes",
        ),
        PlannerRejectionCode::VramBudgetExceeded => (
            "vram_budget_exceeded",
            "required VRAM exceeds the requested VRAM budget",
            "bytes",
        ),
        PlannerRejectionCode::ContextLimitExceeded => (
            "context_limit_exceeded",
            "requested context exceeds candidate capacity",
            "tokens",
        ),
    };
    json!({"plan_id": rejection.plan_id(), "code": code, "reason": reason, "required": rejection.required(), "limit": rejection.limit(), "unit": unit})
}

fn rank_candidate_plans(candidates: &mut [Value]) -> Result<(), String> {
    for candidate in &*candidates {
        let _ = decimal_number(candidate, "/estimate/decode_tokens_per_second")?;
        let _ = decimal_number(candidate, "/resources/startup_seconds")?;
        let _ = string(candidate, "/plan_id")?;
        match candidate
            .pointer("/quality_risk/level")
            .and_then(Value::as_str)
        {
            Some("none" | "unvalidated" | "known_degradation") => {}
            _ => return Err("candidate plan has an invalid quality risk level".to_string()),
        }
    }
    candidates.sort_by(|left, right| {
        let left_decode = decimal_number(left, "/estimate/decode_tokens_per_second")
            .expect("ranking candidates are validated before sorting");
        let right_decode = decimal_number(right, "/estimate/decode_tokens_per_second")
            .expect("ranking candidates are validated before sorting");
        right_decode
            .total_cmp(&left_decode)
            .then_with(|| quality_risk_rank(left).cmp(&quality_risk_rank(right)))
            .then_with(|| {
                decimal_number(left, "/resources/startup_seconds")
                    .expect("ranking candidates are validated before sorting")
                    .total_cmp(
                        &decimal_number(right, "/resources/startup_seconds")
                            .expect("ranking candidates are validated before sorting"),
                    )
            })
            .then_with(|| {
                string(left, "/plan_id")
                    .expect("ranking candidates are validated before sorting")
                    .cmp(
                        string(right, "/plan_id")
                            .expect("ranking candidates are validated before sorting"),
                    )
            })
    });
    Ok(())
}

fn quality_risk_rank(candidate: &Value) -> u8 {
    match candidate
        .pointer("/quality_risk/level")
        .and_then(Value::as_str)
    {
        Some("none") => 0,
        Some("unvalidated") => 1,
        Some("known_degradation") => 2,
        _ => u8::MAX,
    }
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
fn decimal_number(root: &Value, pointer: &str) -> Result<f64, String> {
    let number = value(root, pointer)?
        .as_f64()
        .ok_or_else(|| format!("{pointer} must be a finite number"))?;
    if !number.is_finite() {
        return Err(format!("{pointer} must be a finite number"));
    }
    Ok(number)
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

    #[test]
    fn plan_is_reproducible_and_ranks_the_feasible_ssd_candidate() {
        let (hardware, model) = planner_profiles();
        let options = planner_options(planner_base_ram_bytes(&model));
        let hardware_hash = "a".repeat(64);
        let model_hash = "b".repeat(64);

        let first = plan_document(&options, &hardware, &model, &hardware_hash, &model_hash)
            .expect("first plan");
        let second = plan_document(&options, &hardware, &model, &hardware_hash, &model_hash)
            .expect("second plan");
        assert_eq!(first, second);
        assert_eq!(
            first.pointer("/ranking/0").and_then(Value::as_str),
            Some("backend:cpu/dense:ram/experts:ssd")
        );
        assert_eq!(
            first
                .pointer("/candidate_plans/0/estimate/method")
                .and_then(Value::as_str),
            Some("analytical-v1")
        );
        assert_eq!(
            first
                .pointer("/request/hardware_profile/document_sha256")
                .and_then(Value::as_str),
            Some(hardware_hash.as_str())
        );
    }

    #[test]
    fn plan_admits_exact_ram_boundary_and_explains_one_byte_shortfall() {
        let (hardware, model) = planner_profiles();
        let exact_ram = planner_base_ram_bytes(&model);
        let exact = plan_document(
            &planner_options(exact_ram),
            &hardware,
            &model,
            &"a".repeat(64),
            &"b".repeat(64),
        )
        .expect("exact boundary plan");
        assert_eq!(
            exact
                .pointer("/ranking")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(1)
        );

        let one_byte_short = plan_document(
            &planner_options(exact_ram - 1),
            &hardware,
            &model,
            &"a".repeat(64),
            &"b".repeat(64),
        )
        .expect("short budget plan");
        assert!(
            one_byte_short
                .pointer("/ranking")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
        );
        assert!(
            one_byte_short
                .pointer("/rejections")
                .and_then(Value::as_array)
                .is_some_and(|items| items.iter().any(|item| {
                    item.pointer("/plan_id").and_then(Value::as_str)
                        == Some("backend:cpu/dense:ram/experts:ssd")
                        && item.pointer("/code").and_then(Value::as_str)
                            == Some("ram_budget_exceeded")
                        && item.pointer("/limit").and_then(Value::as_u64) == Some(exact_ram - 1)
                }))
        );
    }

    #[test]
    fn ranking_is_deterministic_across_throughput_quality_startup_and_plan_id() {
        let mut candidates = vec![
            ranking_candidate("a", 2.0, "known_degradation", 0.0),
            ranking_candidate("b", 2.0, "none", 3.0),
            ranking_candidate("c", 2.0, "none", 1.0),
            ranking_candidate("d", 3.0, "known_degradation", 9.0),
            ranking_candidate("e", 2.0, "none", 1.0),
        ];
        rank_candidate_plans(&mut candidates).expect("ranking");
        let ids: Vec<_> = candidates
            .iter()
            .map(|candidate| string(candidate, "/plan_id"))
            .collect::<Result<_, _>>()
            .expect("plan IDs");
        assert_eq!(ids, ["d", "c", "e", "b", "a"]);
    }

    fn planner_profiles() -> (Value, Value) {
        (
            serde_json::from_str(include_str!(
                "../../../docs/benchmarks/m6.1-05-doctor-v1.json"
            ))
            .expect("hardware profile"),
            serde_json::from_str(include_str!(
                "../../../docs/benchmarks/m6.1-05-model-profile-v1.json"
            ))
            .expect("model profile"),
        )
    }

    fn planner_options(ram_budget_bytes: u64) -> PlanOptions {
        PlanOptions {
            hardware_profile: "hardware.json".to_string(),
            model_profile: "model.json".to_string(),
            ram_budget_bytes,
            vram_budget_bytes: 0,
            context_tokens: 128,
            prefill_tokens: 16,
            decode_tokens: 32,
            compute_gflop_per_token: 1.0,
            request_id: "planner-request-test-v1".to_string(),
            result_id: "planner-result-test-v1".to_string(),
            output: "plan.json".to_string(),
        }
    }

    fn planner_base_ram_bytes(model: &Value) -> u64 {
        number(model, "/artifact/dense/bytes").expect("dense bytes")
            + number(model, "/execution/kv_cache/bytes_per_token").expect("KV bytes") * 128
    }

    fn ranking_candidate(
        plan_id: &str,
        decode_tokens_per_second: f64,
        quality: &str,
        startup_seconds: f64,
    ) -> Value {
        json!({
            "plan_id": plan_id,
            "estimate": {"decode_tokens_per_second": decode_tokens_per_second},
            "quality_risk": {"level": quality},
            "resources": {"startup_seconds": startup_seconds}
        })
    }
}
