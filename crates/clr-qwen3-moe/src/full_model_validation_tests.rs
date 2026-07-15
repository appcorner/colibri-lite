use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
    time::Instant,
};

use clr_core::{Tensor, TensorShape};

use crate::{
    PINNED_QWEN3_30B_A3B_CONFIG,
    block::{PreRouterOutput, pre_router_with_weights, route_tokens},
};

const RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer0-runtime-plan-v1.tsv"
));
const BF16_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer0-checkpoint-plan-v1.tsv"
));
const BF16_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-transformers-layer0.safetensors"
));
const F32_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-f32-layer0-checkpoint-plan-v1.tsv"
));
const F32_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-transformers-f32-layer0.safetensors"
));
const INPUT_IDS: [usize; 4] = [9707, 11, 1879, 0];
const POSITION_IDS: [usize; 4] = [0, 1, 2, 3];
const POST_NORM_PROPAGATED_ABSOLUTE_BUDGET: f32 = 4.255_092_6e-6;

#[derive(Debug, Clone)]
struct RangeRecord {
    offset: u64,
    length: usize,
    shape: Vec<usize>,
}

#[derive(Debug)]
struct RuntimePlan {
    payload: PathBuf,
    payload_length: u64,
    tensors: HashMap<String, RangeRecord>,
}

#[derive(Debug)]
struct CheckpointRecord {
    data_type: &'static str,
    range: RangeRecord,
}

#[derive(Debug)]
struct StageMetrics {
    maximum_absolute_difference: f32,
    maximum_relative_difference: f32,
}

fn parse_shape(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|part| part.parse().expect("valid shape dimension"))
        .collect()
}

fn runtime_plan() -> RuntimePlan {
    let mut payload = None;
    let mut payload_length = None;
    let mut tensors = HashMap::new();
    for line in RUNTIME_PLAN.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        match fields[0] {
            "payload" => {
                assert_eq!(fields.len(), 4, "invalid runtime payload record");
                payload = Some(PathBuf::from(fields[1]));
                payload_length = Some(fields[2].parse().expect("payload length"));
            }
            "tensor" => {
                assert_eq!(fields.len(), 5, "invalid runtime tensor record");
                assert!(
                    tensors
                        .insert(
                            fields[1].to_owned(),
                            RangeRecord {
                                offset: fields[2].parse().expect("tensor offset"),
                                length: fields[3].parse().expect("tensor length"),
                                shape: parse_shape(fields[4]),
                            },
                        )
                        .is_none(),
                    "duplicate runtime tensor {}",
                    fields[1]
                );
            }
            _ => {}
        }
    }
    RuntimePlan {
        payload: payload.expect("runtime payload path"),
        payload_length: payload_length.expect("runtime payload length"),
        tensors,
    }
}

fn checkpoint_plan(plan: &str) -> HashMap<String, CheckpointRecord> {
    let mut tensors = HashMap::new();
    for line in plan.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields[0] != "tensor" {
            continue;
        }
        assert_eq!(fields.len(), 6, "invalid checkpoint record");
        assert!(
            tensors
                .insert(
                    fields[1].to_owned(),
                    CheckpointRecord {
                        data_type: match fields[2] {
                            "F32" => "F32",
                            "I64" => "I64",
                            other => panic!("unsupported checkpoint dtype {other}"),
                        },
                        range: RangeRecord {
                            offset: fields[3].parse().expect("checkpoint offset"),
                            length: fields[4].parse().expect("checkpoint length"),
                            shape: parse_shape(fields[5]),
                        },
                    },
                )
                .is_none(),
            "duplicate checkpoint {}",
            fields[1]
        );
    }
    tensors
}

fn read_exact_range(file: &mut File, record: &RangeRecord, bytes_read: &mut u64) -> Vec<u8> {
    file.seek(SeekFrom::Start(record.offset))
        .expect("seek artifact");
    let mut bytes = vec![0_u8; record.length];
    file.read_exact(&mut bytes).expect("read artifact range");
    *bytes_read += u64::try_from(bytes.len()).expect("range length fits u64");
    bytes
}

fn f32_tensor(bytes: &[u8], shape: &[usize]) -> Tensor {
    let data = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect();
    Tensor::new(TensorShape::new(shape.to_vec()), data).expect("valid f32 tensor")
}

fn artifact_tensor(
    file: &mut File,
    plan: &RuntimePlan,
    name: &str,
    bytes_read: &mut u64,
) -> Tensor {
    let record = plan
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing runtime tensor {name}"));
    f32_tensor(&read_exact_range(file, record, bytes_read), &record.shape)
}

fn embedding_rows(file: &mut File, plan: &RuntimePlan, bytes_read: &mut u64) -> Tensor {
    let record = &plan.tensors["model.embed_tokens.weight"];
    assert_eq!(record.shape, [151_936, 2048]);
    let row_bytes = 2048 * 4;
    let mut data = Vec::with_capacity(INPUT_IDS.len() * 2048);
    for token_id in INPUT_IDS {
        let offset = record.offset + u64::try_from(token_id * row_bytes).expect("embedding offset");
        let row = RangeRecord {
            offset,
            length: row_bytes,
            shape: vec![2048],
        };
        data.extend(f32_tensor(&read_exact_range(file, &row, bytes_read), &[2048]).into_data());
    }
    Tensor::new(TensorShape::new([4, 2048]), data).expect("embedding rows")
}

fn checkpoint_f32(bytes: &[u8], plan: &HashMap<String, CheckpointRecord>, name: &str) -> Tensor {
    let record = &plan[name];
    assert_eq!(record.data_type, "F32", "checkpoint {name} dtype");
    let start = usize::try_from(record.range.offset).expect("checkpoint offset fits usize");
    let end = start + record.range.length;
    f32_tensor(&bytes[start..end], &record.range.shape)
}

fn checkpoint_ids(
    bytes: &[u8],
    plan: &HashMap<String, CheckpointRecord>,
    name: &str,
) -> Vec<usize> {
    let record = &plan[name];
    assert_eq!(record.data_type, "I64", "checkpoint {name} dtype");
    let start = usize::try_from(record.range.offset).expect("checkpoint offset fits usize");
    let end = start + record.range.length;
    bytes[start..end]
        .chunks_exact(8)
        .map(|chunk| {
            usize::try_from(i64::from_le_bytes(
                chunk.try_into().expect("eight-byte i64"),
            ))
            .expect("non-negative checkpoint ID")
        })
        .collect()
}

fn assert_stage(
    stage: &'static str,
    actual: &Tensor,
    expected: &Tensor,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> StageMetrics {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "checkpoint shape mismatch at {stage}"
    );
    let width = actual.shape().dimensions().last().copied().unwrap_or(1);
    let mut maximum_absolute_difference = 0.0_f32;
    let mut maximum_relative_difference = 0.0_f32;
    let mut first_mismatch = None;
    for (index, (&actual_value, &expected_value)) in
        actual.data().iter().zip(expected.data()).enumerate()
    {
        let absolute = (actual_value - expected_value).abs();
        let relative = if expected_value == 0.0 {
            0.0
        } else {
            absolute / expected_value.abs()
        };
        maximum_absolute_difference = maximum_absolute_difference.max(absolute);
        maximum_relative_difference = maximum_relative_difference.max(relative);
        let tolerance = absolute_tolerance + relative_tolerance * expected_value.abs();
        if absolute > tolerance && first_mismatch.is_none() {
            first_mismatch = Some((index, expected_value, actual_value, absolute, tolerance));
        }
    }
    if let Some((index, expected_value, actual_value, absolute, tolerance)) = first_mismatch {
        panic!(
            "first mismatch at checkpoint={stage}, token={}, element={}: expected {expected_value}, got {actual_value}, absolute_error={absolute}, tolerance={tolerance}, checkpoint_max_absolute_error={maximum_absolute_difference}, checkpoint_max_relative_error={maximum_relative_difference}",
            index / width,
            index % width,
        );
    }
    StageMetrics {
        maximum_absolute_difference,
        maximum_relative_difference,
    }
}

fn measure_stage(actual: &Tensor, expected: &Tensor) -> StageMetrics {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "diagnostic checkpoint shape mismatch"
    );
    let mut maximum_absolute_difference = 0.0_f32;
    let mut maximum_relative_difference = 0.0_f32;
    for (&actual_value, &expected_value) in actual.data().iter().zip(expected.data()) {
        let absolute = (actual_value - expected_value).abs();
        let relative = if expected_value == 0.0 {
            0.0
        } else {
            absolute / expected_value.abs()
        };
        maximum_absolute_difference = maximum_absolute_difference.max(absolute);
        maximum_relative_difference = maximum_relative_difference.max(relative);
    }
    StageMetrics {
        maximum_absolute_difference,
        maximum_relative_difference,
    }
}

fn compare_three_paths(
    stage: &'static str,
    rust: &Tensor,
    bf16: &Tensor,
    f32_control: &Tensor,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> StageMetrics {
    let primary = record_three_paths(stage, rust, bf16, f32_control);
    assert_stage(
        stage,
        rust,
        f32_control,
        absolute_tolerance,
        relative_tolerance,
    );
    primary
}

fn record_three_paths(
    stage: &'static str,
    rust: &Tensor,
    bf16: &Tensor,
    f32_control: &Tensor,
) -> StageMetrics {
    let width = rust.shape().dimensions().last().copied().unwrap_or(1);
    let per_token = |left: &Tensor, right: &Tensor| -> Vec<f32> {
        left.data()
            .chunks_exact(width)
            .zip(right.data().chunks_exact(width))
            .map(|(left_row, right_row)| {
                left_row
                    .iter()
                    .zip(right_row)
                    .map(|(left_value, right_value)| (left_value - right_value).abs())
                    .fold(0.0_f32, f32::max)
            })
            .collect()
    };
    let bf16_vs_rust = measure_stage(rust, bf16);
    let bf16_vs_f32 = measure_stage(f32_control, bf16);
    let primary = measure_stage(rust, f32_control);
    println!(
        "three_path_tokens checkpoint={stage} f32_vs_rust={:?} bf16_vs_rust={:?} bf16_vs_f32={:?}",
        per_token(f32_control, rust),
        per_token(bf16, rust),
        per_token(bf16, f32_control),
    );
    println!(
        "three_path checkpoint={stage} f32_vs_rust={primary:?} bf16_vs_rust={bf16_vs_rust:?} bf16_vs_f32={bf16_vs_f32:?}"
    );
    primary
}

fn atomic_diagnostic(path: &PathBuf, payload: &[u8]) {
    if path.exists() {
        assert_eq!(fs::read(path).expect("read existing diagnostic"), payload);
        return;
    }
    let temporary = path.with_extension("incomplete");
    assert!(!temporary.exists(), "incomplete diagnostic exists");
    let mut output = File::create(&temporary).expect("create diagnostic");
    output.write_all(payload).expect("write diagnostic");
    output.sync_all().expect("sync diagnostic");
    drop(output);
    fs::rename(temporary, path).expect("promote diagnostic");
}

fn export_rms_diagnostics(residual: &Tensor, post_norm: &Tensor, epsilon: f32) {
    let root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_RMS_DIAGNOSTIC_ROOT must be set for the focused diagnostic");
    assert!(root.is_absolute(), "diagnostic root must be absolute");
    let encode = |tensor: &Tensor| -> Vec<u8> {
        tensor
            .data()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    atomic_diagnostic(
        &root.join("m4.2-02-rust-layer0-residual-f32.bin"),
        &encode(residual),
    );
    atomic_diagnostic(
        &root.join("m4.2-02-rust-layer0-postnorm-f32.bin"),
        &encode(post_norm),
    );

    let mut rows = String::from(
        "token\tordered_sum_of_squares\tmean_square\tepsilon\tmean_plus_epsilon\tsquare_root\treciprocal_rms\tsum_bits\tmean_bits\tplus_bits\tsqrt_bits\treciprocal_bits\n",
    );
    for (token, row) in residual.data().chunks_exact(2048).enumerate() {
        let sum: f32 = row.iter().map(|value| value * value).sum();
        let mean = sum / 2048.0;
        let plus = mean + epsilon;
        let square_root = plus.sqrt();
        let reciprocal = square_root.recip();
        use std::fmt::Write as _;
        writeln!(
            &mut rows,
            "{token}\t{sum:.17e}\t{mean:.17e}\t{epsilon:.17e}\t{plus:.17e}\t{square_root:.17e}\t{reciprocal:.17e}\t{:08x}\t{:08x}\t{:08x}\t{:08x}\t{:08x}",
            sum.to_bits(),
            mean.to_bits(),
            plus.to_bits(),
            square_root.to_bits(),
            reciprocal.to_bits(),
        )
        .expect("write row diagnostic");
    }
    atomic_diagnostic(
        &root.join("m4.2-02-rust-rms-row-diagnostics-v1.tsv"),
        rows.as_bytes(),
    );
}

fn deterministic_top_k(logits: &[f32], top_k: usize) -> Vec<usize> {
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    ids.truncate(top_k);
    ids
}

fn comma_separated<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_router_boundaries(
    output: &PreRouterOutput,
    bf16_logits: &Tensor,
    f32_logits: &Tensor,
    bf16_ids: &[usize],
    f32_ids: &[usize],
) -> Vec<&'static str> {
    let expert_count = 128;
    let top_k = 8;
    let mut assertable = Vec::with_capacity(4);
    let mut evidence = String::from(
        "token\tf32_classification\tbf16_classification\tf32_vs_rust_max_error\tbf16_vs_rust_max_error\tbf16_vs_f32_max_error\tf32_required_margin\tbf16_required_margin\tf32_kth\tf32_highest_unselected\tf32_margin\tbf16_kth\tbf16_highest_unselected\tbf16_margin\ttransformers_f32_ids\ttransformers_bf16_ids\trust_ids\ttransformers_f32_selected_logits\ttransformers_bf16_selected_logits\trust_selected_logits\n",
    );
    for token in 0..4 {
        let bf16_row = &bf16_logits.data()[token * expert_count..(token + 1) * expert_count];
        let f32_row = &f32_logits.data()[token * expert_count..(token + 1) * expert_count];
        let actual_row =
            &output.router.logits.data()[token * expert_count..(token + 1) * expert_count];
        let bf16_selected = &bf16_ids[token * top_k..(token + 1) * top_k];
        let f32_selected = &f32_ids[token * top_k..(token + 1) * top_k];
        let actual_selected = &output.router.selected_experts[token * top_k..(token + 1) * top_k];
        assert_eq!(
            actual_selected,
            deterministic_top_k(actual_row, top_k),
            "Rust deterministic router policy failed at token {token}"
        );
        let selected_set: HashSet<_> = f32_selected.iter().copied().collect();
        let bf16_selected_set: HashSet<_> = bf16_selected.iter().copied().collect();
        let f32_selected_logits: Vec<_> = f32_selected.iter().map(|id| f32_row[*id]).collect();
        let bf16_selected_logits: Vec<_> = bf16_selected.iter().map(|id| bf16_row[*id]).collect();
        let rust_selected_logits: Vec<_> =
            actual_selected.iter().map(|id| actual_row[*id]).collect();
        let kth = f32_selected_logits
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let highest_unselected = f32_row
            .iter()
            .enumerate()
            .filter(|(id, _)| !selected_set.contains(id))
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let margin = kth - highest_unselected;
        let bf16_kth = bf16_selected_logits
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let bf16_highest_unselected = bf16_row
            .iter()
            .enumerate()
            .filter(|(id, _)| !bf16_selected_set.contains(id))
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let bf16_margin = bf16_kth - bf16_highest_unselected;
        let f32_vs_rust_maximum_error = f32_row
            .iter()
            .zip(actual_row)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        let bf16_vs_rust_maximum_error = bf16_row
            .iter()
            .zip(actual_row)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        let bf16_vs_f32_maximum_error = bf16_row
            .iter()
            .zip(f32_row)
            .map(|(bf16, f32)| (bf16 - f32).abs())
            .fold(0.0_f32, f32::max);
        let required_margin = 2.0 * f32_vs_rust_maximum_error;
        let bf16_required_margin = 2.0 * bf16_vs_rust_maximum_error;
        let ids_assertable = margin.is_finite() && margin > 0.0 && margin > required_margin;
        let bf16_ids_assertable =
            bf16_margin.is_finite() && bf16_margin > 0.0 && bf16_margin > bf16_required_margin;
        if ids_assertable {
            assert_eq!(
                actual_selected, f32_selected,
                "safe-margin expert IDs differ at token {token}: margin={margin}, max_error={f32_vs_rust_maximum_error}"
            );
        }
        if bf16_ids_assertable {
            assert_eq!(
                actual_selected, bf16_selected,
                "safe-margin BF16 expert IDs differ at token {token}: margin={bf16_margin}, max_error={bf16_vs_rust_maximum_error}"
            );
        }
        let classification = if ids_assertable {
            "exact_match_safe"
        } else {
            "numerically_ambiguous"
        };
        let bf16_classification = if bf16_ids_assertable {
            "exact_match_safe"
        } else {
            "numerically_ambiguous"
        };
        println!(
            "router_boundary token={token} classification={classification} bf16_classification={bf16_classification} ids_assertable={ids_assertable} transformers_bf16_ids={bf16_selected:?} transformers_f32_ids={f32_selected:?} rust_ids={actual_selected:?} transformers_bf16_selected_logits={bf16_selected_logits:?} transformers_f32_selected_logits={f32_selected_logits:?} rust_selected_logits={rust_selected_logits:?} kth={kth} highest_unselected={highest_unselected} margin={margin} bf16_kth={bf16_kth} bf16_highest_unselected={bf16_highest_unselected} bf16_margin={bf16_margin} f32_vs_rust_max_logit_error={f32_vs_rust_maximum_error} bf16_vs_rust_max_logit_error={bf16_vs_rust_maximum_error} bf16_vs_f32_max_logit_error={bf16_vs_f32_maximum_error} required_margin={required_margin} bf16_required_margin={bf16_required_margin}"
        );
        use std::fmt::Write as _;
        writeln!(
            evidence,
            "{token}\t{classification}\t{bf16_classification}\t{f32_vs_rust_maximum_error:.17e}\t{bf16_vs_rust_maximum_error:.17e}\t{bf16_vs_f32_maximum_error:.17e}\t{required_margin:.17e}\t{bf16_required_margin:.17e}\t{kth:.17e}\t{highest_unselected:.17e}\t{margin:.17e}\t{bf16_kth:.17e}\t{bf16_highest_unselected:.17e}\t{bf16_margin:.17e}\t{}\t{}\t{}\t{}\t{}\t{}",
            comma_separated(f32_selected),
            comma_separated(bf16_selected),
            comma_separated(actual_selected),
            comma_separated(&f32_selected_logits),
            comma_separated(&bf16_selected_logits),
            comma_separated(&rust_selected_logits),
        )
        .expect("write router evidence");
        assertable.push(classification);
    }
    let root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_RMS_DIAGNOSTIC_ROOT must be set for router evidence");
    atomic_diagnostic(
        &root.join("m4.2-02-rust-layer0-router-evidence-v1.tsv"),
        evidence.as_bytes(),
    );
    assertable
}

#[test]
fn pinned_layer_zero_pre_router_matches_transformers() {
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let plan = runtime_plan();
    let payload_path = artifact_root.join(&plan.payload);
    let mut payload = File::open(&payload_path).expect("open canonical dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length,
        "dense payload length"
    );
    let bf16_checkpoints = checkpoint_plan(BF16_CHECKPOINT_PLAN);
    let f32_checkpoints = checkpoint_plan(F32_CHECKPOINT_PLAN);
    assert_eq!(
        checkpoint_ids(BF16_CHECKPOINTS, &bf16_checkpoints, "input_ids"),
        INPUT_IDS
    );
    assert_eq!(
        checkpoint_ids(F32_CHECKPOINTS, &f32_checkpoints, "input_ids"),
        INPUT_IDS
    );
    assert_eq!(
        checkpoint_ids(BF16_CHECKPOINTS, &bf16_checkpoints, "position_ids"),
        POSITION_IDS
    );
    assert_eq!(
        checkpoint_ids(F32_CHECKPOINTS, &f32_checkpoints, "position_ids"),
        POSITION_IDS
    );

    let mask = checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "attention_mask");
    assert_eq!(
        mask,
        checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "attention_mask"),
        "control attention mask differs"
    );
    assert_eq!(mask.shape().dimensions(), [1, 1, 4, 4]);
    for query in 0..4 {
        for key in 0..4 {
            let value = mask.data()[query * 4 + key];
            if key <= query {
                assert_eq!(value, 0.0, "causal mask allowed value");
            } else {
                assert_eq!(value, -3.389_531_4e38, "causal mask blocked value");
            }
        }
    }

    let mut artifact_bytes_read = 0_u64;
    let embedding = embedding_rows(&mut payload, &plan, &mut artifact_bytes_read);
    let input_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.input_layernorm.weight",
        &mut artifact_bytes_read,
    );
    let query = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.q_proj.weight",
        &mut artifact_bytes_read,
    );
    let key = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.k_proj.weight",
        &mut artifact_bytes_read,
    );
    let value = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.v_proj.weight",
        &mut artifact_bytes_read,
    );
    let output = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.o_proj.weight",
        &mut artifact_bytes_read,
    );
    let query_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.q_norm.weight",
        &mut artifact_bytes_read,
    );
    let key_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.k_norm.weight",
        &mut artifact_bytes_read,
    );
    let post_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.post_attention_layernorm.weight",
        &mut artifact_bytes_read,
    );
    let router = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.mlp.gate.weight",
        &mut artifact_bytes_read,
    );

    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let original_embedding = embedding.clone();
    let started = Instant::now();
    let pre_router = pre_router_with_weights(
        embedding.view(),
        input_norm.view(),
        query.view(),
        key.view(),
        value.view(),
        output.view(),
        query_norm.view(),
        key_norm.view(),
        post_norm.view(),
        router.view(),
        config,
    )
    .expect("layer-0 pre-router execution");
    let execution_seconds = started.elapsed().as_secs_f64();
    assert_eq!(
        embedding, original_embedding,
        "pre-router execution mutated its input"
    );
    export_rms_diagnostics(
        &pre_router.residual_output,
        &pre_router.post_attention_norm,
        config.rms_norm_epsilon(),
    );
    println!(
        "layer0_execution artifact_bytes_read={artifact_bytes_read} execution_seconds={execution_seconds}"
    );

    let embedding_metrics = compare_three_paths(
        "embedding_output",
        &embedding,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "embedding_output"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "embedding_output"),
        0.0,
        0.0,
    );
    let input_norm_metrics = compare_three_paths(
        "input_rmsnorm",
        &pre_router.input_norm,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "input_rmsnorm"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "input_rmsnorm"),
        1.0e-6,
        1.0e-5,
    );
    let attention_metrics = compare_three_paths(
        "attention_output",
        &pre_router.attention_output,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "attention_output"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "attention_output"),
        1.0e-6,
        1.0e-5,
    );
    let residual_metrics = compare_three_paths(
        "residual_output",
        &pre_router.residual_output,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "residual_output"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "residual_output"),
        1.0e-6,
        1.0e-5,
    );
    let bf16_post_norm = checkpoint_f32(
        BF16_CHECKPOINTS,
        &bf16_checkpoints,
        "post_attention_rmsnorm",
    );
    let f32_post_norm = checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "post_attention_rmsnorm");
    let post_norm_metrics = compare_three_paths(
        "post_attention_rmsnorm",
        &pre_router.post_attention_norm,
        &bf16_post_norm,
        &f32_post_norm,
        POST_NORM_PROPAGATED_ABSOLUTE_BUDGET,
        0.0,
    );
    let bf16_logits = checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "router_logits");
    let reference_logits = checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "router_logits");
    let isolated_router = route_tokens(f32_post_norm.view(), router.view(), config)
        .expect("same-input F32 router execution");
    let isolated_router_metrics = measure_stage(&isolated_router.logits, &reference_logits);
    println!("isolated_router_same_input metrics={isolated_router_metrics:?}");
    let router_metrics = record_three_paths(
        "router_logits",
        &pre_router.router.logits,
        &bf16_logits,
        &reference_logits,
    );
    let bf16_ids = checkpoint_ids(BF16_CHECKPOINTS, &bf16_checkpoints, "selected_expert_ids");
    let f32_ids = checkpoint_ids(F32_CHECKPOINTS, &f32_checkpoints, "selected_expert_ids");
    let classifications = validate_router_boundaries(
        &pre_router,
        &bf16_logits,
        &reference_logits,
        &bf16_ids,
        &f32_ids,
    );
    let routing_metrics = record_three_paths(
        "routing_weights",
        &pre_router.router.weights,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "routing_weights"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "routing_weights"),
    );

    let explicit_weight_bytes = input_norm.data().len()
        + query.data().len()
        + key.data().len()
        + value.data().len()
        + output.data().len()
        + query_norm.data().len()
        + key_norm.data().len()
        + post_norm.data().len()
        + router.data().len();
    let explicit_runtime_bytes = (explicit_weight_bytes
        + embedding.data().len()
        + pre_router.input_norm.data().len()
        + pre_router.attention_output.data().len()
        + pre_router.residual_output.data().len()
        + pre_router.post_attention_norm.data().len()
        + pre_router.router.logits.data().len()
        + pre_router.router.weights.data().len())
        * 4;
    println!(
        "layer0_metrics artifact_bytes_read={artifact_bytes_read} peak_explicit_runtime_bytes={explicit_runtime_bytes} execution_seconds={execution_seconds} classifications={classifications:?} isolated_router_max_abs={} isolated_router_max_rel={} checkpoint_max_abs=[embedding:{},input_norm:{},attention:{},residual:{},post_norm:{},router:{},routing:{}] checkpoint_max_rel=[embedding:{},input_norm:{},attention:{},residual:{},post_norm:{},router:{},routing:{}]",
        isolated_router_metrics.maximum_absolute_difference,
        isolated_router_metrics.maximum_relative_difference,
        embedding_metrics.maximum_absolute_difference,
        input_norm_metrics.maximum_absolute_difference,
        attention_metrics.maximum_absolute_difference,
        residual_metrics.maximum_absolute_difference,
        post_norm_metrics.maximum_absolute_difference,
        router_metrics.maximum_absolute_difference,
        routing_metrics.maximum_absolute_difference,
        embedding_metrics.maximum_relative_difference,
        input_norm_metrics.maximum_relative_difference,
        attention_metrics.maximum_relative_difference,
        residual_metrics.maximum_relative_difference,
        post_norm_metrics.maximum_relative_difference,
        router_metrics.maximum_relative_difference,
        routing_metrics.maximum_relative_difference,
    );
}
