use std::{
    env,
    fmt::Write as _,
    fs::File,
    path::{Path, PathBuf},
};

use clr_storage::CacheMetrics;

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct RepresentativeTraceRecord {
    ordinal: usize,
    fixture_id: String,
    step: usize,
    input_token_id: usize,
    layer: usize,
    rank: usize,
    expert: usize,
    payload_bytes: usize,
    cache_hit: bool,
    loaded: bool,
    evictions: u64,
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn parse_usize_list(value: &str) -> Vec<usize> {
    value
        .split(',')
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.parse()
                .unwrap_or_else(|_| panic!("invalid integer list: {value}"))
        })
        .collect()
}

fn parse_required_usize(name: &str) -> usize {
    required_env(name)
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be an integer"))
}

fn assert_finite(label: &str, values: &[f32]) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
}

fn json_usize_list(values: &[usize]) -> String {
    let mut output = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("write JSON integer");
    }
    output.push(']');
    output
}

#[test]
fn representative_trace_capture() {
    let fixture_id = required_env("COLIBRI_TRACE_FIXTURE_ID");
    let workload_class = required_env("COLIBRI_TRACE_WORKLOAD_CLASS");
    let input_token_ids = parse_usize_list(&required_env("COLIBRI_TRACE_INPUT_TOKEN_IDS"));
    let requested_generation_length =
        parse_required_usize("COLIBRI_TRACE_REQUESTED_GENERATION_LENGTH");
    let seed = required_env("COLIBRI_TRACE_SEED")
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("COLIBRI_TRACE_SEED must be an integer"));
    let decoding_mode = required_env("COLIBRI_TRACE_DECODING_MODE");
    let kv_cache_capacity = parse_required_usize("COLIBRI_TRACE_KV_CACHE_CAPACITY");
    let artifact_root = PathBuf::from(required_env("COLIBRI_ARTIFACT_ROOT"));
    let trace_output = PathBuf::from(required_env("COLIBRI_EXPERT_TRACE_OUTPUT"));
    let cache_budget = required_env("COLIBRI_EXPERT_CACHE_BUDGET_BYTES")
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("COLIBRI_EXPERT_CACHE_BUDGET_BYTES must be an integer"));

    assert!(!fixture_id.is_empty(), "fixture ID must not be empty");
    assert!(
        !workload_class.is_empty(),
        "workload class must not be empty"
    );
    assert!(!input_token_ids.is_empty(), "prompt must not be empty");
    assert!(
        requested_generation_length > 0,
        "generation length must be positive"
    );
    assert_eq!(decoding_mode, "greedy", "M5.2-01 primary capture is greedy");
    assert_eq!(seed, 0, "M5.2-01 primary capture seed is fixed at zero");
    assert_eq!(
        cache_budget, 18_874_368,
        "M5.2-01 uses the one-expert cache"
    );
    assert!(
        kv_cache_capacity >= input_token_ids.len() + requested_generation_length - 1,
        "KV capacity is smaller than the processed sequence"
    );

    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    assert!(
        input_token_ids
            .iter()
            .all(|&token| token < config.model().vocabulary_size()),
        "prompt token is outside the vocabulary"
    );
    let expert_layout = PackedExpertLayout::for_config(config);
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length
    );
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    assert_finite("final norm weight", final_norm_weight.data());
    let mut store = expert_store_from_plans(
        &[GENERATION_LAYER47_EXPERT_RUNTIME_PLAN],
        &artifact_root,
        48 * 128,
    );
    let mut cache = KvCache::new(48, kv_cache_capacity, 4, 128).expect("representative KV cache");
    let allocation_capacities = cache.allocation_capacities();
    assert_eq!(allocation_capacities.len(), 48, "KV cache layer count");
    assert!(
        allocation_capacities
            .iter()
            .all(|&(key, value)| key == kv_cache_capacity * 4 * 128
                && value == kv_cache_capacity * 4 * 128),
        "fixed KV allocation shapes"
    );

    let processed_steps = input_token_ids.len() + requested_generation_length - 1;
    let mut generated = Vec::with_capacity(requested_generation_length);
    let mut records = Vec::with_capacity(processed_steps * 48 * 8);
    let mut requested_keys = Vec::with_capacity(processed_steps * 48 * 8);
    let mut produced_last_logits = false;
    for step in 0..processed_steps {
        let token_id = if step < input_token_ids.len() {
            input_token_ids[step]
        } else {
            generated[step - input_token_ids.len()]
        };
        assert_eq!(cache.len(), step, "KV cache position before append");
        let cache_prefix: Vec<_> = (0..48)
            .map(|layer| {
                let view = cache.layer(layer).expect("KV layer before append");
                (view.key.to_vec(), view.value.to_vec())
            })
            .collect();
        let mut current = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        assert_finite("embedding", current.data());
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                current.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} input norm: {error}"));
            assert_finite("input norm", input_norm.data());
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("KV layer view"),
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} attention: {error}"));
            assert_finite("attention output", attention.output.data());
            assert_finite("attention key", &attention.key);
            assert_finite("attention value", &attention.value);
            let residual = elementwise_add(current.view(), attention.output.view())
                .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} residual: {error}"));
            assert_finite("attention residual", residual.data());
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} post norm: {error}"));
            assert_finite("post norm", post_norm.data());
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} router: {error}"));
            assert_finite("router logits", router.logits.data());
            assert_finite("routing weights", router.weights.data());
            requested_keys.extend(
                router
                    .selected_experts
                    .iter()
                    .map(|&expert| layer * 128 + expert),
            );
            let moe = streaming_routed_experts_with_request_observer(
                post_norm.view(),
                &router,
                config,
                layer,
                &mut store,
                expert_layout,
                |layer, expert, _token, _position, rank, observation| {
                    records.push(RepresentativeTraceRecord {
                        ordinal: records.len(),
                        fixture_id: fixture_id.clone(),
                        step,
                        input_token_id: token_id,
                        layer,
                        rank,
                        expert,
                        payload_bytes: observation.payload_bytes,
                        cache_hit: observation.cache_hit,
                        loaded: observation.loaded,
                        evictions: observation.evictions,
                    });
                },
                |_, _, _, _| {},
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} experts: {error}"));
            assert_finite("MoE output", moe.data());
            let block = elementwise_add(residual.view(), moe.view())
                .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} block: {error}"));
            assert_finite("block output", block.data());
            updates.push((attention.key, attention.value));
            current = block;
        }
        let normalized = rms_norm(
            current.view(),
            final_norm_weight.view(),
            config.rms_norm_epsilon(),
        )
        .unwrap_or_else(|error| panic!("step-{step} final RMSNorm: {error}"));
        assert_finite("final norm", normalized.data());
        let logits = streaming_language_model_head(
            &mut payload,
            &final_plan,
            &normalized,
            &mut dense_bytes_read,
        );
        assert_finite("language-model logits", logits.data());
        let selected = greedy_token(logits.view()).expect("finite greedy logits");
        if step + 1 >= input_token_ids.len() && generated.len() < requested_generation_length {
            generated.push(selected);
        }
        if step + 1 == processed_steps {
            produced_last_logits = true;
        }
        let updates_view: Vec<_> = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect();
        cache
            .append_token(&updates_view)
            .expect("transactional KV append");
        assert_eq!(cache.len(), step + 1, "KV cache logical length");
        assert_eq!(cache.allocation_capacities(), allocation_capacities);
        for (layer, (prior_key, prior_value)) in cache_prefix.iter().enumerate() {
            let view = cache.layer(layer).expect("KV layer after append");
            assert_eq!(view.len, step + 1, "KV layer logical length");
            assert_eq!(view.key.len(), (step + 1) * 4 * 128, "KV key shape");
            assert_eq!(view.value.len(), (step + 1) * 4 * 128, "KV value shape");
            assert_eq!(
                &view.key[..prior_key.len()],
                prior_key,
                "KV key prefix overwrite"
            );
            assert_eq!(
                &view.value[..prior_value.len()],
                prior_value,
                "KV value prefix overwrite"
            );
        }
    }

    assert_eq!(
        generated.len(),
        requested_generation_length,
        "generated length"
    );
    assert!(produced_last_logits, "final logits were not produced");
    let metrics = store.metrics();
    assert_eq!(
        records.len(),
        processed_steps * 48 * 8,
        "expert occurrence count"
    );
    assert_eq!(metrics.hits + metrics.misses, records.len() as u64);
    assert_eq!(metrics.loads, metrics.misses);
    assert!(metrics.resident_bytes <= cache_budget);
    assert!(metrics.peak_resident_bytes <= cache_budget);
    assert_eq!(metrics.oversized_entry_events, 0);
    assert_eq!(metrics.blocked_eviction_events, 0);
    assert_eq!(cache.len(), processed_steps);
    assert_eq!(requested_keys.len(), records.len());
    assert!(requested_keys.iter().all(|&key| key < 48 * 128));
    assert!(records.iter().all(|record| record.fixture_id == fixture_id));
    assert!(
        records
            .iter()
            .all(|record| record.input_token_id < config.model().vocabulary_size())
    );

    let instrumentation_commit = required_env("COLIBRI_TRACE_INSTRUMENTATION_COMMIT");
    write_representative_trace(
        &trace_output,
        &fixture_id,
        &workload_class,
        &input_token_ids,
        &generated,
        requested_generation_length,
        seed,
        &decoding_mode,
        kv_cache_capacity,
        cache_budget,
        dense_bytes_read,
        cache.byte_size(),
        metrics,
        &records,
        &instrumentation_commit,
    );
    println!(
        "m5_2_trace_capture_complete fixture={fixture_id} generated={generated:?} records={} hits={} loads={} evictions={}",
        records.len() / (48 * 8),
        metrics.hits,
        metrics.loads,
        metrics.evictions,
    );
}

#[allow(clippy::too_many_arguments)]
fn write_representative_trace(
    path: &Path,
    fixture_id: &str,
    workload_class: &str,
    input_token_ids: &[usize],
    generated: &[usize],
    requested_generation_length: usize,
    seed: u64,
    decoding_mode: &str,
    kv_cache_capacity: usize,
    cache_budget: usize,
    dense_bytes_read: u64,
    kv_cache_bytes: usize,
    metrics: CacheMetrics,
    records: &[RepresentativeTraceRecord],
    instrumentation_commit: &str,
) {
    let mut output = String::with_capacity(records.len() * 260 + 2400);
    writeln!(
        output,
        "{{\"schema\":\"colibri-qwen3-moe-m5.2-01-ordered-expert-trace-v2\",\"schema_version\":2,\"trace_id\":\"m5.2-01-{fixture_id}-ordered-expert-requests-v1\",\"fixture_id\":\"{fixture_id}\",\"workload_class\":\"{workload_class}\",\"classification\":\"M5.2-01 deterministic Rust authoritative ordered expert-request trace\",\"baseline_id\":\"qwen3-30b-a3b-colibri-f32-windows-x64-v1\",\"release_id\":\"colibri-lite-rs-m4-qwen3-30b-a3b-f32-v1\",\"release_tag\":\"m4-full-qwen3-baseline-v1\",\"baseline_runtime_source_commit\":\"80099f05246a4450ded6f42baf6b8db5a4b2e623\",\"trace_instrumentation_commit\":\"{instrumentation_commit}\",\"model_repository\":\"Qwen/Qwen3-30B-A3B\",\"model_revision\":\"ad44e777bcd18fa416d9da3bd8f70d33ebb85d39\",\"canonical_artifact_root_sha256\":\"f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2\",\"tokenizer_identity\":\"Qwen2Tokenizer:a66c5b39331656b1a3befd2d695265f15bdc5f16226fbbf7794bfb5ae9220c5e\",\"input_token_ids\":{},\"expected_generated_token_ids\":{},\"requested_generation_length\":{},\"seed\":{},\"decoding\":{{\"mode\":\"{}\",\"temperature\":null}},\"stop_conditions\":{{\"type\":\"fixed_length\",\"eos_token_id\":null}},\"kv_cache\":{{\"capacity\":{cache_capacity},\"final_sequence_length\":{final_sequence_length},\"layers\":48,\"key_shape_per_layer\":[{key_shape},4,128],\"value_shape_per_layer\":[{key_shape},4,128],\"allocated_bytes\":{allocated_bytes}}},\"cache_configuration\":{{\"budget_bytes\":{cache_budget},\"policy\":\"strict_global_lru\",\"payload_accounting\":\"payload_bytes_only\",\"trace_mode\":\"every_logical_request\"}},\"runtime_configuration\":{{\"compute_dtype\":\"F32\",\"kv_cache_dtype\":\"F32\",\"threads\":8,\"target\":\"x86_64-pc-windows-msvc\",\"build_profile\":\"release\",\"mmap\":false,\"prefetch\":false,\"simd\":false,\"threading\":false,\"quantization\":false,\"gpu\":false}},\"counters\":{{\"requested_trace_count\":{requested_count},\"cache_hits\":{cache_hits},\"cache_misses\":{cache_misses},\"loads\":{loads},\"evictions\":{evictions},\"expert_payload_bytes_requested\":{payload_requested},\"expert_bytes_read\":{expert_bytes_read},\"dense_bytes_read\":{dense_bytes_read}}},\"serialization\":\"UTF-8 JSON object with fixed field order, compact separators, trailing newline; no timestamp, process ID, local path, or timing\",\"records\":[",
        json_usize_list(input_token_ids),
        json_usize_list(generated),
        requested_generation_length,
        seed,
        decoding_mode,
        requested_count = records.len(),
        cache_hits = metrics.hits,
        cache_misses = metrics.misses,
        loads = metrics.loads,
        evictions = metrics.evictions,
        expert_bytes_read = metrics.bytes_read,
        cache_budget = cache_budget,
        payload_requested = records.iter().map(|record| record.payload_bytes as u64).sum::<u64>(),
        cache_capacity = kv_cache_capacity,
        final_sequence_length = records.len() / (48 * 8),
        key_shape = kv_cache_capacity * 4 * 128,
        allocated_bytes = kv_cache_bytes,
        dense_bytes_read = dense_bytes_read,
    )
    .expect("write representative trace header");
    for (index, record) in records.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        let prompt_length = input_token_ids.len();
        let phase = if record.step < prompt_length {
            "prefill"
        } else {
            "decode"
        };
        let decode_step = if record.step < prompt_length {
            "null".to_owned()
        } else {
            (record.step - prompt_length).to_string()
        };
        write!(
            output,
            "{{\"global_ordinal\":{},\"fixture_id\":\"{}\",\"phase\":\"{}\",\"generation_step\":{},\"decode_step\":{},\"input_token_id\":{},\"absolute_position\":{},\"layer_index\":{},\"selected_expert_rank\":{},\"expert_id\":{},\"layer_expert_key\":\"layer.{}.expert.{}\",\"payload_bytes\":{},\"cache_hit\":{},\"loaded\":{},\"evictions_caused\":{}}}",
            record.ordinal,
            record.fixture_id,
            phase,
            record.step,
            decode_step,
            record.input_token_id,
            record.step,
            record.layer,
            record.rank,
            record.expert,
            record.layer,
            record.expert,
            record.payload_bytes,
            record.cache_hit,
            record.loaded,
            record.evictions,
        )
        .expect("write representative trace record");
    }
    output.push_str("]}\n");
    atomic_diagnostic(&path.to_path_buf(), output.as_bytes());
}
