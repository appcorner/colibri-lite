use std::{env, fs, path::Path, thread, time::Duration};

use super::*;

const R1_3_TOKENS: [usize; 2] = [9707, 1879];
const R1_3_CACHE_BUDGET_BYTES: usize = 18_874_368;
const R1_3_CANDIDATE_BYTES: u64 = 679_477_248;
const R1_3_CANDIDATE_SHA256: &str =
    "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2";

#[derive(Debug)]
struct R1_3Sample {
    timed_wall_seconds: f64,
    ttft_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    final_argmax: usize,
    dense_logical_bytes: u64,
    f32_expert_logical_bytes: u64,
    candidate_logical_bytes: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_loads: u64,
    cache_evictions: u64,
    f32_cache_peak_resident_bytes: usize,
    candidate_peak_packed_expert_bytes: usize,
    candidate_verification_bytes: u64,
    kv_cache_bytes: usize,
}
fn r1_3_pass(
    artifact_root: &Path,
    store: &mut ExpertStore,
    mut candidate_reader: Option<&mut R1_1CandidateReader>,
) -> R1_3Sample {
    let pass_started = Instant::now();
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R1.3 pinned runtime configuration")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("R1.3 dense payload");
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let cache_before = store.metrics();
    let candidate_bytes_before = candidate_reader
        .as_deref()
        .map_or(0, R1_1CandidateReader::payload_bytes_read);
    let mut cache = KvCache::new(48, R1_3_TOKENS.len(), 4, 128).expect("R1.3 KV cache");
    let mut step_seconds = Vec::with_capacity(R1_3_TOKENS.len());
    let mut last_argmax = None;
    for (position, &token_id) in R1_3_TOKENS.iter().enumerate() {
        let step_started = Instant::now();
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R1.3 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("R1.3 KV layer"),
            )
            .expect("R1.3 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("R1.3 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R1.3 post norm");
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .expect("R1.3 F32 router");
            let moe = if layer == 0 {
                if let Some(reader) = candidate_reader.as_deref_mut() {
                    r1_1_routed_experts_with_observer(
                        post_norm.view(),
                        &router,
                        config,
                        reader,
                        |_, _, _, _| {},
                    )
                    .expect("R1.3 group-32 Layer-0 candidate")
                } else {
                    streaming_routed_experts_with_observer(
                        post_norm.view(),
                        &router,
                        config,
                        layer,
                        store,
                        expert_layout,
                        |_, _, _, _| {},
                    )
                    .expect("R1.3 F32 Layer-0 experts")
                }
            } else {
                streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("R1.3 F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("R1.3 block");
            updates.push((attention.key, attention.value));
        }
        let normalized = rms_norm(
            hidden.view(),
            final_norm_weight.view(),
            config.rms_norm_epsilon(),
        )
        .expect("R1.3 final norm");
        let logits = streaming_language_model_head(
            &mut payload,
            &final_plan,
            &normalized,
            &mut dense_bytes_read,
        );
        last_argmax = Some(greedy_token(logits.view()).expect("R1.3 greedy token"));
        let updates_view = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&updates_view).expect("R1.3 KV append");
        assert_eq!(cache.len(), position + 1);
        step_seconds.push(step_started.elapsed().as_secs_f64());
    }
    let cache_after = store.metrics();
    let candidate_logical_bytes = candidate_reader.as_deref().map_or(0, |reader| {
        reader.payload_bytes_read() - candidate_bytes_before
    });
    let final_argmax = last_argmax.expect("R1.3 final argmax");
    assert_eq!(final_argmax, 0, "R1.3 short-English final argmax");
    R1_3Sample {
        timed_wall_seconds: pass_started.elapsed().as_secs_f64(),
        ttft_seconds: step_seconds[0],
        prefill_tokens_per_second: 1.0 / step_seconds[0],
        decode_tokens_per_second: 1.0 / step_seconds[1],
        final_argmax,
        dense_logical_bytes: dense_bytes_read,
        f32_expert_logical_bytes: cache_after.bytes_read - cache_before.bytes_read,
        candidate_logical_bytes,
        cache_hits: cache_after.hits - cache_before.hits,
        cache_misses: cache_after.misses - cache_before.misses,
        cache_loads: cache_after.loads - cache_before.loads,
        cache_evictions: cache_after.evictions - cache_before.evictions,
        f32_cache_peak_resident_bytes: cache_after.peak_resident_bytes,
        candidate_peak_packed_expert_bytes: candidate_reader
            .as_deref()
            .map_or(0, R1_1CandidateReader::peak_packed_expert_bytes),
        candidate_verification_bytes: candidate_reader
            .as_deref()
            .map_or(0, R1_1CandidateReader::verification_bytes_read),
        kv_cache_bytes: cache.byte_size(),
    }
}

fn r1_3_wait_for_go(go_path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(300);
    while !go_path.exists() {
        assert!(Instant::now() < deadline, "R1.3 GO marker timeout");
        thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn m6_3_r1_3_paired_sample() {
    let setup_started = Instant::now();
    let mode = env::var("COLIBRI_R1_3_MODE").expect("R1.3 mode");
    assert!(matches!(mode.as_str(), "reference" | "candidate"));
    let condition = env::var("COLIBRI_R1_3_CONDITION").expect("R1.3 condition");
    assert!(matches!(
        condition.as_str(),
        "runtime_cache_cold" | "runtime_cache_warm"
    ));
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("R1.3 canonical artifact root"));
    let ready_path = PathBuf::from(env::var_os("COLIBRI_R1_3_READY").expect("R1.3 READY path"));
    let go_path = PathBuf::from(env::var_os("COLIBRI_R1_3_GO").expect("R1.3 GO path"));
    let metrics_path = PathBuf::from(
        env::var_os("COLIBRI_R1_3_METRICS_OUTPUT").expect("R1.3 metrics output path"),
    );
    assert!(!ready_path.exists() && !go_path.exists() && !metrics_path.exists());
    let configured_budget = env::var("COLIBRI_EXPERT_CACHE_BUDGET_BYTES")
        .expect("R1.3 fixed F32 cache budget")
        .parse::<usize>()
        .expect("R1.3 valid cache budget");
    assert_eq!(configured_budget, R1_3_CACHE_BUDGET_BYTES);
    let mut store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    assert_eq!(store.budget(), R1_3_CACHE_BUDGET_BYTES);
    let mut candidate_reader = if mode == "candidate" {
        let candidate_path =
            PathBuf::from(env::var_os("COLIBRI_R1_3_CANDIDATE_PATH").expect("R1.3 candidate path"));
        assert_eq!(
            fs::metadata(&candidate_path)
                .expect("R1.3 candidate metadata")
                .len(),
            R1_3_CANDIDATE_BYTES
        );
        Some(
            R1_1CandidateReader::open(
                &candidate_path,
                R1_1PackedArtifactLayout::canonical(32).expect("R1.3 group-32 layout"),
                R1_3_CANDIDATE_SHA256,
            )
            .expect("R1.3 admitted group-32 artifact"),
        )
    } else {
        None
    };
    if condition == "runtime_cache_warm" {
        let warm = r1_3_pass(&artifact_root, &mut store, candidate_reader.as_mut());
        assert_eq!(warm.final_argmax, 0, "R1.3 warm-up argmax");
    }
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    fs::write(&ready_path, b"ready\n").expect("R1.3 READY marker");
    r1_3_wait_for_go(&go_path);
    candidate_reader = candidate_reader.as_ref().map(|reader| {
        reader
            .reopen_verified_for_measurement()
            .expect("R1.3 reopen verified candidate after GO")
    });
    let sample = r1_3_pass(&artifact_root, &mut store, candidate_reader.as_mut());
    let total_logical_bytes = sample.dense_logical_bytes
        + sample.f32_expert_logical_bytes
        + sample.candidate_logical_bytes;
    let logical_bytes_per_token = reporting_ratio(
        total_logical_bytes,
        u64::try_from(R1_3_TOKENS.len()).expect("R1.3 token count fits u64"),
    );
    let mut evidence = String::from(
        "mode\tcondition\tsetup_seconds\ttimed_wall_seconds\tttft_seconds\tprefill_tokens_per_second\tdecode_tokens_per_second\tfinal_argmax\tdense_logical_bytes\tf32_expert_logical_bytes\tcandidate_logical_bytes\ttotal_logical_bytes\tlogical_bytes_per_token\tcache_hits\tcache_misses\tcache_loads\tcache_evictions\tf32_cache_peak_resident_bytes\tcandidate_peak_packed_expert_bytes\tcandidate_verification_bytes\tkv_cache_bytes\tvram_bytes\n",
    );
    writeln!(
        evidence,
        "{}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0",
        mode,
        condition,
        setup_seconds,
        sample.timed_wall_seconds,
        sample.ttft_seconds,
        sample.prefill_tokens_per_second,
        sample.decode_tokens_per_second,
        sample.final_argmax,
        sample.dense_logical_bytes,
        sample.f32_expert_logical_bytes,
        sample.candidate_logical_bytes,
        total_logical_bytes,
        logical_bytes_per_token,
        sample.cache_hits,
        sample.cache_misses,
        sample.cache_loads,
        sample.cache_evictions,
        sample.f32_cache_peak_resident_bytes,
        sample.candidate_peak_packed_expert_bytes,
        sample.candidate_verification_bytes,
        sample.kv_cache_bytes,
    )
    .expect("R1.3 evidence row");
    fs::write(&metrics_path, evidence).expect("R1.3 metrics output");
}
