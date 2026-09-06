use std::{
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use crate::{
    block::combine_routed_experts,
    r1_1_direct_candidate::{R1_1CandidateReader, R1_1PackedArtifactLayout},
    r2_2_d5_hybrid::D5Layer24HybridReader,
};

use super::*;

const D5C_CACHE_BUDGET_BYTES: usize = 18_874_368;
const D5C_LAYER0_SHA256: &str = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2";
const D5C_LAYER24_SHA256: &str = "d94d12cbea648e2f2911573c893f564254cbde526d60132c600ed24e88317ac2";

#[derive(Debug)]
struct D5CHybridReaders {
    layer0: R1_1CandidateReader,
    layer24: D5Layer24HybridReader,
}

#[derive(Debug)]
struct D5CSample {
    timed_wall_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    generated_ids: Vec<usize>,
    dense_logical_bytes: u64,
    f32_expert_logical_bytes: u64,
    layer0_candidate_logical_bytes: u64,
    layer24_candidate_logical_bytes: u64,
    logical_artifact_bytes_read: u64,
    expert_payload_bytes_read: u64,
    candidate_artifact_bytes_read: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_loads: u64,
    cache_evictions: u64,
    f32_cache_peak_resident_bytes: usize,
    layer0_peak_packed_expert_bytes: usize,
    layer24_peak_hybrid_expert_bytes: usize,
    kv_cache_bytes: usize,
}

fn d5c_layer24_routed(
    hidden_states: clr_core::TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: crate::Qwen3MoeConfig,
    reader: &mut D5Layer24HybridReader,
) -> Tensor {
    let hidden = config.model().hidden_size();
    combine_routed_experts(hidden_states, router, config, |expert_id, occurrences| {
        let expert = reader.load_expert(expert_id)?;
        let mut outputs = Vec::with_capacity(occurrences.len());
        for &(token, _) in occurrences {
            let input = &hidden_states.data()[token * hidden..(token + 1) * hidden];
            outputs.push(expert.apply(input)?);
        }
        Ok::<_, clr_core::RuntimeError>(outputs)
    })
    .expect("D5c Layer24 production hybrid")
}

fn d5c_pass(
    artifact_root: &Path,
    fixture: &TierBReference,
    store: &mut ExpertStore,
    mut hybrid: Option<&mut D5CHybridReaders>,
    expected_generated_ids: &[usize],
) -> D5CSample {
    let pass_started = Instant::now();
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("D5c pinned runtime configuration")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("D5c dense payload");
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let cache_before = store.metrics();
    let layer0_before = hybrid
        .as_deref()
        .map_or(0, |readers| readers.layer0.payload_bytes_read());
    let layer24_before = hybrid
        .as_deref()
        .map_or(0, |readers| readers.layer24.payload_bytes_read());
    let processed_positions = fixture.token_ids.len() + R1_2_GENERATED_TOKEN_COUNT - 1;
    let mut cache = KvCache::new(48, processed_positions, 4, 128).expect("D5c fixed KV cache");
    let mut sequence = fixture.token_ids.clone();
    let mut generated_ids = Vec::with_capacity(R1_2_GENERATED_TOKEN_COUNT);
    let mut step_seconds = Vec::with_capacity(processed_positions);

    for position in 0..processed_positions {
        let step_started = Instant::now();
        let token_id = *sequence
            .get(position)
            .expect("D5c generated token available before position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D5c input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("D5c KV layer"),
            )
            .expect("D5c attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("D5c attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D5c post norm");
            let router =
                route_tokens(post_norm.view(), weights.router.view(), config).expect("D5c router");
            let moe = if let Some(readers) = hybrid.as_deref_mut() {
                match layer {
                    0 => r2_1_routed_experts_with_backend(
                        post_norm.view(),
                        &router,
                        config,
                        &mut readers.layer0,
                        R2_1PackedProjectionBackend::NativeAvx2Fma,
                    )
                    .expect("D5c Layer0 native group32"),
                    24 => {
                        d5c_layer24_routed(post_norm.view(), &router, config, &mut readers.layer24)
                    }
                    _ => streaming_routed_experts_with_observer(
                        post_norm.view(),
                        &router,
                        config,
                        layer,
                        store,
                        expert_layout,
                        |_, _, _, _| {},
                    )
                    .expect("D5c F32 experts"),
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
                .expect("D5c reference F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("D5c block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("D5c KV append");

        if position + 1 >= fixture.token_ids.len() {
            let normalized = rms_norm(
                hidden.view(),
                final_norm_weight.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D5c final norm");
            let logits = streaming_language_model_head(
                &mut payload,
                &final_plan,
                &normalized,
                &mut dense_bytes_read,
            );
            let greedy = greedy_token(logits.view()).expect("D5c finite greedy logits");
            generated_ids.push(greedy);
            if generated_ids.len() < R1_2_GENERATED_TOKEN_COUNT {
                sequence.push(greedy);
            }
        }
        step_seconds.push(step_started.elapsed().as_secs_f64());
    }

    assert_eq!(
        generated_ids, expected_generated_ids,
        "D5c generation guard"
    );
    let prompt_positions = fixture.token_ids.len();
    let prefill_seconds: f64 = step_seconds[..prompt_positions].iter().sum();
    let decode_seconds: f64 = step_seconds[prompt_positions..].iter().sum();
    let decode_positions = processed_positions - prompt_positions;
    assert!(prefill_seconds > 0.0 && decode_seconds > 0.0 && decode_positions > 0);
    let cache_after = store.metrics();
    let layer0_candidate_logical_bytes = hybrid.as_deref().map_or(0, |readers| {
        readers.layer0.payload_bytes_read() - layer0_before
    });
    let layer24_candidate_logical_bytes = hybrid.as_deref().map_or(0, |readers| {
        readers.layer24.payload_bytes_read() - layer24_before
    });
    let f32_expert_logical_bytes = cache_after.bytes_read - cache_before.bytes_read;
    let candidate_artifact_bytes_read =
        layer0_candidate_logical_bytes + layer24_candidate_logical_bytes;
    let expert_payload_bytes_read = f32_expert_logical_bytes + candidate_artifact_bytes_read;
    let logical_artifact_bytes_read = dense_bytes_read + expert_payload_bytes_read;
    D5CSample {
        timed_wall_seconds: pass_started.elapsed().as_secs_f64(),
        prefill_tokens_per_second: f64::from(
            u32::try_from(prompt_positions).expect("D5c prompt count fits u32"),
        ) / prefill_seconds,
        decode_tokens_per_second: f64::from(
            u32::try_from(decode_positions).expect("D5c decode count fits u32"),
        ) / decode_seconds,
        generated_ids,
        dense_logical_bytes: dense_bytes_read,
        f32_expert_logical_bytes,
        layer0_candidate_logical_bytes,
        layer24_candidate_logical_bytes,
        logical_artifact_bytes_read,
        expert_payload_bytes_read,
        candidate_artifact_bytes_read,
        cache_hits: cache_after.hits - cache_before.hits,
        cache_misses: cache_after.misses - cache_before.misses,
        cache_loads: cache_after.loads - cache_before.loads,
        cache_evictions: cache_after.evictions - cache_before.evictions,
        f32_cache_peak_resident_bytes: cache_after.peak_resident_bytes,
        layer0_peak_packed_expert_bytes: hybrid
            .as_deref()
            .map_or(0, |readers| readers.layer0.peak_packed_expert_bytes()),
        layer24_peak_hybrid_expert_bytes: hybrid
            .as_deref()
            .map_or(0, |readers| readers.layer24.peak_loaded_expert_bytes()),
        kv_cache_bytes: cache.byte_size(),
    }
}

fn wait_for_go(go_path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(300);
    while !go_path.exists() {
        assert!(Instant::now() < deadline, "D5c GO marker timeout");
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn m6_3_r2_2_d5c_paired_sample() {
    assert!(
        crate::r2_native::avx2_fma_available(),
        "D5c host must expose AVX2+FMA"
    );
    let setup_started = Instant::now();
    let mode = env::var("COLIBRI_R2_2_D5C_MODE").expect("D5c mode");
    assert!(matches!(
        mode.as_str(),
        "reference_f32" | "production_hybrid"
    ));
    let cache_label = env::var("COLIBRI_R2_2_D5C_CACHE_LABEL").expect("D5c cache label");
    assert!(matches!(
        cache_label.as_str(),
        "first_process_touch" | "likely_warm"
    ));
    let fixture_name = env::var("COLIBRI_R2_2_D5C_FIXTURE").expect("D5c fixture");
    let fixture = tier_b_references()
        .into_iter()
        .find(|fixture| fixture.name == fixture_name)
        .expect("D5c held-out fixture");
    assert!(matches!(
        fixture.name.as_str(),
        "short_english" | "short_thai"
    ));
    let reference_path =
        PathBuf::from(env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("D5c reference path"));
    let frozen = r1_2_frozen_references(&reference_path);
    let expected_generated_ids = frozen[&fixture.name].generated_ids.clone();
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("D5c artifact root"));
    let ready_path = PathBuf::from(env::var_os("COLIBRI_R2_2_D5C_READY").expect("D5c READY"));
    let go_path = PathBuf::from(env::var_os("COLIBRI_R2_2_D5C_GO").expect("D5c GO"));
    let metrics_path =
        PathBuf::from(env::var_os("COLIBRI_R2_2_D5C_METRICS_OUTPUT").expect("D5c metrics"));
    assert!(!ready_path.exists() && !go_path.exists() && !metrics_path.exists());
    let budget = env::var("COLIBRI_EXPERT_CACHE_BUDGET_BYTES")
        .expect("D5c cache budget")
        .parse::<usize>()
        .expect("D5c valid cache budget");
    assert_eq!(budget, D5C_CACHE_BUDGET_BYTES);
    let mut store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    assert_eq!(store.budget(), D5C_CACHE_BUDGET_BYTES);
    let mut hybrid = if mode == "production_hybrid" {
        let layer0_path =
            PathBuf::from(env::var_os("COLIBRI_R2_2_D5C_LAYER0_PATH").expect("D5c Layer0 path"));
        let layer24_path =
            PathBuf::from(env::var_os("COLIBRI_R2_2_D5C_LAYER24_PATH").expect("D5c Layer24 path"));
        Some(D5CHybridReaders {
            layer0: R1_1CandidateReader::open(
                &layer0_path,
                R1_1PackedArtifactLayout::canonical(32).expect("D5c group32 layout"),
                D5C_LAYER0_SHA256,
            )
            .expect("D5c Layer0 candidate"),
            layer24: D5Layer24HybridReader::open(&layer24_path, D5C_LAYER24_SHA256)
                .expect("D5c Layer24 candidate"),
        })
    } else {
        None
    };
    if cache_label == "likely_warm" {
        let warm = d5c_pass(
            &artifact_root,
            &fixture,
            &mut store,
            hybrid.as_mut(),
            &expected_generated_ids,
        );
        assert_eq!(warm.generated_ids, expected_generated_ids);
    }
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    fs::write(&ready_path, b"ready\n").expect("D5c READY marker");
    wait_for_go(&go_path);
    let sample = d5c_pass(
        &artifact_root,
        &fixture,
        &mut store,
        hybrid.as_mut(),
        &expected_generated_ids,
    );
    let generated_text = sample
        .generated_ids
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut evidence = String::from(
        "mode\tfixture\tcache_label\tsetup_seconds\twall_seconds\tprefill_tokens_per_second\tdecode_tokens_per_second\tgenerated_ids\tdense_logical_bytes\tf32_expert_logical_bytes\tlayer0_candidate_logical_bytes\tlayer24_candidate_logical_bytes\tlogical_artifact_bytes_read\texpert_payload_bytes_read\tcandidate_artifact_bytes_read\tcache_hits\tcache_misses\tcache_loads\tcache_evictions\tf32_cache_peak_resident_bytes\tlayer0_peak_packed_expert_bytes\tlayer24_peak_hybrid_expert_bytes\tkv_cache_bytes\n",
    );
    writeln!(
        evidence,
        "{}\t{}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        mode,
        fixture.name,
        cache_label,
        setup_seconds,
        sample.timed_wall_seconds,
        sample.prefill_tokens_per_second,
        sample.decode_tokens_per_second,
        generated_text,
        sample.dense_logical_bytes,
        sample.f32_expert_logical_bytes,
        sample.layer0_candidate_logical_bytes,
        sample.layer24_candidate_logical_bytes,
        sample.logical_artifact_bytes_read,
        sample.expert_payload_bytes_read,
        sample.candidate_artifact_bytes_read,
        sample.cache_hits,
        sample.cache_misses,
        sample.cache_loads,
        sample.cache_evictions,
        sample.f32_cache_peak_resident_bytes,
        sample.layer0_peak_packed_expert_bytes,
        sample.layer24_peak_hybrid_expert_bytes,
        sample.kv_cache_bytes,
    )
    .expect("D5c metrics row");
    fs::write(metrics_path, evidence).expect("D5c metrics output");
}
