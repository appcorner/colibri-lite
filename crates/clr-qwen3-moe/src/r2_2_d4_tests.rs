use crate::{
    block::combine_routed_experts,
    r1_1_direct_candidate::{R1_1CandidateReader, R2_2LayerCandidateReader},
    streaming::{StreamingModelError, r2_d3_load_f32_expert},
};

use super::*;

const PACKED_LAYERS: [usize; 2] = [0, 24];
const LOCAL_MAX_ABS_LIMIT: f32 = 0.001;
const NATIVE_SCALAR_PROMPT_MAX_ABS: f32 = 0.002;

#[derive(Debug)]
struct D4Run {
    prompt_top20_ids: Vec<usize>,
    prompt_argmax: usize,
    prompt_guard_ids: HashMap<usize, Vec<usize>>,
    prompt_logits: Vec<f32>,
    prompt_logits_sha256: String,
    prompt_final_norm_sha256: String,
    prompt_fixed_logit_error: f32,
    prompt_top20_logit_error: f32,
    generated_ids: Vec<usize>,
    candidate_f32_layer_max_abs: HashMap<usize, f32>,
    native_scalar_layer_max_abs: HashMap<usize, f32>,
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn open_readers() -> HashMap<usize, R2_2LayerCandidateReader> {
    let root = PathBuf::from(
        env::var_os("COLIBRI_R2_2_ARTIFACT_ROOT").expect("R2.2 group32 artifact root"),
    );
    let layout = R1_1PackedArtifactLayout::canonical(32).expect("D4 group32 layout");
    [
        (
            0,
            "layer00-group32-repro.bin",
            "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2",
        ),
        (
            24,
            "layer24-group32.bin",
            "890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0",
        ),
    ]
    .into_iter()
    .map(|(layer, name, sha)| {
        let reader = R2_2LayerCandidateReader::open(layer, &root.join(name), layout, sha)
            .expect("D4 frozen group32 artifact");
        (layer, reader)
    })
    .collect()
}
fn layer24_hybrid_routed(
    hidden_states: clr_core::TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: crate::Qwen3MoeConfig,
    reader: &mut R1_1CandidateReader,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
    backend: R2_1PackedProjectionBackend,
) -> Result<Tensor, StreamingModelError> {
    let hidden = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    combine_routed_experts(hidden_states, router, config, |expert_id, occurrences| {
        let packed = reader.load_expert(expert_id)?;
        let f32 = r2_d3_load_f32_expert(24, expert_id, store, layout)?;
        let mut outputs = Vec::with_capacity(occurrences.len());
        for &(token, _) in occurrences {
            let input = &hidden_states.data()[token * hidden..(token + 1) * hidden];
            let trace = crate::block::expert_mlp_trace(
                input,
                &f32.gate,
                &f32.up,
                &f32.down,
                hidden,
                intermediate,
            );
            let output = packed.r2_d4_apply_down_with_backend(&trace.activated_product, backend)?;
            outputs.push(output);
        }
        Ok(outputs)
    })
}
#[allow(clippy::too_many_arguments)]
fn packed_layer_candidate(
    layer: usize,
    hidden_states: clr_core::TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: crate::Qwen3MoeConfig,
    readers: &mut HashMap<usize, R2_2LayerCandidateReader>,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
    backend: R2_1PackedProjectionBackend,
) -> Tensor {
    let reader = readers
        .get_mut(&layer)
        .expect("D4 packed layer reader")
        .reader_mut_for_layer(layer)
        .expect("D4 packed layer identity");
    match layer {
        0 => r2_1_routed_experts_with_backend(hidden_states, router, config, reader, backend)
            .expect("D4 Layer0 packed candidate"),
        24 => layer24_hybrid_routed(
            hidden_states,
            router,
            config,
            reader,
            store,
            layout,
            backend,
        )
        .expect("D4 Layer24 hybrid candidate"),
        _ => panic!("D4 unexpected packed layer {layer}"),
    }
}

fn update_max(map: &mut HashMap<usize, f32>, layer: usize, value: f32) {
    assert!(value.is_finite());
    let maximum = map.entry(layer).or_insert(0.0);
    *maximum = maximum.max(value);
}
fn run_fixture(
    fixture: &TierBReference,
    readers: &mut HashMap<usize, R2_2LayerCandidateReader>,
    backend: R2_1PackedProjectionBackend,
    compare_native_scalar: bool,
) -> D4Run {
    assert!(matches!(
        fixture.name.as_str(),
        "short_english" | "short_thai"
    ));
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("D4 runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open D4 dense payload");
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut normal_store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    let processed_positions = fixture.token_ids.len() + R1_2_GENERATED_TOKEN_COUNT - 1;
    let mut cache = KvCache::new(48, processed_positions, 4, 128).expect("D4 fixed KV cache");
    let mut sequence = fixture.token_ids.clone();
    let mut generated_ids = Vec::with_capacity(R1_2_GENERATED_TOKEN_COUNT);
    let mut prompt_top20_ids = None;
    let mut prompt_argmax = None;
    let mut prompt_guard_ids = HashMap::<usize, Vec<usize>>::new();
    let mut prompt_logits = None;
    let mut prompt_logits_sha256 = None;
    let mut prompt_final_norm_sha256 = None;
    let mut prompt_fixed_logit_error = None;
    let mut prompt_top20_logit_error = None;
    let mut candidate_f32_layer_max_abs = HashMap::<usize, f32>::new();
    let mut native_scalar_layer_max_abs = HashMap::<usize, f32>::new();

    for position in 0..processed_positions {
        let token_id = *sequence
            .get(position)
            .expect("D4 generated token available before position");
        assert_eq!(cache.len(), position, "D4 cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("D4 KV layer"),
            )
            .expect("D4 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("D4 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4 post-attention norm");
            let router =
                route_tokens(post_norm.view(), weights.router.view(), config).expect("D4 router");
            if position + 1 == fixture.token_ids.len() && GENERATION_GUARD_LAYERS.contains(&layer) {
                prompt_guard_ids.insert(layer, router.selected_experts.clone());
            }

            let moe = if PACKED_LAYERS.contains(&layer) {
                let f32 = streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut normal_store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("D4 F32 comparator");
                let candidate = packed_layer_candidate(
                    layer,
                    post_norm.view(),
                    &router,
                    config,
                    readers,
                    &mut normal_store,
                    expert_layout,
                    backend,
                );
                update_max(
                    &mut candidate_f32_layer_max_abs,
                    layer,
                    max_abs(candidate.data(), f32.data()),
                );
                if compare_native_scalar {
                    let scalar = packed_layer_candidate(
                        layer,
                        post_norm.view(),
                        &router,
                        config,
                        readers,
                        &mut normal_store,
                        expert_layout,
                        R2_1PackedProjectionBackend::Scalar,
                    );
                    update_max(
                        &mut native_scalar_layer_max_abs,
                        layer,
                        max_abs(candidate.data(), scalar.data()),
                    );
                }
                candidate
            } else {
                streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut normal_store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("D4 canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("D4 block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("D4 KV append");

        if position + 1 >= fixture.token_ids.len() {
            let normalized = rms_norm(
                hidden.view(),
                final_norm_weight.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4 final norm");
            let logits = streaming_language_model_head(
                &mut payload,
                &final_plan,
                &normalized,
                &mut dense_bytes_read,
            );
            assert!(logits.data().iter().all(|value| value.is_finite()));
            let greedy = greedy_token(logits.view()).expect("D4 finite greedy logits");
            if position + 1 == fixture.token_ids.len() {
                prompt_top20_ids = Some(deterministic_top_ids(&logits, 20));
                prompt_argmax = Some(greedy);
                prompt_logits = Some(logits.data().to_vec());
                prompt_logits_sha256 = Some(f32_little_endian_sha256(logits.data()));
                prompt_final_norm_sha256 = Some(f32_little_endian_sha256(normalized.data()));
                prompt_fixed_logit_error = Some(maximum_indexed_difference(
                    logits.data(),
                    &fixture.fixed_logit_indices,
                    &fixture.fixed_logits,
                ));
                prompt_top20_logit_error = Some(maximum_indexed_difference(
                    logits.data(),
                    &fixture.top20_ids,
                    &fixture.top20_logits,
                ));
            }
            generated_ids.push(greedy);
            if generated_ids.len() < R1_2_GENERATED_TOKEN_COUNT {
                sequence.push(greedy);
            }
        }
    }

    assert_eq!(generated_ids.len(), R1_2_GENERATED_TOKEN_COUNT);
    assert_eq!(cache.len(), processed_positions);
    D4Run {
        prompt_top20_ids: prompt_top20_ids.expect("D4 prompt top20"),
        prompt_argmax: prompt_argmax.expect("D4 prompt argmax"),
        prompt_guard_ids,
        prompt_logits: prompt_logits.expect("D4 prompt logits"),
        prompt_logits_sha256: prompt_logits_sha256.expect("D4 prompt logits hash"),
        prompt_final_norm_sha256: prompt_final_norm_sha256.expect("D4 prompt norm hash"),
        prompt_fixed_logit_error: prompt_fixed_logit_error.expect("D4 fixed-logit error"),
        prompt_top20_logit_error: prompt_top20_logit_error.expect("D4 top20-logit error"),
        generated_ids,
        candidate_f32_layer_max_abs,
        native_scalar_layer_max_abs,
    }
}

fn assert_frozen_semantics(fixture: &TierBReference, frozen: &R1_2FrozenReference, run: &D4Run) {
    assert_eq!(run.generated_ids, frozen.generated_ids);
    assert_eq!(run.prompt_top20_ids, frozen.prompt_top20_ids);
    assert_eq!(run.prompt_argmax, frozen.prompt_argmax);
    for layer in GENERATION_GUARD_LAYERS {
        assert_eq!(
            run.prompt_guard_ids[&layer],
            frozen.prompt_guard_ids[&layer]
        );
        assert_eq!(run.prompt_guard_ids[&layer], fixture.guard_ids[&layer]);
    }
    let observed_f32 = run
        .prompt_fixed_logit_error
        .max(run.prompt_top20_logit_error);
    let allowed_f32 = R1_2_LOGIT_MAX_ABS_ENVELOPE.min(fixture.margin / 4.0);
    assert!(
        observed_f32 <= allowed_f32,
        "{} D4 compact-logit error {observed_f32} exceeds {allowed_f32}",
        fixture.name,
    );
    assert!(run.prompt_logits.iter().all(|value| value.is_finite()));
}

fn assert_scalar_local(run: &D4Run, fixture: &str) {
    for layer in PACKED_LAYERS {
        let local = run.candidate_f32_layer_max_abs[&layer];
        assert!(
            local <= LOCAL_MAX_ABS_LIMIT,
            "{fixture} Layer-{layer} scalar-hybrid/F32 error {local} exceeds {LOCAL_MAX_ABS_LIMIT}",
        );
    }
}

fn assert_native_local(run: &D4Run, fixture: &str) {
    for layer in PACKED_LAYERS {
        let local = run.native_scalar_layer_max_abs[&layer];
        assert!(
            local <= LOCAL_MAX_ABS_LIMIT,
            "{fixture} Layer-{layer} native/scalar error {local} exceeds {LOCAL_MAX_ABS_LIMIT}",
        );
    }
}
#[test]
fn m6_3_r2_2_d4_hybrid_pass_held_out_quality() {
    assert!(
        crate::r2_native::avx2_fma_available(),
        "D4 host must expose AVX2+FMA"
    );
    let reference_path = PathBuf::from(
        env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R1.2 frozen reference path"),
    );
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_2_D4_OUTPUT").expect("D4 quality output path"));
    assert!(!output_path.exists(), "D4 quality output must be new");
    let references = r1_2_frozen_references(&reference_path);
    let mut readers = open_readers();
    let mut evidence = String::from(
        "fixture\tgenerated_ids\tprompt_argmax\tprompt_top20_ids\tguard_layer0_ids\tguard_layer24_ids\tguard_layer47_ids\tscalar_f32_layer0_max_abs\tscalar_f32_layer24_max_abs\tnative_scalar_layer0_max_abs\tnative_scalar_layer24_max_abs\tnative_scalar_prompt_logit_max_abs\tf32_fixed_logit_max_abs\tf32_top20_logit_max_abs\tf32_allowed_logit_max_abs\tnative_prompt_logits_sha256\tscalar_prompt_logits_sha256\tnative_prompt_norm_sha256\trepeatability\n",
    );

    for fixture in tier_b_references()
        .into_iter()
        .filter(|fixture| matches!(fixture.name.as_str(), "short_english" | "short_thai"))
    {
        let frozen = &references[&fixture.name];
        let scalar = run_fixture(
            &fixture,
            &mut readers,
            R2_1PackedProjectionBackend::Scalar,
            false,
        );
        let first = run_fixture(
            &fixture,
            &mut readers,
            R2_1PackedProjectionBackend::NativeAvx2Fma,
            true,
        );
        let second = run_fixture(
            &fixture,
            &mut readers,
            R2_1PackedProjectionBackend::NativeAvx2Fma,
            true,
        );
        for run in [&scalar, &first, &second] {
            assert_frozen_semantics(&fixture, frozen, run);
        }
        assert_scalar_local(&scalar, &fixture.name);
        assert_native_local(&first, &fixture.name);
        assert_native_local(&second, &fixture.name);

        assert_eq!(first.generated_ids, second.generated_ids);
        assert_eq!(first.prompt_top20_ids, second.prompt_top20_ids);
        assert_eq!(first.prompt_argmax, second.prompt_argmax);
        assert_eq!(first.prompt_guard_ids, second.prompt_guard_ids);
        assert_eq!(first.prompt_logits_sha256, second.prompt_logits_sha256);
        assert_eq!(
            first.prompt_final_norm_sha256,
            second.prompt_final_norm_sha256
        );
        assert_eq!(
            first.candidate_f32_layer_max_abs,
            second.candidate_f32_layer_max_abs
        );
        assert_eq!(
            first.native_scalar_layer_max_abs,
            second.native_scalar_layer_max_abs
        );
        let native_scalar_prompt = max_abs(&first.prompt_logits, &scalar.prompt_logits);
        let native_scalar_prompt_second = max_abs(&second.prompt_logits, &scalar.prompt_logits);
        assert!(native_scalar_prompt <= NATIVE_SCALAR_PROMPT_MAX_ABS);
        assert!(native_scalar_prompt_second <= NATIVE_SCALAR_PROMPT_MAX_ABS);
        assert_eq!(
            native_scalar_prompt.to_bits(),
            native_scalar_prompt_second.to_bits()
        );

        let allowed_f32 = R1_2_LOGIT_MAX_ABS_ENVELOPE.min(fixture.margin / 4.0);
        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\texact",
            fixture.name,
            comma_separated(&first.generated_ids),
            first.prompt_argmax,
            comma_separated(&first.prompt_top20_ids),
            comma_separated(&first.prompt_guard_ids[&0]),
            comma_separated(&first.prompt_guard_ids[&24]),
            comma_separated(&first.prompt_guard_ids[&47]),
            scalar.candidate_f32_layer_max_abs[&0],
            scalar.candidate_f32_layer_max_abs[&24],
            first.native_scalar_layer_max_abs[&0],
            first.native_scalar_layer_max_abs[&24],
            native_scalar_prompt,
            first.prompt_fixed_logit_error,
            first.prompt_top20_logit_error,
            allowed_f32,
            first.prompt_logits_sha256,
            scalar.prompt_logits_sha256,
            first.prompt_final_norm_sha256,
        )
        .expect("write D4 quality evidence");
    }

    fs::write(output_path, evidence).expect("write D4 quality evidence file");
}
