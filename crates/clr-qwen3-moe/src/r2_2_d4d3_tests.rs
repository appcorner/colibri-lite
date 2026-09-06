use crate::{
    block::combine_routed_experts,
    r1_1_direct_candidate::{R1_1CandidateReader, R2_2LayerCandidateReader},
    streaming::{StreamingModelError, r2_d3_load_f32_expert},
};

use super::*;

const TARGET_LAYER: usize = 24;
const GROUPS: [usize; 3] = [32, 16, 8];
const LOCAL_MAX_ABS_LIMIT: f32 = 0.001;

#[derive(Debug)]
struct D4D3Run {
    router_ids_by_position: Vec<Vec<usize>>,
    position_max_abs: Vec<f32>,
    candidate_finite: bool,
    greedy_ids: Vec<usize>,
    frozen_greedy_match: Vec<bool>,
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn insert_reader(
    readers: &mut HashMap<(usize, usize), R2_2LayerCandidateReader>,
    layer: usize,
    group_size: usize,
    root: &Path,
    name: &str,
    sha256: &str,
) {
    let layout = R1_1PackedArtifactLayout::canonical(group_size).expect("D4D3 packed layout");
    let reader = R2_2LayerCandidateReader::open(layer, &root.join(name), layout, sha256)
        .expect("D4D3 frozen artifact");
    assert!(readers.insert((layer, group_size), reader).is_none());
}

fn open_readers() -> HashMap<(usize, usize), R2_2LayerCandidateReader> {
    let group32_root = PathBuf::from(
        env::var_os("COLIBRI_R2_2_ARTIFACT_ROOT").expect("R2.2 group32 artifact root"),
    );
    let d2_root =
        PathBuf::from(env::var_os("COLIBRI_R2_2_D2_ARTIFACT_ROOT").expect("D2 artifact root"));
    let mut readers = HashMap::new();
    insert_reader(
        &mut readers,
        0,
        32,
        &group32_root,
        "layer00-group32-repro.bin",
        "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2",
    );
    insert_reader(
        &mut readers,
        TARGET_LAYER,
        32,
        &group32_root,
        "layer24-group32.bin",
        "890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0",
    );
    insert_reader(
        &mut readers,
        TARGET_LAYER,
        16,
        &d2_root,
        "layer24-group16.bin",
        "569b221f448b521488764da8b0e99058f8daded9f41ba10ef7f1f2706ac36416",
    );
    insert_reader(
        &mut readers,
        TARGET_LAYER,
        8,
        &d2_root,
        "layer24-group8.bin",
        "4d51c17a2ba040229a8c20c895c5be8f030ab273db3151ccb739df7289965435",
    );
    readers
}

fn hybrid_expert_bytes(group_size: usize) -> usize {
    match group_size {
        32 => 14_352_384,
        16 => 14_548_992,
        8 => 14_942_208,
        _ => unreachable!("frozen D4D3 group"),
    }
}

fn down_only_routed(
    hidden_states: clr_core::TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: crate::Qwen3MoeConfig,
    reader: &mut R1_1CandidateReader,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
) -> Result<Tensor, StreamingModelError> {
    let hidden = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    combine_routed_experts(hidden_states, router, config, |expert_id, occurrences| {
        let packed = reader.load_expert(expert_id)?;
        let f32 = r2_d3_load_f32_expert(TARGET_LAYER, expert_id, store, layout)?;
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
            outputs.push(packed.r2_d3_apply_down(&trace.activated_product)?);
        }
        Ok(outputs)
    })
}

fn run_group(
    fixture: &TierBReference,
    frozen: &R1_2FrozenReference,
    group_size: usize,
    readers: &mut HashMap<(usize, usize), R2_2LayerCandidateReader>,
) -> D4D3Run {
    assert!(GROUPS.contains(&group_size));
    assert_eq!(frozen.generated_ids.len(), R1_2_GENERATED_TOKEN_COUNT);
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("D4D3 runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload =
        File::open(artifact_root.join(&plan.payload)).expect("open D4D3 dense payload");
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
    let mut cache = KvCache::new(48, processed_positions, 4, 128).expect("D4D3 KV cache");
    let mut sequence = fixture.token_ids.clone();
    sequence.push(frozen.generated_ids[0]);
    assert_eq!(sequence.len(), processed_positions);
    let mut router_ids_by_position = Vec::with_capacity(processed_positions);
    let mut position_max_abs = Vec::with_capacity(processed_positions);
    let mut candidate_finite = true;
    let mut greedy_ids = Vec::with_capacity(R1_2_GENERATED_TOKEN_COUNT);
    let mut frozen_greedy_match = Vec::with_capacity(R1_2_GENERATED_TOKEN_COUNT);

    for (position, &token_id) in sequence.iter().enumerate() {
        assert_eq!(cache.len(), position, "D4D3 cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4D3 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("D4D3 KV layer"),
            )
            .expect("D4D3 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("D4D3 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4D3 post-attention norm");
            let router =
                route_tokens(post_norm.view(), weights.router.view(), config).expect("D4D3 router");

            let moe = if layer == TARGET_LAYER {
                router_ids_by_position.push(router.selected_experts.clone());
                let f32 = streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    TARGET_LAYER,
                    &mut normal_store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("D4D3 same-input F32 comparator");
                assert!(f32.data().iter().all(|value| value.is_finite()));
                let reader = readers
                    .get_mut(&(TARGET_LAYER, group_size))
                    .expect("D4D3 target reader")
                    .reader_mut_for_layer(TARGET_LAYER)
                    .expect("D4D3 target layer identity");
                let candidate = down_only_routed(
                    post_norm.view(),
                    &router,
                    config,
                    reader,
                    &mut normal_store,
                    expert_layout,
                )
                .expect("D4D3 down-only candidate");
                let finite = candidate.data().iter().all(|value| value.is_finite());
                candidate_finite &= finite;
                assert!(finite, "D4D3 candidate output must be finite");
                let local = max_abs(candidate.data(), f32.data());
                assert!(local.is_finite(), "D4D3 local error finite");
                position_max_abs.push(local);
                candidate
            } else if layer == 0 {
                let reader = readers
                    .get_mut(&(0, 32))
                    .expect("D4D3 Layer0 group32 reader")
                    .reader_mut_for_layer(0)
                    .expect("D4D3 Layer0 identity");
                let candidate = r2_1_routed_experts_with_backend(
                    post_norm.view(),
                    &router,
                    config,
                    reader,
                    R2_1PackedProjectionBackend::Scalar,
                )
                .expect("D4D3 Layer0 scalar group32");
                assert!(candidate.data().iter().all(|value| value.is_finite()));
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
                .expect("D4D3 canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("D4D3 block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("D4D3 KV append");

        if position + 1 >= fixture.token_ids.len() {
            let normalized = rms_norm(
                hidden.view(),
                final_norm_weight.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4D3 final norm");
            let logits = streaming_language_model_head(
                &mut payload,
                &final_plan,
                &normalized,
                &mut dense_bytes_read,
            );
            assert!(logits.data().iter().all(|value| value.is_finite()));
            let greedy = greedy_token(logits.view()).expect("D4D3 finite greedy logits");
            let generated_index = position + 1 - fixture.token_ids.len();
            greedy_ids.push(greedy);
            frozen_greedy_match.push(greedy == frozen.generated_ids[generated_index]);
        }
    }

    assert_eq!(router_ids_by_position.len(), processed_positions);
    assert_eq!(position_max_abs.len(), processed_positions);
    assert_eq!(greedy_ids.len(), R1_2_GENERATED_TOKEN_COUNT);
    assert_eq!(frozen_greedy_match.len(), R1_2_GENERATED_TOKEN_COUNT);
    D4D3Run {
        router_ids_by_position,
        position_max_abs,
        candidate_finite,
        greedy_ids,
        frozen_greedy_match,
    }
}
fn ids_by_position_text(values: &[Vec<usize>]) -> String {
    values
        .iter()
        .map(|ids| comma_separated(ids))
        .collect::<Vec<_>>()
        .join("|")
}

fn errors_text(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn m6_3_r2_2_d4d3_characterize_sequence_aware_layer24_down_precision() {
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_2_D4D3_OUTPUT").expect("D4D3 evidence output path"));
    assert!(!output_path.exists(), "D4D3 evidence output must be new");
    let reference_path =
        PathBuf::from(env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R1.2 reference path"));
    let frozen = r1_2_frozen_references(&reference_path);
    let fixtures = tier_b_references()
        .into_iter()
        .filter(|fixture| matches!(fixture.name.as_str(), "short_english" | "short_thai"))
        .collect::<Vec<_>>();
    assert_eq!(fixtures.len(), 2);

    let mut readers = open_readers();
    let mut evidence = String::from(
        "fixture\tgroup_size\thybrid_expert_bytes\thybrid_layer_bytes\ttrajectory_positions\trouter_ids_by_position\tprompt_guard_exact\tposition_max_abs\tlocal_max_abs\tlocal_limit\tpass_local\tcandidate_finite\tgreedy_ids\tfrozen_greedy_match_by_position\tfrozen_greedy_match_all\n",
    );

    for fixture in fixtures {
        let frozen_fixture = &frozen[&fixture.name];
        assert_eq!(fixture.token_ids, frozen_fixture.token_ids);
        for group_size in GROUPS {
            let run = run_group(&fixture, frozen_fixture, group_size, &mut readers);
            let prompt_router = &run.router_ids_by_position[fixture.token_ids.len() - 1];
            let prompt_guard_exact =
                prompt_router == &frozen_fixture.prompt_guard_ids[&TARGET_LAYER];
            let local_max_abs = run.position_max_abs.iter().copied().fold(0.0_f32, f32::max);
            let pass_local = local_max_abs <= LOCAL_MAX_ABS_LIMIT;
            let frozen_greedy_match_all = run.frozen_greedy_match.iter().all(|value| *value);
            writeln!(
                evidence,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\t{}\t{}",
                fixture.name,
                group_size,
                hybrid_expert_bytes(group_size),
                hybrid_expert_bytes(group_size) * 128,
                run.position_max_abs.len(),
                ids_by_position_text(&run.router_ids_by_position),
                prompt_guard_exact,
                errors_text(&run.position_max_abs),
                local_max_abs,
                LOCAL_MAX_ABS_LIMIT,
                pass_local,
                run.candidate_finite,
                comma_separated(&run.greedy_ids),
                run.frozen_greedy_match
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                frozen_greedy_match_all,
            )
            .expect("write D4D3 evidence row");
        }
    }

    fs::write(output_path, evidence).expect("write D4D3 evidence file");
}
