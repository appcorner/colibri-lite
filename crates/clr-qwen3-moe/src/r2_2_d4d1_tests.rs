use crate::{
    block::combine_routed_experts,
    r1_1_direct_candidate::{R1_1CandidateReader, R2_2LayerCandidateReader},
    streaming::{StreamingModelError, r2_d3_load_f32_expert},
};

use super::*;

const TARGET_LAYER: usize = 24;
const GROUPS: [usize; 3] = [32, 16, 8];
const LOCAL_MAX_ABS_LIMIT: f32 = 0.001;

#[derive(Debug, Clone, Copy)]
struct D4D1Context {
    id: &'static str,
    layer0_group32: bool,
}

const CONTEXTS: [D4D1Context; 2] = [
    D4D1Context {
        id: "f32_prefix",
        layer0_group32: false,
    },
    D4D1Context {
        id: "layer0_group32_prefix",
        layer0_group32: true,
    },
];

#[derive(Debug)]
struct D4D1Run {
    router_ids_by_position: Vec<Vec<usize>>,
    position_max_abs: HashMap<usize, Vec<f32>>,
    candidate_finite: HashMap<usize, bool>,
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
    let layout = R1_1PackedArtifactLayout::canonical(group_size).expect("D4D1 packed layout");
    let reader = R2_2LayerCandidateReader::open(layer, &root.join(name), layout, sha256)
        .expect("D4D1 frozen artifact");
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
        _ => unreachable!("frozen D4D1 group"),
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

fn run_context(
    fixture: &TierBReference,
    context: D4D1Context,
    readers: &mut HashMap<(usize, usize), R2_2LayerCandidateReader>,
) -> D4D1Run {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("D4D1 runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload =
        File::open(artifact_root.join(&plan.payload)).expect("open D4D1 dense payload");
    let mut dense_bytes_read = 0_u64;
    let mut normal_store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    let mut cache = KvCache::new(48, fixture.token_ids.len(), 4, 128).expect("D4D1 KV cache");
    let mut router_ids_by_position = Vec::with_capacity(fixture.token_ids.len());
    let mut position_max_abs = GROUPS
        .into_iter()
        .map(|group| (group, Vec::with_capacity(fixture.token_ids.len())))
        .collect::<HashMap<_, _>>();
    let mut candidate_finite = GROUPS
        .into_iter()
        .map(|group| (group, true))
        .collect::<HashMap<_, _>>();

    for (position, &token_id) in fixture.token_ids.iter().enumerate() {
        assert_eq!(cache.len(), position, "D4D1 cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4D1 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("D4D1 KV layer"),
            )
            .expect("D4D1 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("D4D1 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D4D1 post-attention norm");
            let router =
                route_tokens(post_norm.view(), weights.router.view(), config).expect("D4D1 router");

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
                .expect("D4D1 F32 comparator");
                assert!(f32.data().iter().all(|value| value.is_finite()));

                for group_size in GROUPS {
                    let reader = readers
                        .get_mut(&(TARGET_LAYER, group_size))
                        .expect("D4D1 target reader")
                        .reader_mut_for_layer(TARGET_LAYER)
                        .expect("D4D1 target layer identity");
                    let candidate = down_only_routed(
                        post_norm.view(),
                        &router,
                        config,
                        reader,
                        &mut normal_store,
                        expert_layout,
                    )
                    .expect("D4D1 down-only candidate");
                    let finite = candidate.data().iter().all(|value| value.is_finite());
                    candidate_finite
                        .entry(group_size)
                        .and_modify(|value| *value &= finite);
                    assert!(finite, "D4D1 candidate output must be finite");
                    let local = max_abs(candidate.data(), f32.data());
                    assert!(local.is_finite(), "D4D1 local error finite");
                    position_max_abs
                        .get_mut(&group_size)
                        .expect("D4D1 group error vector")
                        .push(local);
                }
                f32
            } else if context.layer0_group32 && layer == 0 {
                let reader = readers
                    .get_mut(&(0, 32))
                    .expect("D4D1 Layer0 group32 reader")
                    .reader_mut_for_layer(0)
                    .expect("D4D1 Layer0 identity");
                let candidate = r2_1_routed_experts_with_backend(
                    post_norm.view(),
                    &router,
                    config,
                    reader,
                    R2_1PackedProjectionBackend::Scalar,
                )
                .expect("D4D1 Layer0 scalar group32");
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
                .expect("D4D1 canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("D4D1 block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("D4D1 KV append");
    }

    assert_eq!(router_ids_by_position.len(), fixture.token_ids.len());
    for group_size in GROUPS {
        assert_eq!(position_max_abs[&group_size].len(), fixture.token_ids.len());
    }
    D4D1Run {
        router_ids_by_position,
        position_max_abs,
        candidate_finite,
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
fn m6_3_r2_2_d4d1_characterize_layer24_down_precision() {
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_2_D4D1_OUTPUT").expect("D4D1 evidence output path"));
    assert!(!output_path.exists(), "D4D1 evidence output must be new");
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
        "fixture\tcontext_index\tcontext_id\tlayer0_group32_prefix\tgroup_size\thybrid_expert_bytes\thybrid_layer_bytes\tprompt_positions\trouter_ids_by_position\tfinal_guard_exact\tposition_max_abs\tlocal_max_abs\tlocal_limit\tpass_local\tcandidate_finite\n",
    );

    for fixture in fixtures {
        let frozen_fixture = &frozen[&fixture.name];
        assert_eq!(fixture.token_ids, frozen_fixture.token_ids);
        for (context_index, context) in CONTEXTS.iter().copied().enumerate() {
            let run = run_context(&fixture, context, &mut readers);
            let final_router = run
                .router_ids_by_position
                .last()
                .expect("D4D1 final Layer24 router IDs");
            let final_guard_exact = final_router == &frozen_fixture.prompt_guard_ids[&TARGET_LAYER];

            for group_size in GROUPS {
                let errors = &run.position_max_abs[&group_size];
                let local_max_abs = errors.iter().copied().fold(0.0_f32, f32::max);
                let pass_local = local_max_abs <= LOCAL_MAX_ABS_LIMIT;
                writeln!(
                    evidence,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{}\t{}",
                    fixture.name,
                    context_index,
                    context.id,
                    context.layer0_group32,
                    group_size,
                    hybrid_expert_bytes(group_size),
                    hybrid_expert_bytes(group_size) * 128,
                    fixture.token_ids.len(),
                    ids_by_position_text(&run.router_ids_by_position),
                    final_guard_exact,
                    errors_text(errors),
                    local_max_abs,
                    LOCAL_MAX_ABS_LIMIT,
                    pass_local,
                    run.candidate_finite[&group_size],
                )
                .expect("write D4D1 evidence row");
            }
        }
    }

    fs::write(output_path, evidence).expect("write D4D1 evidence file");
}
