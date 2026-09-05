use crate::r1_1_direct_candidate::R2_2LayerCandidateReader;

use super::*;

const GROUPS: [usize; 3] = [32, 16, 8];
const LOCAL_MAX_ABS_LIMIT: f32 = 0.001;

#[derive(Debug, Clone)]
struct D2Context {
    id: &'static str,
    target_layer: usize,
    prefix_group32_layers: Vec<usize>,
}

#[derive(Debug)]
struct D2ContextRun {
    router_ids_by_position: Vec<Vec<usize>>,
    position_max_abs: HashMap<usize, Vec<f32>>,
    candidate_finite: HashMap<usize, bool>,
}

fn contexts() -> Vec<D2Context> {
    vec![
        D2Context {
            id: "layer24_f32_prefix",
            target_layer: 24,
            prefix_group32_layers: vec![],
        },
        D2Context {
            id: "layer24_layer0_group32_prefix",
            target_layer: 24,
            prefix_group32_layers: vec![0],
        },
        D2Context {
            id: "layer47_f32_prefix",
            target_layer: 47,
            prefix_group32_layers: vec![],
        },
        D2Context {
            id: "layer47_r2_2_failing_prefix",
            target_layer: 47,
            prefix_group32_layers: vec![0, 24],
        },
    ]
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
    root: &std::path::Path,
    name: &str,
    sha256: &str,
) {
    let layout = R1_1PackedArtifactLayout::canonical(group_size).expect("D2 packed layout");
    let reader = R2_2LayerCandidateReader::open(layer, &root.join(name), layout, sha256)
        .expect("D2 frozen artifact");
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
        24,
        32,
        &group32_root,
        "layer24-group32.bin",
        "890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0",
    );
    insert_reader(
        &mut readers,
        47,
        32,
        &group32_root,
        "layer47-group32.bin",
        "a2a45a1f407d5cd2303286843001684fc7800953cd2b67995a2c5e62a9e66a33",
    );
    insert_reader(
        &mut readers,
        24,
        16,
        &d2_root,
        "layer24-group16.bin",
        "569b221f448b521488764da8b0e99058f8daded9f41ba10ef7f1f2706ac36416",
    );
    insert_reader(
        &mut readers,
        24,
        8,
        &d2_root,
        "layer24-group8.bin",
        "4d51c17a2ba040229a8c20c895c5be8f030ab273db3151ccb739df7289965435",
    );
    insert_reader(
        &mut readers,
        47,
        16,
        &d2_root,
        "layer47-group16.bin",
        "ab11ed88973d394e01e3430b9607f0b42c2752cd1631ed8eff2c0c716a639c19",
    );
    insert_reader(
        &mut readers,
        47,
        8,
        &d2_root,
        "layer47-group8.bin",
        "7a57ce8bcf1d05b0d82491c88e644c994d2adafedf0285596a62da81c29c200d",
    );
    readers
}
fn run_context(
    fixture: &TierBReference,
    context: &D2Context,
    readers: &mut HashMap<(usize, usize), R2_2LayerCandidateReader>,
) -> D2ContextRun {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("D2 runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open D2 dense payload");
    let mut dense_bytes_read = 0_u64;
    let mut normal_store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    let mut cache = KvCache::new(48, fixture.token_ids.len(), 4, 128).expect("D2 KV cache");
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
        assert_eq!(cache.len(), position, "D2 cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D2 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("D2 KV layer"),
            )
            .expect("D2 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("D2 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D2 post-attention norm");
            let router =
                route_tokens(post_norm.view(), weights.router.view(), config).expect("D2 router");
            let moe = if layer == context.target_layer {
                router_ids_by_position.push(router.selected_experts.clone());
                let f32 = streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut normal_store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("D2 F32 comparator");
                assert!(f32.data().iter().all(|value| value.is_finite()));

                for group_size in GROUPS {
                    let reader = readers
                        .get_mut(&(layer, group_size))
                        .expect("D2 target reader")
                        .reader_mut_for_layer(layer)
                        .expect("D2 target layer identity");
                    let candidate = r2_1_routed_experts_with_backend(
                        post_norm.view(),
                        &router,
                        config,
                        reader,
                        R2_1PackedProjectionBackend::Scalar,
                    )
                    .expect("D2 scalar candidate");
                    let finite = candidate.data().iter().all(|value| value.is_finite());
                    candidate_finite.insert(group_size, finite);
                    assert!(finite, "D2 candidate output must be finite");
                    let local = max_abs(candidate.data(), f32.data());
                    assert!(local.is_finite(), "D2 local error finite");
                    position_max_abs
                        .get_mut(&group_size)
                        .expect("D2 group error vector")
                        .push(local);
                }
                f32
            } else if context.prefix_group32_layers.contains(&layer) {
                let reader = readers
                    .get_mut(&(layer, 32))
                    .expect("D2 prefix group32 reader")
                    .reader_mut_for_layer(layer)
                    .expect("D2 prefix layer identity");
                let candidate = r2_1_routed_experts_with_backend(
                    post_norm.view(),
                    &router,
                    config,
                    reader,
                    R2_1PackedProjectionBackend::Scalar,
                )
                .expect("D2 prefix scalar group32");
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
                .expect("D2 canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("D2 block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("D2 KV append");
    }

    assert_eq!(router_ids_by_position.len(), fixture.token_ids.len());
    for group_size in GROUPS {
        assert_eq!(position_max_abs[&group_size].len(), fixture.token_ids.len());
    }
    D2ContextRun {
        router_ids_by_position,
        position_max_abs,
        candidate_finite,
    }
}
fn prefix_label(layers: &[usize]) -> String {
    if layers.is_empty() {
        "f32".to_owned()
    } else {
        comma_separated(layers)
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
fn m6_3_r2_2_d2_characterize_depth_sensitive_precision() {
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_2_D2_OUTPUT").expect("D2 evidence output path"));
    assert!(!output_path.exists(), "D2 evidence output must be new");
    let reference_path =
        PathBuf::from(env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R1.2 reference path"));
    let frozen = r1_2_frozen_references(&reference_path);
    let frozen = &frozen["short_thai"];
    let fixture = tier_b_references()
        .into_iter()
        .find(|item| item.name == "short_thai")
        .expect("D2 short_thai fixture");
    assert_eq!(fixture.token_ids, frozen.token_ids);

    let mut readers = open_readers();
    let mut evidence = String::from(
        "context_index\tlayer\tcontext_id\tprefix_group32_layers\tgroup_size\tartifact_bytes\texpert_bytes\tprompt_positions\trouter_ids_by_position\tfinal_guard_exact\tposition_max_abs\tlocal_max_abs\tlocal_limit\tpass_local\tcandidate_finite\n",
    );

    for (context_index, context) in contexts().iter().enumerate() {
        let run = run_context(&fixture, context, &mut readers);
        let final_router = run
            .router_ids_by_position
            .last()
            .expect("D2 final target router IDs");
        let final_guard_exact = final_router == &frozen.prompt_guard_ids[&context.target_layer];

        for group_size in GROUPS {
            let errors = &run.position_max_abs[&group_size];
            let local_max_abs = errors.iter().copied().fold(0.0_f32, f32::max);
            let pass_local = local_max_abs <= LOCAL_MAX_ABS_LIMIT;
            let layout =
                R1_1PackedArtifactLayout::canonical(group_size).expect("D2 evidence packed layout");
            writeln!(
                evidence,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{}\t{}",
                context_index,
                context.target_layer,
                context.id,
                prefix_label(&context.prefix_group32_layers),
                group_size,
                layout.artifact_bytes().expect("D2 artifact bytes"),
                layout.expert_bytes().expect("D2 expert bytes"),
                fixture.token_ids.len(),
                ids_by_position_text(&run.router_ids_by_position),
                final_guard_exact,
                errors_text(errors),
                local_max_abs,
                LOCAL_MAX_ABS_LIMIT,
                pass_local,
                run.candidate_finite[&group_size],
            )
            .expect("write D2 evidence row");
        }
    }
    fs::write(output_path, evidence).expect("write D2 evidence file");
}
