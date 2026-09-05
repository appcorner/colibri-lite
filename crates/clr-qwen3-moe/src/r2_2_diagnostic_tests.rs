use crate::r1_1_direct_candidate::R2_2LayerCandidateReader;

use super::*;

const SENTINEL_LAYERS: [usize; 3] = [0, 24, 47];
const TRACKED_TOKENS: [usize; 4] = [94482, 69440, 52388, 129_075];
const SENTINEL_ARTIFACTS: [(usize, &str, &str); 3] = [
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
    (
        47,
        "layer47-group32.bin",
        "a2a45a1f407d5cd2303286843001684fc7800953cd2b67995a2c5e62a9e66a33",
    ),
];

#[derive(Debug)]
struct D1Run {
    top20: Vec<usize>,
    argmax: usize,
    guards: HashMap<usize, Vec<usize>>,
    fixed_logit_error: f32,
    top20_logit_error: f32,
    logits_sha256: String,
    final_norm_sha256: String,
    tracked_logits: HashMap<usize, f32>,
    tracked_ranks: HashMap<usize, usize>,
    local_max_abs: HashMap<usize, f32>,
}

fn states() -> Vec<Vec<usize>> {
    vec![
        vec![],
        vec![0],
        vec![24],
        vec![47],
        vec![0, 24],
        vec![0, 47],
        vec![24, 47],
        vec![0, 24, 47],
    ]
}

fn state_label(active: &[usize]) -> String {
    if active.is_empty() {
        return "f32".to_owned();
    }
    active
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn open_readers() -> HashMap<usize, R2_2LayerCandidateReader> {
    let root =
        PathBuf::from(env::var_os("COLIBRI_R2_2_ARTIFACT_ROOT").expect("R2.2 artifact root"));
    let layout = R1_1PackedArtifactLayout::canonical(32).expect("R2.2 group32 layout");
    SENTINEL_ARTIFACTS
        .into_iter()
        .map(|(layer, name, sha256)| {
            let reader = R2_2LayerCandidateReader::open(layer, &root.join(name), layout, sha256)
                .expect("R2.2 frozen sentinel artifact");
            (layer, reader)
        })
        .collect()
}

fn run_state(
    fixture: &TierBReference,
    active: &[usize],
    readers: &mut HashMap<usize, R2_2LayerCandidateReader>,
) -> D1Run {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("D1 runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open D1 dense payload");
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
    let mut cache = KvCache::new(48, fixture.token_ids.len(), 4, 128).expect("D1 KV cache");
    let mut final_hidden = None;
    let mut guards = HashMap::<usize, Vec<usize>>::new();
    let mut local_max_abs = HashMap::<usize, f32>::new();

    for (position, &token_id) in fixture.token_ids.iter().enumerate() {
        assert_eq!(cache.len(), position, "D1 cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D1 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("D1 KV layer"),
            )
            .expect("D1 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("D1 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D1 post-attention norm");
            let router =
                route_tokens(post_norm.view(), weights.router.view(), config).expect("D1 router");
            if position + 1 == fixture.token_ids.len() && SENTINEL_LAYERS.contains(&layer) {
                guards.insert(layer, router.selected_experts.clone());
            }

            let moe = if active.contains(&layer) {
                let f32 = streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut normal_store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("D1 F32 comparator");
                let reader = readers
                    .get_mut(&layer)
                    .expect("D1 sentinel reader")
                    .reader_mut_for_layer(layer)
                    .expect("D1 layer identity");
                let candidate = r2_1_routed_experts_with_backend(
                    post_norm.view(),
                    &router,
                    config,
                    reader,
                    R2_1PackedProjectionBackend::Scalar,
                )
                .expect("D1 scalar group32");
                let local = max_abs(candidate.data(), f32.data());
                assert!(local.is_finite(), "D1 local error finite");
                let entry = local_max_abs.entry(layer).or_insert(0.0);
                *entry = entry.max(local);
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
                .expect("D1 canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("D1 block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("D1 KV append");
        final_hidden = Some(hidden);
    }

    let hidden = final_hidden.expect("D1 final hidden");
    let normalized = rms_norm(
        hidden.view(),
        final_norm_weight.view(),
        config.rms_norm_epsilon(),
    )
    .expect("D1 final norm");
    let logits = streaming_language_model_head(
        &mut payload,
        &final_plan,
        &normalized,
        &mut dense_bytes_read,
    );
    assert!(logits.data().iter().all(|value| value.is_finite()));
    let ranked = deterministic_top_ids(&logits, 151_936);
    let top20 = ranked[..20].to_vec();
    let mut tracked_logits = HashMap::new();
    let mut tracked_ranks = HashMap::new();
    for token in TRACKED_TOKENS {
        tracked_logits.insert(token, logits.data()[token]);
        tracked_ranks.insert(
            token,
            ranked
                .iter()
                .position(|&id| id == token)
                .expect("D1 tracked rank")
                + 1,
        );
    }
    D1Run {
        argmax: top20[0],
        top20,
        guards,
        fixed_logit_error: maximum_indexed_difference(
            logits.data(),
            &fixture.fixed_logit_indices,
            &fixture.fixed_logits,
        ),
        top20_logit_error: maximum_indexed_difference(
            logits.data(),
            &fixture.top20_ids,
            &fixture.top20_logits,
        ),
        logits_sha256: f32_little_endian_sha256(logits.data()),
        final_norm_sha256: f32_little_endian_sha256(normalized.data()),
        tracked_logits,
        tracked_ranks,
        local_max_abs,
    }
}

fn sorted_ids(values: &[usize]) -> Vec<usize> {
    let mut values = values.to_vec();
    values.sort_unstable();
    values
}

fn local_text(run: &D1Run, layer: usize) -> String {
    run.local_max_abs
        .get(&layer)
        .map_or_else(|| "NA".to_owned(), |value| format!("{value:.17e}"))
}

#[test]
fn m6_3_r2_2_d1_localize_short_thai_top20_drift() {
    let output_path = PathBuf::from(env::var_os("COLIBRI_R2_2_D1_OUTPUT").expect("D1 output path"));
    assert!(!output_path.exists(), "D1 output must be new");
    let reference_path =
        PathBuf::from(env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R1.2 reference path"));
    let frozen = r1_2_frozen_references(&reference_path);
    let frozen = &frozen["short_thai"];
    let fixture = tier_b_references()
        .into_iter()
        .find(|item| item.name == "short_thai")
        .expect("D1 short_thai fixture");
    assert_eq!(fixture.token_ids, frozen.token_ids);
    assert_eq!(fixture.top20_ids, frozen.prompt_top20_ids);
    assert_eq!(fixture.argmax, frozen.prompt_argmax);

    let mut readers = open_readers();
    let mut evidence = String::from(
        "state_index\tactive_layers\ttop20_ids\ttop20_exact\ttop20_set_exact\targmax\targmax_exact\tguard0_ids\tguard24_ids\tguard47_ids\tguard0_exact\tguard24_exact\tguard47_exact\tfixed_logit_max_abs\ttop20_logit_max_abs\tlocal_layer0_max_abs\tlocal_layer24_max_abs\tlocal_layer47_max_abs\tlogit_94482\trank_94482\tlogit_69440\trank_69440\tlogit_52388\trank_52388\tlogit_129075\trank_129075\tprompt_logits_sha256\tprompt_final_norm_sha256\n",
    );

    for (state_index, active) in states().iter().enumerate() {
        let run = run_state(&fixture, active, &mut readers);
        let top20_exact = run.top20 == frozen.prompt_top20_ids;
        let top20_set_exact = sorted_ids(&run.top20) == sorted_ids(&frozen.prompt_top20_ids);
        let argmax_exact = run.argmax == frozen.prompt_argmax;
        let guard0_exact = run.guards[&0] == frozen.prompt_guard_ids[&0];
        let guard24_exact = run.guards[&24] == frozen.prompt_guard_ids[&24];
        let guard47_exact = run.guards[&47] == frozen.prompt_guard_ids[&47];

        if active.is_empty() {
            assert!(top20_exact && argmax_exact);
            assert!(guard0_exact && guard24_exact && guard47_exact);
            assert_eq!(run.logits_sha256, frozen.prompt_logits_sha256);
            assert_eq!(run.final_norm_sha256, frozen.prompt_final_norm_sha256);
        }

        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\t{:.17e}\t{}\t{:.17e}\t{}\t{:.17e}\t{}\t{:.17e}\t{}\t{}\t{}",
            state_index,
            state_label(active),
            comma_separated(&run.top20),
            top20_exact,
            top20_set_exact,
            run.argmax,
            argmax_exact,
            comma_separated(&run.guards[&0]),
            comma_separated(&run.guards[&24]),
            comma_separated(&run.guards[&47]),
            guard0_exact,
            guard24_exact,
            guard47_exact,
            run.fixed_logit_error,
            run.top20_logit_error,
            local_text(&run, 0),
            local_text(&run, 24),
            local_text(&run, 47),
            run.tracked_logits[&94482],
            run.tracked_ranks[&94482],
            run.tracked_logits[&69440],
            run.tracked_ranks[&69440],
            run.tracked_logits[&52388],
            run.tracked_ranks[&52388],
            run.tracked_logits[&129_075],
            run.tracked_ranks[&129_075],
            run.logits_sha256,
            run.final_norm_sha256,
        )
        .expect("write D1 evidence");
    }
    fs::write(output_path, evidence).expect("write D1 evidence file");
}
