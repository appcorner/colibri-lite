use clr_core::RuntimeError;

use crate::{
    block::combine_routed_experts,
    r1_1_direct_candidate::{R1_1CandidateReader, R1_1PackedExpert, R2_2LayerCandidateReader},
    streaming::{R2D3F32Expert, r2_d3_load_f32_expert},
};

use super::*;

const LOCAL_MAX_ABS_LIMIT: f32 = 0.001;
const GROUPS: [usize; 3] = [32, 16, 8];
const F32_PROJECTION_BYTES: usize = 6_291_456;
const F32_EXPERT_BYTES: usize = 18_874_368;
const EXPERTS_PER_LAYER: usize = 128;

#[derive(Debug, Clone, Copy)]
struct Variant {
    id: &'static str,
    gate: bool,
    up: bool,
    down: bool,
}

const VARIANTS: [Variant; 7] = [
    Variant {
        id: "gate_up_down",
        gate: true,
        up: true,
        down: true,
    },
    Variant {
        id: "gate_up",
        gate: true,
        up: true,
        down: false,
    },
    Variant {
        id: "gate_down",
        gate: true,
        up: false,
        down: true,
    },
    Variant {
        id: "up_down",
        gate: false,
        up: true,
        down: true,
    },
    Variant {
        id: "gate",
        gate: true,
        up: false,
        down: false,
    },
    Variant {
        id: "up",
        gate: false,
        up: true,
        down: false,
    },
    Variant {
        id: "down",
        gate: false,
        up: false,
        down: true,
    },
];

#[derive(Debug)]
struct HybridExpert {
    f32: R2D3F32Expert,
    packed: R1_1PackedExpert,
}

#[derive(Debug)]
struct SameInputRun {
    max_abs: HashMap<&'static str, f32>,
    finite: HashMap<&'static str, bool>,
    control_first_token_exact: bool,
}

#[derive(Debug)]
struct SequenceRun {
    max_abs: f32,
    finite: bool,
    prompt_first_token_exact: bool,
    router_ids_by_position: Vec<Vec<usize>>,
    trace_sha256: String,
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn f32_projection(input: &[f32], weight: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    assert_eq!(input.len(), columns);
    assert_eq!(weight.len(), rows * columns);
    (0..rows)
        .map(|row| {
            let start = row * columns;
            input
                .iter()
                .zip(&weight[start..start + columns])
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

fn packed_projection_bytes(group_size: usize) -> usize {
    assert!(GROUPS.contains(&group_size));
    1_572_864 + F32_PROJECTION_BYTES / group_size
}

fn packed_count(variant: Variant) -> usize {
    usize::from(variant.gate) + usize::from(variant.up) + usize::from(variant.down)
}

fn logical_expert_bytes(variant: Variant, group_size: usize) -> usize {
    let packed = packed_count(variant);
    packed * packed_projection_bytes(group_size) + (3 - packed) * F32_PROJECTION_BYTES
}

fn hybrid_expert_output(
    input: &[f32],
    expert: &HybridExpert,
    variant: Variant,
    hidden: usize,
    intermediate: usize,
) -> Result<Vec<f32>, RuntimeError> {
    let gate = if variant.gate {
        expert.packed.r2_d3_apply_gate(input)?
    } else {
        f32_projection(input, &expert.f32.gate, intermediate, hidden)
    };
    let up = if variant.up {
        expert.packed.r2_d3_apply_up(input)?
    } else {
        f32_projection(input, &expert.f32.up, intermediate, hidden)
    };
    let activated = gate
        .into_iter()
        .zip(up)
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    if variant.down {
        expert.packed.r2_d3_apply_down(&activated)
    } else {
        Ok(f32_projection(
            &activated,
            &expert.f32.down,
            hidden,
            intermediate,
        ))
    }
}

fn hybrid_routed(
    hidden_states: clr_core::TensorView<'_>,
    router: &crate::block::RouterOutput,
    config: crate::Qwen3MoeConfig,
    experts: &HashMap<usize, HybridExpert>,
    variant: Variant,
) -> Result<Tensor, RuntimeError> {
    let hidden = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    combine_routed_experts(hidden_states, router, config, |expert_id, occurrences| {
        let expert = experts
            .get(&expert_id)
            .expect("R2.3b selected expert loaded");
        let mut outputs = Vec::with_capacity(occurrences.len());
        for &(token, _) in occurrences {
            let input = &hidden_states.data()[token * hidden..(token + 1) * hidden];
            outputs.push(hybrid_expert_output(
                input,
                expert,
                variant,
                hidden,
                intermediate,
            )?);
        }
        Ok(outputs)
    })
}

fn load_hybrid_experts(
    layer: usize,
    router: &crate::block::RouterOutput,
    reader: &mut R1_1CandidateReader,
    store: &mut ExpertStore,
    layout: PackedExpertLayout,
) -> HashMap<usize, HybridExpert> {
    let mut unique = router.selected_experts.clone();
    unique.sort_unstable();
    unique.dedup();
    unique
        .into_iter()
        .map(|expert_id| {
            let f32 = r2_d3_load_f32_expert(layer, expert_id, store, layout)
                .expect("R2.3b canonical F32 expert");
            let packed = reader.load_expert(expert_id).expect("R2.3b packed expert");
            (expert_id, HybridExpert { f32, packed })
        })
        .collect()
}

fn frozen_first_tokens(path: &Path) -> HashMap<String, usize> {
    let source = fs::read_to_string(path).expect("read R2.3a frozen reference");
    let mut lines = source.lines();
    let header = lines
        .next()
        .expect("R2.3a header")
        .split('\t')
        .collect::<Vec<_>>();
    let fixture_index = header
        .iter()
        .position(|name| *name == "fixture")
        .expect("fixture column");
    let generated_index = header
        .iter()
        .position(|name| *name == "generated_ids")
        .expect("generated_ids column");
    let mut result = HashMap::new();
    for line in lines {
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), header.len(), "R2.3a row width");
        let first = fields[generated_index]
            .split(',')
            .next()
            .expect("R2.3a first generated token")
            .parse::<usize>()
            .expect("R2.3a generated token integer");
        assert!(
            result
                .insert(fields[fixture_index].to_owned(), first)
                .is_none()
        );
    }
    assert_eq!(result.len(), 2, "R2.3a bilingual reference");
    result
}

fn open_reader(layer: usize, group_size: usize) -> R2_2LayerCandidateReader {
    let path =
        PathBuf::from(env::var_os("COLIBRI_R2_3B_ARTIFACT_PATH").expect("R2.3b artifact path"));
    let sha256 = env::var("COLIBRI_R2_3B_ARTIFACT_SHA256").expect("R2.3b artifact SHA-256");
    let layout = R1_1PackedArtifactLayout::canonical(group_size).expect("R2.3b packed layout");
    R2_2LayerCandidateReader::open(layer, &path, layout, &sha256)
        .expect("R2.3b verified layer artifact")
}

fn run_same_input(
    fixture: &TierBReference,
    expected_first_token: usize,
    target_layer: usize,
    reader: &mut R2_2LayerCandidateReader,
) -> SameInputRun {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("R2.3b canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.3b runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("R2.3b dense payload");
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * EXPERTS_PER_LAYER,
    );
    let mut cache = KvCache::new(48, fixture.token_ids.len(), 4, 128).expect("R2.3b KV cache");
    let mut maxima = VARIANTS
        .into_iter()
        .map(|variant| (variant.id, 0.0_f32))
        .collect::<HashMap<_, _>>();
    let mut finite = VARIANTS
        .into_iter()
        .map(|variant| (variant.id, true))
        .collect::<HashMap<_, _>>();
    let mut control_first_token_exact = false;

    for (position, &token_id) in fixture.token_ids.iter().enumerate() {
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3b input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("R2.3b KV layer"),
            )
            .expect("R2.3b attention");
            let residual =
                elementwise_add(hidden.view(), attention.output.view()).expect("R2.3b residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3b post norm");
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .expect("R2.3b router");
            let moe = streaming_routed_experts_with_observer(
                post_norm.view(),
                &router,
                config,
                layer,
                &mut store,
                expert_layout,
                |_, _, _, _| {},
            )
            .expect("R2.3b canonical F32 experts");
            if layer == target_layer {
                let packed_reader = reader
                    .reader_mut_for_layer(target_layer)
                    .expect("R2.3b layer identity");
                let experts = load_hybrid_experts(
                    target_layer,
                    &router,
                    packed_reader,
                    &mut store,
                    expert_layout,
                );
                for variant in VARIANTS {
                    let candidate =
                        hybrid_routed(post_norm.view(), &router, config, &experts, variant)
                            .expect("R2.3b same-input candidate");
                    let is_finite = candidate.data().iter().all(|value| value.is_finite());
                    finite
                        .entry(variant.id)
                        .and_modify(|value| *value &= is_finite);
                    let error = max_abs(candidate.data(), moe.data());
                    maxima
                        .entry(variant.id)
                        .and_modify(|value| *value = value.max(error));
                }
            }
            hidden = elementwise_add(residual.view(), moe.view()).expect("R2.3b block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("R2.3b KV append");
        if position + 1 == fixture.token_ids.len() {
            let normalized = rms_norm(
                hidden.view(),
                final_norm_weight.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3b final norm");
            let logits = streaming_language_model_head(
                &mut payload,
                &final_plan,
                &normalized,
                &mut dense_bytes_read,
            );
            control_first_token_exact =
                greedy_token(logits.view()).expect("R2.3b F32 greedy") == expected_first_token;
        }
    }
    SameInputRun {
        max_abs: maxima,
        finite,
        control_first_token_exact,
    }
}

fn run_sequence(
    fixture: &TierBReference,
    expected_first_token: usize,
    target_layer: usize,
    variant: Variant,
    reader: &mut R2_2LayerCandidateReader,
) -> SequenceRun {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("R2.3b canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.3b runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("R2.3b dense payload");
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * EXPERTS_PER_LAYER,
    );
    let mut sequence = fixture.token_ids.clone();
    sequence.push(expected_first_token);
    let mut cache = KvCache::new(48, sequence.len(), 4, 128).expect("R2.3b sequence KV cache");
    let mut maximum = 0.0_f32;
    let mut finite = true;
    let mut prompt_first_token_exact = false;
    let mut router_ids_by_position = Vec::with_capacity(sequence.len());
    let mut trace = Vec::new();

    for (position, &token_id) in sequence.iter().enumerate() {
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3b input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("R2.3b KV layer"),
            )
            .expect("R2.3b attention");
            let residual =
                elementwise_add(hidden.view(), attention.output.view()).expect("R2.3b residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3b post norm");
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .expect("R2.3b router");
            let moe = if layer == target_layer {
                router_ids_by_position.push(router.selected_experts.clone());
                let f32 = streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("R2.3b sequence F32 comparator");
                let packed_reader = reader
                    .reader_mut_for_layer(target_layer)
                    .expect("R2.3b layer identity");
                let experts = load_hybrid_experts(
                    target_layer,
                    &router,
                    packed_reader,
                    &mut store,
                    expert_layout,
                );
                let candidate = hybrid_routed(post_norm.view(), &router, config, &experts, variant)
                    .expect("R2.3b sequence candidate");
                finite &= candidate.data().iter().all(|value| value.is_finite());
                maximum = maximum.max(max_abs(candidate.data(), f32.data()));
                trace.extend_from_slice(candidate.data());
                candidate
            } else {
                streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("R2.3b canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("R2.3b block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("R2.3b KV append");
        if position + 1 == fixture.token_ids.len() {
            let normalized = rms_norm(
                hidden.view(),
                final_norm_weight.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3b final norm");
            let logits = streaming_language_model_head(
                &mut payload,
                &final_plan,
                &normalized,
                &mut dense_bytes_read,
            );
            prompt_first_token_exact = greedy_token(logits.view()).expect("R2.3b candidate greedy")
                == expected_first_token;
        }
    }
    SequenceRun {
        max_abs: maximum,
        finite,
        prompt_first_token_exact,
        router_ids_by_position,
        trace_sha256: f32_little_endian_sha256(&trace),
    }
}

fn ids_by_position_text(values: &[Vec<usize>]) -> String {
    values
        .iter()
        .map(|ids| comma_separated(ids))
        .collect::<Vec<_>>()
        .join("|")
}

#[test]
fn m6_3_r2_3b_characterize_layer_group_candidates() {
    let target_layer = env::var("COLIBRI_R2_3B_LAYER")
        .expect("R2.3b layer")
        .parse::<usize>()
        .expect("integer layer");
    assert!((1..=23).contains(&target_layer) || (25..=46).contains(&target_layer));
    let group_size = env::var("COLIBRI_R2_3B_GROUP_SIZE")
        .expect("R2.3b group")
        .parse::<usize>()
        .expect("integer group");
    assert!(GROUPS.contains(&group_size));
    let output_path = PathBuf::from(env::var_os("COLIBRI_R2_3B_OUTPUT").expect("R2.3b output"));
    assert!(!output_path.exists(), "R2.3b output must be new");
    let reference_path =
        PathBuf::from(env::var_os("COLIBRI_R2_3A_REFERENCE_PATH").expect("R2.3a reference path"));
    let first_tokens = frozen_first_tokens(&reference_path);
    let source_sha256 = env::var("COLIBRI_R2_3B_SOURCE_SHA256").expect("R2.3b source SHA-256");
    let artifact_sha256 =
        env::var("COLIBRI_R2_3B_ARTIFACT_SHA256").expect("R2.3b artifact SHA-256");
    let artifact_bytes = fs::metadata(PathBuf::from(
        env::var_os("COLIBRI_R2_3B_ARTIFACT_PATH").expect("R2.3b artifact path"),
    ))
    .expect("R2.3b artifact metadata")
    .len();
    let mut reader = open_reader(target_layer, group_size);
    let fixtures = tier_b_references()
        .into_iter()
        .filter(|fixture| matches!(fixture.name.as_str(), "short_english" | "short_thai"))
        .collect::<Vec<_>>();
    assert_eq!(fixtures.len(), 2);
    let same = fixtures
        .iter()
        .map(|fixture| {
            let run = run_same_input(
                fixture,
                first_tokens[&fixture.name],
                target_layer,
                &mut reader,
            );
            assert!(
                run.control_first_token_exact,
                "R2.3b F32 control must match R2.3a"
            );
            (fixture.name.clone(), run)
        })
        .collect::<HashMap<_, _>>();
    let mut evidence = String::from(
        "layer\tgroup_size\tsubset_index\tsubset\tpacked_count\tlogical_expert_bytes\tlogical_layer_bytes\tlogical_bytes_saved\tartifact_bytes\tsource_sha256\tartifact_sha256\tsame_input_english_max_abs\tsame_input_thai_max_abs\tsame_input_max_abs\tsame_input_pass\tsequence_english_max_abs\tsequence_thai_max_abs\tsequence_max_abs\tsequence_pass\tcandidate_finite\tprompt_first_token_exact\tenglish_router_ids_by_position\tthai_router_ids_by_position\tenglish_trace_sha256\tthai_trace_sha256\n",
    );
    for (subset_index, variant) in VARIANTS.iter().copied().enumerate() {
        let english = run_sequence(
            &fixtures[0],
            first_tokens[&fixtures[0].name],
            target_layer,
            variant,
            &mut reader,
        );
        let thai = run_sequence(
            &fixtures[1],
            first_tokens[&fixtures[1].name],
            target_layer,
            variant,
            &mut reader,
        );
        let same_english = same["short_english"].max_abs[variant.id];
        let same_thai = same["short_thai"].max_abs[variant.id];
        let same_maximum = same_english.max(same_thai);
        let sequence_maximum = english.max_abs.max(thai.max_abs);
        let candidate_finite = same["short_english"].finite[variant.id]
            && same["short_thai"].finite[variant.id]
            && english.finite
            && thai.finite;
        let prompt_exact = english.prompt_first_token_exact && thai.prompt_first_token_exact;
        let expert_bytes = logical_expert_bytes(variant, group_size);
        let layer_bytes = expert_bytes * EXPERTS_PER_LAYER;
        let bytes_saved = (F32_EXPERT_BYTES - expert_bytes) * EXPERTS_PER_LAYER;
        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            target_layer, group_size, subset_index, variant.id, packed_count(variant), expert_bytes,
            layer_bytes, bytes_saved, artifact_bytes, source_sha256, artifact_sha256,
            same_english, same_thai, same_maximum, same_maximum <= LOCAL_MAX_ABS_LIMIT,
            english.max_abs, thai.max_abs, sequence_maximum, sequence_maximum <= LOCAL_MAX_ABS_LIMIT,
            candidate_finite, prompt_exact, ids_by_position_text(&english.router_ids_by_position),
            ids_by_position_text(&thai.router_ids_by_position), english.trace_sha256, thai.trace_sha256,
        ).expect("write R2.3b evidence row");
    }
    fs::write(output_path, evidence).expect("write R2.3b evidence");
}
