use clr_core::RuntimeError;

use crate::{
    block::combine_routed_experts,
    r1_1_direct_candidate::{R1_1PackedExpert, R2_2LayerCandidateReader},
    streaming::{R2D3F32Expert, r2_d3_load_f32_expert},
};

use super::*;

const LOCAL_MAX_ABS_LIMIT: f32 = 0.001;
const F32_PROJECTION_BYTES: usize = 6_291_456;
const PACKED_PROJECTION_BYTES: usize = 1_769_472;

#[derive(Debug, Clone, Copy)]
struct D3Variant {
    id: &'static str,
    gate: bool,
    up: bool,
    down: bool,
}

const VARIANTS: [D3Variant; 7] = [
    D3Variant {
        id: "gate_up_down",
        gate: true,
        up: true,
        down: true,
    },
    D3Variant {
        id: "gate_up",
        gate: true,
        up: true,
        down: false,
    },
    D3Variant {
        id: "gate_down",
        gate: true,
        up: false,
        down: true,
    },
    D3Variant {
        id: "up_down",
        gate: false,
        up: true,
        down: true,
    },
    D3Variant {
        id: "gate",
        gate: true,
        up: false,
        down: false,
    },
    D3Variant {
        id: "up",
        gate: false,
        up: true,
        down: false,
    },
    D3Variant {
        id: "down",
        gate: false,
        up: false,
        down: true,
    },
];

#[derive(Debug, Clone)]
struct D3Context {
    id: &'static str,
    target_layer: usize,
    prefix_group32_layers: Vec<usize>,
}

#[derive(Debug)]
struct HybridExpert {
    f32: R2D3F32Expert,
    packed: R1_1PackedExpert,
}

#[derive(Debug)]
struct D3ContextRun {
    router_ids_by_position: Vec<Vec<usize>>,
    position_max_abs: HashMap<&'static str, Vec<f32>>,
    candidate_finite: HashMap<&'static str, bool>,
    group32_control_exact: bool,
}

fn contexts() -> Vec<D3Context> {
    vec![
        D3Context {
            id: "layer24_f32_prefix",
            target_layer: 24,
            prefix_group32_layers: vec![],
        },
        D3Context {
            id: "layer24_layer0_group32_prefix",
            target_layer: 24,
            prefix_group32_layers: vec![0],
        },
        D3Context {
            id: "layer47_f32_prefix",
            target_layer: 47,
            prefix_group32_layers: vec![],
        },
        D3Context {
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

fn packed_count(variant: D3Variant) -> usize {
    usize::from(variant.gate) + usize::from(variant.up) + usize::from(variant.down)
}

fn modeled_expert_bytes(variant: D3Variant) -> usize {
    let packed = packed_count(variant);
    packed * PACKED_PROJECTION_BYTES + (3 - packed) * F32_PROJECTION_BYTES
}

fn modeled_layer_bytes(variant: D3Variant) -> usize {
    modeled_expert_bytes(variant) * 128
}

fn packed_label(variant: D3Variant) -> String {
    let mut names = Vec::new();
    if variant.gate {
        names.push("gate");
    }
    if variant.up {
        names.push("up");
    }
    if variant.down {
        names.push("down");
    }
    names.join(",")
}

fn hybrid_expert_output(
    input: &[f32],
    expert: &HybridExpert,
    variant: D3Variant,
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
    variant: D3Variant,
) -> Result<Tensor, RuntimeError> {
    let hidden = config.model().hidden_size();
    let intermediate = config.moe_intermediate_size();
    combine_routed_experts(hidden_states, router, config, |expert_id, occurrences| {
        let expert = experts.get(&expert_id).expect("D3 selected expert loaded");
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

fn open_group32_readers() -> HashMap<usize, R2_2LayerCandidateReader> {
    let root = PathBuf::from(
        env::var_os("COLIBRI_R2_2_ARTIFACT_ROOT").expect("R2.2 group32 artifact root"),
    );
    let layout = R1_1PackedArtifactLayout::canonical(32).expect("D3 group32 layout");
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
        (
            47,
            "layer47-group32.bin",
            "a2a45a1f407d5cd2303286843001684fc7800953cd2b67995a2c5e62a9e66a33",
        ),
    ]
    .into_iter()
    .map(|(layer, name, sha)| {
        let reader = R2_2LayerCandidateReader::open(layer, &root.join(name), layout, sha)
            .expect("D3 frozen group32 artifact");
        (layer, reader)
    })
    .collect()
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
                .expect("D3 canonical F32 expert");
            let packed = reader.load_expert(expert_id).expect("D3 group32 expert");
            (expert_id, HybridExpert { f32, packed })
        })
        .collect()
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

fn run_context(
    fixture: &TierBReference,
    context: &D3Context,
    readers: &mut HashMap<usize, R2_2LayerCandidateReader>,
) -> D3ContextRun {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("D3 runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open D3 dense payload");
    let mut dense_bytes_read = 0_u64;
    let mut normal_store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    let mut cache = KvCache::new(48, fixture.token_ids.len(), 4, 128).expect("D3 KV cache");
    let mut router_ids_by_position = Vec::with_capacity(fixture.token_ids.len());
    let mut position_max_abs = VARIANTS
        .into_iter()
        .map(|variant| (variant.id, Vec::with_capacity(fixture.token_ids.len())))
        .collect::<HashMap<_, _>>();
    let mut candidate_finite = VARIANTS
        .into_iter()
        .map(|variant| (variant.id, true))
        .collect::<HashMap<_, _>>();
    let mut group32_control_exact = true;

    for (position, &token_id) in fixture.token_ids.iter().enumerate() {
        assert_eq!(cache.len(), position, "D3 cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D3 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("D3 KV layer"),
            )
            .expect("D3 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("D3 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("D3 post-attention norm");
            let router =
                route_tokens(post_norm.view(), weights.router.view(), config).expect("D3 router");

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
                .expect("D3 F32 comparator");
                assert!(f32.data().iter().all(|value| value.is_finite()));

                let reader = readers
                    .get_mut(&layer)
                    .expect("D3 target group32 reader")
                    .reader_mut_for_layer(layer)
                    .expect("D3 target layer identity");
                let group32_control = r2_1_routed_experts_with_backend(
                    post_norm.view(),
                    &router,
                    config,
                    reader,
                    R2_1PackedProjectionBackend::Scalar,
                )
                .expect("D3 group32 control");
                let experts =
                    load_hybrid_experts(layer, &router, reader, &mut normal_store, expert_layout);
                for variant in VARIANTS {
                    let candidate =
                        hybrid_routed(post_norm.view(), &router, config, &experts, variant)
                            .expect("D3 hybrid candidate");
                    let finite = candidate.data().iter().all(|value| value.is_finite());
                    candidate_finite
                        .entry(variant.id)
                        .and_modify(|value| *value &= finite);
                    assert!(finite, "D3 hybrid candidate output must be finite");
                    if variant.id == "gate_up_down" {
                        let exact = candidate.data() == group32_control.data();
                        group32_control_exact &= exact;
                        assert!(
                            exact,
                            "D3 all-packed seam must match scalar group32 exactly"
                        );
                    }
                    let local = max_abs(candidate.data(), f32.data());
                    assert!(local.is_finite(), "D3 local error finite");
                    position_max_abs
                        .get_mut(variant.id)
                        .expect("D3 variant error vector")
                        .push(local);
                }
                f32
            } else if context.prefix_group32_layers.contains(&layer) {
                let reader = readers
                    .get_mut(&layer)
                    .expect("D3 prefix group32 reader")
                    .reader_mut_for_layer(layer)
                    .expect("D3 prefix layer identity");
                let candidate = r2_1_routed_experts_with_backend(
                    post_norm.view(),
                    &router,
                    config,
                    reader,
                    R2_1PackedProjectionBackend::Scalar,
                )
                .expect("D3 prefix scalar group32");
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
                .expect("D3 canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("D3 block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("D3 KV append");
    }

    assert_eq!(router_ids_by_position.len(), fixture.token_ids.len());
    for variant in VARIANTS {
        assert_eq!(position_max_abs[variant.id].len(), fixture.token_ids.len());
    }
    assert!(group32_control_exact, "D3 group32 seam validation");
    D3ContextRun {
        router_ids_by_position,
        position_max_abs,
        candidate_finite,
        group32_control_exact,
    }
}
#[test]
fn m6_3_r2_2_d3_localize_projection_sensitivity() {
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_2_D3_OUTPUT").expect("D3 evidence output path"));
    assert!(!output_path.exists(), "D3 evidence output must be new");
    let reference_path =
        PathBuf::from(env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R1.2 reference path"));
    let frozen = r1_2_frozen_references(&reference_path);
    let frozen = &frozen["short_thai"];
    let fixture = tier_b_references()
        .into_iter()
        .find(|item| item.name == "short_thai")
        .expect("D3 short_thai fixture");
    assert_eq!(fixture.token_ids, frozen.token_ids);

    let mut readers = open_group32_readers();
    let mut evidence = String::from(
        "context_index\tlayer\tcontext_id\tprefix_group32_layers\tvariant_index\tvariant_id\tpacked_projections\tpacked_count\tmodeled_expert_bytes\tmodeled_layer_bytes\tprompt_positions\trouter_ids_by_position\tfinal_guard_exact\tposition_max_abs\tlocal_max_abs\tlocal_limit\tpass_local\tcandidate_finite\tgroup32_control_exact\n",
    );

    for (context_index, context) in contexts().iter().enumerate() {
        let run = run_context(&fixture, context, &mut readers);
        let final_router = run
            .router_ids_by_position
            .last()
            .expect("D3 final target router IDs");
        let final_guard_exact = final_router == &frozen.prompt_guard_ids[&context.target_layer];
        for (variant_index, variant) in VARIANTS.iter().copied().enumerate() {
            let errors = &run.position_max_abs[variant.id];
            let local_max_abs = errors.iter().copied().fold(0.0_f32, f32::max);
            let pass_local = local_max_abs <= LOCAL_MAX_ABS_LIMIT;
            writeln!(
                evidence,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{}\t{}\t{}",
                context_index,
                context.target_layer,
                context.id,
                prefix_label(&context.prefix_group32_layers),
                variant_index,
                variant.id,
                packed_label(variant),
                packed_count(variant),
                modeled_expert_bytes(variant),
                modeled_layer_bytes(variant),
                fixture.token_ids.len(),
                ids_by_position_text(&run.router_ids_by_position),
                final_guard_exact,
                errors_text(errors),
                local_max_abs,
                LOCAL_MAX_ABS_LIMIT,
                pass_local,
                run.candidate_finite[variant.id],
                run.group32_control_exact,
            )
            .expect("write D3 evidence row");
        }
    }
    fs::write(output_path, evidence).expect("write D3 evidence file");
}
