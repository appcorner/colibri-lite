use super::*;

const R2_3A_GENERATED_TOKEN_COUNT: usize = 4;

#[derive(Debug)]
struct R2_3AReferenceRun {
    generated_ids: Vec<usize>,
    prompt_argmax: usize,
    prompt_top20_ids: Vec<usize>,
    prompt_top20_logits: Vec<f32>,
    fixed_logit_indices: Vec<usize>,
    fixed_logits: Vec<f32>,
    prompt_guard_ids: HashMap<usize, Vec<usize>>,
    prompt_logits_sha256: String,
    prompt_final_norm_sha256: String,
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn join_f32(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn run_f32_fixture(fixture: &TierBReference) -> R2_3AReferenceRun {
    assert!(matches!(
        fixture.name.as_str(),
        "short_english" | "short_thai"
    ));
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("R2.3a artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.3a runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("R2.3a dense payload");
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
        48 * 128,
    );
    let processed_positions = fixture.token_ids.len() + R2_3A_GENERATED_TOKEN_COUNT - 1;
    let mut cache = KvCache::new(48, processed_positions, 4, 128).expect("R2.3a KV cache");
    let mut sequence = fixture.token_ids.clone();
    let mut generated_ids = Vec::with_capacity(R2_3A_GENERATED_TOKEN_COUNT);
    let mut prompt_argmax = None;
    let mut prompt_top20_ids = None;
    let mut prompt_top20_logits = None;
    let mut fixed_logits = None;
    let mut prompt_guard_ids = HashMap::<usize, Vec<usize>>::new();
    let mut prompt_logits_sha256 = None;
    let mut prompt_final_norm_sha256 = None;

    for position in 0..processed_positions {
        let token_id = *sequence
            .get(position)
            .expect("R2.3a generated token available before position");
        assert_eq!(cache.len(), position, "R2.3a cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3a input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("R2.3a KV layer"),
            )
            .expect("R2.3a attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("R2.3a attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3a post-attention norm");
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .expect("R2.3a router");
            if position + 1 == fixture.token_ids.len() && GENERATION_GUARD_LAYERS.contains(&layer) {
                prompt_guard_ids.insert(layer, router.selected_experts.clone());
            }
            let moe = streaming_routed_experts_with_observer(
                post_norm.view(),
                &router,
                config,
                layer,
                &mut store,
                expert_layout,
                |_, _, _, _| {},
            )
            .expect("R2.3a canonical F32 experts");
            hidden = elementwise_add(residual.view(), moe.view()).expect("R2.3a block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&update_views).expect("R2.3a KV append");

        if position + 1 >= fixture.token_ids.len() {
            let normalized = rms_norm(
                hidden.view(),
                final_norm_weight.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R2.3a final norm");
            let logits = streaming_language_model_head(
                &mut payload,
                &final_plan,
                &normalized,
                &mut dense_bytes_read,
            );
            assert!(logits.data().iter().all(|value| value.is_finite()));
            let greedy = greedy_token(logits.view()).expect("R2.3a greedy token");
            if position + 1 == fixture.token_ids.len() {
                let top20 = deterministic_top_ids(&logits, 20);
                prompt_argmax = Some(greedy);
                prompt_top20_logits = Some(
                    top20
                        .iter()
                        .map(|&index| logits.data()[index])
                        .collect::<Vec<_>>(),
                );
                fixed_logits = Some(
                    fixture
                        .fixed_logit_indices
                        .iter()
                        .map(|&index| logits.data()[index])
                        .collect::<Vec<_>>(),
                );
                prompt_top20_ids = Some(top20);
                prompt_logits_sha256 = Some(f32_little_endian_sha256(logits.data()));
                prompt_final_norm_sha256 = Some(f32_little_endian_sha256(normalized.data()));
            }
            generated_ids.push(greedy);
            if generated_ids.len() < R2_3A_GENERATED_TOKEN_COUNT {
                sequence.push(greedy);
            }
        }
    }

    assert_eq!(generated_ids.len(), R2_3A_GENERATED_TOKEN_COUNT);
    assert_eq!(cache.len(), processed_positions);
    R2_3AReferenceRun {
        generated_ids,
        prompt_argmax: prompt_argmax.expect("R2.3a prompt argmax"),
        prompt_top20_ids: prompt_top20_ids.expect("R2.3a prompt top20"),
        prompt_top20_logits: prompt_top20_logits.expect("R2.3a prompt top20 logits"),
        fixed_logit_indices: fixture.fixed_logit_indices.clone(),
        fixed_logits: fixed_logits.expect("R2.3a fixed logits"),
        prompt_guard_ids,
        prompt_logits_sha256: prompt_logits_sha256.expect("R2.3a prompt logits hash"),
        prompt_final_norm_sha256: prompt_final_norm_sha256.expect("R2.3a prompt norm hash"),
    }
}

#[test]
fn m6_3_r2_3a_freeze_four_token_f32_reference() {
    let old_reference_path = PathBuf::from(
        env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R2.3a old reference path"),
    );
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_3A_OUTPUT").expect("R2.3a output path"));
    assert!(!output_path.exists(), "R2.3a output must be new");
    let old_references = r1_2_frozen_references(&old_reference_path);
    let mut evidence = String::from(
        "fixture\tgenerated_ids\tprompt_argmax\tprompt_top20_ids\tprompt_top20_logits\tfixed_logit_indices\tfixed_logits\tguard_layer0_ids\tguard_layer24_ids\tguard_layer47_ids\tprompt_logits_sha256\tprompt_final_norm_sha256\n",
    );

    for fixture in tier_b_references()
        .into_iter()
        .filter(|fixture| matches!(fixture.name.as_str(), "short_english" | "short_thai"))
    {
        let old = &old_references[&fixture.name];
        let run = run_f32_fixture(&fixture);
        assert_eq!(
            &run.generated_ids[..R1_2_GENERATED_TOKEN_COUNT],
            old.generated_ids
        );
        assert_eq!(run.prompt_argmax, old.prompt_argmax);
        assert_eq!(run.prompt_top20_ids, old.prompt_top20_ids);
        for layer in GENERATION_GUARD_LAYERS {
            assert_eq!(run.prompt_guard_ids[&layer], old.prompt_guard_ids[&layer]);
        }
        assert_eq!(run.prompt_logits_sha256, old.prompt_logits_sha256);
        assert_eq!(run.prompt_final_norm_sha256, old.prompt_final_norm_sha256);
        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            fixture.name,
            join_usize(&run.generated_ids),
            run.prompt_argmax,
            join_usize(&run.prompt_top20_ids),
            join_f32(&run.prompt_top20_logits),
            join_usize(&run.fixed_logit_indices),
            join_f32(&run.fixed_logits),
            join_usize(&run.prompt_guard_ids[&0]),
            join_usize(&run.prompt_guard_ids[&24]),
            join_usize(&run.prompt_guard_ids[&47]),
            run.prompt_logits_sha256,
            run.prompt_final_norm_sha256,
        )
        .expect("R2.3a evidence row");
    }
    fs::write(output_path, evidence).expect("write R2.3a reference");
}
