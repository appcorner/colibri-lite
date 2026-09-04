use std::{
    collections::BTreeSet,
    env,
    fmt::Write as _,
    fs::{self, File},
    path::PathBuf,
    time::Instant,
};

use crate::{
    block::RouterOutput,
    r1_1_direct_candidate::{
        R1_1CandidateReader, R1_1PackedArtifactLayout, R2_1PackedProjectionBackend,
        R2PreloadedCandidateExperts, r1_1_routed_experts_with_observer,
        r2_1_routed_experts_with_backend, r2_1_routed_preloaded_candidate_with_backend,
        r2_preload_candidate_experts, r2_routed_preloaded_candidate,
    },
    r2_localization,
    streaming::{
        PackedExpertLayout, R2PreloadedF32Experts, r2_preload_f32_experts, r2_routed_preloaded_f32,
        streaming_routed_experts_with_observer,
    },
};
use clr_core::{Tensor, TensorShape};

use super::*;

const CONTRACT_SHA256: &str = "02322820758848b5ebfe3f020db026f1aa5e3dd54ce007fb71202f50e54aceca";
const CANDIDATE_SHA256: &str = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2";

#[derive(Debug)]
struct FrozenRuntimeFixture {
    name: String,
    token_ids: Vec<usize>,
    input: Tensor,
    router: RouterOutput,
    reference_output: Option<Tensor>,
}

fn usize_array(values: &[usize]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn f32_le_hex(values: &[f32]) -> String {
    let mut output = String::with_capacity(values.len() * 8);
    for value in values {
        for byte in value.to_le_bytes() {
            write!(output, "{byte:02x}").expect("hex f32 bytes");
        }
    }
    output
}

fn build_runtime_fixture(name: &str, with_reference_output: bool) -> FrozenRuntimeFixture {
    let fixture = tier_b_references()
        .into_iter()
        .find(|item| item.name == name)
        .expect("R2.0 frozen fixture");
    assert!(matches!(name, "short_english" | "short_thai"));

    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(RUNTIME_PLAN);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    let mut dense_bytes_read = 0_u64;
    let mut input_values = Vec::with_capacity(fixture.token_ids.len() * 2048);
    for &token_id in &fixture.token_ids {
        input_values.extend_from_slice(
            embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read).data(),
        );
    }
    let hidden = Tensor::new(
        TensorShape::new([fixture.token_ids.len(), 2048]),
        input_values,
    )
    .expect("R2.0 embedding rows");
    let weights = layer_weights(&mut payload, &plan, 0, &mut dense_bytes_read);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.0 runtime config")
        .runtime_config();
    let pre = pre_router_with_weights(
        hidden.view(),
        weights.input_norm.view(),
        weights.query.view(),
        weights.key.view(),
        weights.value.view(),
        weights.output.view(),
        weights.query_norm.view(),
        weights.key_norm.view(),
        weights.post_norm.view(),
        weights.router.view(),
        config,
    )
    .expect("R2.0 Layer-0 pre-router");
    let top_k = config.experts_per_token();
    let final_ids = &pre.router.selected_experts
        [(fixture.token_ids.len() - 1) * top_k..fixture.token_ids.len() * top_k];
    assert_eq!(
        final_ids, fixture.guard_ids[&0],
        "R2.0 final prompt router guard"
    );
    let reference_output = if with_reference_output {
        let mut store = canonical_layer0_expert_store(&artifact_root);
        Some(
            streaming_routed_experts_with_observer(
                pre.post_attention_norm.view(),
                &pre.router,
                config,
                0,
                &mut store,
                PackedExpertLayout::for_config(config),
                |_, _, _, _| {},
            )
            .expect("R2.0 reference routed experts"),
        )
    } else {
        None
    };
    FrozenRuntimeFixture {
        name: fixture.name,
        token_ids: fixture.token_ids,
        input: pre.post_attention_norm,
        router: pre.router,
        reference_output,
    }
}

fn fixture_record_entry(run: &FrozenRuntimeFixture) -> String {
    let mut unique = BTreeSet::new();
    unique.extend(run.router.selected_experts.iter().copied());
    let unique = unique.into_iter().collect::<Vec<_>>();
    let reference = run.reference_output.as_ref().expect("reference output");
    format!(
        concat!(
            "{{\"fixture_id\":\"{}\",\"token_ids\":[{}],",
            "\"expert_input\":{{\"shape\":[{},2048],\"sha256\":\"{}\",",
            "\"f32_le_hex\":\"{}\"}},",
            "\"selected_expert_ids\":[{}],",
            "\"router_weights\":{{\"shape\":[{},8],\"sha256\":\"{}\",",
            "\"f32_le_hex\":\"{}\"}},",
            "\"unique_selected_expert_ids\":[{}],",
            "\"reference_routed_output\":{{\"shape\":[{},2048],\"sha256\":\"{}\"}}}}"
        ),
        run.name,
        usize_array(&run.token_ids),
        run.token_ids.len(),
        f32_little_endian_sha256(run.input.data()),
        f32_le_hex(run.input.data()),
        usize_array(&run.router.selected_experts),
        run.token_ids.len(),
        f32_little_endian_sha256(run.router.weights.data()),
        f32_le_hex(run.router.weights.data()),
        usize_array(&unique),
        run.token_ids.len(),
        f32_little_endian_sha256(reference.data()),
    )
}

#[test]
fn m6_3_r2_0b_freeze_reference_layer0_fixtures() {
    let output =
        PathBuf::from(env::var_os("COLIBRI_R2_FIXTURE_OUTPUT").expect("R2.0 fixture output path"));
    assert!(!output.exists(), "R2.0 fixture output must be new");
    let english = build_runtime_fixture("short_english", true);
    let thai = build_runtime_fixture("short_thai", true);
    let document = format!(
        concat!(
            "{{\"schema\":\"m6.3-r2.0-reference-fixtures-v1\",",
            "\"contract_sha256\":\"{}\",",
            "\"reference_id\":\"reference-f32-v1\",",
            "\"root_manifest_sha256\":\"f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2\",",
            "\"fixtures\":[{},{}]}}\n"
        ),
        CONTRACT_SHA256,
        fixture_record_entry(&english),
        fixture_record_entry(&thai),
    );
    fs::write(output, document).expect("write R2.0 frozen fixture record");
}

fn parse_expected_ids() -> Vec<usize> {
    env::var("COLIBRI_R2_EXPECT_SELECTED_IDS")
        .expect("R2.0 expected selected IDs")
        .split(',')
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>().expect("selected expert ID"))
        .collect()
}

fn verify_fixture_identity(run: &FrozenRuntimeFixture) {
    assert_eq!(
        f32_little_endian_sha256(run.input.data()),
        env::var("COLIBRI_R2_EXPECT_INPUT_SHA256").expect("expected input SHA"),
        "R2.0 expert input identity",
    );
    assert_eq!(
        f32_little_endian_sha256(run.router.weights.data()),
        env::var("COLIBRI_R2_EXPECT_ROUTER_WEIGHTS_SHA256").expect("expected router weights SHA"),
        "R2.0 router weights identity",
    );
    assert_eq!(
        run.router.selected_experts,
        parse_expected_ids(),
        "R2.0 selected expert identity",
    );
}

fn snapshot_json(snapshot: &r2_localization::Snapshot) -> String {
    let mut entries = Vec::with_capacity(snapshot.events.len());
    for (name, event) in &snapshot.events {
        entries.push(format!(
            concat!(
                "\"{}\":{{\"calls\":{},\"total_nanos\":{},",
                "\"exclusive_nanos\":{},\"min_nanos\":{},",
                "\"median_nanos\":{},\"max_nanos\":{},\"logical_bytes\":{}}}"
            ),
            name,
            event.calls,
            event.total_nanos,
            event.exclusive_nanos,
            event.min_nanos,
            event.median_nanos,
            event.max_nanos,
            event.logical_bytes,
        ));
    }
    format!("{{{}}}", entries.join(","))
}

struct PreparedReference {
    config: crate::Qwen3MoeConfig,
    preloaded: R2PreloadedF32Experts,
}

struct PreparedCandidate {
    config: crate::Qwen3MoeConfig,
    preloaded: R2PreloadedCandidateExperts,
}

fn prepare_preloaded_reference(run: &FrozenRuntimeFixture) -> PreparedReference {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.0 config")
        .runtime_config();
    let layout = PackedExpertLayout::for_config(config);
    let mut store = canonical_layer0_expert_store(&artifact_root);
    let preloaded = r2_preload_f32_experts(0, &run.router.selected_experts, &mut store, layout)
        .expect("preload F32 experts");
    PreparedReference { config, preloaded }
}

fn prepare_preloaded_candidate(run: &FrozenRuntimeFixture) -> PreparedCandidate {
    let candidate_path =
        PathBuf::from(env::var_os("COLIBRI_R2_CANDIDATE_PATH").expect("R2.0 candidate path"));
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.0 config")
        .runtime_config();
    let mut reader = R1_1CandidateReader::open(
        &candidate_path,
        R1_1PackedArtifactLayout::canonical(32).expect("group-32 layout"),
        CANDIDATE_SHA256,
    )
    .expect("verified group-32 candidate");
    let preloaded = r2_preload_candidate_experts(&mut reader, &run.router.selected_experts)
        .expect("preload candidate experts");
    PreparedCandidate { config, preloaded }
}

fn timed_preloaded_reference(
    run: &FrozenRuntimeFixture,
    prepared: &PreparedReference,
    observer_enabled: bool,
) -> (u128, r2_localization::Snapshot, Tensor) {
    for _ in 0..2 {
        std::hint::black_box(
            r2_routed_preloaded_f32(
                run.input.view(),
                &run.router,
                prepared.config,
                &prepared.preloaded,
            )
            .expect("F32 warmup"),
        );
    }
    let session = r2_localization::start(observer_enabled);
    let mut total_nanos = 0_u128;
    let mut last = None;
    for _ in 0..25 {
        let started = Instant::now();
        let output = r2_routed_preloaded_f32(
            run.input.view(),
            &run.router,
            prepared.config,
            &prepared.preloaded,
        )
        .expect("F32 timed compute-only");
        total_nanos = total_nanos.saturating_add(started.elapsed().as_nanos());
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    (total_nanos, snapshot, last.expect("timed F32 output"))
}

fn timed_preloaded_candidate(
    run: &FrozenRuntimeFixture,
    prepared: &PreparedCandidate,
    observer_enabled: bool,
) -> (u128, r2_localization::Snapshot, Tensor) {
    for _ in 0..2 {
        std::hint::black_box(
            r2_routed_preloaded_candidate(
                run.input.view(),
                &run.router,
                prepared.config,
                &prepared.preloaded,
            )
            .expect("candidate warmup"),
        );
    }
    let session = r2_localization::start(observer_enabled);
    let mut total_nanos = 0_u128;
    let mut last = None;
    for _ in 0..25 {
        let started = Instant::now();
        let output = r2_routed_preloaded_candidate(
            run.input.view(),
            &run.router,
            prepared.config,
            &prepared.preloaded,
        )
        .expect("candidate timed compute-only");
        total_nanos = total_nanos.saturating_add(started.elapsed().as_nanos());
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    (total_nanos, snapshot, last.expect("timed candidate output"))
}

fn timed_preloaded_candidate_with_backend(
    run: &FrozenRuntimeFixture,
    prepared: &PreparedCandidate,
    backend: R2_1PackedProjectionBackend,
) -> (u128, r2_localization::Snapshot, Tensor) {
    for _ in 0..2 {
        std::hint::black_box(
            r2_1_routed_preloaded_candidate_with_backend(
                run.input.view(),
                &run.router,
                prepared.config,
                &prepared.preloaded,
                backend,
            )
            .expect("R2.1 candidate warmup"),
        );
    }
    let session = r2_localization::start(true);
    let mut total_nanos = 0_u128;
    let mut last = None;
    for _ in 0..25 {
        let started = Instant::now();
        let output = r2_1_routed_preloaded_candidate_with_backend(
            run.input.view(),
            &run.router,
            prepared.config,
            &prepared.preloaded,
            backend,
        )
        .expect("R2.1 candidate timed compute-only");
        total_nanos = total_nanos.saturating_add(started.elapsed().as_nanos());
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    (
        total_nanos,
        snapshot,
        last.expect("timed R2.1 candidate output"),
    )
}

fn run_preloaded_reference(
    run: &FrozenRuntimeFixture,
    observer_enabled: bool,
) -> (u128, r2_localization::Snapshot, Tensor) {
    let prepared = prepare_preloaded_reference(run);
    timed_preloaded_reference(run, &prepared, observer_enabled)
}

fn run_preloaded_candidate(
    run: &FrozenRuntimeFixture,
    observer_enabled: bool,
) -> (u128, r2_localization::Snapshot, Tensor) {
    let prepared = prepare_preloaded_candidate(run);
    timed_preloaded_candidate(run, &prepared, observer_enabled)
}

#[test]
fn m6_3_r2_0_compute_only_single_sample() {
    let fixture_name = env::var("COLIBRI_R2_FIXTURE").expect("R2.0 fixture name");
    let path = env::var("COLIBRI_R2_PATH").expect("R2.0 path");
    assert!(matches!(path.as_str(), "reference" | "candidate"));
    let observer = env::var("COLIBRI_R2_OBSERVER").expect("R2.0 observer state");
    let observer_enabled = match observer.as_str() {
        "enabled" => true,
        "disabled" => false,
        _ => panic!("invalid R2.0 observer state"),
    };
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_SAMPLE_OUTPUT").expect("R2.0 sample output"));
    assert!(!output_path.exists(), "R2.0 sample output must be new");
    let run = build_runtime_fixture(&fixture_name, false);
    verify_fixture_identity(&run);
    let noop = r2_localization::calibrate_noop(101);
    let (timed_total_nanos, snapshot, output) = if path == "reference" {
        run_preloaded_reference(&run, observer_enabled)
    } else {
        run_preloaded_candidate(&run, observer_enabled)
    };
    let output_sha256 = f32_little_endian_sha256(output.data());
    if path == "reference" {
        assert_eq!(
            output_sha256,
            env::var("COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256")
                .expect("expected reference output SHA"),
            "R2.0 reference routed output identity",
        );
    }
    if observer_enabled {
        assert_eq!(
            snapshot
                .counters
                .get("unique_expert_loads")
                .copied()
                .unwrap_or(0),
            0
        );
        assert!(
            snapshot
                .events
                .values()
                .all(|event| event.logical_bytes == 0)
        );
    }
    let expert_occurrences = snapshot
        .counters
        .get("expert_occurrences")
        .copied()
        .unwrap_or(0);
    let unique_expert_loads = snapshot
        .counters
        .get("unique_expert_loads")
        .copied()
        .unwrap_or(0);
    let document = format!(
        concat!(
            "{{\"schema\":\"m6.3-r2.0-control-sample-v1\",",
            "\"contract_sha256\":\"{}\",\"fixture\":\"{}\",",
            "\"path\":\"{}\",\"observer\":\"{}\",",
            "\"fixture_record_sha256\":\"{}\",",
            "\"release_binary_sha256\":\"{}\",\"host_id\":\"{}\",",
            "\"timed_iterations\":25,\"timed_total_nanos\":{},",
            "\"output_sha256\":\"{}\",\"timer_noop_median_nanos\":{},",
            "\"expert_occurrences\":{},\"unique_expert_loads\":{},",
            "\"stages\":{}}}\n"
        ),
        CONTRACT_SHA256,
        fixture_name,
        path,
        observer,
        env::var("COLIBRI_R2_FIXTURE_RECORD_SHA256").expect("fixture record SHA"),
        env::var("COLIBRI_R2_RELEASE_BINARY_SHA256").expect("binary SHA"),
        env::var("COLIBRI_R2_HOST_ID").expect("host ID"),
        timed_total_nanos,
        output_sha256,
        noop.median_nanos,
        expert_occurrences,
        unique_expert_loads,
        snapshot_json(&snapshot),
    );
    fs::write(output_path, document).expect("write R2.0 single sample");
}

#[derive(Debug)]
struct ObserverPairRun {
    observer: String,
    timed_total_nanos: u128,
    snapshot: r2_localization::Snapshot,
    output_sha256: String,
}

fn observer_enabled(observer: &str) -> bool {
    match observer {
        "enabled" => true,
        "disabled" => false,
        _ => panic!("invalid R2.0 observer state"),
    }
}

fn assert_compute_only_snapshot(run: &ObserverPairRun) {
    if run.observer != "enabled" {
        return;
    }
    assert_eq!(
        run.snapshot
            .counters
            .get("unique_expert_loads")
            .copied()
            .unwrap_or(0),
        0,
        "compute-only timed expert loads",
    );
    assert!(
        run.snapshot
            .events
            .values()
            .all(|event| event.logical_bytes == 0),
        "compute-only timed logical bytes",
    );
}
fn run_observer_pair_reference(
    run: &FrozenRuntimeFixture,
    order: &[String],
) -> Vec<ObserverPairRun> {
    let prepared = prepare_preloaded_reference(run);
    order
        .iter()
        .map(|observer| {
            let (timed_total_nanos, snapshot, output) =
                timed_preloaded_reference(run, &prepared, observer_enabled(observer));
            ObserverPairRun {
                observer: observer.clone(),
                timed_total_nanos,
                snapshot,
                output_sha256: f32_little_endian_sha256(output.data()),
            }
        })
        .collect()
}

fn run_observer_pair_candidate(
    run: &FrozenRuntimeFixture,
    order: &[String],
) -> Vec<ObserverPairRun> {
    let prepared = prepare_preloaded_candidate(run);
    order
        .iter()
        .map(|observer| {
            let (timed_total_nanos, snapshot, output) =
                timed_preloaded_candidate(run, &prepared, observer_enabled(observer));
            ObserverPairRun {
                observer: observer.clone(),
                timed_total_nanos,
                snapshot,
                output_sha256: f32_little_endian_sha256(output.data()),
            }
        })
        .collect()
}

#[test]
fn m6_3_r2_0_observer_pair_sample() {
    let fixture_name = env::var("COLIBRI_R2_FIXTURE").expect("R2.0 fixture name");
    let path = env::var("COLIBRI_R2_PATH").expect("R2.0 path");
    assert!(matches!(path.as_str(), "reference" | "candidate"));
    let order = env::var("COLIBRI_R2_OBSERVER_ORDER")
        .expect("R2.0 observer order")
        .split(',')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(order.len(), 2, "R2.0 observer pair length");
    assert!(
        matches!(order.as_slice(), [left, right] if left != right && matches!(left.as_str(), "enabled" | "disabled") && matches!(right.as_str(), "enabled" | "disabled")),
        "R2.0 observer pair variants",
    );
    let output_path =
        PathBuf::from(env::var_os("COLIBRI_R2_SAMPLE_OUTPUT").expect("R2.0 pair sample output"));
    assert!(!output_path.exists(), "R2.0 pair sample output must be new");
    let run = build_runtime_fixture(&fixture_name, false);
    verify_fixture_identity(&run);
    let noop = r2_localization::calibrate_noop(101);
    let results = if path == "reference" {
        run_observer_pair_reference(&run, &order)
    } else {
        run_observer_pair_candidate(&run, &order)
    };
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].output_sha256, results[1].output_sha256);
    if path == "reference" {
        let expected = env::var("COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256")
            .expect("expected reference output SHA");
        assert_eq!(results[0].output_sha256, expected);
    }
    for result in &results {
        assert_compute_only_snapshot(result);
    }
    let enabled = results
        .iter()
        .find(|result| result.observer == "enabled")
        .expect("enabled observer result");
    let disabled = results
        .iter()
        .find(|result| result.observer == "disabled")
        .expect("disabled observer result");
    let document = format!(
        concat!(
            "{{\"schema\":\"m6.3-r2.0-observer-pair-sample-v2\",",
            "\"contract_sha256\":\"{}\",\"method_contract_sha256\":\"{}\",",
            "\"fixture\":\"{}\",\"path\":\"{}\",",
            "\"observer_order\":[\"{}\",\"{}\"],",
            "\"fixture_record_sha256\":\"{}\",",
            "\"release_binary_sha256\":\"{}\",\"host_id\":\"{}\",",
            "\"timed_iterations_per_state\":25,\"timer_noop_median_nanos\":{},",
            "\"states\":{{",
            "\"enabled\":{{\"timed_total_nanos\":{},\"output_sha256\":\"{}\",\"stages\":{}}},",
            "\"disabled\":{{\"timed_total_nanos\":{},\"output_sha256\":\"{}\",\"stages\":{}}}",
            "}}}}\n"
        ),
        CONTRACT_SHA256,
        env::var("COLIBRI_R2_METHOD_CONTRACT_SHA256").expect("method contract SHA"),
        fixture_name,
        path,
        order[0],
        order[1],
        env::var("COLIBRI_R2_FIXTURE_RECORD_SHA256").expect("fixture record SHA"),
        env::var("COLIBRI_R2_RELEASE_BINARY_SHA256").expect("binary SHA"),
        env::var("COLIBRI_R2_HOST_ID").expect("host ID"),
        noop.median_nanos,
        enabled.timed_total_nanos,
        enabled.output_sha256,
        snapshot_json(&enabled.snapshot),
        disabled.timed_total_nanos,
        disabled.output_sha256,
        snapshot_json(&disabled.snapshot),
    );
    fs::write(output_path, document).expect("write R2.0 observer pair sample");
}

fn warm_reference_for_interleaved_pair(run: &FrozenRuntimeFixture, prepared: &PreparedReference) {
    for _ in 0..4 {
        std::hint::black_box(
            r2_routed_preloaded_f32(
                run.input.view(),
                &run.router,
                prepared.config,
                &prepared.preloaded,
            )
            .expect("F32 interleaved observer warmup"),
        );
    }
}

fn warm_candidate_for_interleaved_pair(run: &FrozenRuntimeFixture, prepared: &PreparedCandidate) {
    for _ in 0..4 {
        std::hint::black_box(
            r2_routed_preloaded_candidate(
                run.input.view(),
                &run.router,
                prepared.config,
                &prepared.preloaded,
            )
            .expect("candidate interleaved observer warmup"),
        );
    }
}

fn timed_reference_mini_block(
    run: &FrozenRuntimeFixture,
    prepared: &PreparedReference,
    observer: &str,
) -> ObserverPairRun {
    let session = r2_localization::start(observer_enabled(observer));
    let mut total_nanos = 0_u128;
    let mut last = None;
    for _ in 0..5 {
        let started = Instant::now();
        let output = r2_routed_preloaded_f32(
            run.input.view(),
            &run.router,
            prepared.config,
            &prepared.preloaded,
        )
        .expect("F32 interleaved observer mini-block");
        total_nanos = total_nanos.saturating_add(started.elapsed().as_nanos());
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    let output = last.expect("F32 interleaved observer output");
    ObserverPairRun {
        observer: observer.to_owned(),
        timed_total_nanos: total_nanos,
        snapshot,
        output_sha256: f32_little_endian_sha256(output.data()),
    }
}

fn timed_candidate_mini_block(
    run: &FrozenRuntimeFixture,
    prepared: &PreparedCandidate,
    observer: &str,
) -> ObserverPairRun {
    let session = r2_localization::start(observer_enabled(observer));
    let mut total_nanos = 0_u128;
    let mut last = None;
    for _ in 0..5 {
        let started = Instant::now();
        let output = r2_routed_preloaded_candidate(
            run.input.view(),
            &run.router,
            prepared.config,
            &prepared.preloaded,
        )
        .expect("candidate interleaved observer mini-block");
        total_nanos = total_nanos.saturating_add(started.elapsed().as_nanos());
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    let output = last.expect("candidate interleaved observer output");
    ObserverPairRun {
        observer: observer.to_owned(),
        timed_total_nanos: total_nanos,
        snapshot,
        output_sha256: f32_little_endian_sha256(output.data()),
    }
}

fn mini_pair_order(base: &[String], mini_pair_index: usize) -> Vec<String> {
    assert_eq!(base.len(), 2, "R2.0 base observer pair length");
    if mini_pair_index % 2 == 0 {
        base.to_vec()
    } else {
        vec![base[1].clone(), base[0].clone()]
    }
}

fn observer_results_json(results: &[ObserverPairRun]) -> String {
    let enabled = results
        .iter()
        .find(|result| result.observer == "enabled")
        .expect("enabled mini-pair state");
    let disabled = results
        .iter()
        .find(|result| result.observer == "disabled")
        .expect("disabled mini-pair state");
    format!(
        concat!(
            "{{\"enabled\":{{\"timed_total_nanos\":{},\"output_sha256\":\"{}\",\"stages\":{}}},",
            "\"disabled\":{{\"timed_total_nanos\":{},\"output_sha256\":\"{}\",\"stages\":{}}}}}"
        ),
        enabled.timed_total_nanos,
        enabled.output_sha256,
        snapshot_json(&enabled.snapshot),
        disabled.timed_total_nanos,
        disabled.output_sha256,
        snapshot_json(&disabled.snapshot),
    )
}

#[test]
fn m6_3_r2_0_observer_interleaved_pair_sample() {
    let fixture_name = env::var("COLIBRI_R2_FIXTURE").expect("R2.0 fixture name");
    let path = env::var("COLIBRI_R2_PATH").expect("R2.0 path");
    assert!(matches!(path.as_str(), "reference" | "candidate"));
    let base_order = env::var("COLIBRI_R2_OBSERVER_ORDER")
        .expect("R2.0 observer order")
        .split(',')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(
        matches!(base_order.as_slice(), [left, right] if left != right && matches!(left.as_str(), "enabled" | "disabled") && matches!(right.as_str(), "enabled" | "disabled")),
        "R2.0 observer base pair variants",
    );
    let output_path = PathBuf::from(
        env::var_os("COLIBRI_R2_SAMPLE_OUTPUT").expect("R2.0 interleaved sample output"),
    );
    assert!(
        !output_path.exists(),
        "R2.0 interleaved sample output must be new"
    );
    let run = build_runtime_fixture(&fixture_name, false);
    verify_fixture_identity(&run);
    let noop = r2_localization::calibrate_noop(101);
    let mut mini_pair_documents = Vec::with_capacity(5);
    let mut frozen_output_sha256: Option<String> = None;
    if path == "reference" {
        let prepared = prepare_preloaded_reference(&run);
        warm_reference_for_interleaved_pair(&run, &prepared);
        let expected = env::var("COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256")
            .expect("expected reference output SHA");
        for mini_index in 0..5 {
            let order = mini_pair_order(&base_order, mini_index);
            let results = order
                .iter()
                .map(|observer| timed_reference_mini_block(&run, &prepared, observer))
                .collect::<Vec<_>>();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].output_sha256, results[1].output_sha256);
            assert_eq!(results[0].output_sha256, expected);
            for result in &results {
                assert_compute_only_snapshot(result);
                if let Some(frozen) = &frozen_output_sha256 {
                    assert_eq!(&result.output_sha256, frozen);
                } else {
                    frozen_output_sha256 = Some(result.output_sha256.clone());
                }
            }
            mini_pair_documents.push(format!(
                "{{\"mini_pair\":{},\"observer_order\":[\"{}\",\"{}\"],\"timed_iterations_per_state\":5,\"states\":{}}}",
                mini_index + 1,
                order[0],
                order[1],
                observer_results_json(&results),
            ));
        }
    } else {
        let prepared = prepare_preloaded_candidate(&run);
        warm_candidate_for_interleaved_pair(&run, &prepared);
        for mini_index in 0..5 {
            let order = mini_pair_order(&base_order, mini_index);
            let results = order
                .iter()
                .map(|observer| timed_candidate_mini_block(&run, &prepared, observer))
                .collect::<Vec<_>>();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].output_sha256, results[1].output_sha256);
            for result in &results {
                assert_compute_only_snapshot(result);
                if let Some(frozen) = &frozen_output_sha256 {
                    assert_eq!(&result.output_sha256, frozen);
                } else {
                    frozen_output_sha256 = Some(result.output_sha256.clone());
                }
            }
            mini_pair_documents.push(format!(
                "{{\"mini_pair\":{},\"observer_order\":[\"{}\",\"{}\"],\"timed_iterations_per_state\":5,\"states\":{}}}",
                mini_index + 1,
                order[0],
                order[1],
                observer_results_json(&results),
            ));
        }
    }
    assert_eq!(mini_pair_documents.len(), 5);
    let document = format!(
        concat!(
            "{{\"schema\":\"m6.3-r2.0-observer-pair-sample-v3\",",
            "\"contract_sha256\":\"{}\",\"method_contract_sha256\":\"{}\",",
            "\"fixture\":\"{}\",\"path\":\"{}\",",
            "\"base_observer_order\":[\"{}\",\"{}\"],",
            "\"fixture_record_sha256\":\"{}\",",
            "\"release_binary_sha256\":\"{}\",\"host_id\":\"{}\",",
            "\"warmups_total\":4,\"mini_pairs_per_process\":5,",
            "\"timed_iterations_per_mini_pair_per_state\":5,",
            "\"total_timed_iterations_per_state\":25,",
            "\"timer_noop_median_nanos\":{},\"output_sha256\":\"{}\",",
            "\"mini_pairs\":[{}]}}\n"
        ),
        CONTRACT_SHA256,
        env::var("COLIBRI_R2_METHOD_CONTRACT_SHA256").expect("method contract SHA"),
        fixture_name,
        path,
        base_order[0],
        base_order[1],
        env::var("COLIBRI_R2_FIXTURE_RECORD_SHA256").expect("fixture record SHA"),
        env::var("COLIBRI_R2_RELEASE_BINARY_SHA256").expect("binary SHA"),
        env::var("COLIBRI_R2_HOST_ID").expect("host ID"),
        noop.median_nanos,
        frozen_output_sha256.expect("frozen interleaved output SHA"),
        mini_pair_documents.join(","),
    );
    fs::write(output_path, document).expect("write R2.0 interleaved observer sample");
}

#[derive(Debug)]
struct LocalizationRun {
    timed_total_nanos: u128,
    snapshot: r2_localization::Snapshot,
    output: Tensor,
    candidate_payload_bytes: u64,
    f32_expert_load_bytes: u64,
}

fn selected_unique_expert_count(run: &FrozenRuntimeFixture) -> usize {
    let mut selected = run.router.selected_experts.clone();
    selected.sort_unstable();
    selected.dedup();
    selected.len()
}

fn warm_load_plus_reference(run: &FrozenRuntimeFixture) {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.0 config")
        .runtime_config();
    let layout = PackedExpertLayout::for_config(config);
    for _ in 0..2 {
        let mut store = canonical_layer0_expert_store(&artifact_root);
        std::hint::black_box(
            streaming_routed_experts_with_observer(
                run.input.view(),
                &run.router,
                config,
                0,
                &mut store,
                layout,
                |_, _, _, _| {},
            )
            .expect("F32 load-plus warmup"),
        );
    }
}

fn run_load_plus_reference(run: &FrozenRuntimeFixture) -> LocalizationRun {
    warm_load_plus_reference(run);
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.0 config")
        .runtime_config();
    let layout = PackedExpertLayout::for_config(config);
    let session = r2_localization::start(true);
    let mut timed_total_nanos = 0_u128;
    let mut f32_expert_load_bytes = 0_u64;
    let mut last = None;
    for _ in 0..10 {
        let mut store = canonical_layer0_expert_store(&artifact_root);
        let started = Instant::now();
        let output = streaming_routed_experts_with_observer(
            run.input.view(),
            &run.router,
            config,
            0,
            &mut store,
            layout,
            |_, _, _, _| {},
        )
        .expect("F32 timed load-plus compute");
        timed_total_nanos = timed_total_nanos.saturating_add(started.elapsed().as_nanos());
        f32_expert_load_bytes = f32_expert_load_bytes.saturating_add(store.metrics().bytes_read);
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    LocalizationRun {
        timed_total_nanos,
        snapshot,
        output: last.expect("F32 load-plus output"),
        candidate_payload_bytes: 0,
        f32_expert_load_bytes,
    }
}

fn verified_candidate_reader() -> R1_1CandidateReader {
    let candidate_path =
        PathBuf::from(env::var_os("COLIBRI_R2_CANDIDATE_PATH").expect("R2.0 candidate path"));
    R1_1CandidateReader::open(
        &candidate_path,
        R1_1PackedArtifactLayout::canonical(32).expect("group-32 layout"),
        CANDIDATE_SHA256,
    )
    .expect("verified group-32 candidate")
}

fn warm_load_plus_candidate(run: &FrozenRuntimeFixture, verified: &R1_1CandidateReader) {
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.0 config")
        .runtime_config();
    for _ in 0..2 {
        let mut reader = verified
            .reopen_verified_for_measurement()
            .expect("reopen candidate warmup reader");
        std::hint::black_box(
            r1_1_routed_experts_with_observer(
                run.input.view(),
                &run.router,
                config,
                &mut reader,
                |_, _, _, _| {},
            )
            .expect("candidate load-plus warmup"),
        );
    }
}

fn run_load_plus_candidate(run: &FrozenRuntimeFixture) -> LocalizationRun {
    let verified = verified_candidate_reader();
    warm_load_plus_candidate(run, &verified);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.0 config")
        .runtime_config();
    let session = r2_localization::start(true);
    let mut timed_total_nanos = 0_u128;
    let mut candidate_payload_bytes = 0_u64;
    let mut last = None;
    for _ in 0..10 {
        let mut reader = verified
            .reopen_verified_for_measurement()
            .expect("reopen candidate timed reader");
        let started = Instant::now();
        let output = r1_1_routed_experts_with_observer(
            run.input.view(),
            &run.router,
            config,
            &mut reader,
            |_, _, _, _| {},
        )
        .expect("candidate timed load-plus compute");
        timed_total_nanos = timed_total_nanos.saturating_add(started.elapsed().as_nanos());
        candidate_payload_bytes =
            candidate_payload_bytes.saturating_add(reader.payload_bytes_read());
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    LocalizationRun {
        timed_total_nanos,
        snapshot,
        output: last.expect("candidate load-plus output"),
        candidate_payload_bytes,
        f32_expert_load_bytes: 0,
    }
}

fn run_load_plus_candidate_with_backend(
    run: &FrozenRuntimeFixture,
    backend: R2_1PackedProjectionBackend,
) -> LocalizationRun {
    let verified = verified_candidate_reader();
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R2.1 config")
        .runtime_config();
    for _ in 0..2 {
        let mut reader = verified
            .reopen_verified_for_measurement()
            .expect("reopen R2.1 candidate warmup reader");
        std::hint::black_box(
            r2_1_routed_experts_with_backend(
                run.input.view(),
                &run.router,
                config,
                &mut reader,
                backend,
            )
            .expect("R2.1 candidate load-plus warmup"),
        );
    }
    let session = r2_localization::start(true);
    let mut timed_total_nanos = 0_u128;
    let mut candidate_payload_bytes = 0_u64;
    let mut last = None;
    for _ in 0..10 {
        let mut reader = verified
            .reopen_verified_for_measurement()
            .expect("reopen R2.1 candidate timed reader");
        let started = Instant::now();
        let output = r2_1_routed_experts_with_backend(
            run.input.view(),
            &run.router,
            config,
            &mut reader,
            backend,
        )
        .expect("R2.1 candidate timed load-plus compute");
        timed_total_nanos = timed_total_nanos.saturating_add(started.elapsed().as_nanos());
        candidate_payload_bytes =
            candidate_payload_bytes.saturating_add(reader.payload_bytes_read());
        std::hint::black_box(output.data().first().copied());
        last = Some(output);
    }
    let snapshot = r2_localization::finish(session);
    LocalizationRun {
        timed_total_nanos,
        snapshot,
        output: last.expect("R2.1 candidate load-plus output"),
        candidate_payload_bytes,
        f32_expert_load_bytes: 0,
    }
}

fn localization_expected_bytes(run: &FrozenRuntimeFixture, view: &str, path: &str) -> u64 {
    if view == "compute_only_preloaded" {
        return 0;
    }
    let unique = u64::try_from(selected_unique_expert_count(run)).expect("unique expert count");
    let per_expert = if path == "reference" {
        let config = PINNED_QWEN3_30B_A3B_CONFIG
            .map_to_f32_runtime()
            .expect("R2.0 config")
            .runtime_config();
        u64::try_from(PackedExpertLayout::for_config(config).total_byte_length)
            .expect("F32 expert bytes")
    } else {
        u64::try_from(
            R1_1PackedArtifactLayout::canonical(32)
                .expect("group-32 layout")
                .expert_bytes()
                .expect("group-32 expert bytes"),
        )
        .expect("candidate expert bytes")
    };
    unique
        .checked_mul(per_expert)
        .and_then(|value| value.checked_mul(10))
        .expect("R2.0 expected logical bytes")
}

fn snapshot_logical_bytes(snapshot: &r2_localization::Snapshot) -> u64 {
    snapshot
        .events
        .values()
        .map(|event| event.logical_bytes)
        .sum()
}

fn run_localization_view(run: &FrozenRuntimeFixture, view: &str, path: &str) -> LocalizationRun {
    match (view, path) {
        ("compute_only_preloaded", "reference") => {
            let (timed_total_nanos, snapshot, output) = run_preloaded_reference(run, true);
            LocalizationRun {
                timed_total_nanos,
                snapshot,
                output,
                candidate_payload_bytes: 0,
                f32_expert_load_bytes: 0,
            }
        }
        ("compute_only_preloaded", "candidate") => {
            let (timed_total_nanos, snapshot, output) = run_preloaded_candidate(run, true);
            LocalizationRun {
                timed_total_nanos,
                snapshot,
                output,
                candidate_payload_bytes: 0,
                f32_expert_load_bytes: 0,
            }
        }
        ("load_plus_compute", "reference") => run_load_plus_reference(run),
        ("load_plus_compute", "candidate") => run_load_plus_candidate(run),
        _ => panic!("invalid R2.0 localization view/path"),
    }
}

#[test]
fn m6_3_r2_0_localization_single_sample() {
    let fixture_name = env::var("COLIBRI_R2_FIXTURE").expect("R2.0 fixture name");
    let view = env::var("COLIBRI_R2_VIEW").expect("R2.0 localization view");
    assert!(matches!(
        view.as_str(),
        "compute_only_preloaded" | "load_plus_compute"
    ));
    let path = env::var("COLIBRI_R2_PATH").expect("R2.0 localization path");
    assert!(matches!(path.as_str(), "reference" | "candidate"));
    let pair: usize = env::var("COLIBRI_R2_PAIR")
        .expect("R2.0 pair")
        .parse()
        .expect("numeric R2.0 pair");
    let order_index: usize = env::var("COLIBRI_R2_ORDER_INDEX")
        .expect("R2.0 order index")
        .parse()
        .expect("numeric R2.0 order index");
    assert!((1..=5).contains(&pair));
    assert!(matches!(order_index, 1 | 2));
    let output_path = PathBuf::from(
        env::var_os("COLIBRI_R2_SAMPLE_OUTPUT").expect("R2.0 localization sample output"),
    );
    assert!(
        !output_path.exists(),
        "R2.0 localization output must be new"
    );
    let run = build_runtime_fixture(&fixture_name, false);
    verify_fixture_identity(&run);
    let noop = r2_localization::calibrate_noop(101);
    let result = run_localization_view(&run, &view, &path);
    let output_sha256 = f32_little_endian_sha256(result.output.data());
    if path == "reference" {
        assert_eq!(
            output_sha256,
            env::var("COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256")
                .expect("expected reference output SHA"),
            "R2.0 localization reference output identity",
        );
    }
    let timed_iterations = if view == "compute_only_preloaded" {
        25
    } else {
        10
    };
    let expected_logical_expert_bytes = localization_expected_bytes(&run, &view, &path);
    let logical_expert_bytes = snapshot_logical_bytes(&result.snapshot);
    assert_eq!(logical_expert_bytes, expected_logical_expert_bytes);
    if path == "candidate" {
        assert_eq!(
            result.candidate_payload_bytes,
            expected_logical_expert_bytes
        );
        assert_eq!(result.f32_expert_load_bytes, 0);
    } else {
        assert_eq!(result.f32_expert_load_bytes, expected_logical_expert_bytes);
        assert_eq!(result.candidate_payload_bytes, 0);
    }
    let unique_expert_loads = result
        .snapshot
        .counters
        .get("unique_expert_loads")
        .copied()
        .unwrap_or(0);
    let expected_unique_expert_loads = if view == "compute_only_preloaded" {
        0
    } else {
        u64::try_from(selected_unique_expert_count(&run))
            .expect("unique expert count")
            .checked_mul(10)
            .expect("expected unique expert loads")
    };
    assert_eq!(unique_expert_loads, expected_unique_expert_loads);
    let expert_occurrences = result
        .snapshot
        .counters
        .get("expert_occurrences")
        .copied()
        .unwrap_or(0);
    assert!(expert_occurrences > 0);
    let expert_total = result
        .snapshot
        .events
        .get("expert_total")
        .expect("R2.0 expert_total stage");
    let exclusive_residual_nanos = expert_total.exclusive_nanos;
    let attempt_ordinal = 1_u32;
    let document = format!(
        concat!(
            "{{\"schema\":\"m6.3-r2.0-localization-sample-v1\",",
            "\"contract_sha256\":\"{}\",\"fixture\":\"{}\",",
            "\"view\":\"{}\",\"path\":\"{}\",",
            "\"pair\":{},\"order_index\":{},\"attempt_ordinal\":{},",
            "\"release_binary_sha256\":\"{}\",\"host_id\":\"{}\",",
            "\"timed_iterations\":{},\"timed_total_nanos\":{},",
            "\"output_sha256\":\"{}\",\"timer_noop_median_nanos\":{},",
            "\"expert_occurrences\":{},\"unique_expert_loads\":{},",
            "\"expected_logical_expert_bytes\":{},\"logical_expert_bytes\":{},",
            "\"timed_candidate_payload_bytes\":{},",
            "\"timed_f32_expert_load_bytes\":{},",
            "\"exclusive_residual_nanos\":{},\"stages\":{}}}\n"
        ),
        CONTRACT_SHA256,
        fixture_name,
        view,
        path,
        pair,
        order_index,
        attempt_ordinal,
        env::var("COLIBRI_R2_RELEASE_BINARY_SHA256").expect("binary SHA"),
        env::var("COLIBRI_R2_HOST_ID").expect("host ID"),
        timed_iterations,
        result.timed_total_nanos,
        output_sha256,
        noop.median_nanos,
        expert_occurrences,
        unique_expert_loads,
        expected_logical_expert_bytes,
        logical_expert_bytes,
        result.candidate_payload_bytes,
        result.f32_expert_load_bytes,
        exclusive_residual_nanos,
        snapshot_json(&result.snapshot),
    );
    fs::write(output_path, document).expect("write R2.0 localization sample");
}

const R2_1_CONTRACT_SHA256: &str =
    "76bfc4a850a8d89fb5901eda338568b7226b36acb7207324575568ea21f6cc2a";

fn run_r2_1_view(run: &FrozenRuntimeFixture, view: &str, path: &str) -> LocalizationRun {
    match (view, path) {
        ("compute_only_preloaded", "reference_f32") => {
            let (timed_total_nanos, snapshot, output) = run_preloaded_reference(run, true);
            LocalizationRun {
                timed_total_nanos,
                snapshot,
                output,
                candidate_payload_bytes: 0,
                f32_expert_load_bytes: 0,
            }
        }
        ("compute_only_preloaded", "scalar_group32") => {
            let prepared = prepare_preloaded_candidate(run);
            let (timed_total_nanos, snapshot, output) = timed_preloaded_candidate_with_backend(
                run,
                &prepared,
                R2_1PackedProjectionBackend::Scalar,
            );
            LocalizationRun {
                timed_total_nanos,
                snapshot,
                output,
                candidate_payload_bytes: 0,
                f32_expert_load_bytes: 0,
            }
        }
        ("compute_only_preloaded", "native_avx2_fma_group32") => {
            let prepared = prepare_preloaded_candidate(run);
            let (timed_total_nanos, snapshot, output) = timed_preloaded_candidate_with_backend(
                run,
                &prepared,
                R2_1PackedProjectionBackend::NativeAvx2Fma,
            );
            LocalizationRun {
                timed_total_nanos,
                snapshot,
                output,
                candidate_payload_bytes: 0,
                f32_expert_load_bytes: 0,
            }
        }
        ("load_plus_compute", "reference_f32") => run_load_plus_reference(run),
        ("load_plus_compute", "scalar_group32") => {
            run_load_plus_candidate_with_backend(run, R2_1PackedProjectionBackend::Scalar)
        }
        ("load_plus_compute", "native_avx2_fma_group32") => {
            run_load_plus_candidate_with_backend(run, R2_1PackedProjectionBackend::NativeAvx2Fma)
        }
        _ => panic!("invalid R2.1 performance view/path"),
    }
}

#[test]
fn m6_3_r2_1_performance_single_sample() {
    assert!(
        crate::r2_native::avx2_fma_available(),
        "R2.1 AVX2+FMA host required"
    );
    let fixture_name = env::var("COLIBRI_R2_FIXTURE").expect("R2.1 fixture name");
    let view = env::var("COLIBRI_R2_VIEW").expect("R2.1 performance view");
    assert!(matches!(
        view.as_str(),
        "compute_only_preloaded" | "load_plus_compute"
    ));
    let path = env::var("COLIBRI_R2_PATH").expect("R2.1 performance path");
    assert!(matches!(
        path.as_str(),
        "native_avx2_fma_group32" | "scalar_group32" | "reference_f32"
    ));
    let triplet: usize = env::var("COLIBRI_R2_PAIR")
        .expect("R2.1 triplet")
        .parse()
        .expect("numeric R2.1 triplet");
    let order_index: usize = env::var("COLIBRI_R2_ORDER_INDEX")
        .expect("R2.1 order index")
        .parse()
        .expect("numeric R2.1 order index");
    assert!((1..=6).contains(&triplet));
    assert!((1..=3).contains(&order_index));
    let output_path = PathBuf::from(
        env::var_os("COLIBRI_R2_SAMPLE_OUTPUT").expect("R2.1 performance sample output"),
    );
    assert!(!output_path.exists(), "R2.1 performance output must be new");
    let run = build_runtime_fixture(&fixture_name, false);
    verify_fixture_identity(&run);
    let result = run_r2_1_view(&run, &view, &path);
    let output_sha256 = f32_little_endian_sha256(result.output.data());
    if path == "reference_f32" {
        assert_eq!(
            output_sha256,
            env::var("COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256")
                .expect("expected reference output SHA"),
        );
    }
    let timed_iterations = if view == "compute_only_preloaded" {
        25
    } else {
        10
    };
    let expected_path = if path == "reference_f32" {
        "reference"
    } else {
        "candidate"
    };
    let expected_logical_expert_bytes = localization_expected_bytes(&run, &view, expected_path);
    let logical_expert_bytes = snapshot_logical_bytes(&result.snapshot);
    assert_eq!(logical_expert_bytes, expected_logical_expert_bytes);
    if path == "reference_f32" {
        assert_eq!(result.f32_expert_load_bytes, expected_logical_expert_bytes);
        assert_eq!(result.candidate_payload_bytes, 0);
    } else {
        assert_eq!(
            result.candidate_payload_bytes,
            expected_logical_expert_bytes
        );
        assert_eq!(result.f32_expert_load_bytes, 0);
    }
    let attempt_ordinal = 1_u32;
    let document = format!(
        concat!(
            "{{\"schema\":\"m6.3-r2.1-performance-sample-v1\",",
            "\"contract_sha256\":\"{}\",\"fixture\":\"{}\",",
            "\"view\":\"{}\",\"path\":\"{}\",",
            "\"triplet\":{},\"order_index\":{},\"attempt_ordinal\":{},",
            "\"release_binary_sha256\":\"{}\",\"host_id\":\"{}\",",
            "\"timed_iterations\":{},\"timed_total_nanos\":{},",
            "\"output_sha256\":\"{}\",",
            "\"expected_logical_expert_bytes\":{},\"logical_expert_bytes\":{},",
            "\"timed_candidate_payload_bytes\":{},",
            "\"timed_f32_expert_load_bytes\":{}}}\n"
        ),
        R2_1_CONTRACT_SHA256,
        fixture_name,
        view,
        path,
        triplet,
        order_index,
        attempt_ordinal,
        env::var("COLIBRI_R2_RELEASE_BINARY_SHA256").expect("binary SHA"),
        env::var("COLIBRI_R2_HOST_ID").expect("host ID"),
        timed_iterations,
        result.timed_total_nanos,
        output_sha256,
        expected_logical_expert_bytes,
        logical_expert_bytes,
        result.candidate_payload_bytes,
        result.f32_expert_load_bytes,
    );
    fs::write(output_path, document).expect("write R2.1 performance sample");
}
