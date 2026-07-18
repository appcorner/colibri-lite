//! Isolated M5.3-01 storage-path measurements.
//!
//! This example exercises the current `ArtifactReader` and two local,
//! non-production read loops. It does not change `ExpertStore`, cache policy,
//! artifact layout, or the runtime reader. All payloads are hash-checked.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::Instant,
};

use clr_core::{DataType, TensorShape};
use clr_storage::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ArtifactReader, ByteOrder, CacheMetrics, ExpertId,
    ExpertKey, ExpertPathMetrics, ExpertRegistration, ExpertStore, ReaderMetrics, ReaderMode,
    TensorLocation, TensorMetadata, sha256_digest,
};

const PAYLOAD_BYTES: u64 = 18_874_368;
const DEFAULT_PLAN: &str = "models/qwen3-30b-a3b/m4.2-04-layer47-expert-runtime-plan-v1.tsv";
const DEFAULT_CROSS_SHARD_PLAN: &str =
    "models/qwen3-30b-a3b/m4.2-02-layer0-selected-expert-runtime-plan-v1.tsv";
const DEFAULT_SEQUENCE: &str = "models/qwen3-30b-a3b/m5.3-01-layer47-miss-sequence-v1.tsv";

#[derive(Debug, Clone)]
struct ExpertRange {
    name: String,
    layer_index: u32,
    expert_id: u32,
    path: String,
    offset: u64,
    length: u64,
    hash: [u8; 32],
}

#[derive(Debug, Clone, Copy)]
struct RunResult {
    elapsed_nanos: u128,
    metrics: ReaderMetrics,
}

#[derive(Debug, Clone, Copy)]
struct StoreRunResult {
    elapsed_nanos: u128,
    cache: CacheMetrics,
    path: ExpertPathMetrics,
    reader: ReaderMetrics,
}

fn env_path(name: &str, default: &str) -> PathBuf {
    std::env::var_os(name).map_or_else(|| PathBuf::from(default), PathBuf::from)
}

fn decode_hash(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64, "SHA-256 text length");
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .expect("lowercase SHA-256 text");
    }
    output
}

fn load_plan(path: &Path) -> Vec<ExpertRange> {
    let text = std::fs::read_to_string(path).expect("read runtime plan");
    text.lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.first().copied() != Some("expert") {
                return None;
            }
            assert_eq!(fields.len(), 8, "expert runtime plan fields");
            let length = fields[6].parse::<u64>().expect("expert length");
            assert_eq!(length, PAYLOAD_BYTES, "packed expert payload length");
            Some(ExpertRange {
                name: fields[3].to_owned(),
                layer_index: fields[1].parse().expect("expert layer index"),
                expert_id: fields[2].parse().expect("expert ID"),
                path: fields[4].to_owned(),
                offset: fields[5].parse().expect("expert offset"),
                length,
                hash: decode_hash(fields[7]),
            })
        })
        .collect()
}

fn build_manifest(ranges: &[ExpertRange]) -> ArtifactManifest {
    let metadata = ranges
        .iter()
        .map(|range| TensorMetadata {
            name: range.name.clone(),
            shape: TensorShape::new([usize::try_from(range.length / 4).expect("F32 length")]),
            data_type: DataType::F32,
            location: TensorLocation {
                path: range.path.clone().into(),
                offset: range.offset,
                length: range.length,
            },
            sha256: range.hash,
        })
        .collect();
    ArtifactManifest::new(ARTIFACT_FORMAT_VERSION, ByteOrder::Little, metadata)
        .expect("valid expert manifest")
}

fn build_reader(artifact_root: &Path, ranges: &[ExpertRange]) -> ArtifactReader {
    build_reader_mode(artifact_root, ranges, ReaderMode::Reference)
}

fn build_reader_mode(
    artifact_root: &Path,
    ranges: &[ExpertRange],
    mode: ReaderMode,
) -> ArtifactReader {
    ArtifactReader::open_with_mode(artifact_root.join("experts"), build_manifest(ranges), mode)
        .expect("open expert reader")
}

fn build_store(artifact_root: &Path, ranges: &[ExpertRange], mode: ReaderMode) -> ExpertStore {
    let registrations = ranges
        .iter()
        .map(|range| ExpertRegistration {
            key: ExpertKey {
                layer_index: range.layer_index,
                expert_id: ExpertId(range.expert_id),
            },
            tensor_name: range.name.clone(),
        })
        .collect();
    ExpertStore::new(
        build_reader_mode(artifact_root, ranges, mode),
        registrations,
        usize::try_from(PAYLOAD_BYTES).expect("payload usize"),
    )
    .expect("expert store")
}

fn read_current(reader: &ArtifactReader, ranges: &[ExpertRange], ids: &[usize]) -> RunResult {
    let before = reader.metrics();
    let started = Instant::now();
    for &id in ids {
        let tensor = reader
            .read_tensor(&ranges[id].name)
            .expect("current reader payload");
        assert_eq!(
            tensor.bytes.len(),
            usize::try_from(PAYLOAD_BYTES).expect("payload usize")
        );
    }
    RunResult {
        elapsed_nanos: started.elapsed().as_nanos(),
        metrics: {
            let after = reader.metrics();
            subtract_metrics(&after, &before)
        },
    }
}

fn read_direct(
    artifact_root: &Path,
    ranges: &[ExpertRange],
    ids: &[usize],
    reusable_buffer: bool,
) -> u128 {
    let path = artifact_root
        .join("experts")
        .join("experts-layer-00047-of-00048.bin");
    let mut file = File::open(path).expect("open persistent benchmark handle");
    let mut buffer = vec![0_u8; usize::try_from(PAYLOAD_BYTES).expect("payload usize")];
    let started = Instant::now();
    for &id in ids {
        file.seek(SeekFrom::Start(ranges[id].offset))
            .expect("benchmark seek");
        if reusable_buffer {
            file.read_exact(&mut buffer)
                .expect("benchmark reusable read");
            assert_eq!(
                sha256_digest(&buffer),
                ranges[id].hash,
                "reusable payload hash"
            );
        } else {
            let mut fresh = vec![0_u8; buffer.len()];
            file.read_exact(&mut fresh).expect("benchmark fresh read");
            assert_eq!(sha256_digest(&fresh), ranges[id].hash, "fresh payload hash");
        }
    }
    started.elapsed().as_nanos()
}

fn read_persistent_reusable(artifact_root: &Path, ranges: &[ExpertRange], ids: &[usize]) -> u128 {
    read_direct(artifact_root, ranges, ids, true)
}

fn read_sequence(path: &Path, ranges: &[ExpertRange]) -> Vec<usize> {
    let text = std::fs::read_to_string(path).expect("read M5.3 sequence");
    let mut ids = Vec::new();
    for line in text
        .lines()
        .skip(1)
        .filter(|line| line.split('\t').nth(1) == Some("8_gib"))
    {
        let expert = line
            .split('\t')
            .nth(4)
            .expect("sequence expert")
            .parse::<usize>()
            .expect("sequence expert integer");
        let id = ranges
            .iter()
            .position(|range| range.name == format!("layer.47.expert.{expert}"))
            .expect("sequence expert in runtime plan");
        ids.push(id);
        if ids.len() == 64 {
            break;
        }
    }
    assert_eq!(ids.len(), 64, "authoritative benchmark subset length");
    ids
}

fn run_store(
    artifact_root: &Path,
    ranges: &[ExpertRange],
    ids: &[usize],
    mode: ReaderMode,
) -> StoreRunResult {
    let mut store = build_store(artifact_root, ranges, mode);
    let started = Instant::now();
    for &id in ids {
        let lease = store
            .load(ExpertKey {
                layer_index: ranges[id].layer_index,
                expert_id: ExpertId(ranges[id].expert_id),
            })
            .expect("expert store benchmark load");
        assert_eq!(
            lease.bytes().len(),
            usize::try_from(PAYLOAD_BYTES).expect("payload usize")
        );
    }
    StoreRunResult {
        elapsed_nanos: started.elapsed().as_nanos(),
        cache: store.metrics(),
        path: store.path_metrics(),
        reader: store.reader_metrics(),
    }
}

fn subtract_metrics(after: &ReaderMetrics, before: &ReaderMetrics) -> ReaderMetrics {
    ReaderMetrics {
        tensor_reads: after.tensor_reads - before.tensor_reads,
        file_open_count: after.file_open_count - before.file_open_count,
        file_handle_reuse_count: after.file_handle_reuse_count - before.file_handle_reuse_count,
        metadata_count: after.metadata_count - before.metadata_count,
        seek_count: after.seek_count - before.seek_count,
        read_call_count: after.read_call_count - before.read_call_count,
        requested_read_bytes: after.requested_read_bytes - before.requested_read_bytes,
        returned_read_bytes: after.returned_read_bytes - before.returned_read_bytes,
        buffer_allocation_count: after.buffer_allocation_count - before.buffer_allocation_count,
        allocated_bytes: after.allocated_bytes - before.allocated_bytes,
        copied_bytes: after.copied_bytes - before.copied_bytes,
        buffer_growth_events: after.buffer_growth_events - before.buffer_growth_events,
        buffer_reuse_count: after.buffer_reuse_count - before.buffer_reuse_count,
        bytes_read_into_reusable_buffers: after.bytes_read_into_reusable_buffers
            - before.bytes_read_into_reusable_buffers,
        bytes_copied_after_read: after.bytes_copied_after_read - before.bytes_copied_after_read,
        peak_buffer_capacity: after.peak_buffer_capacity,
        fallback_allocations: after.fallback_allocations - before.fallback_allocations,
        alignment_failures: after.alignment_failures - before.alignment_failures,
        hash_bytes: after.hash_bytes - before.hash_bytes,
        open_nanos: after.open_nanos - before.open_nanos,
        metadata_nanos: after.metadata_nanos - before.metadata_nanos,
        seek_nanos: after.seek_nanos - before.seek_nanos,
        read_nanos: after.read_nanos - before.read_nanos,
        hash_nanos: after.hash_nanos - before.hash_nanos,
        mmap_mapping_count: after.mmap_mapping_count - before.mmap_mapping_count,
        mmap_shard_reuse_count: after.mmap_shard_reuse_count - before.mmap_shard_reuse_count,
        mmap_active_mapping_count: after.mmap_active_mapping_count,
        mmap_peak_mapping_count: after.mmap_peak_mapping_count,
        mmap_mapped_virtual_bytes: after.mmap_mapped_virtual_bytes,
        mmap_peak_mapped_virtual_bytes: after.mmap_peak_mapped_virtual_bytes,
        mmap_mapping_init_nanos: after.mmap_mapping_init_nanos - before.mmap_mapping_init_nanos,
        mmap_first_touch_nanos: after.mmap_first_touch_nanos - before.mmap_first_touch_nanos,
        mmap_access_nanos: after.mmap_access_nanos - before.mmap_access_nanos,
        mmap_copy_nanos: after.mmap_copy_nanos - before.mmap_copy_nanos,
        mmap_copy_bytes: after.mmap_copy_bytes - before.mmap_copy_bytes,
    }
}

fn metrics_json(metrics: &ReaderMetrics) -> String {
    format!(
        "{{\"tensor_reads\":{},\"file_open_count\":{},\"file_handle_reuse_count\":{},\"metadata_count\":{},\"seek_count\":{},\"read_call_count\":{},\"requested_read_bytes\":{},\"returned_read_bytes\":{},\"buffer_allocation_count\":{},\"allocated_bytes\":{},\"copied_bytes\":{},\"buffer_growth_events\":{},\"buffer_reuse_count\":{},\"bytes_read_into_reusable_buffers\":{},\"bytes_copied_after_read\":{},\"peak_buffer_capacity\":{},\"fallback_allocations\":{},\"alignment_failures\":{},\"hash_bytes\":{},\"open_nanos\":{},\"metadata_nanos\":{},\"seek_nanos\":{},\"read_nanos\":{},\"hash_nanos\":{},\"mmap_mapping_count\":{},\"mmap_shard_reuse_count\":{},\"mmap_active_mapping_count\":{},\"mmap_peak_mapping_count\":{},\"mmap_mapped_virtual_bytes\":{},\"mmap_peak_mapped_virtual_bytes\":{},\"mmap_mapping_init_nanos\":{},\"mmap_first_touch_nanos\":{},\"mmap_access_nanos\":{},\"mmap_copy_nanos\":{},\"mmap_copy_bytes\":{}}}",
        metrics.tensor_reads,
        metrics.file_open_count,
        metrics.file_handle_reuse_count,
        metrics.metadata_count,
        metrics.seek_count,
        metrics.read_call_count,
        metrics.requested_read_bytes,
        metrics.returned_read_bytes,
        metrics.buffer_allocation_count,
        metrics.allocated_bytes,
        metrics.copied_bytes,
        metrics.buffer_growth_events,
        metrics.buffer_reuse_count,
        metrics.bytes_read_into_reusable_buffers,
        metrics.bytes_copied_after_read,
        metrics.peak_buffer_capacity,
        metrics.fallback_allocations,
        metrics.alignment_failures,
        metrics.hash_bytes,
        metrics.open_nanos,
        metrics.metadata_nanos,
        metrics.seek_nanos,
        metrics.read_nanos,
        metrics.hash_nanos,
        metrics.mmap_mapping_count,
        metrics.mmap_shard_reuse_count,
        metrics.mmap_active_mapping_count,
        metrics.mmap_peak_mapping_count,
        metrics.mmap_mapped_virtual_bytes,
        metrics.mmap_peak_mapped_virtual_bytes,
        metrics.mmap_mapping_init_nanos,
        metrics.mmap_first_touch_nanos,
        metrics.mmap_access_nanos,
        metrics.mmap_copy_nanos,
        metrics.mmap_copy_bytes,
    )
}

fn current_run_json(requests: usize, result: &RunResult) -> String {
    format!(
        "{{\"requests\":{},\"elapsed_nanos\":{},\"metrics\":{}}}",
        requests,
        result.elapsed_nanos,
        metrics_json(&result.metrics)
    )
}

fn store_run_json(requests: usize, result: &StoreRunResult) -> String {
    let cache = result.cache;
    let path = result.path;
    let reader = result.reader;
    format!(
        "{{\"requests\":{},\"elapsed_nanos\":{},\"cache\":{{\"hits\":{},\"misses\":{},\"loads\":{},\"evictions\":{},\"bytes_read\":{}}},\"path_metrics\":{{\"request_count\":{},\"cache_hit_count\":{},\"expert_load_count\":{},\"total_nanos\":{},\"cache_lookup_nanos\":{},\"expert_load_nanos\":{},\"bytes_copied_after_read\":{}}},\"reader_metrics\":{}}}",
        requests,
        result.elapsed_nanos,
        cache.hits,
        cache.misses,
        cache.loads,
        cache.evictions,
        cache.bytes_read,
        path.request_count,
        path.cache_hit_count,
        path.expert_load_count,
        path.total_nanos,
        path.cache_lookup_nanos,
        path.expert_load_nanos,
        path.bytes_copied_after_read,
        metrics_json(&reader),
    )
}

fn repeated_store_runs_json(
    artifact_root: &Path,
    ranges: &[ExpertRange],
    ids: &[usize],
    mode: ReaderMode,
    repeats: usize,
) -> String {
    let mut output = String::from("[");
    for repeat in 0..repeats {
        if repeat != 0 {
            output.push(',');
        }
        let result = run_store(artifact_root, ranges, ids, mode);
        output.push_str(&store_run_json(ids.len(), &result));
    }
    output.push(']');
    output
}

fn scenario_store_runs_json(
    artifact_root: &Path,
    ranges: &[ExpertRange],
    scenarios: &[(&str, &[usize])],
    mode: ReaderMode,
    repeats: usize,
) -> String {
    let mut output = String::from("{");
    for (index, (name, ids)) in scenarios.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push('"');
        output.push_str(name);
        output.push_str("\":");
        output.push_str(&repeated_store_runs_json(
            artifact_root,
            ranges,
            ids,
            mode,
            repeats,
        ));
    }
    output.push('}');
    output
}

#[allow(clippy::too_many_lines)]
fn main() {
    let artifact_root = env_path(
        "COLIBRI_ARTIFACT_ROOT",
        r"D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1",
    );
    let plan_path = env_path("COLIBRI_M5_3_RUNTIME_PLAN", DEFAULT_PLAN);
    let cross_shard_plan_path = env_path("COLIBRI_M5_3_CROSS_SHARD_PLAN", DEFAULT_CROSS_SHARD_PLAN);
    let sequence_path = env_path("COLIBRI_M5_3_SEQUENCE", DEFAULT_SEQUENCE);
    let output_path = std::env::var_os("COLIBRI_M5_3_BENCH_OUTPUT").map_or_else(
        || PathBuf::from("m5.3-01-storage-bench.json"),
        PathBuf::from,
    );
    let mut ranges = load_plan(&plan_path);
    assert_eq!(ranges.len(), 128, "layer-47 expert plan count");
    let cross_shard_start = ranges.len();
    ranges.extend(load_plan(&cross_shard_plan_path));
    assert!(
        ranges.len() > cross_shard_start,
        "cross-shard plan must contain at least one expert"
    );
    let reader = build_reader(&artifact_root, &ranges);
    let repeats = std::env::var("COLIBRI_M5_3_BENCH_REPEATS").map_or(2, |value| {
        value.parse().expect("valid benchmark repeat count")
    });
    assert!(repeats > 0, "benchmark repeat count must be positive");
    let repeated = vec![0; 3];
    let randomized: Vec<_> = (0..32).map(|index| (index * 73 + 19) % 128).collect();
    let authoritative = read_sequence(&sequence_path, &ranges);
    let clustered: Vec<_> = (0..32).collect();
    let cross_shard: Vec<_> = (0..8)
        .flat_map(|index| [index, cross_shard_start + index])
        .collect();
    let current_repeated = read_current(&reader, &ranges, &repeated);
    let current_randomized = read_current(&reader, &ranges, &randomized);
    let current_authoritative = read_current(&reader, &ranges, &authoritative);
    let fresh_authoritative = read_direct(&artifact_root, &ranges, &authoritative, false);
    let reusable_authoritative = read_persistent_reusable(&artifact_root, &ranges, &authoritative);
    let reference_store_json = repeated_store_runs_json(
        &artifact_root,
        &ranges,
        &authoritative,
        ReaderMode::Reference,
        repeats,
    );
    let reusable_store_json = repeated_store_runs_json(
        &artifact_root,
        &ranges,
        &authoritative,
        ReaderMode::ReusableAlignedBuffer,
        repeats,
    );
    let scenarios = [
        ("repeated_same_expert", repeated.as_slice()),
        ("randomized_experts", randomized.as_slice()),
        (
            "authoritative_layer47_miss_subset",
            authoritative.as_slice(),
        ),
        ("same_shard_clustered", clustered.as_slice()),
        ("cross_shard_access", cross_shard.as_slice()),
    ];
    let reference_scenarios_json = scenario_store_runs_json(
        &artifact_root,
        &ranges,
        &scenarios,
        ReaderMode::Reference,
        repeats,
    );
    let reusable_scenarios_json = scenario_store_runs_json(
        &artifact_root,
        &ranges,
        &scenarios,
        ReaderMode::ReusableAlignedBuffer,
        repeats,
    );
    #[cfg(feature = "m5-3-mmap")]
    let mmap_scenarios_json = scenario_store_runs_json(
        &artifact_root,
        &ranges,
        &scenarios,
        ReaderMode::MmapReadOnly,
        repeats,
    );
    #[cfg(feature = "m5-3-mmap")]
    let mmap_section = format!(",\"mmap_read_only\":{mmap_scenarios_json}");
    #[cfg(not(feature = "m5-3-mmap"))]
    let mmap_section = String::new();
    #[cfg(feature = "m5-3-mmap")]
    let schema = "colibri-qwen3-moe-m5.3-04-mmap-benchmark-v1";
    #[cfg(not(feature = "m5-3-mmap"))]
    let schema = "colibri-qwen3-moe-m5.3-02-reusable-buffer-benchmark-v1";
    let output = [
        "{\"schema\":\"",
        schema,
        "\",\"schema_version\":1,\"artifact_root_sha256\":\"f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2\",\"payload_bytes\":",
        &PAYLOAD_BYTES.to_string(),
        ",\"warm_cache_state\":\"uncontrolled\",\"current_reader\":{\"repeated_same_expert\":",
        &current_run_json(3, &current_repeated),
        ",\"randomized_experts\":",
        &current_run_json(randomized.len(), &current_randomized),
        ",\"authoritative_layer47_miss_subset\":",
        &current_run_json(authoritative.len(), &current_authoritative),
        "},\"isolated_same_range_access\":{\"authoritative_layer47_miss_subset\":{\"requests\":",
        &authoritative.len().to_string(),
        ",\"fresh_allocation_persistent_handle_elapsed_nanos\":",
        &fresh_authoritative.to_string(),
        ",\"reusable_buffer_persistent_handle_elapsed_nanos\":",
        &reusable_authoritative.to_string(),
        "}},\"expert_store_authoritative_layer47_miss_subset\":{\"repeat_count\":",
        &repeats.to_string(),
        ",\"reference_allocated\":",
        &reference_store_json,
        ",\"reusable_aligned_buffer\":",
        &reusable_store_json,
        "},\"expert_store_scenarios\":{\"reference_allocated\":",
        &reference_scenarios_json,
        ",\"reusable_aligned_buffer\":",
        &reusable_scenarios_json,
        &mmap_section,
        "}}\n",
    ]
    .concat();
    std::fs::write(&output_path, output).expect("write benchmark evidence");
    println!(
        "m5_3_storage_bench_complete output={} requests={}",
        output_path.display(),
        authoritative.len()
    );
}
