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
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ArtifactReader, ByteOrder, ExpertId, ExpertKey,
    ExpertRegistration, ExpertStore, ReaderMetrics, TensorLocation, TensorMetadata, sha256_digest,
};

const PAYLOAD_BYTES: u64 = 18_874_368;
const DEFAULT_PLAN: &str = "models/qwen3-30b-a3b/m4.2-04-layer47-expert-runtime-plan-v1.tsv";
const DEFAULT_SEQUENCE: &str = "models/qwen3-30b-a3b/m5.3-01-layer47-miss-sequence-v1.tsv";

#[derive(Debug, Clone)]
struct ExpertRange {
    name: String,
    offset: u64,
    length: u64,
    hash: [u8; 32],
}

#[derive(Debug, Clone, Copy)]
struct RunResult {
    elapsed_nanos: u128,
    metrics: ReaderMetrics,
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
                path: "experts-layer-00047-of-00048.bin".into(),
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
    ArtifactReader::open(artifact_root.join("experts"), build_manifest(ranges))
        .expect("open expert reader")
}

fn build_store(artifact_root: &Path, ranges: &[ExpertRange]) -> ExpertStore {
    let registrations = ranges
        .iter()
        .enumerate()
        .map(|(index, range)| ExpertRegistration {
            key: ExpertKey {
                layer_index: 47,
                expert_id: ExpertId(u32::try_from(index).expect("expert ID")),
            },
            tensor_name: range.name.clone(),
        })
        .collect();
    ExpertStore::new(
        ArtifactReader::open(artifact_root.join("experts"), build_manifest(ranges))
            .expect("open expert store reader"),
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
        metrics: subtract_metrics(reader.metrics(), before),
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

fn subtract_metrics(after: ReaderMetrics, before: ReaderMetrics) -> ReaderMetrics {
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
        hash_bytes: after.hash_bytes - before.hash_bytes,
        open_nanos: after.open_nanos - before.open_nanos,
        metadata_nanos: after.metadata_nanos - before.metadata_nanos,
        seek_nanos: after.seek_nanos - before.seek_nanos,
        read_nanos: after.read_nanos - before.read_nanos,
        hash_nanos: after.hash_nanos - before.hash_nanos,
    }
}

fn metrics_json(metrics: ReaderMetrics) -> String {
    format!(
        "{{\"tensor_reads\":{},\"file_open_count\":{},\"file_handle_reuse_count\":{},\"metadata_count\":{},\"seek_count\":{},\"read_call_count\":{},\"requested_read_bytes\":{},\"returned_read_bytes\":{},\"buffer_allocation_count\":{},\"allocated_bytes\":{},\"copied_bytes\":{},\"hash_bytes\":{},\"open_nanos\":{},\"metadata_nanos\":{},\"seek_nanos\":{},\"read_nanos\":{},\"hash_nanos\":{}}}",
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
        metrics.hash_bytes,
        metrics.open_nanos,
        metrics.metadata_nanos,
        metrics.seek_nanos,
        metrics.read_nanos,
        metrics.hash_nanos,
    )
}

fn current_run_json(requests: usize, result: RunResult) -> String {
    format!(
        "{{\"requests\":{},\"elapsed_nanos\":{},\"metrics\":{}}}",
        requests,
        result.elapsed_nanos,
        metrics_json(result.metrics)
    )
}

fn expert_store_run_json(requests: usize, elapsed_nanos: u128, store: &ExpertStore) -> String {
    let cache = store.metrics();
    let path = store.path_metrics();
    let reader = store.reader_metrics();
    format!(
        "{{\"requests\":{},\"elapsed_nanos\":{},\"cache\":{{\"hits\":{},\"misses\":{},\"loads\":{},\"evictions\":{},\"bytes_read\":{}}},\"path_metrics\":{{\"request_count\":{},\"cache_hit_count\":{},\"expert_load_count\":{},\"total_nanos\":{},\"cache_lookup_nanos\":{},\"expert_load_nanos\":{}}},\"reader_metrics\":{}}}",
        requests,
        elapsed_nanos,
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
        metrics_json(reader),
    )
}

fn main() {
    let artifact_root = env_path(
        "COLIBRI_ARTIFACT_ROOT",
        r"D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1",
    );
    let plan_path = env_path("COLIBRI_M5_3_RUNTIME_PLAN", DEFAULT_PLAN);
    let sequence_path = env_path("COLIBRI_M5_3_SEQUENCE", DEFAULT_SEQUENCE);
    let output_path = std::env::var_os("COLIBRI_M5_3_BENCH_OUTPUT").map_or_else(
        || PathBuf::from("m5.3-01-storage-bench.json"),
        PathBuf::from,
    );
    let ranges = load_plan(&plan_path);
    assert_eq!(ranges.len(), 128, "layer-47 expert plan count");
    let reader = build_reader(&artifact_root, &ranges);
    let repeated = vec![0; 3];
    let randomized: Vec<_> = (0..32).map(|index| (index * 73 + 19) % 128).collect();
    let authoritative = read_sequence(&sequence_path, &ranges);
    let current_repeated = read_current(&reader, &ranges, &repeated);
    let current_randomized = read_current(&reader, &ranges, &randomized);
    let current_authoritative = read_current(&reader, &ranges, &authoritative);
    let fresh_authoritative = read_direct(&artifact_root, &ranges, &authoritative, false);
    let reusable_authoritative = read_persistent_reusable(&artifact_root, &ranges, &authoritative);
    let mut store = build_store(&artifact_root, &ranges);
    let store_started = Instant::now();
    for &id in &authoritative {
        let lease = store
            .load(ExpertKey {
                layer_index: 47,
                expert_id: ExpertId(u32::try_from(id).expect("expert ID")),
            })
            .expect("expert store benchmark load");
        assert_eq!(
            lease.bytes().len(),
            usize::try_from(PAYLOAD_BYTES).expect("payload usize")
        );
    }
    let store_elapsed_nanos = store_started.elapsed().as_nanos();
    let store_json = expert_store_run_json(authoritative.len(), store_elapsed_nanos, &store);
    let output = [
        "{\"schema\":\"colibri-qwen3-moe-m5.3-01-storage-benchmark-v1\",\"schema_version\":1,\"artifact_root_sha256\":\"f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2\",\"payload_bytes\":",
        &PAYLOAD_BYTES.to_string(),
        ",\"warm_cache_state\":\"uncontrolled\",\"current_reader\":{\"repeated_same_expert\":",
        &current_run_json(3, current_repeated),
        ",\"randomized_experts\":",
        &current_run_json(randomized.len(), current_randomized),
        ",\"authoritative_layer47_miss_subset\":",
        &current_run_json(authoritative.len(), current_authoritative),
        "},\"isolated_same_range_access\":{\"authoritative_layer47_miss_subset\":{\"requests\":",
        &authoritative.len().to_string(),
        ",\"fresh_allocation_persistent_handle_elapsed_nanos\":",
        &fresh_authoritative.to_string(),
        ",\"reusable_buffer_persistent_handle_elapsed_nanos\":",
        &reusable_authoritative.to_string(),
        "}},\"expert_store_authoritative_layer47_miss_subset\":",
        &store_json,
        "}\n",
    ]
    .concat();
    std::fs::write(&output_path, output).expect("write benchmark evidence");
    println!(
        "m5_3_storage_bench_complete output={} requests={}",
        output_path.display(),
        authoritative.len()
    );
}
