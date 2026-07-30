//! Controlled CPU-kernel and RAM-copy microbenchmark for M6.1-02.
//!
//! This is evidence tooling, not an inference kernel. It deliberately uses
//! safe Rust and the standard library only so the recorded baseline does not
//! imply a backend, SIMD policy, or optimization decision.

use std::{env, fs, hint::black_box, path::PathBuf, time::Instant};

const MATRIX_ROWS: usize = 1_024;
const MATRIX_COLUMNS: usize = 4_096;
const CPU_WARMUP_SAMPLES: usize = 3;
const CPU_SAMPLE_COUNT: usize = 9;
const CPU_INNER_REPETITIONS: usize = 16;
const CPU_OPERATIONS_PER_SAMPLE: f64 = 134_217_728.0;

const RAM_PAYLOAD_BYTES: usize = 128 * 1_024 * 1_024;
const RAM_WARMUP_SAMPLES: usize = 2;
const RAM_SAMPLE_COUNT: usize = 9;
const RAM_INNER_REPETITIONS: usize = 4;
const RAM_TRAFFIC_BYTES_PER_SAMPLE: f64 = 1_073_741_824.0;

#[derive(Debug)]
struct Arguments {
    output: PathBuf,
    profile_id: String,
    created_at: String,
    runtime_commit: String,
    os_name: String,
    os_version: String,
    cpu_model: String,
    logical_core_count: usize,
    ram_total_bytes: u64,
}

#[derive(Debug)]
struct Distribution {
    samples: Vec<f64>,
    minimum: f64,
    p10: f64,
    median: f64,
    p90: f64,
    maximum: f64,
}

fn main() -> Result<(), String> {
    let arguments = parse_arguments(env::args().skip(1))?;
    let (cpu_distribution, cpu_checksum) = measure_cpu_kernel();
    let (ram_distribution, ram_checksum) = measure_ram_copy();
    let json = render_json(
        &arguments,
        &cpu_distribution,
        cpu_checksum,
        &ram_distribution,
        ram_checksum,
    );

    if let Some(parent) = arguments.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    fs::write(&arguments.output, json)
        .map_err(|error| format!("failed to write {}: {error}", arguments.output.display()))?;

    println!("wrote {}", arguments.output.display());
    println!(
        "CPU median: {:.3} GFLOP/s; RAM median: {:.3} GiB/s",
        cpu_distribution.median, ram_distribution.median
    );
    Ok(())
}

fn parse_arguments(arguments: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let mut values = std::collections::BTreeMap::new();
    let mut arguments = arguments.peekable();
    while let Some(flag) = arguments.next() {
        if !flag.starts_with("--") {
            return Err(format!("unexpected argument `{flag}`"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for `{flag}`"))?;
        values.insert(flag, value);
    }

    let required = |flag: &str| {
        values
            .get(flag)
            .cloned()
            .ok_or_else(|| format!("missing required argument `{flag}`"))
    };
    let parse_usize = |flag: &str| {
        required(flag)?
            .parse::<usize>()
            .map_err(|error| format!("invalid unsigned integer for `{flag}`: {error}"))
    };
    let parse_u64 = |flag: &str| {
        required(flag)?
            .parse::<u64>()
            .map_err(|error| format!("invalid unsigned integer for `{flag}`: {error}"))
    };

    Ok(Arguments {
        output: PathBuf::from(required("--output")?),
        profile_id: required("--profile-id")?,
        created_at: required("--created-at")?,
        runtime_commit: required("--runtime-commit")?,
        os_name: required("--os-name")?,
        os_version: required("--os-version")?,
        cpu_model: required("--cpu-model")?,
        logical_core_count: parse_usize("--logical-core-count")?,
        ram_total_bytes: parse_u64("--ram-total-bytes")?,
    })
}

fn measure_cpu_kernel() -> (Distribution, f64) {
    let mut state = 0x6d2b_79f5_u32;
    let matrix = (0..MATRIX_ROWS * MATRIX_COLUMNS)
        .map(|_| next_f32(&mut state))
        .collect::<Vec<_>>();
    let vector = (0..MATRIX_COLUMNS)
        .map(|_| next_f32(&mut state))
        .collect::<Vec<_>>();
    let mut output = vec![0.0_f32; MATRIX_ROWS];

    for _ in 0..CPU_WARMUP_SAMPLES {
        black_box(run_matrix_vector(&matrix, &vector, &mut output));
    }

    let mut samples = Vec::with_capacity(CPU_SAMPLE_COUNT);
    let mut checksum = 0.0_f64;
    for _ in 0..CPU_SAMPLE_COUNT {
        let started = Instant::now();
        for _ in 0..CPU_INNER_REPETITIONS {
            checksum += run_matrix_vector(&matrix, &vector, &mut output);
        }
        let elapsed_seconds = started.elapsed().as_secs_f64();
        samples.push(CPU_OPERATIONS_PER_SAMPLE / elapsed_seconds / 1_000_000_000.0);
    }
    black_box(checksum);
    (distribution(samples), checksum)
}

fn run_matrix_vector(matrix: &[f32], vector: &[f32], output: &mut [f32]) -> f64 {
    let mut checksum = 0.0_f64;
    for (row, result) in output.iter_mut().enumerate() {
        let start = row * MATRIX_COLUMNS;
        let mut sum = 0.0_f32;
        for column in 0..MATRIX_COLUMNS {
            sum += matrix[start + column] * vector[column];
        }
        *result = sum;
        checksum += f64::from(sum);
    }
    checksum
}

fn measure_ram_copy() -> (Distribution, u64) {
    let source = (0..RAM_PAYLOAD_BYTES)
        .map(|index| {
            u8::try_from(index % 256)
                .expect("modulo result fits in u8")
                .wrapping_mul(31)
                .wrapping_add(17)
        })
        .collect::<Vec<_>>();
    let mut destination = vec![0_u8; RAM_PAYLOAD_BYTES];

    for _ in 0..RAM_WARMUP_SAMPLES {
        destination.copy_from_slice(&source);
        black_box(destination[RAM_PAYLOAD_BYTES / 2]);
    }

    let mut samples = Vec::with_capacity(RAM_SAMPLE_COUNT);
    let mut checksum = 0_u64;
    for sample_index in 0..RAM_SAMPLE_COUNT {
        let started = Instant::now();
        for _ in 0..RAM_INNER_REPETITIONS {
            destination.copy_from_slice(&source);
        }
        let elapsed_seconds = started.elapsed().as_secs_f64();
        checksum += u64::from(destination[(sample_index * 7_919) % RAM_PAYLOAD_BYTES]);
        samples.push(RAM_TRAFFIC_BYTES_PER_SAMPLE / elapsed_seconds / 1_073_741_824.0);
    }
    black_box(checksum);
    (distribution(samples), checksum)
}

fn next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let upper_bits = u16::try_from(*state >> 16).expect("upper 16 bits fit in u16");
    f32::from(upper_bits) / 65_536.0 - 0.5
}

fn distribution(mut samples: Vec<f64>) -> Distribution {
    samples.sort_by(f64::total_cmp);
    assert_eq!(samples.len(), 9, "M6.1-02 uses exactly nine samples");
    Distribution {
        minimum: samples[0],
        p10: samples[1],
        median: samples[4],
        p90: samples[7],
        maximum: samples[samples.len() - 1],
        samples,
    }
}

fn render_json(
    arguments: &Arguments,
    cpu: &Distribution,
    cpu_checksum: f64,
    ram: &Distribution,
    ram_checksum: u64,
) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"colibri-lite-m6.1-02-cpu-ram-benchmark-v1\",\n",
            "  \"schema_version\": 1,\n",
            "  \"profile_id\": \"{}\",\n",
            "  \"created_at\": \"{}\",\n",
            "  \"runtime\": {{\"name\": \"colibri-lite-rs\", \"commit\": \"{}\", \"target_arch\": \"{}\", \"build_profile\": \"release\"}},\n",
            "  \"host\": {{\"os_name\": \"{}\", \"os_version\": \"{}\", \"cpu_model\": \"{}\", \"logical_core_count\": {}, \"ram_total_bytes\": {}}},\n",
            "  \"measurement_semantics\": {{\n",
            "    \"clock\": \"std::time::Instant monotonic elapsed clock\",\n",
            "    \"sample_percentiles\": \"nearest sample after ascending sort; index=round((n-1)*p)\",\n",
            "    \"cpu_kernel\": {{\"kernel_id\": \"safe_f32_matrix_vector\", \"matrix_rows\": {}, \"matrix_columns\": {}, \"fma_equivalent_operations_per_element\": 2, \"warmup_samples\": {}, \"sample_count\": {}, \"inner_repetitions\": {}}},\n",
            "    \"ram_copy\": {{\"operation\": \"Vec<u8>::copy_from_slice\", \"payload_bytes\": {}, \"traffic_accounting\": \"one payload read plus one payload write per copy\", \"warmup_samples\": {}, \"sample_count\": {}, \"inner_repetitions\": {}}}\n",
            "  }},\n",
            "  \"results\": {{\n",
            "    \"cpu_kernel_gflops\": {},\n",
            "    \"ram_copy_gib_per_second\": {}\n",
            "  }},\n",
            "  \"checksums\": {{\"cpu_output_sum\": {:.9}, \"ram_probe_sum\": {}}},\n",
            "  \"limitations\": [\n",
            "    \"This is a controlled microbenchmark, not end-to-end inference throughput.\",\n",
            "    \"The benchmark does not select or require explicit SIMD, FFI, or a GPU backend.\",\n",
            "    \"RAM copy throughput is a streaming-copy proxy and is not a latency measurement.\",\n",
            "    \"The host was not isolated or clock-pinned; background activity can affect dispersion.\"\n",
            "  ]\n",
            "}}\n"
        ),
        json_escape(&arguments.profile_id),
        json_escape(&arguments.created_at),
        json_escape(&arguments.runtime_commit),
        env::consts::ARCH,
        json_escape(&arguments.os_name),
        json_escape(&arguments.os_version),
        json_escape(&arguments.cpu_model),
        arguments.logical_core_count,
        arguments.ram_total_bytes,
        MATRIX_ROWS,
        MATRIX_COLUMNS,
        CPU_WARMUP_SAMPLES,
        CPU_SAMPLE_COUNT,
        CPU_INNER_REPETITIONS,
        RAM_PAYLOAD_BYTES,
        RAM_WARMUP_SAMPLES,
        RAM_SAMPLE_COUNT,
        RAM_INNER_REPETITIONS,
        render_distribution(cpu),
        render_distribution(ram),
        cpu_checksum,
        ram_checksum,
    )
}

fn render_distribution(distribution: &Distribution) -> String {
    let samples = distribution
        .samples
        .iter()
        .map(|sample| format!("{sample:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{\"samples\": [{samples}], \"minimum\": {:.6}, \"p10\": {:.6}, \"median\": {:.6}, \"p90\": {:.6}, \"maximum\": {:.6}}}",
        distribution.minimum,
        distribution.p10,
        distribution.median,
        distribution.p90,
        distribution.maximum,
    )
}

fn json_escape(value: &str) -> String {
    value
        .chars()
        .flat_map(char::escape_default)
        .collect::<String>()
}
