//! R1.1-only deterministic Layer-0 INT8 candidate converter.
//!
//! The output is deliberately a temporary characterization artifact.  It is
//! never a replacement for the frozen F32 artifact and does not execute a
//! held-out fixture.

use std::{
    env,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Seek, Write},
    path::PathBuf,
};

use clr_qwen3_moe::{PINNED_QWEN3_30B_A3B_CONFIG, PackedExpertLayout};
use clr_storage::Sha256Hasher;

const EXPERTS: usize = 128;
const GATE_UP_ROWS: usize = 768;
const HIDDEN: usize = 2048;

fn main() -> Result<(), String> {
    let mut args = env::args_os();
    let _program = args.next();
    let group_size = args
        .next()
        .ok_or("usage: m6_3_r1_1_convert <32|64> <source> <output.incomplete>")?
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|_| "group size must be 32 or 64")?;
    if !matches!(group_size, 32 | 64) {
        return Err("group size must be 32 or 64".into());
    }
    let source_path = PathBuf::from(
        args.next()
            .ok_or("usage: m6_3_r1_1_convert <32|64> <source> <output.incomplete>")?,
    );
    let output_path = PathBuf::from(
        args.next()
            .ok_or("usage: m6_3_r1_1_convert <32|64> <source> <output.incomplete>")?,
    );
    if args.next().is_some() {
        return Err("unexpected additional arguments".into());
    }
    if output_path.extension().and_then(|value| value.to_str()) != Some("incomplete") {
        return Err("output must be a sibling .incomplete path".into());
    }

    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .map_err(|error| error.to_string())?
        .runtime_config();
    let layout = PackedExpertLayout::for_config(config);
    let mut source =
        BufReader::with_capacity(1024 * 1024, File::open(source_path).map_err(io_error)?);
    let output_file = File::create(output_path).map_err(io_error)?;
    let mut output = BufWriter::with_capacity(1024 * 1024, output_file);
    let mut hasher = Sha256Hasher::new();

    for expert in 0..EXPERTS {
        convert_projection(
            &mut source,
            &mut output,
            &mut hasher,
            layout.gate_offset + expert * layout.total_byte_length,
            GATE_UP_ROWS,
            HIDDEN,
            group_size,
        )?;
        convert_projection(
            &mut source,
            &mut output,
            &mut hasher,
            layout.up_offset + expert * layout.total_byte_length,
            GATE_UP_ROWS,
            HIDDEN,
            group_size,
        )?;
        convert_projection(
            &mut source,
            &mut output,
            &mut hasher,
            layout.down_offset + expert * layout.total_byte_length,
            HIDDEN,
            GATE_UP_ROWS,
            group_size,
        )?;
    }
    output.flush().map_err(io_error)?;
    output.get_ref().sync_all().map_err(io_error)?;
    let expected = artifact_bytes(group_size);
    let actual = output.get_ref().metadata().map_err(io_error)?.len();
    if actual != expected {
        return Err(format!(
            "output length mismatch: expected {expected}, got {actual}"
        ));
    }
    println!(
        "group_size={group_size} output_bytes={actual} output_sha256={}",
        hex(hasher.finalize())
    );
    Ok(())
}

fn convert_projection(
    source: &mut BufReader<File>,
    output: &mut BufWriter<File>,
    hasher: &mut Sha256Hasher,
    offset: usize,
    rows: usize,
    columns: usize,
    group_size: usize,
) -> Result<(), String> {
    if columns % group_size != 0 {
        return Err("group size does not divide projection columns".into());
    }
    source
        .seek(std::io::SeekFrom::Start(
            u64::try_from(offset).map_err(|_| "source offset overflow")?,
        ))
        .map_err(io_error)?;
    let mut row_bytes = vec![0_u8; columns * size_of::<f32>()];
    let mut scales = Vec::with_capacity(rows * (columns / group_size));
    for _ in 0..rows {
        source.read_exact(&mut row_bytes).map_err(io_error)?;
        let mut values = Vec::with_capacity(columns);
        for bytes in row_bytes.chunks_exact(size_of::<f32>()) {
            let value = f32::from_le_bytes(bytes.try_into().map_err(|_| "invalid F32 row")?);
            if !value.is_finite() {
                return Err("source contains non-finite value".into());
            }
            values.push(value);
        }
        for group in values.chunks_exact(group_size) {
            let maximum = group
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            let scale = maximum / 127.0;
            scales.push(scale);
            for value in group {
                let quantized = if scale == 0.0 {
                    0
                } else {
                    quantize(*value, scale)
                };
                let byte = quantized.to_le_bytes();
                output.write_all(&byte).map_err(io_error)?;
                hasher.update(&byte);
            }
        }
    }
    for scale in scales {
        let bytes = scale.to_le_bytes();
        output.write_all(&bytes).map_err(io_error)?;
        hasher.update(&bytes);
    }
    Ok(())
}

fn artifact_bytes(group_size: usize) -> u64 {
    let projection = |rows: usize, columns: usize| {
        rows * columns + rows * (columns / group_size) * size_of::<f32>()
    };
    u64::try_from(
        EXPERTS * (projection(GATE_UP_ROWS, HIDDEN) * 2 + projection(HIDDEN, GATE_UP_ROWS)),
    )
    .expect("bounded artifact bytes")
}

#[allow(clippy::needless_pass_by_value)] // required by Result::map_err's by-value error.
fn io_error(error: io::Error) -> String {
    error.to_string()
}

#[allow(clippy::cast_possible_truncation)]
fn quantize(value: f32, scale: f32) -> i8 {
    // The explicit clamp establishes the i8 range before this intentional cast.
    (value / scale).round().clamp(-127.0, 127.0) as i8
}

fn hex(bytes: [u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
