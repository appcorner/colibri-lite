use std::{fs, hint::black_box, time::Instant};

use clr_core::{DataType, TensorShape};
use clr_storage::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ArtifactReader, ByteOrder, TensorLocation,
    TensorMetadata, sha256_digest,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const PAYLOAD_BYTES: usize = 1024 * 1024;
    const ITERATIONS: usize = 200;

    let root = std::env::temp_dir().join(format!("colibri-portable-bench-{}", std::process::id()));
    fs::create_dir(&root)?;
    let payload: Vec<u8> = (0..PAYLOAD_BYTES)
        .map(|index| u8::try_from(index % 251).expect("value below 251"))
        .collect();
    fs::write(root.join("payload.bin"), &payload)?;
    let manifest = ArtifactManifest::new(
        ARTIFACT_FORMAT_VERSION,
        ByteOrder::Little,
        vec![TensorMetadata {
            name: "benchmark.payload".to_owned(),
            shape: TensorShape::new([PAYLOAD_BYTES / 4]),
            data_type: DataType::F32,
            location: TensorLocation {
                path: "payload.bin".into(),
                offset: 0,
                length: u64::try_from(PAYLOAD_BYTES)?,
            },
            sha256: sha256_digest(&payload),
        }],
    )?;
    let reader = ArtifactReader::open(&root, manifest)?;

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(reader.read_tensor("benchmark.payload")?);
    }
    let elapsed = started.elapsed();
    let total_bytes = PAYLOAD_BYTES * ITERATIONS;
    #[allow(clippy::cast_precision_loss)]
    let mib_per_second = total_bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64();
    println!(
        "{{\"method\":\"portable_open_seek_read_exact_sha256\",\"payload_bytes\":{PAYLOAD_BYTES},\"iterations\":{ITERATIONS},\"total_bytes\":{total_bytes},\"elapsed_seconds\":{:.6},\"mib_per_second\":{mib_per_second:.3}}}",
        elapsed.as_secs_f64()
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
