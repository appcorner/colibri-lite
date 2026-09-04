use std::{env, fmt::Write as _, fs, path::PathBuf};

use crate::r1_1_direct_candidate::{R1_1PackedArtifactLayout, R2_2LayerCandidateReader};

const PACKED_LAYER_BYTES: u64 = 679_477_248;
const PACKED_EXPERT_BYTES: usize = 5_308_416;
const ARTIFACTS: [(usize, &str, &str); 3] = [
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

#[test]
fn m6_3_r2_2_sentinel_artifacts_match_frozen_identities() {
    let root =
        PathBuf::from(env::var_os("COLIBRI_R2_2_ARTIFACT_ROOT").expect("R2.2 artifact root"));
    let output_path = PathBuf::from(
        env::var_os("COLIBRI_R2_2_ARTIFACT_EVIDENCE").expect("R2.2 artifact evidence path"),
    );
    assert!(!output_path.exists(), "R2.2 artifact evidence must be new");
    let layout = R1_1PackedArtifactLayout::canonical(32).expect("group32 layout");
    assert_eq!(
        layout.artifact_bytes().expect("artifact bytes"),
        PACKED_LAYER_BYTES
    );

    let mut evidence = String::from(
        "layer\tartifact\tsha256\tbytes\tverification_bytes\tpayload_bytes\tpeak_expert_bytes\n",
    );
    for (layer, name, sha256) in ARTIFACTS {
        let path = root.join(name);
        let mut bound = R2_2LayerCandidateReader::open(layer, &path, layout, sha256)
            .expect("open frozen layer artifact");
        assert_eq!(bound.layer(), layer);
        let reader = bound
            .reader_mut_for_layer(layer)
            .expect("matching frozen layer identity");
        assert_eq!(reader.sha256(), sha256);
        assert_eq!(reader.verification_bytes_read(), PACKED_LAYER_BYTES);
        reader.load_expert(0).expect("first expert");
        reader.load_expert(127).expect("last expert");
        assert_eq!(
            reader.payload_bytes_read(),
            (PACKED_EXPERT_BYTES * 2) as u64
        );
        assert_eq!(reader.peak_packed_expert_bytes(), PACKED_EXPERT_BYTES);
        writeln!(
            evidence,
            "{layer}\t{}\t{sha256}\t{PACKED_LAYER_BYTES}\t{}\t{}\t{}",
            path.display(),
            reader.verification_bytes_read(),
            reader.payload_bytes_read(),
            reader.peak_packed_expert_bytes(),
        )
        .expect("write artifact evidence");
    }
    fs::write(output_path, evidence).expect("write R2.2 artifact evidence");
}
