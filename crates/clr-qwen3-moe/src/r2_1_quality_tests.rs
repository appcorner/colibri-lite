use std::{env, fmt::Write as _, fs, path::PathBuf};

use super::*;

const NATIVE_SCALAR_LAYER0_MAX_ABS: f32 = 0.001;
const NATIVE_SCALAR_PROMPT_LOGIT_MAX_ABS: f32 = 0.002;

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn m6_3_r2_1_native_group32_passes_held_out_quality() {
    assert!(
        crate::r2_native::avx2_fma_available(),
        "R2.1 host must expose AVX2+FMA"
    );
    let reference_path = PathBuf::from(
        env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R1.2 frozen reference path"),
    );
    let candidate_path = PathBuf::from(
        env::var_os("COLIBRI_R2_1_CANDIDATE_PATH").expect("R2.1 group32 candidate path"),
    );
    let output_path = PathBuf::from(
        env::var_os("COLIBRI_R2_1_QUALITY_OUTPUT").expect("R2.1 quality output path"),
    );
    assert!(!output_path.exists(), "R2.1 quality output must be new");
    assert_eq!(
        fs::metadata(&candidate_path)
            .expect("candidate metadata")
            .len(),
        R1_2_GROUP32_ARTIFACT_BYTES,
    );
    let references = r1_2_frozen_references(&reference_path);
    let mut reader = R1_1CandidateReader::open(
        &candidate_path,
        R1_1PackedArtifactLayout::canonical(32).expect("group32 layout"),
        R1_2_GROUP32_ARTIFACT_SHA256,
    )
    .expect("R2.1 admitted candidate");
    assert_eq!(
        reader.verification_bytes_read(),
        R1_2_GROUP32_ARTIFACT_BYTES
    );

    let mut evidence = String::from(
        "fixture\tgenerated_ids\tprompt_argmax\tprompt_top20_ids\tguard_layer0_ids\tguard_layer24_ids\tguard_layer47_ids\tf32_fixed_logit_max_abs\tf32_top20_logit_max_abs\tf32_allowed_logit_max_abs\tnative_f32_layer0_max_abs\tnative_scalar_layer0_max_abs\tnative_scalar_prompt_logit_max_abs\tnative_prompt_logits_sha256\tscalar_prompt_logits_sha256\trepeatability\n",
    );
    for fixture in tier_b_references()
        .into_iter()
        .filter(|fixture| fixture.name == "short_english" || fixture.name == "short_thai")
    {
        let frozen = &references[&fixture.name];
        let scalar =
            r1_2_fixture_run_with_mode(&fixture, Some(&mut reader), R1_2CandidateMode::Scalar);
        let first = r1_2_fixture_run_with_mode(
            &fixture,
            Some(&mut reader),
            R1_2CandidateMode::NativeAvx2Fma,
        );
        let second = r1_2_fixture_run_with_mode(
            &fixture,
            Some(&mut reader),
            R1_2CandidateMode::NativeAvx2Fma,
        );
        for run in [&scalar, &first, &second] {
            r1_2_assert_frozen_prompt_contract(&fixture, run);
            assert_eq!(run.generated_ids, frozen.generated_ids);
        }
        assert_eq!(first.generated_ids, second.generated_ids);
        assert_eq!(first.prompt_top20_ids, second.prompt_top20_ids);
        assert_eq!(first.prompt_argmax, second.prompt_argmax);
        assert_eq!(first.prompt_guard_ids, second.prompt_guard_ids);
        assert_eq!(first.prompt_logits_sha256, second.prompt_logits_sha256);
        assert_eq!(
            first.prompt_final_norm_sha256,
            second.prompt_final_norm_sha256
        );
        assert_eq!(
            first.native_scalar_layer0_max_abs_error.to_bits(),
            second.native_scalar_layer0_max_abs_error.to_bits(),
        );
        let native_scalar_prompt = max_abs(&first.prompt_logits, &scalar.prompt_logits);
        let native_scalar_prompt_second = max_abs(&second.prompt_logits, &scalar.prompt_logits);
        assert!(
            first.native_scalar_layer0_max_abs_error <= NATIVE_SCALAR_LAYER0_MAX_ABS,
            "{} native/scalar Layer-0 error {} exceeds {}",
            fixture.name,
            first.native_scalar_layer0_max_abs_error,
            NATIVE_SCALAR_LAYER0_MAX_ABS,
        );
        assert!(native_scalar_prompt <= NATIVE_SCALAR_PROMPT_LOGIT_MAX_ABS);
        assert!(native_scalar_prompt_second <= NATIVE_SCALAR_PROMPT_LOGIT_MAX_ABS);
        assert_eq!(
            native_scalar_prompt.to_bits(),
            native_scalar_prompt_second.to_bits()
        );

        let observed_f32_error = first
            .prompt_fixed_logit_error
            .max(first.prompt_top20_logit_error);
        let allowed_f32_error = R1_2_LOGIT_MAX_ABS_ENVELOPE.min(fixture.margin / 4.0);
        assert!(observed_f32_error <= allowed_f32_error);
        assert!(first.layer0_moe_max_abs_error.is_finite());
        assert!(first.layer0_moe_max_abs_error > 0.0);
        assert_eq!(first.prompt_top20_ids, frozen.prompt_top20_ids);
        assert_eq!(first.prompt_guard_ids, frozen.prompt_guard_ids);
        assert!(first.prompt_logits.iter().all(|value| value.is_finite()));
        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\texact",
            fixture.name,
            comma_separated(&first.generated_ids),
            first.prompt_argmax,
            comma_separated(&first.prompt_top20_ids),
            comma_separated(&first.prompt_guard_ids[&0]),
            comma_separated(&first.prompt_guard_ids[&24]),
            comma_separated(&first.prompt_guard_ids[&47]),
            first.prompt_fixed_logit_error,
            first.prompt_top20_logit_error,
            allowed_f32_error,
            first.layer0_moe_max_abs_error,
            first.native_scalar_layer0_max_abs_error,
            native_scalar_prompt,
            first.prompt_logits_sha256,
            scalar.prompt_logits_sha256,
        )
        .expect("write R2.1 quality evidence");
    }
    assert!(reader.payload_bytes_read() > 0);
    assert_eq!(reader.peak_packed_expert_bytes(), 5_308_416);
    fs::write(&output_path, evidence).expect("write R2.1 quality evidence");
}
