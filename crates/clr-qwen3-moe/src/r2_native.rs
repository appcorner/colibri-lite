//! R2.1 isolated native packed-projection boundary.
#![allow(unsafe_code)]

use clr_core::RuntimeError;

#[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
unsafe extern "C" {
    fn clr_qwen3_moe_group32_avx2_fma(
        values: *const i8,
        scales: *const f32,
        input: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
    ) -> i32;
}

pub(crate) fn avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
pub(crate) fn apply_group32_avx2_fma(
    values: &[i8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
) -> Result<bool, RuntimeError> {
    validate(values, scales, input, output, rows, columns)?;
    if !avx2_fma_available() {
        return Ok(false);
    }

    #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
    {
        // SAFETY: validation above proves every slice covers the exact element count the C
        // kernel may access; Rust retains ownership for the entire call; pointers do not
        // escape; the runtime feature check proves AVX2+FMA support before execution.
        let status = unsafe {
            clr_qwen3_moe_group32_avx2_fma(
                values.as_ptr(),
                scales.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                rows,
                columns,
            )
        };
        if status == 0 {
            Ok(true)
        } else {
            Err(error("native kernel rejected validated arguments"))
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc")))]
    {
        Ok(false)
    }
}

fn validate(
    values: &[i8],
    scales: &[f32],
    input: &[f32],
    output: &[f32],
    rows: usize,
    columns: usize,
) -> Result<(), RuntimeError> {
    if rows == 0 || columns == 0 || columns % 32 != 0 {
        return Err(error("invalid native group32 projection layout"));
    }
    let value_count = rows
        .checked_mul(columns)
        .ok_or_else(|| error("native projection value length overflow"))?;
    let scale_count = rows
        .checked_mul(columns / 32)
        .ok_or_else(|| error("native projection scale length overflow"))?;
    if values.len() != value_count || scales.len() != scale_count {
        return Err(error("invalid native packed projection payload length"));
    }
    if input.len() != columns || output.len() != rows {
        return Err(error("invalid native projection input or output length"));
    }
    if values.contains(&i8::MIN)
        || input.iter().any(|value| !value.is_finite())
        || scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
    {
        return Err(error("invalid native projection numeric payload"));
    }
    Ok(())
}

fn error(reason: &'static str) -> RuntimeError {
    RuntimeError::BackendContractViolation {
        context: "M6.3-R2.1 native packed projection",
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_group32_avx2_fma, avx2_fma_available};

    fn scalar(values: &[i8], scales: &[f32], input: &[f32], rows: usize) -> Vec<f32> {
        let columns = input.len();
        let groups = columns / 32;
        (0..rows)
            .map(|row| {
                let mut sum = 0.0_f32;
                for column in 0..columns {
                    let scale = scales[row * groups + column / 32];
                    sum += input[column] * (f32::from(values[row * columns + column]) * scale);
                }
                sum
            })
            .collect()
    }

    #[test]
    fn native_matches_scalar_and_preserves_output_canaries() {
        if !avx2_fma_available() {
            return;
        }
        let rows = 3;
        let columns = 64;
        let values = (0..rows * columns)
            .map(|index| i8::try_from(index % 31).expect("i8") - 15)
            .collect::<Vec<_>>();
        let scales = vec![0.015_625_f32; rows * (columns / 32)];
        let input = vec![0.031_25_f32; columns];
        let expected = scalar(&values, &scales, &input, rows);
        let canary = f32::from_bits(0x7fc0_1234);
        let mut guarded = vec![canary; rows + 2];
        let used = apply_group32_avx2_fma(
            &values,
            &scales,
            &input,
            &mut guarded[1..=rows],
            rows,
            columns,
        )
        .expect("native call");
        assert!(used);
        assert_eq!(guarded[0].to_bits(), canary.to_bits());
        assert_eq!(guarded[rows + 1].to_bits(), canary.to_bits());
        for (actual, expected) in guarded[1..=rows].iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-4,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn validation_rejects_bad_lengths_before_native_call() {
        let values = vec![1_i8; 32];
        let scales = vec![0.25_f32];
        let input = vec![1.0_f32; 31];
        let mut output = vec![123.0_f32];
        let error = apply_group32_avx2_fma(&values, &scales, &input, &mut output, 1, 32)
            .expect_err("invalid input length");
        assert!(format!("{error}").contains("input or output length"));
        assert_eq!(output, [123.0]);
    }
}
