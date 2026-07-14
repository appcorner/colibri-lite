//! Readable scalar `f32` operations required by the frozen M1 fixture.
//!
//! These kernels prioritize checked shapes and independently testable numerical
//! behavior. They are not performance implementations.

use crate::{RuntimeError, Tensor, TensorShape, TensorView};

/// Adds two equal-shaped tensors element by element.
///
/// # Errors
///
/// Returns [`RuntimeError::ShapeMismatch`] when the input shapes differ.
pub fn elementwise_add(
    left: TensorView<'_>,
    right: TensorView<'_>,
) -> Result<Tensor, RuntimeError> {
    require_same_shape(left, right, "elementwise add")?;
    let data = left
        .data()
        .iter()
        .zip(right.data())
        .map(|(left_value, right_value)| left_value + right_value)
        .collect();
    Tensor::new(left.shape().clone(), data)
}

/// Multiplies two equal-shaped tensors element by element.
///
/// # Errors
///
/// Returns [`RuntimeError::ShapeMismatch`] when the input shapes differ.
pub fn elementwise_multiply(
    left: TensorView<'_>,
    right: TensorView<'_>,
) -> Result<Tensor, RuntimeError> {
    require_same_shape(left, right, "elementwise multiply")?;
    let data = left
        .data()
        .iter()
        .zip(right.data())
        .map(|(left_value, right_value)| left_value * right_value)
        .collect();
    Tensor::new(left.shape().clone(), data)
}

/// Sums all elements using scalar `f32` accumulation.
#[must_use]
pub fn sum(input: TensorView<'_>) -> f32 {
    input.data().iter().sum()
}

/// Calculates the arithmetic mean of all elements.
///
/// # Errors
///
/// Returns [`RuntimeError::InvalidShape`] for an empty tensor.
pub fn mean(input: TensorView<'_>) -> Result<f32, RuntimeError> {
    if input.data().is_empty() {
        return Err(RuntimeError::InvalidShape {
            reason: "mean requires at least one element",
        });
    }
    // A scalar f32 reduction necessarily represents its divisor as f32.
    #[allow(clippy::cast_precision_loss)]
    let divisor = input.data().len() as f32;
    Ok(sum(input) / divisor)
}

/// Multiplies a rank-two matrix by a rank-one vector.
///
/// # Errors
///
/// Returns [`RuntimeError::RankMismatch`] for non-matrix/non-vector inputs, or
/// [`RuntimeError::ShapeMismatch`] when their inner dimensions differ.
pub fn matrix_vector_multiply(
    matrix: TensorView<'_>,
    vector: TensorView<'_>,
) -> Result<Tensor, RuntimeError> {
    require_rank(matrix, 2, "matrix-vector matrix")?;
    require_rank(vector, 1, "matrix-vector vector")?;
    let row_count = matrix.shape().dimensions()[0];
    let inner_size = matrix.shape().dimensions()[1];
    if vector.shape().dimensions()[0] != inner_size {
        return Err(RuntimeError::ShapeMismatch {
            operation: "matrix-vector multiply",
            expected: [inner_size].into(),
            actual: vector.shape().dimensions().into(),
        });
    }

    let mut output = vec![0.0; row_count];
    for (row_index, output_value) in output.iter_mut().enumerate() {
        let row_start = row_index * inner_size;
        let matrix_row = &matrix.data()[row_start..row_start + inner_size];
        *output_value = matrix_row
            .iter()
            .zip(vector.data())
            .map(|(matrix_value, vector_value)| matrix_value * vector_value)
            .sum();
    }
    Tensor::new(TensorShape::new([row_count]), output)
}

/// Multiplies two rank-two matrices using scalar row-major loops.
///
/// # Errors
///
/// Returns [`RuntimeError::RankMismatch`] for non-matrix inputs, or
/// [`RuntimeError::ShapeMismatch`] when their inner dimensions differ.
pub fn matrix_multiply(
    left: TensorView<'_>,
    right: TensorView<'_>,
) -> Result<Tensor, RuntimeError> {
    require_rank(left, 2, "matrix-multiply left")?;
    require_rank(right, 2, "matrix-multiply right")?;
    let row_count = left.shape().dimensions()[0];
    let inner_size = left.shape().dimensions()[1];
    let right_inner_size = right.shape().dimensions()[0];
    let column_count = right.shape().dimensions()[1];
    if right_inner_size != inner_size {
        return Err(RuntimeError::ShapeMismatch {
            operation: "matrix multiply",
            expected: [inner_size, column_count].into(),
            actual: right.shape().dimensions().into(),
        });
    }

    let mut output = vec![0.0; row_count * column_count];
    for row_index in 0..row_count {
        for column_index in 0..column_count {
            let mut value = 0.0;
            for inner_index in 0..inner_size {
                value += left.data()[row_index * inner_size + inner_index]
                    * right.data()[inner_index * column_count + column_index];
            }
            output[row_index * column_count + column_index] = value;
        }
    }
    Tensor::new(TensorShape::new([row_count, column_count]), output)
}

/// Applies numerically stable softmax over the final dimension.
///
/// A scalar produces one. A tensor with a zero-sized final dimension remains
/// empty.
///
/// # Errors
///
/// Returns [`RuntimeError::NonFiniteInput`] for NaN or infinite input.
pub fn softmax_last_dim(input: TensorView<'_>) -> Result<Tensor, RuntimeError> {
    require_finite(input, "softmax")?;
    let final_dimension = input.shape().dimensions().last().copied().unwrap_or(1);
    if final_dimension == 0 {
        return Tensor::new(input.shape().clone(), Vec::new());
    }

    let mut output = vec![0.0; input.data().len()];
    for (input_row, output_row) in input
        .data()
        .chunks_exact(final_dimension)
        .zip(output.chunks_exact_mut(final_dimension))
    {
        let maximum = input_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;
        for (output_value, input_value) in output_row.iter_mut().zip(input_row) {
            *output_value = (*input_value - maximum).exp();
            denominator += *output_value;
        }
        for output_value in output_row {
            *output_value /= denominator;
        }
    }
    Tensor::new(input.shape().clone(), output)
}

/// Applies the `SiLU` activation `x / (1 + exp(-x))` element by element.
///
/// # Errors
///
/// Returns [`RuntimeError::NonFiniteInput`] for NaN or infinite input.
pub fn silu(input: TensorView<'_>) -> Result<Tensor, RuntimeError> {
    require_finite(input, "silu")?;
    let output = input
        .data()
        .iter()
        .map(|value| value / (1.0 + (-value).exp()))
        .collect();
    Tensor::new(input.shape().clone(), output)
}

fn require_same_shape(
    left: TensorView<'_>,
    right: TensorView<'_>,
    operation: &'static str,
) -> Result<(), RuntimeError> {
    if left.shape() != right.shape() {
        return Err(RuntimeError::ShapeMismatch {
            operation,
            expected: left.shape().dimensions().into(),
            actual: right.shape().dimensions().into(),
        });
    }
    Ok(())
}

fn require_rank(
    input: TensorView<'_>,
    expected: usize,
    context: &'static str,
) -> Result<(), RuntimeError> {
    let actual = input.shape().rank();
    if actual != expected {
        return Err(RuntimeError::RankMismatch {
            context,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_finite(input: TensorView<'_>, operation: &'static str) -> Result<(), RuntimeError> {
    if let Some(index) = input.data().iter().position(|value| !value.is_finite()) {
        return Err(RuntimeError::NonFiniteInput { operation, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: impl Into<Box<[usize]>>, data: &[f32]) -> Tensor {
        Tensor::new(TensorShape::new(shape), data.to_vec()).expect("valid test tensor")
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "index {index}: expected {expected_value}, got {actual_value}"
            );
        }
    }

    #[test]
    fn elementwise_operations_match_hand_calculated_values() {
        let left = tensor([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let right = tensor([2, 2], &[5.0, 6.0, 7.0, 8.0]);

        let added = elementwise_add(left.view(), right.view()).expect("equal shapes");
        let multiplied = elementwise_multiply(left.view(), right.view()).expect("equal shapes");

        assert_eq!(added.data(), [6.0, 8.0, 10.0, 12.0]);
        assert_eq!(multiplied.data(), [5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn elementwise_operations_reject_shape_mismatch() {
        let left = tensor([2, 2], &[1.0; 4]);
        let right = tensor([4], &[1.0; 4]);

        assert_eq!(
            elementwise_add(left.view(), right.view()),
            Err(RuntimeError::ShapeMismatch {
                operation: "elementwise add",
                expected: [2, 2].into(),
                actual: [4].into(),
            })
        );
    }

    #[test]
    fn reductions_match_hand_calculated_values() {
        let input = tensor([4], &[1.0, 2.0, 3.0, 4.0]);
        let empty = tensor([0], &[]);

        assert_close(&[sum(input.view())], &[10.0], f32::EPSILON);
        assert_close(
            &[mean(input.view()).expect("non-empty mean")],
            &[2.5],
            f32::EPSILON,
        );
        assert_eq!(
            mean(empty.view()),
            Err(RuntimeError::InvalidShape {
                reason: "mean requires at least one element",
            })
        );
    }

    #[test]
    fn matrix_vector_multiply_matches_hand_calculated_values() {
        let matrix = tensor([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let vector = tensor([3], &[1.0, 2.0, 3.0]);

        let output =
            matrix_vector_multiply(matrix.view(), vector.view()).expect("compatible shapes");

        assert_eq!(output.shape().dimensions(), [2]);
        assert_eq!(output.data(), [14.0, 32.0]);
    }

    #[test]
    fn matrix_multiply_matches_hand_calculated_values() {
        let left = tensor([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = tensor([3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let output = matrix_multiply(left.view(), right.view()).expect("compatible shapes");

        assert_eq!(output.shape().dimensions(), [2, 2]);
        assert_eq!(output.data(), [22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn matrix_operations_reject_rank_and_inner_dimension_mismatch() {
        let vector = tensor([3], &[1.0; 3]);
        let wrong_matrix = tensor([2, 2], &[1.0; 4]);

        assert_eq!(
            matrix_vector_multiply(vector.view(), vector.view()),
            Err(RuntimeError::RankMismatch {
                context: "matrix-vector matrix",
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            matrix_vector_multiply(wrong_matrix.view(), vector.view()),
            Err(RuntimeError::ShapeMismatch {
                operation: "matrix-vector multiply",
                expected: [2].into(),
                actual: [3].into(),
            })
        );
    }

    #[test]
    fn softmax_is_stable_per_final_dimension() {
        let input = tensor([2, 3], &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);

        let output = softmax_last_dim(input.view()).expect("finite input");

        assert_close(
            output.data(),
            &[
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                0.090_030_57,
                0.244_728_48,
                0.665_240_94,
            ],
            1.0e-6,
        );
    }

    #[test]
    fn softmax_supports_scalar_and_empty_final_dimension() {
        let scalar = tensor([], &[9.0]);
        let empty = tensor([2, 0], &[]);

        assert_eq!(
            softmax_last_dim(scalar.view())
                .expect("scalar softmax")
                .data(),
            [1.0]
        );
        assert!(
            softmax_last_dim(empty.view())
                .expect("empty softmax")
                .data()
                .is_empty()
        );
    }

    #[test]
    fn silu_matches_independent_values() {
        let input = tensor([3], &[-1.0, 0.0, 1.0]);

        let output = silu(input.view()).expect("finite input");

        assert_close(output.data(), &[-0.268_941_43, 0.0, 0.731_058_6], 1.0e-6);
    }

    #[test]
    fn nonlinear_operations_reject_non_finite_input() {
        let with_nan = tensor([3], &[0.0, f32::NAN, 1.0]);
        let with_infinity = tensor([2], &[0.0, f32::INFINITY]);

        assert_eq!(
            softmax_last_dim(with_nan.view()),
            Err(RuntimeError::NonFiniteInput {
                operation: "softmax",
                index: 1,
            })
        );
        assert_eq!(
            silu(with_infinity.view()),
            Err(RuntimeError::NonFiniteInput {
                operation: "silu",
                index: 1,
            })
        );
    }
}
