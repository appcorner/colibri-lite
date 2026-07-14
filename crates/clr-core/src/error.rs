use std::fmt;

/// Errors produced by runtime value-contract validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// A tensor shape does not satisfy a required invariant.
    InvalidShape {
        /// Human-readable explanation of the violated invariant.
        reason: &'static str,
    },
    /// Checked arithmetic could not represent a derived size.
    ArithmeticOverflow {
        /// Name of the size calculation that overflowed.
        operation: &'static str,
    },
    /// An index is outside the valid range for a value contract.
    IndexOutOfBounds {
        /// Requested zero-based index.
        index: usize,
        /// Exclusive upper bound for the index.
        length: usize,
    },
    /// Tensor storage length does not match its shape-derived element count.
    TensorDataLengthMismatch {
        /// Element count required by the shape.
        expected: usize,
        /// Element count supplied by storage.
        actual: usize,
    },
    /// An operation received a tensor with the wrong rank.
    RankMismatch {
        /// Operation or input whose rank was checked.
        context: &'static str,
        /// Required rank.
        expected: usize,
        /// Supplied rank.
        actual: usize,
    },
    /// Two tensor shapes are incompatible for an operation.
    ShapeMismatch {
        /// Operation whose inputs were checked.
        operation: &'static str,
        /// Expected dimensions.
        expected: Box<[usize]>,
        /// Supplied dimensions.
        actual: Box<[usize]>,
    },
    /// A numerical operation received a NaN or infinity.
    NonFiniteInput {
        /// Operation validating the input.
        operation: &'static str,
        /// Flat element index containing the invalid value.
        index: usize,
    },
    /// A model configuration field or relationship is invalid.
    InvalidModelConfig {
        /// Field primarily responsible for the validation failure.
        field: &'static str,
        /// Human-readable explanation of the violated invariant.
        reason: &'static str,
    },
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { reason } => write!(formatter, "invalid tensor shape: {reason}"),
            Self::ArithmeticOverflow { operation } => {
                write!(
                    formatter,
                    "arithmetic overflow while calculating {operation}"
                )
            }
            Self::IndexOutOfBounds { index, length } => {
                write!(
                    formatter,
                    "index {index} is out of bounds for length {length}"
                )
            }
            Self::TensorDataLengthMismatch { expected, actual } => write!(
                formatter,
                "tensor data length mismatch: expected {expected} elements, got {actual}"
            ),
            Self::RankMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "rank mismatch for {context}: expected rank {expected}, got {actual}"
            ),
            Self::ShapeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "shape mismatch for {operation}: expected {expected:?}, got {actual:?}"
            ),
            Self::NonFiniteInput { operation, index } => write!(
                formatter,
                "non-finite input for {operation} at flat index {index}"
            ),
            Self::InvalidModelConfig { field, reason } => {
                write!(
                    formatter,
                    "invalid model configuration field '{field}': {reason}"
                )
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_categories_are_matchable() {
        let error = RuntimeError::InvalidModelConfig {
            field: "hidden_size",
            reason: "must be greater than zero",
        };

        assert!(matches!(
            error,
            RuntimeError::InvalidModelConfig {
                field: "hidden_size",
                ..
            }
        ));
    }

    #[test]
    fn display_messages_include_actionable_context() {
        let cases = [
            (
                RuntimeError::InvalidShape {
                    reason: "rank is unsupported",
                },
                "invalid tensor shape: rank is unsupported",
            ),
            (
                RuntimeError::ArithmeticOverflow {
                    operation: "tensor byte count",
                },
                "arithmetic overflow while calculating tensor byte count",
            ),
            (
                RuntimeError::IndexOutOfBounds {
                    index: 3,
                    length: 2,
                },
                "index 3 is out of bounds for length 2",
            ),
            (
                RuntimeError::InvalidModelConfig {
                    field: "layer_count",
                    reason: "must be greater than zero",
                },
                "invalid model configuration field 'layer_count': must be greater than zero",
            ),
            (
                RuntimeError::TensorDataLengthMismatch {
                    expected: 6,
                    actual: 5,
                },
                "tensor data length mismatch: expected 6 elements, got 5",
            ),
            (
                RuntimeError::RankMismatch {
                    context: "matrix",
                    expected: 2,
                    actual: 1,
                },
                "rank mismatch for matrix: expected rank 2, got 1",
            ),
            (
                RuntimeError::ShapeMismatch {
                    operation: "elementwise add",
                    expected: [2, 3].into(),
                    actual: [3, 2].into(),
                },
                "shape mismatch for elementwise add: expected [2, 3], got [3, 2]",
            ),
            (
                RuntimeError::NonFiniteInput {
                    operation: "softmax",
                    index: 2,
                },
                "non-finite input for softmax at flat index 2",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }
}
