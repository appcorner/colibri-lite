use crate::{DataType, RuntimeError};

/// Owned dimensions for a contiguous dense tensor.
///
/// An empty dimension list (`[]`) represents a scalar with one element. A
/// shape containing any zero-sized dimension, such as `[2, 0, 3]`, represents
/// an empty tensor with zero elements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape {
    dimensions: Box<[usize]>,
}

impl TensorShape {
    /// Creates a shape from owned dimensions.
    #[must_use]
    pub fn new(dimensions: impl Into<Box<[usize]>>) -> Self {
        Self {
            dimensions: dimensions.into(),
        }
    }

    /// Creates the rank-zero scalar shape.
    #[must_use]
    pub fn scalar() -> Self {
        Self::new([])
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns all dimensions in axis order.
    #[must_use]
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns the size of one dimension.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::IndexOutOfBounds`] when `index` is not less
    /// than the shape rank.
    pub fn dimension(&self, index: usize) -> Result<usize, RuntimeError> {
        self.dimensions
            .get(index)
            .copied()
            .ok_or(RuntimeError::IndexOutOfBounds {
                index,
                length: self.rank(),
            })
    }

    /// Returns whether this is the rank-zero scalar shape.
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }

    /// Returns whether at least one dimension has size zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dimensions.contains(&0)
    }

    /// Calculates the number of tensor elements without wrapping.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ArithmeticOverflow`] when the product cannot be
    /// represented by [`usize`].
    pub fn element_count(&self) -> Result<usize, RuntimeError> {
        self.dimensions.iter().try_fold(1_usize, |count, size| {
            count
                .checked_mul(*size)
                .ok_or(RuntimeError::ArithmeticOverflow {
                    operation: "tensor element count",
                })
        })
    }

    /// Calculates the dense tensor byte count without wrapping.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ArithmeticOverflow`] when either the element
    /// count or byte count cannot be represented by [`usize`].
    pub fn byte_count(&self, data_type: DataType) -> Result<usize, RuntimeError> {
        self.element_count()?
            .checked_mul(data_type.byte_width())
            .ok_or(RuntimeError::ArithmeticOverflow {
                operation: "tensor byte count",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_has_rank_zero_and_one_element() {
        let shape = TensorShape::scalar();

        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.dimensions(), []);
        assert!(shape.is_scalar());
        assert!(!shape.is_empty());
        assert_eq!(shape.element_count(), Ok(1));
        assert_eq!(shape.byte_count(DataType::F32), Ok(4));
    }

    #[test]
    fn vector_and_matrix_report_dimensions_and_sizes() {
        let vector = TensorShape::new([7]);
        let matrix = TensorShape::new([2, 3]);

        assert_eq!(vector.rank(), 1);
        assert_eq!(vector.dimension(0), Ok(7));
        assert_eq!(vector.element_count(), Ok(7));
        assert_eq!(matrix.rank(), 2);
        assert_eq!(matrix.dimensions(), [2, 3]);
        assert_eq!(matrix.element_count(), Ok(6));
        assert_eq!(matrix.byte_count(DataType::BF16), Ok(12));
    }

    #[test]
    fn zero_sized_dimension_makes_tensor_empty() {
        let shape = TensorShape::new([2, 0, 3]);

        assert!(shape.is_empty());
        assert!(!shape.is_scalar());
        assert_eq!(shape.element_count(), Ok(0));
        assert_eq!(shape.byte_count(DataType::F32), Ok(0));
    }

    #[test]
    fn invalid_dimension_access_is_structured() {
        let shape = TensorShape::new([2, 3]);

        assert_eq!(
            shape.dimension(2),
            Err(RuntimeError::IndexOutOfBounds {
                index: 2,
                length: 2,
            })
        );
    }

    #[test]
    fn element_count_reports_overflow() {
        let shape = TensorShape::new([usize::MAX, 2]);

        assert_eq!(
            shape.element_count(),
            Err(RuntimeError::ArithmeticOverflow {
                operation: "tensor element count",
            })
        );
    }

    #[test]
    fn byte_count_reports_overflow() {
        let shape = TensorShape::new([usize::MAX]);

        assert_eq!(
            shape.byte_count(DataType::F32),
            Err(RuntimeError::ArithmeticOverflow {
                operation: "tensor byte count",
            })
        );
    }
}
