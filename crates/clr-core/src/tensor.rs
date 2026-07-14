use crate::{RuntimeError, TensorShape};

/// Owned contiguous row-major tensor storage for the M1 `f32` correctness path.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: TensorShape,
    data: Vec<f32>,
}

impl Tensor {
    /// Creates an owned tensor after validating shape and storage length.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ArithmeticOverflow`] when the shape element count
    /// overflows, or [`RuntimeError::TensorDataLengthMismatch`] when `data` does
    /// not contain exactly one value per shape element.
    pub fn new(shape: TensorShape, data: Vec<f32>) -> Result<Self, RuntimeError> {
        validate_data_length(&shape, data.len())?;
        Ok(Self { shape, data })
    }

    /// Returns the tensor shape.
    #[must_use]
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns contiguous row-major values.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns mutable contiguous row-major values.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Returns a checked immutable tensor view.
    #[must_use]
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            shape: &self.shape,
            data: &self.data,
        }
    }

    /// Returns a checked mutable tensor view.
    #[must_use]
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut {
            shape: &self.shape,
            data: &mut self.data,
        }
    }

    /// Returns one value using contiguous row-major coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::RankMismatch`] when the coordinate count differs
    /// from the tensor rank, or [`RuntimeError::IndexOutOfBounds`] when a
    /// coordinate is outside its dimension.
    pub fn get(&self, coordinates: &[usize]) -> Result<f32, RuntimeError> {
        self.view().get(coordinates)
    }

    /// Consumes the tensor and returns its contiguous values.
    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}

/// Checked immutable view over contiguous row-major `f32` tensor data.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    shape: &'a TensorShape,
    data: &'a [f32],
}

impl<'a> TensorView<'a> {
    /// Creates a view after validating shape and storage length.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ArithmeticOverflow`] when the shape element count
    /// overflows, or [`RuntimeError::TensorDataLengthMismatch`] when `data` does
    /// not contain exactly one value per shape element.
    pub fn new(shape: &'a TensorShape, data: &'a [f32]) -> Result<Self, RuntimeError> {
        validate_data_length(shape, data.len())?;
        Ok(Self { shape, data })
    }

    /// Returns the view shape.
    #[must_use]
    pub const fn shape(self) -> &'a TensorShape {
        self.shape
    }

    /// Returns contiguous row-major values.
    #[must_use]
    pub const fn data(self) -> &'a [f32] {
        self.data
    }

    /// Returns one value using contiguous row-major coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::RankMismatch`] when the coordinate count differs
    /// from the view rank, or [`RuntimeError::IndexOutOfBounds`] when a
    /// coordinate is outside its dimension.
    pub fn get(self, coordinates: &[usize]) -> Result<f32, RuntimeError> {
        let offset = contiguous_offset(self.shape, coordinates)?;
        Ok(self.data[offset])
    }
}

/// Checked mutable view over contiguous row-major `f32` tensor data.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    shape: &'a TensorShape,
    data: &'a mut [f32],
}

impl<'a> TensorViewMut<'a> {
    /// Creates a mutable view after validating shape and storage length.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ArithmeticOverflow`] when the shape element count
    /// overflows, or [`RuntimeError::TensorDataLengthMismatch`] when `data` does
    /// not contain exactly one value per shape element.
    pub fn new(shape: &'a TensorShape, data: &'a mut [f32]) -> Result<Self, RuntimeError> {
        validate_data_length(shape, data.len())?;
        Ok(Self { shape, data })
    }

    /// Returns the view shape.
    #[must_use]
    pub const fn shape(&self) -> &'a TensorShape {
        self.shape
    }

    /// Returns contiguous row-major values.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Returns mutable contiguous row-major values.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }

    /// Returns one value using contiguous row-major coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::RankMismatch`] when the coordinate count differs
    /// from the view rank, or [`RuntimeError::IndexOutOfBounds`] when a
    /// coordinate is outside its dimension.
    pub fn get(&self, coordinates: &[usize]) -> Result<f32, RuntimeError> {
        let offset = contiguous_offset(self.shape, coordinates)?;
        Ok(self.data[offset])
    }

    /// Returns mutable access to one value using row-major coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::RankMismatch`] when the coordinate count differs
    /// from the view rank, or [`RuntimeError::IndexOutOfBounds`] when a
    /// coordinate is outside its dimension.
    pub fn get_mut(&mut self, coordinates: &[usize]) -> Result<&mut f32, RuntimeError> {
        let offset = contiguous_offset(self.shape, coordinates)?;
        Ok(&mut self.data[offset])
    }
}

fn validate_data_length(shape: &TensorShape, actual: usize) -> Result<(), RuntimeError> {
    let expected = shape.element_count()?;
    if expected != actual {
        return Err(RuntimeError::TensorDataLengthMismatch { expected, actual });
    }
    Ok(())
}

fn contiguous_offset(shape: &TensorShape, coordinates: &[usize]) -> Result<usize, RuntimeError> {
    if coordinates.len() != shape.rank() {
        return Err(RuntimeError::RankMismatch {
            context: "tensor index",
            expected: shape.rank(),
            actual: coordinates.len(),
        });
    }

    let mut offset = 0_usize;
    for (&coordinate, &dimension) in coordinates.iter().zip(shape.dimensions()) {
        if coordinate >= dimension {
            return Err(RuntimeError::IndexOutOfBounds {
                index: coordinate,
                length: dimension,
            });
        }
        offset = offset * dimension + coordinate;
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_tensor_validates_shape_and_length() {
        let tensor =
            Tensor::new(TensorShape::new([2, 3]), vec![0.0; 6]).expect("matching storage length");

        assert_eq!(tensor.shape().dimensions(), [2, 3]);
        assert_eq!(tensor.data().len(), 6);
        assert_eq!(
            Tensor::new(TensorShape::new([2, 3]), vec![0.0; 5]),
            Err(RuntimeError::TensorDataLengthMismatch {
                expected: 6,
                actual: 5,
            })
        );
    }

    #[test]
    fn immutable_view_validates_length_and_reads_row_major_values() {
        let shape = TensorShape::new([2, 3]);
        let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let view = TensorView::new(&shape, &values).expect("matching storage length");

        assert_eq!(view.shape(), &shape);
        assert_eq!(view.data(), values);
        assert_eq!(view.get(&[0, 2]), Ok(2.0));
        assert_eq!(view.get(&[1, 0]), Ok(3.0));
    }

    #[test]
    fn mutable_view_writes_row_major_values() {
        let shape = TensorShape::new([2, 2]);
        let mut values = [0.0; 4];
        let mut view = TensorViewMut::new(&shape, &mut values).expect("matching storage length");

        *view.get_mut(&[1, 0]).expect("valid coordinates") = 7.0;

        assert_eq!(view.get(&[1, 0]), Ok(7.0));
        assert_eq!(view.data(), [0.0, 0.0, 7.0, 0.0]);
    }

    #[test]
    fn indexing_rejects_wrong_rank_and_out_of_bounds_coordinates() {
        let tensor =
            Tensor::new(TensorShape::new([2, 3]), vec![0.0; 6]).expect("matching storage length");

        assert_eq!(
            tensor.get(&[1]),
            Err(RuntimeError::RankMismatch {
                context: "tensor index",
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            tensor.get(&[2, 0]),
            Err(RuntimeError::IndexOutOfBounds {
                index: 2,
                length: 2,
            })
        );
    }

    #[test]
    fn scalar_and_empty_tensor_views_are_valid() {
        let scalar = Tensor::new(TensorShape::scalar(), vec![4.0]).expect("scalar storage");
        let empty =
            Tensor::new(TensorShape::new([2, 0, 3]), Vec::new()).expect("empty tensor storage");

        assert_eq!(scalar.get(&[]), Ok(4.0));
        assert_eq!(empty.data(), []);
    }
}
