//! Backend-neutral operation and execution metadata.
//!
//! These contracts describe work after tensor/artifact loading and before an
//! implementation-specific backend executes it. They deliberately contain no
//! file handles, model architecture fields, device API handles, or cache policy.

use std::time::Duration;

use crate::{DataType, RuntimeError, TensorShape};

/// Validated metadata for one contiguous dense tensor boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    shape: TensorShape,
    data_type: DataType,
}

impl TensorDescriptor {
    /// Creates generic tensor metadata without allocating or loading payload.
    #[must_use]
    pub const fn new(shape: TensorShape, data_type: DataType) -> Self {
        Self { shape, data_type }
    }

    /// Returns the tensor shape.
    #[must_use]
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the tensor metadata element type.
    #[must_use]
    pub const fn data_type(&self) -> DataType {
        self.data_type
    }
}

/// Generic tensor operation supported by the first backend contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    /// Add two equal-shaped dense tensors.
    ElementwiseAdd,
    /// Multiply two equal-shaped dense tensors.
    ElementwiseMultiply,
    /// Multiply a rank-two matrix by a rank-one vector.
    MatrixVectorMultiply,
    /// Multiply two rank-two matrices.
    MatrixMultiply,
    /// Apply softmax across the final dimension.
    SoftmaxLastDimension,
    /// Apply the `SiLU` activation element by element.
    SiLU,
}

/// Validated generic operation signature for a backend execution request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationDescriptor {
    kind: OperationKind,
    inputs: Box<[TensorDescriptor]>,
    output: TensorDescriptor,
}

impl OperationDescriptor {
    /// Creates an operation signature and validates its generic tensor rules.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] for invalid input
    /// counts, ranks, shapes, or dtypes.
    pub fn new(
        kind: OperationKind,
        inputs: impl Into<Box<[TensorDescriptor]>>,
        output: TensorDescriptor,
    ) -> Result<Self, RuntimeError> {
        let inputs = inputs.into();
        validate_operation(kind, &inputs, &output)?;
        Ok(Self {
            kind,
            inputs,
            output,
        })
    }

    /// Returns the operation kind.
    #[must_use]
    pub const fn kind(&self) -> OperationKind {
        self.kind
    }

    /// Returns input tensor metadata in declared order.
    #[must_use]
    pub const fn inputs(&self) -> &[TensorDescriptor] {
        &self.inputs
    }

    /// Returns output tensor metadata.
    #[must_use]
    pub const fn output(&self) -> &TensorDescriptor {
        &self.output
    }
}

/// Explicit memory limits admitted for one backend operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionBudget {
    host_memory_limit_bytes: usize,
    device_memory_limit_bytes: usize,
}

impl ExecutionBudget {
    /// Creates a budget. A zero limit explicitly forbids using that memory tier.
    #[must_use]
    pub const fn new(host_memory_limit_bytes: usize, device_memory_limit_bytes: usize) -> Self {
        Self {
            host_memory_limit_bytes,
            device_memory_limit_bytes,
        }
    }

    /// Returns the host-memory limit in bytes.
    #[must_use]
    pub const fn host_memory_limit_bytes(self) -> usize {
        self.host_memory_limit_bytes
    }

    /// Returns the device-memory limit in bytes.
    #[must_use]
    pub const fn device_memory_limit_bytes(self) -> usize {
        self.device_memory_limit_bytes
    }
}

/// Backend-agnostic execution request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRequest {
    operation: OperationDescriptor,
    budget: ExecutionBudget,
}

impl ExecutionRequest {
    /// Creates a request from a validated operation and explicit budget.
    #[must_use]
    pub const fn new(operation: OperationDescriptor, budget: ExecutionBudget) -> Self {
        Self { operation, budget }
    }

    /// Returns the operation to execute.
    #[must_use]
    pub const fn operation(&self) -> &OperationDescriptor {
        &self.operation
    }

    /// Returns the budget admitted for this operation.
    #[must_use]
    pub const fn budget(&self) -> ExecutionBudget {
        self.budget
    }
}

/// Backend-observed resource usage for one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionMetrics {
    elapsed: Duration,
    host_memory_bytes: usize,
    device_memory_bytes: usize,
}

impl ExecutionMetrics {
    /// Creates metrics reported by a backend after execution.
    #[must_use]
    pub const fn new(
        elapsed: Duration,
        host_memory_bytes: usize,
        device_memory_bytes: usize,
    ) -> Self {
        Self {
            elapsed,
            host_memory_bytes,
            device_memory_bytes,
        }
    }

    /// Returns elapsed operation time.
    #[must_use]
    pub const fn elapsed(self) -> Duration {
        self.elapsed
    }

    /// Returns host memory attributed to this operation.
    #[must_use]
    pub const fn host_memory_bytes(self) -> usize {
        self.host_memory_bytes
    }

    /// Returns device memory attributed to this operation.
    #[must_use]
    pub const fn device_memory_bytes(self) -> usize {
        self.device_memory_bytes
    }
}

/// Validated metadata returned by a backend after one operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    output: TensorDescriptor,
    metrics: ExecutionMetrics,
}

impl ExecutionResult {
    /// Validates a backend-reported output and resource usage against a request.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] when output metadata
    /// differs from the requested operation or a memory limit is exceeded.
    pub fn new(
        request: &ExecutionRequest,
        output: TensorDescriptor,
        metrics: ExecutionMetrics,
    ) -> Result<Self, RuntimeError> {
        if output != *request.operation().output() {
            return Err(contract_error(
                "execution result",
                "output metadata differs from requested operation",
            ));
        }
        if metrics.host_memory_bytes() > request.budget().host_memory_limit_bytes() {
            return Err(contract_error(
                "execution result",
                "host memory exceeds admitted budget",
            ));
        }
        if metrics.device_memory_bytes() > request.budget().device_memory_limit_bytes() {
            return Err(contract_error(
                "execution result",
                "device memory exceeds admitted budget",
            ));
        }
        Ok(Self { output, metrics })
    }

    /// Returns backend-reported output metadata.
    #[must_use]
    pub const fn output(&self) -> &TensorDescriptor {
        &self.output
    }

    /// Returns backend-observed execution metrics.
    #[must_use]
    pub const fn metrics(&self) -> ExecutionMetrics {
        self.metrics
    }
}

fn validate_operation(
    kind: OperationKind,
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), RuntimeError> {
    match kind {
        OperationKind::ElementwiseAdd | OperationKind::ElementwiseMultiply => {
            validate_same_shape_binary(inputs, output)
        }
        OperationKind::MatrixVectorMultiply => validate_matrix_vector(inputs, output),
        OperationKind::MatrixMultiply => validate_matrix_multiply(inputs, output),
        OperationKind::SoftmaxLastDimension | OperationKind::SiLU => {
            validate_same_shape_unary(inputs, output)
        }
    }
}

fn validate_same_shape_binary(
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), RuntimeError> {
    require_input_count(inputs, 2)?;
    require_same_dtype(inputs, output)?;
    if inputs[0].shape() != inputs[1].shape() || inputs[0].shape() != output.shape() {
        return Err(contract_error(
            "operation descriptor",
            "binary tensor shapes must match output shape",
        ));
    }
    Ok(())
}

fn validate_same_shape_unary(
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), RuntimeError> {
    require_input_count(inputs, 1)?;
    require_same_dtype(inputs, output)?;
    if inputs[0].shape() != output.shape() {
        return Err(contract_error(
            "operation descriptor",
            "unary tensor shape must match output shape",
        ));
    }
    Ok(())
}

fn validate_matrix_vector(
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), RuntimeError> {
    require_input_count(inputs, 2)?;
    require_same_dtype(inputs, output)?;
    require_rank(inputs[0].shape(), 2)?;
    require_rank(inputs[1].shape(), 1)?;
    require_rank(output.shape(), 1)?;
    if inputs[0].shape().dimension(1)? != inputs[1].shape().dimension(0)? {
        return Err(contract_error(
            "operation descriptor",
            "matrix width must equal vector length",
        ));
    }
    if inputs[0].shape().dimension(0)? != output.shape().dimension(0)? {
        return Err(contract_error(
            "operation descriptor",
            "output length must equal matrix row count",
        ));
    }
    Ok(())
}

fn validate_matrix_multiply(
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), RuntimeError> {
    require_input_count(inputs, 2)?;
    require_same_dtype(inputs, output)?;
    require_rank(inputs[0].shape(), 2)?;
    require_rank(inputs[1].shape(), 2)?;
    require_rank(output.shape(), 2)?;
    if inputs[0].shape().dimension(1)? != inputs[1].shape().dimension(0)? {
        return Err(contract_error(
            "operation descriptor",
            "left matrix width must equal right matrix height",
        ));
    }
    if inputs[0].shape().dimension(0)? != output.shape().dimension(0)?
        || inputs[1].shape().dimension(1)? != output.shape().dimension(1)?
    {
        return Err(contract_error(
            "operation descriptor",
            "output dimensions must equal outer matrix dimensions",
        ));
    }
    Ok(())
}

fn require_input_count(inputs: &[TensorDescriptor], expected: usize) -> Result<(), RuntimeError> {
    if inputs.len() != expected {
        return Err(contract_error(
            "operation descriptor",
            "input count is invalid",
        ));
    }
    Ok(())
}

fn require_same_dtype(
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), RuntimeError> {
    if inputs
        .iter()
        .any(|input| input.data_type() != output.data_type())
    {
        return Err(contract_error(
            "operation descriptor",
            "input and output dtypes must match",
        ));
    }
    Ok(())
}

fn require_rank(shape: &TensorShape, expected: usize) -> Result<(), RuntimeError> {
    if shape.rank() != expected {
        return Err(contract_error(
            "operation descriptor",
            "tensor rank is invalid for operation",
        ));
    }
    Ok(())
}

fn contract_error(context: &'static str, reason: &'static str) -> RuntimeError {
    RuntimeError::BackendContractViolation { context, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(shape: impl Into<Box<[usize]>>) -> TensorDescriptor {
        TensorDescriptor::new(TensorShape::new(shape), DataType::F32)
    }

    #[test]
    fn operation_descriptor_accepts_generic_scalar_contracts() {
        let matrix = descriptor([2, 3]);
        let vector = descriptor([3]);
        let output = descriptor([2]);
        let operation = OperationDescriptor::new(
            OperationKind::MatrixVectorMultiply,
            [matrix, vector],
            output.clone(),
        )
        .expect("valid matrix-vector contract");

        assert_eq!(operation.kind(), OperationKind::MatrixVectorMultiply);
        assert_eq!(operation.inputs().len(), 2);
        assert_eq!(operation.output(), &output);
    }

    #[test]
    fn operation_descriptor_rejects_invalid_shapes_and_dtypes() {
        let matrix = descriptor([2, 3]);
        let wrong_vector = descriptor([2]);
        let output = descriptor([2]);
        assert_eq!(
            OperationDescriptor::new(
                OperationKind::MatrixVectorMultiply,
                [matrix, wrong_vector],
                output,
            ),
            Err(RuntimeError::BackendContractViolation {
                context: "operation descriptor",
                reason: "matrix width must equal vector length",
            })
        );

        let f32_input = descriptor([2]);
        let bf16_output = TensorDescriptor::new(TensorShape::new([2]), DataType::BF16);
        assert_eq!(
            OperationDescriptor::new(OperationKind::SiLU, [f32_input], bf16_output),
            Err(RuntimeError::BackendContractViolation {
                context: "operation descriptor",
                reason: "input and output dtypes must match",
            })
        );
    }

    #[test]
    fn execution_result_enforces_output_and_memory_budgets() {
        let output = descriptor([2]);
        let operation =
            OperationDescriptor::new(OperationKind::SiLU, [output.clone()], output.clone())
                .expect("valid unary contract");
        let request = ExecutionRequest::new(operation, ExecutionBudget::new(16, 8));
        let metrics = ExecutionMetrics::new(Duration::from_micros(7), 16, 8);
        let result = ExecutionResult::new(&request, output.clone(), metrics)
            .expect("result stays within budget");

        assert_eq!(result.output(), &output);
        assert_eq!(result.metrics(), metrics);
        assert_eq!(
            ExecutionResult::new(
                &request,
                output,
                ExecutionMetrics::new(Duration::ZERO, 17, 8),
            ),
            Err(RuntimeError::BackendContractViolation {
                context: "execution result",
                reason: "host memory exceeds admitted budget",
            })
        );
    }
}
