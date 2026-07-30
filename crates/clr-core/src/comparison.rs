//! Deterministic, backend-neutral differential comparison contracts.
//!
//! Candidate outputs are compared in caller-supplied stage order. Tensor
//! metadata and value lengths are checked before floating-point values so a
//! report always identifies the earliest meaningful divergence.

use crate::{RuntimeError, TensorDescriptor};

/// A documented absolute-and-relative tolerance for one comparison stage.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonTolerance {
    absolute: f32,
    relative: f32,
    source: Box<str>,
}

impl ComparisonTolerance {
    /// Creates a finite, non-negative tolerance with a documented source.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] when either tolerance
    /// is non-finite or negative, or when `source` is empty.
    pub fn new(
        absolute: f32,
        relative: f32,
        source: impl Into<Box<str>>,
    ) -> Result<Self, RuntimeError> {
        let source = source.into();
        if !absolute.is_finite() || !relative.is_finite() || absolute < 0.0 || relative < 0.0 {
            return Err(contract_error(
                "comparison tolerance",
                "absolute and relative tolerances must be finite and non-negative",
            ));
        }
        if source.trim().is_empty() {
            return Err(contract_error(
                "comparison tolerance",
                "source must not be empty",
            ));
        }
        Ok(Self {
            absolute,
            relative,
            source,
        })
    }

    /// Returns the absolute tolerance term.
    #[must_use]
    pub const fn absolute(&self) -> f32 {
        self.absolute
    }

    /// Returns the relative tolerance term.
    #[must_use]
    pub const fn relative(&self) -> f32 {
        self.relative
    }

    /// Returns the document or contract identifier defining this tolerance.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    fn allowed_error(&self, reference: f32) -> f32 {
        self.absolute + self.relative * reference.abs()
    }
}

/// One ordered candidate-versus-reference stage comparison.
#[derive(Debug, Clone)]
pub struct StageComparison<'a> {
    stage_id: Box<str>,
    reference_descriptor: TensorDescriptor,
    candidate_descriptor: TensorDescriptor,
    reference_values: &'a [f32],
    candidate_values: &'a [f32],
    tolerance: ComparisonTolerance,
}

impl<'a> StageComparison<'a> {
    /// Creates one ordered comparison stage.
    ///
    /// Value lengths are intentionally checked during comparison so an invalid
    /// candidate result is reported as the first divergence rather than causing
    /// an unrelated caller-side construction failure.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] when `stage_id` is
    /// empty.
    pub fn new(
        stage_id: impl Into<Box<str>>,
        reference_descriptor: TensorDescriptor,
        candidate_descriptor: TensorDescriptor,
        reference_values: &'a [f32],
        candidate_values: &'a [f32],
        tolerance: ComparisonTolerance,
    ) -> Result<Self, RuntimeError> {
        let stage_id = stage_id.into();
        if stage_id.trim().is_empty() {
            return Err(contract_error(
                "stage comparison",
                "stage ID must not be empty",
            ));
        }
        Ok(Self {
            stage_id,
            reference_descriptor,
            candidate_descriptor,
            reference_values,
            candidate_values,
            tolerance,
        })
    }
}

/// Overall result of a differential comparison sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentialStatus {
    /// Every applicable value is exactly equal.
    ExactMatch,
    /// At least one value differs, but all differences meet their tolerances.
    WithinTolerance,
    /// A structural, finite-value, or numerical mismatch was found.
    Diverged,
}

/// Category of the first detected difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceKind {
    /// Reference and candidate tensor metadata differ.
    TensorMetadataMismatch,
    /// A value slice length does not match its tensor metadata.
    ValueLengthMismatch,
    /// The frozen reference unexpectedly contains NaN or infinity.
    NonFiniteReference,
    /// The candidate contains NaN or infinity.
    NonFiniteCandidate,
    /// A finite value exceeds its documented tolerance.
    NumericalToleranceExceeded,
}

/// Detailed first divergence, including the tolerance selected for its stage.
#[derive(Debug, Clone, PartialEq)]
pub struct FirstDivergence {
    stage_index: usize,
    stage_id: Box<str>,
    kind: DivergenceKind,
    element_index: Option<usize>,
    reference_value: Option<f32>,
    candidate_value: Option<f32>,
    absolute_error: Option<f32>,
    allowed_error: Option<f32>,
    tolerance: ComparisonTolerance,
}

impl FirstDivergence {
    /// Returns the zero-based comparison-stage index.
    #[must_use]
    pub const fn stage_index(&self) -> usize {
        self.stage_index
    }

    /// Returns the stable stage identifier.
    #[must_use]
    pub fn stage_id(&self) -> &str {
        &self.stage_id
    }

    /// Returns the failure category.
    #[must_use]
    pub const fn kind(&self) -> DivergenceKind {
        self.kind
    }

    /// Returns the flat value index, if a value was compared.
    #[must_use]
    pub const fn element_index(&self) -> Option<usize> {
        self.element_index
    }

    /// Returns the frozen reference value, if applicable.
    #[must_use]
    pub const fn reference_value(&self) -> Option<f32> {
        self.reference_value
    }

    /// Returns the candidate value, if applicable.
    #[must_use]
    pub const fn candidate_value(&self) -> Option<f32> {
        self.candidate_value
    }

    /// Returns absolute error, if both values were finite.
    #[must_use]
    pub const fn absolute_error(&self) -> Option<f32> {
        self.absolute_error
    }

    /// Returns the tolerance threshold, if a numerical comparison occurred.
    #[must_use]
    pub const fn allowed_error(&self) -> Option<f32> {
        self.allowed_error
    }

    /// Returns the tolerance selected for the divergent stage.
    #[must_use]
    pub const fn tolerance(&self) -> &ComparisonTolerance {
        &self.tolerance
    }
}

/// Deterministic report for an ordered stage-comparison sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentialReport {
    status: DifferentialStatus,
    checked_stage_count: usize,
    first_divergence: Option<FirstDivergence>,
}

impl DifferentialReport {
    /// Compares ordered stages and returns the first structural or numerical
    /// divergence. Later stages are not inspected after a failure.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] for an empty stage
    /// sequence.
    pub fn compare(stages: &[StageComparison<'_>]) -> Result<Self, RuntimeError> {
        if stages.is_empty() {
            return Err(contract_error(
                "differential comparison",
                "at least one comparison stage is required",
            ));
        }

        let mut status = DifferentialStatus::ExactMatch;
        for (stage_index, stage) in stages.iter().enumerate() {
            match compare_stage(stage_index, stage) {
                StageOutcome::Exact => {}
                StageOutcome::WithinTolerance => status = DifferentialStatus::WithinTolerance,
                StageOutcome::Diverged(first_divergence) => {
                    return Ok(Self {
                        status: DifferentialStatus::Diverged,
                        checked_stage_count: stage_index + 1,
                        first_divergence: Some(first_divergence),
                    });
                }
            }
        }
        Ok(Self {
            status,
            checked_stage_count: stages.len(),
            first_divergence: None,
        })
    }

    /// Returns the resulting comparison classification.
    #[must_use]
    pub const fn status(&self) -> DifferentialStatus {
        self.status
    }

    /// Returns how many ordered stages were inspected.
    #[must_use]
    pub const fn checked_stage_count(&self) -> usize {
        self.checked_stage_count
    }

    /// Returns the first mismatch, if one was found.
    #[must_use]
    pub const fn first_divergence(&self) -> Option<&FirstDivergence> {
        self.first_divergence.as_ref()
    }
}

enum StageOutcome {
    Exact,
    WithinTolerance,
    Diverged(FirstDivergence),
}

#[derive(Clone, Copy, Default)]
struct DivergenceDetails {
    element_index: Option<usize>,
    reference_value: Option<f32>,
    candidate_value: Option<f32>,
    absolute_error: Option<f32>,
    allowed_error: Option<f32>,
}

fn compare_stage(stage_index: usize, stage: &StageComparison<'_>) -> StageOutcome {
    if stage.reference_descriptor != stage.candidate_descriptor {
        return StageOutcome::Diverged(divergence(
            stage_index,
            stage,
            DivergenceKind::TensorMetadataMismatch,
            DivergenceDetails::default(),
        ));
    }

    let Ok(expected_length) = stage.reference_descriptor.shape().element_count() else {
        return StageOutcome::Diverged(divergence(
            stage_index,
            stage,
            DivergenceKind::ValueLengthMismatch,
            DivergenceDetails::default(),
        ));
    };
    if stage.reference_values.len() != expected_length
        || stage.candidate_values.len() != expected_length
    {
        return StageOutcome::Diverged(divergence(
            stage_index,
            stage,
            DivergenceKind::ValueLengthMismatch,
            DivergenceDetails::default(),
        ));
    }

    let mut within_tolerance = false;
    for (element_index, (&reference, &candidate)) in stage
        .reference_values
        .iter()
        .zip(stage.candidate_values)
        .enumerate()
    {
        if !reference.is_finite() {
            return StageOutcome::Diverged(divergence(
                stage_index,
                stage,
                DivergenceKind::NonFiniteReference,
                DivergenceDetails {
                    element_index: Some(element_index),
                    reference_value: Some(reference),
                    candidate_value: Some(candidate),
                    ..DivergenceDetails::default()
                },
            ));
        }
        if !candidate.is_finite() {
            return StageOutcome::Diverged(divergence(
                stage_index,
                stage,
                DivergenceKind::NonFiniteCandidate,
                DivergenceDetails {
                    element_index: Some(element_index),
                    reference_value: Some(reference),
                    candidate_value: Some(candidate),
                    ..DivergenceDetails::default()
                },
            ));
        }
        let absolute_error = (reference - candidate).abs();
        let allowed_error = stage.tolerance.allowed_error(reference);
        if absolute_error > allowed_error {
            return StageOutcome::Diverged(divergence(
                stage_index,
                stage,
                DivergenceKind::NumericalToleranceExceeded,
                DivergenceDetails {
                    element_index: Some(element_index),
                    reference_value: Some(reference),
                    candidate_value: Some(candidate),
                    absolute_error: Some(absolute_error),
                    allowed_error: Some(allowed_error),
                },
            ));
        }
        if absolute_error != 0.0 {
            within_tolerance = true;
        }
    }
    if within_tolerance {
        StageOutcome::WithinTolerance
    } else {
        StageOutcome::Exact
    }
}

fn divergence(
    stage_index: usize,
    stage: &StageComparison<'_>,
    kind: DivergenceKind,
    details: DivergenceDetails,
) -> FirstDivergence {
    FirstDivergence {
        stage_index,
        stage_id: stage.stage_id.clone(),
        kind,
        element_index: details.element_index,
        reference_value: details.reference_value,
        candidate_value: details.candidate_value,
        absolute_error: details.absolute_error,
        allowed_error: details.allowed_error,
        tolerance: stage.tolerance.clone(),
    }
}

fn contract_error(context: &'static str, reason: &'static str) -> RuntimeError {
    RuntimeError::BackendContractViolation { context, reason }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataType, TensorShape};

    fn descriptor(shape: impl Into<Box<[usize]>>) -> TensorDescriptor {
        TensorDescriptor::new(TensorShape::new(shape), DataType::F32)
    }

    fn tolerance() -> ComparisonTolerance {
        ComparisonTolerance::new(1.0e-6, 1.0e-5, "frozen-contract/test").expect("valid tolerance")
    }

    #[test]
    fn reports_exact_and_within_tolerance_results() {
        let descriptor = descriptor([2]);
        let exact = StageComparison::new(
            "embedding",
            descriptor.clone(),
            descriptor.clone(),
            &[1.0, 2.0],
            &[1.0, 2.0],
            tolerance(),
        )
        .expect("valid exact stage");
        let tolerated = StageComparison::new(
            "rmsnorm",
            descriptor.clone(),
            descriptor,
            &[100.0, 2.0],
            &[100.0005, 2.0],
            tolerance(),
        )
        .expect("valid tolerated stage");

        assert_eq!(
            DifferentialReport::compare(&[exact])
                .expect("exact comparison")
                .status(),
            DifferentialStatus::ExactMatch
        );
        let report = DifferentialReport::compare(&[tolerated]).expect("tolerated comparison");
        assert_eq!(report.status(), DifferentialStatus::WithinTolerance);
        assert_eq!(report.checked_stage_count(), 1);
        assert_eq!(report.first_divergence(), None);
    }

    #[test]
    fn reports_first_structural_divergence_before_values() {
        let stage = StageComparison::new(
            "embedding",
            descriptor([2]),
            descriptor([1, 2]),
            &[f32::NAN, 2.0],
            &[1.0, 2.0],
            tolerance(),
        )
        .expect("valid stage metadata container");

        let report = DifferentialReport::compare(&[stage]).expect("comparison report");
        let divergence = report.first_divergence().expect("first divergence");
        assert_eq!(report.status(), DifferentialStatus::Diverged);
        assert_eq!(divergence.kind(), DivergenceKind::TensorMetadataMismatch);
        assert_eq!(divergence.stage_id(), "embedding");
        assert_eq!(divergence.tolerance().source(), "frozen-contract/test");
        assert_eq!(divergence.element_index(), None);
    }

    #[test]
    fn stops_at_first_numerical_divergence_and_records_tolerance() {
        let descriptor = descriptor([2]);
        let passing = StageComparison::new(
            "embedding",
            descriptor.clone(),
            descriptor.clone(),
            &[1.0, 2.0],
            &[1.0, 2.0],
            tolerance(),
        )
        .expect("valid passing stage");
        let failing = StageComparison::new(
            "router_logits",
            descriptor.clone(),
            descriptor,
            &[1.0, 2.0],
            &[1.1, 2.0],
            tolerance(),
        )
        .expect("valid failing stage");

        let report = DifferentialReport::compare(&[passing, failing]).expect("comparison report");
        let divergence = report.first_divergence().expect("first divergence");
        assert_eq!(report.checked_stage_count(), 2);
        assert_eq!(divergence.stage_index(), 1);
        assert_eq!(
            divergence.kind(),
            DivergenceKind::NumericalToleranceExceeded
        );
        assert_eq!(divergence.element_index(), Some(0));
        assert_eq!(
            divergence.absolute_error().map(f32::to_bits),
            Some(0.100_000_024_f32.to_bits())
        );
        assert_eq!(
            divergence.allowed_error().map(f32::to_bits),
            Some(0.000_011_f32.to_bits())
        );
        assert_eq!(
            divergence.tolerance().absolute().to_bits(),
            1.0e-6_f32.to_bits()
        );
        assert_eq!(
            divergence.tolerance().relative().to_bits(),
            1.0e-5_f32.to_bits()
        );
    }

    #[test]
    fn rejects_invalid_tolerance_and_empty_stage_sequence() {
        assert_eq!(
            ComparisonTolerance::new(-1.0, 0.0, "source"),
            Err(RuntimeError::BackendContractViolation {
                context: "comparison tolerance",
                reason: "absolute and relative tolerances must be finite and non-negative",
            })
        );
        assert_eq!(
            DifferentialReport::compare(&[]),
            Err(RuntimeError::BackendContractViolation {
                context: "differential comparison",
                reason: "at least one comparison stage is required",
            })
        );
    }
}
