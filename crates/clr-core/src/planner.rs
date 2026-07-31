//! Analytical placement-planner cost model contracts.
//!
//! The model is deliberately generic: callers supply model-derived work and
//! measured hardware rates. It does not probe the host, select a backend, or
//! contain a machine-specific throughput constant.

use crate::RuntimeError;

const GIGAFLOPS_TO_FLOPS: f64 = 1_000_000_000.0;
const GIB_TO_BYTES: f64 = 1_073_741_824.0;
const MIB_TO_BYTES: f64 = 1_048_576.0;
const MAX_EXACT_F64_INTEGER: u64 = 1 << 53;

/// One positive, measured rate cited by a planner input profile.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredRate {
    per_second: f64,
    measurement_id: Box<str>,
}

impl MeasuredRate {
    /// Creates a finite, positive rate with its source measurement identifier.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] when the rate is not
    /// finite and positive, or when the measurement identifier is empty.
    pub fn new(per_second: f64, measurement_id: impl Into<Box<str>>) -> Result<Self, RuntimeError> {
        if !per_second.is_finite() || per_second <= 0.0 {
            return Err(contract_error(
                "planner measured rate",
                "rate must be finite and greater than zero",
            ));
        }
        let measurement_id = measurement_id.into();
        if measurement_id.trim().is_empty() {
            return Err(contract_error(
                "planner measured rate",
                "measurement identifier must not be empty",
            ));
        }
        Ok(Self {
            per_second,
            measurement_id,
        })
    }

    /// Returns the measured rate in the unit required by its input field.
    #[must_use]
    pub const fn per_second(&self) -> f64 {
        self.per_second
    }

    /// Returns the immutable source measurement identifier.
    #[must_use]
    pub fn measurement_id(&self) -> &str {
        &self.measurement_id
    }
}

/// Model-derived resource demand for processing one token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenWork {
    compute_gflop: f64,
    ram_bytes: u64,
    storage_bytes: u64,
}

impl TokenWork {
    /// Creates non-negative compute, RAM, and storage demand for one token.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] when compute work is
    /// non-finite or negative.
    pub fn new(
        compute_gflop: f64,
        ram_bytes: u64,
        storage_bytes: u64,
    ) -> Result<Self, RuntimeError> {
        if !compute_gflop.is_finite() || compute_gflop < 0.0 {
            return Err(contract_error(
                "planner token work",
                "compute work must be finite and non-negative",
            ));
        }
        Ok(Self {
            compute_gflop,
            ram_bytes,
            storage_bytes,
        })
    }

    /// Returns compute work in GFLOP per token.
    #[must_use]
    pub const fn compute_gflop(self) -> f64 {
        self.compute_gflop
    }

    /// Returns RAM traffic in bytes per token.
    #[must_use]
    pub const fn ram_bytes(self) -> u64 {
        self.ram_bytes
    }

    /// Returns storage traffic in bytes per token.
    #[must_use]
    pub const fn storage_bytes(self) -> u64 {
        self.storage_bytes
    }
}

/// Explicit prompt and decode lengths for an analytical estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerWorkload {
    prefill_tokens: u64,
    decode_tokens: u64,
}

impl PlannerWorkload {
    /// Creates a workload with a non-zero decode target.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] when decode tokens
    /// are zero.
    pub fn new(prefill_tokens: u64, decode_tokens: u64) -> Result<Self, RuntimeError> {
        if decode_tokens == 0 {
            return Err(contract_error(
                "planner workload",
                "decode token count must be greater than zero",
            ));
        }
        Ok(Self {
            prefill_tokens,
            decode_tokens,
        })
    }

    /// Returns prompt tokens processed during prefill.
    #[must_use]
    pub const fn prefill_tokens(self) -> u64 {
        self.prefill_tokens
    }

    /// Returns tokens generated during decode.
    #[must_use]
    pub const fn decode_tokens(self) -> u64 {
        self.decode_tokens
    }
}

/// Availability state copied from a versioned profile measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileMeasurementStatus {
    /// The required path has a matching measured profile record.
    Measured,
    /// The profile explicitly says the path is unavailable.
    Unavailable,
    /// The measurement was intentionally not run.
    NotRun,
}

impl ProfileMeasurementStatus {
    #[must_use]
    const fn is_measured(self) -> bool {
        matches!(self, Self::Measured)
    }
}

/// Memory tier used by a dense or expert component of a candidate plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementTier {
    /// Host RAM.
    Ram,
    /// Device VRAM.
    Vram,
    /// SSD-backed expert storage.
    Ssd,
}

impl PlacementTier {
    const DENSE_TIERS: [Self; 2] = [Self::Ram, Self::Vram];
    const EXPERT_TIERS: [Self; 3] = [Self::Ram, Self::Vram, Self::Ssd];

    const fn identifier(self) -> &'static str {
        match self {
            Self::Ram => "ram",
            Self::Vram => "vram",
            Self::Ssd => "ssd",
        }
    }
}

/// Profile-derived evidence needed to admit a placement before budget checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementCapabilities {
    backend_id: Box<str>,
    ram_compute: ProfileMeasurementStatus,
    vram_compute: ProfileMeasurementStatus,
    ram_access: ProfileMeasurementStatus,
    vram_access: ProfileMeasurementStatus,
    ssd_access: ProfileMeasurementStatus,
    host_to_device_transfer: ProfileMeasurementStatus,
}

impl PlacementCapabilities {
    /// Creates evidence-backed capabilities for one backend identifier.
    ///
    /// A tier is eligible only when the matching status is [`ProfileMeasurementStatus::Measured`].
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] when `backend_id` is
    /// empty.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend_id: impl Into<Box<str>>,
        ram_compute: ProfileMeasurementStatus,
        vram_compute: ProfileMeasurementStatus,
        ram_access: ProfileMeasurementStatus,
        vram_access: ProfileMeasurementStatus,
        ssd_access: ProfileMeasurementStatus,
        host_to_device_transfer: ProfileMeasurementStatus,
    ) -> Result<Self, RuntimeError> {
        let backend_id = backend_id.into();
        if backend_id.trim().is_empty() {
            return Err(contract_error(
                "planner placement capabilities",
                "backend identifier must not be empty",
            ));
        }
        Ok(Self {
            backend_id,
            ram_compute,
            vram_compute,
            ram_access,
            vram_access,
            ssd_access,
            host_to_device_transfer,
        })
    }

    /// Enumerates all and only placements whose required evidence is measured.
    #[must_use]
    pub fn enumerate_supported(&self) -> Box<[CandidatePlacement]> {
        let mut candidates = Vec::new();
        for dense_location in PlacementTier::DENSE_TIERS {
            for expert_location in PlacementTier::EXPERT_TIERS {
                if self.supports(dense_location, expert_location) {
                    candidates.push(CandidatePlacement {
                        backend_id: self.backend_id.clone(),
                        dense_location,
                        expert_location,
                    });
                }
            }
        }
        candidates.into_boxed_slice()
    }

    fn supports(&self, dense_location: PlacementTier, expert_location: PlacementTier) -> bool {
        self.supports_dense(dense_location)
            && self.supports_experts(expert_location)
            && (!requires_host_to_device_transfer(dense_location, expert_location)
                || self.host_to_device_transfer.is_measured())
    }

    const fn supports_dense(&self, location: PlacementTier) -> bool {
        match location {
            PlacementTier::Ram => self.ram_compute.is_measured(),
            PlacementTier::Vram => self.vram_compute.is_measured(),
            PlacementTier::Ssd => false,
        }
    }

    const fn supports_experts(&self, location: PlacementTier) -> bool {
        match location {
            PlacementTier::Ram => self.ram_access.is_measured(),
            PlacementTier::Vram => self.vram_access.is_measured(),
            PlacementTier::Ssd => self.ssd_access.is_measured(),
        }
    }
}

/// A deterministic, evidence-supported dense/expert placement candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidatePlacement {
    backend_id: Box<str>,
    dense_location: PlacementTier,
    expert_location: PlacementTier,
}

impl CandidatePlacement {
    /// Returns the backend identifier whose measurements admitted this candidate.
    #[must_use]
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Returns the dense component memory tier.
    #[must_use]
    pub const fn dense_location(&self) -> PlacementTier {
        self.dense_location
    }

    /// Returns the expert component memory tier.
    #[must_use]
    pub const fn expert_location(&self) -> PlacementTier {
        self.expert_location
    }

    /// Returns the stable candidate identifier used by later ranking steps.
    #[must_use]
    pub fn plan_id(&self) -> String {
        format!(
            "backend:{}/dense:{}/experts:{}",
            self.backend_id,
            self.dense_location.identifier(),
            self.expert_location.identifier()
        )
    }
}

/// Explicit resource limits selected for one planner request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerBudgets {
    ram_bytes: u64,
    vram_bytes: u64,
    context_tokens: u64,
}

impl PlannerBudgets {
    /// Creates RAM, VRAM, and requested-context limits without host inference.
    #[must_use]
    pub const fn new(ram_bytes: u64, vram_bytes: u64, context_tokens: u64) -> Self {
        Self {
            ram_bytes,
            vram_bytes,
            context_tokens,
        }
    }

    /// Returns the user-selected RAM limit in bytes.
    #[must_use]
    pub const fn ram_bytes(self) -> u64 {
        self.ram_bytes
    }

    /// Returns the user-selected VRAM limit in bytes.
    #[must_use]
    pub const fn vram_bytes(self) -> u64 {
        self.vram_bytes
    }

    /// Returns the requested context length in tokens.
    #[must_use]
    pub const fn context_tokens(self) -> u64 {
        self.context_tokens
    }
}

/// Calculated resources required by a candidate before admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateResourceRequirements {
    ram_bytes: u64,
    vram_bytes: u64,
    max_context_tokens: u64,
}

impl CandidateResourceRequirements {
    /// Creates calculated RAM/VRAM demand and supported context capacity.
    #[must_use]
    pub const fn new(ram_bytes: u64, vram_bytes: u64, max_context_tokens: u64) -> Self {
        Self {
            ram_bytes,
            vram_bytes,
            max_context_tokens,
        }
    }

    /// Returns required RAM in bytes.
    #[must_use]
    pub const fn ram_bytes(self) -> u64 {
        self.ram_bytes
    }

    /// Returns required VRAM in bytes.
    #[must_use]
    pub const fn vram_bytes(self) -> u64 {
        self.vram_bytes
    }

    /// Returns the candidate's maximum supported context length in tokens.
    #[must_use]
    pub const fn max_context_tokens(self) -> u64 {
        self.max_context_tokens
    }
}

/// Machine-readable reason a candidate cannot be admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerRejectionCode {
    /// Required RAM exceeds the explicit RAM budget.
    RamBudgetExceeded,
    /// Required VRAM exceeds the explicit VRAM budget.
    VramBudgetExceeded,
    /// Requested context exceeds the candidate capacity.
    ContextLimitExceeded,
}

/// One ordered resource rejection with calculated requirement and limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerRejection {
    plan_id: Box<str>,
    code: PlannerRejectionCode,
    required: u64,
    limit: u64,
}

impl PlannerRejection {
    /// Returns the stable candidate identifier.
    #[must_use]
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }

    /// Returns the machine-readable rejection code.
    #[must_use]
    pub const fn code(&self) -> PlannerRejectionCode {
        self.code
    }

    /// Returns the calculated required bytes or requested tokens.
    #[must_use]
    pub const fn required(&self) -> u64 {
        self.required
    }

    /// Returns the applicable budget bytes or capacity tokens.
    #[must_use]
    pub const fn limit(&self) -> u64 {
        self.limit
    }
}

/// Result of admitting one evidence-supported placement against explicit limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetAdmission {
    /// Every RAM, VRAM, and context constraint is satisfied.
    Admitted,
    /// One or more ordered constraints were exceeded.
    Rejected(Box<[PlannerRejection]>),
}

impl BudgetAdmission {
    /// Returns whether the candidate is admissible.
    #[must_use]
    pub const fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted)
    }

    /// Returns all ordered rejection explanations, if any.
    #[must_use]
    pub fn rejections(&self) -> &[PlannerRejection] {
        match self {
            Self::Admitted => &[],
            Self::Rejected(rejections) => rejections,
        }
    }
}

/// Enforces RAM, VRAM, and context constraints for one candidate placement.
#[must_use]
pub fn admit_candidate(
    candidate: &CandidatePlacement,
    requirements: CandidateResourceRequirements,
    budgets: PlannerBudgets,
) -> BudgetAdmission {
    let mut rejections = Vec::new();
    let plan_id = candidate.plan_id();
    if requirements.ram_bytes() > budgets.ram_bytes() {
        rejections.push(PlannerRejection {
            plan_id: plan_id.clone().into_boxed_str(),
            code: PlannerRejectionCode::RamBudgetExceeded,
            required: requirements.ram_bytes(),
            limit: budgets.ram_bytes(),
        });
    }
    if requirements.vram_bytes() > budgets.vram_bytes() {
        rejections.push(PlannerRejection {
            plan_id: plan_id.clone().into_boxed_str(),
            code: PlannerRejectionCode::VramBudgetExceeded,
            required: requirements.vram_bytes(),
            limit: budgets.vram_bytes(),
        });
    }
    if budgets.context_tokens() > requirements.max_context_tokens() {
        rejections.push(PlannerRejection {
            plan_id: plan_id.into_boxed_str(),
            code: PlannerRejectionCode::ContextLimitExceeded,
            required: budgets.context_tokens(),
            limit: requirements.max_context_tokens(),
        });
    }
    if rejections.is_empty() {
        BudgetAdmission::Admitted
    } else {
        BudgetAdmission::Rejected(rejections.into_boxed_slice())
    }
}

const fn requires_host_to_device_transfer(
    dense_location: PlacementTier,
    expert_location: PlacementTier,
) -> bool {
    matches!(dense_location, PlacementTier::Vram) != matches!(expert_location, PlacementTier::Vram)
}

/// Measured profile rates used by the initial analytical model.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyticalCostModel {
    compute: MeasuredRate,
    ram: MeasuredRate,
    storage: MeasuredRate,
}

impl AnalyticalCostModel {
    /// Creates the model from matching measured compute, RAM, and storage rates.
    #[must_use]
    pub const fn new(compute: MeasuredRate, ram: MeasuredRate, storage: MeasuredRate) -> Self {
        Self {
            compute,
            ram,
            storage,
        }
    }

    /// Returns the profile compute measurement.
    #[must_use]
    pub const fn compute_gflop_per_second(&self) -> &MeasuredRate {
        &self.compute
    }

    /// Returns the profile RAM-bandwidth measurement.
    #[must_use]
    pub const fn ram_gib_per_second(&self) -> &MeasuredRate {
        &self.ram
    }

    /// Returns the profile storage-throughput measurement.
    #[must_use]
    pub const fn storage_mib_per_second(&self) -> &MeasuredRate {
        &self.storage
    }

    /// Estimates serial analytical costs for the explicit workload.
    ///
    /// The result treats compute, RAM traffic, and storage traffic as additive
    /// terms. It is a transparent baseline, not an end-to-end measurement.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::BackendContractViolation`] if a derived duration
    /// cannot remain finite.
    pub fn estimate(
        &self,
        workload: PlannerWorkload,
        prefill_work: TokenWork,
        decode_work: TokenWork,
        startup_storage_bytes: u64,
    ) -> Result<AnalyticalEstimate, RuntimeError> {
        let prefill_per_token = self.estimate_token(prefill_work)?;
        let decode_per_token = self.estimate_token(decode_work)?;
        let prefill_seconds =
            scale_seconds(prefill_per_token.total_seconds(), workload.prefill_tokens())?;
        let decode_seconds =
            scale_seconds(decode_per_token.total_seconds(), workload.decode_tokens())?;
        let startup_seconds = seconds_from_rate(
            exact_u64_as_f64(startup_storage_bytes)?,
            self.storage.per_second() * MIB_TO_BYTES,
        )?;
        let decode_tokens_per_second = reciprocal_seconds(decode_per_token.total_seconds())?;
        Ok(AnalyticalEstimate {
            prefill_per_token,
            decode_per_token,
            prefill_seconds,
            decode_seconds,
            startup_seconds,
            decode_tokens_per_second,
        })
    }

    fn estimate_token(&self, work: TokenWork) -> Result<TokenCostEstimate, RuntimeError> {
        let compute_seconds = seconds_from_rate(
            work.compute_gflop() * GIGAFLOPS_TO_FLOPS,
            self.compute.per_second() * GIGAFLOPS_TO_FLOPS,
        )?;
        let ram_seconds = seconds_from_rate(
            exact_u64_as_f64(work.ram_bytes())?,
            self.ram.per_second() * GIB_TO_BYTES,
        )?;
        let storage_seconds = seconds_from_rate(
            exact_u64_as_f64(work.storage_bytes())?,
            self.storage.per_second() * MIB_TO_BYTES,
        )?;
        let total_seconds = compute_seconds + ram_seconds + storage_seconds;
        if !total_seconds.is_finite() || total_seconds <= 0.0 {
            return Err(contract_error(
                "planner analytical estimate",
                "per-token cost must be finite and greater than zero",
            ));
        }
        Ok(TokenCostEstimate {
            compute: compute_seconds,
            ram: ram_seconds,
            storage: storage_seconds,
            total: total_seconds,
        })
    }
}

/// Transparent per-token component costs in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenCostEstimate {
    compute: f64,
    ram: f64,
    storage: f64,
    total: f64,
}

impl TokenCostEstimate {
    /// Returns compute cost in seconds per token.
    #[must_use]
    pub const fn compute_seconds(self) -> f64 {
        self.compute
    }

    /// Returns RAM traffic cost in seconds per token.
    #[must_use]
    pub const fn ram_seconds(self) -> f64 {
        self.ram
    }

    /// Returns storage traffic cost in seconds per token.
    #[must_use]
    pub const fn storage_seconds(self) -> f64 {
        self.storage
    }

    /// Returns the serial sum of all component costs in seconds per token.
    #[must_use]
    pub const fn total_seconds(self) -> f64 {
        self.total
    }
}

/// Analytical prefill, decode, and startup predictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyticalEstimate {
    prefill_per_token: TokenCostEstimate,
    decode_per_token: TokenCostEstimate,
    prefill_seconds: f64,
    decode_seconds: f64,
    startup_seconds: f64,
    decode_tokens_per_second: f64,
}

impl AnalyticalEstimate {
    /// Returns per-token prefill costs.
    #[must_use]
    pub const fn prefill_per_token(self) -> TokenCostEstimate {
        self.prefill_per_token
    }

    /// Returns per-token decode costs.
    #[must_use]
    pub const fn decode_per_token(self) -> TokenCostEstimate {
        self.decode_per_token
    }

    /// Returns projected total prefill time in seconds.
    #[must_use]
    pub const fn prefill_seconds(self) -> f64 {
        self.prefill_seconds
    }

    /// Returns projected total decode time in seconds.
    #[must_use]
    pub const fn decode_seconds(self) -> f64 {
        self.decode_seconds
    }

    /// Returns projected startup storage-read time in seconds.
    #[must_use]
    pub const fn startup_seconds(self) -> f64 {
        self.startup_seconds
    }

    /// Returns the predicted decode throughput in tokens per second.
    #[must_use]
    pub const fn decode_tokens_per_second(self) -> f64 {
        self.decode_tokens_per_second
    }
}

fn seconds_from_rate(demand: f64, rate: f64) -> Result<f64, RuntimeError> {
    let seconds = demand / rate;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(contract_error(
            "planner analytical estimate",
            "derived duration must be finite and non-negative",
        ));
    }
    Ok(seconds)
}

fn reciprocal_seconds(seconds: f64) -> Result<f64, RuntimeError> {
    let rate = 1.0 / seconds;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(contract_error(
            "planner analytical estimate",
            "decode throughput must be finite and greater than zero",
        ));
    }
    Ok(rate)
}

fn scale_seconds(seconds_per_token: f64, token_count: u64) -> Result<f64, RuntimeError> {
    let seconds = seconds_per_token * exact_u64_as_f64(token_count)?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(contract_error(
            "planner analytical estimate",
            "workload duration must be finite and non-negative",
        ));
    }
    Ok(seconds)
}

fn exact_u64_as_f64(value: u64) -> Result<f64, RuntimeError> {
    if value > MAX_EXACT_F64_INTEGER {
        return Err(contract_error(
            "planner analytical estimate",
            "integer input exceeds the exact f64 range",
        ));
    }
    let high = u32::try_from(value >> 32).map_err(|_| {
        contract_error(
            "planner analytical estimate",
            "integer high word cannot be represented",
        )
    })?;
    let low = u32::try_from(value & u64::from(u32::MAX)).map_err(|_| {
        contract_error(
            "planner analytical estimate",
            "integer low word cannot be represented",
        )
    })?;
    Ok(f64::from(high) * 4_294_967_296.0 + f64::from(low))
}

fn contract_error(context: &'static str, reason: &'static str) -> RuntimeError {
    RuntimeError::BackendContractViolation { context, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> AnalyticalCostModel {
        AnalyticalCostModel::new(
            MeasuredRate::new(1.0, "hardware.cpu.compute.median").unwrap(),
            MeasuredRate::new(1.0, "hardware.ram.median").unwrap(),
            MeasuredRate::new(1.0, "hardware.storage.random-read.median").unwrap(),
        )
    }

    #[test]
    fn combines_only_caller_supplied_profile_rates_and_model_work() {
        let work = TokenWork::new(1.0, 1_073_741_824, 1_048_576).unwrap();
        let estimate = model()
            .estimate(
                PlannerWorkload::new(2, 4).unwrap(),
                work,
                work,
                4 * 1_048_576,
            )
            .unwrap();

        assert_close(estimate.prefill_per_token().compute_seconds(), 1.0);
        assert_close(estimate.prefill_per_token().ram_seconds(), 1.0);
        assert_close(estimate.prefill_per_token().storage_seconds(), 1.0);
        assert_close(estimate.prefill_per_token().total_seconds(), 3.0);
        assert_close(estimate.prefill_seconds(), 6.0);
        assert_close(estimate.decode_seconds(), 12.0);
        assert_close(estimate.startup_seconds(), 4.0);
        assert_close(estimate.decode_tokens_per_second(), 1.0 / 3.0);
    }

    #[test]
    fn changed_profile_rate_changes_the_prediction_without_hidden_machine_constants() {
        let work = TokenWork::new(0.0, 0, 1_048_576).unwrap();
        let fast_storage = AnalyticalCostModel::new(
            MeasuredRate::new(1.0, "compute").unwrap(),
            MeasuredRate::new(1.0, "ram").unwrap(),
            MeasuredRate::new(2.0, "storage").unwrap(),
        );

        let baseline = model()
            .estimate(PlannerWorkload::new(0, 1).unwrap(), work, work, 0)
            .unwrap();
        let faster = fast_storage
            .estimate(PlannerWorkload::new(0, 1).unwrap(), work, work, 0)
            .unwrap();
        assert_close(baseline.decode_per_token().storage_seconds(), 1.0);
        assert_close(faster.decode_per_token().storage_seconds(), 0.5);
        assert_close(faster.decode_tokens_per_second(), 2.0);
    }

    #[test]
    fn rejects_missing_or_invalid_measurements_and_zero_cost_decode() {
        assert!(MeasuredRate::new(0.0, "measurement").is_err());
        assert!(MeasuredRate::new(1.0, " ").is_err());
        assert!(TokenWork::new(f64::NAN, 0, 0).is_err());
        assert!(PlannerWorkload::new(1, 0).is_err());

        let zero_work = TokenWork::new(0.0, 0, 0).unwrap();
        assert!(
            model()
                .estimate(PlannerWorkload::new(0, 1).unwrap(), zero_work, zero_work, 0)
                .is_err()
        );
        assert!(
            model()
                .estimate(
                    PlannerWorkload::new(MAX_EXACT_F64_INTEGER + 1, 1).unwrap(),
                    TokenWork::new(1.0, 0, 0).unwrap(),
                    TokenWork::new(1.0, 0, 0).unwrap(),
                    0,
                )
                .is_err()
        );
    }

    #[test]
    fn enumerates_cpu_ram_and_ssd_candidates_from_measured_profile_statuses() {
        let capabilities = PlacementCapabilities::new(
            "cpu",
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Unavailable,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Unavailable,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::NotRun,
        )
        .unwrap();

        let plan_ids: Vec<_> = capabilities
            .enumerate_supported()
            .iter()
            .map(CandidatePlacement::plan_id)
            .collect();
        assert_eq!(
            plan_ids,
            [
                "backend:cpu/dense:ram/experts:ram",
                "backend:cpu/dense:ram/experts:ssd",
            ]
        );
    }

    #[test]
    fn enumerates_vram_only_when_compute_access_and_transfers_are_measured() {
        let capabilities = PlacementCapabilities::new(
            "gpu",
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
        )
        .unwrap();

        let plan_ids: Vec<_> = capabilities
            .enumerate_supported()
            .iter()
            .map(CandidatePlacement::plan_id)
            .collect();
        assert_eq!(
            plan_ids,
            [
                "backend:gpu/dense:ram/experts:ram",
                "backend:gpu/dense:ram/experts:vram",
                "backend:gpu/dense:ram/experts:ssd",
                "backend:gpu/dense:vram/experts:ram",
                "backend:gpu/dense:vram/experts:vram",
                "backend:gpu/dense:vram/experts:ssd",
            ]
        );
    }

    #[test]
    fn omits_mixed_vram_candidates_without_transfer_evidence_and_rejects_empty_backend() {
        assert!(
            PlacementCapabilities::new(
                " ",
                ProfileMeasurementStatus::Measured,
                ProfileMeasurementStatus::Measured,
                ProfileMeasurementStatus::Measured,
                ProfileMeasurementStatus::Measured,
                ProfileMeasurementStatus::Measured,
                ProfileMeasurementStatus::Measured,
            )
            .is_err()
        );

        let capabilities = PlacementCapabilities::new(
            "gpu",
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::NotRun,
            ProfileMeasurementStatus::NotRun,
        )
        .unwrap();
        let plan_ids: Vec<_> = capabilities
            .enumerate_supported()
            .iter()
            .map(CandidatePlacement::plan_id)
            .collect();
        assert_eq!(
            plan_ids,
            [
                "backend:gpu/dense:ram/experts:ram",
                "backend:gpu/dense:vram/experts:vram",
            ]
        );
    }

    #[test]
    fn admits_exact_ram_vram_and_context_boundaries() {
        let candidate = cpu_candidate();
        let requirements = CandidateResourceRequirements::new(100, 0, 128);
        let admission = admit_candidate(&candidate, requirements, PlannerBudgets::new(100, 0, 128));

        assert!(admission.is_admitted());
        assert!(admission.rejections().is_empty());
    }

    #[test]
    fn reports_every_exceeded_budget_with_stable_plan_id_and_order() {
        let candidate = cpu_candidate();
        let admission = admit_candidate(
            &candidate,
            CandidateResourceRequirements::new(101, 202, 127),
            PlannerBudgets::new(100, 200, 128),
        );

        assert!(!admission.is_admitted());
        let rejections = admission.rejections();
        assert_eq!(rejections.len(), 3);
        assert_eq!(
            rejections[0].code(),
            PlannerRejectionCode::RamBudgetExceeded
        );
        assert_eq!(rejections[0].required(), 101);
        assert_eq!(rejections[0].limit(), 100);
        assert_eq!(
            rejections[1].code(),
            PlannerRejectionCode::VramBudgetExceeded
        );
        assert_eq!(rejections[1].required(), 202);
        assert_eq!(rejections[1].limit(), 200);
        assert_eq!(
            rejections[2].code(),
            PlannerRejectionCode::ContextLimitExceeded
        );
        assert_eq!(rejections[2].required(), 128);
        assert_eq!(rejections[2].limit(), 127);
        for rejection in rejections {
            assert_eq!(rejection.plan_id(), "backend:cpu/dense:ram/experts:ram");
        }
    }

    fn cpu_candidate() -> CandidatePlacement {
        PlacementCapabilities::new(
            "cpu",
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Unavailable,
            ProfileMeasurementStatus::Measured,
            ProfileMeasurementStatus::Unavailable,
            ProfileMeasurementStatus::Unavailable,
            ProfileMeasurementStatus::NotRun,
        )
        .unwrap()
        .enumerate_supported()
        .into_iter()
        .next()
        .unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < f64::EPSILON);
    }
}
