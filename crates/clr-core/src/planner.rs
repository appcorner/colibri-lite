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

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < f64::EPSILON);
    }
}
