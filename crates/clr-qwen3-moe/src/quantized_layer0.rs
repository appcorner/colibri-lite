//! Direct-consumption Layer-0 group-128 INT8 expert candidate.
//!
//! This module deliberately owns no F32 weight projection. It reads each INT8
//! value and its F32 group scale inside the accumulation loop. Router and
//! block integration remain on the frozen F32 path until later M6.3 tasks.

use clr_core::RuntimeError;

const INPUT_GROUP_SIZE: usize = 128;

/// One row-major INT8 projection with F32 scales for input groups of 128.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Group128Projection<'a> {
    values: &'a [i8],
    scales: &'a [f32],
    rows: usize,
    columns: usize,
}

impl<'a> Group128Projection<'a> {
    /// Creates a validated direct-consumption projection view.
    pub(crate) fn new(
        values: &'a [i8],
        scales: &'a [f32],
        rows: usize,
        columns: usize,
    ) -> Result<Self, RuntimeError> {
        if rows == 0 || columns == 0 || columns % INPUT_GROUP_SIZE != 0 {
            return Err(contract_error(
                "group-128 projection",
                "rows and columns must be non-zero and columns must be divisible by 128",
            ));
        }
        let expected_values = rows.checked_mul(columns).ok_or_else(|| {
            contract_error("group-128 projection", "quantized value length overflowed")
        })?;
        let expected_scales = rows
            .checked_mul(columns / INPUT_GROUP_SIZE)
            .ok_or_else(|| contract_error("group-128 projection", "scale length overflowed"))?;
        if values.len() != expected_values || scales.len() != expected_scales {
            return Err(contract_error(
                "group-128 projection",
                "quantized value or scale length does not match the projection shape",
            ));
        }
        if values.contains(&i8::MIN) {
            return Err(contract_error(
                "group-128 projection",
                "INT8 value -128 is reserved and must not be present",
            ));
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
        {
            return Err(contract_error(
                "group-128 projection",
                "scales must be finite and non-negative",
            ));
        }
        Ok(Self {
            values,
            scales,
            rows,
            columns,
        })
    }

    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        if input.len() != self.columns || input.iter().any(|value| !value.is_finite()) {
            return Err(contract_error(
                "group-128 projection input",
                "input must be finite and match the projection column count",
            ));
        }
        let groups_per_row = self.columns / INPUT_GROUP_SIZE;
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut sum = 0.0;
            let row_start = row * self.columns;
            let scale_start = row * groups_per_row;
            for group in 0..groups_per_row {
                let scale = self.scales[scale_start + group];
                let value_start = row_start + group * INPUT_GROUP_SIZE;
                for column_in_group in 0..INPUT_GROUP_SIZE {
                    let column = group * INPUT_GROUP_SIZE + column_in_group;
                    let quantized = self.values[value_start + column_in_group];
                    sum += input[column] * (f32::from(quantized) * scale);
                }
            }
            output.push(sum);
        }
        Ok(output)
    }

    const fn value_count(&self) -> usize {
        self.rows * self.columns
    }
}

/// Direct-consumption group-128 weights for one Layer-0 routed expert.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Layer0QuantizedExpert<'a> {
    gate: Group128Projection<'a>,
    up: Group128Projection<'a>,
    down: Group128Projection<'a>,
}

impl<'a> Layer0QuantizedExpert<'a> {
    /// Creates one Layer-0 expert whose three projections are direct INT8 views.
    pub(crate) fn new(
        gate: Group128Projection<'a>,
        up: Group128Projection<'a>,
        down: Group128Projection<'a>,
    ) -> Result<Self, RuntimeError> {
        if gate.columns != up.columns || gate.rows != up.rows || down.columns != gate.rows {
            return Err(contract_error(
                "Layer-0 group-128 expert",
                "gate/up/down projection shapes are incompatible",
            ));
        }
        Ok(Self { gate, up, down })
    }

    /// Evaluates the expert without materializing a F32 weight projection.
    pub(crate) fn evaluate(
        &self,
        input: &[f32],
    ) -> Result<DirectQuantizedExpertOutput, RuntimeError> {
        let gate_projection = self.gate.apply(input)?;
        let up_projection = self.up.apply(input)?;
        let activated_product = gate_projection
            .iter()
            .zip(&up_projection)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let down_projection = self.down.apply(&activated_product)?;
        Ok(DirectQuantizedExpertOutput {
            gate_projection,
            up_projection,
            activated_product,
            down_projection,
            direct_consumption: DirectConsumptionMetrics {
                quantized_weight_values_read: self.gate.value_count()
                    + self.up.value_count()
                    + self.down.value_count(),
                f32_weight_values_materialized: 0,
            },
        })
    }
}

/// Checkpoints and direct-consumption evidence from one expert evaluation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DirectQuantizedExpertOutput {
    pub(crate) gate_projection: Vec<f32>,
    pub(crate) up_projection: Vec<f32>,
    pub(crate) activated_product: Vec<f32>,
    pub(crate) down_projection: Vec<f32>,
    pub(crate) direct_consumption: DirectConsumptionMetrics,
}

/// Weight-materialization accounting for the candidate path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectConsumptionMetrics {
    pub(crate) quantized_weight_values_read: usize,
    pub(crate) f32_weight_values_materialized: usize,
}

fn contract_error(context: &'static str, reason: &'static str) -> RuntimeError {
    RuntimeError::BackendContractViolation { context, reason }
}

#[cfg(test)]
mod tests {
    use super::{Group128Projection, Layer0QuantizedExpert};

    fn projection_values(rows: usize, columns: usize) -> Vec<i8> {
        (0..rows * columns)
            .map(|index| i8::try_from(index % 15).expect("small deterministic value") - 7)
            .collect()
    }

    fn reference_projection(input: &[f32], values: &[i8], scales: &[f32], rows: usize) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                (0..input.len())
                    .map(|column| {
                        let group = column / 128;
                        input[column]
                            * (f32::from(values[row * input.len() + column])
                                * scales[row * (input.len() / 128) + group])
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn group128_projection_matches_reference_without_f32_weight_storage() {
        let input = (0..128)
            .map(|index| f32::from(u8::try_from(index).expect("small test index")) / 64.0 - 1.0)
            .collect::<Vec<_>>();
        let values = projection_values(2, 128);
        let scales = vec![0.125, 0.25];
        let projection = Group128Projection::new(&values, &scales, 2, 128).expect("projection");

        assert_eq!(
            projection.apply(&input).expect("direct projection"),
            reference_projection(&input, &values, &scales, 2)
        );
    }

    #[test]
    fn layer0_expert_consumes_all_weights_directly_and_materializes_none() {
        let input = vec![0.5; 128];
        let gate_values = projection_values(128, 128);
        let up_values = projection_values(128, 128);
        let down_values = projection_values(128, 128);
        let gate_scales = vec![0.125; 128];
        let up_scales = vec![0.25; 128];
        let down_scales = vec![0.5; 128];
        let gate = Group128Projection::new(&gate_values, &gate_scales, 128, 128).expect("gate");
        let up = Group128Projection::new(&up_values, &up_scales, 128, 128).expect("up");
        let down = Group128Projection::new(&down_values, &down_scales, 128, 128).expect("down");
        let expert = Layer0QuantizedExpert::new(gate, up, down).expect("expert");

        let output = expert.evaluate(&input).expect("direct expert");
        assert_eq!(output.gate_projection.len(), 128);
        assert_eq!(output.up_projection.len(), 128);
        assert_eq!(output.activated_product.len(), 128);
        assert_eq!(output.down_projection.len(), 128);
        assert_eq!(output.direct_consumption.f32_weight_values_materialized, 0);
        assert_eq!(
            output.direct_consumption.quantized_weight_values_read,
            49_152
        );
    }

    #[test]
    fn group128_projection_rejects_invalid_layout_and_non_finite_inputs() {
        assert!(Group128Projection::new(&[0; 127], &[1.0], 1, 127).is_err());
        assert!(Group128Projection::new(&[0; 128], &[f32::NAN], 1, 128).is_err());
        assert!(Group128Projection::new(&[i8::MIN; 128], &[1.0], 1, 128).is_err());

        let projection = Group128Projection::new(&[0; 128], &[1.0], 1, 128).expect("projection");
        assert!(projection.apply(&[f32::INFINITY; 128]).is_err());
    }
}
