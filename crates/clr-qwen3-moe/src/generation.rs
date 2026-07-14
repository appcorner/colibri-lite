use clr_core::{RuntimeError, TensorView};

use crate::Qwen3MoeModel;

/// Reproducible `SplitMix64` random number generator for sampling tests/runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        let upper_24 = self.next_u64() >> 40;
        #[allow(clippy::cast_precision_loss)]
        let numerator = upper_24 as f32;
        numerator / 16_777_216.0
    }
}

/// Selects the highest-scoring token from the final logits row.
///
/// Equal scores select the lower token ID.
///
/// # Errors
///
/// Returns a rank/shape error for non-matrix or empty logits, and
/// [`RuntimeError::NonFiniteInput`] for NaN or infinite scores.
pub fn greedy_token(logits: TensorView<'_>) -> Result<usize, RuntimeError> {
    if logits.shape().rank() != 2 {
        return Err(RuntimeError::RankMismatch {
            context: "greedy logits",
            expected: 2,
            actual: logits.shape().rank(),
        });
    }
    let sequence_length = logits.shape().dimensions()[0];
    let vocabulary_size = logits.shape().dimensions()[1];
    if sequence_length == 0 || vocabulary_size == 0 {
        return Err(RuntimeError::InvalidShape {
            reason: "greedy logits require non-empty sequence and vocabulary dimensions",
        });
    }
    let row_start = (sequence_length - 1) * vocabulary_size;
    let final_row = &logits.data()[row_start..row_start + vocabulary_size];
    if let Some(index) = final_row.iter().position(|value| !value.is_finite()) {
        return Err(RuntimeError::NonFiniteInput {
            operation: "greedy decoding",
            index: row_start + index,
        });
    }
    let mut selected = 0;
    for token_id in 1..vocabulary_size {
        if final_row[token_id] > final_row[selected] {
            selected = token_id;
        }
    }
    Ok(selected)
}

/// Samples a token from the final logits row using temperature and seeded RNG.
///
/// # Errors
///
/// Returns a structured error for invalid temperature, logits shape, or
/// non-finite scores.
pub fn sample_token(
    logits: TensorView<'_>,
    temperature: f32,
    rng: &mut SeededRng,
) -> Result<usize, RuntimeError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(RuntimeError::InvalidModelConfig {
            field: "temperature",
            reason: "must be finite and greater than zero",
        });
    }
    if logits.shape().rank() != 2 {
        return Err(RuntimeError::RankMismatch {
            context: "sampling logits",
            expected: 2,
            actual: logits.shape().rank(),
        });
    }
    let sequence = logits.shape().dimensions()[0];
    let vocabulary = logits.shape().dimensions()[1];
    if sequence == 0 || vocabulary == 0 {
        return Err(RuntimeError::InvalidShape {
            reason: "sampling logits require non-empty dimensions",
        });
    }
    let start = (sequence - 1) * vocabulary;
    let row = &logits.data()[start..start + vocabulary];
    if let Some(index) = row.iter().position(|value| !value.is_finite()) {
        return Err(RuntimeError::NonFiniteInput {
            operation: "temperature sampling",
            index: start + index,
        });
    }
    let maximum = row
        .iter()
        .copied()
        .map(|value| value / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<f32> = row
        .iter()
        .map(|value| (value / temperature - maximum).exp())
        .collect();
    let sum: f32 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= sum;
    }
    let draw = rng.next_unit_f32();
    let mut cumulative = 0.0;
    for (token_id, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return Ok(token_id);
        }
    }
    Ok(vocabulary - 1)
}

impl Qwen3MoeModel {
    /// Generates token IDs greedily, recomputing the complete sequence per step.
    ///
    /// The returned vector contains only newly generated token IDs.
    ///
    /// # Errors
    ///
    /// Returns a model-forward or greedy-logit validation error. Empty prompts
    /// are rejected even when `max_new_tokens` is zero.
    pub fn generate_greedy(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, RuntimeError> {
        if prompt.is_empty() {
            return Err(RuntimeError::InvalidShape {
                reason: "greedy generation prompt must not be empty",
            });
        }
        let mut sequence = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let output = self.forward(&sequence)?;
            let token_id = greedy_token(output.logits.view())?;
            sequence.push(token_id);
            generated.push(token_id);
        }
        Ok(generated)
    }

    /// Generates token IDs with reproducible temperature sampling.
    ///
    /// # Errors
    ///
    /// Returns model-forward, temperature, or sampling validation errors.
    pub fn generate_temperature(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<usize>, RuntimeError> {
        if prompt.is_empty() {
            return Err(RuntimeError::InvalidShape {
                reason: "temperature generation prompt must not be empty",
            });
        }
        let mut rng = SeededRng::new(seed);
        let mut sequence = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let output = self.forward(&sequence)?;
            let token_id = sample_token(output.logits.view(), temperature, &mut rng)?;
            sequence.push(token_id);
            generated.push(token_id);
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use clr_core::{Tensor, TensorShape};

    use super::*;
    use crate::{Qwen3MoeModelWeightsSpec, test_fixture};

    fn fixture_model() -> Qwen3MoeModel {
        Qwen3MoeModel::new(
            test_fixture::config(),
            Qwen3MoeModelWeightsSpec {
                token_embeddings: test_fixture::token_embeddings(),
                blocks: vec![
                    test_fixture::block_weights(0),
                    test_fixture::block_weights(1),
                ],
                final_norm: test_fixture::final_norm_weight(),
                language_model_head: test_fixture::language_model_head(),
            },
        )
        .expect("frozen model")
    }

    fn tensor(shape: &[usize], values: &[f32]) -> Tensor {
        Tensor::new(TensorShape::new(shape.to_vec()), values.to_vec()).expect("test logits")
    }

    #[test]
    fn greedy_uses_final_row_and_lower_id_tie_break() {
        let logits = tensor(&[2, 4], &[9.0, 0.0, 0.0, 0.0, 0.5, 2.0, 2.0, 1.0]);

        assert_eq!(greedy_token(logits.view()), Ok(1));
    }

    #[test]
    fn greedy_rejects_rank_empty_and_non_finite_logits() {
        assert!(matches!(
            greedy_token(tensor(&[3], &[1.0, 2.0, 3.0]).view()),
            Err(RuntimeError::RankMismatch { .. })
        ));
        assert!(matches!(
            greedy_token(tensor(&[0, 4], &[]).view()),
            Err(RuntimeError::InvalidShape { .. })
        ));
        assert!(matches!(
            greedy_token(tensor(&[1, 2], &[0.0, f32::NAN]).view()),
            Err(RuntimeError::NonFiniteInput {
                operation: "greedy decoding",
                index: 1,
            })
        ));
    }

    #[test]
    fn frozen_prompt_first_greedy_token_matches_oracle() {
        let model = fixture_model();

        let generated = model
            .generate_greedy(&test_fixture::token_ids(), 1)
            .expect("greedy generation");

        assert_eq!(generated, [10]);
    }

    #[test]
    fn greedy_generation_is_repeatable_and_rejects_empty_prompt() {
        let model = fixture_model();
        let prompt = test_fixture::token_ids();

        assert_eq!(
            model.generate_greedy(&prompt, 3),
            model.generate_greedy(&prompt, 3)
        );
        assert!(matches!(
            model.generate_greedy(&[], 0),
            Err(RuntimeError::InvalidShape { .. })
        ));
    }

    #[test]
    fn splitmix64_has_a_stable_seeded_sequence() {
        let mut rng = SeededRng::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn temperature_sampling_is_seeded_and_validated() {
        let model = fixture_model();
        let prompt = test_fixture::token_ids();
        let first = model
            .generate_temperature(&prompt, 4, 0.8, 42)
            .expect("first sampled sequence");
        let second = model
            .generate_temperature(&prompt, 4, 0.8, 42)
            .expect("second sampled sequence");

        assert_eq!(first, second);
        assert!(matches!(
            model.generate_temperature(&prompt, 1, 0.0, 42),
            Err(RuntimeError::InvalidModelConfig {
                field: "temperature",
                ..
            })
        ));
    }
}
