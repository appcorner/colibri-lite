use clr_core::{DataType, ModelConfig, RuntimeError};

/// Unvalidated Qwen3-MoE-specific settings layered over generic model values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3MoeConfigSpec {
    /// Validated architecture-neutral decoder dimensions.
    pub model: ModelConfig,
    /// Epsilon added by every RMS normalization.
    pub rms_norm_epsilon: f32,
    /// Base frequency used by default rotary position embeddings.
    pub rope_theta: f32,
    /// Number of routed experts per sparse layer.
    pub expert_count: usize,
    /// Number of experts selected for each token.
    pub experts_per_token: usize,
    /// Hidden width inside each routed expert.
    pub moe_intermediate_size: usize,
    /// Whether selected router probabilities are renormalized to sum to one.
    pub normalize_topk_probabilities: bool,
}

/// Validated configuration for the M1 Qwen3-MoE correctness path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3MoeConfig {
    model: ModelConfig,
    rms_norm_epsilon: f32,
    rope_theta: f32,
    expert_count: usize,
    experts_per_token: usize,
    moe_intermediate_size: usize,
    normalize_topk_probabilities: bool,
}

impl Qwen3MoeConfig {
    /// Validates and creates Qwen3-MoE-specific configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidModelConfig`] when the correctness path
    /// is not F32, a Qwen dimension is invalid, or a floating-point setting is
    /// non-finite/non-positive.
    pub fn new(spec: Qwen3MoeConfigSpec) -> Result<Self, RuntimeError> {
        if spec.model.data_type() != DataType::F32 {
            return Err(invalid("data_type", "M1 computation requires f32"));
        }
        if spec.model.hidden_size() / spec.model.attention_head_count() % 2 != 0 {
            return Err(invalid(
                "head_dimension",
                "must be even for rotary embeddings",
            ));
        }
        require_positive_finite("rms_norm_epsilon", spec.rms_norm_epsilon)?;
        require_positive_finite("rope_theta", spec.rope_theta)?;
        require_nonzero("expert_count", spec.expert_count)?;
        require_nonzero("experts_per_token", spec.experts_per_token)?;
        require_nonzero("moe_intermediate_size", spec.moe_intermediate_size)?;
        if spec.experts_per_token > spec.expert_count {
            return Err(invalid("experts_per_token", "must not exceed expert_count"));
        }

        Ok(Self {
            model: spec.model,
            rms_norm_epsilon: spec.rms_norm_epsilon,
            rope_theta: spec.rope_theta,
            expert_count: spec.expert_count,
            experts_per_token: spec.experts_per_token,
            moe_intermediate_size: spec.moe_intermediate_size,
            normalize_topk_probabilities: spec.normalize_topk_probabilities,
        })
    }

    /// Returns the generic decoder configuration.
    #[must_use]
    pub const fn model(self) -> ModelConfig {
        self.model
    }

    /// Returns the width of one query/key/value head.
    #[must_use]
    pub const fn head_dimension(self) -> usize {
        self.model.hidden_size() / self.model.attention_head_count()
    }

    /// Returns the number of query heads sharing each key/value head.
    #[must_use]
    pub const fn key_value_group_count(self) -> usize {
        self.model.attention_head_count() / self.model.key_value_head_count()
    }

    /// Returns the RMS normalization epsilon.
    #[must_use]
    pub const fn rms_norm_epsilon(self) -> f32 {
        self.rms_norm_epsilon
    }

    /// Returns the default rotary-embedding base frequency.
    #[must_use]
    pub const fn rope_theta(self) -> f32 {
        self.rope_theta
    }

    /// Returns the number of experts.
    #[must_use]
    pub const fn expert_count(self) -> usize {
        self.expert_count
    }

    /// Returns the number of selected experts per token.
    #[must_use]
    pub const fn experts_per_token(self) -> usize {
        self.experts_per_token
    }

    /// Returns each expert's intermediate width.
    #[must_use]
    pub const fn moe_intermediate_size(self) -> usize {
        self.moe_intermediate_size
    }

    /// Returns whether selected routing probabilities are renormalized.
    #[must_use]
    pub const fn normalize_topk_probabilities(self) -> bool {
        self.normalize_topk_probabilities
    }
}

fn require_nonzero(field: &'static str, value: usize) -> Result<(), RuntimeError> {
    if value == 0 {
        return Err(invalid(field, "must be greater than zero"));
    }
    Ok(())
}

fn require_positive_finite(field: &'static str, value: f32) -> Result<(), RuntimeError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(field, "must be finite and greater than zero"));
    }
    Ok(())
}

const fn invalid(field: &'static str, reason: &'static str) -> RuntimeError {
    RuntimeError::InvalidModelConfig { field, reason }
}

#[cfg(test)]
mod tests {
    use clr_core::ModelConfigSpec;

    use super::*;

    #[allow(dead_code)]
    mod frozen_fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../python/reference/fixtures/tiny-qwen3-moe/rust-config.rs"
        ));
    }

    fn generic_config(data_type: DataType, hidden_size: usize) -> ModelConfig {
        ModelConfig::new(ModelConfigSpec {
            vocabulary_size: frozen_fixture::VOCABULARY_SIZE,
            hidden_size,
            layer_count: frozen_fixture::LAYER_COUNT,
            attention_head_count: frozen_fixture::ATTENTION_HEAD_COUNT,
            key_value_head_count: frozen_fixture::KEY_VALUE_HEAD_COUNT,
            intermediate_size: frozen_fixture::INTERMEDIATE_SIZE,
            max_sequence_length: frozen_fixture::MAX_SEQUENCE_LENGTH,
            data_type,
        })
        .expect("valid generic config")
    }

    fn valid_spec() -> Qwen3MoeConfigSpec {
        Qwen3MoeConfigSpec {
            model: generic_config(DataType::F32, frozen_fixture::HIDDEN_SIZE),
            rms_norm_epsilon: frozen_fixture::RMS_NORM_EPSILON,
            rope_theta: frozen_fixture::ROPE_THETA,
            expert_count: frozen_fixture::EXPERT_COUNT,
            experts_per_token: frozen_fixture::EXPERTS_PER_TOKEN,
            moe_intermediate_size: frozen_fixture::MOE_INTERMEDIATE_SIZE,
            normalize_topk_probabilities: frozen_fixture::NORMALIZE_TOPK_PROBABILITIES,
        }
    }

    #[test]
    fn maps_the_frozen_tiny_qwen_configuration() {
        let config = Qwen3MoeConfig::new(valid_spec()).expect("valid Qwen config");

        assert_eq!(config.model().hidden_size(), 16);
        assert_eq!(config.head_dimension(), 4);
        assert_eq!(config.key_value_group_count(), 2);
        assert_eq!(config.expert_count(), 4);
        assert_eq!(config.experts_per_token(), 2);
        assert_eq!(config.moe_intermediate_size(), 24);
        assert!(!config.normalize_topk_probabilities());
        assert_eq!(config.rope_theta().to_bits(), 10_000.0_f32.to_bits());
    }

    #[test]
    fn rejects_non_f32_and_odd_head_dimensions() {
        let mut spec = valid_spec();
        spec.model = generic_config(DataType::BF16, 16);
        assert_eq!(
            Qwen3MoeConfig::new(spec),
            Err(invalid("data_type", "M1 computation requires f32"))
        );

        let mut spec = valid_spec();
        spec.model = generic_config(DataType::F32, 12);
        assert_eq!(
            Qwen3MoeConfig::new(spec),
            Err(invalid(
                "head_dimension",
                "must be even for rotary embeddings"
            ))
        );
    }

    #[test]
    fn rejects_invalid_qwen_specific_values() {
        let mut cases = Vec::new();

        let mut spec = valid_spec();
        spec.rms_norm_epsilon = f32::NAN;
        cases.push((
            spec,
            invalid("rms_norm_epsilon", "must be finite and greater than zero"),
        ));

        let mut spec = valid_spec();
        spec.rope_theta = 0.0;
        cases.push((
            spec,
            invalid("rope_theta", "must be finite and greater than zero"),
        ));

        let mut spec = valid_spec();
        spec.expert_count = 0;
        cases.push((spec, invalid("expert_count", "must be greater than zero")));

        let mut spec = valid_spec();
        spec.experts_per_token = 0;
        cases.push((
            spec,
            invalid("experts_per_token", "must be greater than zero"),
        ));

        let mut spec = valid_spec();
        spec.moe_intermediate_size = 0;
        cases.push((
            spec,
            invalid("moe_intermediate_size", "must be greater than zero"),
        ));

        let mut spec = valid_spec();
        spec.experts_per_token = 5;
        cases.push((
            spec,
            invalid("experts_per_token", "must not exceed expert_count"),
        ));

        for (spec, expected) in cases {
            assert_eq!(Qwen3MoeConfig::new(spec), Err(expected));
        }
    }
}
