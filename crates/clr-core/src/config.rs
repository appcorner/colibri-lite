use crate::{DataType, RuntimeError};

/// Unvalidated architecture-neutral dimensions used to construct a model
/// configuration.
///
/// Expert counts, routing policy, rotary-embedding settings, and other
/// Qwen-specific fields intentionally do not belong in this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfigSpec {
    /// Number of token embeddings and output logits.
    pub vocabulary_size: usize,
    /// Width of the decoder hidden state.
    pub hidden_size: usize,
    /// Number of decoder layers.
    pub layer_count: usize,
    /// Number of query attention heads.
    pub attention_head_count: usize,
    /// Number of key/value attention heads.
    pub key_value_head_count: usize,
    /// Width of the decoder feed-forward intermediate state.
    pub intermediate_size: usize,
    /// Maximum number of token positions accepted by the model.
    pub max_sequence_length: usize,
    /// Dense element type recorded by model metadata.
    pub data_type: DataType,
}

/// Validated architecture-neutral dimensions for a decoder model.
///
/// Qwen-specific configuration belongs to `clr-qwen3-moe`. This contract owns
/// only dimensions shared by the tensor/runtime boundary used in M1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfig {
    vocabulary_size: usize,
    hidden_size: usize,
    layer_count: usize,
    attention_head_count: usize,
    key_value_head_count: usize,
    intermediate_size: usize,
    max_sequence_length: usize,
    data_type: DataType,
}

impl ModelConfig {
    /// Validates and creates a model configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidModelConfig`] when a required dimension
    /// is zero, the hidden size is not divisible by the query-head count, or
    /// the query-head count is not divisible by the key/value-head count.
    pub fn new(spec: ModelConfigSpec) -> Result<Self, RuntimeError> {
        require_nonzero("vocabulary_size", spec.vocabulary_size)?;
        require_nonzero("hidden_size", spec.hidden_size)?;
        require_nonzero("layer_count", spec.layer_count)?;
        require_nonzero("attention_head_count", spec.attention_head_count)?;
        require_nonzero("key_value_head_count", spec.key_value_head_count)?;
        require_nonzero("intermediate_size", spec.intermediate_size)?;
        require_nonzero("max_sequence_length", spec.max_sequence_length)?;

        if spec.hidden_size % spec.attention_head_count != 0 {
            return Err(RuntimeError::InvalidModelConfig {
                field: "attention_head_count",
                reason: "must divide hidden_size evenly",
            });
        }

        if spec.attention_head_count % spec.key_value_head_count != 0 {
            return Err(RuntimeError::InvalidModelConfig {
                field: "key_value_head_count",
                reason: "must divide attention_head_count evenly",
            });
        }

        Ok(Self {
            vocabulary_size: spec.vocabulary_size,
            hidden_size: spec.hidden_size,
            layer_count: spec.layer_count,
            attention_head_count: spec.attention_head_count,
            key_value_head_count: spec.key_value_head_count,
            intermediate_size: spec.intermediate_size,
            max_sequence_length: spec.max_sequence_length,
            data_type: spec.data_type,
        })
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub const fn vocabulary_size(self) -> usize {
        self.vocabulary_size
    }

    /// Returns the decoder hidden-state width.
    #[must_use]
    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    /// Returns the decoder layer count.
    #[must_use]
    pub const fn layer_count(self) -> usize {
        self.layer_count
    }

    /// Returns the query attention-head count.
    #[must_use]
    pub const fn attention_head_count(self) -> usize {
        self.attention_head_count
    }

    /// Returns the key/value attention-head count.
    #[must_use]
    pub const fn key_value_head_count(self) -> usize {
        self.key_value_head_count
    }

    /// Returns the decoder feed-forward intermediate width.
    #[must_use]
    pub const fn intermediate_size(self) -> usize {
        self.intermediate_size
    }

    /// Returns the maximum supported token sequence length.
    #[must_use]
    pub const fn max_sequence_length(self) -> usize {
        self.max_sequence_length
    }

    /// Returns the model metadata element type.
    #[must_use]
    pub const fn data_type(self) -> DataType {
        self.data_type
    }
}

fn require_nonzero(field: &'static str, value: usize) -> Result<(), RuntimeError> {
    if value == 0 {
        return Err(RuntimeError::InvalidModelConfig {
            field,
            reason: "must be greater than zero",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_spec() -> ModelConfigSpec {
        ModelConfigSpec {
            vocabulary_size: 128,
            hidden_size: 32,
            layer_count: 2,
            attention_head_count: 4,
            key_value_head_count: 2,
            intermediate_size: 64,
            max_sequence_length: 256,
            data_type: DataType::F32,
        }
    }

    #[test]
    fn valid_configuration_exposes_dimensions() {
        let config = ModelConfig::new(valid_spec()).expect("valid configuration");

        assert_eq!(config.vocabulary_size(), 128);
        assert_eq!(config.hidden_size(), 32);
        assert_eq!(config.layer_count(), 2);
        assert_eq!(config.attention_head_count(), 4);
        assert_eq!(config.key_value_head_count(), 2);
        assert_eq!(config.intermediate_size(), 64);
        assert_eq!(config.max_sequence_length(), 256);
        assert_eq!(config.data_type(), DataType::F32);
    }

    #[test]
    fn zero_required_dimensions_are_rejected_independently() {
        let cases = [
            ("vocabulary_size", 0),
            ("hidden_size", 1),
            ("layer_count", 2),
            ("attention_head_count", 3),
            ("key_value_head_count", 4),
            ("intermediate_size", 5),
            ("max_sequence_length", 6),
        ];

        for (expected_field, zeroed_field) in cases {
            let mut spec = valid_spec();
            match zeroed_field {
                0 => spec.vocabulary_size = 0,
                1 => spec.hidden_size = 0,
                2 => spec.layer_count = 0,
                3 => spec.attention_head_count = 0,
                4 => spec.key_value_head_count = 0,
                5 => spec.intermediate_size = 0,
                6 => spec.max_sequence_length = 0,
                _ => unreachable!(),
            }

            assert_eq!(
                ModelConfig::new(spec),
                Err(RuntimeError::InvalidModelConfig {
                    field: expected_field,
                    reason: "must be greater than zero",
                })
            );
        }
    }

    #[test]
    fn hidden_size_must_be_divisible_by_attention_heads() {
        let mut spec = valid_spec();
        spec.hidden_size = 30;

        assert_eq!(
            ModelConfig::new(spec),
            Err(RuntimeError::InvalidModelConfig {
                field: "attention_head_count",
                reason: "must divide hidden_size evenly",
            })
        );
    }

    #[test]
    fn attention_heads_must_be_divisible_by_key_value_heads() {
        let mut spec = valid_spec();
        spec.attention_head_count = 4;
        spec.key_value_head_count = 3;

        assert_eq!(
            ModelConfig::new(spec),
            Err(RuntimeError::InvalidModelConfig {
                field: "key_value_head_count",
                reason: "must divide attention_head_count evenly",
            })
        );
    }

    #[test]
    fn model_config_contains_only_reviewed_architecture_neutral_fields() {
        let config = ModelConfig::new(valid_spec()).expect("valid configuration");

        let ModelConfig {
            vocabulary_size: _,
            hidden_size: _,
            layer_count: _,
            attention_head_count: _,
            key_value_head_count: _,
            intermediate_size: _,
            max_sequence_length: _,
            data_type: _,
        } = config;
    }
}
