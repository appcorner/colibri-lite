use clr_core::{DataType, ModelConfig, ModelConfigSpec, RuntimeError};

use crate::{Qwen3MoeConfig, Qwen3MoeConfigSpec};

/// Upstream Hugging Face configuration fields required by Qwen3-MoE runtime
/// mapping.
///
/// Independent boolean fields intentionally mirror the external schema rather
/// than being collapsed into runtime state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Qwen3MoeSourceConfig {
    /// Sole entry from upstream `architectures`.
    pub architecture: &'static str,
    /// Upstream `model_type` discriminator.
    pub model_type: &'static str,
    /// Tensor storage dtype declared by upstream metadata.
    pub source_data_type: DataType,
    /// Upstream `vocab_size`.
    pub vocabulary_size: usize,
    /// Upstream `hidden_size`.
    pub hidden_size: usize,
    /// Upstream dense `intermediate_size` metadata.
    pub intermediate_size: usize,
    /// Upstream `num_hidden_layers`.
    pub layer_count: usize,
    /// Upstream `num_attention_heads`.
    pub attention_head_count: usize,
    /// Upstream `num_key_value_heads`.
    pub key_value_head_count: usize,
    /// Upstream explicit `head_dim`.
    pub head_dimension: usize,
    /// Upstream model `max_position_embeddings`.
    pub max_position_embeddings: usize,
    /// Upstream `rms_norm_eps`.
    pub rms_norm_epsilon: f32,
    /// Default `RoPE` base from upstream configuration.
    pub rope_theta: f32,
    /// Whether upstream requests non-default `RoPE` scaling.
    pub rope_scaling_enabled: bool,
    /// Upstream `hidden_act` name.
    pub hidden_activation: &'static str,
    /// Whether attention projections include bias.
    pub attention_bias: bool,
    /// Upstream inference attention-dropout probability.
    pub attention_dropout: f32,
    /// Upstream `num_experts`.
    pub expert_count: usize,
    /// Upstream `num_experts_per_tok`.
    pub experts_per_token: usize,
    /// Upstream `moe_intermediate_size`.
    pub moe_intermediate_size: usize,
    /// Upstream `norm_topk_prob` routing policy.
    pub normalize_topk_probabilities: bool,
    /// Upstream sparse-layer cadence.
    pub decoder_sparse_step: usize,
    /// Decoder layer IDs forced to use dense MLPs.
    pub mlp_only_layers: &'static [usize],
    /// Whether upstream enables sliding-window attention.
    pub use_sliding_window: bool,
    /// Optional upstream sliding-window length.
    pub sliding_window: Option<usize>,
    /// Whether token embeddings and LM head share weights.
    pub tie_word_embeddings: bool,
}

/// Validated bridge from immutable source metadata to the F32 correctness
/// runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3MoeConfigMapping {
    runtime_config: Qwen3MoeConfig,
    source_data_type: DataType,
    model_max_position_count: usize,
}

impl Qwen3MoeConfigMapping {
    /// Returns the validated F32 runtime configuration.
    #[must_use]
    pub const fn runtime_config(self) -> Qwen3MoeConfig {
        self.runtime_config
    }

    /// Returns the immutable upstream tensor storage type.
    #[must_use]
    pub const fn source_data_type(self) -> DataType {
        self.source_data_type
    }

    /// Returns the model configuration's position limit.
    ///
    /// This is distinct from tokenizer metadata and session capacity.
    #[must_use]
    pub const fn model_max_position_count(self) -> usize {
        self.model_max_position_count
    }
}

impl Qwen3MoeSourceConfig {
    /// Validates supported pinned-source semantics and maps them to the F32
    /// correctness runtime.
    ///
    /// `source_data_type` remains provenance metadata. This method does not
    /// add BF16 computation; artifact decoding may convert BF16 values to F32.
    ///
    /// # Errors
    ///
    /// Returns a structured configuration or arithmetic error for an
    /// unsupported architecture feature or invalid dimension.
    pub fn map_to_f32_runtime(self) -> Result<Qwen3MoeConfigMapping, RuntimeError> {
        require_equal(
            "architectures",
            self.architecture == "Qwen3MoeForCausalLM",
            "must contain Qwen3MoeForCausalLM",
        )?;
        require_equal(
            "model_type",
            self.model_type == "qwen3_moe",
            "must be qwen3_moe",
        )?;
        require_equal(
            "torch_dtype",
            self.source_data_type == DataType::BF16,
            "pinned source storage must be bfloat16",
        )?;
        require_equal(
            "hidden_act",
            self.hidden_activation == "silu",
            "runtime supports silu only",
        )?;
        require_equal(
            "attention_bias",
            !self.attention_bias,
            "runtime does not support attention bias",
        )?;
        require_equal(
            "attention_dropout",
            self.attention_dropout == 0.0,
            "inference mapping requires zero attention dropout",
        )?;
        require_equal(
            "rope_scaling",
            !self.rope_scaling_enabled,
            "runtime does not support rope scaling",
        )?;
        require_equal(
            "decoder_sparse_step",
            self.decoder_sparse_step == 1,
            "runtime requires every decoder layer to be sparse",
        )?;
        require_equal(
            "mlp_only_layers",
            self.mlp_only_layers.is_empty(),
            "runtime does not support dense-only decoder layers",
        )?;
        require_equal(
            "use_sliding_window",
            !self.use_sliding_window && self.sliding_window.is_none(),
            "runtime does not support sliding-window attention",
        )?;
        require_equal(
            "tie_word_embeddings",
            !self.tie_word_embeddings,
            "runtime requires a separate language-model head",
        )?;

        let model = ModelConfig::new(ModelConfigSpec {
            vocabulary_size: self.vocabulary_size,
            hidden_size: self.hidden_size,
            layer_count: self.layer_count,
            attention_head_count: self.attention_head_count,
            key_value_head_count: self.key_value_head_count,
            head_dimension: self.head_dimension,
            intermediate_size: self.intermediate_size,
            max_sequence_length: self.max_position_embeddings,
            data_type: DataType::F32,
        })?;
        let runtime_config = Qwen3MoeConfig::new(Qwen3MoeConfigSpec {
            model,
            rms_norm_epsilon: self.rms_norm_epsilon,
            rope_theta: self.rope_theta,
            expert_count: self.expert_count,
            experts_per_token: self.experts_per_token,
            moe_intermediate_size: self.moe_intermediate_size,
            normalize_topk_probabilities: self.normalize_topk_probabilities,
        })?;
        Ok(Qwen3MoeConfigMapping {
            runtime_config,
            source_data_type: self.source_data_type,
            model_max_position_count: self.max_position_embeddings,
        })
    }
}

fn require_equal(
    field: &'static str,
    condition: bool,
    reason: &'static str,
) -> Result<(), RuntimeError> {
    if !condition {
        return Err(RuntimeError::InvalidModelConfig { field, reason });
    }
    Ok(())
}

/// Exact required configuration fields from the pinned Qwen3-30B-A3B source
/// revision in source manifest v1.
pub const PINNED_QWEN3_30B_A3B_CONFIG: Qwen3MoeSourceConfig = Qwen3MoeSourceConfig {
    architecture: "Qwen3MoeForCausalLM",
    model_type: "qwen3_moe",
    source_data_type: DataType::BF16,
    vocabulary_size: 151_936,
    hidden_size: 2_048,
    intermediate_size: 6_144,
    layer_count: 48,
    attention_head_count: 32,
    key_value_head_count: 4,
    head_dimension: 128,
    max_position_embeddings: 40_960,
    rms_norm_epsilon: 1.0e-6,
    rope_theta: 1_000_000.0,
    rope_scaling_enabled: false,
    hidden_activation: "silu",
    attention_bias: false,
    attention_dropout: 0.0,
    expert_count: 128,
    experts_per_token: 8,
    moe_intermediate_size: 768,
    normalize_topk_probabilities: true,
    decoder_sparse_step: 1,
    mlp_only_layers: &[],
    use_sliding_window: false,
    sliding_window: None,
    tie_word_embeddings: false,
};

#[cfg(test)]
mod tests {
    use super::*;

    type SourceMutator = fn(&mut Qwen3MoeSourceConfig);

    #[test]
    fn pinned_source_maps_explicit_head_dimension_and_separate_storage_type() {
        let mapping = PINNED_QWEN3_30B_A3B_CONFIG
            .map_to_f32_runtime()
            .expect("pinned source mapping");
        let runtime = mapping.runtime_config();

        assert_eq!(mapping.source_data_type(), DataType::BF16);
        assert_eq!(runtime.model().data_type(), DataType::F32);
        assert_eq!(runtime.model().hidden_size(), 2_048);
        assert_eq!(runtime.model().attention_head_count(), 32);
        assert_eq!(runtime.model().key_value_head_count(), 4);
        assert_eq!(runtime.model().head_dimension(), 128);
        assert_eq!(runtime.model().query_projection_width(), 4_096);
        assert_eq!(runtime.model().key_value_projection_width(), 512);
        assert_eq!(mapping.model_max_position_count(), 40_960);
        assert_eq!(runtime.model().max_sequence_length(), 40_960);
        assert_eq!(runtime.expert_count(), 128);
        assert_eq!(runtime.experts_per_token(), 8);
        assert_eq!(runtime.moe_intermediate_size(), 768);
        assert!(runtime.normalize_topk_probabilities());
    }

    #[test]
    fn unsupported_source_semantics_are_structured() {
        let cases: &[(&'static str, SourceMutator)] = &[
            ("architectures", |source| source.architecture = "Other"),
            ("model_type", |source| source.model_type = "other"),
            ("torch_dtype", |source| {
                source.source_data_type = DataType::F16;
            }),
            ("hidden_act", |source| source.hidden_activation = "gelu"),
            ("attention_bias", |source| source.attention_bias = true),
            ("attention_dropout", |source| {
                source.attention_dropout = 0.1;
            }),
            ("rope_scaling", |source| {
                source.rope_scaling_enabled = true;
            }),
            ("decoder_sparse_step", |source| {
                source.decoder_sparse_step = 2;
            }),
            ("mlp_only_layers", |source| {
                source.mlp_only_layers = &[0];
            }),
            ("use_sliding_window", |source| {
                source.use_sliding_window = true;
            }),
            ("tie_word_embeddings", |source| {
                source.tie_word_embeddings = true;
            }),
        ];

        for (field, mutate) in cases {
            let mut source = PINNED_QWEN3_30B_A3B_CONFIG;
            mutate(&mut source);
            assert!(matches!(
                source.map_to_f32_runtime(),
                Err(RuntimeError::InvalidModelConfig {
                    field: actual,
                    ..
                }) if actual == *field
            ));
        }
    }

    #[test]
    fn source_dimension_errors_preserve_generic_categories() {
        let mut zero = PINNED_QWEN3_30B_A3B_CONFIG;
        zero.head_dimension = 0;
        assert!(matches!(
            zero.map_to_f32_runtime(),
            Err(RuntimeError::InvalidModelConfig {
                field: "head_dimension",
                ..
            })
        ));

        let mut overflow = PINNED_QWEN3_30B_A3B_CONFIG;
        overflow.attention_head_count = usize::MAX;
        overflow.key_value_head_count = 1;
        overflow.head_dimension = 2;
        assert_eq!(
            overflow.map_to_f32_runtime(),
            Err(RuntimeError::ArithmeticOverflow {
                operation: "query projection width",
            })
        );
    }
}
