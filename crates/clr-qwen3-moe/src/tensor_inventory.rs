use std::{collections::HashSet, fmt};

use clr_core::{DataType, RuntimeError, TensorShape};

use crate::{Qwen3MoeConfig, Qwen3MoeSourceConfig};

/// Number of Safetensors shards in the pinned Qwen3-30B-A3B source snapshot.
pub const PINNED_QWEN3_30B_A3B_SHARD_COUNT: usize = 16;

/// Safetensors-index metadata required to validate one source tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeTensorMetadata {
    name: String,
    data_type: DataType,
    shape: TensorShape,
    shard_index: usize,
}

impl Qwen3MoeTensorMetadata {
    /// Creates source metadata without reading or decoding a tensor payload.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        data_type: DataType,
        shape: TensorShape,
        shard_index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            data_type,
            shape,
            shard_index,
        }
    }

    /// Returns the exact Safetensors tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the source storage type.
    #[must_use]
    pub const fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the indexed source shape.
    #[must_use]
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the zero-based source shard index.
    #[must_use]
    pub const fn shard_index(&self) -> usize {
        self.shard_index
    }
}

/// Numerical role assigned to one canonical Qwen3-MoE source tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen3MoeTensorRole {
    /// Token embedding table.
    TokenEmbedding,
    /// Final decoder RMS-normalization weight.
    FinalNorm,
    /// Separate language-model output projection.
    LanguageModelHead,
    /// Pre-attention RMS-normalization weight.
    InputNorm { layer: usize },
    /// Pre-MoE RMS-normalization weight.
    PostAttentionNorm { layer: usize },
    /// Query projection weight.
    QueryProjection { layer: usize },
    /// Key projection weight.
    KeyProjection { layer: usize },
    /// Value projection weight.
    ValueProjection { layer: usize },
    /// Attention output projection weight.
    OutputProjection { layer: usize },
    /// Per-head query RMS-normalization weight.
    QueryNorm { layer: usize },
    /// Per-head key RMS-normalization weight.
    KeyNorm { layer: usize },
    /// Sparse expert router weight.
    Router { layer: usize },
    /// Gated-activation projection for one routed expert.
    ExpertGate { layer: usize, expert: usize },
    /// Up projection for one routed expert.
    ExpertUp { layer: usize, expert: usize },
    /// Down projection for one routed expert.
    ExpertDown { layer: usize, expert: usize },
}

/// Validated source metadata paired with its runtime tensor role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeMappedTensor {
    metadata: Qwen3MoeTensorMetadata,
    role: Qwen3MoeTensorRole,
}

impl Qwen3MoeMappedTensor {
    /// Returns the validated source metadata.
    #[must_use]
    pub const fn metadata(&self) -> &Qwen3MoeTensorMetadata {
        &self.metadata
    }

    /// Returns the classified numerical role.
    #[must_use]
    pub const fn role(&self) -> Qwen3MoeTensorRole {
        self.role
    }
}

/// Complete, duplicate-free Qwen3-MoE tensor inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeTensorInventory {
    tensors: Box<[Qwen3MoeMappedTensor]>,
}

impl Qwen3MoeTensorInventory {
    /// Returns every classified tensor in supplied index order.
    #[must_use]
    pub const fn tensors(&self) -> &[Qwen3MoeMappedTensor] {
        &self.tensors
    }

    /// Returns the number of classified tensors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether the inventory contains no tensors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Structured failures from Qwen3-MoE source inventory validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen3MoeTensorInventoryError {
    /// The source configuration is invalid or unsupported.
    SourceConfig(RuntimeError),
    /// Checked expected-count arithmetic overflowed.
    ArithmeticOverflow { operation: &'static str },
    /// A tensor name occurs more than once.
    DuplicateTensor { name: String },
    /// A required canonical tensor is absent.
    MissingTensor { name: String },
    /// A tensor does not match the supported naming grammar.
    UnknownTensor { name: String },
    /// A parsed decoder layer is outside the configured range.
    LayerOutOfRange {
        name: String,
        layer: usize,
        layer_count: usize,
    },
    /// A parsed routed expert is outside the configured range.
    ExpertOutOfRange {
        name: String,
        expert: usize,
        expert_count: usize,
    },
    /// A tensor references a shard outside the source inventory.
    ShardIndexOutOfRange {
        name: String,
        shard_index: usize,
        shard_count: usize,
    },
    /// Source storage metadata differs from the pinned dtype.
    DataTypeMismatch {
        name: String,
        expected: DataType,
        actual: DataType,
    },
    /// A source tensor has the wrong rank.
    RankMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// A source tensor has the wrong dimensions.
    ShapeMismatch {
        name: String,
        expected: Box<[usize]>,
        actual: Box<[usize]>,
    },
}

impl fmt::Display for Qwen3MoeTensorInventoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceConfig(error) => write!(formatter, "invalid source configuration: {error}"),
            Self::ArithmeticOverflow { operation } => {
                write!(
                    formatter,
                    "arithmetic overflow while calculating {operation}"
                )
            }
            Self::DuplicateTensor { name } => write!(formatter, "duplicate tensor '{name}'"),
            Self::MissingTensor { name } => write!(formatter, "missing required tensor '{name}'"),
            Self::UnknownTensor { name } => write!(formatter, "unknown tensor '{name}'"),
            Self::LayerOutOfRange {
                name,
                layer,
                layer_count,
            } => write!(
                formatter,
                "tensor '{name}' uses layer {layer}, outside 0..{layer_count}"
            ),
            Self::ExpertOutOfRange {
                name,
                expert,
                expert_count,
            } => write!(
                formatter,
                "tensor '{name}' uses expert {expert}, outside 0..{expert_count}"
            ),
            Self::ShardIndexOutOfRange {
                name,
                shard_index,
                shard_count,
            } => write!(
                formatter,
                "tensor '{name}' uses shard {shard_index}, outside 0..{shard_count}"
            ),
            Self::DataTypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor '{name}' dtype mismatch: expected {expected}, got {actual}"
            ),
            Self::RankMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor '{name}' rank mismatch: expected {expected}, got {actual}"
            ),
            Self::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor '{name}' shape mismatch: expected {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl std::error::Error for Qwen3MoeTensorInventoryError {}

impl From<RuntimeError> for Qwen3MoeTensorInventoryError {
    fn from(error: RuntimeError) -> Self {
        Self::SourceConfig(error)
    }
}

/// Classifies and validates a complete Qwen3-MoE Safetensors inventory.
///
/// Shapes are derived exclusively from the validated source configuration.
/// The function reads metadata only and neither opens shards nor decodes tensor
/// payloads. Unknown tensors are errors rather than being silently ignored.
///
/// # Errors
///
/// Returns a structured error for invalid source configuration, checked
/// arithmetic overflow, incomplete coverage, duplicate/unknown names,
/// out-of-range layer/expert/shard indices, or dtype/rank/shape mismatches.
pub fn validate_qwen3_moe_tensor_inventory(
    source: Qwen3MoeSourceConfig,
    shard_count: usize,
    tensors: &[Qwen3MoeTensorMetadata],
) -> Result<Qwen3MoeTensorInventory, Qwen3MoeTensorInventoryError> {
    let mapping = source.map_to_f32_runtime()?;
    let config = mapping.runtime_config();
    let expected_count = expected_tensor_count(config)?;
    let mut names = HashSet::with_capacity(tensors.len());
    let mut mapped = Vec::with_capacity(tensors.len());

    for tensor in tensors {
        if !names.insert(tensor.name()) {
            return Err(Qwen3MoeTensorInventoryError::DuplicateTensor {
                name: tensor.name.clone(),
            });
        }
        if tensor.shard_index >= shard_count {
            return Err(Qwen3MoeTensorInventoryError::ShardIndexOutOfRange {
                name: tensor.name.clone(),
                shard_index: tensor.shard_index,
                shard_count,
            });
        }

        let role = parse_role(tensor.name(), config)?;
        if tensor.data_type != mapping.source_data_type() {
            return Err(Qwen3MoeTensorInventoryError::DataTypeMismatch {
                name: tensor.name.clone(),
                expected: mapping.source_data_type(),
                actual: tensor.data_type,
            });
        }
        let expected_shape = expected_shape(role, config);
        if tensor.shape.rank() != expected_shape.rank() {
            return Err(Qwen3MoeTensorInventoryError::RankMismatch {
                name: tensor.name.clone(),
                expected: expected_shape.rank(),
                actual: tensor.shape.rank(),
            });
        }
        if tensor.shape != expected_shape {
            return Err(Qwen3MoeTensorInventoryError::ShapeMismatch {
                name: tensor.name.clone(),
                expected: expected_shape.dimensions().into(),
                actual: tensor.shape.dimensions().into(),
            });
        }
        mapped.push(Qwen3MoeMappedTensor {
            metadata: tensor.clone(),
            role,
        });
    }

    for_each_expected_name(config, |name| {
        if names.contains(name.as_str()) {
            Ok(())
        } else {
            Err(Qwen3MoeTensorInventoryError::MissingTensor { name })
        }
    })?;

    debug_assert_eq!(mapped.len(), expected_count);
    Ok(Qwen3MoeTensorInventory {
        tensors: mapped.into_boxed_slice(),
    })
}

fn expected_tensor_count(config: Qwen3MoeConfig) -> Result<usize, Qwen3MoeTensorInventoryError> {
    let expert_tensors = config.expert_count().checked_mul(3).ok_or(
        Qwen3MoeTensorInventoryError::ArithmeticOverflow {
            operation: "per-layer expert tensor count",
        },
    )?;
    let per_layer = 9_usize.checked_add(expert_tensors).ok_or(
        Qwen3MoeTensorInventoryError::ArithmeticOverflow {
            operation: "per-layer tensor count",
        },
    )?;
    config
        .model()
        .layer_count()
        .checked_mul(per_layer)
        .and_then(|count| count.checked_add(3))
        .ok_or(Qwen3MoeTensorInventoryError::ArithmeticOverflow {
            operation: "complete tensor count",
        })
}

fn parse_role(
    name: &str,
    config: Qwen3MoeConfig,
) -> Result<Qwen3MoeTensorRole, Qwen3MoeTensorInventoryError> {
    match name {
        "model.embed_tokens.weight" => return Ok(Qwen3MoeTensorRole::TokenEmbedding),
        "model.norm.weight" => return Ok(Qwen3MoeTensorRole::FinalNorm),
        "lm_head.weight" => return Ok(Qwen3MoeTensorRole::LanguageModelHead),
        _ => {}
    }

    let parts: Vec<_> = name.split('.').collect();
    if parts.len() < 5 || parts[0] != "model" || parts[1] != "layers" {
        return Err(unknown(name));
    }
    let Some(layer) = parts[2].parse::<usize>().ok() else {
        return Err(unknown(name));
    };
    if layer >= config.model().layer_count() {
        return Err(Qwen3MoeTensorInventoryError::LayerOutOfRange {
            name: name.to_owned(),
            layer,
            layer_count: config.model().layer_count(),
        });
    }

    let role = match parts.as_slice() {
        ["model", "layers", _, "input_layernorm", "weight"] => {
            Qwen3MoeTensorRole::InputNorm { layer }
        }
        ["model", "layers", _, "post_attention_layernorm", "weight"] => {
            Qwen3MoeTensorRole::PostAttentionNorm { layer }
        }
        ["model", "layers", _, "self_attn", "q_proj", "weight"] => {
            Qwen3MoeTensorRole::QueryProjection { layer }
        }
        ["model", "layers", _, "self_attn", "k_proj", "weight"] => {
            Qwen3MoeTensorRole::KeyProjection { layer }
        }
        ["model", "layers", _, "self_attn", "v_proj", "weight"] => {
            Qwen3MoeTensorRole::ValueProjection { layer }
        }
        ["model", "layers", _, "self_attn", "o_proj", "weight"] => {
            Qwen3MoeTensorRole::OutputProjection { layer }
        }
        ["model", "layers", _, "self_attn", "q_norm", "weight"] => {
            Qwen3MoeTensorRole::QueryNorm { layer }
        }
        ["model", "layers", _, "self_attn", "k_norm", "weight"] => {
            Qwen3MoeTensorRole::KeyNorm { layer }
        }
        ["model", "layers", _, "mlp", "gate", "weight"] => Qwen3MoeTensorRole::Router { layer },
        [
            "model",
            "layers",
            _,
            "mlp",
            "experts",
            expert,
            projection,
            "weight",
        ] => {
            let Some(expert) = expert.parse::<usize>().ok() else {
                return Err(unknown(name));
            };
            if expert >= config.expert_count() {
                return Err(Qwen3MoeTensorInventoryError::ExpertOutOfRange {
                    name: name.to_owned(),
                    expert,
                    expert_count: config.expert_count(),
                });
            }
            match *projection {
                "gate_proj" => Qwen3MoeTensorRole::ExpertGate { layer, expert },
                "up_proj" => Qwen3MoeTensorRole::ExpertUp { layer, expert },
                "down_proj" => Qwen3MoeTensorRole::ExpertDown { layer, expert },
                _ => return Err(unknown(name)),
            }
        }
        _ => return Err(unknown(name)),
    };
    Ok(role)
}

fn expected_shape(role: Qwen3MoeTensorRole, config: Qwen3MoeConfig) -> TensorShape {
    let model = config.model();
    let hidden = model.hidden_size();
    let query = model.query_projection_width();
    let key_value = model.key_value_projection_width();
    let head = model.head_dimension();
    let expert_hidden = config.moe_intermediate_size();

    match role {
        Qwen3MoeTensorRole::TokenEmbedding | Qwen3MoeTensorRole::LanguageModelHead => {
            TensorShape::new([model.vocabulary_size(), hidden])
        }
        Qwen3MoeTensorRole::FinalNorm
        | Qwen3MoeTensorRole::InputNorm { .. }
        | Qwen3MoeTensorRole::PostAttentionNorm { .. } => TensorShape::new([hidden]),
        Qwen3MoeTensorRole::QueryProjection { .. } => TensorShape::new([query, hidden]),
        Qwen3MoeTensorRole::KeyProjection { .. } | Qwen3MoeTensorRole::ValueProjection { .. } => {
            TensorShape::new([key_value, hidden])
        }
        Qwen3MoeTensorRole::OutputProjection { .. } => TensorShape::new([hidden, query]),
        Qwen3MoeTensorRole::QueryNorm { .. } | Qwen3MoeTensorRole::KeyNorm { .. } => {
            TensorShape::new([head])
        }
        Qwen3MoeTensorRole::Router { .. } => TensorShape::new([config.expert_count(), hidden]),
        Qwen3MoeTensorRole::ExpertGate { .. } | Qwen3MoeTensorRole::ExpertUp { .. } => {
            TensorShape::new([expert_hidden, hidden])
        }
        Qwen3MoeTensorRole::ExpertDown { .. } => TensorShape::new([hidden, expert_hidden]),
    }
}

fn for_each_expected_name(
    config: Qwen3MoeConfig,
    mut visit: impl FnMut(String) -> Result<(), Qwen3MoeTensorInventoryError>,
) -> Result<(), Qwen3MoeTensorInventoryError> {
    visit("model.embed_tokens.weight".to_owned())?;
    visit("model.norm.weight".to_owned())?;
    visit("lm_head.weight".to_owned())?;

    for layer in 0..config.model().layer_count() {
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate.weight",
        ] {
            visit(format!("model.layers.{layer}.{suffix}"))?;
        }
        for expert in 0..config.expert_count() {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                visit(format!(
                    "model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                ))?;
            }
        }
    }
    Ok(())
}

fn unknown(name: &str) -> Qwen3MoeTensorInventoryError {
    Qwen3MoeTensorInventoryError::UnknownTensor {
        name: name.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::PINNED_QWEN3_30B_A3B_CONFIG;

    fn valid_inventory() -> Vec<Qwen3MoeTensorMetadata> {
        let mapping = PINNED_QWEN3_30B_A3B_CONFIG
            .map_to_f32_runtime()
            .expect("pinned mapping");
        let config = mapping.runtime_config();
        let mut tensors = Vec::with_capacity(expected_tensor_count(config).expect("count"));
        for_each_expected_name(config, |name| {
            let role = parse_role(&name, config).expect("generated canonical role");
            let shard_index = tensors.len() % PINNED_QWEN3_30B_A3B_SHARD_COUNT;
            tensors.push(Qwen3MoeTensorMetadata::new(
                name,
                mapping.source_data_type(),
                expected_shape(role, config),
                shard_index,
            ));
            Ok(())
        })
        .expect("generate inventory");
        tensors
    }

    fn validate(
        tensors: &[Qwen3MoeTensorMetadata],
    ) -> Result<Qwen3MoeTensorInventory, Qwen3MoeTensorInventoryError> {
        validate_qwen3_moe_tensor_inventory(
            PINNED_QWEN3_30B_A3B_CONFIG,
            PINNED_QWEN3_30B_A3B_SHARD_COUNT,
            tensors,
        )
    }

    #[test]
    fn validates_complete_pinned_inventory_and_role_counts() {
        let inventory = validate(&valid_inventory()).expect("complete inventory");
        let mut counts = HashMap::new();
        for tensor in inventory.tensors() {
            let key = std::mem::discriminant(&tensor.role());
            *counts.entry(key).or_insert(0_usize) += 1;
        }

        assert_eq!(inventory.len(), 18_867);
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::TokenEmbedding)],
            1
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::FinalNorm)],
            1
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::LanguageModelHead)],
            1
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::InputNorm { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::PostAttentionNorm { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::QueryProjection { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::KeyProjection { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::ValueProjection { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::OutputProjection { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::QueryNorm { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::KeyNorm { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::Router { layer: 0 })],
            48
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::ExpertGate {
                layer: 0,
                expert: 0
            })],
            6_144
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::ExpertUp {
                layer: 0,
                expert: 0
            })],
            6_144
        );
        assert_eq!(
            counts[&std::mem::discriminant(&Qwen3MoeTensorRole::ExpertDown {
                layer: 0,
                expert: 0
            })],
            6_144
        );
    }

    #[test]
    fn pinned_projection_shapes_use_explicit_head_dimension() {
        let inventory = validate(&valid_inventory()).expect("complete inventory");
        let shape = |name: &str| {
            inventory
                .tensors()
                .iter()
                .find(|tensor| tensor.metadata().name() == name)
                .expect("pinned tensor")
                .metadata()
                .shape()
                .dimensions()
        };
        assert_eq!(
            shape("model.layers.0.self_attn.q_proj.weight"),
            [4_096, 2_048]
        );
        assert_eq!(
            shape("model.layers.0.self_attn.k_proj.weight"),
            [512, 2_048]
        );
        assert_eq!(
            shape("model.layers.0.self_attn.v_proj.weight"),
            [512, 2_048]
        );
        assert_eq!(
            shape("model.layers.0.self_attn.o_proj.weight"),
            [2_048, 4_096]
        );
        assert_eq!(shape("model.layers.0.mlp.gate.weight"), [128, 2_048]);
        assert_eq!(
            shape("model.layers.0.mlp.experts.0.gate_proj.weight"),
            [768, 2_048]
        );
        assert_eq!(
            shape("model.layers.0.mlp.experts.0.down_proj.weight"),
            [2_048, 768]
        );
    }

    #[test]
    fn rejects_missing_tensor() {
        let mut tensors = valid_inventory();
        let removed = tensors.remove(0);
        assert_eq!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::MissingTensor { name: removed.name })
        );
    }

    #[test]
    fn rejects_duplicate_tensor() {
        let mut tensors = valid_inventory();
        tensors.push(tensors[0].clone());
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::DuplicateTensor { .. })
        ));
    }

    #[test]
    fn rejects_unknown_tensor() {
        let mut tensors = valid_inventory();
        tensors[0].name = "model.layers.0.self_attn.rotary_emb.weight".to_owned();
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::UnknownTensor { .. })
        ));
    }

    #[test]
    fn rejects_wrong_rank() {
        let mut tensors = valid_inventory();
        tensors[0].shape = TensorShape::new([151_936, 2_048, 1]);
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::RankMismatch {
                expected: 2,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn rejects_wrong_shape() {
        let mut tensors = valid_inventory();
        tensors[0].shape = TensorShape::new([151_936, 2_049]);
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_layer() {
        let mut tensors = valid_inventory();
        tensors[0].name = "model.layers.48.input_layernorm.weight".to_owned();
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::LayerOutOfRange {
                layer: 48,
                layer_count: 48,
                ..
            })
        ));
    }

    #[test]
    fn rejects_out_of_range_expert() {
        let mut tensors = valid_inventory();
        tensors[0].name = "model.layers.0.mlp.experts.128.gate_proj.weight".to_owned();
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::ExpertOutOfRange {
                expert: 128,
                expert_count: 128,
                ..
            })
        ));
    }

    #[test]
    fn rejects_wrong_source_dtype() {
        let mut tensors = valid_inventory();
        tensors[0].data_type = DataType::F32;
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::DataTypeMismatch {
                expected: DataType::BF16,
                actual: DataType::F32,
                ..
            })
        ));
    }

    #[test]
    fn rejects_out_of_range_shard_index() {
        let mut tensors = valid_inventory();
        tensors[0].shard_index = PINNED_QWEN3_30B_A3B_SHARD_COUNT;
        assert!(matches!(
            validate(&tensors),
            Err(Qwen3MoeTensorInventoryError::ShardIndexOutOfRange {
                shard_index: 16,
                shard_count: 16,
                ..
            })
        ));
    }

    #[test]
    fn checked_expected_count_rejects_overflow() {
        let mut source = PINNED_QWEN3_30B_A3B_CONFIG;
        source.layer_count = usize::MAX;
        assert_eq!(
            validate_qwen3_moe_tensor_inventory(source, 16, &[]),
            Err(Qwen3MoeTensorInventoryError::ArithmeticOverflow {
                operation: "complete tensor count",
            })
        );
    }
}
