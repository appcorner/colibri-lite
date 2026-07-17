use std::{collections::HashSet, fmt, path::PathBuf};

use clr_storage::{
    DenseConversionError, DenseConversionSpec, DenseConversionSummary, DenseSourceShard,
    DenseSourceTensor, convert_dense_bf16_to_f32,
};

use crate::{
    PINNED_QWEN3_30B_A3B_CONFIG, Qwen3MoeTensorInventoryError, Qwen3MoeTensorMetadata,
    Qwen3MoeTensorRole,
    tensor_inventory::{for_each_expected_dense_name, validate_qwen3_moe_tensor_metadata},
};

/// Immutable upstream model identifier used by M4 conversion.
pub const PINNED_QWEN3_30B_A3B_MODEL_ID: &str = "Qwen/Qwen3-30B-A3B";
/// Immutable upstream source revision used by M4 conversion.
pub const PINNED_QWEN3_30B_A3B_REVISION: &str = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39";

/// Completeness requirement for a pinned dense conversion transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3MoeDenseConversionScope {
    /// Validate and convert a named real-tensor subset.
    VerticalSlice,
    /// Require all 435 non-expert tensors from the pinned inventory.
    Complete,
}

/// Qwen-validated source metadata plus its absolute shard byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeDenseSourceTensor {
    /// Canonical name, BF16 dtype, expected shape, and source shard index.
    pub metadata: Qwen3MoeTensorMetadata,
    /// Absolute byte offset from the start of the complete source shard.
    pub offset: u64,
    /// Exact BF16 source payload byte length.
    pub length: u64,
}

/// Inputs for one pinned Qwen3-MoE dense conversion transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeDenseConversionSpec {
    /// Whether this is the initial subset or the complete dense inventory.
    pub scope: Qwen3MoeDenseConversionScope,
    /// Complete pinned source shards referenced by selected tensors.
    pub source_shards: Vec<DenseSourceShard>,
    /// Selected Qwen dense tensors and verified source ranges.
    pub tensors: Vec<Qwen3MoeDenseSourceTensor>,
    /// Directory receiving the shared payload and deterministic manifest.
    pub output_directory: PathBuf,
    /// Caller-observed free space used by preflight.
    pub available_space_bytes: u64,
    /// Even, non-zero BF16 source chunk size.
    pub source_chunk_bytes: usize,
}

/// Structured Qwen role/config or generic storage conversion failure.
#[derive(Debug)]
pub enum Qwen3MoeDenseConversionError {
    /// The vertical slice contains no tensor.
    EmptyVerticalSlice,
    /// Pinned name, dtype, shape, layer, expert, or shard validation failed.
    Inventory(Qwen3MoeTensorInventoryError),
    /// Expert tensors are reserved for M4.1-06.
    ExpertTensor { name: String },
    /// Source integrity, range, preflight, conversion, or output failed.
    Storage(DenseConversionError),
}

impl fmt::Display for Qwen3MoeDenseConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVerticalSlice => write!(formatter, "dense vertical slice must not be empty"),
            Self::Inventory(error) => write!(formatter, "invalid Qwen dense tensor: {error}"),
            Self::ExpertTensor { name } => {
                write!(formatter, "expert tensor '{name}' is deferred to M4.1-06")
            }
            Self::Storage(error) => write!(formatter, "dense conversion failed: {error}"),
        }
    }
}

impl std::error::Error for Qwen3MoeDenseConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inventory(error) => Some(error),
            Self::Storage(error) => Some(error),
            _ => None,
        }
    }
}

/// Validates pinned Qwen roles and converts only non-expert tensors.
///
/// # Errors
///
/// Returns a structured error before source decoding for any invalid Qwen
/// tensor, expert role, incomplete full inventory, or storage conversion
/// failure.
pub fn convert_pinned_qwen3_moe_dense_tensors(
    spec: &Qwen3MoeDenseConversionSpec,
) -> Result<DenseConversionSummary, Qwen3MoeDenseConversionError> {
    if spec.scope == Qwen3MoeDenseConversionScope::VerticalSlice && spec.tensors.is_empty() {
        return Err(Qwen3MoeDenseConversionError::EmptyVerticalSlice);
    }
    let mut names = HashSet::with_capacity(spec.tensors.len());
    let mut tensors = Vec::with_capacity(spec.tensors.len());
    for tensor in &spec.tensors {
        let role = validate_qwen3_moe_tensor_metadata(
            PINNED_QWEN3_30B_A3B_CONFIG,
            spec.source_shards.len(),
            &tensor.metadata,
        )
        .map_err(Qwen3MoeDenseConversionError::Inventory)?;
        if matches!(
            role,
            Qwen3MoeTensorRole::ExpertGate { .. }
                | Qwen3MoeTensorRole::ExpertUp { .. }
                | Qwen3MoeTensorRole::ExpertDown { .. }
        ) {
            return Err(Qwen3MoeDenseConversionError::ExpertTensor {
                name: tensor.metadata.name().to_owned(),
            });
        }
        names.insert(tensor.metadata.name());
        tensors.push(DenseSourceTensor {
            name: tensor.metadata.name().to_owned(),
            shape: tensor.metadata.shape().clone(),
            data_type: tensor.metadata.data_type(),
            shard_index: tensor.metadata.shard_index(),
            offset: tensor.offset,
            length: tensor.length,
        });
    }

    if spec.scope == Qwen3MoeDenseConversionScope::Complete {
        let config = PINNED_QWEN3_30B_A3B_CONFIG
            .map_to_f32_runtime()
            .map_err(Qwen3MoeTensorInventoryError::from)
            .map_err(Qwen3MoeDenseConversionError::Inventory)?
            .runtime_config();
        for_each_expected_dense_name(config, |name| {
            if names.contains(name.as_str()) {
                Ok(())
            } else {
                Err(Qwen3MoeTensorInventoryError::MissingTensor { name })
            }
        })
        .map_err(Qwen3MoeDenseConversionError::Inventory)?;
    }

    convert_dense_bf16_to_f32(&DenseConversionSpec {
        model_id: PINNED_QWEN3_30B_A3B_MODEL_ID.to_owned(),
        model_revision: PINNED_QWEN3_30B_A3B_REVISION.to_owned(),
        source_shards: spec.source_shards.clone(),
        tensors,
        output_directory: spec.output_directory.clone(),
        available_space_bytes: spec.available_space_bytes,
        source_chunk_bytes: spec.source_chunk_bytes,
    })
    .map_err(Qwen3MoeDenseConversionError::Storage)
}

#[cfg(test)]
mod tests {
    use clr_core::{DataType, TensorShape};
    use clr_storage::DEFAULT_CONVERSION_CHUNK_BYTES;

    use super::*;
    use crate::PINNED_QWEN3_30B_A3B_SHARD_COUNT;

    fn metadata(name: &str, shape: impl Into<Box<[usize]>>) -> Qwen3MoeTensorMetadata {
        Qwen3MoeTensorMetadata::new(name, DataType::BF16, TensorShape::new(shape), 0)
    }

    fn spec(tensors: Vec<Qwen3MoeDenseSourceTensor>) -> Qwen3MoeDenseConversionSpec {
        Qwen3MoeDenseConversionSpec {
            scope: Qwen3MoeDenseConversionScope::VerticalSlice,
            source_shards: vec![DenseSourceShard {
                path: "unused.safetensors".into(),
                byte_length: 1_000_000,
                sha256: [0; 32],
            }],
            tensors,
            output_directory: "unused-output".into(),
            available_space_bytes: u64::MAX,
            source_chunk_bytes: DEFAULT_CONVERSION_CHUNK_BYTES,
        }
    }

    #[test]
    fn rejects_wrong_pinned_shape_before_storage_access() {
        let tensor = Qwen3MoeDenseSourceTensor {
            metadata: metadata("model.norm.weight", [2_047]),
            offset: 0,
            length: 4_094,
        };

        assert!(matches!(
            convert_pinned_qwen3_moe_dense_tensors(&spec(vec![tensor])),
            Err(Qwen3MoeDenseConversionError::Inventory(
                Qwen3MoeTensorInventoryError::ShapeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn rejects_expert_tensor_before_storage_access() {
        let tensor = Qwen3MoeDenseSourceTensor {
            metadata: metadata(
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                [768, 2_048],
            ),
            offset: 0,
            length: 768 * 2_048 * 2,
        };

        assert!(matches!(
            convert_pinned_qwen3_moe_dense_tensors(&spec(vec![tensor])),
            Err(Qwen3MoeDenseConversionError::ExpertTensor { .. })
        ));
    }

    #[test]
    fn complete_scope_requires_every_dense_tensor() {
        let mut incomplete = spec(vec![Qwen3MoeDenseSourceTensor {
            metadata: metadata("model.norm.weight", [2_048]),
            offset: 0,
            length: 4_096,
        }]);
        incomplete.scope = Qwen3MoeDenseConversionScope::Complete;

        assert!(matches!(
            convert_pinned_qwen3_moe_dense_tensors(&incomplete),
            Err(Qwen3MoeDenseConversionError::Inventory(
                Qwen3MoeTensorInventoryError::MissingTensor { .. }
            ))
        ));
    }

    #[test]
    fn vertical_slice_must_not_be_empty() {
        assert!(matches!(
            convert_pinned_qwen3_moe_dense_tensors(&spec(Vec::new())),
            Err(Qwen3MoeDenseConversionError::EmptyVerticalSlice)
        ));
        assert_eq!(PINNED_QWEN3_30B_A3B_SHARD_COUNT, 16);
    }
}
