use std::{collections::HashSet, fmt, path::PathBuf};

use clr_core::{DataType, TensorShape};
use clr_storage::{ArtifactManifest, DenseSourceShard, ExpertId, ExpertKey, ExpertRegistration};

use crate::{
    PINNED_QWEN3_30B_A3B_CONFIG, PINNED_QWEN3_30B_A3B_MODEL_ID, PINNED_QWEN3_30B_A3B_REVISION,
    PackedExpertLayout, Qwen3MoeConfig, Qwen3MoeTensorInventoryError, Qwen3MoeTensorMetadata,
    Qwen3MoeTensorRole, tensor_inventory::validate_qwen3_moe_tensor_metadata,
};

mod io;

/// Source projection identity and packed output order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertProjectionKind {
    /// Gated activation projection, packed first.
    Gate,
    /// Up projection, packed second.
    Up,
    /// Down projection, packed third.
    Down,
}

/// Completeness requirement for one expert conversion transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3MoeExpertConversionScope {
    /// A reviewed subset containing experts from at least two layers.
    VerticalSlice,
    /// All 6,144 experts and all 18,432 source projections.
    Complete,
}

/// One Qwen-validated expert source projection and its absolute shard range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeExpertSourceProjection {
    /// Declared zero-based decoder layer.
    pub layer_index: usize,
    /// Declared zero-based expert within the layer.
    pub expert_index: usize,
    /// Gate/up/down identity and required packed order.
    pub projection: ExpertProjectionKind,
    /// Canonical source name, BF16 dtype, shape, and source shard index.
    pub metadata: Qwen3MoeTensorMetadata,
    /// Absolute byte offset from the start of the complete source shard.
    pub offset: u64,
    /// Exact contiguous BF16 source payload length.
    pub length: u64,
}

/// Inputs for one pinned expert conversion transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeExpertConversionSpec {
    /// Vertical-slice or complete-inventory validation policy.
    pub scope: Qwen3MoeExpertConversionScope,
    /// Complete pinned source shards referenced by selected projections.
    pub source_shards: Vec<DenseSourceShard>,
    /// Projections in layer/expert/gate-up-down order.
    pub projections: Vec<Qwen3MoeExpertSourceProjection>,
    /// New directory receiving layer shards and manifest.
    pub output_directory: PathBuf,
    /// Caller-observed free bytes used by exact logical-byte preflight.
    pub available_space_bytes: u64,
    /// Even, non-zero BF16 source chunk size.
    pub source_chunk_bytes: usize,
}

/// Relative packed range and source matrix shape for one projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProjectionArtifactRange {
    /// Byte offset relative to the logical expert payload.
    pub offset: u64,
    /// F32 byte length.
    pub length: u64,
    /// Matrix shape in proven runtime orientation.
    pub shape: TensorShape,
}

/// Complete manifest record for one logical packed expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeExpertArtifactRecord {
    /// Existing layer/expert runtime cache key.
    pub key: ExpertKey,
    /// Arbitrary physical shard ID.
    pub shard_id: usize,
    /// Byte offset within the physical shard.
    pub payload_offset: u64,
    /// Complete gate/up/down F32 payload length.
    pub payload_length: u64,
    /// Relative gate range and shape.
    pub gate: ExpertProjectionArtifactRange,
    /// Relative up range and shape.
    pub up: ExpertProjectionArtifactRange,
    /// Relative down range and shape.
    pub down: ExpertProjectionArtifactRange,
    /// Source storage dtype.
    pub source_data_type: DataType,
    /// Artifact storage dtype.
    pub artifact_data_type: DataType,
    /// SHA-256 of exactly this logical payload.
    pub sha256: [u8; 32],
}

/// One finalized physical expert container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeExpertShardRecord {
    /// Arbitrary shard ID referenced by expert records.
    pub shard_id: usize,
    /// Artifact-root-relative physical path.
    pub path: PathBuf,
    /// Complete physical shard byte length.
    pub byte_length: u64,
    /// Complete physical shard SHA-256.
    pub sha256: [u8; 32],
}

/// Runtime mappings plus detailed versioned expert conversion records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeExpertArtifactManifest {
    runtime_manifest: ArtifactManifest,
    registrations: Vec<ExpertRegistration>,
    shards: Vec<Qwen3MoeExpertShardRecord>,
    experts: Vec<Qwen3MoeExpertArtifactRecord>,
}

impl Qwen3MoeExpertArtifactManifest {
    /// Returns the unchanged artifact reader contract.
    #[must_use]
    pub const fn runtime_manifest(&self) -> &ArtifactManifest {
        &self.runtime_manifest
    }

    /// Returns mappings consumed unchanged by `ExpertStore`.
    #[must_use]
    pub fn registrations(&self) -> &[ExpertRegistration] {
        &self.registrations
    }

    /// Returns deterministic physical shard records.
    #[must_use]
    pub fn shards(&self) -> &[Qwen3MoeExpertShardRecord] {
        &self.shards
    }

    /// Returns deterministic logical expert records.
    #[must_use]
    pub fn experts(&self) -> &[Qwen3MoeExpertArtifactRecord] {
        &self.experts
    }
}

/// Evidence returned after a committed expert artifact transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeExpertConversionSummary {
    /// Runtime and detailed artifact mappings.
    pub manifest: Qwen3MoeExpertArtifactManifest,
    /// Final deterministic JSON manifest path.
    pub manifest_path: PathBuf,
    /// Deterministic JSON manifest SHA-256.
    pub manifest_sha256: [u8; 32],
    /// Complete source bytes read while verifying shard hashes.
    pub source_verification_bytes_read: u64,
    /// BF16 tensor bytes read by conversion and exact verification.
    pub source_payload_bytes_read: u64,
    /// Total F32 expert shard bytes written.
    pub artifact_bytes_written: u64,
    /// Exact logical payload plus manifest bytes checked before output.
    pub preflight_required_bytes: u64,
    /// Maximum explicitly allocated conversion/verification buffers.
    pub peak_buffer_bytes: usize,
}

/// Structured failures from pinned expert validation and conversion.
#[derive(Debug)]
pub enum Qwen3MoeExpertConversionError {
    /// Checked size arithmetic overflowed.
    ArithmeticOverflow { operation: &'static str },
    /// Source chunk size is zero, odd, or overflows working buffers.
    InvalidChunkSize { actual: usize },
    /// M4.1-04 name/dtype/shape/shard validation failed.
    Inventory(Qwen3MoeTensorInventoryError),
    /// Input is not in gate/up/down order.
    InvalidProjectionOrder {
        position: usize,
        expected: ExpertProjectionKind,
        actual: ExpertProjectionKind,
    },
    /// Declared layer/expert/projection differs from the canonical name.
    ProjectionIdentityMismatch { name: String },
    /// A logical layer/expert occurs more than once.
    DuplicateExpert { layer: usize, expert: usize },
    /// A vertical slice does not cover at least two layers.
    InsufficientVerticalSliceLayers { actual: usize },
    /// Complete mode is missing or misorders an expected expert.
    IncompleteExpertInventory {
        expected_layer: usize,
        expected_expert: usize,
    },
    /// Shape-derived BF16 bytes differ from the source range.
    SourceLengthMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    /// A selected source range exceeds its shard.
    InvalidSourceRange { name: String },
    /// A local source shard length differs from pinned provenance.
    SourceShardLengthMismatch { path: PathBuf },
    /// A complete local source shard hash differs from pinned provenance.
    SourceShardHashMismatch { path: PathBuf },
    /// Exact logical output bytes exceed caller-observed free space.
    InsufficientDiskSpace { required: u64, available: u64 },
    /// The requested final output directory already exists.
    OutputExists { path: PathBuf },
    /// Written F32 bytes differ from direct BF16 decoding.
    RoundTripMismatch {
        layer: usize,
        expert: usize,
        element: u64,
    },
    /// Existing artifact format v1 validation failed.
    Artifact(clr_storage::StorageError),
    /// A filesystem operation failed.
    Io {
        action: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    #[cfg(test)]
    InjectedIncompleteOutput,
}

impl fmt::Display for Qwen3MoeExpertConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow { operation } => {
                write!(
                    formatter,
                    "arithmetic overflow while calculating {operation}"
                )
            }
            Self::InvalidChunkSize { actual } => {
                write!(formatter, "invalid BF16 chunk size {actual}")
            }
            Self::Inventory(error) => write!(formatter, "invalid expert tensor: {error}"),
            Self::InvalidProjectionOrder {
                position,
                expected,
                actual,
            } => write!(
                formatter,
                "projection {position} must be {expected:?}, got {actual:?}"
            ),
            Self::ProjectionIdentityMismatch { name } => {
                write!(formatter, "projection identity does not match '{name}'")
            }
            Self::DuplicateExpert { layer, expert } => {
                write!(formatter, "duplicate expert {layer}:{expert}")
            }
            Self::InsufficientVerticalSliceLayers { actual } => write!(
                formatter,
                "expert vertical slice must cover at least two layers, got {actual}"
            ),
            Self::IncompleteExpertInventory {
                expected_layer,
                expected_expert,
            } => write!(
                formatter,
                "incomplete expert inventory at {expected_layer}:{expected_expert}"
            ),
            Self::SourceLengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "source tensor '{name}' length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidSourceRange { name } => {
                write!(formatter, "invalid source range for '{name}'")
            }
            Self::SourceShardLengthMismatch { path } => {
                write!(
                    formatter,
                    "source shard '{}' length mismatch",
                    path.display()
                )
            }
            Self::SourceShardHashMismatch { path } => {
                write!(
                    formatter,
                    "source shard '{}' SHA-256 mismatch",
                    path.display()
                )
            }
            Self::InsufficientDiskSpace {
                required,
                available,
            } => write!(
                formatter,
                "expert conversion requires {required} bytes, only {available} available"
            ),
            Self::OutputExists { path } => {
                write!(formatter, "output '{}' already exists", path.display())
            }
            Self::RoundTripMismatch {
                layer,
                expert,
                element,
            } => write!(
                formatter,
                "expert {layer}:{expert} differs after F32 round trip at element {element}"
            ),
            Self::Artifact(error) => write!(formatter, "invalid expert artifact: {error}"),
            Self::Io {
                action,
                path,
                source,
            } => {
                write!(
                    formatter,
                    "failed to {action} '{}': {source}",
                    path.display()
                )
            }
            #[cfg(test)]
            Self::InjectedIncompleteOutput => write!(formatter, "injected incomplete output"),
        }
    }
}

impl std::error::Error for Qwen3MoeExpertConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inventory(error) => Some(error),
            Self::Artifact(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<clr_storage::StorageError> for Qwen3MoeExpertConversionError {
    fn from(error: clr_storage::StorageError) -> Self {
        Self::Artifact(error)
    }
}

/// Validates, converts, verifies, and atomically commits packed expert shards.
///
/// # Errors
///
/// Returns a structured identity, inventory, integrity, range, disk, exact
/// round-trip, artifact, or filesystem error. Failed transactions remove all
/// incomplete and finalized outputs created by the transaction.
pub fn convert_pinned_qwen3_moe_experts(
    spec: &Qwen3MoeExpertConversionSpec,
) -> Result<Qwen3MoeExpertConversionSummary, Qwen3MoeExpertConversionError> {
    io::convert(spec, None)
}

#[derive(Debug, Clone)]
struct ValidatedExpert {
    layer_index: usize,
    expert_index: usize,
    shard_id: usize,
    payload_offset: u64,
    projections: [Qwen3MoeExpertSourceProjection; 3],
}

#[derive(Debug)]
struct ValidatedSpec {
    config: Qwen3MoeConfig,
    layout: PackedExpertLayout,
    experts: Vec<ValidatedExpert>,
    used_source_shards: Vec<bool>,
    output_shard_lengths: Vec<(usize, u64)>,
    source_verification_bytes: u64,
    source_payload_bytes: u64,
    artifact_bytes: u64,
    peak_buffer_bytes: usize,
}

fn validate_spec(
    spec: &Qwen3MoeExpertConversionSpec,
) -> Result<ValidatedSpec, Qwen3MoeExpertConversionError> {
    if spec.source_chunk_bytes == 0 || spec.source_chunk_bytes % 2 != 0 {
        return Err(Qwen3MoeExpertConversionError::InvalidChunkSize {
            actual: spec.source_chunk_bytes,
        });
    }
    let peak_buffer_bytes = spec.source_chunk_bytes.checked_mul(5).ok_or(
        Qwen3MoeExpertConversionError::ArithmeticOverflow {
            operation: "expert conversion buffers",
        },
    )?;
    if spec.projections.len() % 3 != 0 {
        return Err(Qwen3MoeExpertConversionError::IncompleteExpertInventory {
            expected_layer: 0,
            expected_expert: spec.projections.len() / 3,
        });
    }
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .map_err(Qwen3MoeTensorInventoryError::from)
        .map_err(Qwen3MoeExpertConversionError::Inventory)?
        .runtime_config();
    let layout = PackedExpertLayout::for_config(config);
    let mut used_source_shards = vec![false; spec.source_shards.len()];
    let (experts, source_payload_bytes) =
        validate_experts(spec, config, layout, &mut used_source_shards)?;
    validate_scope(spec.scope, config, &experts)?;
    let output_shard_lengths = output_shard_lengths(&experts, layout)?;
    let artifact_bytes = output_shard_lengths
        .iter()
        .try_fold(0_u64, |total, (_, length)| {
            total
                .checked_add(*length)
                .ok_or(Qwen3MoeExpertConversionError::ArithmeticOverflow {
                    operation: "complete expert artifact bytes",
                })
        })?;
    let source_verification_bytes = spec
        .source_shards
        .iter()
        .zip(&used_source_shards)
        .filter(|(_, used)| **used)
        .try_fold(0_u64, |total, (shard, _)| {
            total.checked_add(shard.byte_length).ok_or(
                Qwen3MoeExpertConversionError::ArithmeticOverflow {
                    operation: "expert source verification bytes",
                },
            )
        })?;
    Ok(ValidatedSpec {
        config,
        layout,
        experts,
        used_source_shards,
        output_shard_lengths,
        source_verification_bytes,
        source_payload_bytes,
        artifact_bytes,
        peak_buffer_bytes,
    })
}

fn validate_experts(
    spec: &Qwen3MoeExpertConversionSpec,
    config: Qwen3MoeConfig,
    layout: PackedExpertLayout,
    used_source_shards: &mut [bool],
) -> Result<(Vec<ValidatedExpert>, u64), Qwen3MoeExpertConversionError> {
    let mut seen = HashSet::with_capacity(spec.projections.len() / 3);
    let mut experts: Vec<ValidatedExpert> = Vec::with_capacity(spec.projections.len() / 3);
    let mut source_payload_bytes = 0_u64;
    let mut last_layer = None;
    let mut layer_offset = 0_u64;

    for (group_index, group) in spec.projections.chunks_exact(3).enumerate() {
        for (projection_index, expected) in [
            ExpertProjectionKind::Gate,
            ExpertProjectionKind::Up,
            ExpertProjectionKind::Down,
        ]
        .into_iter()
        .enumerate()
        {
            if group[projection_index].projection != expected {
                return Err(Qwen3MoeExpertConversionError::InvalidProjectionOrder {
                    position: group_index * 3 + projection_index,
                    expected,
                    actual: group[projection_index].projection,
                });
            }
        }
        let layer = group[0].layer_index;
        let expert = group[0].expert_index;
        if group
            .iter()
            .any(|projection| projection.layer_index != layer || projection.expert_index != expert)
        {
            return Err(Qwen3MoeExpertConversionError::ProjectionIdentityMismatch {
                name: group[0].metadata.name().to_owned(),
            });
        }
        if !seen.insert((layer, expert)) {
            return Err(Qwen3MoeExpertConversionError::DuplicateExpert { layer, expert });
        }
        if let Some(previous) = experts.last() {
            if (layer, expert) <= (previous.layer_index, previous.expert_index) {
                return Err(Qwen3MoeExpertConversionError::IncompleteExpertInventory {
                    expected_layer: previous.layer_index,
                    expected_expert: previous.expert_index + 1,
                });
            }
        }
        for projection in group {
            validate_projection(spec, projection, config, used_source_shards)?;
            source_payload_bytes = source_payload_bytes.checked_add(projection.length).ok_or(
                Qwen3MoeExpertConversionError::ArithmeticOverflow {
                    operation: "expert source payload bytes",
                },
            )?;
        }
        if last_layer != Some(layer) {
            last_layer = Some(layer);
            layer_offset = 0;
        }
        experts.push(ValidatedExpert {
            layer_index: layer,
            expert_index: expert,
            shard_id: layer,
            payload_offset: layer_offset,
            projections: group.to_vec().try_into().expect("three projection group"),
        });
        layer_offset = layer_offset
            .checked_add(layout.total_byte_length as u64)
            .ok_or(Qwen3MoeExpertConversionError::ArithmeticOverflow {
                operation: "expert layer shard offset",
            })?;
    }

    Ok((experts, source_payload_bytes))
}

fn validate_projection(
    spec: &Qwen3MoeExpertConversionSpec,
    projection: &Qwen3MoeExpertSourceProjection,
    config: Qwen3MoeConfig,
    used_shards: &mut [bool],
) -> Result<(), Qwen3MoeExpertConversionError> {
    let role = validate_qwen3_moe_tensor_metadata(
        PINNED_QWEN3_30B_A3B_CONFIG,
        spec.source_shards.len(),
        &projection.metadata,
    )
    .map_err(Qwen3MoeExpertConversionError::Inventory)?;
    let identity = match role {
        Qwen3MoeTensorRole::ExpertGate { layer, expert } => {
            (layer, expert, ExpertProjectionKind::Gate)
        }
        Qwen3MoeTensorRole::ExpertUp { layer, expert } => (layer, expert, ExpertProjectionKind::Up),
        Qwen3MoeTensorRole::ExpertDown { layer, expert } => {
            (layer, expert, ExpertProjectionKind::Down)
        }
        _ => {
            return Err(Qwen3MoeExpertConversionError::ProjectionIdentityMismatch {
                name: projection.metadata.name().to_owned(),
            });
        }
    };
    if identity
        != (
            projection.layer_index,
            projection.expert_index,
            projection.projection,
        )
    {
        return Err(Qwen3MoeExpertConversionError::ProjectionIdentityMismatch {
            name: projection.metadata.name().to_owned(),
        });
    }
    let expected = projection
        .metadata
        .shape()
        .byte_count(DataType::BF16)
        .map_err(|_| Qwen3MoeExpertConversionError::ArithmeticOverflow {
            operation: "expert BF16 projection bytes",
        })? as u64;
    if expected != projection.length {
        return Err(Qwen3MoeExpertConversionError::SourceLengthMismatch {
            name: projection.metadata.name().to_owned(),
            expected,
            actual: projection.length,
        });
    }
    let shard_index = projection.metadata.shard_index();
    let shard = &spec.source_shards[shard_index];
    let end = projection
        .offset
        .checked_add(projection.length)
        .ok_or_else(|| Qwen3MoeExpertConversionError::InvalidSourceRange {
            name: projection.metadata.name().to_owned(),
        })?;
    if end > shard.byte_length
        || projection.layer_index >= config.model().layer_count()
        || projection.expert_index >= config.expert_count()
    {
        return Err(Qwen3MoeExpertConversionError::InvalidSourceRange {
            name: projection.metadata.name().to_owned(),
        });
    }
    used_shards[shard_index] = true;
    Ok(())
}

fn validate_scope(
    scope: Qwen3MoeExpertConversionScope,
    config: Qwen3MoeConfig,
    experts: &[ValidatedExpert],
) -> Result<(), Qwen3MoeExpertConversionError> {
    match scope {
        Qwen3MoeExpertConversionScope::VerticalSlice => {
            let layers = experts
                .iter()
                .map(|expert| expert.layer_index)
                .collect::<HashSet<_>>()
                .len();
            if layers < 2 {
                return Err(
                    Qwen3MoeExpertConversionError::InsufficientVerticalSliceLayers {
                        actual: layers,
                    },
                );
            }
        }
        Qwen3MoeExpertConversionScope::Complete => {
            let expected_count = config.model().layer_count() * config.expert_count();
            if experts.len() != expected_count {
                return Err(Qwen3MoeExpertConversionError::IncompleteExpertInventory {
                    expected_layer: experts.len() / config.expert_count(),
                    expected_expert: experts.len() % config.expert_count(),
                });
            }
            for (index, expert) in experts.iter().enumerate() {
                let expected = (index / config.expert_count(), index % config.expert_count());
                if (expert.layer_index, expert.expert_index) != expected {
                    return Err(Qwen3MoeExpertConversionError::IncompleteExpertInventory {
                        expected_layer: expected.0,
                        expected_expert: expected.1,
                    });
                }
            }
        }
    }
    Ok(())
}

fn output_shard_lengths(
    experts: &[ValidatedExpert],
    layout: PackedExpertLayout,
) -> Result<Vec<(usize, u64)>, Qwen3MoeExpertConversionError> {
    let mut output = Vec::new();
    for expert in experts {
        let end = expert
            .payload_offset
            .checked_add(layout.total_byte_length as u64)
            .ok_or(Qwen3MoeExpertConversionError::ArithmeticOverflow {
                operation: "expert layer shard length",
            })?;
        match output.last_mut() {
            Some((shard_id, length)) if *shard_id == expert.shard_id => *length = end,
            _ => output.push((expert.shard_id, end)),
        }
    }
    Ok(output)
}

fn expert_key(layer: usize, expert: usize) -> ExpertKey {
    ExpertKey {
        layer_index: u32::try_from(layer).expect("validated layer fits u32"),
        expert_id: ExpertId(u32::try_from(expert).expect("validated expert fits u32")),
    }
}

fn logical_name(layer: usize, expert: usize) -> String {
    format!("layer.{layer}.expert.{expert}")
}

fn shard_path(layer: usize) -> PathBuf {
    format!("experts-layer-{layer:05}-of-00048.bin").into()
}

fn projection_ranges(
    layout: PackedExpertLayout,
    config: Qwen3MoeConfig,
) -> [ExpertProjectionArtifactRange; 3] {
    let gate_up_shape =
        TensorShape::new([config.moe_intermediate_size(), config.model().hidden_size()]);
    let down_shape =
        TensorShape::new([config.model().hidden_size(), config.moe_intermediate_size()]);
    [
        ExpertProjectionArtifactRange {
            offset: layout.gate_offset as u64,
            length: layout.gate_length as u64,
            shape: gate_up_shape.clone(),
        },
        ExpertProjectionArtifactRange {
            offset: layout.up_offset as u64,
            length: layout.up_length as u64,
            shape: gate_up_shape,
        },
        ExpertProjectionArtifactRange {
            offset: layout.down_offset as u64,
            length: layout.down_length as u64,
            shape: down_shape,
        },
    ]
}

fn overflow(operation: &'static str) -> Qwen3MoeExpertConversionError {
    Qwen3MoeExpertConversionError::ArithmeticOverflow { operation }
}

#[cfg(test)]
mod tests;
