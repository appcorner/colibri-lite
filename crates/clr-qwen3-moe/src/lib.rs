#![doc = "Qwen3-MoE architecture implementation for colibri-lite-rs."]

mod block;
mod cache;
mod config;
mod dense_conversion;
mod expert_conversion;
mod generation;
mod model;
mod session;
mod source_config;
mod streaming;
mod tensor_inventory;
mod test_fixture;

pub use block::{Qwen3MoeBlock, Qwen3MoeBlockOutput, Qwen3MoeBlockWeightsSpec};
pub use cache::KvCache;
pub use config::{Qwen3MoeConfig, Qwen3MoeConfigSpec};
pub use dense_conversion::{
    PINNED_QWEN3_30B_A3B_MODEL_ID, PINNED_QWEN3_30B_A3B_REVISION, Qwen3MoeDenseConversionError,
    Qwen3MoeDenseConversionScope, Qwen3MoeDenseConversionSpec, Qwen3MoeDenseSourceTensor,
    convert_pinned_qwen3_moe_dense_tensors,
};
pub use expert_conversion::{
    ExpertProjectionArtifactRange, ExpertProjectionKind, Qwen3MoeExpertArtifactManifest,
    Qwen3MoeExpertArtifactRecord, Qwen3MoeExpertConversionError, Qwen3MoeExpertConversionScope,
    Qwen3MoeExpertConversionSpec, Qwen3MoeExpertConversionSummary, Qwen3MoeExpertShardRecord,
    Qwen3MoeExpertSourceProjection, convert_pinned_qwen3_moe_experts,
};
pub use generation::{SeededRng, greedy_token, sample_token};
pub use model::{Qwen3MoeModel, Qwen3MoeModelOutput, Qwen3MoeModelWeightsSpec};
pub use session::{GenerationError, GenerationSession, PrefillOutput};
pub use source_config::{PINNED_QWEN3_30B_A3B_CONFIG, Qwen3MoeConfigMapping, Qwen3MoeSourceConfig};
pub use streaming::{
    PackedExpertLayout, StreamingBlockWeightsSpec, StreamingModelError, StreamingModelWeightsSpec,
    StreamingQwen3MoeModel,
};
pub use tensor_inventory::{
    PINNED_QWEN3_30B_A3B_SHARD_COUNT, Qwen3MoeMappedTensor, Qwen3MoeTensorInventory,
    Qwen3MoeTensorInventoryError, Qwen3MoeTensorMetadata, Qwen3MoeTensorRole,
    validate_qwen3_moe_tensor_inventory,
};
pub use test_fixture::{frozen_tiny_model, frozen_tiny_prompt};
