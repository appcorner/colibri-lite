#![doc = "Qwen3-MoE architecture implementation for colibri-lite-rs."]

mod block;
mod cache;
mod config;
mod generation;
mod model;
mod session;
mod streaming;
#[cfg(test)]
mod test_fixture;

pub use block::{Qwen3MoeBlock, Qwen3MoeBlockOutput, Qwen3MoeBlockWeightsSpec};
pub use cache::KvCache;
pub use config::{Qwen3MoeConfig, Qwen3MoeConfigSpec};
pub use generation::{SeededRng, greedy_token, sample_token};
pub use model::{Qwen3MoeModel, Qwen3MoeModelOutput, Qwen3MoeModelWeightsSpec};
pub use session::{GenerationError, GenerationSession, PrefillOutput};
pub use streaming::{
    PackedExpertLayout, StreamingBlockWeightsSpec, StreamingModelError, StreamingModelWeightsSpec,
    StreamingQwen3MoeModel,
};
