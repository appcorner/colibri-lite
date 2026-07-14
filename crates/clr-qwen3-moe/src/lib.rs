#![doc = "Qwen3-MoE architecture implementation for colibri-lite-rs."]

mod block;
mod config;
mod model;
mod streaming;
#[cfg(test)]
mod test_fixture;

pub use block::{Qwen3MoeBlock, Qwen3MoeBlockOutput, Qwen3MoeBlockWeightsSpec};
pub use config::{Qwen3MoeConfig, Qwen3MoeConfigSpec};
pub use model::{Qwen3MoeModel, Qwen3MoeModelOutput, Qwen3MoeModelWeightsSpec};
pub use streaming::{
    PackedExpertLayout, StreamingBlockWeightsSpec, StreamingModelError, StreamingModelWeightsSpec,
    StreamingQwen3MoeModel,
};
