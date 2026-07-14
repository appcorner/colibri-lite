#![doc = "Qwen3-MoE architecture implementation for colibri-lite-rs."]

mod block;
mod config;
mod model;
#[cfg(test)]
mod test_fixture;

pub use block::{Qwen3MoeBlock, Qwen3MoeBlockOutput, Qwen3MoeBlockWeightsSpec};
pub use config::{Qwen3MoeConfig, Qwen3MoeConfigSpec};
pub use model::{Qwen3MoeModel, Qwen3MoeModelOutput, Qwen3MoeModelWeightsSpec};
