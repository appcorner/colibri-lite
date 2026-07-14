#![doc = "Qwen3-MoE architecture implementation for colibri-lite-rs."]

mod block;
mod config;

pub use block::{Qwen3MoeBlock, Qwen3MoeBlockOutput, Qwen3MoeBlockWeightsSpec};
pub use config::{Qwen3MoeConfig, Qwen3MoeConfigSpec};
