#![doc = "Core runtime abstractions for colibri-lite-rs."]

mod config;
mod dtype;
mod error;
mod runtime;
mod shape;

pub use config::{ModelConfig, ModelConfigSpec};
pub use dtype::DataType;
pub use error::RuntimeError;
pub use runtime::{RuntimeInfo, runtime_info};
pub use shape::TensorShape;
