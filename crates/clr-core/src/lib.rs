#![doc = "Core runtime abstractions for colibri-lite-rs."]

mod config;
mod dtype;
mod error;
pub mod ops;
mod runtime;
mod shape;
mod tensor;

pub use config::{ModelConfig, ModelConfigSpec};
pub use dtype::DataType;
pub use error::RuntimeError;
pub use runtime::{RuntimeInfo, runtime_info};
pub use shape::TensorShape;
pub use tensor::{Tensor, TensorView, TensorViewMut};
