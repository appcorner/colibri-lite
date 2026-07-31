#![doc = "Core runtime abstractions for colibri-lite-rs."]

mod backend;
mod comparison;
mod config;
mod dtype;
mod error;
pub mod ops;
mod planner;
mod runtime;
mod shape;
mod tensor;

pub use backend::{
    ExecutionBudget, ExecutionMetrics, ExecutionRequest, ExecutionResult, OperationDescriptor,
    OperationKind, TensorDescriptor,
};
pub use comparison::{
    ComparisonTolerance, DifferentialReport, DifferentialStatus, DivergenceKind, FirstDivergence,
    StageComparison,
};
pub use config::{ModelConfig, ModelConfigSpec};
pub use dtype::DataType;
pub use error::RuntimeError;
pub use planner::{
    AnalyticalCostModel, AnalyticalEstimate, CandidatePlacement, MeasuredRate,
    PlacementCapabilities, PlacementTier, PlannerWorkload, ProfileMeasurementStatus,
    TokenCostEstimate, TokenWork,
};
pub use runtime::{RuntimeInfo, runtime_info};
pub use shape::TensorShape;
pub use tensor::{Tensor, TensorView, TensorViewMut};
