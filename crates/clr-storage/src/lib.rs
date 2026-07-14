#![doc = "Storage and expert-cache subsystem for colibri-lite-rs."]

mod converter;
mod error;
mod expert;
mod hash;
mod manifest;
mod reader;

pub use converter::{
    DEFAULT_CONVERSION_CHUNK_BYTES, DenseConversionError, DenseConversionSpec,
    DenseConversionSummary, DenseSourceShard, DenseSourceTensor, convert_dense_bf16_to_f32,
    decode_bf16, dense_conversion_preflight_bytes,
};
pub use error::StorageError;
pub use expert::{
    CacheMetrics, ExpertCache, ExpertId, ExpertKey, ExpertLease, ExpertRegistration, ExpertStore,
};
pub use manifest::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ByteOrder, TensorLocation, TensorMetadata,
};
pub use reader::{ArtifactReader, TensorBytes};

/// Calculates SHA-256 for artifact construction and integrity tests.
#[must_use]
pub fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    hash::sha256(bytes)
}
