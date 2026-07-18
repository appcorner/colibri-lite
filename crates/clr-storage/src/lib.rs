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
#[cfg(feature = "m5-3-instrumentation")]
pub use expert::ExpertPathMetrics;
pub use expert::{
    CacheMetrics, ExpertCache, ExpertId, ExpertKey, ExpertLease, ExpertLoadObservation,
    ExpertRegistration, ExpertStore,
};
pub use hash::Sha256Hasher;
pub use manifest::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ByteOrder, TensorLocation, TensorMetadata,
};
#[cfg(feature = "m5-3-instrumentation")]
pub use reader::ReaderMetrics;
pub use reader::{ArtifactReader, TensorBytes};

/// Calculates SHA-256 for artifact construction and integrity tests.
#[must_use]
pub fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    hash::sha256(bytes)
}
