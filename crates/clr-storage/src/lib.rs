#![doc = "Storage and expert-cache subsystem for colibri-lite-rs."]

mod error;
mod expert;
mod hash;
mod manifest;
mod reader;

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
