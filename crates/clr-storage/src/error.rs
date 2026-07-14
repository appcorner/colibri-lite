use std::{fmt, io, path::PathBuf};

use crate::ByteOrder;

/// Errors produced while validating or reading model artifacts.
#[derive(Debug)]
pub enum StorageError {
    /// The manifest format version is not supported.
    UnsupportedFormatVersion { actual: u32 },
    /// The artifact byte order cannot be consumed by the current contract.
    UnsupportedByteOrder { actual: ByteOrder },
    /// A tensor name occurs more than once.
    DuplicateTensorName { name: String },
    /// A tensor path is empty, absolute, or contains traversal components.
    InvalidRelativePath { tensor: String, path: PathBuf },
    /// A tensor byte range overflows `u64`.
    ByteRangeOverflow { tensor: String },
    /// Shape/dtype byte count does not match declared storage length.
    TensorLengthMismatch {
        tensor: String,
        expected: u64,
        actual: u64,
    },
    /// Two tensor ranges overlap within one artifact file.
    OverlappingTensorRanges { first: String, second: String },
    /// A requested tensor does not exist in the manifest.
    TensorNotFound { name: String },
    /// A canonical artifact path escaped the configured root.
    PathEscapesRoot { tensor: String, path: PathBuf },
    /// The backing file is shorter than the declared tensor range.
    TruncatedTensor {
        tensor: String,
        required_end: u64,
        file_length: u64,
    },
    /// Tensor bytes do not match the manifest digest.
    HashMismatch {
        tensor: String,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// A filesystem operation failed.
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
}

impl fmt::Display for StorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFormatVersion { actual } => {
                write!(formatter, "unsupported artifact format version {actual}")
            }
            Self::UnsupportedByteOrder { actual } => {
                write!(formatter, "unsupported artifact byte order {actual:?}")
            }
            Self::DuplicateTensorName { name } => {
                write!(formatter, "duplicate tensor name '{name}'")
            }
            Self::InvalidRelativePath { tensor, path } => write!(
                formatter,
                "invalid relative path '{}' for tensor '{tensor}'",
                path.display()
            ),
            Self::ByteRangeOverflow { tensor } => {
                write!(formatter, "byte range overflows for tensor '{tensor}'")
            }
            Self::TensorLengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor '{tensor}' length mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::OverlappingTensorRanges { first, second } => {
                write!(formatter, "tensor ranges overlap: '{first}' and '{second}'")
            }
            Self::TensorNotFound { name } => write!(formatter, "tensor '{name}' not found"),
            Self::PathEscapesRoot { tensor, path } => write!(
                formatter,
                "tensor '{tensor}' path escapes artifact root: '{}'",
                path.display()
            ),
            Self::TruncatedTensor {
                tensor,
                required_end,
                file_length,
            } => write!(
                formatter,
                "tensor '{tensor}' requires byte {required_end}, file length is {file_length}"
            ),
            Self::HashMismatch { tensor, .. } => {
                write!(formatter, "SHA-256 mismatch for tensor '{tensor}'")
            }
            Self::Io {
                action,
                path,
                source,
            } => write!(
                formatter,
                "failed to {action} '{}': {source}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
