use std::{collections::HashSet, path::Component, path::PathBuf};

use clr_core::{DataType, TensorShape};

use crate::StorageError;

/// The only artifact manifest version supported by this runtime revision.
pub const ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Byte order of numeric tensor payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Least-significant byte first.
    Little,
    /// Most-significant byte first.
    Big,
}

/// File and byte range containing one tensor payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLocation {
    /// Artifact-root-relative file path.
    pub path: PathBuf,
    /// Byte offset from the start of the file.
    pub offset: u64,
    /// Tensor payload length in bytes.
    pub length: u64,
}

/// Versioned metadata required to validate and locate one dense tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    /// Stable unique tensor name.
    pub name: String,
    /// Dense row-major tensor shape.
    pub shape: TensorShape,
    /// Dense element type.
    pub data_type: DataType,
    /// File and byte-range location.
    pub location: TensorLocation,
    /// SHA-256 of exactly the tensor payload bytes.
    pub sha256: [u8; 32],
}

/// Validated, versioned collection of tensor locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactManifest {
    format_version: u32,
    byte_order: ByteOrder,
    tensors: Vec<TensorMetadata>,
}

impl ArtifactManifest {
    /// Validates and creates an artifact manifest.
    ///
    /// # Errors
    ///
    /// Rejects unsupported versions/endianness, duplicate names, unsafe paths,
    /// overflowing/mismatched byte ranges, and overlapping tensor ranges.
    pub fn new(
        format_version: u32,
        byte_order: ByteOrder,
        tensors: Vec<TensorMetadata>,
    ) -> Result<Self, StorageError> {
        if format_version != ARTIFACT_FORMAT_VERSION {
            return Err(StorageError::UnsupportedFormatVersion {
                actual: format_version,
            });
        }
        if byte_order != ByteOrder::Little {
            return Err(StorageError::UnsupportedByteOrder { actual: byte_order });
        }
        validate_tensors(&tensors)?;
        Ok(Self {
            format_version,
            byte_order,
            tensors,
        })
    }

    /// Returns the artifact format version.
    #[must_use]
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the artifact numeric byte order.
    #[must_use]
    pub const fn byte_order(&self) -> ByteOrder {
        self.byte_order
    }

    /// Returns validated tensors in manifest order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorMetadata] {
        &self.tensors
    }

    pub(crate) fn tensor(&self, name: &str) -> Option<&TensorMetadata> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

fn validate_tensors(tensors: &[TensorMetadata]) -> Result<(), StorageError> {
    let mut names = HashSet::with_capacity(tensors.len());
    for tensor in tensors {
        if !names.insert(tensor.name.as_str()) {
            return Err(StorageError::DuplicateTensorName {
                name: tensor.name.clone(),
            });
        }
        if tensor.name.is_empty() || !valid_relative_path(&tensor.location.path) {
            return Err(StorageError::InvalidRelativePath {
                tensor: tensor.name.clone(),
                path: tensor.location.path.clone(),
            });
        }
        tensor
            .location
            .offset
            .checked_add(tensor.location.length)
            .ok_or_else(|| StorageError::ByteRangeOverflow {
                tensor: tensor.name.clone(),
            })?;
        let expected = u64::try_from(tensor.shape.byte_count(tensor.data_type).map_err(|_| {
            StorageError::ByteRangeOverflow {
                tensor: tensor.name.clone(),
            }
        })?)
        .map_err(|_| StorageError::ByteRangeOverflow {
            tensor: tensor.name.clone(),
        })?;
        if expected != tensor.location.length {
            return Err(StorageError::TensorLengthMismatch {
                tensor: tensor.name.clone(),
                expected,
                actual: tensor.location.length,
            });
        }
    }

    for (index, left) in tensors.iter().enumerate() {
        for right in &tensors[index + 1..] {
            if left.location.path == right.location.path && overlaps(left, right) {
                return Err(StorageError::OverlappingTensorRanges {
                    first: left.name.clone(),
                    second: right.name.clone(),
                });
            }
        }
    }
    Ok(())
}

fn valid_relative_path(path: &std::path::Path) -> bool {
    let mut components = path.components();
    let Some(first) = components.next() else {
        return false;
    };
    matches!(first, Component::Normal(_))
        && components.all(|component| matches!(component, Component::Normal(_)))
}

fn overlaps(left: &TensorMetadata, right: &TensorMetadata) -> bool {
    if left.location.length == 0 || right.location.length == 0 {
        return false;
    }
    let left_end = left.location.offset + left.location.length;
    let right_end = right.location.offset + right.location.length;
    left.location.offset < right_end && right.location.offset < left_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256;

    fn tensor(name: &str, path: &str, offset: u64, shape: &[usize]) -> TensorMetadata {
        TensorMetadata {
            name: name.to_owned(),
            shape: TensorShape::new(shape.to_vec()),
            data_type: DataType::F32,
            location: TensorLocation {
                path: path.into(),
                offset,
                length: u64::try_from(shape.iter().product::<usize>() * 4).expect("test length"),
            },
            sha256: sha256(&[]),
        }
    }

    #[test]
    fn accepts_non_overlapping_little_endian_manifest() {
        let manifest = ArtifactManifest::new(
            ARTIFACT_FORMAT_VERSION,
            ByteOrder::Little,
            vec![
                tensor("first", "weights.bin", 0, &[2]),
                tensor("second", "weights.bin", 8, &[1]),
            ],
        )
        .expect("valid manifest");

        assert_eq!(manifest.format_version(), 1);
        assert_eq!(manifest.byte_order(), ByteOrder::Little);
        assert_eq!(manifest.tensors().len(), 2);
    }

    #[test]
    fn rejects_version_endian_duplicates_and_paths() {
        assert!(matches!(
            ArtifactManifest::new(2, ByteOrder::Little, Vec::new()),
            Err(StorageError::UnsupportedFormatVersion { actual: 2 })
        ));
        assert!(matches!(
            ArtifactManifest::new(1, ByteOrder::Big, Vec::new()),
            Err(StorageError::UnsupportedByteOrder {
                actual: ByteOrder::Big
            })
        ));
        assert!(matches!(
            ArtifactManifest::new(
                1,
                ByteOrder::Little,
                vec![
                    tensor("same", "a.bin", 0, &[1]),
                    tensor("same", "b.bin", 0, &[1])
                ]
            ),
            Err(StorageError::DuplicateTensorName { .. })
        ));
        assert!(matches!(
            ArtifactManifest::new(
                1,
                ByteOrder::Little,
                vec![tensor("escape", "../weights.bin", 0, &[1])]
            ),
            Err(StorageError::InvalidRelativePath { .. })
        ));
    }

    #[test]
    fn rejects_length_overflow_and_overlap() {
        let mut wrong_length = tensor("wrong", "a.bin", 0, &[2]);
        wrong_length.location.length = 4;
        assert!(matches!(
            ArtifactManifest::new(1, ByteOrder::Little, vec![wrong_length]),
            Err(StorageError::TensorLengthMismatch {
                expected: 8,
                actual: 4,
                ..
            })
        ));

        let mut overflow = tensor("overflow", "a.bin", 0, &[1]);
        overflow.location.offset = u64::MAX;
        assert!(matches!(
            ArtifactManifest::new(1, ByteOrder::Little, vec![overflow]),
            Err(StorageError::ByteRangeOverflow { .. })
        ));

        assert!(matches!(
            ArtifactManifest::new(
                1,
                ByteOrder::Little,
                vec![
                    tensor("first", "a.bin", 0, &[2]),
                    tensor("second", "a.bin", 4, &[1])
                ]
            ),
            Err(StorageError::OverlappingTensorRanges { .. })
        ));
    }
}
