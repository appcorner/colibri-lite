use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::{ArtifactManifest, StorageError, TensorMetadata, hash::sha256};

/// Validated bytes returned for one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBytes {
    /// Stable manifest tensor name.
    pub name: String,
    /// Exact payload bytes after SHA-256 verification.
    pub bytes: Vec<u8>,
}

/// Portable read-at artifact access rooted at one canonical directory.
#[derive(Debug)]
pub struct ArtifactReader {
    root: PathBuf,
    manifest: ArtifactManifest,
}

impl ArtifactReader {
    /// Creates a reader rooted at an existing artifact directory.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Io`] when the root cannot be canonicalized.
    pub fn open(root: impl AsRef<Path>, manifest: ArtifactManifest) -> Result<Self, StorageError> {
        let requested = root.as_ref();
        let root = requested
            .canonicalize()
            .map_err(|source| StorageError::Io {
                action: "canonicalize artifact root",
                path: requested.to_owned(),
                source,
            })?;
        Ok(Self { root, manifest })
    }

    /// Reads and verifies one tensor payload by name.
    ///
    /// Files are opened for this operation and closed before return.
    ///
    /// # Errors
    ///
    /// Returns a structured error for unknown tensors, escaped paths, file I/O,
    /// truncation, or SHA-256 mismatch.
    pub fn read_tensor(&self, name: &str) -> Result<TensorBytes, StorageError> {
        let tensor = self
            .manifest
            .tensor(name)
            .ok_or_else(|| StorageError::TensorNotFound {
                name: name.to_owned(),
            })?;
        let path = self.canonical_tensor_path(tensor)?;
        let mut file = File::open(&path).map_err(|source| StorageError::Io {
            action: "open tensor file",
            path: path.clone(),
            source,
        })?;
        let file_length = file
            .metadata()
            .map_err(|source| StorageError::Io {
                action: "read tensor file metadata",
                path: path.clone(),
                source,
            })?
            .len();
        let required_end = tensor.location.offset + tensor.location.length;
        if file_length < required_end {
            return Err(StorageError::TruncatedTensor {
                tensor: tensor.name.clone(),
                required_end,
                file_length,
            });
        }
        file.seek(SeekFrom::Start(tensor.location.offset))
            .map_err(|source| StorageError::Io {
                action: "seek tensor file",
                path: path.clone(),
                source,
            })?;
        let length = usize::try_from(tensor.location.length).map_err(|_| {
            StorageError::ByteRangeOverflow {
                tensor: tensor.name.clone(),
            }
        })?;
        let mut bytes = vec![0_u8; length];
        file.read_exact(&mut bytes)
            .map_err(|source| StorageError::Io {
                action: "read tensor bytes",
                path,
                source,
            })?;
        let actual = sha256(&bytes);
        if actual != tensor.sha256 {
            return Err(StorageError::HashMismatch {
                tensor: tensor.name.clone(),
                expected: tensor.sha256,
                actual,
            });
        }
        Ok(TensorBytes {
            name: tensor.name.clone(),
            bytes,
        })
    }

    fn canonical_tensor_path(&self, tensor: &TensorMetadata) -> Result<PathBuf, StorageError> {
        let joined = self.root.join(&tensor.location.path);
        let canonical = joined.canonicalize().map_err(|source| StorageError::Io {
            action: "canonicalize tensor path",
            path: joined,
            source,
        })?;
        if !canonical.starts_with(&self.root) {
            return Err(StorageError::PathEscapesRoot {
                tensor: tensor.name.clone(),
                path: canonical,
            });
        }
        Ok(canonical)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use clr_core::{DataType, TensorShape};

    use super::*;
    use crate::{ARTIFACT_FORMAT_VERSION, ByteOrder, TensorLocation};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("colibri-storage-test-{}-{id}", std::process::id()));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove test directory");
        }
    }

    fn manifest(bytes: &[u8], declared_length: u64, hash: [u8; 32]) -> ArtifactManifest {
        ArtifactManifest::new(
            ARTIFACT_FORMAT_VERSION,
            ByteOrder::Little,
            vec![TensorMetadata {
                name: "layer.weight".to_owned(),
                shape: TensorShape::new([bytes.len() / 4]),
                data_type: DataType::F32,
                location: TensorLocation {
                    path: "weights.bin".into(),
                    offset: 4,
                    length: declared_length,
                },
                sha256: hash,
            }],
        )
        .expect("valid test manifest")
    }

    #[test]
    fn reads_exact_range_and_releases_file_handle() {
        let directory = TestDirectory::new();
        let tensor_bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let mut file_bytes = vec![9_u8; 4];
        file_bytes.extend_from_slice(&tensor_bytes);
        let path = directory.0.join("weights.bin");
        fs::write(&path, file_bytes).expect("write test artifact");
        let reader = ArtifactReader::open(
            &directory.0,
            manifest(&tensor_bytes, 8, sha256(&tensor_bytes)),
        )
        .expect("open reader");

        let output = reader.read_tensor("layer.weight").expect("read tensor");

        assert_eq!(output.name, "layer.weight");
        assert_eq!(output.bytes, tensor_bytes);
        fs::remove_file(&path).expect("Windows handle must be closed after read");
    }

    #[test]
    fn rejects_unknown_truncated_and_corrupted_tensors() {
        let directory = TestDirectory::new();
        let tensor_bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let path = directory.0.join("weights.bin");
        fs::write(&path, [0_u8; 8]).expect("write truncated artifact");
        let reader = ArtifactReader::open(
            &directory.0,
            manifest(&tensor_bytes, 8, sha256(&tensor_bytes)),
        )
        .expect("open reader");

        assert!(matches!(
            reader.read_tensor("missing"),
            Err(StorageError::TensorNotFound { .. })
        ));
        assert!(matches!(
            reader.read_tensor("layer.weight"),
            Err(StorageError::TruncatedTensor { .. })
        ));

        fs::write(&path, [0_u8; 12]).expect("write corrupted artifact");
        assert!(matches!(
            reader.read_tensor("layer.weight"),
            Err(StorageError::HashMismatch { .. })
        ));
    }
}
