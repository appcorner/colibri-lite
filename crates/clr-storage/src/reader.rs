use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[cfg(feature = "m5-3-instrumentation")]
use std::{
    sync::{Arc, Mutex},
    time::Instant,
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

/// Counters and timing estimates for the current portable reader path.
///
/// The read counter is at the Rust `Read::read` boundary. It is not an OS
/// kernel-syscall counter. Timing fields are operational evidence only and
/// must not be included in deterministic artifact or trace hashes.
#[cfg(feature = "m5-3-instrumentation")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReaderMetrics {
    pub tensor_reads: u64,
    pub file_open_count: u64,
    pub file_handle_reuse_count: u64,
    pub metadata_count: u64,
    pub seek_count: u64,
    pub read_call_count: u64,
    pub requested_read_bytes: u64,
    pub returned_read_bytes: u64,
    pub buffer_allocation_count: u64,
    pub allocated_bytes: u64,
    pub copied_bytes: u64,
    pub hash_bytes: u64,
    pub open_nanos: u128,
    pub metadata_nanos: u128,
    pub seek_nanos: u128,
    pub read_nanos: u128,
    pub hash_nanos: u128,
}

#[cfg(feature = "m5-3-instrumentation")]
impl ReaderMetrics {
    fn add_assign(&mut self, delta: Self) {
        self.tensor_reads += delta.tensor_reads;
        self.file_open_count += delta.file_open_count;
        self.file_handle_reuse_count += delta.file_handle_reuse_count;
        self.metadata_count += delta.metadata_count;
        self.seek_count += delta.seek_count;
        self.read_call_count += delta.read_call_count;
        self.requested_read_bytes += delta.requested_read_bytes;
        self.returned_read_bytes += delta.returned_read_bytes;
        self.buffer_allocation_count += delta.buffer_allocation_count;
        self.allocated_bytes += delta.allocated_bytes;
        self.copied_bytes += delta.copied_bytes;
        self.hash_bytes += delta.hash_bytes;
        self.open_nanos += delta.open_nanos;
        self.metadata_nanos += delta.metadata_nanos;
        self.seek_nanos += delta.seek_nanos;
        self.read_nanos += delta.read_nanos;
        self.hash_nanos += delta.hash_nanos;
    }
}

/// Portable read-at artifact access rooted at one canonical directory.
#[derive(Debug)]
pub struct ArtifactReader {
    root: PathBuf,
    manifest: ArtifactManifest,
    #[cfg(feature = "m5-3-instrumentation")]
    metrics: Arc<Mutex<ReaderMetrics>>,
}

#[cfg(feature = "m5-3-instrumentation")]
#[derive(Debug)]
struct CountingReader<'a> {
    file: &'a mut File,
    metrics: ReaderMetrics,
}

#[cfg(feature = "m5-3-instrumentation")]
impl Read for CountingReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let result = self.file.read(buffer);
        self.metrics.read_call_count += 1;
        if let Ok(returned) = result {
            self.metrics.returned_read_bytes += u64::try_from(returned).unwrap_or(u64::MAX);
        }
        result
    }
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
        Ok(Self {
            root,
            manifest,
            #[cfg(feature = "m5-3-instrumentation")]
            metrics: Arc::new(Mutex::new(ReaderMetrics::default())),
        })
    }

    /// Returns cumulative instrumentation metrics for the current reader.
    #[cfg(feature = "m5-3-instrumentation")]
    #[must_use]
    ///
    /// # Panics
    ///
    /// Panics only if the instrumentation mutex was poisoned by a prior
    /// instrumentation failure.
    pub fn metrics(&self) -> ReaderMetrics {
        *self.metrics.lock().expect("reader metrics mutex")
    }

    #[cfg(feature = "m5-3-instrumentation")]
    fn record(&self, delta: ReaderMetrics) {
        self.metrics
            .lock()
            .expect("reader metrics mutex")
            .add_assign(delta);
    }

    /// Reads and verifies one tensor payload by name.
    ///
    /// Files are opened for this operation and closed before return.
    ///
    /// # Errors
    ///
    /// Returns a structured error for unknown tensors, escaped paths, file I/O,
    /// truncation, or SHA-256 mismatch.
    #[allow(clippy::too_many_lines)]
    pub fn read_tensor(&self, name: &str) -> Result<TensorBytes, StorageError> {
        let tensor = self
            .manifest
            .tensor(name)
            .ok_or_else(|| StorageError::TensorNotFound {
                name: name.to_owned(),
            })?;
        let path = self.canonical_tensor_path(tensor)?;
        #[cfg(feature = "m5-3-instrumentation")]
        let open_started = Instant::now();
        let mut file = File::open(&path).map_err(|source| StorageError::Io {
            action: "open tensor file",
            path: path.clone(),
            source,
        })?;
        #[cfg(feature = "m5-3-instrumentation")]
        let open_nanos = open_started.elapsed().as_nanos();
        #[cfg(feature = "m5-3-instrumentation")]
        let metadata_started = Instant::now();
        let file_length = file
            .metadata()
            .map_err(|source| StorageError::Io {
                action: "read tensor file metadata",
                path: path.clone(),
                source,
            })?
            .len();
        #[cfg(feature = "m5-3-instrumentation")]
        let metadata_nanos = metadata_started.elapsed().as_nanos();
        let required_end = tensor.location.offset + tensor.location.length;
        if file_length < required_end {
            return Err(StorageError::TruncatedTensor {
                tensor: tensor.name.clone(),
                required_end,
                file_length,
            });
        }
        #[cfg(feature = "m5-3-instrumentation")]
        let seek_started = Instant::now();
        file.seek(SeekFrom::Start(tensor.location.offset))
            .map_err(|source| StorageError::Io {
                action: "seek tensor file",
                path: path.clone(),
                source,
            })?;
        #[cfg(feature = "m5-3-instrumentation")]
        let seek_nanos = seek_started.elapsed().as_nanos();
        let length = usize::try_from(tensor.location.length).map_err(|_| {
            StorageError::ByteRangeOverflow {
                tensor: tensor.name.clone(),
            }
        })?;
        let mut bytes = vec![0_u8; length];
        #[cfg(feature = "m5-3-instrumentation")]
        let read_started = Instant::now();
        #[cfg(feature = "m5-3-instrumentation")]
        let mut counting_file = CountingReader {
            file: &mut file,
            metrics: ReaderMetrics {
                requested_read_bytes: tensor.location.length,
                buffer_allocation_count: 1,
                allocated_bytes: tensor.location.length,
                ..ReaderMetrics::default()
            },
        };
        #[cfg(feature = "m5-3-instrumentation")]
        let read_result = counting_file.read_exact(&mut bytes);
        #[cfg(not(feature = "m5-3-instrumentation"))]
        let read_result = file.read_exact(&mut bytes);
        read_result.map_err(|source| StorageError::Io {
            action: "read tensor bytes",
            path,
            source,
        })?;
        #[cfg(feature = "m5-3-instrumentation")]
        let read_nanos = read_started.elapsed().as_nanos();
        #[cfg(feature = "m5-3-instrumentation")]
        let mut delta = counting_file.metrics;
        #[cfg(feature = "m5-3-instrumentation")]
        {
            delta.tensor_reads = 1;
            delta.file_open_count = 1;
            delta.metadata_count = 1;
            delta.seek_count = 1;
            delta.copied_bytes = delta.returned_read_bytes;
            delta.open_nanos = open_nanos;
            delta.metadata_nanos = metadata_nanos;
            delta.seek_nanos = seek_nanos;
            delta.read_nanos = read_nanos;
        }
        #[cfg(feature = "m5-3-instrumentation")]
        let hash_started = Instant::now();
        let actual = sha256(&bytes);
        #[cfg(feature = "m5-3-instrumentation")]
        {
            delta.hash_bytes = tensor.location.length;
            delta.hash_nanos = hash_started.elapsed().as_nanos();
            self.record(delta);
        }
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

    #[cfg(feature = "m5-3-instrumentation")]
    #[test]
    fn instrumentation_counts_one_portable_read_without_handle_reuse() {
        let directory = TestDirectory::new();
        let tensor_bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let mut file_bytes = vec![9_u8; 4];
        file_bytes.extend_from_slice(&tensor_bytes);
        fs::write(directory.0.join("weights.bin"), file_bytes).expect("write test artifact");
        let reader = ArtifactReader::open(
            &directory.0,
            manifest(&tensor_bytes, 8, sha256(&tensor_bytes)),
        )
        .expect("open reader");

        reader.read_tensor("layer.weight").expect("read tensor");

        let metrics = reader.metrics();
        assert_eq!(metrics.tensor_reads, 1);
        assert_eq!(metrics.file_open_count, 1);
        assert_eq!(metrics.file_handle_reuse_count, 0);
        assert_eq!(metrics.metadata_count, 1);
        assert_eq!(metrics.seek_count, 1);
        assert_eq!(metrics.read_call_count, 1);
        assert_eq!(metrics.requested_read_bytes, 8);
        assert_eq!(metrics.returned_read_bytes, 8);
        assert_eq!(metrics.buffer_allocation_count, 1);
        assert_eq!(metrics.allocated_bytes, 8);
        assert_eq!(metrics.copied_bytes, 8);
        assert_eq!(metrics.hash_bytes, 8);
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
