use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[cfg(feature = "m5-3-mmap")]
use std::collections::HashMap;

#[cfg(any(feature = "m5-3-instrumentation", feature = "m5-3-mmap"))]
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(feature = "m5-3-mmap")]
use crate::mmap::ReadOnlyMappedShard;
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
    pub buffer_growth_events: u64,
    pub buffer_reuse_count: u64,
    pub bytes_read_into_reusable_buffers: u64,
    pub bytes_copied_after_read: u64,
    pub peak_buffer_capacity: usize,
    pub fallback_allocations: u64,
    pub alignment_failures: u64,
    pub hash_bytes: u64,
    pub open_nanos: u128,
    pub metadata_nanos: u128,
    pub seek_nanos: u128,
    pub read_nanos: u128,
    pub hash_nanos: u128,
    pub mmap_mapping_count: u64,
    pub mmap_shard_reuse_count: u64,
    pub mmap_active_mapping_count: u64,
    pub mmap_peak_mapping_count: u64,
    pub mmap_mapped_virtual_bytes: u64,
    pub mmap_peak_mapped_virtual_bytes: u64,
    pub mmap_mapping_init_nanos: u128,
    pub mmap_first_touch_nanos: u128,
    pub mmap_access_nanos: u128,
    pub mmap_copy_nanos: u128,
    pub mmap_copy_bytes: u64,
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
        self.buffer_growth_events += delta.buffer_growth_events;
        self.buffer_reuse_count += delta.buffer_reuse_count;
        self.bytes_read_into_reusable_buffers += delta.bytes_read_into_reusable_buffers;
        self.bytes_copied_after_read += delta.bytes_copied_after_read;
        self.peak_buffer_capacity = self.peak_buffer_capacity.max(delta.peak_buffer_capacity);
        self.fallback_allocations += delta.fallback_allocations;
        self.alignment_failures += delta.alignment_failures;
        self.hash_bytes += delta.hash_bytes;
        self.open_nanos += delta.open_nanos;
        self.metadata_nanos += delta.metadata_nanos;
        self.seek_nanos += delta.seek_nanos;
        self.read_nanos += delta.read_nanos;
        self.hash_nanos += delta.hash_nanos;
        self.mmap_mapping_count += delta.mmap_mapping_count;
        self.mmap_shard_reuse_count += delta.mmap_shard_reuse_count;
        self.mmap_active_mapping_count = delta.mmap_active_mapping_count;
        self.mmap_peak_mapping_count = self
            .mmap_peak_mapping_count
            .max(delta.mmap_peak_mapping_count);
        self.mmap_mapped_virtual_bytes = delta.mmap_mapped_virtual_bytes;
        self.mmap_peak_mapped_virtual_bytes = self
            .mmap_peak_mapped_virtual_bytes
            .max(delta.mmap_peak_mapped_virtual_bytes);
        self.mmap_mapping_init_nanos += delta.mmap_mapping_init_nanos;
        self.mmap_first_touch_nanos += delta.mmap_first_touch_nanos;
        self.mmap_access_nanos += delta.mmap_access_nanos;
        self.mmap_copy_nanos += delta.mmap_copy_nanos;
        self.mmap_copy_bytes += delta.mmap_copy_bytes;
    }
}

/// Portable read-at artifact access rooted at one canonical directory.
#[derive(Debug)]
pub struct ArtifactReader {
    root: PathBuf,
    manifest: ArtifactManifest,
    #[cfg(feature = "m5-3-instrumentation")]
    metrics: Arc<Mutex<ReaderMetrics>>,
    #[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
    mode: ReaderMode,
    #[cfg(feature = "m5-3-reusable-buffer")]
    reusable_buffer: Mutex<ReusableReadBuffer>,
    #[cfg(feature = "m5-3-mmap")]
    mapped_shards: Mutex<HashMap<PathBuf, Arc<ReadOnlyMappedShard>>>,
}

/// Controlled storage reader mode for the M5.3 storage prototypes.
#[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderMode {
    /// Allocate the exact payload buffer for every read.
    Reference,
    /// Read into one synchronously leased reusable staging buffer.
    #[cfg(feature = "m5-3-reusable-buffer")]
    ReusableAlignedBuffer,
    /// Read exact payloads from lazily mapped, read-only complete shards.
    #[cfg(feature = "m5-3-mmap")]
    MmapReadOnly,
}

#[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
impl ReaderMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference_allocated",
            #[cfg(feature = "m5-3-reusable-buffer")]
            Self::ReusableAlignedBuffer => "reusable_aligned_buffer",
            #[cfg(feature = "m5-3-mmap")]
            Self::MmapReadOnly => "mmap_read_only",
        }
    }
}

#[cfg(feature = "m5-3-reusable-buffer")]
#[derive(Debug, Default)]
struct ReusableReadBuffer {
    bytes: Vec<u8>,
}

#[cfg(feature = "m5-3-reusable-buffer")]
impl ReusableReadBuffer {
    const REQUIRED_ALIGNMENT: usize = 1;

    fn prepare(&mut self, length: usize) -> BufferPreparation {
        let capacity_before = self.bytes.capacity();
        let reused = capacity_before >= length;
        if self.bytes.len() < length {
            self.bytes.resize(length, 0);
        }
        let capacity_after = self.bytes.capacity();
        BufferPreparation {
            reused,
            grew: capacity_after > capacity_before,
            allocated_bytes: capacity_after.saturating_sub(capacity_before),
            capacity: capacity_after,
        }
    }
}

#[cfg(feature = "m5-3-reusable-buffer")]
#[derive(Debug, Clone, Copy)]
struct BufferPreparation {
    reused: bool,
    grew: bool,
    allocated_bytes: usize,
    capacity: usize,
}

#[derive(Debug)]
struct PreparedTensor {
    tensor: TensorMetadata,
    path: PathBuf,
    file: File,
    length: usize,
    #[cfg(feature = "m5-3-instrumentation")]
    open_nanos: u128,
    #[cfg(feature = "m5-3-instrumentation")]
    metadata_nanos: u128,
    #[cfg(feature = "m5-3-instrumentation")]
    seek_nanos: u128,
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
            #[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
            mode: ReaderMode::Reference,
            #[cfg(feature = "m5-3-reusable-buffer")]
            reusable_buffer: Mutex::new(ReusableReadBuffer::default()),
            #[cfg(feature = "m5-3-mmap")]
            mapped_shards: Mutex::new(HashMap::new()),
        })
    }

    /// Opens a reader with an explicitly selected M5.3 prototype mode.
    #[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
    /// Opens an artifact reader with an explicitly selected reader mode.
    ///
    /// # Errors
    ///
    /// Returns a storage error when the artifact root or manifest cannot be
    /// initialized.
    pub fn open_with_mode(
        root: impl AsRef<Path>,
        manifest: ArtifactManifest,
        mode: ReaderMode,
    ) -> Result<Self, StorageError> {
        let mut reader = Self::open(root, manifest)?;
        reader.mode = mode;
        Ok(reader)
    }

    /// Returns the active reader mode for runtime evidence.
    #[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
    #[must_use]
    pub const fn reader_mode_name(&self) -> &'static str {
        self.mode.as_str()
    }

    /// Returns the active reader mode for runtime evidence.
    #[cfg(not(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap")))]
    #[must_use]
    pub const fn reader_mode_name(&self) -> &'static str {
        "reference_allocated"
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
    fn record(&self, delta: &ReaderMetrics) {
        self.metrics
            .lock()
            .expect("reader metrics mutex")
            .add_assign(*delta);
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
        #[cfg(feature = "m5-3-mmap")]
        if self.mode == ReaderMode::MmapReadOnly {
            return self.read_tensor_mmap(name);
        }
        #[cfg(feature = "m5-3-reusable-buffer")]
        if self.mode == ReaderMode::ReusableAlignedBuffer {
            let bytes = self.read_tensor_reusable_arc(name)?;
            return Ok(TensorBytes {
                name: name.to_owned(),
                bytes: bytes.as_ref().to_vec(),
            });
        }
        self.read_tensor_reference(name)
    }

    #[allow(clippy::too_many_lines)]
    fn read_tensor_reference(&self, name: &str) -> Result<TensorBytes, StorageError> {
        let prepared = self.prepare_tensor(name)?;
        let PreparedTensor {
            tensor,
            path,
            mut file,
            length,
            #[cfg(feature = "m5-3-instrumentation")]
            open_nanos,
            #[cfg(feature = "m5-3-instrumentation")]
            metadata_nanos,
            #[cfg(feature = "m5-3-instrumentation")]
            seek_nanos,
        } = prepared;
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
                peak_buffer_capacity: length,
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
            self.record(&delta);
        }
        if actual != tensor.sha256 {
            return Err(StorageError::HashMismatch {
                tensor: tensor.name.clone(),
                expected: tensor.sha256,
                actual,
            });
        }
        Ok(TensorBytes {
            name: tensor.name,
            bytes,
        })
    }

    #[cfg(feature = "m5-3-mmap")]
    fn read_tensor_mmap(&self, name: &str) -> Result<TensorBytes, StorageError> {
        let tensor = self
            .manifest
            .tensor(name)
            .ok_or_else(|| StorageError::TensorNotFound {
                name: name.to_owned(),
            })?
            .clone();
        let path = self.canonical_tensor_path(&tensor)?;
        let required_end = tensor
            .location
            .offset
            .checked_add(tensor.location.length)
            .ok_or_else(|| StorageError::ByteRangeOverflow {
                tensor: tensor.name.clone(),
            })?;
        let (shard, newly_mapped, active_mappings, mapped_virtual_bytes) =
            self.mapped_shard(&path)?;
        let file_length = u64::try_from(shard.len()).unwrap_or(u64::MAX);
        if file_length < required_end {
            return Err(StorageError::TruncatedTensor {
                tensor: tensor.name.clone(),
                required_end,
                file_length,
            });
        }

        let access_started = Instant::now();
        let bytes = shard
            .range(tensor.location.offset, tensor.location.length)
            .ok_or_else(|| StorageError::ByteRangeOverflow {
                tensor: tensor.name.clone(),
            })?;
        let access_nanos = access_started.elapsed().as_nanos();
        let touch_started = Instant::now();
        let actual = sha256(bytes);
        let hash_nanos = touch_started.elapsed().as_nanos();
        let copy_started = Instant::now();
        let owned = bytes.to_vec();
        let copy_nanos = copy_started.elapsed().as_nanos();
        let first_touch_nanos = if newly_mapped {
            touch_started.elapsed().as_nanos()
        } else {
            0
        };
        let length = bytes.len();
        self.record(&ReaderMetrics {
            tensor_reads: 1,
            file_handle_reuse_count: u64::from(!newly_mapped),
            requested_read_bytes: tensor.location.length,
            returned_read_bytes: tensor.location.length,
            buffer_allocation_count: 1,
            allocated_bytes: tensor.location.length,
            copied_bytes: tensor.location.length,
            bytes_copied_after_read: tensor.location.length,
            peak_buffer_capacity: length,
            hash_bytes: tensor.location.length,
            hash_nanos,
            mmap_shard_reuse_count: u64::from(!newly_mapped),
            mmap_active_mapping_count: active_mappings,
            mmap_peak_mapping_count: active_mappings,
            mmap_mapped_virtual_bytes: mapped_virtual_bytes,
            mmap_peak_mapped_virtual_bytes: mapped_virtual_bytes,
            mmap_first_touch_nanos: first_touch_nanos,
            mmap_access_nanos: access_nanos,
            mmap_copy_nanos: copy_nanos,
            mmap_copy_bytes: tensor.location.length,
            ..ReaderMetrics::default()
        });
        if actual != tensor.sha256 {
            return Err(StorageError::HashMismatch {
                tensor: tensor.name,
                expected: tensor.sha256,
                actual,
            });
        }
        Ok(TensorBytes {
            name: tensor.name,
            bytes: owned,
        })
    }

    #[cfg(feature = "m5-3-mmap")]
    fn mapped_shard(
        &self,
        path: &Path,
    ) -> Result<(Arc<ReadOnlyMappedShard>, bool, u64, u64), StorageError> {
        let mut mappings = self
            .mapped_shards
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = mappings.get(path) {
            let active = u64::try_from(mappings.len()).unwrap_or(u64::MAX);
            let virtual_bytes = mappings.values().fold(0_u64, |total, mapping| {
                total.saturating_add(u64::try_from(mapping.len()).unwrap_or(u64::MAX))
            });
            return Ok((Arc::clone(existing), false, active, virtual_bytes));
        }

        let mapping_started = Instant::now();
        let mapping =
            Arc::new(
                ReadOnlyMappedShard::open(path).map_err(|source| StorageError::Io {
                    action: "map read-only expert shard",
                    path: path.to_owned(),
                    source,
                })?,
            );
        let mapping_nanos = mapping_started.elapsed().as_nanos();
        mappings.insert(path.to_owned(), Arc::clone(&mapping));
        let active = u64::try_from(mappings.len()).unwrap_or(u64::MAX);
        let virtual_bytes = mappings.values().fold(0_u64, |total, item| {
            total.saturating_add(u64::try_from(item.len()).unwrap_or(u64::MAX))
        });
        drop(mappings);
        self.record(&ReaderMetrics {
            file_open_count: 1,
            mmap_mapping_count: 1,
            mmap_active_mapping_count: active,
            mmap_peak_mapping_count: active,
            mmap_mapped_virtual_bytes: virtual_bytes,
            mmap_peak_mapped_virtual_bytes: virtual_bytes,
            mmap_mapping_init_nanos: mapping_nanos,
            ..ReaderMetrics::default()
        });
        Ok((mapping, true, active, virtual_bytes))
    }

    #[cfg(feature = "m5-3-reusable-buffer")]
    pub(crate) fn read_tensor_reusable_arc(&self, name: &str) -> Result<Arc<[u8]>, StorageError> {
        let prepared = self.prepare_tensor(name)?;
        let PreparedTensor {
            tensor,
            path,
            mut file,
            length,
            open_nanos,
            metadata_nanos,
            seek_nanos,
        } = prepared;
        let mut reusable_buffer = self
            .reusable_buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let preparation = reusable_buffer.prepare(length);
        let alignment_failure = reusable_buffer
            .bytes
            .as_ptr()
            .align_offset(ReusableReadBuffer::REQUIRED_ALIGNMENT)
            != 0;
        let mut delta = ReaderMetrics {
            requested_read_bytes: tensor.location.length,
            buffer_growth_events: u64::from(preparation.grew),
            buffer_reuse_count: u64::from(preparation.reused),
            bytes_read_into_reusable_buffers: tensor.location.length,
            allocated_bytes: u64::try_from(preparation.allocated_bytes).unwrap_or(u64::MAX),
            buffer_allocation_count: u64::from(preparation.grew),
            peak_buffer_capacity: preparation.capacity,
            alignment_failures: u64::from(alignment_failure),
            ..ReaderMetrics::default()
        };
        let buffer = &mut reusable_buffer.bytes[..length];
        let read_started = Instant::now();
        let mut counting_file = CountingReader {
            file: &mut file,
            metrics: ReaderMetrics::default(),
        };
        counting_file
            .read_exact(buffer)
            .map_err(|source| StorageError::Io {
                action: "read tensor bytes into reusable buffer",
                path,
                source,
            })?;
        let read_nanos = read_started.elapsed().as_nanos();
        let hash_started = Instant::now();
        let actual = sha256(buffer);
        let hash_nanos = hash_started.elapsed().as_nanos();
        delta.tensor_reads = 1;
        delta.file_open_count = 1;
        delta.metadata_count = 1;
        delta.seek_count = 1;
        delta.read_call_count = counting_file.metrics.read_call_count;
        delta.returned_read_bytes = counting_file.metrics.returned_read_bytes;
        delta.copied_bytes = tensor.location.length;
        delta.bytes_copied_after_read = tensor.location.length;
        delta.open_nanos = open_nanos;
        delta.metadata_nanos = metadata_nanos;
        delta.seek_nanos = seek_nanos;
        delta.read_nanos = read_nanos;
        delta.hash_bytes = tensor.location.length;
        delta.hash_nanos = hash_nanos;
        if actual != tensor.sha256 {
            self.record(&delta);
            return Err(StorageError::HashMismatch {
                tensor: tensor.name,
                expected: tensor.sha256,
                actual,
            });
        }
        let bytes: Arc<[u8]> = Arc::from(&buffer[..]);
        self.record(&delta);
        Ok(bytes)
    }

    fn prepare_tensor(&self, name: &str) -> Result<PreparedTensor, StorageError> {
        let tensor = self
            .manifest
            .tensor(name)
            .ok_or_else(|| StorageError::TensorNotFound {
                name: name.to_owned(),
            })?;
        let path = self.canonical_tensor_path(tensor)?;
        let tensor = tensor.clone();
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
        Ok(PreparedTensor {
            tensor,
            path,
            file,
            length,
            #[cfg(feature = "m5-3-instrumentation")]
            open_nanos,
            #[cfg(feature = "m5-3-instrumentation")]
            metadata_nanos,
            #[cfg(feature = "m5-3-instrumentation")]
            seek_nanos,
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

    #[cfg(feature = "m5-3-reusable-buffer")]
    #[test]
    fn reusable_buffer_is_byte_equivalent_and_grows_without_shrinking() {
        let directory = TestDirectory::new();
        let first = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let second = [9_u8, 10, 11, 12];
        let third = [13_u8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
        let mut file_bytes = vec![0_u8; 4];
        file_bytes.extend_from_slice(&first);
        file_bytes.extend_from_slice(&second);
        file_bytes.extend_from_slice(&third);
        fs::write(directory.0.join("weights.bin"), file_bytes).expect("write variable artifact");
        let metadata = vec![
            TensorMetadata {
                name: "first".to_owned(),
                shape: TensorShape::new([2]),
                data_type: DataType::F32,
                location: TensorLocation {
                    path: "weights.bin".into(),
                    offset: 4,
                    length: 8,
                },
                sha256: sha256(&first),
            },
            TensorMetadata {
                name: "second".to_owned(),
                shape: TensorShape::new([1]),
                data_type: DataType::F32,
                location: TensorLocation {
                    path: "weights.bin".into(),
                    offset: 12,
                    length: 4,
                },
                sha256: sha256(&second),
            },
            TensorMetadata {
                name: "third".to_owned(),
                shape: TensorShape::new([3]),
                data_type: DataType::F32,
                location: TensorLocation {
                    path: "weights.bin".into(),
                    offset: 16,
                    length: 12,
                },
                sha256: sha256(&third),
            },
        ];
        let manifest = ArtifactManifest::new(ARTIFACT_FORMAT_VERSION, ByteOrder::Little, metadata)
            .expect("variable-size manifest");
        let reference = ArtifactReader::open(&directory.0, manifest.clone()).expect("reference");
        let reusable = ArtifactReader::open_with_mode(
            &directory.0,
            manifest,
            ReaderMode::ReusableAlignedBuffer,
        )
        .expect("reusable");

        for name in ["first", "second", "third", "first"] {
            assert_eq!(
                reusable.read_tensor(name).expect("reusable read"),
                reference.read_tensor(name).expect("reference read")
            );
        }
        assert_eq!(reusable.reader_mode_name(), "reusable_aligned_buffer");
        let metrics = reusable.metrics();
        assert_eq!(metrics.tensor_reads, 4);
        assert_eq!(metrics.buffer_growth_events, 2);
        assert_eq!(metrics.buffer_reuse_count, 2);
        assert_eq!(metrics.buffer_allocation_count, 2);
        assert_eq!(metrics.bytes_read_into_reusable_buffers, 32);
        assert_eq!(metrics.bytes_copied_after_read, 32);
        assert!(metrics.peak_buffer_capacity >= 12);
        assert_eq!(metrics.alignment_failures, 0);
        assert_eq!(ReusableReadBuffer::REQUIRED_ALIGNMENT, 1);
    }

    #[cfg(feature = "m5-3-reusable-buffer")]
    #[test]
    fn reusable_buffer_recovers_after_truncated_read_without_leaking_a_handle() {
        let directory = TestDirectory::new();
        let payload = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let path = directory.0.join("weights.bin");
        fs::write(&path, [0_u8; 4]).expect("write truncated artifact");
        let reader = ArtifactReader::open_with_mode(
            &directory.0,
            manifest(&payload, 8, sha256(&payload)),
            ReaderMode::ReusableAlignedBuffer,
        )
        .expect("open reusable reader");
        assert!(matches!(
            reader.read_tensor("layer.weight"),
            Err(StorageError::TruncatedTensor { .. })
        ));
        let mut repaired = vec![0_u8; 4];
        repaired.extend_from_slice(&payload);
        fs::write(&path, repaired).expect("repair artifact");
        assert_eq!(
            reader
                .read_tensor("layer.weight")
                .expect("recovered read")
                .bytes,
            payload
        );
        fs::remove_file(path).expect("reusable reader releases file handle");
    }

    #[cfg(feature = "m5-3-mmap")]
    #[test]
    fn mmap_reader_is_byte_equivalent_across_shards_and_reuses_mappings() {
        let directory = TestDirectory::new();
        let first = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let second = [9_u8, 10, 11, 12];
        let third = [13_u8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
        let mut shard_a = vec![0xaa_u8; 4];
        shard_a.extend_from_slice(&first);
        shard_a.extend_from_slice(&second);
        let mut shard_b = vec![0xbb_u8; 2];
        shard_b.extend_from_slice(&third);
        fs::write(directory.0.join("shard-a.bin"), shard_a).expect("write shard a");
        fs::write(directory.0.join("shard-b.bin"), shard_b).expect("write shard b");

        let manifest = ArtifactManifest::new(
            ARTIFACT_FORMAT_VERSION,
            ByteOrder::Little,
            vec![
                TensorMetadata {
                    name: "a-first".to_owned(),
                    shape: TensorShape::new([2]),
                    data_type: DataType::F32,
                    location: TensorLocation {
                        path: "shard-a.bin".into(),
                        offset: 4,
                        length: 8,
                    },
                    sha256: sha256(&first),
                },
                TensorMetadata {
                    name: "a-second".to_owned(),
                    shape: TensorShape::new([1]),
                    data_type: DataType::F32,
                    location: TensorLocation {
                        path: "shard-a.bin".into(),
                        offset: 12,
                        length: 4,
                    },
                    sha256: sha256(&second),
                },
                TensorMetadata {
                    name: "b-third".to_owned(),
                    shape: TensorShape::new([3]),
                    data_type: DataType::F32,
                    location: TensorLocation {
                        path: "shard-b.bin".into(),
                        offset: 2,
                        length: 12,
                    },
                    sha256: sha256(&third),
                },
            ],
        )
        .expect("valid multi-shard manifest");
        let reference = ArtifactReader::open(&directory.0, manifest.clone()).expect("reference");
        let mmap = ArtifactReader::open_with_mode(&directory.0, manifest, ReaderMode::MmapReadOnly)
            .expect("mmap");

        for name in [
            "a-first", "b-third", "a-second", "a-first", "b-third", "b-third",
        ] {
            assert_eq!(
                mmap.read_tensor(name).expect("mmap read"),
                reference.read_tensor(name).expect("reference read")
            );
        }
        assert_eq!(mmap.reader_mode_name(), "mmap_read_only");
        let metrics = mmap.metrics();
        assert_eq!(metrics.tensor_reads, 6);
        assert_eq!(metrics.file_open_count, 2);
        assert_eq!(metrics.file_handle_reuse_count, 4);
        assert_eq!(metrics.read_call_count, 0);
        assert_eq!(metrics.mmap_mapping_count, 2);
        assert_eq!(metrics.mmap_shard_reuse_count, 4);
        assert_eq!(metrics.mmap_active_mapping_count, 2);
        assert_eq!(metrics.mmap_peak_mapping_count, 2);
        assert_eq!(metrics.mmap_mapped_virtual_bytes, 30);
        assert_eq!(metrics.mmap_peak_mapped_virtual_bytes, 30);
        assert_eq!(metrics.requested_read_bytes, 56);
        assert_eq!(metrics.returned_read_bytes, 56);
        assert_eq!(metrics.mmap_copy_bytes, 56);
    }

    #[cfg(feature = "m5-3-mmap")]
    #[test]
    fn mmap_reader_rejects_truncated_and_missing_shards() {
        let directory = TestDirectory::new();
        let payload = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let path = directory.0.join("weights.bin");
        fs::write(&path, [0_u8; 4]).expect("write truncated shard");
        let reader = ArtifactReader::open_with_mode(
            &directory.0,
            manifest(&payload, 8, sha256(&payload)),
            ReaderMode::MmapReadOnly,
        )
        .expect("open mmap reader");
        assert!(matches!(
            reader.read_tensor("layer.weight"),
            Err(StorageError::TruncatedTensor { .. })
        ));
        drop(reader);

        fs::remove_file(&path).expect("remove truncated shard");
        let missing = ArtifactReader::open_with_mode(
            &directory.0,
            manifest(&payload, 8, sha256(&payload)),
            ReaderMode::MmapReadOnly,
        )
        .expect("open mmap reader for missing shard");
        assert!(matches!(
            missing.read_tensor("layer.weight"),
            Err(StorageError::Io { .. })
        ));
    }

    #[cfg(feature = "m5-3-mmap")]
    #[test]
    fn mmap_reader_releases_file_and_mapping_after_drop() {
        let directory = TestDirectory::new();
        let payload = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let path = directory.0.join("weights.bin");
        let mut file_bytes = vec![0_u8; 4];
        file_bytes.extend_from_slice(&payload);
        fs::write(&path, file_bytes).expect("write test shard");
        {
            let reader = ArtifactReader::open_with_mode(
                &directory.0,
                manifest(&payload, 8, sha256(&payload)),
                ReaderMode::MmapReadOnly,
            )
            .expect("open mmap reader");
            assert_eq!(
                reader.read_tensor("layer.weight").expect("mmap read").bytes,
                payload
            );
        }
        fs::remove_file(path).expect("mapping and file handle must be released after drop");
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
