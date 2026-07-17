use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{ArtifactReader, StorageError};

/// Stable expert number within one sparse layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertId(pub u32);

/// Stable cache key for one layer's complete expert payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    /// Zero-based decoder layer index.
    pub layer_index: u32,
    /// Expert number within the layer.
    pub expert_id: ExpertId,
}

/// Cumulative cache and I/O evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub loads: u64,
    pub evictions: u64,
    pub resident_bytes: usize,
    pub peak_resident_bytes: usize,
    pub bytes_read: u64,
}

/// Observation of one completed expert-store request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertLoadObservation {
    pub key: ExpertKey,
    pub payload_bytes: usize,
    pub cache_hit: bool,
    pub loaded: bool,
    pub evictions: u64,
}

/// Pinned access to one cached expert payload.
#[derive(Debug, Clone)]
pub struct ExpertLease {
    key: ExpertKey,
    bytes: Arc<[u8]>,
}

impl ExpertLease {
    #[must_use]
    pub const fn key(&self) -> ExpertKey {
        self.key
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug)]
struct CacheEntry {
    bytes: Arc<[u8]>,
    last_used: u64,
}

/// Deterministic byte-budgeted LRU cache with lease-based pinning.
#[derive(Debug)]
pub struct ExpertCache {
    budget: usize,
    clock: u64,
    entries: HashMap<ExpertKey, CacheEntry>,
    metrics: CacheMetrics,
}

impl ExpertCache {
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            clock: 0,
            entries: HashMap::new(),
            metrics: CacheMetrics::default(),
        }
    }

    /// Returns a cached expert or loads and admits it under the byte budget.
    ///
    /// # Errors
    ///
    /// Returns a loader error, [`StorageError::ExpertExceedsBudget`], or
    /// [`StorageError::CacheBudgetExhausted`] when pinned entries prevent room.
    pub fn get_or_load<F>(&mut self, key: ExpertKey, loader: F) -> Result<ExpertLease, StorageError>
    where
        F: FnOnce() -> Result<Vec<u8>, StorageError>,
    {
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = self.clock;
            self.metrics.hits += 1;
            return Ok(ExpertLease {
                key,
                bytes: Arc::clone(&entry.bytes),
            });
        }
        self.metrics.misses += 1;
        let payload = loader()?;
        let length = payload.len();
        if length > self.budget {
            return Err(StorageError::ExpertExceedsBudget {
                key,
                bytes: length,
                budget: self.budget,
            });
        }
        while self.metrics.resident_bytes + length > self.budget {
            let candidate = self
                .entries
                .iter()
                .filter(|(_, entry)| Arc::strong_count(&entry.bytes) == 1)
                .min_by_key(|(candidate_key, entry)| (entry.last_used, **candidate_key))
                .map(|(candidate_key, _)| *candidate_key);
            let Some(candidate) = candidate else {
                return Err(StorageError::CacheBudgetExhausted {
                    requested: length,
                    budget: self.budget,
                    resident: self.metrics.resident_bytes,
                });
            };
            let Some(removed) = self.entries.remove(&candidate) else {
                return Err(StorageError::CacheBudgetExhausted {
                    requested: length,
                    budget: self.budget,
                    resident: self.metrics.resident_bytes,
                });
            };
            self.metrics.resident_bytes -= removed.bytes.len();
            self.metrics.evictions += 1;
        }
        let bytes: Arc<[u8]> = payload.into();
        self.metrics.loads += 1;
        self.metrics.bytes_read += u64::try_from(length).unwrap_or(u64::MAX);
        self.metrics.resident_bytes += length;
        self.metrics.peak_resident_bytes = self
            .metrics
            .peak_resident_bytes
            .max(self.metrics.resident_bytes);
        self.entries.insert(
            key,
            CacheEntry {
                bytes: Arc::clone(&bytes),
                last_used: self.clock,
            },
        );
        Ok(ExpertLease { key, bytes })
    }

    #[must_use]
    pub const fn metrics(&self) -> CacheMetrics {
        self.metrics
    }
}

/// Mapping from a stable expert key to one artifact tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertRegistration {
    pub key: ExpertKey,
    pub tensor_name: String,
}

/// On-demand expert loader backed by an artifact reader and byte-budgeted cache.
#[derive(Debug)]
pub struct ExpertStore {
    reader: ArtifactReader,
    names: HashMap<ExpertKey, String>,
    cache: ExpertCache,
}

impl ExpertStore {
    /// Creates an expert store after validating unique cache keys.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::DuplicateExpertKey`] for duplicate mappings.
    pub fn new(
        reader: ArtifactReader,
        registrations: Vec<ExpertRegistration>,
        budget: usize,
    ) -> Result<Self, StorageError> {
        let mut seen = HashSet::with_capacity(registrations.len());
        let mut names = HashMap::with_capacity(registrations.len());
        for registration in registrations {
            if !seen.insert(registration.key) {
                return Err(StorageError::DuplicateExpertKey {
                    key: registration.key,
                });
            }
            names.insert(registration.key, registration.tensor_name);
        }
        Ok(Self {
            reader,
            names,
            cache: ExpertCache::new(budget),
        })
    }

    /// Loads or returns a cached expert lease.
    ///
    /// # Errors
    ///
    /// Returns a registration, artifact, or cache-budget error.
    pub fn load(&mut self, key: ExpertKey) -> Result<ExpertLease, StorageError> {
        let name = self
            .names
            .get(&key)
            .ok_or(StorageError::ExpertNotRegistered { key })?
            .clone();
        self.cache.get_or_load(key, || {
            self.reader.read_tensor(&name).map(|tensor| tensor.bytes)
        })
    }

    /// Loads an expert without changing cache behavior and reports the
    /// resulting hit/load/eviction deltas after the lease is acquired.
    ///
    /// # Errors
    ///
    /// Returns the same storage error as [`ExpertStore::load`] when the
    /// payload cannot be loaded or validated.
    pub fn load_with_observer<F>(
        &mut self,
        key: ExpertKey,
        observe: F,
    ) -> Result<ExpertLease, StorageError>
    where
        F: FnOnce(ExpertLoadObservation),
    {
        let before = self.metrics();
        let lease = self.load(key)?;
        let after = self.metrics();
        observe(ExpertLoadObservation {
            key,
            payload_bytes: lease.bytes().len(),
            cache_hit: after.hits > before.hits,
            loaded: after.loads > before.loads,
            evictions: after.evictions - before.evictions,
        });
        Ok(lease)
    }

    #[must_use]
    pub const fn metrics(&self) -> CacheMetrics {
        self.cache.metrics()
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
    use crate::{
        ARTIFACT_FORMAT_VERSION, ArtifactManifest, ByteOrder, TensorLocation, TensorMetadata,
        hash::sha256,
    };

    static NEXT_STORE_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    const fn key(expert: u32) -> ExpertKey {
        ExpertKey {
            layer_index: 0,
            expert_id: ExpertId(expert),
        }
    }

    #[test]
    fn records_hits_misses_loads_and_deterministic_eviction() {
        let mut cache = ExpertCache::new(8);
        drop(
            cache
                .get_or_load(key(0), || Ok(vec![0; 4]))
                .expect("load 0"),
        );
        drop(
            cache
                .get_or_load(key(1), || Ok(vec![1; 4]))
                .expect("load 1"),
        );
        drop(
            cache
                .get_or_load(key(0), || panic!("cache hit"))
                .expect("hit 0"),
        );
        drop(
            cache
                .get_or_load(key(2), || Ok(vec![2; 4]))
                .expect("load 2"),
        );
        let metrics = cache.metrics();

        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 3);
        assert_eq!(metrics.loads, 3);
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.resident_bytes, 8);
        assert_eq!(metrics.peak_resident_bytes, 8);
        assert_eq!(metrics.bytes_read, 12);
        drop(
            cache
                .get_or_load(key(0), || panic!("0 remains MRU"))
                .expect("hit 0"),
        );
        drop(
            cache
                .get_or_load(key(1), || Ok(vec![1; 4]))
                .expect("1 was evicted"),
        );
    }

    #[test]
    fn leases_pin_entries_and_oversize_payloads_are_rejected() {
        let mut cache = ExpertCache::new(8);
        let pinned_zero = cache
            .get_or_load(key(0), || Ok(vec![0; 4]))
            .expect("load 0");
        let pinned_one = cache
            .get_or_load(key(1), || Ok(vec![1; 4]))
            .expect("load 1");

        assert!(matches!(
            cache.get_or_load(key(2), || Ok(vec![2; 4])),
            Err(StorageError::CacheBudgetExhausted { .. })
        ));
        drop(pinned_one);
        let two = cache
            .get_or_load(key(2), || Ok(vec![2; 4]))
            .expect("evict unpinned 1");
        assert_eq!(pinned_zero.bytes(), [0; 4]);
        assert_eq!(two.bytes(), [2; 4]);
        drop(two);
        drop(pinned_zero);

        assert!(matches!(
            cache.get_or_load(key(3), || Ok(vec![3; 9])),
            Err(StorageError::ExpertExceedsBudget { .. })
        ));
        assert!(cache.metrics().resident_bytes <= 8);
    }

    #[test]
    fn expert_store_loads_on_demand_through_artifact_reader() {
        let id = NEXT_STORE_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "colibri-expert-store-test-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("create store test directory");
        let bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        fs::write(root.join("experts.bin"), bytes).expect("write expert artifact");
        let manifest = ArtifactManifest::new(
            ARTIFACT_FORMAT_VERSION,
            ByteOrder::Little,
            vec![TensorMetadata {
                name: "layer.0.expert.0".to_owned(),
                shape: TensorShape::new([2]),
                data_type: DataType::F32,
                location: TensorLocation {
                    path: "experts.bin".into(),
                    offset: 0,
                    length: 8,
                },
                sha256: sha256(&bytes),
            }],
        )
        .expect("valid expert manifest");
        let reader = ArtifactReader::open(&root, manifest).expect("open expert reader");
        let registration = ExpertRegistration {
            key: key(0),
            tensor_name: "layer.0.expert.0".to_owned(),
        };
        let mut store = ExpertStore::new(reader, vec![registration], 8).expect("expert store");

        drop(store.load(key(0)).expect("first on-demand load"));
        drop(store.load(key(0)).expect("cached load"));

        assert_eq!(store.metrics().hits, 1);
        assert_eq!(store.metrics().misses, 1);
        assert_eq!(store.metrics().loads, 1);
        assert_eq!(store.metrics().bytes_read, 8);
        assert!(matches!(
            store.load(key(1)),
            Err(StorageError::ExpertNotRegistered { .. })
        ));
        fs::remove_dir_all(root).expect("remove store test directory");
    }
}
