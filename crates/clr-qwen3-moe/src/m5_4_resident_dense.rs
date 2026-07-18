//! M5.4-02 test-only resident-dense measurement support.
//!
//! This module is deliberately unavailable to normal library consumers. It
//! backs the full-model validation harness when its explicit feature and
//! environment mode are selected; the reference `File` reader remains the
//! default execution path.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

/// Fixed non-expert allowance used by the M5.4-02 prototype.
///
/// It preserves the M5.4-01 fixed-runtime/safety allowance and therefore
/// leaves a smaller expert-cache capacity than the raw process address space.
pub const FIXED_RUNTIME_MEMORY_BYTES: usize = 377_384_088;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentDenseBudget {
    pub total_budget: usize,
    pub expert_cache_budget: usize,
    pub fixed_runtime_memory: usize,
}

impl ResidentDenseBudget {
    pub fn validate(self, dense_payload_bytes: usize) -> Result<(), String> {
        let reserved = dense_payload_bytes
            .checked_add(self.fixed_runtime_memory)
            .ok_or_else(|| "resident dense budget reservation overflow".to_owned())?;
        let required = reserved
            .checked_add(self.expert_cache_budget)
            .ok_or_else(|| "resident dense total budget overflow".to_owned())?;
        if required > self.total_budget {
            return Err(format!(
                "resident dense configuration exceeds total budget: required={required} budget={}",
                self.total_budget
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseReadMetrics {
    pub resident_dense_bytes: usize,
    /// Bytes copied from the filesystem-facing reader into the resident Vec.
    pub initialization_bytes_read: u64,
    /// Logical tensor-range accesses made by the execution path.
    pub execution_bytes_accessed: u64,
}

/// Dense source selected only by `COLIBRI_DENSE_RESIDENCY_MODE` in the
/// full-model validation harness.
#[derive(Debug)]
pub enum DenseSource {
    Streaming(File),
    Resident {
        bytes: Vec<u8>,
        metrics: DenseReadMetrics,
    },
}

impl DenseSource {
    pub fn streaming(path: &Path) -> Result<Self, String> {
        File::open(path)
            .map(Self::Streaming)
            .map_err(|error| format!("open streamed dense payload {}: {error}", path.display()))
    }

    pub fn resident(path: &Path, budget: ResidentDenseBudget) -> Result<Self, String> {
        let payload_bytes = usize::try_from(
            path.metadata()
                .map_err(|error| format!("stat dense payload {}: {error}", path.display()))?
                .len(),
        )
        .map_err(|_| "dense payload length does not fit usize".to_owned())?;
        budget.validate(payload_bytes)?;

        let mut file = File::open(path)
            .map_err(|error| format!("open resident dense payload {}: {error}", path.display()))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(payload_bytes)
            .map_err(|error| format!("reserve resident dense payload: {error}"))?;
        file.read_to_end(&mut bytes)
            .map_err(|error| format!("load resident dense payload {}: {error}", path.display()))?;
        if bytes.len() != payload_bytes {
            return Err("resident dense payload length changed while loading".to_owned());
        }
        Ok(Self::Resident {
            bytes,
            metrics: DenseReadMetrics {
                resident_dense_bytes: payload_bytes,
                initialization_bytes_read: u64::try_from(payload_bytes).unwrap_or(u64::MAX),
                execution_bytes_accessed: 0,
            },
        })
    }

    pub fn read_exact_at(&mut self, offset: u64, destination: &mut [u8]) -> Result<(), String> {
        match self {
            Self::Streaming(file) => {
                file.seek(SeekFrom::Start(offset))
                    .map_err(|error| format!("seek streamed dense payload: {error}"))?;
                file.read_exact(destination)
                    .map_err(|error| format!("read streamed dense payload: {error}"))
            }
            Self::Resident { bytes, metrics } => {
                let start = usize::try_from(offset)
                    .map_err(|_| "resident dense offset does not fit usize".to_owned())?;
                let end = start
                    .checked_add(destination.len())
                    .ok_or_else(|| "resident dense range overflow".to_owned())?;
                let source = bytes
                    .get(start..end)
                    .ok_or_else(|| "resident dense range exceeds payload".to_owned())?;
                destination.copy_from_slice(source);
                metrics.execution_bytes_accessed = metrics
                    .execution_bytes_accessed
                    .saturating_add(u64::try_from(destination.len()).unwrap_or(u64::MAX));
                Ok(())
            }
        }
    }

    #[must_use]
    pub const fn metrics(&self) -> DenseReadMetrics {
        match self {
            Self::Streaming(_) => DenseReadMetrics {
                resident_dense_bytes: 0,
                initialization_bytes_read: 0,
                execution_bytes_accessed: 0,
            },
            Self::Resident { metrics, .. } => *metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn temp_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "clr-m5-4-resident-dense-{}-{}.bin",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn resident_dense_loads_once_and_tracks_logical_accesses() {
        let path = temp_path();
        fs::write(&path, [1_u8, 2, 3, 4]).expect("write dense fixture");
        let mut source = DenseSource::resident(
            &path,
            ResidentDenseBudget {
                total_budget: 32,
                expert_cache_budget: 16,
                fixed_runtime_memory: 8,
            },
        )
        .expect("resident source");
        let mut bytes = [0_u8; 2];
        source.read_exact_at(1, &mut bytes).expect("resident read");
        assert_eq!(bytes, [2, 3]);
        assert_eq!(source.metrics().resident_dense_bytes, 4);
        assert_eq!(source.metrics().initialization_bytes_read, 4);
        assert_eq!(source.metrics().execution_bytes_accessed, 2);
        drop(source);
        fs::remove_file(path).expect("release resident source file");
    }

    #[test]
    fn rejects_invalid_or_over_budget_resident_configuration() {
        let path = temp_path();
        fs::write(&path, [0_u8; 4]).expect("write dense fixture");
        let error = DenseSource::resident(
            &path,
            ResidentDenseBudget {
                total_budget: 10,
                expert_cache_budget: 4,
                fixed_runtime_memory: 4,
            },
        )
        .expect_err("over-budget configuration must fail");
        assert!(error.contains("exceeds total budget"));
        fs::remove_file(path).expect("remove dense fixture");
    }

    #[test]
    fn resident_range_bounds_are_checked() {
        let path = temp_path();
        fs::write(&path, [0_u8; 4]).expect("write dense fixture");
        let mut source = DenseSource::resident(
            &path,
            ResidentDenseBudget {
                total_budget: 16,
                expert_cache_budget: 4,
                fixed_runtime_memory: 4,
            },
        )
        .expect("resident source");
        let mut destination = [0_u8; 2];
        assert!(source.read_exact_at(3, &mut destination).is_err());
        drop(source);
        fs::remove_file(path).expect("remove dense fixture");
    }
}
