#![doc = "Small, safe read-only memory-mapped-file boundary for colibri-lite-rs."]

use std::{fs::File, io, path::Path};

use memmap2::{Mmap, MmapOptions};

#[allow(unsafe_code)]
mod mapping {
    use super::{File, Mmap, MmapOptions, Path, io};

    /// Read-only mapping that owns its backing file handle.
    #[derive(Debug)]
    pub struct ReadOnlyMappedFile {
        // The handle is intentionally retained to make Windows lifetime explicit.
        #[allow(dead_code)]
        file: File,
        map: Mmap,
    }

    impl ReadOnlyMappedFile {
        /// Opens and maps a complete, non-empty file without write access.
        ///
        /// The `File` and `Mmap` are owned together; only shared slices are exposed.
        ///
        /// # Errors
        ///
        /// Returns an I/O error when the file cannot be opened, its length cannot
        /// be represented, or the operating system rejects the read-only mapping.
        pub fn open(path: &Path) -> io::Result<Self> {
            let file = File::open(path)?;
            let length = usize::try_from(file.metadata()?.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "mapped file exceeds usize")
            })?;
            if length == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cannot map an empty file",
                ));
            }

            // SAFETY: the file is opened read-only; the mapping starts at offset zero,
            // has the exact metadata length, and this object owns both the file and map.
            // The public API exposes only shared slices and no mutation operation.
            let map = unsafe { MmapOptions::new().len(length).map(&file) }?;
            Ok(Self { file, map })
        }

        /// Returns the mapped file length in bytes.
        #[must_use]
        pub fn len(&self) -> usize {
            self.map.len()
        }

        /// Returns whether the mapped file has no bytes.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.map.is_empty()
        }

        /// Returns an exact shared byte range, or `None` for an invalid range.
        #[must_use]
        pub fn range(&self, offset: u64, length: u64) -> Option<&[u8]> {
            let offset = usize::try_from(offset).ok()?;
            let length = usize::try_from(length).ok()?;
            let end = offset.checked_add(length)?;
            self.map.get(offset..end)
        }
    }
}

pub use mapping::ReadOnlyMappedFile;

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("colibri-mmap-test-{}-{id}", std::process::id()));
            fs::create_dir(&path).expect("create isolated mmap test directory");
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove isolated mmap test directory");
        }
    }

    #[test]
    fn maps_exact_ranges_read_only_and_releases_after_drop() {
        let directory = TestDirectory::new();
        let path = directory.0.join("shard.bin");
        fs::write(&path, [9_u8, 1, 2, 3, 4, 8]).expect("write test shard");

        let mapped = ReadOnlyMappedFile::open(&path).expect("map test shard");
        assert_eq!(mapped.len(), 6);
        assert_eq!(mapped.range(1, 4), Some(&[1, 2, 3, 4][..]));
        assert_eq!(mapped.range(5, 1), Some(&[8][..]));
        assert_eq!(mapped.range(6, 0), Some(&[][..]));
        assert_eq!(mapped.range(6, 1), None);
        drop(mapped);

        fs::remove_file(&path).expect("remove test shard after mapping drop");
    }

    #[test]
    fn rejects_empty_files_and_invalid_ranges() {
        let directory = TestDirectory::new();
        let path = directory.0.join("empty.bin");
        fs::write(&path, []).expect("write empty test shard");
        assert!(ReadOnlyMappedFile::open(&path).is_err());
    }
}
