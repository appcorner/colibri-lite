//! Safe adapter around the isolated M5.3-04 mapping boundary.

use std::{io, path::Path};

use clr_mmap::ReadOnlyMappedFile;

/// One complete, read-only mapping retained for the lifetime of its reader.
#[derive(Debug)]
pub(crate) struct ReadOnlyMappedShard {
    inner: ReadOnlyMappedFile,
}

impl ReadOnlyMappedShard {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        Ok(Self {
            inner: ReadOnlyMappedFile::open(path)?,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn range(&self, offset: u64, length: u64) -> Option<&[u8]> {
        self.inner.range(offset, length)
    }
}
