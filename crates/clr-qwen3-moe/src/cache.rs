use clr_core::RuntimeError;

/// Key/value vectors for one token at one decoder layer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerKvUpdate<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
}

/// Borrowed initialized prefix for one layer's key/value cache.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerKvView<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub len: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct LayerKvCache {
    key: Vec<f32>,
    value: Vec<f32>,
}

/// Fixed-capacity contiguous F32 KV cache shared by generation backends.
///
/// Every layer owns separate key and value allocations with logical layout
/// `[capacity, kv_heads, head_dim]`. Allocation never grows after construction.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    capacity: usize,
    len: usize,
    values_per_token: usize,
    byte_size: usize,
}

impl KvCache {
    /// Creates a zero-initialized fixed-capacity cache.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ArithmeticOverflow`] when any allocation or byte
    /// count cannot be represented by `usize`.
    pub fn new(
        layer_count: usize,
        capacity: usize,
        key_value_heads: usize,
        head_dimension: usize,
    ) -> Result<Self, RuntimeError> {
        let values_per_token = key_value_heads.checked_mul(head_dimension).ok_or(
            RuntimeError::ArithmeticOverflow {
                operation: "KV values per token",
            },
        )?;
        let values_per_layer =
            capacity
                .checked_mul(values_per_token)
                .ok_or(RuntimeError::ArithmeticOverflow {
                    operation: "KV values per layer",
                })?;
        let total_values = layer_count
            .checked_mul(values_per_layer)
            .and_then(|value| value.checked_mul(2))
            .ok_or(RuntimeError::ArithmeticOverflow {
                operation: "KV total values",
            })?;
        let byte_size =
            total_values
                .checked_mul(size_of::<f32>())
                .ok_or(RuntimeError::ArithmeticOverflow {
                    operation: "KV cache byte size",
                })?;
        let layers = (0..layer_count)
            .map(|_| LayerKvCache {
                key: vec![0.0; values_per_layer],
                value: vec![0.0; values_per_layer],
            })
            .collect();
        Ok(Self {
            layers,
            capacity,
            len: 0,
            values_per_token,
            byte_size,
        })
    }

    /// Returns the fixed number of token positions allocated per layer.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of initialized token positions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the cache contains no initialized token positions.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the exact bytes allocated for F32 key and value elements.
    ///
    /// This excludes `Vec` and cache-structure metadata.
    #[must_use]
    pub const fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// Appends one token's K/V vectors for every layer transactionally.
    ///
    /// Validation completes before any cache bytes or `len` are changed.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ContextLengthExceeded`] before mutation when
    /// full, or a shape error when layer/update lengths differ.
    pub(crate) fn append_token(
        &mut self,
        updates: &[LayerKvUpdate<'_>],
    ) -> Result<(), RuntimeError> {
        let requested = self
            .len
            .checked_add(1)
            .ok_or(RuntimeError::ArithmeticOverflow {
                operation: "KV cache length",
            })?;
        if requested > self.capacity {
            return Err(RuntimeError::ContextLengthExceeded {
                requested,
                capacity: self.capacity,
            });
        }
        if updates.len() != self.layers.len()
            || updates.iter().any(|update| {
                update.key.len() != self.values_per_token
                    || update.value.len() != self.values_per_token
            })
        {
            return Err(RuntimeError::InvalidShape {
                reason: "KV append layer or vector length mismatch",
            });
        }
        let start = self.len * self.values_per_token;
        let end = start + self.values_per_token;
        for (layer, update) in self.layers.iter_mut().zip(updates) {
            layer.key[start..end].copy_from_slice(update.key);
            layer.value[start..end].copy_from_slice(update.value);
        }
        self.len = requested;
        Ok(())
    }

    pub(crate) fn layer(&self, index: usize) -> Result<LayerKvView<'_>, RuntimeError> {
        let layer = self
            .layers
            .get(index)
            .ok_or(RuntimeError::IndexOutOfBounds {
                index,
                length: self.layers.len(),
            })?;
        let initialized = self.len * self.values_per_token;
        Ok(LayerKvView {
            key: &layer.key[..initialized],
            value: &layer.value[..initialized],
            len: self.len,
        })
    }

    #[cfg(test)]
    pub(crate) fn allocation_capacities(&self) -> Vec<(usize, usize)> {
        self.layers
            .iter()
            .map(|layer| (layer.key.capacity(), layer.value.capacity()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_and_byte_accounting_are_checked_and_fixed() {
        let cache = KvCache::new(2, 8, 2, 4).expect("cache");

        assert_eq!(cache.capacity(), 8);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.byte_size(), 2 * 8 * 2 * 4 * 2 * 4);
        assert_eq!(cache.allocation_capacities(), vec![(64, 64), (64, 64)]);
        assert!(matches!(
            KvCache::new(usize::MAX, usize::MAX, 2, 4),
            Err(RuntimeError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn append_is_transactional_and_never_grows_allocations() {
        let mut cache = KvCache::new(2, 1, 1, 2).expect("cache");
        let before = cache.allocation_capacities();
        cache
            .append_token(&[
                LayerKvUpdate {
                    key: &[1.0, 2.0],
                    value: &[3.0, 4.0],
                },
                LayerKvUpdate {
                    key: &[5.0, 6.0],
                    value: &[7.0, 8.0],
                },
            ])
            .expect("append");

        assert_eq!(cache.len(), 1);
        let layer = cache.layer(1).expect("layer");
        assert_eq!(layer.key, [5.0, 6.0]);
        assert_eq!(layer.value, [7.0, 8.0]);
        assert_eq!(layer.len, 1);
        assert_eq!(cache.allocation_capacities(), before);

        let snapshot = cache.clone();
        assert!(matches!(
            cache.append_token(&[
                LayerKvUpdate {
                    key: &[9.0, 9.0],
                    value: &[9.0, 9.0],
                },
                LayerKvUpdate {
                    key: &[9.0, 9.0],
                    value: &[9.0, 9.0],
                },
            ]),
            Err(RuntimeError::ContextLengthExceeded {
                requested: 2,
                capacity: 1,
            })
        ));
        assert_eq!(cache, snapshot);
    }

    #[test]
    fn invalid_layer_vectors_do_not_mutate_cache() {
        let mut cache = KvCache::new(2, 2, 1, 2).expect("cache");
        let snapshot = cache.clone();

        assert!(matches!(
            cache.append_token(&[LayerKvUpdate {
                key: &[1.0],
                value: &[2.0],
            }]),
            Err(RuntimeError::InvalidShape { .. })
        ));
        assert_eq!(cache, snapshot);
    }

    #[test]
    fn repeated_create_and_drop_keeps_contract_independent() {
        for _ in 0..100 {
            let mut cache = KvCache::new(2, 4, 1, 2).expect("cache");
            cache
                .append_token(&[
                    LayerKvUpdate {
                        key: &[1.0, 2.0],
                        value: &[3.0, 4.0],
                    },
                    LayerKvUpdate {
                        key: &[5.0, 6.0],
                        value: &[7.0, 8.0],
                    },
                ])
                .expect("append");
            assert_eq!(cache.len(), 1);
        }
    }
}
