use std::fmt;

/// Dense floating-point element types understood by runtime metadata.
///
/// M0 and M1 define computation for [`DataType::F32`] only. The `F16` and
/// `BF16` variants describe model metadata and do not imply that arithmetic
/// kernels for those types exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// IEEE 754 binary32 floating point.
    F32,
    /// IEEE 754 binary16 floating point.
    F16,
    /// Brain floating-point format with an eight-bit exponent.
    BF16,
}

impl DataType {
    /// Returns the number of bytes used by one dense element.
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    /// Returns the stable lowercase name used in diagnostics and metadata.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        }
    }

    /// Returns whether this metadata type represents floating-point values.
    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        true
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_have_stable_metadata() {
        let cases = [
            (DataType::F32, 4, "f32"),
            (DataType::F16, 2, "f16"),
            (DataType::BF16, 2, "bf16"),
        ];

        for (data_type, byte_width, name) in cases {
            assert_eq!(data_type.byte_width(), byte_width);
            assert_eq!(data_type.name(), name);
            assert_eq!(data_type.to_string(), name);
            assert!(data_type.is_floating_point());
        }
    }
}
