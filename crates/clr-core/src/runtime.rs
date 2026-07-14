/// Metadata describing the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// Public runtime name.
    pub name: &'static str,
    /// Current package version.
    pub version: &'static str,
    /// Host operating system selected during compilation.
    pub operating_system: &'static str,
    /// Host architecture selected during compilation.
    pub architecture: &'static str,
}

/// Returns compile-time information about colibri-lite-rs.
#[must_use]
pub const fn runtime_info() -> RuntimeInfo {
    RuntimeInfo {
        name: "colibri-lite-rs",
        version: env!("CARGO_PKG_VERSION"),
        operating_system: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_has_expected_name() {
        let info = runtime_info();

        assert_eq!(info.name, "colibri-lite-rs");
        assert!(!info.version.is_empty());
        assert!(!info.operating_system.is_empty());
        assert!(!info.architecture.is_empty());
    }
}
