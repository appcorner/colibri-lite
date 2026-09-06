//! D5 production-like Layer24 hybrid artifact reader.
//!
//! Each expert is stored as F32 gate, F32 up, then grouped-i8 down values/scales.
//! The representation intentionally has no F32 down buffer.

use std::{
    fmt::Write as _,
    fs::File,
    io::{Read, Seek, SeekFrom},
    mem::size_of,
    path::{Path, PathBuf},
};

use clr_core::RuntimeError;
use clr_storage::Sha256Hasher;

use crate::r1_1_direct_candidate::R1_1PackedProjection;

const CANONICAL_EXPERTS: usize = 128;
const CANONICAL_HIDDEN: usize = 2048;
const CANONICAL_INTERMEDIATE: usize = 768;
const D5_GROUP_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct D5Layer24HybridLayout {
    experts: usize,
    hidden: usize,
    intermediate: usize,
    group_size: usize,
}

impl D5Layer24HybridLayout {
    pub(crate) fn canonical() -> Self {
        Self {
            experts: CANONICAL_EXPERTS,
            hidden: CANONICAL_HIDDEN,
            intermediate: CANONICAL_INTERMEDIATE,
            group_size: D5_GROUP_SIZE,
        }
    }

    fn new(
        experts: usize,
        hidden: usize,
        intermediate: usize,
        group_size: usize,
    ) -> Result<Self, RuntimeError> {
        if experts == 0
            || hidden == 0
            || intermediate == 0
            || group_size != D5_GROUP_SIZE
            || hidden % group_size != 0
            || intermediate % group_size != 0
        {
            return Err(error("invalid D5 hybrid artifact layout"));
        }
        Ok(Self {
            experts,
            hidden,
            intermediate,
            group_size,
        })
    }

    pub(crate) fn f32_projection_bytes(self) -> Result<usize, RuntimeError> {
        self.hidden
            .checked_mul(self.intermediate)
            .and_then(|value| value.checked_mul(size_of::<f32>()))
            .ok_or_else(|| error("D5 F32 projection byte length overflow"))
    }

    pub(crate) fn down_value_bytes(self) -> Result<usize, RuntimeError> {
        self.hidden
            .checked_mul(self.intermediate)
            .ok_or_else(|| error("D5 down value byte length overflow"))
    }

    pub(crate) fn down_scale_bytes(self) -> Result<usize, RuntimeError> {
        self.down_value_bytes()?
            .checked_div(self.group_size)
            .and_then(|value| value.checked_mul(size_of::<f32>()))
            .ok_or_else(|| error("D5 down scale byte length overflow"))
    }

    pub(crate) fn expert_bytes(self) -> Result<usize, RuntimeError> {
        self.f32_projection_bytes()?
            .checked_mul(2)
            .and_then(|value| value.checked_add(self.down_value_bytes().ok()?))
            .and_then(|value| value.checked_add(self.down_scale_bytes().ok()?))
            .ok_or_else(|| error("D5 hybrid expert byte length overflow"))
    }

    pub(crate) fn artifact_bytes(self) -> Result<u64, RuntimeError> {
        let bytes = self
            .expert_bytes()?
            .checked_mul(self.experts)
            .ok_or_else(|| error("D5 hybrid artifact byte length overflow"))?;
        u64::try_from(bytes).map_err(|_| error("D5 hybrid artifact byte length overflow"))
    }
}

#[derive(Debug)]
pub(crate) struct D5Layer24HybridExpert {
    gate: Vec<f32>,
    up: Vec<f32>,
    down_values: Vec<i8>,
    down_scales: Vec<f32>,
    layout: D5Layer24HybridLayout,
}

impl D5Layer24HybridExpert {
    pub(crate) fn apply(&self, input: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        if input.len() != self.layout.hidden || input.iter().any(|value| !value.is_finite()) {
            return Err(error("invalid D5 hybrid expert input"));
        }
        let mut activated = vec![0.0_f32; self.layout.intermediate];
        for (intermediate_index, activated_value) in activated.iter_mut().enumerate() {
            let start = intermediate_index * self.layout.hidden;
            let gate_value = dot(input, &self.gate[start..start + self.layout.hidden]);
            let up_value = dot(input, &self.up[start..start + self.layout.hidden]);
            *activated_value = gate_value / (1.0 + (-gate_value).exp()) * up_value;
        }
        let down = R1_1PackedProjection::new(
            &self.down_values,
            &self.down_scales,
            self.layout.hidden,
            self.layout.intermediate,
            self.layout.group_size,
        )?;
        down.apply_direct(&activated).map(|(output, _)| output)
    }
}

#[derive(Debug)]
pub(crate) struct D5Layer24HybridReader {
    path: PathBuf,
    file: File,
    layout: D5Layer24HybridLayout,
    sha256: String,
    verification_bytes_read: u64,
    payload_bytes_read: u64,
    peak_loaded_expert_bytes: usize,
}

impl D5Layer24HybridReader {
    pub(crate) fn open(path: &Path, expected_sha256: &str) -> Result<Self, RuntimeError> {
        Self::open_with_layout(path, D5Layer24HybridLayout::canonical(), expected_sha256)
    }

    fn open_with_layout(
        path: &Path,
        layout: D5Layer24HybridLayout,
        expected_sha256: &str,
    ) -> Result<Self, RuntimeError> {
        if expected_sha256.len() != 64
            || !expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(error("invalid D5 hybrid artifact hash"));
        }
        let mut file = File::open(path).map_err(|_| error("cannot open D5 hybrid artifact"))?;
        let actual_bytes = file
            .metadata()
            .map_err(|_| error("cannot inspect D5 hybrid artifact"))?
            .len();
        if actual_bytes != layout.artifact_bytes()? {
            return Err(error("D5 hybrid artifact length mismatch"));
        }
        let actual_sha256 = hash_reader(&mut file)?;
        if actual_sha256 != expected_sha256 {
            return Err(error("D5 hybrid artifact hash mismatch"));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|_| error("cannot rewind D5 hybrid artifact"))?;
        Ok(Self {
            path: path.to_owned(),
            file,
            layout,
            sha256: actual_sha256,
            verification_bytes_read: actual_bytes,
            payload_bytes_read: 0,
            peak_loaded_expert_bytes: 0,
        })
    }

    pub(crate) fn load_expert(
        &mut self,
        expert: usize,
    ) -> Result<D5Layer24HybridExpert, RuntimeError> {
        if expert >= self.layout.experts {
            return Err(error("D5 hybrid expert ID out of range"));
        }
        let expert_bytes = self.layout.expert_bytes()?;
        let base = expert
            .checked_mul(expert_bytes)
            .ok_or_else(|| error("D5 hybrid expert offset overflow"))?;
        self.file
            .seek(SeekFrom::Start(
                u64::try_from(base).map_err(|_| error("D5 hybrid expert offset overflow"))?,
            ))
            .map_err(|_| error("cannot seek D5 hybrid artifact"))?;

        let projection_values = self
            .layout
            .hidden
            .checked_mul(self.layout.intermediate)
            .ok_or_else(|| error("D5 projection value count overflow"))?;
        let gate = self.read_f32(projection_values)?;
        let up = self.read_f32(projection_values)?;
        let down_values = self.read_i8(self.layout.down_value_bytes()?)?;
        let down_scales =
            self.read_f32(self.layout.down_value_bytes()? / self.layout.group_size)?;
        self.peak_loaded_expert_bytes = self.peak_loaded_expert_bytes.max(expert_bytes);
        Ok(D5Layer24HybridExpert {
            gate,
            up,
            down_values,
            down_scales,
            layout: self.layout,
        })
    }

    pub(crate) fn payload_bytes_read(&self) -> u64 {
        self.payload_bytes_read
    }

    pub(crate) fn verification_bytes_read(&self) -> u64 {
        self.verification_bytes_read
    }

    pub(crate) fn peak_loaded_expert_bytes(&self) -> usize {
        self.peak_loaded_expert_bytes
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }

    fn read_f32(&mut self, count: usize) -> Result<Vec<f32>, RuntimeError> {
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| error("D5 F32 read byte length overflow"))?;
        let mut raw = vec![0_u8; bytes];
        self.file
            .read_exact(&mut raw)
            .map_err(|_| error("short D5 hybrid F32 read"))?;
        self.payload_bytes_read = self
            .payload_bytes_read
            .checked_add(bytes as u64)
            .ok_or_else(|| error("D5 payload byte counter overflow"))?;
        Ok(raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32 payload")))
            .collect())
    }

    fn read_i8(&mut self, count: usize) -> Result<Vec<i8>, RuntimeError> {
        let mut raw = vec![0_u8; count];
        self.file
            .read_exact(&mut raw)
            .map_err(|_| error("short D5 hybrid i8 read"))?;
        self.payload_bytes_read = self
            .payload_bytes_read
            .checked_add(count as u64)
            .ok_or_else(|| error("D5 payload byte counter overflow"))?;
        Ok(raw
            .into_iter()
            .map(|value| i8::from_le_bytes([value]))
            .collect())
    }
}

fn hash_reader(reader: &mut File) -> Result<String, RuntimeError> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|_| error("cannot rewind D5 hybrid artifact for hashing"))?;
    let mut hasher = Sha256Hasher::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|_| error("cannot hash D5 hybrid artifact"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_sha256(hasher.finalize()))
}

fn hex_sha256(bytes: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("write SHA-256 hex");
    }
    output
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn error(reason: &'static str) -> RuntimeError {
    RuntimeError::BackendContractViolation {
        context: "M6.3-R2.2-D5 Layer24 hybrid artifact",
        reason,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Write,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "colibri-d5-{name}-{}-{stamp}.bin",
            std::process::id()
        ))
    }

    fn write_synthetic_artifact(path: &Path, layout: D5Layer24HybridLayout) -> String {
        let mut file = File::create(path).expect("create D5 synthetic artifact");
        let projection_values = layout.hidden * layout.intermediate;
        for expert in 0..layout.experts {
            let expert_value = f32::from(u16::try_from(expert).expect("small synthetic expert"));
            let gate = 0.25_f32 + expert_value;
            let up = 0.5_f32 + expert_value;
            for _ in 0..projection_values {
                file.write_all(&gate.to_le_bytes()).expect("write gate");
            }
            for _ in 0..projection_values {
                file.write_all(&up.to_le_bytes()).expect("write up");
            }
            file.write_all(&vec![1_u8; projection_values])
                .expect("write down values");
            for _ in 0..projection_values / layout.group_size {
                file.write_all(&1.0_f32.to_le_bytes())
                    .expect("write down scales");
            }
        }
        file.flush().expect("flush synthetic artifact");
        let mut reader = File::open(path).expect("open synthetic artifact for hash");
        hash_reader(&mut reader).expect("hash synthetic artifact")
    }

    #[test]
    fn canonical_layout_matches_frozen_d5_contract() {
        let layout = D5Layer24HybridLayout::canonical();
        assert_eq!(
            layout.f32_projection_bytes().expect("projection bytes"),
            6_291_456
        );
        assert_eq!(layout.down_value_bytes().expect("down values"), 1_572_864);
        assert_eq!(layout.down_scale_bytes().expect("down scales"), 786_432);
        assert_eq!(layout.expert_bytes().expect("expert bytes"), 14_942_208);
        assert_eq!(
            layout.artifact_bytes().expect("artifact bytes"),
            1_912_602_624
        );
    }

    #[test]
    fn reader_consumes_f32_gate_up_and_group8_down_without_f32_down() {
        let layout = D5Layer24HybridLayout::new(2, 8, 8, 8).expect("synthetic layout");
        let path = temp_path("reader");
        let sha256 = write_synthetic_artifact(&path, layout);
        let mut reader = D5Layer24HybridReader::open_with_layout(&path, layout, &sha256)
            .expect("open D5 hybrid reader");
        let expert = reader.load_expert(1).expect("load synthetic expert");
        let output = expert.apply(&[1.0; 8]).expect("apply synthetic expert");
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|value| value.is_finite() && *value > 0.0));
        assert_eq!(
            reader.payload_bytes_read(),
            layout.expert_bytes().expect("expert bytes") as u64
        );
        assert_eq!(
            reader.verification_bytes_read(),
            layout.artifact_bytes().expect("artifact bytes")
        );
        assert_eq!(
            reader.peak_loaded_expert_bytes(),
            layout.expert_bytes().expect("expert bytes")
        );
        assert_eq!(reader.path(), path.as_path());
        assert_eq!(reader.sha256(), sha256);
        fs::remove_file(path).expect("remove synthetic artifact");
    }

    #[test]
    fn reader_rejects_wrong_length_hash_and_expert_id() {
        let layout = D5Layer24HybridLayout::new(1, 8, 8, 8).expect("synthetic layout");
        let path = temp_path("invalid");
        let sha256 = write_synthetic_artifact(&path, layout);
        assert!(D5Layer24HybridReader::open_with_layout(&path, layout, &"0".repeat(64)).is_err());
        let mut reader = D5Layer24HybridReader::open_with_layout(&path, layout, &sha256)
            .expect("open valid synthetic artifact");
        assert!(reader.load_expert(1).is_err());
        fs::remove_file(&path).expect("remove valid synthetic artifact");

        let short_path = temp_path("short");
        fs::write(&short_path, [0_u8; 8]).expect("write short artifact");
        assert!(
            D5Layer24HybridReader::open_with_layout(&short_path, layout, &"0".repeat(64)).is_err()
        );
        fs::remove_file(short_path).expect("remove short artifact");
    }
}
