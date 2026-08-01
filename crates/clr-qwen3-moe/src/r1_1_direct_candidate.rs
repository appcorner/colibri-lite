//! R1.1 direct-consumption reader for the new group-32/group-64 layouts.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use clr_core::{RuntimeError, Tensor, TensorView};
use clr_storage::Sha256Hasher;

use crate::{
    Qwen3MoeConfig,
    block::{RouterOutput, combine_routed_experts},
};

const CANONICAL_EXPERTS: usize = 128;
const CANONICAL_INTERMEDIATE: usize = 768;
const CANONICAL_HIDDEN: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct R1_1PackedArtifactLayout {
    group_size: usize,
    experts: usize,
    intermediate: usize,
    hidden: usize,
}

impl R1_1PackedArtifactLayout {
    pub(crate) fn canonical(group_size: usize) -> Result<Self, RuntimeError> {
        Self::new(
            group_size,
            CANONICAL_EXPERTS,
            CANONICAL_INTERMEDIATE,
            CANONICAL_HIDDEN,
        )
    }

    fn new(
        group_size: usize,
        experts: usize,
        intermediate: usize,
        hidden: usize,
    ) -> Result<Self, RuntimeError> {
        if !matches!(group_size, 32 | 64)
            || experts == 0
            || intermediate == 0
            || hidden == 0
            || hidden % group_size != 0
            || intermediate % group_size != 0
        {
            return Err(error("invalid R1.1 packed artifact layout"));
        }
        let layout = Self {
            group_size,
            experts,
            intermediate,
            hidden,
        };
        layout.artifact_bytes()?;
        Ok(layout)
    }

    pub(crate) fn group_size(self) -> usize {
        self.group_size
    }

    pub(crate) fn expert_bytes(self) -> Result<usize, RuntimeError> {
        let gate_up = self.projection_bytes(self.intermediate, self.hidden)?;
        gate_up
            .checked_mul(2)
            .and_then(|value| {
                value.checked_add(self.projection_bytes(self.hidden, self.intermediate).ok()?)
            })
            .ok_or_else(|| error("R1.1 expert byte length overflow"))
    }

    pub(crate) fn artifact_bytes(self) -> Result<u64, RuntimeError> {
        let bytes = self
            .expert_bytes()?
            .checked_mul(self.experts)
            .ok_or_else(|| error("R1.1 artifact byte length overflow"))?;
        u64::try_from(bytes).map_err(|_| error("R1.1 artifact byte length overflow"))
    }

    fn projection_bytes(self, rows: usize, columns: usize) -> Result<usize, RuntimeError> {
        let values = rows
            .checked_mul(columns)
            .ok_or_else(|| error("R1.1 projection value length overflow"))?;
        let scales = rows
            .checked_mul(columns / self.group_size)
            .and_then(|value| value.checked_mul(size_of::<f32>()))
            .ok_or_else(|| error("R1.1 projection scale length overflow"))?;
        values
            .checked_add(scales)
            .ok_or_else(|| error("R1.1 projection byte length overflow"))
    }
}

#[derive(Debug)]
pub(crate) struct R1_1PackedExpert {
    gate_values: Vec<i8>,
    gate_scales: Vec<f32>,
    up_values: Vec<i8>,
    up_scales: Vec<f32>,
    down_values: Vec<i8>,
    down_scales: Vec<f32>,
    layout: R1_1PackedArtifactLayout,
}

impl R1_1PackedExpert {
    pub(crate) fn apply_direct(&self, input: &[f32]) -> Result<(Vec<f32>, usize), RuntimeError> {
        let gate = R1_1PackedProjection::new(
            &self.gate_values,
            &self.gate_scales,
            self.layout.intermediate,
            self.layout.hidden,
            self.layout.group_size,
        )?;
        let up = R1_1PackedProjection::new(
            &self.up_values,
            &self.up_scales,
            self.layout.intermediate,
            self.layout.hidden,
            self.layout.group_size,
        )?;
        let down = R1_1PackedProjection::new(
            &self.down_values,
            &self.down_scales,
            self.layout.hidden,
            self.layout.intermediate,
            self.layout.group_size,
        )?;
        let (gate, gate_expanded) = gate.apply_direct(input)?;
        let (up, up_expanded) = up.apply_direct(input)?;
        let activated = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let (output, down_expanded) = down.apply_direct(&activated)?;
        Ok((output, gate_expanded + up_expanded + down_expanded))
    }
}

#[derive(Debug)]
pub(crate) struct R1_1CandidateReader {
    path: PathBuf,
    file: File,
    layout: R1_1PackedArtifactLayout,
    sha256: String,
    verification_bytes_read: u64,
    payload_bytes_read: u64,
    peak_packed_expert_bytes: usize,
}

impl R1_1CandidateReader {
    pub(crate) fn open(
        path: &Path,
        layout: R1_1PackedArtifactLayout,
        expected_sha256: &str,
    ) -> Result<Self, RuntimeError> {
        if expected_sha256.len() != 64
            || !expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(error("invalid R1.1 expected artifact hash"));
        }
        let mut file =
            File::open(path).map_err(|_| error("cannot open R1.1 candidate artifact"))?;
        let actual_bytes = file
            .metadata()
            .map_err(|_| error("cannot inspect R1.1 candidate artifact"))?
            .len();
        if actual_bytes != layout.artifact_bytes()? {
            return Err(error("R1.1 candidate artifact length mismatch"));
        }
        let actual_sha256 = hash_reader(&mut file)?;
        if actual_sha256 != expected_sha256 {
            return Err(error("R1.1 candidate artifact hash mismatch"));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|_| error("cannot rewind R1.1 candidate artifact"))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            layout,
            sha256: actual_sha256,
            verification_bytes_read: actual_bytes,
            payload_bytes_read: 0,
            peak_packed_expert_bytes: 0,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(crate) fn verification_bytes_read(&self) -> u64 {
        self.verification_bytes_read
    }

    pub(crate) fn payload_bytes_read(&self) -> u64 {
        self.payload_bytes_read
    }

    pub(crate) fn peak_packed_expert_bytes(&self) -> usize {
        self.peak_packed_expert_bytes
    }

    pub(crate) fn load_expert(&mut self, expert: usize) -> Result<R1_1PackedExpert, RuntimeError> {
        if expert >= self.layout.experts {
            return Err(error("R1.1 candidate expert ID out of range"));
        }
        let expert_bytes = self.layout.expert_bytes()?;
        let base = expert
            .checked_mul(expert_bytes)
            .ok_or_else(|| error("R1.1 candidate expert offset overflow"))?;
        if base % 64 != 0 {
            return Err(error("R1.1 candidate expert offset is not 64-byte aligned"));
        }
        self.file
            .seek(SeekFrom::Start(u64::try_from(base).map_err(|_| {
                error("R1.1 candidate expert offset overflow")
            })?))
            .map_err(|_| error("cannot seek R1.1 candidate artifact"))?;
        let (gate_values, gate_scales) =
            self.read_projection(self.layout.intermediate, self.layout.hidden)?;
        let (up_values, up_scales) =
            self.read_projection(self.layout.intermediate, self.layout.hidden)?;
        let (down_values, down_scales) =
            self.read_projection(self.layout.hidden, self.layout.intermediate)?;
        self.peak_packed_expert_bytes = self.peak_packed_expert_bytes.max(expert_bytes);
        Ok(R1_1PackedExpert {
            gate_values,
            gate_scales,
            up_values,
            up_scales,
            down_values,
            down_scales,
            layout: self.layout,
        })
    }

    fn read_projection(
        &mut self,
        rows: usize,
        columns: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), RuntimeError> {
        let position = self
            .file
            .stream_position()
            .map_err(|_| error("cannot inspect R1.1 candidate offset"))?;
        if position % 64 != 0 {
            return Err(error(
                "R1.1 candidate projection offset is not 64-byte aligned",
            ));
        }
        let value_count = rows
            .checked_mul(columns)
            .ok_or_else(|| error("R1.1 projection value length overflow"))?;
        let scale_count = rows
            .checked_mul(columns / self.layout.group_size)
            .ok_or_else(|| error("R1.1 projection scale length overflow"))?;
        let mut value_bytes = vec![0; value_count];
        self.file
            .read_exact(&mut value_bytes)
            .map_err(|_| error("truncated R1.1 candidate values"))?;
        let values = value_bytes
            .into_iter()
            .map(|value| i8::from_le_bytes([value]))
            .collect::<Vec<_>>();
        if values.contains(&i8::MIN) {
            return Err(error("R1.1 candidate contains forbidden -128 value"));
        }
        let mut scale_bytes = vec![0; scale_count * size_of::<f32>()];
        self.file
            .read_exact(&mut scale_bytes)
            .map_err(|_| error("truncated R1.1 candidate scales"))?;
        let scales = scale_bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte scale")))
            .collect::<Vec<_>>();
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
        {
            return Err(error("R1.1 candidate contains invalid scale"));
        }
        self.payload_bytes_read = self
            .payload_bytes_read
            .checked_add(
                u64::try_from(value_count + scale_bytes.len())
                    .map_err(|_| error("R1.1 read accounting overflow"))?,
            )
            .ok_or_else(|| error("R1.1 read accounting overflow"))?;
        Ok((values, scales))
    }
}

pub(crate) fn r1_1_routed_experts_with_observer<F>(
    hidden_states: TensorView<'_>,
    router: &RouterOutput,
    config: Qwen3MoeConfig,
    reader: &mut R1_1CandidateReader,
    mut observe: F,
) -> Result<Tensor, RuntimeError>
where
    F: FnMut(usize, usize, usize, &[f32]),
{
    let hidden_size = config.model().hidden_size();
    combine_routed_experts(
        hidden_states,
        router,
        config,
        |expert_id, occurrences| -> Result<Vec<Vec<f32>>, RuntimeError> {
            let expert = reader.load_expert(expert_id)?;
            let mut outputs = Vec::with_capacity(occurrences.len());
            for &(token, rank) in occurrences {
                let input = &hidden_states.data()[token * hidden_size..(token + 1) * hidden_size];
                let (output, complete_f32_weight_materializations) = expert.apply_direct(input)?;
                if complete_f32_weight_materializations != 0 {
                    return Err(error("R1.1 direct path expanded a complete F32 weight"));
                }
                observe(expert_id, token, rank, &output);
                outputs.push(output);
            }
            Ok(outputs)
        },
    )
}

fn hash_reader(file: &mut File) -> Result<String, RuntimeError> {
    let mut hasher = Sha256Hasher::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| error("cannot hash R1.1 candidate artifact"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher
        .finalize()
        .into_iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            use std::fmt::Write as _;
            write!(output, "{byte:02x}").expect("write hash to String");
            output
        }))
}

/// A packed R1.1 projection. It intentionally has no dequantized weight buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct R1_1PackedProjection<'a> {
    values: &'a [i8],
    scales: &'a [f32],
    rows: usize,
    columns: usize,
    group_size: usize,
}

impl<'a> R1_1PackedProjection<'a> {
    pub(crate) fn new(
        values: &'a [i8],
        scales: &'a [f32],
        rows: usize,
        columns: usize,
        group_size: usize,
    ) -> Result<Self, RuntimeError> {
        if rows == 0 || columns == 0 || !matches!(group_size, 32 | 64) || columns % group_size != 0
        {
            return Err(error("invalid R1.1 packed projection layout"));
        }
        if values.len()
            != rows
                .checked_mul(columns)
                .ok_or_else(|| error("R1.1 value length overflow"))?
            || scales.len()
                != rows
                    .checked_mul(columns / group_size)
                    .ok_or_else(|| error("R1.1 scale length overflow"))?
            || values.contains(&i8::MIN)
            || scales
                .iter()
                .any(|scale| !scale.is_finite() || *scale < 0.0)
        {
            return Err(error("invalid R1.1 packed projection payload"));
        }
        Ok(Self {
            values,
            scales,
            rows,
            columns,
            group_size,
        })
    }

    pub(crate) fn apply_direct(&self, input: &[f32]) -> Result<(Vec<f32>, usize), RuntimeError> {
        if input.len() != self.columns || input.iter().any(|value| !value.is_finite()) {
            return Err(error("invalid R1.1 projection input"));
        }
        let groups = self.columns / self.group_size;
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut sum = 0.0;
            for (column, input_value) in input.iter().enumerate() {
                let scale = self.scales[row * groups + column / self.group_size];
                sum += *input_value * (f32::from(self.values[row * self.columns + column]) * scale);
            }
            output.push(sum);
        }
        Ok((output, 0))
    }
}

fn error(reason: &'static str) -> RuntimeError {
    RuntimeError::BackendContractViolation {
        context: "M6.3-R1.1 packed projection",
        reason,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Write,
        sync::atomic::{AtomicU64, Ordering},
    };

    use clr_storage::Sha256Hasher;

    use super::{R1_1CandidateReader, R1_1PackedArtifactLayout, R1_1PackedProjection};

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (std::path::PathBuf, R1_1PackedArtifactLayout, String) {
        let layout = R1_1PackedArtifactLayout::new(32, 2, 32, 32).expect("layout");
        let mut bytes = Vec::with_capacity(
            usize::try_from(layout.artifact_bytes().expect("bytes")).expect("usize"),
        );
        for _ in 0..2 {
            for (rows, columns) in [(32, 32), (32, 32), (32, 32)] {
                bytes.extend(std::iter::repeat_n(1_u8, rows * columns));
                for _ in 0..rows * (columns / 32) {
                    bytes.extend_from_slice(&0.5_f32.to_le_bytes());
                }
            }
        }
        let mut hasher = Sha256Hasher::new();
        hasher.update(&bytes);
        let hash = hasher
            .finalize()
            .into_iter()
            .fold(String::new(), |mut output, byte| {
                use std::fmt::Write as _;
                write!(output, "{byte:02x}").expect("hash string");
                output
            });
        let path = std::env::temp_dir().join(format!(
            "colibri-r1-1-reader-{}-{}.bin",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::create(&path).expect("fixture file");
        file.write_all(&bytes).expect("fixture payload");
        file.sync_all().expect("sync fixture");
        (path, layout, hash)
    }

    #[test]
    fn group64_consumes_packed_values_and_scales_without_f32_weight_expansion() {
        let values = vec![2_i8; 64];
        let scales = vec![0.25_f32];
        let projection = R1_1PackedProjection::new(&values, &scales, 1, 64, 64).expect("valid");
        let (output, expanded_weights) = projection.apply_direct(&vec![1.0; 64]).expect("direct");
        assert_eq!(output, vec![32.0]);
        assert_eq!(expanded_weights, 0);
    }

    #[test]
    fn canonical_layout_sizes_match_frozen_artifacts() {
        assert_eq!(
            R1_1PackedArtifactLayout::canonical(64)
                .expect("group-64")
                .artifact_bytes()
                .expect("bytes"),
            641_728_512
        );
        assert_eq!(
            R1_1PackedArtifactLayout::canonical(32)
                .expect("group-32")
                .artifact_bytes()
                .expect("bytes"),
            679_477_248
        );
    }

    #[test]
    fn reader_validates_hash_and_directly_consumes_one_expert() {
        let (path, layout, hash) = fixture();
        let mut reader = R1_1CandidateReader::open(&path, layout, &hash).expect("reader");
        assert_eq!(reader.path(), path);
        assert_eq!(reader.sha256(), hash);
        let hash_read = layout.artifact_bytes().expect("artifact bytes");
        assert_eq!(reader.verification_bytes_read(), hash_read);
        assert_eq!(reader.payload_bytes_read(), 0);
        let expert = reader.load_expert(1).expect("expert");
        let (output, expanded) = expert.apply_direct(&[1.0; 32]).expect("direct expert");
        assert_eq!(output.len(), 32);
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(expanded, 0);
        assert_eq!(
            reader.payload_bytes_read(),
            u64::try_from(layout.expert_bytes().expect("expert bytes")).expect("u64")
        );
        assert_eq!(
            reader.peak_packed_expert_bytes(),
            layout.expert_bytes().expect("expert bytes")
        );
        fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn reader_rejects_hash_size_expert_and_payload_failures() {
        let (path, layout, hash) = fixture();
        assert!(R1_1CandidateReader::open(&path, layout, &format!("0{}", &hash[1..])).is_err());
        let mut reader = R1_1CandidateReader::open(&path, layout, &hash).expect("reader");
        assert!(reader.load_expert(2).is_err());
        drop(reader);

        let mut bytes = fs::read(&path).expect("fixture bytes");
        bytes[0] = 0x80;
        fs::write(&path, &bytes).expect("tampered values");
        let tampered_hash = {
            let mut hasher = Sha256Hasher::new();
            hasher.update(&bytes);
            hasher
                .finalize()
                .into_iter()
                .fold(String::new(), |mut output, byte| {
                    use std::fmt::Write as _;
                    write!(output, "{byte:02x}").expect("hash string");
                    output
                })
        };
        let mut tampered =
            R1_1CandidateReader::open(&path, layout, &tampered_hash).expect("tampered reader");
        assert!(tampered.load_expert(0).is_err());
        drop(tampered);

        bytes.pop();
        fs::write(&path, &bytes).expect("truncated fixture");
        assert!(R1_1CandidateReader::open(&path, layout, &tampered_hash).is_err());
        fs::remove_file(path).expect("remove fixture");
    }
}
