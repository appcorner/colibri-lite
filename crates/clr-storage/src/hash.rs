const INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Incremental dependency-free SHA-256 state for bounded artifact I/O.
#[derive(Debug, Clone)]
pub struct Sha256Hasher {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_length: usize,
    total_bytes: u64,
}

impl Sha256Hasher {
    /// Creates an empty SHA-256 state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: INITIAL_STATE,
            buffer: [0; 64],
            buffer_length: 0,
            total_bytes: 0,
        }
    }

    /// Appends bytes to the digest state.
    pub fn update(&mut self, mut input: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(input.len() as u64);
        if self.buffer_length != 0 {
            let copied = (64 - self.buffer_length).min(input.len());
            self.buffer[self.buffer_length..self.buffer_length + copied]
                .copy_from_slice(&input[..copied]);
            self.buffer_length += copied;
            input = &input[copied..];
            if self.buffer_length < 64 {
                return;
            }
            compress(&mut self.state, &self.buffer);
            self.buffer_length = 0;
        }
        while input.len() >= 64 {
            compress(&mut self.state, &input[..64]);
            input = &input[64..];
        }
        self.buffer[..input.len()].copy_from_slice(input);
        self.buffer_length = input.len();
    }

    /// Finalizes and returns the 32-byte SHA-256 digest.
    #[must_use]
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_length = self.total_bytes.wrapping_mul(8);
        self.buffer[self.buffer_length] = 0x80;
        self.buffer_length += 1;
        if self.buffer_length > 56 {
            self.buffer[self.buffer_length..].fill(0);
            compress(&mut self.state, &self.buffer);
            self.buffer_length = 0;
        }
        self.buffer[self.buffer_length..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_length.to_be_bytes());
        compress(&mut self.state, &self.buffer);

        state_bytes(self.state)
    }
}

impl Default for Sha256Hasher {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn sha256(input: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256Hasher::new();
    hasher.update(input);
    hasher.finalize()
}

fn state_bytes(state: [u32; 8]) -> [u8; 32] {
    let mut output = [0_u8; 32];
    for (chunk, value) in output.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    output
}

#[allow(clippy::many_single_char_names)] // Names follow the SHA-256 specification.
fn compress(state: &mut [u32; 8], block: &[u8]) {
    let mut schedule = [0_u32; 64];
    for (index, word) in block.chunks_exact(4).enumerate() {
        schedule[index] = u32::from_be_bytes(word.try_into().expect("four-byte SHA-256 word"));
    }
    for index in 16..64 {
        let first = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let second = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(first)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(second);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let choose = (e & f) ^ ((!e) & g);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let sum_zero = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let sum_one = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let temporary_one = h
            .wrapping_add(sum_one)
            .wrapping_add(choose)
            .wrapping_add(ROUND_CONSTANTS[index])
            .wrapping_add(schedule[index]);
        let temporary_two = sum_zero.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temporary_one);
        d = c;
        c = b;
        b = a;
        a = temporary_one.wrapping_add(temporary_two);
    }

    for (target, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *target = target.wrapping_add(value);
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use super::*;

    fn hex(bytes: &[u8]) -> String {
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            write!(&mut output, "{byte:02x}").expect("write to string");
        }
        output
    }

    #[test]
    fn matches_published_sha256_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn incremental_updates_match_one_shot_hashing() {
        let input = (0_u8..=255).cycle().take(10_003).collect::<Vec<_>>();
        let expected = sha256(&input);

        for chunk_size in [1, 3, 63, 64, 65, 1_024] {
            let mut hasher = Sha256Hasher::new();
            for chunk in input.chunks(chunk_size) {
                hasher.update(chunk);
            }
            assert_eq!(hasher.finalize(), expected, "chunk size {chunk_size}");
        }
    }
}
