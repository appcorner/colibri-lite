#![allow(clippy::float_cmp, clippy::too_many_lines)]

use std::{
    collections::{HashMap, HashSet},
    env,
    fmt::Write as _,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use clr_core::{DataType, Tensor, TensorShape, ops::elementwise_add};
#[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
use clr_storage::ReaderMode;
use clr_storage::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ArtifactReader, ByteOrder, ExpertId, ExpertKey,
    ExpertLoadObservation, ExpertRegistration, ExpertStore, Sha256Hasher, TensorLocation,
    TensorMetadata,
};

#[cfg(feature = "m5-4-resident-dense")]
use crate::m5_4_resident_dense::{DenseSource, FIXED_RUNTIME_MEMORY_BYTES, ResidentDenseBudget};
#[cfg(feature = "m6-3-r2-native")]
use crate::r1_1_direct_candidate::{R2_1PackedProjectionBackend, r2_1_routed_experts_with_backend};
use crate::{
    KvCache, PINNED_QWEN3_30B_A3B_CONFIG,
    block::{
        ExpertMlpTrace, PreRouterOutput, cached_attention_with_weights, pre_router_with_weights,
        rms_norm, route_tokens,
    },
    cache::LayerKvUpdate,
    generation::greedy_token,
    quantized_layer0::{
        Group128Projection, Layer0QuantizedExpert, routed_layer0_quantized_experts_with_loader,
    },
    r1_1_direct_candidate::{
        R1_1CandidateReader, R1_1PackedArtifactLayout, r1_1_routed_experts_with_observer,
    },
    streaming::{
        PackedExpertLayout, streaming_routed_experts_with_observer,
        streaming_routed_experts_with_request_observer,
        streaming_routed_experts_with_trace_observer,
    },
};

#[path = "m5_2_trace_capture.rs"]
mod m5_2_trace_capture;
#[path = "r1_3_benchmark_tests.rs"]
mod r1_3_benchmark_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_1_quality_tests.rs"]
mod r2_1_quality_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d2_tests.rs"]
mod r2_2_d2_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d3_tests.rs"]
mod r2_2_d3_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d4_tests.rs"]
mod r2_2_d4_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d4d1_tests.rs"]
mod r2_2_d4d1_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d4d2_tests.rs"]
mod r2_2_d4d2_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d4d3_tests.rs"]
mod r2_2_d4d3_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d4d4_tests.rs"]
mod r2_2_d4d4_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d5b_tests.rs"]
mod r2_2_d5b_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_d5c_benchmark_tests.rs"]
mod r2_2_d5c_benchmark_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_diagnostic_tests.rs"]
mod r2_2_diagnostic_tests;
#[cfg(feature = "m6-3-r2-native")]
#[path = "r2_2_quality_tests.rs"]
mod r2_2_quality_tests;
#[path = "r2_3a_reference_tests.rs"]
mod r2_3a_reference_tests;
#[path = "r2_3b_characterization_tests.rs"]
mod r2_3b_characterization_tests;
#[cfg(feature = "m6-3-r2-localization")]
#[path = "r2_benchmark_tests.rs"]
mod r2_benchmark_tests;

const RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer0-runtime-plan-v1.tsv"
));
const BF16_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer0-checkpoint-plan-v1.tsv"
));
const BF16_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-transformers-layer0.safetensors"
));
const F32_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-f32-layer0-checkpoint-plan-v1.tsv"
));
const F32_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-transformers-f32-layer0.safetensors"
));
const LAYER1_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer1-dense-runtime-plan-v1.tsv"
));
const LAYER0_EXPERT_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer0-selected-expert-runtime-plan-v1.tsv"
));
const LAYER1_BF16_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer1-transformers-bf16-checkpoint-plan-v1.tsv"
));
const LAYER1_BF16_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer1-transformers-bf16-v1.safetensors"
));
const LAYER1_F32_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer1-transformers-f32-checkpoint-plan-v1.tsv"
));
const LAYER1_F32_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer1-transformers-f32-v1.safetensors"
));
const LAYER24_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer24-dense-runtime-plan-v1.tsv"
));
const LAYER24_EXPERT_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer24-expert-runtime-plan-v1.tsv"
));
const LAYER24_BF16_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer24-transformers-bf16-checkpoint-plan-v1.tsv"
));
const LAYER24_BF16_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer24-transformers-bf16-v1.safetensors"
));
const LAYER24_F32_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer24-transformers-f32-checkpoint-plan-v1.tsv"
));
const LAYER24_F32_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer24-transformers-f32-v1.safetensors"
));
const LAYER47_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer47-dense-runtime-plan-v1.tsv"
));
const LAYER47_EXPERT_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer47-expert-runtime-plan-v1.tsv"
));
const LAYER47_BF16_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer47-transformers-bf16-checkpoint-plan-v1.tsv"
));
const LAYER47_BF16_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer47-transformers-bf16-v1.safetensors"
));
const LAYER47_F32_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer47-transformers-f32-checkpoint-plan-v1.tsv"
));
const LAYER47_F32_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-02-layer47-transformers-f32-v1.safetensors"
));
const INTERMEDIATE_BF16_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-03-transformers-bf16-intermediate-plan-v1.tsv"
));
const INTERMEDIATE_BF16_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-03-transformers-bf16-intermediate-v1.safetensors"
));
const INTERMEDIATE_F32_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-03-transformers-f32-intermediate-plan-v1.tsv"
));
const INTERMEDIATE_F32_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-03-transformers-f32-intermediate-v1.safetensors"
));
const INTERMEDIATE_STRUCTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-03-intermediate-structure-v1.tsv"
));
const LAYER47_SELECTED_EXPERT_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-03-layer47-selected-expert-runtime-plan-v1.tsv"
));
const GENERATION_FINAL_DENSE_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-04-final-dense-runtime-plan-v1.tsv"
));
const GENERATION_LAYER47_EXPERT_RUNTIME_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-04-layer47-expert-runtime-plan-v1.tsv"
));
const GENERATION_BF16_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-04-transformers-bf16-generation-plan-v1.tsv"
));
const GENERATION_BF16_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-04-transformers-bf16-generation-v1.safetensors"
));
const GENERATION_F32_CHECKPOINT_PLAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-04-transformers-f32-generation-plan-v1.tsv"
));
const GENERATION_F32_CHECKPOINTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.2-04-transformers-f32-generation-v1.safetensors"
));
const TIER_B_F32_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m4.3-01-tier-b-transformers-f32-v1.tsv"
));
const M6_3_04_GROUP128_ARTIFACT_CONTRACT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/qwen3-30b-a3b/m6.3-04-layer0-group128-artifact-contract-v1.json"
));
const GENERATION_INPUT_TOKENS: [usize; 6] = [9707, 11, 1879, 0, 1096, 374];
const GENERATION_GUARD_LAYERS: [usize; 3] = [0, 24, 47];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrderedExpertTraceRecord {
    ordinal: usize,
    step: usize,
    position: usize,
    layer: usize,
    rank: usize,
    expert: usize,
    payload_bytes: usize,
    cache_hit: bool,
    loaded: bool,
    evictions: u64,
}
const GENERATION_DENSE_READS_PER_TOKEN: u64 = 1 + 48 * 9 + 594;
const GENERATION_EXPERT_READS_PER_TOKEN: u64 = 48 * 8;
const INTERMEDIATE_CASES: [(usize, usize, usize, usize); 8] = [
    (0, 0, 0, 62),
    (0, 0, 7, 91),
    (1, 0, 0, 68),
    (1, 0, 7, 127),
    (24, 1, 0, 85),
    (24, 1, 7, 8),
    (47, 0, 0, 54),
    (47, 0, 7, 36),
];
const INPUT_IDS: [usize; 4] = [9707, 11, 1879, 0];
const POSITION_IDS: [usize; 4] = [0, 1, 2, 3];
const POST_NORM_PROPAGATED_ABSOLUTE_BUDGET: f32 = 4.255_093e-6;
const LAYER1_INPUT_MAXIMUM_ERROR: f32 = 1.907_348_6e-6;
const LAYER1_INPUT_NORM_BUDGET: f32 = 6.222_046e-6;
const LAYER1_ATTENTION_BUDGET: f32 = 2.123_415_5e-6;
const LAYER1_RESIDUAL_BUDGET: f32 = 4.030_764e-6;
const LAYER1_POST_NORM_BUDGET: f32 = 6.222_046e-6;
const LAYER1_ROUTER_LOGIT_BUDGET: f32 = 2.217_292_8e-5;
const LAYER1_ROUTING_WEIGHT_BUDGET: f32 = 1.0e-6;
const LAYER24_FROZEN_INPUT_ERRORS: [f32; 25] = [
    0.0,
    1.907_348_6e-6,
    3.509_521_5e-4,
    7.934_570_3e-4,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
];
const LAYER24_FROZEN_INPUT_NORM_ERRORS: [f32; 25] = [
    2.086_162_6e-7,
    2.905_726_4e-7,
    5.364_418e-7,
    1.549_720_8e-6,
    9.536_743e-7,
    9.238_72e-7,
    1.192_092_9e-6,
    1.132_488_3e-6,
    1.192_092_9e-6,
    1.370_906_8e-6,
    1.251_697_5e-6,
    1.311_302_2e-6,
    1.788_139_3e-6,
    1.609_325_4e-6,
    2.026_558e-6,
    1.668_93e-6,
    1.907_348_6e-6,
    2.264_976_5e-6,
    2.384_185_8e-6,
    2.741_813_7e-6,
    2.980_232_2e-6,
    2.980_232_2e-6,
    2.741_813_7e-6,
    3.457_069_4e-6,
    3.457_069_4e-6,
];
const LAYER24_FROZEN_ATTENTION_ERRORS: [f32; 25] = [
    1.251_697_5e-6,
    5.755_573_5e-7,
    8.642_673_5e-7,
    7.748_604e-7,
    4.470_348_4e-7,
    1.430_511_5e-6,
    1.996_755_6e-6,
    3.039_836_9e-6,
    6.854_534e-6,
    1.966_953_3e-6,
    2.175_569_5e-6,
    1.171_603_8e-6,
    1.505_017_3e-6,
    1.728_534_7e-6,
    1.311_302_2e-6,
    1.110_136_5e-6,
    1.341_104_5e-6,
    1.549_720_8e-6,
    6.496_906_3e-6,
    6.020_069e-6,
    4.544_854e-6,
    3.159_046_2e-6,
    2.622_604_4e-6,
    1.713_633_5e-6,
    1.847_744e-6,
];
const LAYER24_FROZEN_RESIDUAL_ERRORS: [f32; 25] = [
    1.251_697_5e-6,
    1.907_348_6e-6,
    3.509_521_5e-4,
    7.934_570_3e-4,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
    2.319_336e-3,
];
const LAYER24_FROZEN_POST_NORM_ERRORS: [f32; 25] = [
    3.993_511e-6,
    2.622_604_4e-6,
    4.410_743_7e-6,
    2.136_230_5e-4,
    8.392_334e-5,
    9.918_213e-5,
    1.106_262_2e-4,
    9.536_743e-5,
    9.536_743e-5,
    1.029_968_3e-4,
    1.029_968_3e-4,
    1.029_968_3e-4,
    1.106_262_2e-4,
    1.335_144e-4,
    1.258_850_1e-4,
    1.373_291e-4,
    1.220_703_1e-4,
    1.068_115_2e-4,
    1.373_291e-4,
    1.602_172_9e-4,
    1.449_585e-4,
    1.296_997e-4,
    1.373_291e-4,
    1.602_172_9e-4,
    1.487_732e-4,
];
const LAYER24_FROZEN_ROUTER_ERRORS: [f32; 25] = [
    1.144_409_2e-5,
    1.239_776_6e-5,
    1.716_613_8e-5,
    2.622_604_4e-5,
    3.242_492_7e-5,
    2.861_023e-5,
    2.431_869_5e-5,
    2.241_134_6e-5,
    2.574_920_7e-5,
    2.288_818_4e-5,
    2.956_390_4e-5,
    2.193_451e-5,
    2.813_339_2e-5,
    3.242_492_7e-5,
    2.765_655_5e-5,
    3.719_33e-5,
    2.288_818_4e-5,
    2.098_083_5e-5,
    3.623_962_4e-5,
    3.051_757_8e-5,
    3.004_074_1e-5,
    2.908_706_7e-5,
    3.242_492_7e-5,
    3.242_492_7e-5,
    3.719_33e-5,
];
const LAYER24_FROZEN_MOE_ERRORS: [f32; 24] = [
    7.152_557_4e-7,
    3.509_521_5e-4,
    4.272_461e-4,
    1.495_361_3e-3,
    1.132_488_3e-6,
    8.642_673_5e-7,
    7.450_580_6e-7,
    2.712_011_3e-6,
    3.337_86e-6,
    7.450_580_6e-7,
    7.599_592e-7,
    6.854_534e-7,
    1.102_685_9e-6,
    2.086_162_6e-6,
    1.251_697_5e-6,
    9.089_708_3e-7,
    2.980_232_2e-6,
    8.791_685e-7,
    9.238_72e-7,
    1.735_985_3e-6,
    5.006_79e-6,
    1.370_906_8e-6,
    1.847_744e-6,
    1.281_499_9e-6,
];

#[derive(Debug, Clone, Copy)]
struct Layer24PropagatedBudgets {
    input: f32,
    input_norm: f32,
    attention: f32,
    residual: f32,
    post_norm: f32,
    router: f32,
    routing: f32,
    moe: Option<f32>,
    block: Option<f32>,
}

fn layer24_propagated_budgets(layer: usize) -> Layer24PropagatedBudgets {
    let completed_layer_budgets = |completed_layer: usize| {
        let moe = 3.0 * LAYER24_FROZEN_POST_NORM_ERRORS[completed_layer]
            + LAYER24_FROZEN_MOE_ERRORS[completed_layer];
        let block = LAYER24_FROZEN_RESIDUAL_ERRORS[completed_layer] + moe;
        (moe, block)
    };
    let input = if layer == 0 {
        0.0
    } else {
        3.0 * LAYER24_FROZEN_INPUT_ERRORS[layer] + 5.0e-7
    };
    let input_norm = 3.0 * LAYER24_FROZEN_INPUT_ERRORS[layer] + 5.0e-7;
    let attention =
        3.0 * LAYER24_FROZEN_INPUT_NORM_ERRORS[layer] + LAYER24_FROZEN_ATTENTION_ERRORS[layer];
    let residual = LAYER24_FROZEN_INPUT_ERRORS[layer] + attention;
    let post_norm = 3.0 * LAYER24_FROZEN_RESIDUAL_ERRORS[layer] + 5.0e-7;
    let router = 3.0 * LAYER24_FROZEN_POST_NORM_ERRORS[layer] + 1.430_511_5e-5;
    let routing = 0.5 * LAYER24_FROZEN_ROUTER_ERRORS[layer] + 1.0e-7;
    let (moe, block) = if layer < 24 {
        let (moe, block) = completed_layer_budgets(layer);
        (Some(moe), Some(block))
    } else {
        (None, None)
    };
    Layer24PropagatedBudgets {
        input,
        input_norm,
        attention,
        residual,
        post_norm,
        router,
        routing,
        moe,
        block,
    }
}

const LAYER47_FROZEN_INPUT_NORM_SUFFIX: [f32; 23] = [
    3.576_278_7e-6,
    5.006_79e-6,
    4.291_534_4e-6,
    9.059_906e-6,
    5.722_046e-6,
    5.006_79e-6,
    9.298_325e-6,
    8.344_65e-6,
    6.675_72e-6,
    5.722_046e-6,
    7.390_976e-6,
    5.245_208_7e-6,
    5.245_208_7e-6,
    1.096_725_5e-5,
    8.106_232e-6,
    2.098_083_5e-5,
    1.478_195_2e-5,
    1.478_195_2e-5,
    1.239_776_6e-5,
    1.478_195_2e-5,
    2.574_920_7e-5,
    2.288_818_4e-5,
    3.051_757_8e-5,
];
const LAYER47_FROZEN_ATTENTION_SUFFIX: [f32; 23] = [
    3.606_081e-6,
    1.370_906_8e-6,
    3.039_836_9e-6,
    2.548_098_6e-6,
    1.221_895_2e-6,
    2.503_395e-6,
    6.318_092_3e-6,
    5.706_213_4e-6,
    6.563_961_5e-6,
    3.218_650_8e-6,
    4.082_918e-6,
    1.527_369e-6,
    6.169_080_7e-6,
    6.675_72e-6,
    3.814_697_3e-6,
    1.335_144e-5,
    1.049_041_75e-5,
    2.050_399_8e-5,
    1.335_144e-5,
    2.765_655_5e-5,
    8.821_487e-6,
    4.291_534_4e-5,
    1.373_291e-4,
];
const LAYER47_FROZEN_POST_NORM_SUFFIX: [f32; 23] = [
    1.525_878_9e-4,
    1.678_466_8e-4,
    2.136_230_5e-4,
    1.449_585e-4,
    2.059_936_5e-4,
    2.136_230_5e-4,
    1.983_642_6e-4,
    1.907_348_6e-4,
    1.602_172_9e-4,
    1.602_172_9e-4,
    1.449_585e-4,
    1.602_172_9e-4,
    1.907_348_6e-4,
    1.602_172_9e-4,
    2.136_230_5e-4,
    2.136_230_5e-4,
    1.983_642_6e-4,
    2.365_112_3e-4,
    1.831_054_7e-4,
    1.907_348_6e-4,
    1.754_760_7e-4,
    2.441_406_3e-4,
    7.629_394_5e-5,
];
const LAYER47_FROZEN_ROUTER_SUFFIX: [f32; 23] = [
    3.242_492_7e-5,
    3.433_227_5e-5,
    4.673_004e-5,
    2.288_818_4e-5,
    3.147_125_2e-5,
    3.051_757_8e-5,
    2.622_604_4e-5,
    3.910_064_7e-5,
    3.242_492_7e-5,
    3.528_595e-5,
    2.717_971_8e-5,
    3.623_962_4e-5,
    4.482_269_3e-5,
    4.005_432e-5,
    3.814_697_3e-5,
    2.670_288e-5,
    2.431_869_5e-5,
    2.813_339_2e-5,
    2.384_185_8e-5,
    2.479_553_2e-5,
    2.479_553_2e-5,
    2.574_920_7e-5,
    3.004_074_1e-5,
];
const LAYER47_FROZEN_MOE_SUFFIX: [f32; 23] = [
    5.364_418e-6,
    3.337_86e-6,
    1.144_409_2e-5,
    1.259_148_1e-6,
    8.106_232e-6,
    1.192_092_9e-6,
    1.169_741_2e-6,
    1.292_675_7e-6,
    7.867_813e-6,
    2.771_616e-6,
    1.817_941_7e-6,
    2.682_209e-6,
    2.861_023e-6,
    5.364_418e-6,
    1.966_953_3e-6,
    3.457_069_4e-6,
    9.059_906e-6,
    5.722_046e-6,
    1.025_199_9e-5,
    3.242_492_7e-5,
    1.668_93e-5,
    1.239_776_6e-5,
    1.342_773_4e-3,
];

fn layer47_frozen_input_error(layer: usize) -> f32 {
    if layer <= 24 {
        LAYER24_FROZEN_INPUT_ERRORS[layer]
    } else if layer < 47 {
        2.319_336e-3
    } else {
        9.765_625e-4
    }
}

fn layer47_frozen_residual_error(layer: usize) -> f32 {
    if layer <= 24 {
        LAYER24_FROZEN_RESIDUAL_ERRORS[layer]
    } else if layer < 47 {
        2.319_336e-3
    } else {
        9.155_273_4e-4
    }
}

fn layer47_suffix_value(layer: usize, prefix: &[f32; 25], suffix: &[f32; 23]) -> f32 {
    if layer <= 24 {
        prefix[layer]
    } else {
        suffix[layer - 25]
    }
}

fn layer47_moe_error(layer: usize) -> f32 {
    if layer < 24 {
        LAYER24_FROZEN_MOE_ERRORS[layer]
    } else {
        LAYER47_FROZEN_MOE_SUFFIX[layer - 24]
    }
}

fn layer47_propagated_budgets(layer: usize) -> Layer24PropagatedBudgets {
    let input_error = layer47_frozen_input_error(layer);
    let input_norm_error = layer47_suffix_value(
        layer,
        &LAYER24_FROZEN_INPUT_NORM_ERRORS,
        &LAYER47_FROZEN_INPUT_NORM_SUFFIX,
    );
    let attention_error = layer47_suffix_value(
        layer,
        &LAYER24_FROZEN_ATTENTION_ERRORS,
        &LAYER47_FROZEN_ATTENTION_SUFFIX,
    );
    let residual_error = layer47_frozen_residual_error(layer);
    let post_norm_error = layer47_suffix_value(
        layer,
        &LAYER24_FROZEN_POST_NORM_ERRORS,
        &LAYER47_FROZEN_POST_NORM_SUFFIX,
    );
    let router_error = layer47_suffix_value(
        layer,
        &LAYER24_FROZEN_ROUTER_ERRORS,
        &LAYER47_FROZEN_ROUTER_SUFFIX,
    );
    let input = if layer == 0 {
        0.0
    } else {
        3.0 * input_error + 5.0e-7
    };
    let input_norm = 3.0 * input_error + 5.0e-7;
    let attention = 3.0 * input_norm_error + attention_error;
    let residual = input_error + attention;
    let post_norm = 3.0 * residual_error + 5.0e-7;
    let router = 3.0 * post_norm_error + 1.430_511_5e-5;
    let routing = 0.5 * router_error + 1.0e-7;
    let (moe, block) = if layer < 47 {
        let moe = 3.0 * post_norm_error + layer47_moe_error(layer);
        (Some(moe), Some(residual_error + moe))
    } else {
        (None, None)
    };
    Layer24PropagatedBudgets {
        input,
        input_norm,
        attention,
        residual,
        post_norm,
        router,
        routing,
        moe,
        block,
    }
}

fn selected_intermediate_budget(layer: usize, checkpoint: &str) -> f32 {
    if checkpoint == "routing_weight" {
        return layer47_propagated_budgets(layer).routing;
    }
    match (layer, checkpoint) {
        (0, "expert_input") => 7.116_116e-6,
        (0, "gate_projection") => 7.006_790_2e-6,
        (0, "up_projection") => 7.722_046e-6,
        (0, "activated_gate") => 3.092_802e-6,
        (0, "activated_product") => 6.722_046e-6,
        (0, "down_projection") => 2.296_401e-6,
        (0, "weighted_expert_output") => 1.625_848_8e-6,
        (0, "aggregated_moe_output") => 2.117_587e-6,
        (0, "moe_residual_addition" | "final_block_output") => 3.503_395e-6,
        (1, "expert_input") => 8.367_813e-6,
        (1, "gate_projection") => 3.633_227_5e-5,
        (1, "up_projection") => 2.488_818_4e-5,
        (1, "activated_gate") => 3.483_227_5e-5,
        (1, "activated_product") => 8.249_746e-4,
        (1, "down_projection") => 1.648_949_2e-3,
        (
            1,
            "weighted_expert_output"
            | "aggregated_moe_output"
            | "moe_residual_addition"
            | "final_block_output",
        ) => 1.053_856_4e-3,
        (24, "expert_input") => 1.265_934_8e-5,
        (24, "gate_projection") => 1.988_139_3e-5,
        (24, "up_projection") => 1.558_986e-5,
        (24, "activated_gate") => 9.440_697e-6,
        (24, "activated_product") => 9.225_441e-6,
        (24, "down_projection") => 4.017_485e-6,
        (24, "weighted_expert_output") => 2.788_139_3e-6,
        (24, "aggregated_moe_output") => 3.145_767e-6,
        (24, "moe_residual_addition" | "final_block_output") => 3.390_176_4e-5,
        (47, "expert_input") => 2.293_818_4e-4,
        (47, "gate_projection") => 1.193_019_4e-4,
        (47, "up_projection") => 7.638_66e-5,
        (47, "activated_gate") => 1.178_019_4e-4,
        (47, "activated_product") => 6.604_658e-4,
        (47, "down_projection") => 1.408_623_3e-3,
        (47, "weighted_expert_output") => 2.985_464e-4,
        (47, "aggregated_moe_output") => 1.030_968_3e-3,
        (47, "moe_residual_addition" | "final_block_output") => 2.106_713e-3,
        _ => panic!("missing Layer-{layer} {checkpoint} intermediate budget"),
    }
}

fn short_generation_budget(step: usize, checkpoint: &str) -> f32 {
    assert!(step < 6, "short-generation budget step");
    match checkpoint {
        "expert_outputs" => [
            1.980_828e-3,
            5.417_333_4e-4,
            2.069_936_5e-4,
            3.686_414_5e-4,
            2.527_700_2e-4,
            2.985_464e-4,
        ][step],
        "routing_weights" => [
            2.759_857_3e-6,
            2.424_581e-6,
            1.709_325_4e-6,
            1.843_435_9e-6,
            1.284_642_3e-6,
            1.575_215e-6,
        ][step],
        "moe_output" => [
            1.328_514_6e-3,
            2.642_141e-4,
            6.680_353e-5,
            2.642_141e-4,
            8.647_306e-5,
            1.383_291e-4,
        ][step],
        "block_output" => [
            1.557_396_5e-3,
            3.443_227_5e-4,
            5.732_046e-4,
            5.045_400_4e-4,
            3.900_991_2e-4,
            1.841_054_7e-4,
        ][step],
        "final_norm" => [
            1.556_896_5e-3,
            3.438_227_5e-4,
            5.727_046e-4,
            5.040_400_4e-4,
            3.895_991_2e-4,
            1.836_054_7e-4,
        ][step],
        "logits" => [
            2.051_326_3e-4,
            1.564_952_4e-4,
            1.336_070_6e-4,
            1.021_358e-4,
            1.164_409_2e-4,
            1.364_680_8e-4,
        ][step],
        _ => panic!("missing M4.2-04 {checkpoint} budget"),
    }
}

fn tier_b_fixture_budget(fixture: &str, checkpoint: &str) -> f32 {
    let observed = match (fixture, checkpoint) {
        ("single_low_token", "final_norm") => 3.208_22e-5,
        ("short_english", "final_norm") => 4.172_325e-6,
        ("short_thai", "final_norm") => 1.144_409_2e-5,
        ("code_newline", "final_norm") => 3.695_488e-6,
        ("repeated_pattern", "final_norm") => 2.920_627_6e-6,
        ("special_token", "final_norm") => 7.271_766_7e-6,
        ("single_low_token", "logits") => 2.841_949_5e-4,
        ("short_english", "logits") => 4.768_371_6e-5,
        ("short_thai", "logits") => 4.386_902e-5,
        ("code_newline", "logits") => 3.051_757_8e-5,
        ("repeated_pattern" | "special_token", "logits") => 2.098_083_5e-5,
        _ => panic!("missing Tier B {fixture} {checkpoint} budget observation"),
    };
    match checkpoint {
        "final_norm" => 3.0 * observed + 5.0e-7,
        "logits" => 3.0 * observed + 2.0e-6,
        _ => unreachable!(),
    }
}

fn maximum_magnitude(tensor: &Tensor) -> f32 {
    tensor
        .data()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max)
}

#[derive(Debug, Clone)]
struct RangeRecord {
    offset: u64,
    length: usize,
    shape: Vec<usize>,
}

#[derive(Debug)]
struct RuntimePlan {
    payload: PathBuf,
    payload_length: u64,
    tensors: HashMap<String, RangeRecord>,
}

#[derive(Debug)]
struct CheckpointRecord {
    data_type: &'static str,
    range: RangeRecord,
}

#[derive(Debug, Clone, Copy)]
struct StageMetrics {
    maximum_absolute_difference: f32,
    maximum_relative_difference: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReuseSummary {
    unique_requests: usize,
    repeated_requests: usize,
    minimum_distance: usize,
    median_distance: usize,
    maximum_distance: usize,
    distance_at_most_384: usize,
    distance_385_through_768: usize,
    distance_above_768: usize,
}

fn summarize_reuse_distances(requests: &[usize]) -> ReuseSummary {
    let mut last_seen = HashMap::new();
    let mut distances = Vec::new();
    for (index, &key) in requests.iter().enumerate() {
        if let Some(previous) = last_seen.insert(key, index) {
            distances.push(index - previous);
        }
    }
    distances.sort_unstable();
    let repeated_requests = distances.len();
    ReuseSummary {
        unique_requests: last_seen.len(),
        repeated_requests,
        minimum_distance: distances.first().copied().unwrap_or(0),
        median_distance: distances
            .get(repeated_requests.saturating_sub(1) / 2)
            .copied()
            .unwrap_or(0),
        maximum_distance: distances.last().copied().unwrap_or(0),
        distance_at_most_384: distances.iter().filter(|&&value| value <= 384).count(),
        distance_385_through_768: distances
            .iter()
            .filter(|&&value| (385..=768).contains(&value))
            .count(),
        distance_above_768: distances.iter().filter(|&&value| value > 768).count(),
    }
}

#[allow(clippy::cast_precision_loss)]
fn reporting_ratio(numerator: u64, denominator: u64) -> f64 {
    numerator as f64 / denominator as f64
}

#[test]
fn m4_2_05_counter_reconciliation_and_reuse_summary_are_exact() {
    let summary = summarize_reuse_distances(&[1, 2, 1, 3, 1, 2, 4]);
    assert_eq!(summary.unique_requests, 4);
    assert_eq!(summary.repeated_requests, 3);
    assert_eq!(summary.minimum_distance, 2);
    assert_eq!(summary.median_distance, 2);
    assert_eq!(summary.maximum_distance, 4);
    assert_eq!(summary.distance_at_most_384, 3);
    assert_eq!(summary.distance_385_through_768, 0);
    assert_eq!(summary.distance_above_768, 0);
    let misses = 7_u64;
    let loads = 7_u64;
    let capacity = 1_u64;
    let evictions = 6_u64;
    assert_eq!(misses, loads);
    assert_eq!(evictions, loads - capacity);
}

fn parse_shape(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|part| part.parse().expect("valid shape dimension"))
        .collect()
}

fn runtime_plan(source: &str) -> RuntimePlan {
    let mut payload = None;
    let mut payload_length = None;
    let mut tensors = HashMap::new();
    for line in source.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        match fields[0] {
            "payload" => {
                assert_eq!(fields.len(), 4, "invalid runtime payload record");
                payload = Some(PathBuf::from(fields[1]));
                payload_length = Some(fields[2].parse().expect("payload length"));
            }
            "tensor" => {
                assert_eq!(fields.len(), 5, "invalid runtime tensor record");
                assert!(
                    tensors
                        .insert(
                            fields[1].to_owned(),
                            RangeRecord {
                                offset: fields[2].parse().expect("tensor offset"),
                                length: fields[3].parse().expect("tensor length"),
                                shape: parse_shape(fields[4]),
                            },
                        )
                        .is_none(),
                    "duplicate runtime tensor {}",
                    fields[1]
                );
            }
            _ => {}
        }
    }
    RuntimePlan {
        payload: payload.expect("runtime payload path"),
        payload_length: payload_length.expect("runtime payload length"),
        tensors,
    }
}

fn checkpoint_plan(plan: &str) -> HashMap<String, CheckpointRecord> {
    let mut tensors = HashMap::new();
    for line in plan.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields[0] != "tensor" {
            continue;
        }
        assert_eq!(fields.len(), 6, "invalid checkpoint record");
        assert!(
            tensors
                .insert(
                    fields[1].to_owned(),
                    CheckpointRecord {
                        data_type: match fields[2] {
                            "F32" => "F32",
                            "I64" => "I64",
                            other => panic!("unsupported checkpoint dtype {other}"),
                        },
                        range: RangeRecord {
                            offset: fields[3].parse().expect("checkpoint offset"),
                            length: fields[4].parse().expect("checkpoint length"),
                            shape: parse_shape(fields[5]),
                        },
                    },
                )
                .is_none(),
            "duplicate checkpoint {}",
            fields[1]
        );
    }
    tensors
}

trait DenseReadAt {
    fn read_exact_at(&mut self, offset: u64, destination: &mut [u8]);
}

impl DenseReadAt for File {
    fn read_exact_at(&mut self, offset: u64, destination: &mut [u8]) {
        self.seek(SeekFrom::Start(offset)).expect("seek artifact");
        self.read_exact(destination).expect("read artifact range");
    }
}

#[cfg(feature = "m5-4-resident-dense")]
impl DenseReadAt for DenseSource {
    fn read_exact_at(&mut self, offset: u64, destination: &mut [u8]) {
        DenseSource::read_exact_at(self, offset, destination).expect("read resident dense range");
    }
}

fn read_exact_range(
    reader: &mut impl DenseReadAt,
    record: &RangeRecord,
    bytes_read: &mut u64,
) -> Vec<u8> {
    let mut bytes = vec![0_u8; record.length];
    reader.read_exact_at(record.offset, &mut bytes);
    *bytes_read += u64::try_from(bytes.len()).expect("range length fits u64");
    bytes
}

#[cfg(feature = "m5-4-resident-dense")]
fn dense_source_for_generation(path: &Path, expert_cache_budget_bytes: usize) -> DenseSource {
    match env::var("COLIBRI_DENSE_RESIDENCY_MODE").as_deref() {
        Ok("resident_dense") => {
            let total_budget_bytes = env::var("COLIBRI_TOTAL_RAM_BUDGET_BYTES")
                .expect("resident dense requires COLIBRI_TOTAL_RAM_BUDGET_BYTES")
                .parse::<usize>()
                .expect("valid resident dense total budget");
            DenseSource::resident(
                path,
                ResidentDenseBudget {
                    total_budget: total_budget_bytes,
                    expert_cache_budget: expert_cache_budget_bytes,
                    fixed_runtime_memory: FIXED_RUNTIME_MEMORY_BYTES,
                },
            )
            .expect("resident dense initialization")
        }
        Ok("streamed_dense") | Err(_) => {
            DenseSource::streaming(path).expect("streamed dense initialization")
        }
        Ok(other) => panic!("unsupported COLIBRI_DENSE_RESIDENCY_MODE: {other}"),
    }
}

fn f32_tensor(bytes: &[u8], shape: &[usize]) -> Tensor {
    let data = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect();
    Tensor::new(TensorShape::new(shape.to_vec()), data).expect("valid f32 tensor")
}

fn artifact_tensor(
    reader: &mut impl DenseReadAt,
    plan: &RuntimePlan,
    name: &str,
    bytes_read: &mut u64,
) -> Tensor {
    let record = plan
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing runtime tensor {name}"));
    f32_tensor(&read_exact_range(reader, record, bytes_read), &record.shape)
}

fn embedding_rows(
    reader: &mut impl DenseReadAt,
    plan: &RuntimePlan,
    bytes_read: &mut u64,
) -> Tensor {
    let record = &plan.tensors["model.embed_tokens.weight"];
    assert_eq!(record.shape, [151_936, 2048]);
    let row_bytes = 2048 * 4;
    let mut data = Vec::with_capacity(INPUT_IDS.len() * 2048);
    for token_id in INPUT_IDS {
        let offset = record.offset + u64::try_from(token_id * row_bytes).expect("embedding offset");
        let row = RangeRecord {
            offset,
            length: row_bytes,
            shape: vec![2048],
        };
        data.extend(f32_tensor(&read_exact_range(reader, &row, bytes_read), &[2048]).into_data());
    }
    Tensor::new(TensorShape::new([4, 2048]), data).expect("embedding rows")
}

fn embedding_row(
    reader: &mut impl DenseReadAt,
    plan: &RuntimePlan,
    token_id: usize,
    bytes_read: &mut u64,
) -> Tensor {
    let record = &plan.tensors["model.embed_tokens.weight"];
    assert_eq!(record.shape, [151_936, 2048]);
    let row_bytes = 2048 * 4;
    let row = RangeRecord {
        offset: record.offset + u64::try_from(token_id * row_bytes).expect("embedding offset"),
        length: row_bytes,
        shape: vec![1, 2048],
    };
    f32_tensor(&read_exact_range(reader, &row, bytes_read), &[1, 2048])
}

fn streaming_language_model_head(
    reader: &mut impl DenseReadAt,
    plan: &RuntimePlan,
    hidden: &Tensor,
    bytes_read: &mut u64,
) -> Tensor {
    const ROWS_PER_CHUNK: usize = 256;
    const HIDDEN: usize = 2048;
    let record = &plan.tensors["lm_head.weight"];
    assert_eq!(record.shape, [151_936, 2048]);
    assert_eq!(hidden.shape().dimensions(), [1, 2048]);
    let mut logits = Vec::with_capacity(151_936);
    let row_bytes = HIDDEN * 4;
    for row_start in (0..151_936).step_by(ROWS_PER_CHUNK) {
        let row_count = ROWS_PER_CHUNK.min(151_936 - row_start);
        let chunk = RangeRecord {
            offset: record.offset
                + u64::try_from(row_start * row_bytes).expect("LM-head chunk offset"),
            length: row_count * row_bytes,
            shape: vec![row_count, HIDDEN],
        };
        let weights = f32_tensor(
            &read_exact_range(reader, &chunk, bytes_read),
            &[row_count, HIDDEN],
        );
        logits.extend(weights.data().chunks_exact(HIDDEN).map(|row| {
            hidden
                .data()
                .iter()
                .zip(row)
                .map(|(left, right)| left * right)
                .sum::<f32>()
        }));
    }
    Tensor::new(TensorShape::new([1, 151_936]), logits).expect("LM-head logits")
}

fn deterministic_top_ids(logits: &Tensor, count: usize) -> Vec<usize> {
    assert_eq!(logits.shape().dimensions(), [1, 151_936]);
    let mut ids: Vec<_> = (0..151_936).collect();
    ids.sort_by(|left, right| {
        logits.data()[*right]
            .total_cmp(&logits.data()[*left])
            .then_with(|| left.cmp(right))
    });
    ids.truncate(count);
    ids
}

struct LayerWeights {
    input_norm: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    post_norm: Tensor,
    router: Tensor,
}

fn layer_weights(
    reader: &mut impl DenseReadAt,
    plan: &RuntimePlan,
    layer: usize,
    bytes_read: &mut u64,
) -> LayerWeights {
    let prefix = format!("model.layers.{layer}");
    let mut read =
        |suffix: &str| artifact_tensor(reader, plan, &format!("{prefix}.{suffix}"), bytes_read);
    LayerWeights {
        input_norm: read("input_layernorm.weight"),
        query: read("self_attn.q_proj.weight"),
        key: read("self_attn.k_proj.weight"),
        value: read("self_attn.v_proj.weight"),
        output: read("self_attn.o_proj.weight"),
        query_norm: read("self_attn.q_norm.weight"),
        key_norm: read("self_attn.k_norm.weight"),
        post_norm: read("post_attention_layernorm.weight"),
        router: read("mlp.gate.weight"),
    }
}

fn decode_sha256(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64, "SHA-256 text length");
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .expect("lowercase SHA-256 hex");
    }
    output
}

fn selected_expert_store(artifact_root: &std::path::Path) -> ExpertStore {
    expert_store_from_plan(LAYER0_EXPERT_RUNTIME_PLAN, artifact_root, 27)
}

/// Reads the canonical v1 writer grammar just far enough for this diagnostic
/// seam. This is deliberately private/test-only: production continues to use
/// its established runtime plans and never parses artifact JSON at inference.
fn canonical_layer0_expert_store(artifact_root: &Path) -> ExpertStore {
    const SHARD: &str = "experts-layer-00000-of-00048.bin";
    const PAYLOAD: u64 = 18_874_368;
    const PROJECTIONS: [&str; 3] = [
        "\"gate\": {\"offset\": 0, \"length\": 6291456, \"shape\": [768, 2048]}",
        "\"up\": {\"offset\": 6291456, \"length\": 6291456, \"shape\": [768, 2048]}",
        "\"down\": {\"offset\": 12582912, \"length\": 6291456, \"shape\": [2048, 768]}",
    ];
    let text = fs::read_to_string(artifact_root.join("experts/expert-manifest-v1.json"))
        .expect("read canonical expert manifest");
    for required in [
        "\"format_version\": 1",
        "\"model_id\": \"Qwen/Qwen3-30B-A3B\"",
        "\"model_revision\": \"ad44e777bcd18fa416d9da3bd8f70d33ebb85d39\"",
        "\"source_dtype\": \"BF16\"",
        "\"artifact_dtype\": \"F32\"",
        "\"endianness\": \"little\"",
        SHARD,
    ] {
        assert!(
            text.contains(required),
            "canonical expert manifest missing {required}"
        );
    }
    let shard_length = fs::metadata(artifact_root.join("experts").join(SHARD))
        .expect("canonical Layer-0 shard metadata")
        .len();
    let mut metadata = Vec::with_capacity(128);
    let mut registrations = Vec::with_capacity(128);
    let mut seen = HashSet::new();
    let mut record = String::new();
    let mut in_record = false;
    for line in text.lines() {
        if line.trim_start().starts_with("\"layer\":") {
            in_record = true;
            record.clear();
        }
        if in_record {
            record.push_str(line.trim());
            if line.trim() == "}," || line.trim() == "}" {
                in_record = false;
                if r1_1a_manifest_number(&record, "layer") != 0 {
                    continue;
                }
                assert!(
                    PROJECTIONS
                        .iter()
                        .all(|projection| record.contains(projection)),
                    "invalid Layer-0 projection layout"
                );
                assert!(record.contains("\"source_dtype\": \"BF16\""));
                assert!(record.contains("\"artifact_dtype\": \"F32\""));
                let expert = r1_1a_manifest_number(&record, "expert");
                let shard_id = r1_1a_manifest_number(&record, "shard_id");
                let offset = r1_1a_manifest_number(&record, "payload_offset");
                let length = r1_1a_manifest_number(&record, "payload_length");
                assert_eq!(shard_id, 0, "Layer-0 must use shard zero");
                assert_eq!(length, PAYLOAD, "Layer-0 payload length");
                assert!(
                    expert < 128 && seen.insert(expert),
                    "duplicate/out-of-range Layer-0 expert"
                );
                assert!(
                    offset
                        .checked_add(length)
                        .is_some_and(|end| end <= shard_length),
                    "Layer-0 payload range out of bounds"
                );
                let hash = record
                    .rsplit_once("\"sha256\": \"")
                    .and_then(|(_, tail)| tail.split_once('"'))
                    .map(|(hash, _)| hash)
                    .expect("expert payload SHA-256");
                let name = format!("model.layers.0.mlp.experts.{expert}");
                metadata.push(TensorMetadata {
                    name: name.clone(),
                    shape: TensorShape::new([usize::try_from(length / 4).expect("payload shape")]),
                    data_type: DataType::F32,
                    location: TensorLocation {
                        path: SHARD.into(),
                        offset,
                        length,
                    },
                    sha256: decode_sha256(hash),
                });
                registrations.push(ExpertRegistration {
                    key: ExpertKey {
                        layer_index: 0,
                        expert_id: ExpertId(u32::try_from(expert).expect("expert ID")),
                    },
                    tensor_name: name,
                });
            }
        }
    }
    assert_eq!(
        seen,
        (0..128).collect(),
        "Layer-0 IDs must be exactly 0..127"
    );
    let manifest = ArtifactManifest::new(ARTIFACT_FORMAT_VERSION, ByteOrder::Little, metadata)
        .expect("validated Layer-0 diagnostic manifest");
    let reader = ArtifactReader::open(artifact_root.join("experts"), manifest)
        .expect("open canonical Layer-0 reader");
    ExpertStore::new(
        reader,
        registrations,
        usize::try_from(PAYLOAD).expect("Layer-0 payload length fits usize"),
    )
    .expect("canonical Layer-0 store")
}

fn r1_1a_manifest_number(record: &str, key: &str) -> u64 {
    let prefix = format!("\"{key}\":");
    record
        .split_once(&prefix)
        .and_then(|(_, tail)| {
            tail.trim_start()
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok()
        })
        .expect("canonical manifest numeric field")
}

fn expert_store_from_plan(
    plan: &str,
    artifact_root: &std::path::Path,
    expected_registrations: usize,
) -> ExpertStore {
    expert_store_from_plans(&[plan], artifact_root, expected_registrations)
}

fn expert_store_from_plans(
    plans: &[&str],
    artifact_root: &std::path::Path,
    expected_registrations: usize,
) -> ExpertStore {
    let mut metadata = Vec::new();
    let mut registrations = Vec::new();
    for plan in plans {
        for line in plan.lines() {
            let fields: Vec<_> = line.split('\t').collect();
            if fields[0] != "expert" {
                continue;
            }
            assert_eq!(fields.len(), 8, "invalid selected-expert plan record");
            let layer_index: u32 = fields[1].parse().expect("expert layer");
            let expert_id: u32 = fields[2].parse().expect("expert ID");
            let name = fields[3].to_owned();
            let length: u64 = fields[6].parse().expect("expert payload length");
            assert_eq!(length, 18_874_368, "packed expert payload length");
            metadata.push(TensorMetadata {
                name: name.clone(),
                shape: TensorShape::new([
                    usize::try_from(length / 4).expect("expert F32 element count")
                ]),
                data_type: DataType::F32,
                location: TensorLocation {
                    path: fields[4].into(),
                    offset: fields[5].parse().expect("expert payload offset"),
                    length,
                },
                sha256: decode_sha256(fields[7]),
            });
            registrations.push(ExpertRegistration {
                key: ExpertKey {
                    layer_index,
                    expert_id: ExpertId(expert_id),
                },
                tensor_name: name,
            });
        }
    }
    assert_eq!(
        registrations.len(),
        expected_registrations,
        "expert registration count"
    );
    let manifest = ArtifactManifest::new(ARTIFACT_FORMAT_VERSION, ByteOrder::Little, metadata)
        .expect("selected expert artifact manifest");
    #[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
    let reader_mode = match env::var("COLIBRI_EXPERT_READER_MODE").as_deref() {
        Ok("reference_allocated") | Err(_) => ReaderMode::Reference,
        #[cfg(feature = "m5-3-reusable-buffer")]
        Ok("reusable_aligned_buffer") => ReaderMode::ReusableAlignedBuffer,
        #[cfg(feature = "m5-3-mmap")]
        Ok("mmap_read_only") => ReaderMode::MmapReadOnly,
        Ok(other) => panic!("unsupported COLIBRI_EXPERT_READER_MODE: {other}"),
    };
    #[cfg(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap"))]
    let reader =
        ArtifactReader::open_with_mode(artifact_root.join("experts"), manifest, reader_mode)
            .expect("canonical selected expert reader");
    #[cfg(not(any(feature = "m5-3-reusable-buffer", feature = "m5-3-mmap")))]
    let reader = ArtifactReader::open(artifact_root.join("experts"), manifest)
        .expect("canonical selected expert reader");
    let budget = env::var("COLIBRI_EXPERT_CACHE_BUDGET_BYTES").map_or(18_874_368, |value| {
        value.parse::<usize>().expect("valid expert cache budget")
    });
    ExpertStore::new(reader, registrations, budget).expect("configured expert cache budget")
}

fn checkpoint_f32(bytes: &[u8], plan: &HashMap<String, CheckpointRecord>, name: &str) -> Tensor {
    let record = &plan[name];
    assert_eq!(record.data_type, "F32", "checkpoint {name} dtype");
    let start = usize::try_from(record.range.offset).expect("checkpoint offset fits usize");
    let end = start + record.range.length;
    f32_tensor(&bytes[start..end], &record.range.shape)
}

fn checkpoint_ids(
    bytes: &[u8],
    plan: &HashMap<String, CheckpointRecord>,
    name: &str,
) -> Vec<usize> {
    let record = &plan[name];
    assert_eq!(record.data_type, "I64", "checkpoint {name} dtype");
    let start = usize::try_from(record.range.offset).expect("checkpoint offset fits usize");
    let end = start + record.range.length;
    bytes[start..end]
        .chunks_exact(8)
        .map(|chunk| {
            usize::try_from(i64::from_le_bytes(
                chunk.try_into().expect("eight-byte i64"),
            ))
            .expect("non-negative checkpoint ID")
        })
        .collect()
}

fn tensor_row(tensor: &Tensor, row: usize) -> Tensor {
    let width = tensor.shape().dimensions()[1];
    Tensor::new(
        TensorShape::new([width]),
        tensor.data()[row * width..(row + 1) * width].to_vec(),
    )
    .expect("valid tensor row")
}

fn intermediate_case_name(
    layer: usize,
    token: usize,
    position: usize,
    expert: usize,
    checkpoint: &str,
) -> String {
    format!("layer{layer}_token{token}_position{position}_expert{expert}_{checkpoint}")
}

fn intermediate_layer_name(layer: usize, token: usize, checkpoint: &str) -> String {
    format!("layer{layer}_token{token}_{checkpoint}")
}

fn validate_intermediate_structure(layout: PackedExpertLayout) {
    let aggregation = [
        "62,21,75,87,36,126,34,91",
        "68,114,55,90,0,9,30,127",
        "85,65,28,50,24,105,78,8",
        "54,32,73,34,85,111,117,36",
    ];
    let mut records = INTERMEDIATE_STRUCTURE.lines();
    assert_eq!(
        records.next(),
        Some(
            "record\tlayer\ttoken\tposition\texpert\trole\tshape\torientation\tsource_name\tsource_shard\tsource_offset\tsource_length\tartifact_path\tartifact_payload_offset\tartifact_projection_offset\tartifact_projection_length\taggregation_experts"
        )
    );
    for (case_index, &(layer, token, position, expert)) in INTERMEDIATE_CASES.iter().enumerate() {
        for role in ["gate", "up", "down"] {
            let fields: Vec<_> = records
                .next()
                .expect("missing intermediate structure record")
                .split('\t')
                .collect();
            assert_eq!(fields.len(), 17, "intermediate structure field count");
            assert_eq!(fields[0], "projection");
            assert_eq!(fields[1].parse::<usize>().unwrap(), layer);
            assert_eq!(fields[2].parse::<usize>().unwrap(), token);
            assert_eq!(fields[3].parse::<usize>().unwrap(), position);
            assert_eq!(fields[4].parse::<usize>().unwrap(), expert);
            assert_eq!(fields[5], role, "gate/up/down ordering");
            let (shape, orientation, projection_offset, projection_length) = match role {
                "gate" => (
                    "768,2048",
                    "output_by_input",
                    layout.gate_offset,
                    layout.gate_length,
                ),
                "up" => (
                    "768,2048",
                    "output_by_input",
                    layout.up_offset,
                    layout.up_length,
                ),
                "down" => (
                    "2048,768",
                    "output_by_intermediate",
                    layout.down_offset,
                    layout.down_length,
                ),
                _ => unreachable!(),
            };
            assert_eq!(fields[6], shape);
            assert_eq!(fields[7], orientation);
            assert_eq!(
                fields[8],
                format!("model.layers.{layer}.mlp.experts.{expert}.{role}_proj.weight")
            );
            assert!(fields[9].parse::<usize>().is_ok(), "source shard ID");
            assert!(fields[10].parse::<u64>().is_ok(), "source offset");
            assert_eq!(fields[11].parse::<usize>().unwrap(), 3_145_728);
            assert_eq!(
                fields[12],
                format!("experts/experts-layer-{layer:05}-of-00048.bin")
            );
            assert_eq!(
                fields[13].parse::<usize>().unwrap(),
                expert * layout.total_byte_length
            );
            assert_eq!(fields[14].parse::<usize>().unwrap(), projection_offset);
            assert_eq!(fields[15].parse::<usize>().unwrap(), projection_length);
            assert_eq!(fields[16], aggregation[case_index / 2]);
        }
    }
    assert!(records.next().is_none(), "unexpected structure record");
}

fn assert_stage(
    stage: &str,
    actual: &Tensor,
    expected: &Tensor,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> StageMetrics {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "checkpoint shape mismatch at {stage}"
    );
    let width = actual.shape().dimensions().last().copied().unwrap_or(1);
    let mut maximum_absolute_difference = 0.0_f32;
    let mut maximum_relative_difference = 0.0_f32;
    let mut first_mismatch = None;
    for (index, (&actual_value, &expected_value)) in
        actual.data().iter().zip(expected.data()).enumerate()
    {
        let absolute = (actual_value - expected_value).abs();
        let relative = if expected_value == 0.0 {
            0.0
        } else {
            absolute / expected_value.abs()
        };
        maximum_absolute_difference = maximum_absolute_difference.max(absolute);
        maximum_relative_difference = maximum_relative_difference.max(relative);
        let tolerance = absolute_tolerance + relative_tolerance * expected_value.abs();
        if absolute > tolerance && first_mismatch.is_none() {
            first_mismatch = Some((index, expected_value, actual_value, absolute, tolerance));
        }
    }
    if let Some((index, expected_value, actual_value, absolute, tolerance)) = first_mismatch {
        panic!(
            "first mismatch at checkpoint={stage}, token={}, element={}: expected {expected_value}, got {actual_value}, absolute_error={absolute}, tolerance={tolerance}, checkpoint_max_absolute_error={maximum_absolute_difference}, checkpoint_max_relative_error={maximum_relative_difference}",
            index / width,
            index % width,
        );
    }
    StageMetrics {
        maximum_absolute_difference,
        maximum_relative_difference,
    }
}

fn measure_stage(actual: &Tensor, expected: &Tensor) -> StageMetrics {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "diagnostic checkpoint shape mismatch"
    );
    let mut maximum_absolute_difference = 0.0_f32;
    let mut maximum_relative_difference = 0.0_f32;
    for (&actual_value, &expected_value) in actual.data().iter().zip(expected.data()) {
        let absolute = (actual_value - expected_value).abs();
        let relative = if expected_value == 0.0 {
            0.0
        } else {
            absolute / expected_value.abs()
        };
        maximum_absolute_difference = maximum_absolute_difference.max(absolute);
        maximum_relative_difference = maximum_relative_difference.max(relative);
    }
    StageMetrics {
        maximum_absolute_difference,
        maximum_relative_difference,
    }
}

fn compare_three_paths(
    stage: &str,
    rust: &Tensor,
    bf16: &Tensor,
    f32_control: &Tensor,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> StageMetrics {
    let primary = record_three_paths(stage, rust, bf16, f32_control);
    assert_stage(
        stage,
        rust,
        f32_control,
        absolute_tolerance,
        relative_tolerance,
    );
    primary
}

fn record_three_paths(
    stage: &str,
    rust: &Tensor,
    bf16: &Tensor,
    f32_control: &Tensor,
) -> StageMetrics {
    let width = rust.shape().dimensions().last().copied().unwrap_or(1);
    let per_token = |left: &Tensor, right: &Tensor| -> Vec<f32> {
        left.data()
            .chunks_exact(width)
            .zip(right.data().chunks_exact(width))
            .map(|(left_row, right_row)| {
                left_row
                    .iter()
                    .zip(right_row)
                    .map(|(left_value, right_value)| (left_value - right_value).abs())
                    .fold(0.0_f32, f32::max)
            })
            .collect()
    };
    let bf16_vs_rust = measure_stage(rust, bf16);
    let bf16_vs_f32 = measure_stage(f32_control, bf16);
    let primary = measure_stage(rust, f32_control);
    if env::var_os("COLIBRI_COMPACT_VALIDATION").is_none() {
        println!(
            "three_path_tokens checkpoint={stage} f32_vs_rust={:?} bf16_vs_rust={:?} bf16_vs_f32={:?}",
            per_token(f32_control, rust),
            per_token(bf16, rust),
            per_token(bf16, f32_control),
        );
        println!(
            "three_path checkpoint={stage} f32_vs_rust={primary:?} bf16_vs_rust={bf16_vs_rust:?} bf16_vs_f32={bf16_vs_f32:?}"
        );
    }
    primary
}

#[allow(clippy::too_many_arguments)]
fn record_intermediate_checkpoint(
    evidence: &mut String,
    layer: usize,
    token: usize,
    position: Option<usize>,
    expert: Option<usize>,
    checkpoint: &str,
    actual: &Tensor,
    bf16: &Tensor,
    f32_control: &Tensor,
    budget: Option<f32>,
) -> StageMetrics {
    let metrics = record_three_paths(checkpoint, actual, bf16, f32_control);
    let bf16_metrics = measure_stage(actual, bf16);
    if let Some(budget) = budget {
        assert_stage(checkpoint, actual, f32_control, budget, 0.0);
    }
    writeln!(
        evidence,
        "{layer}\t{token}\t{}\t{}\t{checkpoint}\t{:.17e}\t{}\t{:.17e}\t{:.17e}\t{:.17e}",
        position.map_or_else(|| "NA".to_owned(), |value| value.to_string()),
        expert.map_or_else(|| "NA".to_owned(), |value| value.to_string()),
        metrics.maximum_absolute_difference,
        budget.map_or_else(
            || "CHARACTERIZE".to_owned(),
            |value| format!("{value:.17e}")
        ),
        metrics.maximum_relative_difference,
        bf16_metrics.maximum_absolute_difference,
        bf16_metrics.maximum_relative_difference,
    )
    .expect("write intermediate evidence");
    metrics
}

fn record_generation_checkpoint(
    evidence: &mut String,
    step: usize,
    checkpoint: &str,
    actual: &Tensor,
    bf16: &Tensor,
    f32_control: &Tensor,
    budget: Option<f32>,
) -> StageMetrics {
    let metrics = record_three_paths(checkpoint, actual, bf16, f32_control);
    let bf16_metrics = measure_stage(actual, bf16);
    if let Some(budget) = budget {
        assert_stage(checkpoint, actual, f32_control, budget, 0.0);
    }
    writeln!(
        evidence,
        "checkpoint\t{step}\t{checkpoint}\t{:.17e}\t{}\t{:.17e}\t{:.17e}",
        metrics.maximum_absolute_difference,
        budget.map_or_else(
            || "CHARACTERIZE".to_owned(),
            |value| format!("{value:.17e}")
        ),
        metrics.maximum_relative_difference,
        bf16_metrics.maximum_absolute_difference,
    )
    .expect("write generation checkpoint evidence");
    metrics
}

fn token_selection_classification(
    actual_id: usize,
    reference_id: usize,
    margin: f32,
    maximum_error: f32,
) -> &'static str {
    let safe = margin > 2.0 * maximum_error;
    if actual_id == reference_id && safe {
        "exact_match_safe"
    } else if actual_id != reference_id && safe {
        "true_mismatch"
    } else {
        "numerically_ambiguous"
    }
}

fn atomic_diagnostic(path: &PathBuf, payload: &[u8]) {
    if path.exists() {
        assert_eq!(fs::read(path).expect("read existing diagnostic"), payload);
        return;
    }
    let temporary = path.with_extension("incomplete");
    assert!(!temporary.exists(), "incomplete diagnostic exists");
    let mut output = File::create(&temporary).expect("create diagnostic");
    output.write_all(payload).expect("write diagnostic");
    output.sync_all().expect("sync diagnostic");
    drop(output);
    fs::rename(temporary, path).expect("promote diagnostic");
}

fn f32_bytes(tensor: &Tensor) -> Vec<u8> {
    tensor
        .data()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn export_layer1_moe_diagnostic(
    root: &Path,
    pre_router: &PreRouterOutput,
    moe_output: &Tensor,
    expert_outputs: &HashMap<String, Tensor>,
) {
    assert!(
        root.is_absolute(),
        "Layer-1 diagnostic root must be absolute"
    );
    atomic_diagnostic(
        &root.join("layer1-expert-input-f32.bin"),
        &f32_bytes(&pre_router.post_attention_norm),
    );
    atomic_diagnostic(
        &root.join("layer1-routing-weights-f32.bin"),
        &f32_bytes(&pre_router.router.weights),
    );
    atomic_diagnostic(
        &root.join("layer1-moe-output-f32.bin"),
        &f32_bytes(moe_output),
    );
    let mut plan = String::from("token\tposition\texpert\toffset\tlength\n");
    let mut payload = Vec::with_capacity(32 * 2048 * 4);
    for token in 0..4 {
        for position in 0..8 {
            let expert = pre_router.router.selected_experts[token * 8 + position];
            let name = format!("layer1_expert_output_t{token}_p{position}_e{expert}");
            let output = &expert_outputs[&name];
            writeln!(
                plan,
                "{token}\t{position}\t{expert}\t{}\t{}",
                payload.len(),
                output.data().len() * 4,
            )
            .expect("write Layer-1 diagnostic plan");
            payload.extend(f32_bytes(output));
        }
    }
    atomic_diagnostic(
        &root.join("layer1-expert-output-plan-v1.tsv"),
        plan.as_bytes(),
    );
    atomic_diagnostic(&root.join("layer1-expert-outputs-f32.bin"), &payload);
}

fn export_rms_diagnostics(residual: &Tensor, post_norm: &Tensor, epsilon: f32) {
    let root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_RMS_DIAGNOSTIC_ROOT must be set for the focused diagnostic");
    assert!(root.is_absolute(), "diagnostic root must be absolute");
    let encode = |tensor: &Tensor| -> Vec<u8> {
        tensor
            .data()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    atomic_diagnostic(
        &root.join("m4.2-02-rust-layer0-residual-f32.bin"),
        &encode(residual),
    );
    atomic_diagnostic(
        &root.join("m4.2-02-rust-layer0-postnorm-f32.bin"),
        &encode(post_norm),
    );

    let mut rows = String::from(
        "token\tordered_sum_of_squares\tmean_square\tepsilon\tmean_plus_epsilon\tsquare_root\treciprocal_rms\tsum_bits\tmean_bits\tplus_bits\tsqrt_bits\treciprocal_bits\n",
    );
    for (token, row) in residual.data().chunks_exact(2048).enumerate() {
        let sum: f32 = row.iter().map(|value| value * value).sum();
        let mean = sum / 2048.0;
        let plus = mean + epsilon;
        let square_root = plus.sqrt();
        let reciprocal = square_root.recip();
        writeln!(
            &mut rows,
            "{token}\t{sum:.17e}\t{mean:.17e}\t{epsilon:.17e}\t{plus:.17e}\t{square_root:.17e}\t{reciprocal:.17e}\t{:08x}\t{:08x}\t{:08x}\t{:08x}\t{:08x}",
            sum.to_bits(),
            mean.to_bits(),
            plus.to_bits(),
            square_root.to_bits(),
            reciprocal.to_bits(),
        )
        .expect("write row diagnostic");
    }
    atomic_diagnostic(
        &root.join("m4.2-02-rust-rms-row-diagnostics-v1.tsv"),
        rows.as_bytes(),
    );
}

fn deterministic_top_k(logits: &[f32], top_k: usize) -> Vec<usize> {
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    ids.truncate(top_k);
    ids
}

fn comma_separated<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_router_boundaries(
    output: &PreRouterOutput,
    bf16_logits: &Tensor,
    f32_logits: &Tensor,
    bf16_ids: &[usize],
    f32_ids: &[usize],
    evidence_file: &str,
) -> Vec<&'static str> {
    let expert_count = 128;
    let top_k = 8;
    let mut assertable = Vec::with_capacity(4);
    let mut evidence = String::from(
        "token\tf32_classification\tbf16_classification\tf32_vs_rust_max_error\tbf16_vs_rust_max_error\tbf16_vs_f32_max_error\tf32_required_margin\tbf16_required_margin\tf32_kth\tf32_highest_unselected\tf32_margin\tbf16_kth\tbf16_highest_unselected\tbf16_margin\ttransformers_f32_ids\ttransformers_bf16_ids\trust_ids\ttransformers_f32_selected_logits\ttransformers_bf16_selected_logits\trust_selected_logits\n",
    );
    for token in 0..4 {
        let bf16_row = &bf16_logits.data()[token * expert_count..(token + 1) * expert_count];
        let f32_row = &f32_logits.data()[token * expert_count..(token + 1) * expert_count];
        let actual_row =
            &output.router.logits.data()[token * expert_count..(token + 1) * expert_count];
        let bf16_selected = &bf16_ids[token * top_k..(token + 1) * top_k];
        let f32_selected = &f32_ids[token * top_k..(token + 1) * top_k];
        let actual_selected = &output.router.selected_experts[token * top_k..(token + 1) * top_k];
        assert_eq!(
            actual_selected,
            deterministic_top_k(actual_row, top_k),
            "Rust deterministic router policy failed at token {token}"
        );
        let selected_set: HashSet<_> = f32_selected.iter().copied().collect();
        let bf16_selected_set: HashSet<_> = bf16_selected.iter().copied().collect();
        let f32_selected_logits: Vec<_> = f32_selected.iter().map(|id| f32_row[*id]).collect();
        let bf16_selected_logits: Vec<_> = bf16_selected.iter().map(|id| bf16_row[*id]).collect();
        let rust_selected_logits: Vec<_> =
            actual_selected.iter().map(|id| actual_row[*id]).collect();
        let kth = f32_selected_logits
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let highest_unselected = f32_row
            .iter()
            .enumerate()
            .filter(|(id, _)| !selected_set.contains(id))
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let margin = kth - highest_unselected;
        let bf16_kth = bf16_selected_logits
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let bf16_highest_unselected = bf16_row
            .iter()
            .enumerate()
            .filter(|(id, _)| !bf16_selected_set.contains(id))
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let bf16_margin = bf16_kth - bf16_highest_unselected;
        let f32_vs_rust_maximum_error = f32_row
            .iter()
            .zip(actual_row)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        let bf16_vs_rust_maximum_error = bf16_row
            .iter()
            .zip(actual_row)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        let bf16_vs_f32_maximum_error = bf16_row
            .iter()
            .zip(f32_row)
            .map(|(bf16, f32)| (bf16 - f32).abs())
            .fold(0.0_f32, f32::max);
        let required_margin = 2.0 * f32_vs_rust_maximum_error;
        let bf16_required_margin = 2.0 * bf16_vs_rust_maximum_error;
        let ids_assertable = margin.is_finite() && margin > 0.0 && margin > required_margin;
        let bf16_ids_assertable =
            bf16_margin.is_finite() && bf16_margin > 0.0 && bf16_margin > bf16_required_margin;
        if ids_assertable {
            assert_eq!(
                actual_selected, f32_selected,
                "safe-margin expert IDs differ at token {token}: margin={margin}, max_error={f32_vs_rust_maximum_error}"
            );
        }
        if bf16_ids_assertable {
            assert_eq!(
                actual_selected, bf16_selected,
                "safe-margin BF16 expert IDs differ at token {token}: margin={bf16_margin}, max_error={bf16_vs_rust_maximum_error}"
            );
        }
        let classification = if ids_assertable {
            "exact_match_safe"
        } else {
            "numerically_ambiguous"
        };
        let bf16_classification = if bf16_ids_assertable {
            "exact_match_safe"
        } else {
            "numerically_ambiguous"
        };
        println!(
            "router_boundary token={token} classification={classification} bf16_classification={bf16_classification} ids_assertable={ids_assertable} transformers_bf16_ids={bf16_selected:?} transformers_f32_ids={f32_selected:?} rust_ids={actual_selected:?} transformers_bf16_selected_logits={bf16_selected_logits:?} transformers_f32_selected_logits={f32_selected_logits:?} rust_selected_logits={rust_selected_logits:?} kth={kth} highest_unselected={highest_unselected} margin={margin} bf16_kth={bf16_kth} bf16_highest_unselected={bf16_highest_unselected} bf16_margin={bf16_margin} f32_vs_rust_max_logit_error={f32_vs_rust_maximum_error} bf16_vs_rust_max_logit_error={bf16_vs_rust_maximum_error} bf16_vs_f32_max_logit_error={bf16_vs_f32_maximum_error} required_margin={required_margin} bf16_required_margin={bf16_required_margin}"
        );
        writeln!(
            evidence,
            "{token}\t{classification}\t{bf16_classification}\t{f32_vs_rust_maximum_error:.17e}\t{bf16_vs_rust_maximum_error:.17e}\t{bf16_vs_f32_maximum_error:.17e}\t{required_margin:.17e}\t{bf16_required_margin:.17e}\t{kth:.17e}\t{highest_unselected:.17e}\t{margin:.17e}\t{bf16_kth:.17e}\t{bf16_highest_unselected:.17e}\t{bf16_margin:.17e}\t{}\t{}\t{}\t{}\t{}\t{}",
            comma_separated(f32_selected),
            comma_separated(bf16_selected),
            comma_separated(actual_selected),
            comma_separated(&f32_selected_logits),
            comma_separated(&bf16_selected_logits),
            comma_separated(&rust_selected_logits),
        )
        .expect("write router evidence");
        assertable.push(classification);
    }
    let root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_RMS_DIAGNOSTIC_ROOT must be set for router evidence");
    atomic_diagnostic(&root.join(evidence_file), evidence.as_bytes());
    assertable
}

#[test]
fn pinned_layer_zero_pre_router_matches_transformers() {
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let plan = runtime_plan(RUNTIME_PLAN);
    let payload_path = artifact_root.join(&plan.payload);
    let mut payload = File::open(&payload_path).expect("open canonical dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length,
        "dense payload length"
    );
    let bf16_checkpoints = checkpoint_plan(BF16_CHECKPOINT_PLAN);
    let f32_checkpoints = checkpoint_plan(F32_CHECKPOINT_PLAN);
    assert_eq!(
        checkpoint_ids(BF16_CHECKPOINTS, &bf16_checkpoints, "input_ids"),
        INPUT_IDS
    );
    assert_eq!(
        checkpoint_ids(F32_CHECKPOINTS, &f32_checkpoints, "input_ids"),
        INPUT_IDS
    );
    assert_eq!(
        checkpoint_ids(BF16_CHECKPOINTS, &bf16_checkpoints, "position_ids"),
        POSITION_IDS
    );
    assert_eq!(
        checkpoint_ids(F32_CHECKPOINTS, &f32_checkpoints, "position_ids"),
        POSITION_IDS
    );

    let mask = checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "attention_mask");
    assert_eq!(
        mask,
        checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "attention_mask"),
        "control attention mask differs"
    );
    assert_eq!(mask.shape().dimensions(), [1, 1, 4, 4]);
    for query in 0..4 {
        for key in 0..4 {
            let value = mask.data()[query * 4 + key];
            if key <= query {
                assert_eq!(value, 0.0, "causal mask allowed value");
            } else {
                assert_eq!(value, -3.389_531_4e38, "causal mask blocked value");
            }
        }
    }

    let mut artifact_bytes_read = 0_u64;
    let embedding = embedding_rows(&mut payload, &plan, &mut artifact_bytes_read);
    let input_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.input_layernorm.weight",
        &mut artifact_bytes_read,
    );
    let query = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.q_proj.weight",
        &mut artifact_bytes_read,
    );
    let key = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.k_proj.weight",
        &mut artifact_bytes_read,
    );
    let value = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.v_proj.weight",
        &mut artifact_bytes_read,
    );
    let output = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.o_proj.weight",
        &mut artifact_bytes_read,
    );
    let query_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.q_norm.weight",
        &mut artifact_bytes_read,
    );
    let key_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.self_attn.k_norm.weight",
        &mut artifact_bytes_read,
    );
    let post_norm = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.post_attention_layernorm.weight",
        &mut artifact_bytes_read,
    );
    let router = artifact_tensor(
        &mut payload,
        &plan,
        "model.layers.0.mlp.gate.weight",
        &mut artifact_bytes_read,
    );

    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let original_embedding = embedding.clone();
    let started = Instant::now();
    let pre_router = pre_router_with_weights(
        embedding.view(),
        input_norm.view(),
        query.view(),
        key.view(),
        value.view(),
        output.view(),
        query_norm.view(),
        key_norm.view(),
        post_norm.view(),
        router.view(),
        config,
    )
    .expect("layer-0 pre-router execution");
    let execution_seconds = started.elapsed().as_secs_f64();
    assert_eq!(
        embedding, original_embedding,
        "pre-router execution mutated its input"
    );
    export_rms_diagnostics(
        &pre_router.residual_output,
        &pre_router.post_attention_norm,
        config.rms_norm_epsilon(),
    );
    println!(
        "layer0_execution artifact_bytes_read={artifact_bytes_read} execution_seconds={execution_seconds}"
    );

    let embedding_metrics = compare_three_paths(
        "embedding_output",
        &embedding,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "embedding_output"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "embedding_output"),
        0.0,
        0.0,
    );
    let input_norm_metrics = compare_three_paths(
        "input_rmsnorm",
        &pre_router.input_norm,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "input_rmsnorm"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "input_rmsnorm"),
        1.0e-6,
        1.0e-5,
    );
    let attention_metrics = compare_three_paths(
        "attention_output",
        &pre_router.attention_output,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "attention_output"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "attention_output"),
        1.0e-6,
        1.0e-5,
    );
    let residual_metrics = compare_three_paths(
        "residual_output",
        &pre_router.residual_output,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "residual_output"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "residual_output"),
        1.0e-6,
        1.0e-5,
    );
    let bf16_post_norm = checkpoint_f32(
        BF16_CHECKPOINTS,
        &bf16_checkpoints,
        "post_attention_rmsnorm",
    );
    let f32_post_norm = checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "post_attention_rmsnorm");
    let post_norm_metrics = compare_three_paths(
        "post_attention_rmsnorm",
        &pre_router.post_attention_norm,
        &bf16_post_norm,
        &f32_post_norm,
        POST_NORM_PROPAGATED_ABSOLUTE_BUDGET,
        0.0,
    );
    let bf16_logits = checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "router_logits");
    let reference_logits = checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "router_logits");
    let isolated_router = route_tokens(f32_post_norm.view(), router.view(), config)
        .expect("same-input F32 router execution");
    let isolated_router_metrics = measure_stage(&isolated_router.logits, &reference_logits);
    println!("isolated_router_same_input metrics={isolated_router_metrics:?}");
    let router_metrics = record_three_paths(
        "router_logits",
        &pre_router.router.logits,
        &bf16_logits,
        &reference_logits,
    );
    let bf16_ids = checkpoint_ids(BF16_CHECKPOINTS, &bf16_checkpoints, "selected_expert_ids");
    let f32_ids = checkpoint_ids(F32_CHECKPOINTS, &f32_checkpoints, "selected_expert_ids");
    let classifications = validate_router_boundaries(
        &pre_router,
        &bf16_logits,
        &reference_logits,
        &bf16_ids,
        &f32_ids,
        "m4.2-02-rust-layer0-router-evidence-v1.tsv",
    );
    let routing_metrics = record_three_paths(
        "routing_weights",
        &pre_router.router.weights,
        &checkpoint_f32(BF16_CHECKPOINTS, &bf16_checkpoints, "routing_weights"),
        &checkpoint_f32(F32_CHECKPOINTS, &f32_checkpoints, "routing_weights"),
    );

    let explicit_weight_bytes = input_norm.data().len()
        + query.data().len()
        + key.data().len()
        + value.data().len()
        + output.data().len()
        + query_norm.data().len()
        + key_norm.data().len()
        + post_norm.data().len()
        + router.data().len();
    let explicit_runtime_bytes = (explicit_weight_bytes
        + embedding.data().len()
        + pre_router.input_norm.data().len()
        + pre_router.attention_output.data().len()
        + pre_router.residual_output.data().len()
        + pre_router.post_attention_norm.data().len()
        + pre_router.router.logits.data().len()
        + pre_router.router.weights.data().len())
        * 4;
    println!(
        "layer0_metrics artifact_bytes_read={artifact_bytes_read} peak_explicit_runtime_bytes={explicit_runtime_bytes} execution_seconds={execution_seconds} classifications={classifications:?} isolated_router_max_abs={} isolated_router_max_rel={} checkpoint_max_abs=[embedding:{},input_norm:{},attention:{},residual:{},post_norm:{},router:{},routing:{}] checkpoint_max_rel=[embedding:{},input_norm:{},attention:{},residual:{},post_norm:{},router:{},routing:{}]",
        isolated_router_metrics.maximum_absolute_difference,
        isolated_router_metrics.maximum_relative_difference,
        embedding_metrics.maximum_absolute_difference,
        input_norm_metrics.maximum_absolute_difference,
        attention_metrics.maximum_absolute_difference,
        residual_metrics.maximum_absolute_difference,
        post_norm_metrics.maximum_absolute_difference,
        router_metrics.maximum_absolute_difference,
        routing_metrics.maximum_absolute_difference,
        embedding_metrics.maximum_relative_difference,
        input_norm_metrics.maximum_relative_difference,
        attention_metrics.maximum_relative_difference,
        residual_metrics.maximum_relative_difference,
        post_norm_metrics.maximum_relative_difference,
        router_metrics.maximum_relative_difference,
        routing_metrics.maximum_relative_difference,
    );
}

#[test]
fn pinned_layer_one_router_uses_genuine_streaming_layer_zero_output() {
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let plan = runtime_plan(LAYER1_RUNTIME_PLAN);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length,
        "dense payload length"
    );
    let bf16_plan = checkpoint_plan(LAYER1_BF16_CHECKPOINT_PLAN);
    let f32_plan = checkpoint_plan(LAYER1_F32_CHECKPOINT_PLAN);
    assert_eq!(
        checkpoint_ids(LAYER1_F32_CHECKPOINTS, &f32_plan, "input_ids"),
        INPUT_IDS
    );
    assert_eq!(
        checkpoint_ids(LAYER1_F32_CHECKPOINTS, &f32_plan, "position_ids"),
        POSITION_IDS
    );

    let mut dense_bytes_read = 0_u64;
    let embedding = embedding_rows(&mut payload, &plan, &mut dense_bytes_read);
    let layer0 = layer_weights(&mut payload, &plan, 0, &mut dense_bytes_read);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let started = Instant::now();
    let layer0_pre_router = pre_router_with_weights(
        embedding.view(),
        layer0.input_norm.view(),
        layer0.query.view(),
        layer0.key.view(),
        layer0.value.view(),
        layer0.output.view(),
        layer0.query_norm.view(),
        layer0.key_norm.view(),
        layer0.post_norm.view(),
        layer0.router.view(),
        config,
    )
    .expect("Layer-0 pre-router execution");

    let bf16_layer0_ids = checkpoint_ids(
        LAYER1_BF16_CHECKPOINTS,
        &bf16_plan,
        "layer0_selected_expert_ids",
    );
    let f32_layer0_ids = checkpoint_ids(
        LAYER1_F32_CHECKPOINTS,
        &f32_plan,
        "layer0_selected_expert_ids",
    );
    assert_eq!(bf16_layer0_ids, f32_layer0_ids, "Layer-0 reference IDs");
    assert_eq!(
        layer0_pre_router.router.selected_experts, f32_layer0_ids,
        "approved Layer-0 Rust expert IDs changed"
    );
    compare_three_paths(
        "layer0_expert_input",
        &layer0_pre_router.post_attention_norm,
        &checkpoint_f32(LAYER1_BF16_CHECKPOINTS, &bf16_plan, "layer0_expert_input"),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer0_expert_input"),
        POST_NORM_PROPAGATED_ABSOLUTE_BUDGET,
        0.0,
    );
    compare_three_paths(
        "layer0_routing_weights",
        &layer0_pre_router.router.weights,
        &checkpoint_f32(
            LAYER1_BF16_CHECKPOINTS,
            &bf16_plan,
            "layer0_routing_weights",
        ),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer0_routing_weights"),
        1.0e-6,
        1.0e-5,
    );

    let mut store = selected_expert_store(&artifact_root);
    let mut observed_expert_outputs = HashMap::new();
    let expert_layout = PackedExpertLayout::for_config(config);
    let layer0_moe = streaming_routed_experts_with_observer(
        layer0_pre_router.post_attention_norm.view(),
        &layer0_pre_router.router,
        config,
        0,
        &mut store,
        expert_layout,
        |expert, token, position, values| {
            let name = format!("layer0_expert_output_t{token}_p{position}_e{expert}");
            let tensor = Tensor::new(TensorShape::new([2048]), values.to_vec())
                .expect("observed expert output");
            assert!(
                observed_expert_outputs.insert(name, tensor).is_none(),
                "duplicate observed expert output"
            );
        },
    )
    .expect("stream selected Layer-0 experts");
    assert_eq!(observed_expert_outputs.len(), 32, "expert occurrence count");

    let mut expert_evidence = String::from(
        "checkpoint\tmaximum_f32_vs_rust_absolute_error\tmaximum_f32_vs_rust_relative_error\n",
    );
    for token in 0..4 {
        for position in 0..8 {
            let expert = layer0_pre_router.router.selected_experts[token * 8 + position];
            let name = format!("layer0_expert_output_t{token}_p{position}_e{expert}");
            let metrics = compare_three_paths(
                &name,
                &observed_expert_outputs[&name],
                &checkpoint_f32(LAYER1_BF16_CHECKPOINTS, &bf16_plan, &name),
                &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, &name),
                1.0e-6,
                1.0e-5,
            );
            writeln!(
                expert_evidence,
                "{name}\t{:.17e}\t{:.17e}",
                metrics.maximum_absolute_difference, metrics.maximum_relative_difference,
            )
            .expect("write selected expert evidence");
        }
    }

    let layer0_moe_metrics = compare_three_paths(
        "layer0_moe_output",
        &layer0_moe,
        &checkpoint_f32(LAYER1_BF16_CHECKPOINTS, &bf16_plan, "layer0_moe_output"),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer0_moe_output"),
        1.0e-6,
        1.0e-5,
    );
    let layer0_block = elementwise_add(layer0_pre_router.residual_output.view(), layer0_moe.view())
        .expect("Layer-0 final residual");
    let layer0_block_metrics = compare_three_paths(
        "layer0_block_output",
        &layer0_block,
        &checkpoint_f32(LAYER1_BF16_CHECKPOINTS, &bf16_plan, "layer0_block_output"),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer0_block_output"),
        1.0e-6,
        1.0e-5,
    );
    let cache_metrics = store.metrics();
    writeln!(
        expert_evidence,
        "layer0_moe_output\t{:.17e}\t{:.17e}",
        layer0_moe_metrics.maximum_absolute_difference,
        layer0_moe_metrics.maximum_relative_difference,
    )
    .expect("write Layer-0 MoE evidence");
    writeln!(
        expert_evidence,
        "layer0_block_output\t{:.17e}\t{:.17e}",
        layer0_block_metrics.maximum_absolute_difference,
        layer0_block_metrics.maximum_relative_difference,
    )
    .expect("write Layer-0 block evidence");
    writeln!(
        expert_evidence,
        "cache\thits={}\tmisses={}\tloads={}\tevictions={}\tresident_bytes={}\tpeak_resident_bytes={}\tbytes_read={}",
        cache_metrics.hits,
        cache_metrics.misses,
        cache_metrics.loads,
        cache_metrics.evictions,
        cache_metrics.resident_bytes,
        cache_metrics.peak_resident_bytes,
        cache_metrics.bytes_read,
    )
    .expect("write Layer-0 cache evidence");
    let diagnostic_root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("diagnostic root for Layer-0 expert evidence");
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-02-rust-layer0-expert-evidence-v1.tsv"),
        expert_evidence.as_bytes(),
    );

    let layer1 = layer_weights(&mut payload, &plan, 1, &mut dense_bytes_read);
    let layer1_pre_router = pre_router_with_weights(
        layer0_block.view(),
        layer1.input_norm.view(),
        layer1.query.view(),
        layer1.key.view(),
        layer1.value.view(),
        layer1.output.view(),
        layer1.query_norm.view(),
        layer1.key_norm.view(),
        layer1.post_norm.view(),
        layer1.router.view(),
        config,
    )
    .expect("Layer-1 pre-router execution");
    let layer1_input_metrics = compare_three_paths(
        "layer1_input",
        &layer0_block,
        &checkpoint_f32(LAYER1_BF16_CHECKPOINTS, &bf16_plan, "layer1_input"),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer1_input"),
        1.0e-6,
        1.0e-5,
    );
    assert_eq!(
        layer1_input_metrics.maximum_absolute_difference, LAYER1_INPUT_MAXIMUM_ERROR,
        "frozen genuine Layer-1 incoming error changed"
    );
    let layer1_input_norm_metrics = compare_three_paths(
        "layer1_input_rmsnorm",
        &layer1_pre_router.input_norm,
        &checkpoint_f32(LAYER1_BF16_CHECKPOINTS, &bf16_plan, "layer1_input_rmsnorm"),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer1_input_rmsnorm"),
        LAYER1_INPUT_NORM_BUDGET,
        0.0,
    );
    let layer1_attention_metrics = compare_three_paths(
        "layer1_attention_output",
        &layer1_pre_router.attention_output,
        &checkpoint_f32(
            LAYER1_BF16_CHECKPOINTS,
            &bf16_plan,
            "layer1_attention_output",
        ),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer1_attention_output"),
        LAYER1_ATTENTION_BUDGET,
        0.0,
    );
    let layer1_residual_metrics = compare_three_paths(
        "layer1_residual_output",
        &layer1_pre_router.residual_output,
        &checkpoint_f32(
            LAYER1_BF16_CHECKPOINTS,
            &bf16_plan,
            "layer1_residual_output",
        ),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer1_residual_output"),
        LAYER1_RESIDUAL_BUDGET,
        0.0,
    );
    let layer1_post_norm_metrics = compare_three_paths(
        "layer1_post_attention_rmsnorm",
        &layer1_pre_router.post_attention_norm,
        &checkpoint_f32(
            LAYER1_BF16_CHECKPOINTS,
            &bf16_plan,
            "layer1_post_attention_rmsnorm",
        ),
        &checkpoint_f32(
            LAYER1_F32_CHECKPOINTS,
            &f32_plan,
            "layer1_post_attention_rmsnorm",
        ),
        LAYER1_POST_NORM_BUDGET,
        0.0,
    );
    let bf16_layer1_logits =
        checkpoint_f32(LAYER1_BF16_CHECKPOINTS, &bf16_plan, "layer1_router_logits");
    let f32_layer1_logits =
        checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer1_router_logits");
    let layer1_router_metrics = compare_three_paths(
        "layer1_router_logits",
        &layer1_pre_router.router.logits,
        &bf16_layer1_logits,
        &f32_layer1_logits,
        LAYER1_ROUTER_LOGIT_BUDGET,
        0.0,
    );
    let bf16_layer1_ids = checkpoint_ids(
        LAYER1_BF16_CHECKPOINTS,
        &bf16_plan,
        "layer1_selected_expert_ids",
    );
    let f32_layer1_ids = checkpoint_ids(
        LAYER1_F32_CHECKPOINTS,
        &f32_plan,
        "layer1_selected_expert_ids",
    );
    let classifications = validate_router_boundaries(
        &layer1_pre_router,
        &bf16_layer1_logits,
        &f32_layer1_logits,
        &bf16_layer1_ids,
        &f32_layer1_ids,
        "m4.2-02-rust-layer1-router-evidence-v1.tsv",
    );
    let layer1_routing_metrics = compare_three_paths(
        "layer1_routing_weights",
        &layer1_pre_router.router.weights,
        &checkpoint_f32(
            LAYER1_BF16_CHECKPOINTS,
            &bf16_plan,
            "layer1_routing_weights",
        ),
        &checkpoint_f32(LAYER1_F32_CHECKPOINTS, &f32_plan, "layer1_routing_weights"),
        LAYER1_ROUTING_WEIGHT_BUDGET,
        0.0,
    );

    let mut checkpoint_evidence = String::from(
        "checkpoint\tmaximum_f32_vs_rust_absolute_error\tabsolute_budget\tmaximum_f32_vs_rust_relative_error\n",
    );
    for (name, metrics, budget) in [
        (
            "layer1_input",
            layer1_input_metrics,
            LAYER1_INPUT_MAXIMUM_ERROR,
        ),
        (
            "layer1_input_rmsnorm",
            layer1_input_norm_metrics,
            LAYER1_INPUT_NORM_BUDGET,
        ),
        (
            "layer1_attention_output",
            layer1_attention_metrics,
            LAYER1_ATTENTION_BUDGET,
        ),
        (
            "layer1_residual_output",
            layer1_residual_metrics,
            LAYER1_RESIDUAL_BUDGET,
        ),
        (
            "layer1_post_attention_rmsnorm",
            layer1_post_norm_metrics,
            LAYER1_POST_NORM_BUDGET,
        ),
        (
            "layer1_router_logits",
            layer1_router_metrics,
            LAYER1_ROUTER_LOGIT_BUDGET,
        ),
        (
            "layer1_routing_weights",
            layer1_routing_metrics,
            LAYER1_ROUTING_WEIGHT_BUDGET,
        ),
    ] {
        writeln!(
            checkpoint_evidence,
            "{name}\t{:.17e}\t{budget:.17e}\t{:.17e}",
            metrics.maximum_absolute_difference, metrics.maximum_relative_difference,
        )
        .expect("write Layer-1 checkpoint evidence");
    }
    let runtime_elements = embedding.data().len()
        + layer0_pre_router.input_norm.data().len()
        + layer0_pre_router.attention_output.data().len()
        + layer0_pre_router.residual_output.data().len()
        + layer0_pre_router.post_attention_norm.data().len()
        + layer0_pre_router.router.logits.data().len()
        + layer0_pre_router.router.weights.data().len()
        + observed_expert_outputs
            .values()
            .map(|tensor| tensor.data().len())
            .sum::<usize>()
        + layer0_moe.data().len()
        + layer0_block.data().len()
        + layer1_pre_router.input_norm.data().len()
        + layer1_pre_router.attention_output.data().len()
        + layer1_pre_router.residual_output.data().len()
        + layer1_pre_router.post_attention_norm.data().len()
        + layer1_pre_router.router.logits.data().len()
        + layer1_pre_router.router.weights.data().len()
        + bf16_layer1_logits.data().len()
        + f32_layer1_logits.data().len();
    let peak_explicit_bytes = usize::try_from(dense_bytes_read).expect("dense bytes fit usize")
        + runtime_elements * 4
        + cache_metrics.peak_resident_bytes
        + expert_layout.total_byte_length;
    writeln!(
        checkpoint_evidence,
        "resources\tdense_bytes_read={dense_bytes_read}\texpert_bytes_read={}\texpert_peak_resident_bytes={}\tpeak_explicit_bytes={peak_explicit_bytes}",
        cache_metrics.bytes_read, cache_metrics.peak_resident_bytes,
    )
    .expect("write Layer-1 resource evidence");
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-02-rust-layer1-checkpoint-evidence-v1.tsv"),
        checkpoint_evidence.as_bytes(),
    );
    println!(
        "layer1_end_to_end dense_bytes_read={dense_bytes_read} expert_metrics={cache_metrics:?} peak_explicit_bytes={peak_explicit_bytes} elapsed_seconds={} layer0_moe={layer0_moe_metrics:?} layer0_block={layer0_block_metrics:?} layer1_input={layer1_input_metrics:?} layer1_input_norm={layer1_input_norm_metrics:?} layer1_attention={layer1_attention_metrics:?} layer1_residual={layer1_residual_metrics:?} layer1_post_norm={layer1_post_norm_metrics:?} layer1_router={layer1_router_metrics:?} layer1_routing={layer1_routing_metrics:?} classifications={classifications:?}",
        started.elapsed().as_secs_f64(),
    );
}

#[test]
fn pinned_layer_twenty_four_router_uses_genuine_streaming_prefix() {
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let diagnostic_root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("diagnostic root for Layer-24 evidence");
    assert!(
        diagnostic_root.is_absolute(),
        "diagnostic root must be absolute"
    );
    let plan = runtime_plan(LAYER24_RUNTIME_PLAN);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length,
        "dense payload length"
    );
    let bf16_plan = checkpoint_plan(LAYER24_BF16_CHECKPOINT_PLAN);
    let f32_plan = checkpoint_plan(LAYER24_F32_CHECKPOINT_PLAN);
    assert_eq!(
        checkpoint_ids(LAYER24_F32_CHECKPOINTS, &f32_plan, "input_ids"),
        INPUT_IDS
    );
    assert_eq!(
        checkpoint_ids(LAYER24_F32_CHECKPOINTS, &f32_plan, "position_ids"),
        POSITION_IDS
    );

    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut store = expert_store_from_plan(LAYER24_EXPERT_RUNTIME_PLAN, &artifact_root, 24 * 128);
    let mut dense_bytes_read = 0_u64;
    let mut current = embedding_rows(&mut payload, &plan, &mut dense_bytes_read);
    let embedding_metrics = record_three_paths(
        "layer24_embedding_output",
        &current,
        &checkpoint_f32(LAYER24_BF16_CHECKPOINTS, &bf16_plan, "embedding_output"),
        &checkpoint_f32(LAYER24_F32_CHECKPOINTS, &f32_plan, "embedding_output"),
    );
    assert_eq!(
        embedding_metrics.maximum_absolute_difference, 0.0,
        "embedding must remain exact"
    );

    let mut layer_evidence = String::from(
        "layer\tinput_max_abs\tinput_budget\trouter_max_abs\trouter_budget\trouting_max_abs\trouting_budget\tunique_experts\texpert_occurrences\tmoe_max_abs\tmoe_budget\tblock_max_abs\tblock_budget\ttransformers_f32_ids\ttransformers_bf16_ids\trust_ids\n",
    );
    let mut checkpoint_evidence = String::from(
        "layer\tcheckpoint\tmaximum_f32_vs_rust_absolute_error\tabsolute_budget\tmaximum_f32_vs_rust_relative_error\n",
    );
    let mut total_unique_experts = 0_usize;
    let mut total_expert_occurrences = 0_usize;
    let mut maximum_dense_layer_bytes = 0_u64;
    let mut maximum_runtime_elements = current.data().len();
    let started = Instant::now();
    let mut layer24_classifications = Vec::new();

    for layer in 0..=24 {
        let layer_started = Instant::now();
        let dense_before = dense_bytes_read;
        let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
        maximum_dense_layer_bytes = maximum_dense_layer_bytes.max(dense_bytes_read - dense_before);
        let pre_router = pre_router_with_weights(
            current.view(),
            weights.input_norm.view(),
            weights.query.view(),
            weights.key.view(),
            weights.value.view(),
            weights.output.view(),
            weights.query_norm.view(),
            weights.key_norm.view(),
            weights.post_norm.view(),
            weights.router.view(),
            config,
        )
        .unwrap_or_else(|error| panic!("Layer-{layer} pre-router execution failed: {error}"));
        drop(weights);
        let prefix = format!("layer{layer}");
        let propagated_budgets = layer24_propagated_budgets(layer);
        let bf16_input = checkpoint_f32(
            LAYER24_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_input"),
        );
        let f32_input = checkpoint_f32(
            LAYER24_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_input"),
        );
        let input_metrics = record_three_paths(
            &format!("{prefix}_input"),
            &current,
            &bf16_input,
            &f32_input,
        );
        let input_budget = propagated_budgets.input;
        assert_stage(
            &format!("{prefix}_input"),
            &current,
            &f32_input,
            input_budget,
            0.0,
        );
        let bf16_logits = checkpoint_f32(
            LAYER24_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_router_logits"),
        );
        let f32_logits = checkpoint_f32(
            LAYER24_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_router_logits"),
        );
        let router_metrics = record_three_paths(
            &format!("{prefix}_router_logits"),
            &pre_router.router.logits,
            &bf16_logits,
            &f32_logits,
        );
        assert_stage(
            &format!("{prefix}_router_logits"),
            &pre_router.router.logits,
            &f32_logits,
            propagated_budgets.router,
            0.0,
        );
        let bf16_routing = checkpoint_f32(
            LAYER24_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_routing_weights"),
        );
        let f32_routing = checkpoint_f32(
            LAYER24_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_routing_weights"),
        );
        let routing_metrics = record_three_paths(
            &format!("{prefix}_routing_weights"),
            &pre_router.router.weights,
            &bf16_routing,
            &f32_routing,
        );
        assert_stage(
            &format!("{prefix}_routing_weights"),
            &pre_router.router.weights,
            &f32_routing,
            propagated_budgets.routing,
            0.0,
        );
        let bf16_ids = checkpoint_ids(
            LAYER24_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_selected_expert_ids"),
        );
        let f32_ids = checkpoint_ids(
            LAYER24_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_selected_expert_ids"),
        );
        assert_eq!(
            pre_router.router.selected_experts, f32_ids,
            "Layer-{layer} Rust and Transformers F32 expert IDs differ"
        );

        for (suffix, actual) in [
            ("input_rmsnorm", &pre_router.input_norm),
            ("attention_output", &pre_router.attention_output),
            ("residual_output", &pre_router.residual_output),
            ("post_attention_rmsnorm", &pre_router.post_attention_norm),
        ] {
            let name = format!("{prefix}_{suffix}");
            let bf16_reference = checkpoint_f32(LAYER24_BF16_CHECKPOINTS, &bf16_plan, &name);
            let f32_reference = checkpoint_f32(LAYER24_F32_CHECKPOINTS, &f32_plan, &name);
            let metrics = record_three_paths(&name, actual, &bf16_reference, &f32_reference);
            let budget = match suffix {
                "input_rmsnorm" => propagated_budgets.input_norm,
                "attention_output" => propagated_budgets.attention,
                "residual_output" => propagated_budgets.residual,
                "post_attention_rmsnorm" => propagated_budgets.post_norm,
                _ => unreachable!("known pre-router checkpoint"),
            };
            assert_stage(&name, actual, &f32_reference, budget, 0.0);
            writeln!(
                checkpoint_evidence,
                "{layer}\t{suffix}\t{:.17e}\t{budget:.17e}\t{:.17e}",
                metrics.maximum_absolute_difference, metrics.maximum_relative_difference,
            )
            .expect("write checkpoint evidence");
        }

        maximum_runtime_elements = maximum_runtime_elements.max(
            current.data().len()
                + pre_router.input_norm.data().len()
                + pre_router.attention_output.data().len()
                + pre_router.residual_output.data().len()
                + pre_router.post_attention_norm.data().len()
                + pre_router.router.logits.data().len()
                + pre_router.router.weights.data().len()
                + bf16_logits.data().len()
                + f32_logits.data().len(),
        );
        if layer == 24 {
            layer24_classifications = validate_router_boundaries(
                &pre_router,
                &bf16_logits,
                &f32_logits,
                &bf16_ids,
                &f32_ids,
                "m4.2-02-rust-layer24-router-evidence-v1.tsv",
            );
            writeln!(
                layer_evidence,
                "{layer}\t{:.17e}\t{input_budget:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t0\t0\tNA\tNA\tNA\tNA\t{}\t{}\t{}",
                input_metrics.maximum_absolute_difference,
                router_metrics.maximum_absolute_difference,
                propagated_budgets.router,
                routing_metrics.maximum_absolute_difference,
                propagated_budgets.routing,
                comma_separated(&f32_ids),
                comma_separated(&bf16_ids),
                comma_separated(&pre_router.router.selected_experts),
            )
            .expect("write Layer-24 evidence");
            println!(
                "layer_execution layer={layer} elapsed_seconds={} input_max_abs={} router_max_abs={} routing_max_abs={} stop=router",
                layer_started.elapsed().as_secs_f64(),
                input_metrics.maximum_absolute_difference,
                router_metrics.maximum_absolute_difference,
                routing_metrics.maximum_absolute_difference,
            );
            break;
        }

        let unique_experts = pre_router
            .router
            .selected_experts
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len();
        let expert_occurrences = pre_router.router.selected_experts.len();
        let mut diagnostic_expert_outputs = HashMap::new();
        let moe_output = streaming_routed_experts_with_observer(
            pre_router.post_attention_norm.view(),
            &pre_router.router,
            config,
            layer,
            &mut store,
            expert_layout,
            |expert, token, position, values| {
                if layer == 1 {
                    let name = format!("layer1_expert_output_t{token}_p{position}_e{expert}");
                    let output = Tensor::new(TensorShape::new([2048]), values.to_vec())
                        .expect("Layer-1 diagnostic expert output");
                    assert!(
                        diagnostic_expert_outputs.insert(name, output).is_none(),
                        "duplicate Layer-1 diagnostic expert output"
                    );
                }
            },
        )
        .unwrap_or_else(|error| panic!("Layer-{layer} expert execution failed: {error}"));
        let bf16_moe = checkpoint_f32(
            LAYER24_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_moe_output"),
        );
        let f32_moe = checkpoint_f32(
            LAYER24_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_moe_output"),
        );
        let moe_metrics = record_three_paths(
            &format!("{prefix}_moe_output"),
            &moe_output,
            &bf16_moe,
            &f32_moe,
        );
        let moe_budget = propagated_budgets.moe.expect("completed layer MoE budget");
        if layer == 1
            && let Some(root) = env::var_os("COLIBRI_LAYER1_MOE_DIAGNOSTIC_ROOT")
        {
            assert_eq!(
                diagnostic_expert_outputs.len(),
                32,
                "Layer-1 diagnostic occurrence count"
            );
            export_layer1_moe_diagnostic(
                &PathBuf::from(root),
                &pre_router,
                &moe_output,
                &diagnostic_expert_outputs,
            );
        }
        assert_stage(
            &format!("{prefix}_moe_output"),
            &moe_output,
            &f32_moe,
            moe_budget,
            0.0,
        );
        let block_output = elementwise_add(pre_router.residual_output.view(), moe_output.view())
            .unwrap_or_else(|error| panic!("Layer-{layer} final residual failed: {error}"));
        let bf16_block = checkpoint_f32(
            LAYER24_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_block_output"),
        );
        let f32_block = checkpoint_f32(
            LAYER24_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_block_output"),
        );
        let block_metrics = record_three_paths(
            &format!("{prefix}_block_output"),
            &block_output,
            &bf16_block,
            &f32_block,
        );
        let block_budget = propagated_budgets
            .block
            .expect("completed layer block budget")
            + f32::EPSILON * maximum_magnitude(&f32_block);
        assert_stage(
            &format!("{prefix}_block_output"),
            &block_output,
            &f32_block,
            block_budget,
            0.0,
        );
        writeln!(
            layer_evidence,
            "{layer}\t{:.17e}\t{input_budget:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{unique_experts}\t{expert_occurrences}\t{:.17e}\t{moe_budget:.17e}\t{:.17e}\t{block_budget:.17e}\t{}\t{}\t{}",
            input_metrics.maximum_absolute_difference,
            router_metrics.maximum_absolute_difference,
            propagated_budgets.router,
            routing_metrics.maximum_absolute_difference,
            propagated_budgets.routing,
            moe_metrics.maximum_absolute_difference,
            block_metrics.maximum_absolute_difference,
            comma_separated(&f32_ids),
            comma_separated(&bf16_ids),
            comma_separated(&pre_router.router.selected_experts),
        )
        .expect("write completed-layer evidence");
        total_unique_experts += unique_experts;
        total_expert_occurrences += expert_occurrences;
        maximum_runtime_elements = maximum_runtime_elements.max(
            current.data().len()
                + pre_router.input_norm.data().len()
                + pre_router.attention_output.data().len()
                + pre_router.residual_output.data().len()
                + pre_router.post_attention_norm.data().len()
                + pre_router.router.logits.data().len()
                + pre_router.router.weights.data().len()
                + moe_output.data().len()
                + block_output.data().len(),
        );
        println!(
            "layer_execution layer={layer} elapsed_seconds={} input_max_abs={} router_max_abs={} routing_max_abs={} unique_experts={unique_experts} expert_occurrences={expert_occurrences} moe_max_abs={} block_max_abs={}",
            layer_started.elapsed().as_secs_f64(),
            input_metrics.maximum_absolute_difference,
            router_metrics.maximum_absolute_difference,
            routing_metrics.maximum_absolute_difference,
            moe_metrics.maximum_absolute_difference,
            block_metrics.maximum_absolute_difference,
        );
        current = block_output;
    }

    let cache_metrics = store.metrics();
    assert_eq!(cache_metrics.loads, total_unique_experts as u64);
    assert_eq!(cache_metrics.misses, total_unique_experts as u64);
    assert_eq!(cache_metrics.hits, 0);
    assert_eq!(cache_metrics.evictions, cache_metrics.loads - 1);
    assert_eq!(total_expert_occurrences, 24 * 32);
    let checkpoint_static_bytes = LAYER24_BF16_CHECKPOINTS.len()
        + LAYER24_F32_CHECKPOINTS.len()
        + LAYER24_BF16_CHECKPOINT_PLAN.len()
        + LAYER24_F32_CHECKPOINT_PLAN.len();
    let modeled_peak_explicit_bytes = usize::try_from(maximum_dense_layer_bytes)
        .expect("dense layer bytes fit usize")
        + cache_metrics.peak_resident_bytes
        + expert_layout.total_byte_length
        + maximum_runtime_elements * 4
        + checkpoint_static_bytes;
    writeln!(
        checkpoint_evidence,
        "resources\tdense_bytes_read\t{dense_bytes_read}\texpert_bytes_read={}\ttotal_artifact_bytes_read={}\ttotal_unique_experts={total_unique_experts}\ttotal_expert_occurrences={total_expert_occurrences}\thits={}\tmisses={}\tloads={}\tevictions={}\tresident_bytes={}\tpeak_expert_resident_bytes={}\tmodeled_peak_explicit_bytes={modeled_peak_explicit_bytes}",
        cache_metrics.bytes_read,
        dense_bytes_read + cache_metrics.bytes_read,
        cache_metrics.hits,
        cache_metrics.misses,
        cache_metrics.loads,
        cache_metrics.evictions,
        cache_metrics.resident_bytes,
        cache_metrics.peak_resident_bytes,
    )
    .expect("write Layer-24 resource evidence");
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-02-rust-layer24-layer-evidence-v1.tsv"),
        layer_evidence.as_bytes(),
    );
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-02-rust-layer24-checkpoint-evidence-v1.tsv"),
        checkpoint_evidence.as_bytes(),
    );
    println!(
        "layer24_end_to_end dense_bytes_read={dense_bytes_read} expert_metrics={cache_metrics:?} total_unique_experts={total_unique_experts} total_expert_occurrences={total_expert_occurrences} modeled_peak_explicit_bytes={modeled_peak_explicit_bytes} elapsed_seconds={} classifications={layer24_classifications:?}",
        started.elapsed().as_secs_f64(),
    );
}

#[test]
fn pinned_layer_forty_seven_router_uses_genuine_streaming_prefix() {
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let diagnostic_root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("diagnostic root for Layer-47 evidence");
    assert!(
        diagnostic_root.is_absolute(),
        "diagnostic root must be absolute"
    );
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length,
        "dense payload length"
    );
    let bf16_plan = checkpoint_plan(LAYER47_BF16_CHECKPOINT_PLAN);
    let f32_plan = checkpoint_plan(LAYER47_F32_CHECKPOINT_PLAN);
    assert_eq!(
        checkpoint_ids(LAYER47_F32_CHECKPOINTS, &f32_plan, "input_ids"),
        INPUT_IDS
    );
    assert_eq!(
        checkpoint_ids(LAYER47_F32_CHECKPOINTS, &f32_plan, "position_ids"),
        POSITION_IDS
    );

    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut store = expert_store_from_plan(LAYER47_EXPERT_RUNTIME_PLAN, &artifact_root, 47 * 128);
    let mut dense_bytes_read = 0_u64;
    let mut current = embedding_rows(&mut payload, &plan, &mut dense_bytes_read);
    let embedding_metrics = record_three_paths(
        "layer47_embedding_output",
        &current,
        &checkpoint_f32(LAYER47_BF16_CHECKPOINTS, &bf16_plan, "embedding_output"),
        &checkpoint_f32(LAYER47_F32_CHECKPOINTS, &f32_plan, "embedding_output"),
    );
    assert_eq!(
        embedding_metrics.maximum_absolute_difference, 0.0,
        "embedding must remain exact"
    );

    let mut layer_evidence = String::from(
        "layer\tinput_max_abs\tinput_budget\trouter_max_abs\trouter_budget\trouting_max_abs\trouting_budget\tunique_experts\texpert_occurrences\tmoe_max_abs\tmoe_budget\tblock_max_abs\tblock_budget\ttransformers_f32_ids\ttransformers_bf16_ids\trust_ids\n",
    );
    let mut checkpoint_evidence = String::from(
        "layer\tcheckpoint\tmaximum_f32_vs_rust_absolute_error\tabsolute_budget\tmaximum_f32_vs_rust_relative_error\n",
    );
    let budget_text = |budget: Option<f32>| {
        budget.map_or_else(|| "NA".to_owned(), |value| format!("{value:.17e}"))
    };
    let mut total_unique_experts = 0_usize;
    let mut total_expert_occurrences = 0_usize;
    let mut maximum_dense_layer_bytes = 0_u64;
    let mut maximum_runtime_elements = current.data().len();
    let started = Instant::now();
    let mut layer47_classifications = Vec::new();

    for layer in 0..=47 {
        let layer_started = Instant::now();
        let dense_before = dense_bytes_read;
        let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
        maximum_dense_layer_bytes = maximum_dense_layer_bytes.max(dense_bytes_read - dense_before);
        let pre_router = pre_router_with_weights(
            current.view(),
            weights.input_norm.view(),
            weights.query.view(),
            weights.key.view(),
            weights.value.view(),
            weights.output.view(),
            weights.query_norm.view(),
            weights.key_norm.view(),
            weights.post_norm.view(),
            weights.router.view(),
            config,
        )
        .unwrap_or_else(|error| panic!("Layer-{layer} pre-router execution failed: {error}"));
        drop(weights);
        let prefix = format!("layer{layer}");
        let approved_budgets = Some(layer47_propagated_budgets(layer));
        let bf16_input = checkpoint_f32(
            LAYER47_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_input"),
        );
        let f32_input = checkpoint_f32(
            LAYER47_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_input"),
        );
        let input_metrics = record_three_paths(
            &format!("{prefix}_input"),
            &current,
            &bf16_input,
            &f32_input,
        );
        let input_budget = approved_budgets.map(|budgets| budgets.input);
        if let Some(budget) = input_budget {
            assert_stage(
                &format!("{prefix}_input"),
                &current,
                &f32_input,
                budget,
                0.0,
            );
        }

        let bf16_logits = checkpoint_f32(
            LAYER47_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_router_logits"),
        );
        let f32_logits = checkpoint_f32(
            LAYER47_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_router_logits"),
        );
        let router_metrics = record_three_paths(
            &format!("{prefix}_router_logits"),
            &pre_router.router.logits,
            &bf16_logits,
            &f32_logits,
        );
        let router_budget = approved_budgets.map(|budgets| budgets.router);
        if let Some(budget) = router_budget {
            assert_stage(
                &format!("{prefix}_router_logits"),
                &pre_router.router.logits,
                &f32_logits,
                budget,
                0.0,
            );
        }
        let bf16_routing = checkpoint_f32(
            LAYER47_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_routing_weights"),
        );
        let f32_routing = checkpoint_f32(
            LAYER47_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_routing_weights"),
        );
        let routing_metrics = record_three_paths(
            &format!("{prefix}_routing_weights"),
            &pre_router.router.weights,
            &bf16_routing,
            &f32_routing,
        );
        let routing_budget = approved_budgets.map(|budgets| budgets.routing);
        if let Some(budget) = routing_budget {
            assert_stage(
                &format!("{prefix}_routing_weights"),
                &pre_router.router.weights,
                &f32_routing,
                budget,
                0.0,
            );
        }
        let bf16_ids = checkpoint_ids(
            LAYER47_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_selected_expert_ids"),
        );
        let f32_ids = checkpoint_ids(
            LAYER47_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_selected_expert_ids"),
        );
        assert_eq!(
            pre_router.router.selected_experts, f32_ids,
            "Layer-{layer} Rust and Transformers F32 expert IDs differ"
        );

        for (suffix, actual) in [
            ("input_rmsnorm", &pre_router.input_norm),
            ("attention_output", &pre_router.attention_output),
            ("residual_output", &pre_router.residual_output),
            ("post_attention_rmsnorm", &pre_router.post_attention_norm),
        ] {
            let name = format!("{prefix}_{suffix}");
            let bf16_reference = checkpoint_f32(LAYER47_BF16_CHECKPOINTS, &bf16_plan, &name);
            let f32_reference = checkpoint_f32(LAYER47_F32_CHECKPOINTS, &f32_plan, &name);
            let metrics = record_three_paths(&name, actual, &bf16_reference, &f32_reference);
            let budget = approved_budgets.map(|budgets| match suffix {
                "input_rmsnorm" => budgets.input_norm,
                "attention_output" => budgets.attention,
                "residual_output" => budgets.residual,
                "post_attention_rmsnorm" => budgets.post_norm,
                _ => unreachable!("known pre-router checkpoint"),
            });
            if let Some(budget) = budget {
                assert_stage(&name, actual, &f32_reference, budget, 0.0);
            }
            writeln!(
                checkpoint_evidence,
                "{layer}\t{suffix}\t{:.17e}\t{}\t{:.17e}",
                metrics.maximum_absolute_difference,
                budget_text(budget),
                metrics.maximum_relative_difference,
            )
            .expect("write Layer-47 checkpoint evidence");
        }

        maximum_runtime_elements = maximum_runtime_elements.max(
            current.data().len()
                + pre_router.input_norm.data().len()
                + pre_router.attention_output.data().len()
                + pre_router.residual_output.data().len()
                + pre_router.post_attention_norm.data().len()
                + pre_router.router.logits.data().len()
                + pre_router.router.weights.data().len()
                + bf16_logits.data().len()
                + f32_logits.data().len(),
        );
        if layer == 47 {
            layer47_classifications = validate_router_boundaries(
                &pre_router,
                &bf16_logits,
                &f32_logits,
                &bf16_ids,
                &f32_ids,
                "m4.2-02-rust-layer47-router-evidence-v1.tsv",
            );
            writeln!(
                layer_evidence,
                "{layer}\t{:.17e}\t{}\t{:.17e}\t{}\t{:.17e}\t{}\t0\t0\tNA\tNA\tNA\tNA\t{}\t{}\t{}",
                input_metrics.maximum_absolute_difference,
                budget_text(input_budget),
                router_metrics.maximum_absolute_difference,
                budget_text(router_budget),
                routing_metrics.maximum_absolute_difference,
                budget_text(routing_budget),
                comma_separated(&f32_ids),
                comma_separated(&bf16_ids),
                comma_separated(&pre_router.router.selected_experts),
            )
            .expect("write Layer-47 stop evidence");
            println!(
                "layer47_execution layer={layer} elapsed_seconds={} input_max_abs={} router_max_abs={} routing_max_abs={} stop=router",
                layer_started.elapsed().as_secs_f64(),
                input_metrics.maximum_absolute_difference,
                router_metrics.maximum_absolute_difference,
                routing_metrics.maximum_absolute_difference,
            );
            break;
        }

        let unique_experts = pre_router
            .router
            .selected_experts
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len();
        let expert_occurrences = pre_router.router.selected_experts.len();
        let moe_output = streaming_routed_experts_with_observer(
            pre_router.post_attention_norm.view(),
            &pre_router.router,
            config,
            layer,
            &mut store,
            expert_layout,
            |_, _, _, _| {},
        )
        .unwrap_or_else(|error| panic!("Layer-{layer} expert execution failed: {error}"));
        let bf16_moe = checkpoint_f32(
            LAYER47_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_moe_output"),
        );
        let f32_moe = checkpoint_f32(
            LAYER47_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_moe_output"),
        );
        let moe_metrics = record_three_paths(
            &format!("{prefix}_moe_output"),
            &moe_output,
            &bf16_moe,
            &f32_moe,
        );
        let moe_budget = approved_budgets.and_then(|budgets| budgets.moe);
        if let Some(budget) = moe_budget {
            assert_stage(
                &format!("{prefix}_moe_output"),
                &moe_output,
                &f32_moe,
                budget,
                0.0,
            );
        }
        let block_output = elementwise_add(pre_router.residual_output.view(), moe_output.view())
            .unwrap_or_else(|error| panic!("Layer-{layer} final residual failed: {error}"));
        let bf16_block = checkpoint_f32(
            LAYER47_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("{prefix}_block_output"),
        );
        let f32_block = checkpoint_f32(
            LAYER47_F32_CHECKPOINTS,
            &f32_plan,
            &format!("{prefix}_block_output"),
        );
        let block_metrics = record_three_paths(
            &format!("{prefix}_block_output"),
            &block_output,
            &bf16_block,
            &f32_block,
        );
        let block_budget = approved_budgets.and_then(|budgets| {
            budgets
                .block
                .map(|value| value + f32::EPSILON * maximum_magnitude(&f32_block))
        });
        if let Some(budget) = block_budget {
            assert_stage(
                &format!("{prefix}_block_output"),
                &block_output,
                &f32_block,
                budget,
                0.0,
            );
        }
        writeln!(
            layer_evidence,
            "{layer}\t{:.17e}\t{}\t{:.17e}\t{}\t{:.17e}\t{}\t{unique_experts}\t{expert_occurrences}\t{:.17e}\t{}\t{:.17e}\t{}\t{}\t{}\t{}",
            input_metrics.maximum_absolute_difference,
            budget_text(input_budget),
            router_metrics.maximum_absolute_difference,
            budget_text(router_budget),
            routing_metrics.maximum_absolute_difference,
            budget_text(routing_budget),
            moe_metrics.maximum_absolute_difference,
            budget_text(moe_budget),
            block_metrics.maximum_absolute_difference,
            budget_text(block_budget),
            comma_separated(&f32_ids),
            comma_separated(&bf16_ids),
            comma_separated(&pre_router.router.selected_experts),
        )
        .expect("write Layer-47 completed-layer evidence");
        total_unique_experts += unique_experts;
        total_expert_occurrences += expert_occurrences;
        maximum_runtime_elements = maximum_runtime_elements.max(
            current.data().len()
                + pre_router.input_norm.data().len()
                + pre_router.attention_output.data().len()
                + pre_router.residual_output.data().len()
                + pre_router.post_attention_norm.data().len()
                + pre_router.router.logits.data().len()
                + pre_router.router.weights.data().len()
                + moe_output.data().len()
                + block_output.data().len(),
        );
        println!(
            "layer47_execution layer={layer} elapsed_seconds={} input_max_abs={} router_max_abs={} routing_max_abs={} unique_experts={unique_experts} expert_occurrences={expert_occurrences} moe_max_abs={} block_max_abs={}",
            layer_started.elapsed().as_secs_f64(),
            input_metrics.maximum_absolute_difference,
            router_metrics.maximum_absolute_difference,
            routing_metrics.maximum_absolute_difference,
            moe_metrics.maximum_absolute_difference,
            block_metrics.maximum_absolute_difference,
        );
        current = block_output;
    }

    let cache_metrics = store.metrics();
    assert_eq!(cache_metrics.loads, total_unique_experts as u64);
    assert_eq!(cache_metrics.misses, total_unique_experts as u64);
    assert_eq!(cache_metrics.hits, 0);
    assert_eq!(cache_metrics.evictions, cache_metrics.loads - 1);
    assert_eq!(total_expert_occurrences, 47 * 32);
    let checkpoint_static_bytes = LAYER47_BF16_CHECKPOINTS.len()
        + LAYER47_F32_CHECKPOINTS.len()
        + LAYER47_BF16_CHECKPOINT_PLAN.len()
        + LAYER47_F32_CHECKPOINT_PLAN.len();
    let modeled_peak_explicit_bytes = usize::try_from(maximum_dense_layer_bytes)
        .expect("dense layer bytes fit usize")
        + cache_metrics.peak_resident_bytes
        + expert_layout.total_byte_length
        + maximum_runtime_elements * 4
        + checkpoint_static_bytes;
    writeln!(
        checkpoint_evidence,
        "resources\tdense_bytes_read\t{dense_bytes_read}\texpert_bytes_read={}\ttotal_artifact_bytes_read={}\ttotal_unique_experts={total_unique_experts}\ttotal_expert_occurrences={total_expert_occurrences}\thits={}\tmisses={}\tloads={}\tevictions={}\tresident_bytes={}\tpeak_expert_resident_bytes={}\tmodeled_peak_explicit_bytes={modeled_peak_explicit_bytes}",
        cache_metrics.bytes_read,
        dense_bytes_read + cache_metrics.bytes_read,
        cache_metrics.hits,
        cache_metrics.misses,
        cache_metrics.loads,
        cache_metrics.evictions,
        cache_metrics.resident_bytes,
        cache_metrics.peak_resident_bytes,
    )
    .expect("write Layer-47 resource evidence");
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-02-rust-layer47-layer-evidence-v1.tsv"),
        layer_evidence.as_bytes(),
    );
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-02-rust-layer47-checkpoint-evidence-v1.tsv"),
        checkpoint_evidence.as_bytes(),
    );
    println!(
        "layer47_end_to_end dense_bytes_read={dense_bytes_read} expert_metrics={cache_metrics:?} total_unique_experts={total_unique_experts} total_expert_occurrences={total_expert_occurrences} modeled_peak_explicit_bytes={modeled_peak_explicit_bytes} elapsed_seconds={} classifications={layer47_classifications:?}",
        started.elapsed().as_secs_f64(),
    );
}

#[test]
fn selected_expert_intermediates_match_transformers() {
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let diagnostic_root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("diagnostic root for intermediate evidence");
    assert!(
        diagnostic_root.is_absolute(),
        "intermediate diagnostic root must be absolute"
    );
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length,
        "dense payload length"
    );
    let full_f32_plan = checkpoint_plan(LAYER47_F32_CHECKPOINT_PLAN);
    let intermediate_bf16_plan = checkpoint_plan(INTERMEDIATE_BF16_CHECKPOINT_PLAN);
    let intermediate_f32_plan = checkpoint_plan(INTERMEDIATE_F32_CHECKPOINT_PLAN);
    assert_eq!(
        intermediate_bf16_plan.keys().collect::<HashSet<_>>(),
        intermediate_f32_plan.keys().collect::<HashSet<_>>(),
        "intermediate checkpoint names"
    );

    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    validate_intermediate_structure(expert_layout);
    let layer47_selected_count = LAYER47_SELECTED_EXPERT_RUNTIME_PLAN
        .lines()
        .filter(|line| line.starts_with("expert\t"))
        .count();
    assert_eq!(layer47_selected_count, 21, "Layer-47 selected expert count");
    let mut store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            LAYER47_SELECTED_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        47 * 128 + layer47_selected_count,
    );
    let mut dense_bytes_read = 0_u64;
    let mut current = embedding_rows(&mut payload, &plan, &mut dense_bytes_read);
    let mut traces: HashMap<(usize, usize, usize, usize), ExpertMlpTrace> = HashMap::new();
    let mut evidence = String::from(
        "layer\ttoken\tposition\texpert\tcheckpoint\tmaximum_f32_vs_rust_absolute_error\tabsolute_budget\tmaximum_f32_vs_rust_relative_error\tmaximum_bf16_vs_rust_absolute_error\tmaximum_bf16_vs_rust_relative_error\n",
    );
    let mut total_unique_experts = 0_usize;
    let mut total_expert_occurrences = 0_usize;
    let mut maximum_dense_layer_bytes = 0_u64;
    let mut maximum_runtime_elements = current.data().len();
    let started = Instant::now();

    for layer in 0..=47 {
        let dense_before = dense_bytes_read;
        let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
        maximum_dense_layer_bytes = maximum_dense_layer_bytes.max(dense_bytes_read - dense_before);
        let pre_router = pre_router_with_weights(
            current.view(),
            weights.input_norm.view(),
            weights.query.view(),
            weights.key.view(),
            weights.value.view(),
            weights.output.view(),
            weights.query_norm.view(),
            weights.key_norm.view(),
            weights.post_norm.view(),
            weights.router.view(),
            config,
        )
        .unwrap_or_else(|error| panic!("Layer-{layer} pre-router execution failed: {error}"));
        drop(weights);
        let prefix = format!("layer{layer}");
        let budgets = layer47_propagated_budgets(layer);
        let f32_input = checkpoint_f32(
            LAYER47_F32_CHECKPOINTS,
            &full_f32_plan,
            &format!("{prefix}_input"),
        );
        assert_stage(
            &format!("{prefix}_input"),
            &current,
            &f32_input,
            budgets.input,
            0.0,
        );
        for (suffix, actual, budget) in [
            ("input_rmsnorm", &pre_router.input_norm, budgets.input_norm),
            (
                "attention_output",
                &pre_router.attention_output,
                budgets.attention,
            ),
            (
                "residual_output",
                &pre_router.residual_output,
                budgets.residual,
            ),
            (
                "post_attention_rmsnorm",
                &pre_router.post_attention_norm,
                budgets.post_norm,
            ),
            ("router_logits", &pre_router.router.logits, budgets.router),
            (
                "routing_weights",
                &pre_router.router.weights,
                budgets.routing,
            ),
        ] {
            let name = format!("{prefix}_{suffix}");
            let reference = checkpoint_f32(LAYER47_F32_CHECKPOINTS, &full_f32_plan, &name);
            assert_stage(&name, actual, &reference, budget, 0.0);
        }
        let f32_ids = checkpoint_ids(
            LAYER47_F32_CHECKPOINTS,
            &full_f32_plan,
            &format!("{prefix}_selected_expert_ids"),
        );
        assert_eq!(
            pre_router.router.selected_experts, f32_ids,
            "Layer-{layer} selected expert IDs"
        );
        let unique_experts = pre_router
            .router
            .selected_experts
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len();
        let expert_occurrences = pre_router.router.selected_experts.len();
        let trace_layer = INTERMEDIATE_CASES
            .iter()
            .any(|&(case_layer, _, _, _)| case_layer == layer);
        let moe_output = if trace_layer {
            streaming_routed_experts_with_trace_observer(
                pre_router.post_attention_norm.view(),
                &pre_router.router,
                config,
                layer,
                &mut store,
                expert_layout,
                |expert, token, position| {
                    INTERMEDIATE_CASES.contains(&(layer, token, position, expert))
                },
                |expert, token, position, output, trace| {
                    assert_eq!(
                        output, trace.down_projection,
                        "traced and normal expert output differ"
                    );
                    assert!(
                        traces
                            .insert((layer, token, position, expert), trace.clone())
                            .is_none(),
                        "duplicate intermediate trace"
                    );
                },
            )
        } else {
            streaming_routed_experts_with_observer(
                pre_router.post_attention_norm.view(),
                &pre_router.router,
                config,
                layer,
                &mut store,
                expert_layout,
                |_, _, _, _| {},
            )
        }
        .unwrap_or_else(|error| panic!("Layer-{layer} expert execution failed: {error}"));
        let block_output = elementwise_add(pre_router.residual_output.view(), moe_output.view())
            .unwrap_or_else(|error| panic!("Layer-{layer} final residual failed: {error}"));

        if layer < 47 {
            let f32_moe = checkpoint_f32(
                LAYER47_F32_CHECKPOINTS,
                &full_f32_plan,
                &format!("{prefix}_moe_output"),
            );
            assert_stage(
                &format!("{prefix}_moe_output"),
                &moe_output,
                &f32_moe,
                budgets.moe.expect("completed-layer MoE budget"),
                0.0,
            );
            let f32_block = checkpoint_f32(
                LAYER47_F32_CHECKPOINTS,
                &full_f32_plan,
                &format!("{prefix}_block_output"),
            );
            assert_stage(
                &format!("{prefix}_block_output"),
                &block_output,
                &f32_block,
                budgets.block.expect("completed-layer block budget")
                    + f32::EPSILON * maximum_magnitude(&f32_block),
                0.0,
            );
        }

        for &(case_layer, token, position, expert) in &INTERMEDIATE_CASES {
            if case_layer != layer {
                continue;
            }
            let trace = &traces[&(layer, token, position, expert)];
            let expert_input = tensor_row(&pre_router.post_attention_norm, token);
            let routing_weight = Tensor::new(
                TensorShape::new([1]),
                vec![pre_router.router.weights.data()[token * 8 + position]],
            )
            .expect("routing weight tensor");
            let weighted_output = Tensor::new(
                TensorShape::new([2048]),
                trace
                    .down_projection
                    .iter()
                    .map(|value| value * routing_weight.data()[0])
                    .collect(),
            )
            .expect("weighted expert output tensor");
            let checkpoint_tensors = [
                ("expert_input", expert_input),
                (
                    "gate_projection",
                    Tensor::new(TensorShape::new([768]), trace.gate_projection.clone()).unwrap(),
                ),
                (
                    "up_projection",
                    Tensor::new(TensorShape::new([768]), trace.up_projection.clone()).unwrap(),
                ),
                (
                    "activated_gate",
                    Tensor::new(TensorShape::new([768]), trace.activated_gate.clone()).unwrap(),
                ),
                (
                    "activated_product",
                    Tensor::new(TensorShape::new([768]), trace.activated_product.clone()).unwrap(),
                ),
                (
                    "down_projection",
                    Tensor::new(TensorShape::new([2048]), trace.down_projection.clone()).unwrap(),
                ),
                ("routing_weight", routing_weight),
                ("weighted_expert_output", weighted_output),
            ];
            for (checkpoint, actual) in checkpoint_tensors {
                let name = intermediate_case_name(layer, token, position, expert, checkpoint);
                let bf16 = checkpoint_f32(
                    INTERMEDIATE_BF16_CHECKPOINTS,
                    &intermediate_bf16_plan,
                    &name,
                );
                let f32_control =
                    checkpoint_f32(INTERMEDIATE_F32_CHECKPOINTS, &intermediate_f32_plan, &name);
                record_intermediate_checkpoint(
                    &mut evidence,
                    layer,
                    token,
                    Some(position),
                    Some(expert),
                    checkpoint,
                    &actual,
                    &bf16,
                    &f32_control,
                    Some(selected_intermediate_budget(layer, checkpoint)),
                );
            }
        }

        if let Some(&(_, token, _, _)) = INTERMEDIATE_CASES
            .iter()
            .find(|&&(case_layer, _, _, _)| case_layer == layer)
        {
            let aggregate = tensor_row(&moe_output, token);
            let residual_addition = tensor_row(&block_output, token);
            for (checkpoint, actual) in [
                ("aggregated_moe_output", aggregate),
                ("moe_residual_addition", residual_addition.clone()),
                ("final_block_output", residual_addition),
            ] {
                let name = intermediate_layer_name(layer, token, checkpoint);
                let bf16 = checkpoint_f32(
                    INTERMEDIATE_BF16_CHECKPOINTS,
                    &intermediate_bf16_plan,
                    &name,
                );
                let f32_control =
                    checkpoint_f32(INTERMEDIATE_F32_CHECKPOINTS, &intermediate_f32_plan, &name);
                record_intermediate_checkpoint(
                    &mut evidence,
                    layer,
                    token,
                    None,
                    None,
                    checkpoint,
                    &actual,
                    &bf16,
                    &f32_control,
                    Some(selected_intermediate_budget(layer, checkpoint)),
                );
            }
        }

        total_unique_experts += unique_experts;
        total_expert_occurrences += expert_occurrences;
        maximum_runtime_elements = maximum_runtime_elements.max(
            current.data().len()
                + pre_router.input_norm.data().len()
                + pre_router.attention_output.data().len()
                + pre_router.residual_output.data().len()
                + pre_router.post_attention_norm.data().len()
                + pre_router.router.logits.data().len()
                + pre_router.router.weights.data().len()
                + moe_output.data().len()
                + block_output.data().len(),
        );
        current = block_output;
    }

    assert_eq!(traces.len(), INTERMEDIATE_CASES.len());
    assert_eq!(total_expert_occurrences, 48 * 32);
    let cache_metrics = store.metrics();
    assert_eq!(cache_metrics.loads, total_unique_experts as u64);
    assert_eq!(cache_metrics.misses, total_unique_experts as u64);
    assert_eq!(cache_metrics.hits, 0);
    assert_eq!(cache_metrics.evictions, cache_metrics.loads - 1);
    let retained_trace_elements: usize = traces
        .values()
        .map(|trace| {
            trace.gate_projection.len()
                + trace.up_projection.len()
                + trace.activated_gate.len()
                + trace.activated_product.len()
                + trace.down_projection.len()
        })
        .sum();
    let checkpoint_static_bytes = LAYER47_F32_CHECKPOINTS.len()
        + LAYER47_F32_CHECKPOINT_PLAN.len()
        + INTERMEDIATE_BF16_CHECKPOINTS.len()
        + INTERMEDIATE_BF16_CHECKPOINT_PLAN.len()
        + INTERMEDIATE_F32_CHECKPOINTS.len()
        + INTERMEDIATE_F32_CHECKPOINT_PLAN.len();
    let modeled_peak_explicit_bytes = usize::try_from(maximum_dense_layer_bytes)
        .expect("dense layer bytes fit usize")
        + cache_metrics.peak_resident_bytes
        + expert_layout.total_byte_length
        + (maximum_runtime_elements + retained_trace_elements) * 4
        + checkpoint_static_bytes;
    writeln!(
        evidence,
        "resources\tNA\tNA\tNA\tdense_bytes_read={dense_bytes_read}\texpert_bytes_read={}\ttotal_artifact_bytes_read={}\ttotal_unique_experts={total_unique_experts}\ttotal_expert_occurrences={total_expert_occurrences}\thits={}\tmisses={}\tloads={}\tevictions={}\tpeak_expert_resident_bytes={}\tmodeled_peak_explicit_bytes={modeled_peak_explicit_bytes}",
        cache_metrics.bytes_read,
        dense_bytes_read + cache_metrics.bytes_read,
        cache_metrics.hits,
        cache_metrics.misses,
        cache_metrics.loads,
        cache_metrics.evictions,
        cache_metrics.peak_resident_bytes,
    )
    .expect("write intermediate resources");
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-03-rust-intermediate-evidence-v1.tsv"),
        evidence.as_bytes(),
    );
    println!(
        "selected_intermediate_validation elapsed_seconds={} dense_bytes_read={dense_bytes_read} expert_bytes_read={} modeled_peak_explicit_bytes={modeled_peak_explicit_bytes}",
        started.elapsed().as_secs_f64(),
        cache_metrics.bytes_read,
    );
}

#[test]
fn short_cached_generation_matches_transformers() {
    #[cfg(feature = "m5-3-compute-profiling")]
    let profiling_session = crate::profiling::start_from_env();
    #[cfg(feature = "m5-3-compute-profiling")]
    let model_profile = crate::profiling::scope("model.total");
    let metrics_output = env::var_os("COLIBRI_METRICS_OUTPUT").map(PathBuf::from);
    let filesystem_cache_assumption =
        env::var("COLIBRI_FS_CACHE_ASSUMPTION").unwrap_or_else(|_| "not_recorded".to_owned());
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let diagnostic_root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("diagnostic root for short generation evidence");
    assert!(
        diagnostic_root.is_absolute(),
        "diagnostic root must be absolute"
    );
    let full_logits_root = env::var_os("COLIBRI_FULL_LOGITS_ROOT")
        .map(PathBuf::from)
        .expect("temporary full-logit root");
    assert!(
        full_logits_root.is_absolute(),
        "full-logit root must be absolute"
    );

    let bf16_plan = checkpoint_plan(GENERATION_BF16_CHECKPOINT_PLAN);
    let f32_plan = checkpoint_plan(GENERATION_F32_CHECKPOINT_PLAN);
    let trace_only = env::var_os("COLIBRI_TRACE_ONLY").is_some();
    let (bf16_full_bytes, f32_full_bytes, bf16_full_plan, f32_full_plan) = if trace_only {
        // The committed generation bundle contains fixed-vocabulary and top-20
        // checkpoints, while the full-vocabulary logits are deliberately kept
        // in the temporary M4 reference run. Trace capture still validates
        // generated IDs, guard routing, finite outputs, and all other frozen
        // checkpoints without inventing a full-logit reference.
        (Vec::new(), Vec::new(), HashMap::new(), HashMap::new())
    } else {
        let bf16_full_bytes = fs::read(full_logits_root.join("bf16-full-logits.safetensors"))
            .expect("read temporary BF16 full logits");
        let f32_full_bytes = fs::read(full_logits_root.join("f32-full-logits.safetensors"))
            .expect("read temporary F32 full logits");
        let bf16_full_plan = checkpoint_plan(
            &fs::read_to_string(full_logits_root.join("bf16-full-logits-plan.tsv"))
                .expect("read BF16 full-logit plan"),
        );
        let f32_full_plan = checkpoint_plan(
            &fs::read_to_string(full_logits_root.join("f32-full-logits-plan.tsv"))
                .expect("read F32 full-logit plan"),
        );
        (
            bf16_full_bytes,
            f32_full_bytes,
            bf16_full_plan,
            f32_full_plan,
        )
    };

    // Reference evidence is prepared before this boundary and is excluded from
    // the storage-aware Rust runtime timings below.
    let runtime_started = Instant::now();
    let cache_budget = env::var("COLIBRI_EXPERT_CACHE_BUDGET_BYTES").map_or(18_874_368, |value| {
        value.parse::<usize>().expect("valid expert cache budget")
    });
    let runtime_validation = env::var_os("COLIBRI_RUNTIME_VALIDATION").is_some();
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    assert_eq!(plan.payload, final_plan.payload, "dense payload identity");
    assert_eq!(
        plan.payload_length, final_plan.payload_length,
        "dense payload length"
    );
    assert_eq!(
        fs::metadata(artifact_root.join(&plan.payload))
            .expect("dense payload metadata")
            .len(),
        plan.payload_length,
    );
    #[cfg(feature = "m5-4-resident-dense")]
    let mut payload = dense_source_for_generation(&artifact_root.join(&plan.payload), cache_budget);
    #[cfg(not(feature = "m5-4-resident-dense"))]
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut cache =
        KvCache::new(48, GENERATION_INPUT_TOKENS.len(), 4, 128).expect("fixed full-model KV cache");
    let allocation_capacities = cache.allocation_capacities();
    assert_eq!(allocation_capacities.len(), 48, "KV cache layer count");
    assert!(
        allocation_capacities
            .iter()
            .all(|&(key, value)| key == 6 * 4 * 128 && value == 6 * 4 * 128),
        "fixed KV allocation shapes"
    );
    assert_eq!(cache.byte_size(), 1_179_648, "KV cache byte size");

    let mut evidence = String::from(
        "record\tstep\tcheckpoint\tmaximum_f32_vs_rust_absolute_error\tabsolute_budget\tmaximum_f32_vs_rust_relative_error\tmaximum_bf16_vs_rust_absolute_error\n",
    );
    let mut selection_evidence = String::from(
        "step\tinput_token\tposition\tselected_token\tf32_argmax\tbf16_argmax\tf32_argmax_logit\tf32_second_logit\tf32_margin\tf32_required_margin\tf32_classification\tbf16_margin\tbf16_required_margin\tbf16_classification\ttop20_rank_agreement\tcache_length\tdense_bytes_read\texpert_bytes_read\tloads\tevictions\tguard_router_ids\n",
    );
    let mut maximum_dense_layer_bytes = 0_u64;
    let mut maximum_inference_runtime_elements = 0_usize;
    let started = Instant::now();
    let mut generated = Vec::new();
    let initialization_seconds = runtime_started.elapsed().as_secs_f64();
    let mut step_seconds = Vec::with_capacity(6);
    let mut step_dense_bytes = Vec::with_capacity(6);
    let mut step_expert_bytes = Vec::with_capacity(6);
    let mut step_hits = Vec::with_capacity(6);
    let mut step_misses = Vec::with_capacity(6);
    let mut step_loads = Vec::with_capacity(6);
    let mut step_evictions = Vec::with_capacity(6);
    let mut expert_request_sequence = Vec::with_capacity(6 * 48 * 8);
    let mut ordered_expert_trace = Vec::with_capacity(6 * 48 * 8);
    let mut repeated_requests_within_token = 0_usize;
    if runtime_validation {
        println!("m5_2_runtime_phase phase=initialization");
        println!("m5_2_runtime_phase phase=prefill");
    }

    for (step, &token_id) in GENERATION_INPUT_TOKENS.iter().enumerate() {
        #[cfg(feature = "m5-3-compute-profiling")]
        crate::profiling::set_phase(if step < INPUT_IDS.len() {
            "prefill".to_owned()
        } else {
            format!("decode_{}", step - INPUT_IDS.len() + 1)
        });
        if runtime_validation && step == 4 {
            println!("m5_2_runtime_phase phase=decode");
        }
        let step_started = Instant::now();
        assert_eq!(cache.len(), step, "decode position before append");
        let dense_before = dense_bytes_read;
        let expert_before = store.metrics();
        let cache_prefix: Vec<_> = (0..48)
            .map(|layer| {
                let view = cache.layer(layer).expect("KV layer before append");
                (view.key.to_vec(), view.value.to_vec())
            })
            .collect();
        #[cfg(feature = "m5-3-compute-profiling")]
        let embedding_profile = crate::profiling::scope("embedding.lookup");
        let mut current = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        #[cfg(feature = "m5-3-compute-profiling")]
        drop(embedding_profile);
        let mut updates = Vec::with_capacity(48);
        let mut guard_ids = Vec::with_capacity(24);
        let mut layer47_checkpoints = None;
        let token_request_start = expert_request_sequence.len();

        for layer in 0..48 {
            #[cfg(feature = "m5-3-compute-profiling")]
            crate::profiling::set_layer(Some(layer));
            #[cfg(feature = "m5-3-compute-profiling")]
            let layer_profile = crate::profiling::scope(&format!("decoder.layer.{layer}"));
            let layer_dense_before = dense_bytes_read;
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            maximum_dense_layer_bytes =
                maximum_dense_layer_bytes.max(dense_bytes_read - layer_dense_before);
            #[cfg(feature = "m5-3-compute-profiling")]
            let input_norm_profile = crate::profiling::scope("layer.input_rms_norm");
            let input_norm = rms_norm(
                current.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} input norm: {error}"));
            #[cfg(feature = "m5-3-compute-profiling")]
            drop(input_norm_profile);
            #[cfg(feature = "m5-3-compute-profiling")]
            let attention_profile = crate::profiling::scope("attention.total");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("KV layer view"),
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} attention: {error}"));
            #[cfg(feature = "m5-3-compute-profiling")]
            drop(attention_profile);
            #[cfg(feature = "m5-3-compute-profiling")]
            let attention_residual_profile = crate::profiling::scope("attention.residual_add");
            let residual =
                elementwise_add(current.view(), attention.output.view()).unwrap_or_else(|error| {
                    panic!("step-{step} Layer-{layer} attention residual: {error}")
                });
            #[cfg(feature = "m5-3-compute-profiling")]
            drop(attention_residual_profile);
            #[cfg(feature = "m5-3-compute-profiling")]
            let post_norm_profile = crate::profiling::scope("layer.post_attention_rms_norm");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} post norm: {error}"));
            #[cfg(feature = "m5-3-compute-profiling")]
            drop(post_norm_profile);
            #[cfg(feature = "m5-3-compute-profiling")]
            let router_profile = crate::profiling::scope("router.top_k_selection");
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} router: {error}"));
            #[cfg(feature = "m5-3-compute-profiling")]
            drop(router_profile);
            let mut requested_experts = router.selected_experts.clone();
            requested_experts.sort_unstable();
            expert_request_sequence
                .extend(requested_experts.iter().map(|expert| layer * 128 + expert));
            if GENERATION_GUARD_LAYERS.contains(&layer) {
                guard_ids.extend_from_slice(&router.selected_experts);
            }
            let mut layer47_outputs = vec![0.0_f32; 8 * 2048];
            let moe = streaming_routed_experts_with_request_observer(
                post_norm.view(),
                &router,
                config,
                layer,
                &mut store,
                expert_layout,
                |layer, expert, _token, position, rank, observation: ExpertLoadObservation| {
                    ordered_expert_trace.push(OrderedExpertTraceRecord {
                        ordinal: ordered_expert_trace.len(),
                        step,
                        position,
                        layer,
                        rank,
                        expert,
                        payload_bytes: observation.payload_bytes,
                        cache_hit: observation.cache_hit,
                        loaded: observation.loaded,
                        evictions: observation.evictions,
                    });
                },
                |_, token, position, output| {
                    assert_eq!(token, 0, "single-token expert occurrence");
                    if layer == 47 {
                        layer47_outputs[position * 2048..(position + 1) * 2048]
                            .copy_from_slice(output);
                    }
                },
            )
            .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} experts: {error}"));
            #[cfg(feature = "m5-3-compute-profiling")]
            let final_residual_profile = crate::profiling::scope("layer.final_residual_add");
            let block = elementwise_add(residual.view(), moe.view())
                .unwrap_or_else(|error| panic!("step-{step} Layer-{layer} block: {error}"));
            #[cfg(feature = "m5-3-compute-profiling")]
            drop(final_residual_profile);
            if layer == 47 {
                let actual_experts = Tensor::new(TensorShape::new([8, 2048]), layer47_outputs)
                    .expect("Layer-47 expert output tensor");
                let actual_routing =
                    Tensor::new(TensorShape::new([8]), router.weights.data().to_vec())
                        .expect("Layer-47 routing tensor");
                layer47_checkpoints = Some((
                    actual_experts,
                    actual_routing,
                    tensor_row(&moe, 0),
                    tensor_row(&block, 0),
                ));
            }
            maximum_inference_runtime_elements = maximum_inference_runtime_elements.max(
                current.data().len()
                    + input_norm.data().len()
                    + attention.output.data().len()
                    + residual.data().len()
                    + post_norm.data().len()
                    + router.logits.data().len()
                    + router.weights.data().len()
                    + moe.data().len()
                    + block.data().len(),
            );
            updates.push((attention.key, attention.value));
            current = block;
            drop(weights);
            #[cfg(feature = "m5-3-compute-profiling")]
            drop(layer_profile);
        }
        #[cfg(feature = "m5-3-compute-profiling")]
        crate::profiling::set_layer(None);
        let token_requests = &expert_request_sequence[token_request_start..];
        repeated_requests_within_token +=
            token_requests.len() - token_requests.iter().copied().collect::<HashSet<_>>().len();

        #[cfg(feature = "m5-3-compute-profiling")]
        let final_norm_profile = crate::profiling::scope("final_norm");
        let normalized = rms_norm(
            current.view(),
            final_norm_weight.view(),
            config.rms_norm_epsilon(),
        )
        .unwrap_or_else(|error| panic!("step-{step} final RMSNorm: {error}"));
        #[cfg(feature = "m5-3-compute-profiling")]
        drop(final_norm_profile);
        #[cfg(feature = "m5-3-compute-profiling")]
        let lm_head_profile = crate::profiling::scope("lm_head");
        let logits = streaming_language_model_head(
            &mut payload,
            &final_plan,
            &normalized,
            &mut dense_bytes_read,
        );
        #[cfg(feature = "m5-3-compute-profiling")]
        drop(lm_head_profile);
        let selected = greedy_token(logits.view()).expect("finite greedy logits");
        let updates_view: Vec<_> = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect();
        #[cfg(feature = "m5-3-compute-profiling")]
        let cache_append_profile = crate::profiling::scope("cache.append");
        cache
            .append_token(&updates_view)
            .expect("transactional KV append");
        #[cfg(feature = "m5-3-compute-profiling")]
        drop(cache_append_profile);
        let inference_seconds = step_started.elapsed().as_secs_f64();
        let inference_metrics = store.metrics();
        step_seconds.push(inference_seconds);
        step_dense_bytes.push(dense_bytes_read - dense_before);
        step_expert_bytes.push(inference_metrics.bytes_read - expert_before.bytes_read);
        step_hits.push(inference_metrics.hits - expert_before.hits);
        step_misses.push(inference_metrics.misses - expert_before.misses);
        step_loads.push(inference_metrics.loads - expert_before.loads);
        step_evictions.push(inference_metrics.evictions - expert_before.evictions);

        assert_eq!(cache.len(), step + 1, "KV cache length after append");
        assert_eq!(cache.allocation_capacities(), allocation_capacities);
        for (layer, (prior_key, prior_value)) in cache_prefix.iter().enumerate() {
            let view = cache.layer(layer).expect("KV layer after append");
            assert_eq!(view.len, step + 1, "KV layer logical length");
            assert_eq!(view.key.len(), (step + 1) * 4 * 128, "KV key shape");
            assert_eq!(view.value.len(), (step + 1) * 4 * 128, "KV value shape");
            assert_eq!(
                &view.key[..prior_key.len()],
                prior_key,
                "KV key prefix overwrite"
            );
            assert_eq!(
                &view.value[..prior_value.len()],
                prior_value,
                "KV value prefix overwrite"
            );
        }

        let expected_guard_ids = checkpoint_ids(
            GENERATION_F32_CHECKPOINTS,
            &f32_plan,
            &format!("step{step}_guard_router_ids"),
        );
        assert_eq!(
            guard_ids, expected_guard_ids,
            "step-{step} guard router IDs"
        );
        assert_eq!(
            checkpoint_ids(
                GENERATION_F32_CHECKPOINTS,
                &f32_plan,
                &format!("step{step}_input_token"),
            ),
            [token_id],
            "frozen input token"
        );
        let (actual_experts, actual_routing, actual_moe, actual_block) =
            layer47_checkpoints.expect("Layer-47 checkpoints");
        let prefix = format!("step{step}_layer47");
        for (checkpoint, actual) in [
            ("expert_outputs", actual_experts),
            ("routing_weights", actual_routing),
            ("moe_output", actual_moe),
            ("block_output", actual_block),
        ] {
            let name = format!("{prefix}_{checkpoint}");
            let bf16 = checkpoint_f32(GENERATION_BF16_CHECKPOINTS, &bf16_plan, &name);
            let f32_control = checkpoint_f32(GENERATION_F32_CHECKPOINTS, &f32_plan, &name);
            let budget = Some(short_generation_budget(step, checkpoint));
            record_generation_checkpoint(
                &mut evidence,
                step,
                checkpoint,
                &actual,
                &bf16,
                &f32_control,
                budget,
            );
        }
        let bf16_norm = checkpoint_f32(
            GENERATION_BF16_CHECKPOINTS,
            &bf16_plan,
            &format!("step{step}_final_norm"),
        );
        let f32_norm = checkpoint_f32(
            GENERATION_F32_CHECKPOINTS,
            &f32_plan,
            &format!("step{step}_final_norm"),
        );
        let norm_budget = Some(short_generation_budget(step, "final_norm"));
        record_generation_checkpoint(
            &mut evidence,
            step,
            "final_norm",
            &tensor_row(&normalized, 0),
            &bf16_norm,
            &f32_norm,
            norm_budget,
        );
        let frozen_top = checkpoint_ids(
            GENERATION_F32_CHECKPOINTS,
            &f32_plan,
            &format!("step{step}_top20_ids"),
        );
        let (f32_logits, bf16_logits, logit_metrics) = if trace_only {
            let f32_values = checkpoint_f32(
                GENERATION_F32_CHECKPOINTS,
                &f32_plan,
                &format!("step{step}_top20_logits"),
            );
            let bf16_values = checkpoint_f32(
                GENERATION_BF16_CHECKPOINTS,
                &checkpoint_plan(GENERATION_BF16_CHECKPOINT_PLAN),
                &format!("step{step}_top20_logits"),
            );
            let mut f32_data = vec![f32::NEG_INFINITY; 151_936];
            let mut bf16_data = vec![f32::NEG_INFINITY; 151_936];
            for (index, &token) in frozen_top.iter().enumerate() {
                f32_data[token] = f32_values.data()[index];
                bf16_data[token] = bf16_values.data()[index];
            }
            (
                Tensor::new(TensorShape::new([1, 151_936]), f32_data)
                    .expect("trace-only F32 top-logit row"),
                Tensor::new(TensorShape::new([1, 151_936]), bf16_data)
                    .expect("trace-only BF16 top-logit row"),
                StageMetrics {
                    maximum_absolute_difference: 0.0,
                    maximum_relative_difference: 0.0,
                },
            )
        } else {
            let bf16_logits_flat = checkpoint_f32(
                &bf16_full_bytes,
                &bf16_full_plan,
                &format!("step{step}_logits"),
            );
            let f32_logits_flat = checkpoint_f32(
                &f32_full_bytes,
                &f32_full_plan,
                &format!("step{step}_logits"),
            );
            let bf16_logits =
                Tensor::new(TensorShape::new([1, 151_936]), bf16_logits_flat.into_data())
                    .expect("BF16 logit row");
            let f32_logits =
                Tensor::new(TensorShape::new([1, 151_936]), f32_logits_flat.into_data())
                    .expect("F32 logit row");
            let logits_budget = Some(short_generation_budget(step, "logits"));
            let metrics = record_generation_checkpoint(
                &mut evidence,
                step,
                "logits",
                &logits,
                &bf16_logits,
                &f32_logits,
                logits_budget,
            );
            (f32_logits, bf16_logits, metrics)
        };
        assert_eq!(
            logits.data().iter().filter(|value| value.is_nan()).count(),
            0,
            "step-{step} NaN logits"
        );
        assert_eq!(
            logits
                .data()
                .iter()
                .filter(|value| value.is_infinite())
                .count(),
            0,
            "step-{step} infinite logits"
        );

        let rust_top = deterministic_top_ids(&logits, 20);
        let f32_top = deterministic_top_ids(&f32_logits, 20);
        let bf16_top = deterministic_top_ids(&bf16_logits, 20);
        assert_eq!(f32_top, frozen_top, "frozen F32 top-20 IDs");
        let f32_argmax = f32_top[0];
        let bf16_argmax = bf16_top[0];
        let f32_margin = f32_logits.data()[f32_top[0]] - f32_logits.data()[f32_top[1]];
        let bf16_margin = bf16_logits.data()[bf16_top[0]] - bf16_logits.data()[bf16_top[1]];
        let bf16_metrics = if trace_only {
            StageMetrics {
                maximum_absolute_difference: 0.0,
                maximum_relative_difference: 0.0,
            }
        } else {
            measure_stage(&logits, &bf16_logits)
        };
        let f32_classification = token_selection_classification(
            selected,
            f32_argmax,
            f32_margin,
            logit_metrics.maximum_absolute_difference,
        );
        let bf16_classification = token_selection_classification(
            selected,
            bf16_argmax,
            bf16_margin,
            bf16_metrics.maximum_absolute_difference,
        );
        assert_ne!(
            f32_classification, "true_mismatch",
            "step-{step} true token mismatch"
        );
        assert_eq!(selected, f32_argmax, "step-{step} selected F32 token");
        if step == 3 || step == 4 {
            generated.push(selected);
            assert_eq!(
                selected,
                GENERATION_INPUT_TOKENS[step + 1],
                "generated token must drive next cached input"
            );
            let recompute = checkpoint_ids(
                GENERATION_F32_CHECKPOINTS,
                &f32_plan,
                &format!("step{step}_recompute_argmax"),
            );
            assert_eq!(recompute, [selected], "cached versus recomputed argmax");
        }

        let metrics = store.metrics();
        writeln!(
            selection_evidence,
            "{step}\t{token_id}\t{step}\t{selected}\t{f32_argmax}\t{bf16_argmax}\t{:.17e}\t{:.17e}\t{f32_margin:.17e}\t{:.17e}\t{f32_classification}\t{bf16_margin:.17e}\t{:.17e}\t{bf16_classification}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            f32_logits.data()[f32_top[0]],
            f32_logits.data()[f32_top[1]],
            2.0 * logit_metrics.maximum_absolute_difference,
            2.0 * bf16_metrics.maximum_absolute_difference,
            rust_top == f32_top,
            cache.len(),
            dense_bytes_read - dense_before,
            metrics.bytes_read - expert_before.bytes_read,
            metrics.loads - expert_before.loads,
            metrics.evictions - expert_before.evictions,
            comma_separated(&guard_ids),
        )
        .expect("write generation selection evidence");
        maximum_inference_runtime_elements = maximum_inference_runtime_elements
            .max(current.data().len() + normalized.data().len() + logits.data().len());
        println!(
            "short_generation step={step} input={token_id} selected={selected} elapsed_seconds={} cache_len={} f32_max_abs={} f32_margin={} classification={f32_classification}",
            inference_seconds,
            cache.len(),
            logit_metrics.maximum_absolute_difference,
            f32_margin,
        );
    }

    assert_eq!(generated, [1096, 374], "short greedy sequence");
    let metrics = store.metrics();
    #[cfg(feature = "m5-4-resident-dense")]
    let requires_cache_hit =
        env::var("COLIBRI_DENSE_RESIDENCY_MODE").as_deref() != Ok("resident_dense");
    #[cfg(not(feature = "m5-4-resident-dense"))]
    let requires_cache_hit = true;
    if cache_budget == 18_874_368 {
        assert_eq!(metrics.hits, 0, "one-expert cache expected zero hits");
    } else if requires_cache_hit {
        assert!(metrics.hits > 0, "larger cache should produce cache hits");
    }
    assert_eq!(metrics.loads, metrics.misses, "every miss loads one expert");
    assert_eq!(metrics.hits + metrics.misses, 6 * 48 * 8);
    assert!(metrics.evictions <= metrics.loads);
    assert!(metrics.resident_bytes <= cache_budget);
    assert!(metrics.peak_resident_bytes <= cache_budget);
    assert_eq!(cache.len(), 6);
    assert_eq!(cache.allocation_capacities(), allocation_capacities);
    assert_eq!(expert_request_sequence.len(), 6 * 48 * 8);
    assert_eq!(step_seconds.len(), 6);
    assert_eq!(step_dense_bytes.len(), 6);
    assert_eq!(step_expert_bytes.len(), 6);
    assert_eq!(step_hits.iter().sum::<u64>(), metrics.hits);
    assert_eq!(step_misses.iter().sum::<u64>(), metrics.misses);
    assert_eq!(step_loads.iter().sum::<u64>(), metrics.loads);
    assert_eq!(step_evictions.iter().sum::<u64>(), metrics.evictions);
    if runtime_validation {
        println!("m5_2_runtime_phase phase=complete");
    }

    let expert_payload_bytes = expert_layout.total_byte_length;
    assert_eq!(expert_payload_bytes, 18_874_368);
    assert_eq!(
        metrics.bytes_read,
        expert_payload_bytes as u64 * metrics.loads
    );
    let reuse = summarize_reuse_distances(&expert_request_sequence);
    assert_eq!(reuse.unique_requests + reuse.repeated_requests, 6 * 48 * 8);
    let repeated_requests_across_tokens = reuse
        .repeated_requests
        .checked_sub(repeated_requests_within_token)
        .expect("within-token repeats are included in all repeats");
    let requests_per_token = 48 * 8;
    let prompt_requests = &expert_request_sequence[..4 * requests_per_token];
    let decode_one_requests =
        &expert_request_sequence[4 * requests_per_token..5 * requests_per_token];
    let decode_two_requests = &expert_request_sequence[5 * requests_per_token..];
    let unique_expert_requests =
        |requests: &[usize]| requests.iter().copied().collect::<HashSet<_>>().len();
    let prompt_unique_experts = unique_expert_requests(prompt_requests);
    let decode_one_unique_experts = unique_expert_requests(decode_one_requests);
    let decode_two_unique_experts = unique_expert_requests(decode_two_requests);

    let embedding_row_bytes = 2048_u64 * 4;
    let lm_head_bytes = final_plan.tensors["lm_head.weight"].length as u64;
    let final_norm_bytes = final_plan.tensors["model.norm.weight"].length as u64;
    let layer_dense_bytes = step_dense_bytes[0] - embedding_row_bytes - lm_head_bytes;
    assert!(
        step_dense_bytes
            .iter()
            .all(|&value| value == layer_dense_bytes + embedding_row_bytes + lm_head_bytes)
    );
    assert_eq!(final_norm_bytes, 8_192);
    assert_eq!(
        dense_bytes_read,
        final_norm_bytes + step_dense_bytes.iter().sum::<u64>()
    );
    let unique_embedding_tokens = GENERATION_INPUT_TOKENS
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len() as u64;
    let unique_dense_bytes = final_norm_bytes
        + layer_dense_bytes
        + lm_head_bytes
        + unique_embedding_tokens * embedding_row_bytes;
    let unique_expert_bytes = reuse.unique_requests as u64 * expert_payload_bytes as u64;
    let total_artifact_bytes = dense_bytes_read + metrics.bytes_read;
    let useful_unique_bytes = unique_dense_bytes + unique_expert_bytes;
    let repeated_dense_bytes = dense_bytes_read - unique_dense_bytes;
    let repeated_expert_bytes = metrics.bytes_read - unique_expert_bytes;
    let repeated_artifact_bytes = total_artifact_bytes - useful_unique_bytes;

    assert_eq!(GENERATION_DENSE_READS_PER_TOKEN, 1_027);
    let dense_read_operations = 1 + 6 * GENERATION_DENSE_READS_PER_TOKEN;
    let expert_read_operations = metrics.loads;
    let artifact_read_operations = dense_read_operations + expert_read_operations;
    let artifact_file_opens = 1 + metrics.loads;
    assert_eq!(dense_read_operations, 6_163);
    assert_eq!(
        artifact_read_operations,
        dense_read_operations + metrics.loads
    );
    assert_eq!(artifact_file_opens, 1 + metrics.loads);

    let checkpoint_static_bytes = GENERATION_BF16_CHECKPOINTS.len()
        + GENERATION_F32_CHECKPOINTS.len()
        + bf16_full_bytes.len()
        + f32_full_bytes.len();
    let dense_buffer_bytes = usize::try_from(maximum_dense_layer_bytes)
        .expect("dense layer bytes fit usize")
        + 256 * 2048 * 4;
    let decoded_expert_buffer_bytes = expert_layout.total_byte_length;
    let inference_tensor_bytes = maximum_inference_runtime_elements * 4;
    let temporary_validation_buffer_bytes = checkpoint_static_bytes + 2 * 151_936 * 4;
    let modeled_peak_explicit_bytes = dense_buffer_bytes
        + decoded_expert_buffer_bytes
        + metrics.peak_resident_bytes
        + cache.byte_size()
        + inference_tensor_bytes
        + temporary_validation_buffer_bytes;
    if trace_only && cache_budget == 18_874_368 {
        assert_eq!(modeled_peak_explicit_bytes, 120_529_096);
    } else if !trace_only && cache_budget == 18_874_368 {
        assert_eq!(modeled_peak_explicit_bytes, 127_823_000);
    }
    evidence.push_str(&selection_evidence);
    writeln!(
        evidence,
        "resources\tNA\tdense_bytes_read={dense_bytes_read}\texpert_bytes_read={}\ttotal_artifact_bytes_read={}\thits={}\tmisses={}\tloads={}\tevictions={}\tpeak_expert_resident_bytes={}\tkv_cache_bytes={}\tmodeled_peak_explicit_bytes={modeled_peak_explicit_bytes}",
        metrics.bytes_read,
        dense_bytes_read + metrics.bytes_read,
        metrics.hits,
        metrics.misses,
        metrics.loads,
        metrics.evictions,
        metrics.peak_resident_bytes,
        cache.byte_size(),
    )
    .expect("write generation resource evidence");
    atomic_diagnostic(
        &diagnostic_root.join("m4.2-04-rust-short-generation-evidence-v1.tsv"),
        evidence.as_bytes(),
    );
    #[cfg(feature = "m5-4-resident-dense")]
    let resident_dense_metrics = payload.metrics();
    #[cfg(feature = "m5-4-resident-dense")]
    let dense_residency_mode =
        env::var("COLIBRI_DENSE_RESIDENCY_MODE").unwrap_or_else(|_| "streamed_dense".to_owned());
    if let Some(metrics_path) = metrics_output {
        let prefill_seconds = step_seconds[..4].iter().sum::<f64>();
        let decode_one_seconds = step_seconds[4];
        let decode_two_seconds = step_seconds[5];
        let decode_seconds = decode_one_seconds + decode_two_seconds;
        let inference_seconds = initialization_seconds + step_seconds.iter().sum::<f64>();
        let prefill_dense_bytes = step_dense_bytes[..4].iter().sum::<u64>();
        let prefill_expert_bytes = step_expert_bytes[..4].iter().sum::<u64>();
        let mut baseline = String::from("record\tphase\tmetric\tvalue\tunit\n");
        let mut record = |kind: &str, phase: &str, metric: &str, value: String, unit: &str| {
            writeln!(baseline, "{kind}\t{phase}\t{metric}\t{value}\t{unit}")
                .expect("write baseline metric");
        };

        record(
            "schema",
            "total",
            "schema_version",
            "1".to_owned(),
            "integer",
        );
        record(
            "configuration",
            "total",
            "filesystem_cache_assumption",
            filesystem_cache_assumption,
            "label",
        );
        #[cfg(feature = "m5-4-resident-dense")]
        {
            record(
                "configuration",
                "total",
                "dense_residency_mode",
                dense_residency_mode,
                "label",
            );
            record(
                "io",
                "initialization",
                "resident_dense_bytes",
                resident_dense_metrics.resident_dense_bytes.to_string(),
                "bytes",
            );
            record(
                "io",
                "initialization",
                "dense_payload_initialization_read_bytes",
                resident_dense_metrics.initialization_bytes_read.to_string(),
                "logical_bytes_not_physical_io",
            );
            record(
                "io",
                "total",
                "dense_execution_access_bytes",
                resident_dense_metrics.execution_bytes_accessed.to_string(),
                "logical_bytes",
            );
            if resident_dense_metrics.resident_dense_bytes > 0 {
                let total_budget = env::var("COLIBRI_TOTAL_RAM_BUDGET_BYTES")
                    .expect("resident dense requires total budget metric")
                    .parse::<usize>()
                    .expect("valid resident dense total budget metric");
                let accounted_peak = resident_dense_metrics
                    .resident_dense_bytes
                    .checked_add(FIXED_RUNTIME_MEMORY_BYTES)
                    .and_then(|value| value.checked_add(metrics.peak_resident_bytes))
                    .expect("resident dense accounted peak overflow");
                assert!(
                    accounted_peak <= total_budget,
                    "resident dense accounted peak exceeds total budget"
                );
                record(
                    "memory",
                    "total",
                    "configured_total_ram_budget_bytes",
                    total_budget.to_string(),
                    "bytes",
                );
                record(
                    "memory",
                    "total",
                    "accounted_peak_resident_bytes",
                    accounted_peak.to_string(),
                    "bytes",
                );
            }
        }
        record(
            "configuration",
            "total",
            "input_token_ids",
            "9707,11,1879,0".to_owned(),
            "token_ids",
        );
        record(
            "correctness",
            "total",
            "generated_token_ids",
            "1096,374".to_owned(),
            "token_ids",
        );
        record(
            "correctness",
            "total",
            "f32_classifications",
            "exact_match_safe,exact_match_safe".to_owned(),
            "labels",
        );
        record(
            "timing",
            "initialization",
            "wall_seconds",
            format!("{initialization_seconds:.17e}"),
            "seconds",
        );
        record(
            "timing",
            "prefill",
            "wall_seconds",
            format!("{prefill_seconds:.17e}"),
            "seconds",
        );
        record(
            "timing",
            "decode_1",
            "wall_seconds",
            format!("{decode_one_seconds:.17e}"),
            "seconds",
        );
        record(
            "timing",
            "decode_2",
            "wall_seconds",
            format!("{decode_two_seconds:.17e}"),
            "seconds",
        );
        record(
            "timing",
            "decode",
            "average_seconds_per_token",
            format!("{:.17e}", decode_seconds / 2.0),
            "seconds_per_token",
        );
        record(
            "timing",
            "prefill",
            "tokens_per_second",
            format!("{:.17e}", 4.0 / prefill_seconds),
            "tokens_per_second",
        );
        record(
            "timing",
            "decode",
            "tokens_per_second",
            format!("{:.17e}", 2.0 / decode_seconds),
            "tokens_per_second",
        );
        record(
            "timing",
            "total",
            "inference_wall_seconds",
            format!("{inference_seconds:.17e}"),
            "seconds",
        );

        let phases = [
            (
                "initialization",
                final_norm_bytes,
                0_u64,
                1_u64,
                0_u64,
                1_u64,
                1_u64,
            ),
            (
                "prefill",
                prefill_dense_bytes,
                prefill_expert_bytes,
                4 * GENERATION_DENSE_READS_PER_TOKEN,
                4 * GENERATION_EXPERT_READS_PER_TOKEN,
                4 * GENERATION_EXPERT_READS_PER_TOKEN,
                49_u64,
            ),
            (
                "decode_1",
                step_dense_bytes[4],
                step_expert_bytes[4],
                GENERATION_DENSE_READS_PER_TOKEN,
                GENERATION_EXPERT_READS_PER_TOKEN,
                GENERATION_EXPERT_READS_PER_TOKEN,
                49_u64,
            ),
            (
                "decode_2",
                step_dense_bytes[5],
                step_expert_bytes[5],
                GENERATION_DENSE_READS_PER_TOKEN,
                GENERATION_EXPERT_READS_PER_TOKEN,
                GENERATION_EXPERT_READS_PER_TOKEN,
                49_u64,
            ),
            (
                "total",
                dense_bytes_read,
                metrics.bytes_read,
                dense_read_operations,
                expert_read_operations,
                artifact_file_opens,
                49_u64,
            ),
        ];
        for (phase, dense, expert, dense_reads, expert_reads, file_opens, unique_files) in phases {
            record(
                "io",
                phase,
                "dense_bytes_requested_read",
                dense.to_string(),
                "logical_bytes",
            );
            record(
                "io",
                phase,
                "expert_bytes_requested_read",
                expert.to_string(),
                "logical_bytes",
            );
            record(
                "io",
                phase,
                "total_artifact_bytes_read",
                (dense + expert).to_string(),
                "logical_bytes",
            );
            record(
                "io",
                phase,
                "dense_read_operations",
                dense_reads.to_string(),
                "operations",
            );
            record(
                "io",
                phase,
                "expert_read_operations",
                expert_reads.to_string(),
                "operations",
            );
            record(
                "io",
                phase,
                "artifact_read_operations",
                (dense_reads + expert_reads).to_string(),
                "operations",
            );
            record(
                "io",
                phase,
                "file_open_count",
                file_opens.to_string(),
                "opens",
            );
            record(
                "io",
                phase,
                "unique_files_accessed",
                unique_files.to_string(),
                "files",
            );
            record(
                "io",
                phase,
                "artifact_manifest_metadata_bytes",
                "0".to_owned(),
                "logical_bytes",
            );
        }
        record(
            "io",
            "decode_1",
            "bytes_per_generated_token",
            (step_dense_bytes[4] + step_expert_bytes[4]).to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "decode_2",
            "bytes_per_generated_token",
            (step_dense_bytes[5] + step_expert_bytes[5]).to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "end_to_end_bytes_per_generated_token",
            (total_artifact_bytes / 2).to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "unique_dense_payload_bytes",
            unique_dense_bytes.to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "unique_expert_payload_bytes",
            unique_expert_bytes.to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "useful_unique_payload_bytes",
            useful_unique_bytes.to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "repeated_dense_bytes",
            repeated_dense_bytes.to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "repeated_expert_bytes",
            repeated_expert_bytes.to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "repeated_artifact_bytes",
            repeated_artifact_bytes.to_string(),
            "logical_bytes",
        );
        record(
            "io",
            "total",
            "read_amplification",
            format!(
                "{:.17e}",
                reporting_ratio(total_artifact_bytes, useful_unique_bytes)
            ),
            "ratio",
        );
        record(
            "io",
            "total",
            "expert_read_amplification",
            format!(
                "{:.17e}",
                reporting_ratio(metrics.bytes_read, unique_expert_bytes)
            ),
            "ratio",
        );

        record(
            "cache",
            "total",
            "configured_byte_budget",
            cache_budget.to_string(),
            "bytes",
        );
        record(
            "cache",
            "total",
            "expert_payload_bytes",
            expert_payload_bytes.to_string(),
            "bytes",
        );
        record(
            "cache",
            "total",
            "theoretical_expert_capacity",
            (cache_budget / expert_payload_bytes).to_string(),
            "experts",
        );
        record(
            "cache",
            "total",
            "peak_resident_expert_count",
            metrics.peak_entry_count.to_string(),
            "experts",
        );
        record(
            "cache",
            "total",
            "expert_occurrences",
            expert_request_sequence.len().to_string(),
            "requests",
        );
        record(
            "cache",
            "prefill",
            "unique_layer_expert_requests",
            prompt_unique_experts.to_string(),
            "keys",
        );
        record(
            "cache",
            "decode_1",
            "unique_layer_expert_requests",
            decode_one_unique_experts.to_string(),
            "keys",
        );
        record(
            "cache",
            "decode_2",
            "unique_layer_expert_requests",
            decode_two_unique_experts.to_string(),
            "keys",
        );
        record(
            "cache",
            "total",
            "unique_layer_expert_requests",
            reuse.unique_requests.to_string(),
            "keys",
        );
        record(
            "cache",
            "total",
            "hits",
            metrics.hits.to_string(),
            "requests",
        );
        record(
            "cache",
            "total",
            "misses",
            metrics.misses.to_string(),
            "requests",
        );
        record(
            "cache",
            "total",
            "loads",
            metrics.loads.to_string(),
            "loads",
        );
        record(
            "cache",
            "total",
            "evictions",
            metrics.evictions.to_string(),
            "evictions",
        );
        record(
            "cache",
            "total",
            "hit_rate",
            format!(
                "{:.17e}",
                reporting_ratio(metrics.hits, metrics.hits + metrics.misses)
            ),
            "ratio",
        );
        record(
            "cache",
            "total",
            "repeated_requests_within_token",
            repeated_requests_within_token.to_string(),
            "requests",
        );
        record(
            "cache",
            "total",
            "repeated_requests_across_tokens",
            repeated_requests_across_tokens.to_string(),
            "requests",
        );
        record(
            "cache",
            "total",
            "reuse_distance_minimum",
            reuse.minimum_distance.to_string(),
            "requests",
        );
        record(
            "cache",
            "total",
            "reuse_distance_median",
            reuse.median_distance.to_string(),
            "requests",
        );
        record(
            "cache",
            "total",
            "reuse_distance_maximum",
            reuse.maximum_distance.to_string(),
            "requests",
        );
        record(
            "cache",
            "total",
            "reuse_distance_at_most_384",
            reuse.distance_at_most_384.to_string(),
            "reuses",
        );
        record(
            "cache",
            "total",
            "reuse_distance_385_through_768",
            reuse.distance_385_through_768.to_string(),
            "reuses",
        );
        record(
            "cache",
            "total",
            "reuse_distance_above_768",
            reuse.distance_above_768.to_string(),
            "reuses",
        );

        record(
            "memory",
            "total",
            "modeled_explicit_tensor_bytes",
            modeled_peak_explicit_bytes.to_string(),
            "bytes",
        );
        record(
            "memory",
            "total",
            "dense_buffer_bytes",
            dense_buffer_bytes.to_string(),
            "bytes",
        );
        record(
            "memory",
            "total",
            "decoded_expert_buffer_bytes",
            decoded_expert_buffer_bytes.to_string(),
            "bytes",
        );
        record(
            "memory",
            "total",
            "expert_cache_resident_bytes",
            metrics.peak_resident_bytes.to_string(),
            "bytes",
        );
        record(
            "memory",
            "total",
            "kv_cache_bytes",
            cache.byte_size().to_string(),
            "bytes",
        );
        record(
            "memory",
            "total",
            "inference_tensor_bytes",
            inference_tensor_bytes.to_string(),
            "bytes",
        );
        record(
            "memory",
            "total",
            "temporary_validation_buffer_bytes",
            temporary_validation_buffer_bytes.to_string(),
            "bytes",
        );
        record(
            "kv_cache",
            "total",
            "layer_count",
            "48".to_owned(),
            "layers",
        );
        record(
            "kv_cache",
            "total",
            "configured_capacity",
            "6".to_owned(),
            "tokens",
        );
        record(
            "kv_cache",
            "total",
            "final_sequence_length",
            cache.len().to_string(),
            "tokens",
        );
        record(
            "kv_cache",
            "total",
            "key_shape_per_layer",
            "6,4,128".to_owned(),
            "dimensions",
        );
        record(
            "kv_cache",
            "total",
            "value_shape_per_layer",
            "6,4,128".to_owned(),
            "dimensions",
        );
        record(
            "kv_cache",
            "total",
            "allocated_bytes",
            cache.byte_size().to_string(),
            "bytes",
        );
        record(
            "kv_cache",
            "total",
            "payload_allocation_count",
            "96".to_owned(),
            "allocations",
        );
        record(
            "kv_cache",
            "total",
            "allocation_growth_during_decode",
            "false".to_owned(),
            "boolean",
        );
        record(
            "kv_cache",
            "total",
            "previous_position_overwrite",
            "false".to_owned(),
            "boolean",
        );
        atomic_diagnostic(&metrics_path, baseline.as_bytes());
    }
    if let Some(trace_path) = env::var_os("COLIBRI_EXPERT_TRACE_OUTPUT") {
        let instrumentation_commit = env::var("COLIBRI_TRACE_INSTRUMENTATION_COMMIT")
            .expect("COLIBRI_TRACE_INSTRUMENTATION_COMMIT must identify the trace code");
        write_ordered_expert_trace(
            Path::new(&trace_path),
            &ordered_expert_trace,
            &instrumentation_commit,
            cache_budget,
            env::var_os("COLIBRI_RUNTIME_VALIDATION").is_some(),
        );
    }
    #[cfg(feature = "m5-3-compute-profiling")]
    drop(model_profile);
    #[cfg(feature = "m5-3-compute-profiling")]
    let profile_snapshot = crate::profiling::finish(profiling_session);
    #[cfg(feature = "m5-3-compute-profiling")]
    if let Some(profile_output) = env::var_os("COLIBRI_COMPUTE_PROFILE_OUTPUT") {
        crate::profiling::write_json(
            Path::new(&profile_output),
            &profile_snapshot,
            "tier_a_control",
            cache_budget,
            &GENERATION_INPUT_TOKENS,
            &generated,
        );
    }
    #[cfg(any(
        feature = "m5-3-reusable-buffer",
        feature = "m5-3-compute-profiling",
        feature = "m5-3-mmap"
    ))]
    if let Some(storage_output) = env::var_os("COLIBRI_M5_3_STORAGE_METRICS_OUTPUT") {
        m5_2_trace_capture::write_m5_3_storage_metrics(Path::new(&storage_output), &store, metrics);
    }
    println!(
        "short_generation_complete elapsed_seconds={} generated={generated:?} dense_bytes_read={dense_bytes_read} expert_metrics={metrics:?} kv_cache_bytes={} modeled_peak_explicit_bytes={modeled_peak_explicit_bytes}",
        started.elapsed().as_secs_f64(),
        cache.byte_size(),
    );
}

fn write_ordered_expert_trace(
    path: &Path,
    records: &[OrderedExpertTraceRecord],
    instrumentation_commit: &str,
    cache_budget: usize,
    runtime_validation: bool,
) {
    let trace_budget = if runtime_validation {
        cache_budget
    } else {
        18_874_368
    };
    let trace_policy = if runtime_validation {
        "strict_global_lru"
    } else {
        "strict_lru"
    };
    let mut output = String::with_capacity(records.len() * 220 + 1800);
    writeln!(
        output,
        "{{\"schema\":\"colibri-qwen3-moe-m5.1-00-ordered-expert-trace-v1\",\"schema_version\":1,\"trace_id\":\"m4-tier-a-short-generation-ordered-expert-requests-v1\",\"classification\":\"M5 measurement supplement replaying the frozen M4 baseline configuration\",\"baseline_id\":\"qwen3-30b-a3b-colibri-f32-windows-x64-v1\",\"release_id\":\"colibri-lite-rs-m4-qwen3-30b-a3b-f32-v1\",\"release_tag\":\"m4-full-qwen3-baseline-v1\",\"baseline_runtime_source_commit\":\"80099f05246a4450ded6f42baf6b8db5a4b2e623\",\"trace_instrumentation_commit\":\"{instrumentation_commit}\",\"model_repository\":\"Qwen/Qwen3-30B-A3B\",\"model_revision\":\"ad44e777bcd18fa416d9da3bd8f70d33ebb85d39\",\"canonical_artifact_root_sha256\":\"f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2\",\"tokenizer_identity\":\"Qwen2Tokenizer:a66c5b39331656b1a3befd2d695265f15bdc5f16226fbbf7794bfb5ae9220c5e\",\"input_token_ids\":[9707,11,1879,0],\"expected_generated_token_ids\":[1096,374],\"cache_budget_bytes\":{trace_budget},\"cache_policy\":\"{trace_policy}\",\"runtime_configuration\":{{\"compute_dtype\":\"F32\",\"kv_cache_capacity\":6,\"threads\":8,\"target\":\"x86_64-pc-windows-msvc\",\"build_profile\":\"release\"}},\"requested_trace_count\":{},\"serialization\":\"UTF-8 JSON object with fixed header and record field order, compact separators, trailing newline, no timestamp or local path\",\"records\":[",
        records.len()
    )
    .expect("write trace header");
    for (index, record) in records.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        let phase = if record.step < 4 { "prefill" } else { "decode" };
        let decode_step = if record.step < 4 {
            "null".to_owned()
        } else {
            (record.step - 4).to_string()
        };
        write!(
            output,
            "{{\"global_ordinal\":{},\"phase\":\"{}\",\"generation_step\":{},\"decode_step\":{},\"input_token_id\":{},\"absolute_position\":{},\"layer_index\":{},\"selected_expert_rank\":{},\"expert_id\":{},\"layer_expert_key\":\"layer.{}.expert.{}\",\"payload_bytes\":{},\"cache_hit\":{},\"loaded\":{},\"evictions_caused\":{}}}",
            record.ordinal,
            phase,
            record.step,
            decode_step,
            GENERATION_INPUT_TOKENS[record.step],
            record.step,
            record.layer,
            record.rank,
            record.expert,
            record.layer,
            record.expert,
            record.payload_bytes,
            record.cache_hit,
            record.loaded,
            record.evictions,
        )
        .expect("write trace record");
    }
    output.push_str("]}\n");
    fs::write(path, output).expect("write ordered expert trace");
}

#[derive(Debug, Default)]
struct TierBReference {
    name: String,
    token_ids: Vec<usize>,
    final_norm_indices: Vec<usize>,
    final_norm_values: Vec<f32>,
    final_norm_sha256: String,
    guard_ids: HashMap<usize, Vec<usize>>,
    fixed_logit_indices: Vec<usize>,
    fixed_logits: Vec<f32>,
    top20_ids: Vec<usize>,
    top20_logits: Vec<f32>,
    argmax: usize,
    argmax_logit: f32,
    second_logit: f32,
    margin: f32,
    nan_count: usize,
    positive_infinity_count: usize,
    negative_infinity_count: usize,
}

fn parse_usize_list(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|item| item.parse().expect("Tier B integer"))
        .collect()
}

fn parse_f32_list(value: &str) -> Vec<f32> {
    value
        .split(',')
        .map(|item| item.parse().expect("Tier B F32"))
        .collect()
}

fn tier_b_references() -> Vec<TierBReference> {
    let mut fixtures = Vec::<TierBReference>::new();
    for line in TIER_B_F32_REFERENCE.lines().skip(1) {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "Tier B TSV field count");
        if fields[0] == "fixture" {
            fixtures.push(TierBReference {
                name: fields[1].to_owned(),
                token_ids: parse_usize_list(fields[2]),
                ..TierBReference::default()
            });
            continue;
        }
        let fixture = fixtures
            .iter_mut()
            .find(|fixture| fixture.name == fields[1])
            .expect("Tier B record follows fixture");
        let values: Vec<_> = fields[2].split(';').collect();
        match fields[0] {
            "final_norm" => {
                assert_eq!(values.len(), 3);
                fixture.final_norm_indices = parse_usize_list(values[0]);
                fixture.final_norm_values = parse_f32_list(values[1]);
                values[2].clone_into(&mut fixture.final_norm_sha256);
            }
            guard if guard.starts_with("guard_ids_") => {
                let layer = guard["guard_ids_".len()..]
                    .parse()
                    .expect("Tier B guard layer");
                assert!(
                    fixture
                        .guard_ids
                        .insert(layer, parse_usize_list(values[0]))
                        .is_none(),
                    "duplicate Tier B guard layer"
                );
            }
            "logits" => {
                assert_eq!(values.len(), 11);
                fixture.fixed_logit_indices = parse_usize_list(values[0]);
                fixture.fixed_logits = parse_f32_list(values[1]);
                fixture.top20_ids = parse_usize_list(values[2]);
                fixture.top20_logits = parse_f32_list(values[3]);
                fixture.argmax = values[4].parse().expect("Tier B argmax");
                fixture.argmax_logit = values[5].parse().expect("Tier B argmax logit");
                fixture.second_logit = values[6].parse().expect("Tier B second logit");
                fixture.margin = values[7].parse().expect("Tier B top-1 margin");
                fixture.nan_count = values[8].parse().expect("Tier B NaN count");
                fixture.positive_infinity_count =
                    values[9].parse().expect("Tier B positive infinity count");
                fixture.negative_infinity_count =
                    values[10].parse().expect("Tier B negative infinity count");
            }
            record => panic!("unsupported Tier B record {record}"),
        }
    }
    assert_eq!(fixtures.len(), 6, "Tier B fixture count");
    for fixture in &fixtures {
        assert_eq!(fixture.final_norm_indices.len(), 5);
        assert_eq!(fixture.final_norm_values.len(), 5);
        assert_eq!(fixture.final_norm_sha256.len(), 64);
        assert_eq!(fixture.guard_ids.len(), 3);
        assert_eq!(fixture.fixed_logit_indices.len(), 10);
        assert_eq!(fixture.fixed_logits.len(), 10);
        assert_eq!(fixture.top20_ids.len(), 20);
        assert_eq!(fixture.top20_logits.len(), 20);
        assert_eq!(fixture.argmax, fixture.top20_ids[0]);
        assert_eq!(fixture.argmax_logit, fixture.top20_logits[0]);
        assert_eq!(fixture.second_logit, fixture.top20_logits[1]);
        assert_eq!(fixture.margin, fixture.argmax_logit - fixture.second_logit);
        assert_eq!(fixture.nan_count, 0);
        assert_eq!(fixture.positive_infinity_count, 0);
        assert_eq!(fixture.negative_infinity_count, 0);
    }
    fixtures
}

#[test]
fn m4_3_01_tier_b_reference_schema_and_budgets_are_frozen() {
    let fixtures = tier_b_references();
    assert_eq!(
        fixtures
            .iter()
            .map(|fixture| fixture.name.as_str())
            .collect::<Vec<_>>(),
        [
            "single_low_token",
            "short_english",
            "short_thai",
            "code_newline",
            "repeated_pattern",
            "special_token",
        ]
    );
    assert_eq!(
        fixtures
            .iter()
            .map(|fixture| fixture.token_ids.len())
            .sum::<usize>(),
        11
    );
    for fixture in fixtures {
        assert!(tier_b_fixture_budget(&fixture.name, "final_norm") > 0.0);
        assert!(tier_b_fixture_budget(&fixture.name, "logits") > 0.0);
        assert!(fixture.margin > 0.0);
        assert_eq!(fixture.top20_ids[0], fixture.argmax);
        assert_eq!(
            fixture.guard_ids.keys().copied().collect::<HashSet<_>>(),
            [0, 24, 47].into()
        );
    }
}

fn maximum_indexed_difference(actual: &[f32], indices: &[usize], expected: &[f32]) -> f32 {
    assert_eq!(indices.len(), expected.len());
    indices
        .iter()
        .zip(expected)
        .map(|(&index, &expected)| (actual[index] - expected).abs())
        .fold(0.0_f32, f32::max)
}

fn f32_little_endian_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256Hasher::new();
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            write!(output, "{byte:02x}").expect("write SHA-256 hex");
            output
        })
}

fn f32_little_endian_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

const M6_3_04_GROUP_SIZE: usize = 128;
const M6_3_04_GATE_UP_ROWS: usize = 768;
const M6_3_04_HIDDEN_SIZE: usize = 2048;
const M6_3_04_EXPERT_COUNT: usize = 128;

#[derive(Debug)]
struct M6_3_04OwnedExpert {
    gate_values: Vec<i8>,
    gate_scales: Vec<f32>,
    up_values: Vec<i8>,
    up_scales: Vec<f32>,
    down_values: Vec<i8>,
    down_scales: Vec<f32>,
}

impl M6_3_04OwnedExpert {
    fn view(&self) -> Layer0QuantizedExpert<'_> {
        Layer0QuantizedExpert::new(
            Group128Projection::new(
                &self.gate_values,
                &self.gate_scales,
                M6_3_04_GATE_UP_ROWS,
                M6_3_04_HIDDEN_SIZE,
            )
            .expect("validated Layer-0 gate projection"),
            Group128Projection::new(
                &self.up_values,
                &self.up_scales,
                M6_3_04_GATE_UP_ROWS,
                M6_3_04_HIDDEN_SIZE,
            )
            .expect("validated Layer-0 up projection"),
            Group128Projection::new(
                &self.down_values,
                &self.down_scales,
                M6_3_04_HIDDEN_SIZE,
                M6_3_04_GATE_UP_ROWS,
            )
            .expect("validated Layer-0 down projection"),
        )
        .expect("validated Layer-0 expert")
    }
}

fn m6_3_04_projection_bytes(rows: usize, columns: usize) -> usize {
    rows * columns + rows * (columns / M6_3_04_GROUP_SIZE) * size_of::<f32>()
}

fn m6_3_04_expert_bytes() -> usize {
    m6_3_04_projection_bytes(M6_3_04_GATE_UP_ROWS, M6_3_04_HIDDEN_SIZE) * 2
        + m6_3_04_projection_bytes(M6_3_04_HIDDEN_SIZE, M6_3_04_GATE_UP_ROWS)
}

fn m6_3_04_sha256_hex(hasher: Sha256Hasher) -> String {
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            write!(output, "{byte:02x}").expect("write SHA-256 hex");
            output
        })
}

#[allow(clippy::cast_possible_truncation)]
fn m6_3_04_quantize(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    let quantized = (value / scale).round().clamp(-127.0, 127.0);
    assert!(quantized.is_finite());
    quantized as i8
}

fn m6_3_04_write_projection(
    source: &mut File,
    source_offset: u64,
    rows: usize,
    columns: usize,
    output: &mut File,
    source_hasher: &mut Sha256Hasher,
    output_hasher: &mut Sha256Hasher,
) {
    assert_eq!(columns % M6_3_04_GROUP_SIZE, 0);
    source
        .seek(SeekFrom::Start(source_offset))
        .expect("seek Layer-0 F32 projection");
    let mut row_bytes = vec![0_u8; columns * size_of::<f32>()];
    let mut scales = Vec::with_capacity(rows * (columns / M6_3_04_GROUP_SIZE));
    for _ in 0..rows {
        source
            .read_exact(&mut row_bytes)
            .expect("read Layer-0 F32 projection row");
        source_hasher.update(&row_bytes);
        let values = row_bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 source value")))
            .collect::<Vec<_>>();
        assert!(values.iter().all(|value| value.is_finite()));
        let mut quantized = Vec::with_capacity(columns);
        for group in values.chunks_exact(M6_3_04_GROUP_SIZE) {
            let maximum = group
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            let scale = maximum / 127.0;
            scales.push(scale);
            quantized.extend(group.iter().map(|value| m6_3_04_quantize(*value, scale)));
        }
        let bytes = quantized
            .iter()
            .map(|value| value.to_ne_bytes()[0])
            .collect::<Vec<_>>();
        output.write_all(&bytes).expect("write group-128 values");
        output_hasher.update(&bytes);
    }
    for scale in scales {
        let bytes = scale.to_le_bytes();
        output.write_all(&bytes).expect("write group-128 scale");
        output_hasher.update(&bytes);
    }
}

fn m6_3_04_convert_layer0_group128(source_path: &Path, output_path: &Path) -> (String, String) {
    let layout = PackedExpertLayout::for_config(
        PINNED_QWEN3_30B_A3B_CONFIG
            .map_to_f32_runtime()
            .expect("pinned configuration")
            .runtime_config(),
    );
    assert_eq!(
        layout.gate_length,
        M6_3_04_GATE_UP_ROWS * M6_3_04_HIDDEN_SIZE * 4
    );
    assert_eq!(layout.up_length, layout.gate_length);
    assert_eq!(layout.down_length, layout.gate_length);
    let mut source = File::open(source_path).expect("open canonical Layer-0 experts");
    let mut output = File::create(output_path).expect("create incomplete group-128 artifact");
    let mut source_hasher = Sha256Hasher::new();
    let mut output_hasher = Sha256Hasher::new();
    for expert in 0..M6_3_04_EXPERT_COUNT {
        let expert_offset =
            u64::try_from(expert * layout.total_byte_length).expect("expert offset");
        m6_3_04_write_projection(
            &mut source,
            expert_offset + u64::try_from(layout.gate_offset).expect("gate offset"),
            M6_3_04_GATE_UP_ROWS,
            M6_3_04_HIDDEN_SIZE,
            &mut output,
            &mut source_hasher,
            &mut output_hasher,
        );
        m6_3_04_write_projection(
            &mut source,
            expert_offset + u64::try_from(layout.up_offset).expect("up offset"),
            M6_3_04_GATE_UP_ROWS,
            M6_3_04_HIDDEN_SIZE,
            &mut output,
            &mut source_hasher,
            &mut output_hasher,
        );
        m6_3_04_write_projection(
            &mut source,
            expert_offset + u64::try_from(layout.down_offset).expect("down offset"),
            M6_3_04_HIDDEN_SIZE,
            M6_3_04_GATE_UP_ROWS,
            &mut output,
            &mut source_hasher,
            &mut output_hasher,
        );
    }
    output
        .sync_all()
        .expect("sync incomplete group-128 artifact");
    assert_eq!(
        output.metadata().expect("group-128 metadata").len(),
        u64::try_from(m6_3_04_expert_bytes() * M6_3_04_EXPERT_COUNT).expect("artifact length")
    );
    (
        m6_3_04_sha256_hex(source_hasher),
        m6_3_04_sha256_hex(output_hasher),
    )
}

fn m6_3_04_read_exact(file: &mut File, offset: u64, length: usize) -> Vec<u8> {
    file.seek(SeekFrom::Start(offset))
        .expect("seek group-128 artifact");
    let mut bytes = vec![0_u8; length];
    file.read_exact(&mut bytes)
        .expect("read group-128 artifact");
    bytes
}

fn m6_3_04_load_expert(file: &mut File, expert: usize) -> M6_3_04OwnedExpert {
    assert!(expert < M6_3_04_EXPERT_COUNT);
    let base = u64::try_from(expert * m6_3_04_expert_bytes()).expect("expert base");
    let gate_values_len = M6_3_04_GATE_UP_ROWS * M6_3_04_HIDDEN_SIZE;
    let gate_scales_len = M6_3_04_GATE_UP_ROWS * (M6_3_04_HIDDEN_SIZE / M6_3_04_GROUP_SIZE);
    let down_values_len = M6_3_04_HIDDEN_SIZE * M6_3_04_GATE_UP_ROWS;
    let down_scales_len = M6_3_04_HIDDEN_SIZE * (M6_3_04_GATE_UP_ROWS / M6_3_04_GROUP_SIZE);
    let mut offset = base;
    let read_values = |file: &mut File, offset: &mut u64, length: usize| {
        let bytes = m6_3_04_read_exact(file, *offset, length);
        *offset += u64::try_from(length).expect("value range");
        bytes
            .into_iter()
            .map(|value| i8::from_ne_bytes([value]))
            .collect::<Vec<_>>()
    };
    let read_scales = |file: &mut File, offset: &mut u64, count: usize| {
        let length = count * size_of::<f32>();
        let bytes = m6_3_04_read_exact(file, *offset, length);
        *offset += u64::try_from(length).expect("scale range");
        bytes
            .chunks_exact(size_of::<f32>())
            .map(|value| f32::from_le_bytes(value.try_into().expect("F32 scale")))
            .collect::<Vec<_>>()
    };
    let gate_values = read_values(file, &mut offset, gate_values_len);
    let gate_scales = read_scales(file, &mut offset, gate_scales_len);
    let up_values = read_values(file, &mut offset, gate_values_len);
    let up_scales = read_scales(file, &mut offset, gate_scales_len);
    let down_values = read_values(file, &mut offset, down_values_len);
    let down_scales = read_scales(file, &mut offset, down_scales_len);
    assert_eq!(
        offset - base,
        u64::try_from(m6_3_04_expert_bytes()).expect("expert bytes")
    );
    M6_3_04OwnedExpert {
        gate_values,
        gate_scales,
        up_values,
        up_scales,
        down_values,
        down_scales,
    }
}

#[derive(Debug, Default)]
struct M6_3_05CandidateCacheMetrics {
    hits: u64,
    misses: u64,
    logical_bytes_read: u64,
    peak_resident_bytes: usize,
}

struct M6_3_05CandidateCache {
    file: File,
    experts: HashMap<usize, M6_3_04OwnedExpert>,
    metrics: M6_3_05CandidateCacheMetrics,
}

impl M6_3_05CandidateCache {
    fn open(path: &Path) -> Self {
        Self {
            file: File::open(path).expect("open Layer-0 group-128 candidate"),
            experts: HashMap::new(),
            metrics: M6_3_05CandidateCacheMetrics::default(),
        }
    }

    fn clear_for_cold_run(&mut self) {
        self.experts.clear();
    }

    fn prepare(&mut self, selected_experts: &[usize]) {
        for &expert in selected_experts {
            if self.experts.contains_key(&expert) {
                self.metrics.hits += 1;
                continue;
            }
            self.metrics.misses += 1;
            self.metrics.logical_bytes_read +=
                u64::try_from(m6_3_04_expert_bytes()).expect("candidate expert bytes");
            self.experts
                .insert(expert, m6_3_04_load_expert(&mut self.file, expert));
            self.metrics.peak_resident_bytes = self
                .metrics
                .peak_resident_bytes
                .max(self.experts.len() * m6_3_04_expert_bytes());
        }
    }

    fn expert(&self, expert: usize) -> Layer0QuantizedExpert<'_> {
        self.experts[&expert].view()
    }
}

#[derive(Debug)]
struct M6_3_05Sample {
    ttft_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    dense_logical_bytes: u64,
    f32_expert_logical_bytes: u64,
    candidate_logical_bytes: u64,
    cache_hits: u64,
    cache_misses: u64,
    candidate_peak_resident_bytes: usize,
    f32_peak_resident_bytes: usize,
    kv_cache_bytes: usize,
}

fn m6_3_05_short_english_sample(
    artifact_root: &Path,
    candidate_cache: &mut M6_3_05CandidateCache,
) -> M6_3_05Sample {
    const TOKENS: [usize; 2] = [9707, 1879];
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime configuration")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        artifact_root,
        48 * 128,
    );
    let mut cache = KvCache::new(48, TOKENS.len(), 4, 128).expect("benchmark KV cache");
    let candidate_before_hits = candidate_cache.metrics.hits;
    let candidate_before_misses = candidate_cache.metrics.misses;
    let candidate_before_bytes = candidate_cache.metrics.logical_bytes_read;
    let mut step_seconds = Vec::with_capacity(TOKENS.len());
    let mut last_argmax = None;
    for (position, &token_id) in TOKENS.iter().enumerate() {
        let step_started = Instant::now();
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("benchmark input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("benchmark KV layer"),
            )
            .expect("benchmark attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("benchmark attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("benchmark post norm");
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .expect("benchmark F32 router");
            let moe = if layer == 0 {
                candidate_cache.prepare(&router.selected_experts);
                routed_layer0_quantized_experts_with_loader(
                    post_norm.view(),
                    &router,
                    config,
                    |expert| Ok(candidate_cache.expert(expert)),
                )
                .expect("benchmark Layer-0 candidate")
            } else {
                streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("benchmark F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("benchmark block");
            updates.push((attention.key, attention.value));
        }
        let normalized = rms_norm(
            hidden.view(),
            final_norm_weight.view(),
            config.rms_norm_epsilon(),
        )
        .expect("benchmark final norm");
        let logits = streaming_language_model_head(
            &mut payload,
            &final_plan,
            &normalized,
            &mut dense_bytes_read,
        );
        last_argmax = Some(greedy_token(logits.view()).expect("benchmark greedy token"));
        let updates_view = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache
            .append_token(&updates_view)
            .expect("benchmark KV append");
        assert_eq!(cache.len(), position + 1);
        step_seconds.push(step_started.elapsed().as_secs_f64());
    }
    assert_eq!(last_argmax, Some(0), "short-English candidate argmax");
    let f32 = store.metrics();
    let candidate_hits = candidate_cache.metrics.hits - candidate_before_hits;
    let candidate_misses = candidate_cache.metrics.misses - candidate_before_misses;
    M6_3_05Sample {
        ttft_seconds: step_seconds[0],
        prefill_tokens_per_second: 1.0 / step_seconds[0],
        decode_tokens_per_second: 1.0 / step_seconds[1],
        dense_logical_bytes: dense_bytes_read,
        f32_expert_logical_bytes: f32.bytes_read,
        candidate_logical_bytes: candidate_cache.metrics.logical_bytes_read
            - candidate_before_bytes,
        cache_hits: f32.hits + candidate_hits,
        cache_misses: f32.misses + candidate_misses,
        candidate_peak_resident_bytes: candidate_cache.metrics.peak_resident_bytes,
        f32_peak_resident_bytes: f32.peak_resident_bytes,
        kv_cache_bytes: cache.byte_size(),
    }
}

#[test]
fn m6_3_05_layer0_group128_cold_warm_benchmark() {
    let run_root = PathBuf::from(
        env::var_os("COLIBRI_M63_RUN_ROOT").expect("M6.3-05 requires a unique run root"),
    );
    assert!(run_root.is_absolute() && run_root.is_dir());
    assert!(
        fs::read_dir(&run_root)
            .expect("read M6.3-05 run root")
            .next()
            .is_none(),
        "M6.3-05 run root must be empty after preflight"
    );
    let expected_output = u64::try_from(m6_3_04_expert_bytes() * M6_3_04_EXPERT_COUNT)
        .expect("candidate output bytes");
    let free_bytes = env::var("COLIBRI_M63_FREE_BYTES")
        .expect("M6.3-05 requires recorded disk free bytes")
        .parse::<u64>()
        .expect("recorded disk free bytes");
    assert!(
        free_bytes >= expected_output + 1024 * 1024 * 1024,
        "M6.3-05 preflight requires output plus one-GiB reserve"
    );
    let _cleanup = M6_3_04RunCleanup(run_root.clone());
    let artifact_root = PathBuf::from(
        env::var_os("COLIBRI_ARTIFACT_ROOT").expect("M6.3-05 requires canonical artifact root"),
    );
    let incomplete_path = run_root.join("layer0-group128-int8-v1.bin.incomplete");
    let candidate_path = run_root.join("layer0-group128-int8-v1.bin");
    let (source_sha256, output_sha256) = m6_3_04_convert_layer0_group128(
        &artifact_root.join("experts/experts-layer-00000-of-00048.bin"),
        &incomplete_path,
    );
    fs::rename(&incomplete_path, &candidate_path).expect("promote benchmark candidate artifact");
    let mut cache = M6_3_05CandidateCache::open(&candidate_path);
    let repetitions = env::var("COLIBRI_M63_BENCHMARK_REPETITIONS").map_or(3, |value| {
        value
            .parse::<usize>()
            .expect("valid benchmark repetition count")
    });
    assert!(repetitions > 0, "benchmark repetitions must be positive");
    let mut evidence = String::from(
        "condition\trepetition\tttft_seconds\tprefill_tokens_per_second\tdecode_tokens_per_second\tdense_logical_bytes\tf32_expert_logical_bytes\tcandidate_logical_bytes\tbytes_per_token\tcache_hits\tcache_misses\tcache_hit_rate\tcandidate_peak_resident_bytes\tf32_peak_resident_bytes\tkv_cache_bytes\tvram_bytes\tphysical_read_bytes_status\n",
    );
    for repetition in 0..repetitions {
        cache.clear_for_cold_run();
        for condition in ["runtime_cache_cold", "runtime_cache_warm"] {
            let sample = m6_3_05_short_english_sample(&artifact_root, &mut cache);
            let total_bytes = sample.dense_logical_bytes
                + sample.f32_expert_logical_bytes
                + sample.candidate_logical_bytes;
            let requests = sample.cache_hits + sample.cache_misses;
            writeln!(
                evidence,
                "{condition}\t{repetition}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\t{:.17e}\t{}\t{}\t{:.17e}\t{}\t{}\t{}\t0\tnot_measured",
                sample.ttft_seconds,
                sample.prefill_tokens_per_second,
                sample.decode_tokens_per_second,
                sample.dense_logical_bytes,
                sample.f32_expert_logical_bytes,
                sample.candidate_logical_bytes,
                reporting_ratio(total_bytes, 2),
                sample.cache_hits,
                sample.cache_misses,
                if requests == 0 { 0.0 } else { reporting_ratio(sample.cache_hits, requests) },
                sample.candidate_peak_resident_bytes,
                sample.f32_peak_resident_bytes,
                sample.kv_cache_bytes,
            )
            .expect("write M6.3-05 evidence");
        }
    }
    fs::write(run_root.join("m6.3-05-evidence-v1.tsv"), &evidence).expect("write M6.3-05 evidence");
    println!(
        "m6_3_05 artifact_bytes={expected_output} source_sha256={source_sha256} output_sha256={output_sha256}\n{evidence}"
    );
}

struct M6_3_04RunCleanup(PathBuf);

impl Drop for M6_3_04RunCleanup {
    fn drop(&mut self) {
        if env::var_os("COLIBRI_M63_RETAIN_DEBUG").is_none() {
            fs::remove_dir_all(&self.0).expect("remove successful M6.3-04 run directory");
        }
    }
}

fn m6_3_04_top20_overlap(candidate: &[usize], reference: &[usize]) -> usize {
    candidate.iter().filter(|id| reference.contains(id)).count()
}

#[test]
fn m6_3_04_layer0_group128_matches_reference_f32() {
    assert!(M6_3_04_GROUP128_ARTIFACT_CONTRACT.contains("group128-artifact-v1"));
    assert!(
        M6_3_04_GROUP128_ARTIFACT_CONTRACT
            .contains("f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2")
    );
    let run_root = PathBuf::from(
        env::var_os("COLIBRI_M63_RUN_ROOT").expect("M6.3-04 requires a unique run root"),
    );
    assert!(run_root.is_absolute(), "M6.3-04 run root must be absolute");
    assert!(
        run_root.is_dir(),
        "M6.3-04 run root must exist after preflight"
    );
    assert!(
        fs::read_dir(&run_root)
            .expect("read M6.3-04 run root")
            .next()
            .is_none(),
        "M6.3-04 run root must be empty before conversion"
    );
    let expected_output = u64::try_from(m6_3_04_expert_bytes() * M6_3_04_EXPERT_COUNT)
        .expect("expected group-128 output bytes");
    let safety_reserve = 1024_u64 * 1024 * 1024;
    let free_bytes = env::var("COLIBRI_M63_FREE_BYTES")
        .expect("M6.3-04 requires recorded disk free bytes")
        .parse::<u64>()
        .expect("recorded disk free bytes");
    assert!(
        free_bytes >= expected_output + safety_reserve,
        "M6.3-04 preflight requires output plus one-GiB reserve"
    );
    let _cleanup = M6_3_04RunCleanup(run_root.clone());
    let artifact_root = PathBuf::from(
        env::var_os("COLIBRI_ARTIFACT_ROOT")
            .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact"),
    );
    let source_path = artifact_root.join("experts/experts-layer-00000-of-00048.bin");
    let incomplete_path = run_root.join("layer0-group128-int8-v1.bin.incomplete");
    let candidate_path = run_root.join("layer0-group128-int8-v1.bin");
    let (source_sha256, output_sha256) =
        m6_3_04_convert_layer0_group128(&source_path, &incomplete_path);
    fs::rename(&incomplete_path, &candidate_path).expect("atomically promote group-128 artifact");
    let generation_date = env::var("COLIBRI_M63_GENERATION_DATE")
        .expect("M6.3-04 requires an ISO-8601 generation date");
    let tool_versions =
        env::var("COLIBRI_M63_TOOL_VERSIONS").expect("M6.3-04 requires recorded tool versions");
    let conversion_command =
        env::var("COLIBRI_M63_COMMAND").expect("M6.3-04 requires the exact conversion command");
    let manifest = format!(
        "{{\"schema\":\"colibri-qwen3-moe-layer0-group128-artifact-v1\",\"schema_version\":1,\"model_id\":\"Qwen/Qwen3-30B-A3B\",\"model_revision\":\"ad44e777bcd18fa416d9da3bd8f70d33ebb85d39\",\"license\":\"Apache-2.0\",\"canonical_f32_root_manifest_sha256\":\"f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2\",\"source_relative_path\":\"experts/experts-layer-00000-of-00048.bin\",\"source_sha256\":\"{source_sha256}\",\"output_relative_path\":\"layer0-group128-int8-v1.bin\",\"output_sha256\":\"{output_sha256}\",\"output_bytes\":{expected_output},\"group_axis\":\"input_columns\",\"group_size\":128,\"rounding\":\"IEEE-754 f32 round-half-away-from-zero\",\"saturation\":\"[-127,127]\",\"byte_order\":\"little-endian\",\"conversion_command\":\"{conversion_command}\",\"tool_versions\":\"{tool_versions}\",\"generation_date\":\"{generation_date}\"}}\n"
    );
    fs::write(
        run_root.join("layer0-group128-int8-v1.manifest.json"),
        manifest,
    )
    .expect("write group-128 provenance manifest");

    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    assert_eq!(plan.payload, final_plan.payload, "dense payload identity");
    let mut payload =
        File::open(artifact_root.join(&plan.payload)).expect("open canonical dense payload");
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut evidence = String::from(
        "fixture\ttoken_ids\trouter_ids_exact\tlayer0_moe_max_abs_error\tfixed_logit_max_abs_error\ttop20_logit_max_abs_error\targmax\treference_argmax\ttop20_overlap\tselected_quantized_experts\n",
    );
    for fixture in tier_b_references()
        .into_iter()
        .filter(|fixture| fixture.name == "short_english" || fixture.name == "short_thai")
    {
        let mut store = expert_store_from_plans(
            &[
                LAYER47_EXPERT_RUNTIME_PLAN,
                GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
            ],
            &artifact_root,
            48 * 128,
        );
        let mut candidate_file = File::open(&candidate_path).expect("open group-128 candidate");
        let mut candidate_experts = HashMap::<usize, M6_3_04OwnedExpert>::new();
        let mut cache = KvCache::new(48, fixture.token_ids.len(), 4, 128).expect("Tier B KV cache");
        let mut current = None;
        let mut layer0_moe_error = 0.0_f32;
        let mut router_records = Vec::new();
        for (position, &token_id) in fixture.token_ids.iter().enumerate() {
            let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
            let mut updates = Vec::with_capacity(48);
            for layer in 0..48 {
                let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
                let input_norm = rms_norm(
                    hidden.view(),
                    weights.input_norm.view(),
                    config.rms_norm_epsilon(),
                )
                .expect("candidate input norm");
                let attention = cached_attention_with_weights(
                    input_norm.view(),
                    config,
                    weights.query.view(),
                    weights.key.view(),
                    weights.value.view(),
                    weights.output.view(),
                    weights.query_norm.view(),
                    weights.key_norm.view(),
                    cache.layer(layer).expect("KV layer view"),
                )
                .expect("candidate attention");
                let residual = elementwise_add(hidden.view(), attention.output.view())
                    .expect("candidate attention residual");
                let post_norm = rms_norm(
                    residual.view(),
                    weights.post_norm.view(),
                    config.rms_norm_epsilon(),
                )
                .expect("candidate post-attention norm");
                let router = route_tokens(post_norm.view(), weights.router.view(), config)
                    .expect("candidate F32 router");
                let moe = if layer == 0 {
                    router_records.push(format!(
                        "{position}:{}",
                        comma_separated(&router.selected_experts)
                    ));
                    for expert in &router.selected_experts {
                        candidate_experts
                            .entry(*expert)
                            .or_insert_with(|| m6_3_04_load_expert(&mut candidate_file, *expert));
                    }
                    let candidate = routed_layer0_quantized_experts_with_loader(
                        post_norm.view(),
                        &router,
                        config,
                        |expert| Ok(candidate_experts[&expert].view()),
                    )
                    .expect("direct Layer-0 group-128 experts");
                    let reference = streaming_routed_experts_with_observer(
                        post_norm.view(),
                        &router,
                        config,
                        layer,
                        &mut store,
                        expert_layout,
                        |_, _, _, _| {},
                    )
                    .expect("independent F32 Layer-0 reference");
                    layer0_moe_error = layer0_moe_error.max(
                        candidate
                            .data()
                            .iter()
                            .zip(reference.data())
                            .map(|(left, right)| (left - right).abs())
                            .fold(0.0_f32, f32::max),
                    );
                    candidate
                } else {
                    streaming_routed_experts_with_observer(
                        post_norm.view(),
                        &router,
                        config,
                        layer,
                        &mut store,
                        expert_layout,
                        |_, _, _, _| {},
                    )
                    .expect("F32 non-Layer-0 experts")
                };
                hidden = elementwise_add(residual.view(), moe.view()).expect("candidate block");
                updates.push((attention.key, attention.value));
            }
            let updates_view = updates
                .iter()
                .map(|(key, value)| LayerKvUpdate { key, value })
                .collect::<Vec<_>>();
            cache
                .append_token(&updates_view)
                .expect("candidate KV append");
            current = Some(hidden);
        }
        assert_eq!(
            router_records.last().expect("Layer-0 router record"),
            &format!(
                "{}:{}",
                fixture.token_ids.len() - 1,
                comma_separated(&fixture.guard_ids[&0])
            ),
            "final Layer-0 router IDs must remain exact"
        );
        let normalized = rms_norm(
            current.expect("non-empty candidate fixture").view(),
            final_norm_weight.view(),
            config.rms_norm_epsilon(),
        )
        .expect("candidate final RMSNorm");
        let logits = streaming_language_model_head(
            &mut payload,
            &final_plan,
            &normalized,
            &mut dense_bytes_read,
        );
        assert!(
            logits.data().iter().all(|value| value.is_finite()),
            "candidate logits finite"
        );
        let top20 = deterministic_top_ids(&logits, 20);
        let argmax = greedy_token(logits.view()).expect("candidate greedy token");
        let fixed_logit_error = maximum_indexed_difference(
            logits.data(),
            &fixture.fixed_logit_indices,
            &fixture.fixed_logits,
        );
        let top20_logit_error =
            maximum_indexed_difference(logits.data(), &fixture.top20_ids, &fixture.top20_logits);
        let mut selected_quantized_experts = candidate_experts.keys().copied().collect::<Vec<_>>();
        selected_quantized_experts.sort_unstable();
        writeln!(
            evidence,
            "{}\t{}\t{}\t{layer0_moe_error:.17e}\t{fixed_logit_error:.17e}\t{top20_logit_error:.17e}\t{argmax}\t{}\t{}\t{}",
            fixture.name,
            comma_separated(&fixture.token_ids),
            router_records.join("|"),
            fixture.argmax,
            m6_3_04_top20_overlap(&top20, &fixture.top20_ids),
            comma_separated(&selected_quantized_experts),
        )
        .expect("write M6.3-04 evidence");
    }
    fs::write(run_root.join("m6.3-04-evidence-v1.tsv"), &evidence).expect("write M6.3-04 evidence");
    println!(
        "m6_3_04 artifact_bytes={expected_output} source_sha256={source_sha256} output_sha256={output_sha256}\n{evidence}"
    );
}

/// R1.1a's private seam deliberately changes only the Layer-0 expert store.
/// Attention, router, layers 1--47, final norm, and LM head remain the exact
/// canonical F32 test-runtime implementation below.
struct R1_1aRun {
    ids: Vec<usize>,
    router_logits: Vec<f32>,
    weights: Vec<f32>,
    expert_input: Vec<f32>,
    expert_outputs: Vec<f32>,
    aggregation_events: Vec<(usize, usize, usize)>,
    moe: Vec<f32>,
    block: Vec<f32>,
    logits: Tensor,
    #[allow(dead_code)] // consumed by the pre-registered candidate execution tests added next.
    candidate_telemetry: Option<R1_1aCandidateTelemetry>,
}

#[derive(Debug, PartialEq, Eq)]
struct R1_1aCandidateTelemetry {
    candidate_id: String,
    artifact_path: PathBuf,
    artifact_sha256: String,
    group_size: usize,
    verification_bytes_read: u64,
    payload_bytes_read: u64,
    peak_packed_expert_bytes: usize,
    complete_f32_weight_materializations: usize,
}

struct R1_1aCandidateSpec<'a> {
    candidate_id: &'a str,
    artifact_path: &'a Path,
    artifact_sha256: &'a str,
    group_size: usize,
}

// Frozen by ADR 0064 from two pre-candidate canonical F32 control processes.
const R1_1A_CODE_NEWLINE_INPUT_BUDGET: f32 = 1.529_685_3e-5;
const R1_1A_CODE_NEWLINE_ROUTER_BUDGET: f32 = 2.910_196_8e-5;
const R1_1A_CODE_NEWLINE_ROUTING_BUDGET: f32 = 1.106_725_5e-5;
const R1_1A_CODE_NEWLINE_SELECTED_OUTPUT_BUDGET: f32 = 8.510_186e-6;
const R1_1A_CODE_NEWLINE_MOE_BUDGET: f32 = 4.039_837e-6;

fn r1_1a_code_newline_run(override_layer0_f32: bool) -> R1_1aRun {
    r1_1a_code_newline_run_with_source(override_layer0_f32, None)
}

#[allow(dead_code)] // execution remains prohibited until telemetry validation closes.
fn r1_1a_candidate_code_newline_run(spec: R1_1aCandidateSpec<'_>) -> R1_1aRun {
    r1_1a_code_newline_run_with_source(false, Some(spec))
}

fn r1_1a_code_newline_run_with_source(
    override_layer0_f32: bool,
    candidate: Option<R1_1aCandidateSpec<'_>>,
) -> R1_1aRun {
    assert!(
        !(override_layer0_f32 && candidate.is_some()),
        "control override and candidate source are mutually exclusive"
    );
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let mut payload =
        File::open(artifact_root.join(&plan.payload)).expect("open canonical dense payload");
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("config")
        .runtime_config();
    let layout = PackedExpertLayout::for_config(config);
    let mut bytes_read = 0;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut bytes_read,
    );
    let mut normal_store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    // This independent store is the control override source; it reads the same
    // canonical Layer-0 shard through the production F32 ExpertStore contract.
    let mut ordinary_layer0_store = canonical_layer0_expert_store(&artifact_root);
    let mut override_store = canonical_layer0_expert_store(&artifact_root);
    let mut candidate_reader = candidate.as_ref().map(|spec| {
        R1_1CandidateReader::open(
            spec.artifact_path,
            R1_1PackedArtifactLayout::canonical(spec.group_size).expect("candidate layout"),
            spec.artifact_sha256,
        )
        .expect("validated candidate artifact")
    });
    let mut cache = KvCache::new(48, 4, 4, 128).expect("KV cache");
    let mut ids = Vec::new();
    let mut router_logits = Vec::with_capacity(4 * 128);
    let mut weights = Vec::with_capacity(4 * 8);
    let mut expert_input = Vec::with_capacity(4 * 2048);
    let mut expert_outputs = vec![0.0; 4 * 8 * 2048];
    let mut aggregation_events = Vec::with_capacity(32);
    let mut moe = Vec::with_capacity(4 * 2048);
    let mut block = Vec::with_capacity(4 * 2048);
    let mut current = None;
    for (position, &token_id) in [87_usize, 28, 16, 198].iter().enumerate() {
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let layer_weights = layer_weights(&mut payload, &plan, layer, &mut bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                layer_weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                layer_weights.query.view(),
                layer_weights.key.view(),
                layer_weights.value.view(),
                layer_weights.output.view(),
                layer_weights.query_norm.view(),
                layer_weights.key_norm.view(),
                cache.layer(layer).expect("KV layer"),
            )
            .expect("attention");
            let residual =
                elementwise_add(hidden.view(), attention.output.view()).expect("residual");
            let post_norm = rms_norm(
                residual.view(),
                layer_weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("post norm");
            let router = route_tokens(post_norm.view(), layer_weights.router.view(), config)
                .expect("router");
            let expert_result = if layer == 0 {
                ids.extend_from_slice(&router.selected_experts);
                router_logits.extend_from_slice(router.logits.data());
                weights.extend_from_slice(router.weights.data());
                expert_input.extend_from_slice(post_norm.data());
                if let Some(reader) = candidate_reader.as_mut() {
                    r1_1_routed_experts_with_observer(
                        post_norm.view(),
                        &router,
                        config,
                        reader,
                        |expert, token, rank, output| {
                            let offset = (position * 8 + rank) * 2048;
                            assert_eq!(token, 0, "one-token candidate layer-0 call");
                            assert_eq!(expert, router.selected_experts[rank]);
                            aggregation_events.push((position, expert, rank));
                            expert_outputs[offset..offset + 2048].copy_from_slice(output);
                        },
                    )
                    .expect("Layer-0 candidate expert source")
                } else {
                    let store = if override_layer0_f32 {
                        &mut override_store
                    } else {
                        &mut ordinary_layer0_store
                    };
                    streaming_routed_experts_with_observer(
                        post_norm.view(),
                        &router,
                        config,
                        layer,
                        store,
                        layout,
                        |expert, token, rank, output| {
                            let offset = (position * 8 + rank) * 2048;
                            assert_eq!(token, 0, "one-token streaming layer-0 call");
                            assert_eq!(expert, router.selected_experts[rank]);
                            aggregation_events.push((position, expert, rank));
                            expert_outputs[offset..offset + 2048].copy_from_slice(output);
                        },
                    )
                    .expect("Layer-0 F32 expert source")
                }
            } else {
                streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut normal_store,
                    layout,
                    |_, _, _, _| {},
                )
                .expect("canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), expert_result.view()).expect("block output");
            if layer == 0 {
                moe.extend_from_slice(expert_result.data());
                block.extend_from_slice(hidden.data());
            }
            updates.push((attention.key, attention.value));
        }
        let views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache.append_token(&views).expect("append token");
        current = Some(hidden);
    }
    let normalized = rms_norm(
        current.expect("final hidden").view(),
        final_norm_weight.view(),
        config.rms_norm_epsilon(),
    )
    .expect("final norm");
    let logits =
        streaming_language_model_head(&mut payload, &final_plan, &normalized, &mut bytes_read);
    let candidate_telemetry = candidate.map(|spec| {
        let reader = candidate_reader.expect("candidate reader");
        R1_1aCandidateTelemetry {
            candidate_id: spec.candidate_id.to_owned(),
            artifact_path: reader.path().to_path_buf(),
            artifact_sha256: reader.sha256().to_owned(),
            group_size: spec.group_size,
            verification_bytes_read: reader.verification_bytes_read(),
            payload_bytes_read: reader.payload_bytes_read(),
            peak_packed_expert_bytes: reader.peak_packed_expert_bytes(),
            complete_f32_weight_materializations: 0,
        }
    });
    R1_1aRun {
        ids,
        router_logits,
        weights,
        expert_input,
        expert_outputs,
        aggregation_events,
        moe,
        block,
        logits,
        candidate_telemetry,
    }
}

fn r1_1a_max_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn r1_1a_finite_max_difference(left: &[f32], right: &[f32], checkpoint: &str) -> f32 {
    assert_eq!(left.len(), right.len(), "{checkpoint} length");
    left.iter()
        .zip(right)
        .enumerate()
        .map(|(index, (actual, expected))| {
            assert!(actual.is_finite(), "{checkpoint} actual[{index}] finite");
            assert!(
                expected.is_finite(),
                "{checkpoint} expected[{index}] finite"
            );
            (actual - expected).abs()
        })
        .fold(0.0, f32::max)
}

fn r1_1a_checkpoint_hash(run: &R1_1aRun) -> String {
    let mut hasher = Sha256Hasher::new();
    for (name, values) in [
        ("router_logits", run.router_logits.as_slice()),
        ("routing_weights", run.weights.as_slice()),
        ("expert_input", run.expert_input.as_slice()),
        ("selected_expert_output", run.expert_outputs.as_slice()),
        ("aggregated_moe_output", run.moe.as_slice()),
        ("layer0_block_output", run.block.as_slice()),
        ("final_logits", run.logits.data()),
    ] {
        hasher.update(name.as_bytes());
        hasher.update(&[0]);
        for value in values {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    hasher.update(b"selected_expert_ids\0");
    for &expert in &run.ids {
        hasher.update(
            &u64::try_from(expert)
                .expect("expert ID fits u64")
                .to_le_bytes(),
        );
    }
    m6_3_04_sha256_hex(hasher)
}

fn r1_1a_assert_runs_exact(ordinary: &R1_1aRun, overridden: &R1_1aRun) {
    assert_eq!(
        ordinary.ids, overridden.ids,
        "router IDs are exact across the seam"
    );
    for (name, left, right) in [
        (
            "router_logits",
            ordinary.router_logits.as_slice(),
            overridden.router_logits.as_slice(),
        ),
        (
            "routing_weights",
            ordinary.weights.as_slice(),
            overridden.weights.as_slice(),
        ),
        (
            "expert_input",
            ordinary.expert_input.as_slice(),
            overridden.expert_input.as_slice(),
        ),
        (
            "selected_expert_output",
            ordinary.expert_outputs.as_slice(),
            overridden.expert_outputs.as_slice(),
        ),
        (
            "aggregated_moe_output",
            ordinary.moe.as_slice(),
            overridden.moe.as_slice(),
        ),
        (
            "layer0_block_output",
            ordinary.block.as_slice(),
            overridden.block.as_slice(),
        ),
        (
            "final_logits",
            ordinary.logits.data(),
            overridden.logits.data(),
        ),
    ] {
        assert_eq!(
            r1_1a_max_difference(left, right),
            0.0,
            "{name} control override drift"
        );
    }
}

type R1_1aFrozenCheckpointPayload = (Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>);

fn r1_1a_frozen_checkpoint_payload() -> R1_1aFrozenCheckpointPayload {
    let raw =
        fs::read("../../models/qwen3-30b-a3b/m6.3-r1-1b-code-newline-layer0-checkpoints-v1.bin")
            .expect("read frozen R1.1b payload");
    assert_eq!(raw.len(), 330_811);
    let mut hasher = Sha256Hasher::new();
    hasher.update(&raw);
    assert_eq!(
        hasher.finalize(),
        decode_sha256("6a614737eab3c775fac1ed02f3eeabbf5f0d69eb7ab20ba9010bba0c1b9b0978")
    );
    assert_eq!(&raw[..10], b"CLR-R1-1B\0");
    let header_length = usize::try_from(u64::from_le_bytes(
        raw[10..18].try_into().expect("header length"),
    ))
    .expect("header fits");
    let header = std::str::from_utf8(&raw[18..18 + header_length]).expect("header UTF-8");
    for name in [
        "layer0.post_attention_rmsnorm",
        "layer0.router_logits",
        "layer0.selected_expert_ids",
        "layer0.routing_weights",
        "layer0.selected_expert_output",
        "layer0.aggregated_moe_output",
    ] {
        assert!(header.contains(name), "missing frozen checkpoint {name}");
    }
    let mut cursor = 18 + header_length;
    let f32s = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32")))
            .collect::<Vec<_>>()
    };
    let input = f32s(&raw[cursor..cursor + 32_768]);
    cursor += 32_768;
    let router = f32s(&raw[cursor..cursor + 2_048]);
    cursor += 2_048;
    let ids = raw[cursor..cursor + 256]
        .chunks_exact(8)
        .map(|chunk| {
            usize::try_from(i64::from_le_bytes(chunk.try_into().expect("id")))
                .expect("non-negative ID")
        })
        .collect();
    cursor += 256;
    let weights = f32s(&raw[cursor..cursor + 128]);
    cursor += 128;
    let outputs = f32s(&raw[cursor..cursor + 262_144]);
    cursor += 262_144;
    let moe = f32s(&raw[cursor..cursor + 32_768]);
    cursor += 32_768;
    assert_eq!(cursor, raw.len());
    (input, router, ids, weights, outputs, moe)
}

fn r1_1a_selected_output_diagnostic(
    actual: &[f32],
    expected: &[f32],
    ids: &[usize],
    tolerance: f32,
) {
    let mut rows = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| {
            let error = (actual - expected).abs();
            let token = index / 16_384;
            let within = index % 16_384;
            (
                error,
                index,
                token,
                within / 2048,
                ids[token * 8 + within / 2048],
                within % 2048,
                *expected,
                *actual,
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let worst = rows.first().expect("selected outputs");
    let exceeded = rows.iter().filter(|row| row.0 > tolerance).count();
    let relative = rows
        .iter()
        .map(|row| row.0 / row.6.abs().max(f32::MIN_POSITIVE))
        .fold(0.0, f32::max);
    let worst_ten = rows
        .iter()
        .take(10)
        .map(|row| {
            format!(
                "t={} r={} e={} i={} expected={:.9e} actual={:.9e} abs={:.9e}",
                row.2, row.3, row.4, row.5, row.6, row.7, row.0
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        worst.0 <= tolerance,
        "selected_expert_output max_abs={:.9e} max_relative={:.9e} allowed={:.9e} first=t:{} r:{} e:{} i:{} expected:{:.9e} actual:{:.9e} abs:{:.9e} exceeds:{} worst10:[{}]",
        worst.0,
        relative,
        tolerance,
        worst.2,
        worst.3,
        worst.4,
        worst.5,
        worst.6,
        worst.7,
        worst.0,
        exceeded,
        worst_ten
    );
}

fn r1_1a_recompute_selected_outputs(run: &R1_1aRun, root: &Path) -> Vec<f32> {
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("config")
        .runtime_config();
    let router = crate::block::RouterOutput {
        logits: Tensor::new(TensorShape::new([4, 128]), run.router_logits.clone()).expect("logits"),
        weights: Tensor::new(TensorShape::new([4, 8]), run.weights.clone()).expect("weights"),
        selected_experts: run.ids.clone(),
    };
    let input = Tensor::new(TensorShape::new([4, 2048]), run.expert_input.clone()).expect("input");
    let mut output = vec![0.0; 65_536];
    let mut store = canonical_layer0_expert_store(root);
    let _ = streaming_routed_experts_with_observer(
        input.view(),
        &router,
        config,
        0,
        &mut store,
        PackedExpertLayout::for_config(config),
        |expert, token, rank, values| {
            assert_eq!(expert, run.ids[token * 8 + rank]);
            output[(token * 8 + rank) * 2048..(token * 8 + rank + 1) * 2048]
                .copy_from_slice(values);
        },
    )
    .expect("same-input recomputation");
    output
}

fn r1_1a_weighted_sum(
    outputs: &[f32],
    weights: &[f32],
    events: &[(usize, usize, usize)],
) -> Vec<f32> {
    let mut total = vec![0.0; 8192];
    for &(token, _expert, rank) in events {
        for hidden in 0..2048 {
            total[token * 2048 + hidden] +=
                weights[token * 8 + rank] * outputs[(token * 8 + rank) * 2048 + hidden];
        }
    }
    total
}

#[test]
fn m6_3_r1_1a_full_path_selected_output_triangle() {
    let run = r1_1a_code_newline_run(false);
    let (frozen_input, _logits, ids, _weights, frozen_outputs, _moe) =
        r1_1a_frozen_checkpoint_payload();
    assert_eq!(run.ids, ids);
    let root = PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical root"));
    let recomputed = r1_1a_recompute_selected_outputs(&run, &root);
    assert_eq!(
        r1_1a_max_difference(&run.expert_outputs, &recomputed),
        0.0,
        "C vs B capture/execution mismatch"
    );
    assert_eq!(
        r1_1a_max_difference(
            &r1_1a_weighted_sum(&run.expert_outputs, &run.weights, &run.aggregation_events),
            &run.moe
        ),
        0.0,
        "C weighted sum vs full MoE mismatch"
    );
    assert_eq!(
        r1_1a_max_difference(
            &r1_1a_weighted_sum(&recomputed, &run.weights, &run.aggregation_events),
            &run.moe
        ),
        0.0,
        "B weighted sum vs full MoE mismatch"
    );
    eprintln!(
        "R1.1a triangle full_input_vs_frozen_max_abs={:.9e}",
        r1_1a_max_difference(&run.expert_input, &frozen_input)
    );
    r1_1a_selected_output_diagnostic(
        &recomputed,
        &frozen_outputs,
        &run.ids,
        R1_1A_CODE_NEWLINE_SELECTED_OUTPUT_BUDGET,
    );
}

#[test]
fn m6_3_r1_1a_isolated_selected_expert_output_diagnosis() {
    let (input, logits, ids, weights, expected, _moe) = r1_1a_frozen_checkpoint_payload();
    let root = PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical root"));
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("config")
        .runtime_config();
    let router = crate::block::RouterOutput {
        logits: Tensor::new(TensorShape::new([4, 128]), logits).expect("logits"),
        weights: Tensor::new(TensorShape::new([4, 8]), weights).expect("weights"),
        selected_experts: ids.clone(),
    };
    let input = Tensor::new(TensorShape::new([4, 2048]), input).expect("input");
    let mut actual = vec![0.0; 65_536];
    let mut store = canonical_layer0_expert_store(&root);
    let _ = streaming_routed_experts_with_observer(
        input.view(),
        &router,
        config,
        0,
        &mut store,
        PackedExpertLayout::for_config(config),
        |expert, token, rank, output| {
            assert_eq!(expert, ids[token * 8 + rank]);
            actual[(token * 8 + rank) * 2048..(token * 8 + rank + 1) * 2048]
                .copy_from_slice(output);
        },
    )
    .expect("isolated MLP");
    r1_1a_selected_output_diagnostic(&actual, &expected, &ids, 2.2964e-6);
}

#[test]
fn m6_3_r1_1a_code_newline_tolerance_budget_bits_are_frozen() {
    assert_eq!(R1_1A_CODE_NEWLINE_INPUT_BUDGET.to_bits(), 0x3780_51be);
    assert_eq!(R1_1A_CODE_NEWLINE_ROUTER_BUDGET.to_bits(), 0x37f4_2000);
    assert_eq!(R1_1A_CODE_NEWLINE_ROUTING_BUDGET.to_bits(), 0x3739_ad80);
    assert_eq!(
        R1_1A_CODE_NEWLINE_SELECTED_OUTPUT_BUDGET.to_bits(),
        0x370e_c6f8
    );
    assert_eq!(R1_1A_CODE_NEWLINE_MOE_BUDGET.to_bits(), 0x3687_8df0);
}

#[test]
fn m6_3_r1_1a_control_matches_frozen_r1_1b_checkpoints() {
    let run = r1_1a_code_newline_run(false);
    let (input, router, ids, weights, outputs, moe) = r1_1a_frozen_checkpoint_payload();
    assert_eq!(run.ids, ids, "selected expert IDs are exact");
    for (name, actual, expected, tolerance) in [
        (
            "post_attention_rmsnorm/expert_input",
            run.expert_input.as_slice(),
            input.as_slice(),
            R1_1A_CODE_NEWLINE_INPUT_BUDGET,
        ),
        (
            "router_logits",
            run.router_logits.as_slice(),
            router.as_slice(),
            R1_1A_CODE_NEWLINE_ROUTER_BUDGET,
        ),
        (
            "routing_weights",
            run.weights.as_slice(),
            weights.as_slice(),
            R1_1A_CODE_NEWLINE_ROUTING_BUDGET,
        ),
        (
            "selected_expert_output",
            run.expert_outputs.as_slice(),
            outputs.as_slice(),
            R1_1A_CODE_NEWLINE_SELECTED_OUTPUT_BUDGET,
        ),
        (
            "aggregated_moe_output",
            run.moe.as_slice(),
            moe.as_slice(),
            R1_1A_CODE_NEWLINE_MOE_BUDGET,
        ),
    ] {
        if name == "selected_expert_output" {
            r1_1a_selected_output_diagnostic(actual, expected, &run.ids, tolerance);
        } else {
            assert!(
                r1_1a_max_difference(actual, expected) <= tolerance,
                "{name} exceeds frozen tolerance"
            );
        }
    }
    // `block` is intentionally excluded: it is an internal exact-only diagnostic,
    // not a frozen/oracle checkpoint and never an admission gate.
    assert_eq!(run.block.len(), 4 * 2048);
}

#[test]
fn m6_3_r1_1a_code_newline_control_calibration_observation() {
    let ordinary = r1_1a_code_newline_run(false);
    let overridden = r1_1a_code_newline_run(true);
    r1_1a_assert_runs_exact(&ordinary, &overridden);
    assert_eq!(ordinary.ids.len(), 32);
    assert_eq!(&ordinary.ids[24..], &[16, 114, 1, 98, 84, 100, 52, 53]);

    let (input, router, ids, weights, outputs, moe) = r1_1a_frozen_checkpoint_payload();
    assert_eq!(ordinary.ids, ids, "selected expert IDs are exact");
    let input_error = r1_1a_finite_max_difference(
        &ordinary.expert_input,
        &input,
        "layer0.post_attention_rmsnorm",
    );
    let router_error =
        r1_1a_finite_max_difference(&ordinary.router_logits, &router, "layer0.router_logits");
    let routing_error =
        r1_1a_finite_max_difference(&ordinary.weights, &weights, "layer0.routing_weights");
    let selected_output_error = r1_1a_finite_max_difference(
        &ordinary.expert_outputs,
        &outputs,
        "layer0.selected_expert_output",
    );
    let moe_error =
        r1_1a_finite_max_difference(&ordinary.moe, &moe, "layer0.aggregated_moe_output");

    let recomputed = r1_1a_recompute_selected_outputs(
        &ordinary,
        &PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical root")),
    );
    assert_eq!(
        r1_1a_max_difference(&ordinary.expert_outputs, &recomputed),
        0.0,
        "captured selected outputs match same-input recomputation"
    );
    assert_eq!(
        r1_1a_max_difference(
            &r1_1a_weighted_sum(
                &ordinary.expert_outputs,
                &ordinary.weights,
                &ordinary.aggregation_events,
            ),
            &ordinary.moe,
        ),
        0.0,
        "runtime-order aggregation replay"
    );

    let fixture = tier_b_references()
        .into_iter()
        .find(|item| item.name == "code_newline")
        .expect("code_newline reference");
    let fixed_logit_error = maximum_indexed_difference(
        ordinary.logits.data(),
        &fixture.fixed_logit_indices,
        &fixture.fixed_logits,
    );
    assert!(fixed_logit_error.is_finite());
    assert!(
        fixed_logit_error <= tier_b_fixture_budget("code_newline", "logits"),
        "fixed final logits exceed the frozen F32 tolerance: {fixed_logit_error}"
    );

    println!(
        "R1_1A_CONTROL_CALIBRATION_V1 input_max_abs={input_error:.17e} router_max_abs={router_error:.17e} routing_max_abs={routing_error:.17e} selected_output_max_abs={selected_output_error:.17e} moe_max_abs={moe_error:.17e} fixed_logit_max_abs={fixed_logit_error:.17e} checkpoint_sha256={}",
        r1_1a_checkpoint_hash(&ordinary),
    );
}

#[test]
fn m6_3_r1_1a_canonical_layer0_registration_preflight() {
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical root"));
    let mut store = canonical_layer0_expert_store(&artifact_root);
    let lease = store
        .load(ExpertKey {
            layer_index: 0,
            expert_id: ExpertId(4),
        })
        .expect("frozen code_newline expert must resolve");
    assert_eq!(lease.bytes().len(), 18_874_368);
    drop(lease);
    assert!(
        store
            .load(ExpertKey {
                layer_index: 0,
                expert_id: ExpertId(128)
            })
            .is_err()
    );
    drop(store);
    assert!(File::open(artifact_root.join("experts/experts-layer-00000-of-00048.bin")).is_ok());
}

#[test]
fn m6_3_r1_1a_control_override_reuses_authoritative_f32_path() {
    let ordinary = r1_1a_code_newline_run(false);
    let overridden = r1_1a_code_newline_run(true);
    r1_1a_assert_runs_exact(&ordinary, &overridden);
    assert_eq!(ordinary.ids.len(), 32);
    assert_eq!(&ordinary.ids[24..], &[16, 114, 1, 98, 84, 100, 52, 53]);
    let fixture = tier_b_references()
        .into_iter()
        .find(|item| item.name == "code_newline")
        .expect("code_newline reference");
    let fixed_logit_error = maximum_indexed_difference(
        overridden.logits.data(),
        &fixture.fixed_logit_indices,
        &fixture.fixed_logits,
    );
    assert!(
        fixed_logit_error <= tier_b_fixture_budget("code_newline", "logits"),
        "fixed final logits exceed the frozen F32 tolerance: {fixed_logit_error}"
    );
}

fn r1_1a_characterize_candidate(
    expected_candidate_id: &str,
    group_size: usize,
    expected_sha256: &str,
) {
    let candidate_id = env::var("COLIBRI_R1_1A_CANDIDATE_ID").expect("candidate ID");
    assert_eq!(candidate_id, expected_candidate_id, "candidate ID binding");
    let candidate_path = PathBuf::from(
        env::var_os("COLIBRI_R1_1A_CANDIDATE_PATH").expect("candidate artifact path"),
    );
    let run = r1_1a_candidate_code_newline_run(R1_1aCandidateSpec {
        candidate_id: &candidate_id,
        artifact_path: &candidate_path,
        artifact_sha256: expected_sha256,
        group_size,
    });
    let (input, router, ids, weights, outputs, moe) = r1_1a_frozen_checkpoint_payload();
    assert_eq!(run.ids, ids, "candidate must preserve Layer-0 router IDs");
    let input_error =
        r1_1a_finite_max_difference(&run.expert_input, &input, "layer0.post_attention_rmsnorm");
    let router_error =
        r1_1a_finite_max_difference(&run.router_logits, &router, "layer0.router_logits");
    let routing_error =
        r1_1a_finite_max_difference(&run.weights, &weights, "layer0.routing_weights");
    assert!(input_error <= R1_1A_CODE_NEWLINE_INPUT_BUDGET);
    assert!(router_error <= R1_1A_CODE_NEWLINE_ROUTER_BUDGET);
    assert!(routing_error <= R1_1A_CODE_NEWLINE_ROUTING_BUDGET);
    let selected_output_error = r1_1a_finite_max_difference(
        &run.expert_outputs,
        &outputs,
        "layer0.selected_expert_output",
    );
    let moe_error = r1_1a_finite_max_difference(&run.moe, &moe, "layer0.aggregated_moe_output");
    assert_eq!(
        r1_1a_max_difference(
            &r1_1a_weighted_sum(&run.expert_outputs, &run.weights, &run.aggregation_events),
            &run.moe,
        ),
        0.0,
        "candidate runtime-order aggregation replay"
    );
    let fixture = tier_b_references()
        .into_iter()
        .find(|item| item.name == "code_newline")
        .expect("code_newline reference");
    let fixed_logit_error = maximum_indexed_difference(
        run.logits.data(),
        &fixture.fixed_logit_indices,
        &fixture.fixed_logits,
    );
    assert!(
        fixed_logit_error.is_finite(),
        "candidate fixed logits finite"
    );
    let first_divergence = if selected_output_error > R1_1A_CODE_NEWLINE_SELECTED_OUTPUT_BUDGET {
        "layer0.selected_expert_output"
    } else if moe_error > R1_1A_CODE_NEWLINE_MOE_BUDGET {
        "layer0.aggregated_moe_output"
    } else if fixed_logit_error > tier_b_fixture_budget("code_newline", "logits") {
        "final_logits"
    } else {
        "none"
    };
    let telemetry = run
        .candidate_telemetry
        .as_ref()
        .expect("candidate telemetry");
    assert_eq!(telemetry.candidate_id, candidate_id);
    assert_eq!(telemetry.artifact_path, candidate_path);
    assert_eq!(telemetry.artifact_sha256, expected_sha256);
    assert_eq!(telemetry.group_size, group_size);
    assert_eq!(
        telemetry.verification_bytes_read,
        R1_1PackedArtifactLayout::canonical(group_size)
            .expect("layout")
            .artifact_bytes()
            .expect("artifact bytes")
    );
    assert_eq!(
        telemetry.payload_bytes_read,
        u64::try_from(
            R1_1PackedArtifactLayout::canonical(group_size)
                .expect("layout")
                .expert_bytes()
                .expect("expert bytes")
                * 32
        )
        .expect("payload bytes")
    );
    assert_eq!(telemetry.complete_f32_weight_materializations, 0);
    println!(
        "R1_1A_CANDIDATE_V1 candidate_id={} group_size={} artifact_sha256={} verification_bytes_read={} payload_bytes_read={} peak_packed_expert_bytes={} complete_f32_weight_materializations={} input_max_abs={:.17e} router_max_abs={:.17e} routing_max_abs={:.17e} selected_output_max_abs={:.17e} moe_max_abs={:.17e} fixed_logit_max_abs={:.17e} logit_cap_pass={} first_divergence={} checkpoint_sha256={} final_logits_sha256={}",
        telemetry.candidate_id,
        telemetry.group_size,
        telemetry.artifact_sha256,
        telemetry.verification_bytes_read,
        telemetry.payload_bytes_read,
        telemetry.peak_packed_expert_bytes,
        telemetry.complete_f32_weight_materializations,
        input_error,
        router_error,
        routing_error,
        selected_output_error,
        moe_error,
        fixed_logit_error,
        fixed_logit_error <= 0.05,
        first_divergence,
        r1_1a_checkpoint_hash(&run),
        f32_little_endian_sha256(run.logits.data()),
    );
}

#[test]
fn m6_3_r1_1a_characterize_group64_code_newline() {
    r1_1a_characterize_candidate(
        "cpu-safe-rust-int8-group64-layer0-r1-1a",
        64,
        "35a3ef6aba723d302fb1a7fded6ede4543a0c3dfcd35651596b78f1c5158cad2",
    );
}

#[test]
fn m6_3_r1_1a_characterize_group32_code_newline() {
    r1_1a_characterize_candidate(
        "cpu-safe-rust-int8-group32-layer0-r1-1a",
        32,
        "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2",
    );
}

const R1_2_GENERATED_TOKEN_COUNT: usize = 2;
const R1_2_GROUP32_ARTIFACT_BYTES: u64 = 679_477_248;
const R1_2_GROUP32_ARTIFACT_SHA256: &str =
    "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2";
const R1_2_LOGIT_MAX_ABS_ENVELOPE: f32 = 0.05;

#[derive(Debug)]
struct R1_2Run {
    prompt_top20_ids: Vec<usize>,
    prompt_argmax: usize,
    prompt_guard_ids: HashMap<usize, Vec<usize>>,
    prompt_logits_sha256: String,
    #[cfg(feature = "m6-3-r2-native")]
    prompt_logits: Vec<f32>,
    prompt_final_norm_sha256: String,
    prompt_fixed_logit_error: f32,
    prompt_top20_logit_error: f32,
    generated_ids: Vec<usize>,
    layer0_moe_max_abs_error: f32,
    #[cfg(feature = "m6-3-r2-native")]
    native_scalar_layer0_max_abs_error: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum R1_2CandidateMode {
    Scalar,
    #[cfg(feature = "m6-3-r2-native")]
    NativeAvx2Fma,
}

#[derive(Debug)]
struct R1_2FrozenReference {
    token_ids: Vec<usize>,
    generated_ids: Vec<usize>,
    prompt_top20_ids: Vec<usize>,
    prompt_argmax: usize,
    prompt_guard_ids: HashMap<usize, Vec<usize>>,
    prompt_logits_sha256: String,
    prompt_final_norm_sha256: String,
}

fn r1_2_fixture_run(
    fixture: &TierBReference,
    candidate_reader: Option<&mut R1_1CandidateReader>,
) -> R1_2Run {
    r1_2_fixture_run_with_mode(fixture, candidate_reader, R1_2CandidateMode::Scalar)
}

fn r1_2_fixture_run_with_mode(
    fixture: &TierBReference,
    mut candidate_reader: Option<&mut R1_1CandidateReader>,
    candidate_mode: R1_2CandidateMode,
) -> R1_2Run {
    assert!(
        fixture.name == "short_english" || fixture.name == "short_thai",
        "R1.2 is restricted to the admitted held-out fixtures"
    );
    let artifact_root =
        PathBuf::from(env::var_os("COLIBRI_ARTIFACT_ROOT").expect("canonical artifact root"));
    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("R1.2 runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut payload =
        File::open(artifact_root.join(&plan.payload)).expect("open R1.2 dense payload");
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut normal_store = expert_store_from_plans(
        &[
            LAYER47_EXPERT_RUNTIME_PLAN,
            GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
        ],
        &artifact_root,
        48 * 128,
    );
    let mut reference_layer0_store = canonical_layer0_expert_store(&artifact_root);
    let processed_positions = fixture.token_ids.len() + R1_2_GENERATED_TOKEN_COUNT - 1;
    let mut cache = KvCache::new(48, processed_positions, 4, 128).expect("R1.2 fixed KV cache");
    let mut sequence = fixture.token_ids.clone();
    let mut generated_ids = Vec::with_capacity(R1_2_GENERATED_TOKEN_COUNT);
    let mut prompt_top20_ids = None;
    let mut prompt_argmax = None;
    let mut prompt_guard_ids = HashMap::<usize, Vec<usize>>::new();
    let mut prompt_logits_sha256 = None;
    #[cfg(feature = "m6-3-r2-native")]
    let mut prompt_logits = None;
    let mut prompt_final_norm_sha256 = None;
    let mut prompt_fixed_logit_error = None;
    let mut prompt_top20_logit_error = None;
    let mut layer0_moe_max_abs_error = 0.0_f32;
    #[cfg(feature = "m6-3-r2-native")]
    let mut native_scalar_layer0_max_abs_error = 0.0_f32;

    for position in 0..processed_positions {
        let token_id = *sequence
            .get(position)
            .expect("R1.2 generated token must be available before its position");
        assert_eq!(cache.len(), position, "R1.2 cache position");
        let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
        let mut updates = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
            let input_norm = rms_norm(
                hidden.view(),
                weights.input_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R1.2 input norm");
            let attention = cached_attention_with_weights(
                input_norm.view(),
                config,
                weights.query.view(),
                weights.key.view(),
                weights.value.view(),
                weights.output.view(),
                weights.query_norm.view(),
                weights.key_norm.view(),
                cache.layer(layer).expect("R1.2 KV layer"),
            )
            .expect("R1.2 attention");
            let residual = elementwise_add(hidden.view(), attention.output.view())
                .expect("R1.2 attention residual");
            let post_norm = rms_norm(
                residual.view(),
                weights.post_norm.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R1.2 post-attention norm");
            let router = route_tokens(post_norm.view(), weights.router.view(), config)
                .expect("R1.2 F32 router");
            if position + 1 == fixture.token_ids.len() && GENERATION_GUARD_LAYERS.contains(&layer) {
                prompt_guard_ids.insert(layer, router.selected_experts.clone());
            }
            let moe = if layer == 0 {
                let reference = streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut reference_layer0_store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("R1.2 independent F32 Layer-0 reference");
                if let Some(reader) = candidate_reader.as_deref_mut() {
                    let candidate = match candidate_mode {
                        R1_2CandidateMode::Scalar => r1_1_routed_experts_with_observer(
                            post_norm.view(),
                            &router,
                            config,
                            reader,
                            |_, _, _, _| {},
                        )
                        .expect("R1.2 direct group-32 Layer-0 candidate"),
                        #[cfg(feature = "m6-3-r2-native")]
                        R1_2CandidateMode::NativeAvx2Fma => {
                            let native = r2_1_routed_experts_with_backend(
                                post_norm.view(),
                                &router,
                                config,
                                reader,
                                R2_1PackedProjectionBackend::NativeAvx2Fma,
                            )
                            .expect("R2.1 native group-32 Layer-0 candidate");
                            let scalar = r2_1_routed_experts_with_backend(
                                post_norm.view(),
                                &router,
                                config,
                                reader,
                                R2_1PackedProjectionBackend::Scalar,
                            )
                            .expect("R2.1 scalar group-32 Layer-0 comparator");
                            let native_scalar_error = native
                                .data()
                                .iter()
                                .zip(scalar.data())
                                .map(|(left, right)| (left - right).abs())
                                .fold(0.0_f32, f32::max);
                            assert!(native_scalar_error.is_finite());
                            native_scalar_layer0_max_abs_error =
                                native_scalar_layer0_max_abs_error.max(native_scalar_error);
                            native
                        }
                    };
                    let local_error = candidate
                        .data()
                        .iter()
                        .zip(reference.data())
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(local_error.is_finite(), "R1.2 Layer-0 error must be finite");
                    layer0_moe_max_abs_error = layer0_moe_max_abs_error.max(local_error);
                    candidate
                } else {
                    reference
                }
            } else {
                streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut normal_store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .expect("R1.2 canonical F32 experts")
            };
            hidden = elementwise_add(residual.view(), moe.view()).expect("R1.2 block output");
            updates.push((attention.key, attention.value));
        }
        let update_views = updates
            .iter()
            .map(|(key, value)| LayerKvUpdate { key, value })
            .collect::<Vec<_>>();
        cache
            .append_token(&update_views)
            .expect("R1.2 transactional KV append");

        if position + 1 >= fixture.token_ids.len() {
            let normalized = rms_norm(
                hidden.view(),
                final_norm_weight.view(),
                config.rms_norm_epsilon(),
            )
            .expect("R1.2 final norm");
            let logits = streaming_language_model_head(
                &mut payload,
                &final_plan,
                &normalized,
                &mut dense_bytes_read,
            );
            assert!(
                logits.data().iter().all(|value| value.is_finite()),
                "R1.2 logits must be finite"
            );
            let greedy = greedy_token(logits.view()).expect("R1.2 finite greedy logits");
            if position + 1 == fixture.token_ids.len() {
                prompt_top20_ids = Some(deterministic_top_ids(&logits, 20));
                prompt_argmax = Some(greedy);
                prompt_logits_sha256 = Some(f32_little_endian_sha256(logits.data()));
                #[cfg(feature = "m6-3-r2-native")]
                {
                    prompt_logits = Some(logits.data().to_vec());
                }
                prompt_final_norm_sha256 = Some(f32_little_endian_sha256(normalized.data()));
                prompt_fixed_logit_error = Some(maximum_indexed_difference(
                    logits.data(),
                    &fixture.fixed_logit_indices,
                    &fixture.fixed_logits,
                ));
                prompt_top20_logit_error = Some(maximum_indexed_difference(
                    logits.data(),
                    &fixture.top20_ids,
                    &fixture.top20_logits,
                ));
            }
            generated_ids.push(greedy);
            if generated_ids.len() < R1_2_GENERATED_TOKEN_COUNT {
                sequence.push(greedy);
            }
        }
    }

    assert_eq!(generated_ids.len(), R1_2_GENERATED_TOKEN_COUNT);
    assert_eq!(cache.len(), processed_positions);
    R1_2Run {
        prompt_top20_ids: prompt_top20_ids.expect("R1.2 prompt top-20"),
        prompt_argmax: prompt_argmax.expect("R1.2 prompt argmax"),
        prompt_guard_ids,
        prompt_logits_sha256: prompt_logits_sha256.expect("R1.2 prompt logits hash"),
        #[cfg(feature = "m6-3-r2-native")]
        prompt_logits: prompt_logits.expect("R1.2 prompt logits"),
        prompt_final_norm_sha256: prompt_final_norm_sha256.expect("R1.2 prompt norm hash"),
        prompt_fixed_logit_error: prompt_fixed_logit_error.expect("R1.2 fixed-logit error"),
        prompt_top20_logit_error: prompt_top20_logit_error.expect("R1.2 top-20 logit error"),
        generated_ids,
        layer0_moe_max_abs_error,
        #[cfg(feature = "m6-3-r2-native")]
        native_scalar_layer0_max_abs_error,
    }
}

fn r1_2_assert_frozen_prompt_contract(fixture: &TierBReference, run: &R1_2Run) {
    assert_eq!(
        run.prompt_top20_ids, fixture.top20_ids,
        "R1.2 frozen prompt top-20 IDs"
    );
    assert_eq!(
        run.prompt_argmax, fixture.argmax,
        "R1.2 frozen prompt greedy ID"
    );
    assert_eq!(
        run.generated_ids[0], fixture.argmax,
        "R1.2 first generated token"
    );
    for layer in GENERATION_GUARD_LAYERS {
        assert_eq!(
            run.prompt_guard_ids[&layer], fixture.guard_ids[&layer],
            "R1.2 frozen Layer-{layer} router IDs"
        );
    }
}

fn r1_2_frozen_references(path: &Path) -> HashMap<String, R1_2FrozenReference> {
    let text = fs::read_to_string(path).expect("read R1.2 frozen F32 reference");
    let mut references = HashMap::new();
    for line in text.lines().skip(1) {
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 9, "R1.2 reference TSV field count");
        let generated_ids = parse_usize_list(fields[2]);
        let mut guards = HashMap::new();
        guards.insert(0, parse_usize_list(fields[4]));
        guards.insert(24, parse_usize_list(fields[5]));
        guards.insert(47, parse_usize_list(fields[6]));
        let record = R1_2FrozenReference {
            token_ids: parse_usize_list(fields[1]),
            prompt_argmax: generated_ids[0],
            generated_ids,
            prompt_top20_ids: parse_usize_list(fields[3]),
            prompt_guard_ids: guards,
            prompt_logits_sha256: fields[7].to_owned(),
            prompt_final_norm_sha256: fields[8].to_owned(),
        };
        assert!(references.insert(fields[0].to_owned(), record).is_none());
    }
    assert_eq!(references.len(), 2, "R1.2 frozen fixture count");
    references
}

#[test]
fn m6_3_r1_2_freeze_reference_f32_multitoken() {
    let output_path = PathBuf::from(
        env::var_os("COLIBRI_R1_2_REFERENCE_OUTPUT").expect("R1.2 reference output path"),
    );
    assert!(!output_path.exists(), "R1.2 reference output must be new");
    let mut evidence = String::from(
        "fixture\ttoken_ids\tgenerated_ids\tprompt_top20_ids\tguard_layer0_ids\tguard_layer24_ids\tguard_layer47_ids\tprompt_logits_sha256\tprompt_final_norm_sha256\n",
    );
    for fixture in tier_b_references()
        .into_iter()
        .filter(|fixture| fixture.name == "short_english" || fixture.name == "short_thai")
    {
        let run = r1_2_fixture_run(&fixture, None);
        r1_2_assert_frozen_prompt_contract(&fixture, &run);
        let compact_error = run
            .prompt_fixed_logit_error
            .max(run.prompt_top20_logit_error);
        assert!(
            compact_error <= tier_b_fixture_budget(&fixture.name, "logits"),
            "R1.2 F32 reference must reproduce the existing frozen compact-logit budget"
        );
        assert_eq!(run.layer0_moe_max_abs_error, 0.0);
        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            fixture.name,
            comma_separated(&fixture.token_ids),
            comma_separated(&run.generated_ids),
            comma_separated(&run.prompt_top20_ids),
            comma_separated(&run.prompt_guard_ids[&0]),
            comma_separated(&run.prompt_guard_ids[&24]),
            comma_separated(&run.prompt_guard_ids[&47]),
            run.prompt_logits_sha256,
            run.prompt_final_norm_sha256,
        )
        .expect("write R1.2 F32 reference evidence");
    }
    fs::write(&output_path, evidence).expect("freeze R1.2 F32 reference evidence");
}

#[test]
fn m6_3_r1_2_group32_matches_held_out_quality() {
    let reference_path = PathBuf::from(
        env::var_os("COLIBRI_R1_2_REFERENCE_PATH").expect("R1.2 frozen reference path"),
    );
    let result_path =
        PathBuf::from(env::var_os("COLIBRI_R1_2_RESULT_OUTPUT").expect("R1.2 result output path"));
    let candidate_path = PathBuf::from(
        env::var_os("COLIBRI_R1_2_CANDIDATE_PATH").expect("R1.2 group-32 candidate path"),
    );
    assert!(!result_path.exists(), "R1.2 result output must be new");
    assert_eq!(
        fs::metadata(&candidate_path)
            .expect("R1.2 candidate metadata")
            .len(),
        R1_2_GROUP32_ARTIFACT_BYTES,
        "R1.2 admitted candidate byte length"
    );
    let references = r1_2_frozen_references(&reference_path);
    let mut candidate_reader = R1_1CandidateReader::open(
        &candidate_path,
        R1_1PackedArtifactLayout::canonical(32).expect("R1.2 group-32 layout"),
        R1_2_GROUP32_ARTIFACT_SHA256,
    )
    .expect("R1.2 admitted candidate artifact");
    assert_eq!(
        candidate_reader.verification_bytes_read(),
        R1_2_GROUP32_ARTIFACT_BYTES,
        "R1.2 candidate full-file verification"
    );

    let mut evidence = String::from(
        "fixture\ttoken_ids\tgenerated_ids\tprompt_argmax\tprompt_top20_ids\tguard_layer0_ids\tguard_layer24_ids\tguard_layer47_ids\tprompt_fixed_logit_max_abs\tprompt_top20_logit_max_abs\tallowed_logit_max_abs\tlayer0_moe_max_abs\tprompt_logits_sha256\tprompt_final_norm_sha256\trepeatability\n",
    );
    for fixture in tier_b_references()
        .into_iter()
        .filter(|fixture| fixture.name == "short_english" || fixture.name == "short_thai")
    {
        let frozen = &references[&fixture.name];
        assert_eq!(
            frozen.token_ids, fixture.token_ids,
            "R1.2 frozen prompt identity"
        );
        assert_eq!(
            frozen.prompt_top20_ids, fixture.top20_ids,
            "R1.2 frozen top-20 identity"
        );
        assert_eq!(
            frozen.prompt_argmax, fixture.argmax,
            "R1.2 frozen greedy identity"
        );
        for layer in GENERATION_GUARD_LAYERS {
            assert_eq!(
                frozen.prompt_guard_ids[&layer], fixture.guard_ids[&layer],
                "R1.2 frozen router identity"
            );
        }

        let first = r1_2_fixture_run(&fixture, Some(&mut candidate_reader));
        let second = r1_2_fixture_run(&fixture, Some(&mut candidate_reader));
        r1_2_assert_frozen_prompt_contract(&fixture, &first);
        r1_2_assert_frozen_prompt_contract(&fixture, &second);
        assert_eq!(
            first.generated_ids, frozen.generated_ids,
            "R1.2 frozen multi-token sequence"
        );
        assert_eq!(
            second.generated_ids, frozen.generated_ids,
            "R1.2 repeated multi-token sequence"
        );
        assert_eq!(
            first.prompt_top20_ids, second.prompt_top20_ids,
            "R1.2 top-20 repeatability"
        );
        assert_eq!(
            first.prompt_argmax, second.prompt_argmax,
            "R1.2 greedy repeatability"
        );
        assert_eq!(
            first.prompt_guard_ids, second.prompt_guard_ids,
            "R1.2 router repeatability"
        );
        assert_eq!(
            first.prompt_logits_sha256, second.prompt_logits_sha256,
            "R1.2 logit repeatability"
        );
        assert_eq!(
            first.prompt_final_norm_sha256, second.prompt_final_norm_sha256,
            "R1.2 final-norm repeatability"
        );
        assert_eq!(
            first.layer0_moe_max_abs_error.to_bits(),
            second.layer0_moe_max_abs_error.to_bits(),
            "R1.2 Layer-0 error repeatability"
        );
        assert!(first.layer0_moe_max_abs_error.is_finite());
        assert!(first.layer0_moe_max_abs_error > 0.0);
        let observed_logit_error = first
            .prompt_fixed_logit_error
            .max(first.prompt_top20_logit_error);
        let allowed_logit_error = R1_2_LOGIT_MAX_ABS_ENVELOPE.min(fixture.margin / 4.0);
        assert!(
            observed_logit_error <= allowed_logit_error,
            "{} R1.2 prompt logit error {observed_logit_error} exceeds {allowed_logit_error}",
            fixture.name
        );
        assert_ne!(
            first.prompt_logits_sha256, frozen.prompt_logits_sha256,
            "R1.2 candidate should remain a numerically distinct quantized path"
        );
        assert_ne!(
            first.prompt_final_norm_sha256, frozen.prompt_final_norm_sha256,
            "R1.2 candidate final norm should reflect the quantized Layer-0 path"
        );
        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\texact",
            fixture.name,
            comma_separated(&fixture.token_ids),
            comma_separated(&first.generated_ids),
            first.prompt_argmax,
            comma_separated(&first.prompt_top20_ids),
            comma_separated(&first.prompt_guard_ids[&0]),
            comma_separated(&first.prompt_guard_ids[&24]),
            comma_separated(&first.prompt_guard_ids[&47]),
            first.prompt_fixed_logit_error,
            first.prompt_top20_logit_error,
            allowed_logit_error,
            first.layer0_moe_max_abs_error,
            first.prompt_logits_sha256,
            first.prompt_final_norm_sha256,
        )
        .expect("write R1.2 candidate evidence");
    }
    assert!(candidate_reader.payload_bytes_read() > 0);
    assert_eq!(
        candidate_reader.peak_packed_expert_bytes(),
        5_308_416,
        "R1.2 peak direct packed expert bytes"
    );
    fs::write(&result_path, evidence).expect("write R1.2 candidate evidence");
}

#[test]
fn m4_3_01_tier_b_full_forward_matches_transformers_f32() {
    let fixture_filter = env::var("COLIBRI_TIER_B_FIXTURE").ok();
    let retain_diagnostic = env::var_os("COLIBRI_TIER_B_RETAIN_DIAGNOSTIC").is_some();
    let artifact_root = env::var_os("COLIBRI_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .expect("COLIBRI_ARTIFACT_ROOT must name the stable canonical artifact");
    assert!(
        artifact_root.is_absolute(),
        "artifact root must be absolute"
    );
    let diagnostic_root = env::var_os("COLIBRI_RMS_DIAGNOSTIC_ROOT")
        .map(PathBuf::from)
        .expect("diagnostic root for Tier B evidence");
    assert!(
        diagnostic_root.is_absolute(),
        "diagnostic root must be absolute"
    );

    let plan = runtime_plan(LAYER47_RUNTIME_PLAN);
    let final_plan = runtime_plan(GENERATION_FINAL_DENSE_RUNTIME_PLAN);
    assert_eq!(plan.payload, final_plan.payload, "dense payload identity");
    assert_eq!(
        plan.payload_length, final_plan.payload_length,
        "dense payload length"
    );
    let mut payload = File::open(artifact_root.join(&plan.payload)).expect("open dense payload");
    assert_eq!(
        payload.metadata().expect("dense payload metadata").len(),
        plan.payload_length
    );
    let config = PINNED_QWEN3_30B_A3B_CONFIG
        .map_to_f32_runtime()
        .expect("pinned runtime config")
        .runtime_config();
    let expert_layout = PackedExpertLayout::for_config(config);
    let mut dense_bytes_read = 0_u64;
    let final_norm_weight = artifact_tensor(
        &mut payload,
        &final_plan,
        "model.norm.weight",
        &mut dense_bytes_read,
    );
    let mut evidence = String::from(
        "fixture\ttoken_ids\tfinal_norm_reference_sha256\tfinal_norm_rust_sha256\tmaximum_fixed_final_norm_error\tfinal_norm_fixture_budget\tmaximum_fixed_logit_error\tmaximum_top20_logit_error\tlogit_fixture_budget\targmax\ttop1_margin\trequired_safe_margin\tclassification\ttop20_ids\tguard_layer0_ids\tguard_layer24_ids\tguard_layer47_ids\tkv_cache_bytes\tdense_bytes_read\texpert_bytes_read\texpert_loads\texpert_evictions\n",
    );

    for fixture in tier_b_references().into_iter().filter(|fixture| {
        fixture_filter
            .as_ref()
            .is_none_or(|filter| filter == &fixture.name)
    }) {
        let fixture_dense_before = dense_bytes_read;
        let mut store = expert_store_from_plans(
            &[
                LAYER47_EXPERT_RUNTIME_PLAN,
                GENERATION_LAYER47_EXPERT_RUNTIME_PLAN,
            ],
            &artifact_root,
            48 * 128,
        );
        let mut cache =
            KvCache::new(48, fixture.token_ids.len(), 4, 128).expect("fixed Tier B KV cache");
        let allocation_capacities = cache.allocation_capacities();
        let mut final_guards = HashMap::<usize, Vec<usize>>::new();
        let mut current = None;

        for (position, &token_id) in fixture.token_ids.iter().enumerate() {
            assert_eq!(cache.len(), position, "Tier B cache position");
            let mut hidden = embedding_row(&mut payload, &plan, token_id, &mut dense_bytes_read);
            let mut updates = Vec::with_capacity(48);
            for layer in 0..48 {
                let weights = layer_weights(&mut payload, &plan, layer, &mut dense_bytes_read);
                let input_norm = rms_norm(
                    hidden.view(),
                    weights.input_norm.view(),
                    config.rms_norm_epsilon(),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "{} position-{position} Layer-{layer} input norm: {error}",
                        fixture.name
                    )
                });
                let attention = cached_attention_with_weights(
                    input_norm.view(),
                    config,
                    weights.query.view(),
                    weights.key.view(),
                    weights.value.view(),
                    weights.output.view(),
                    weights.query_norm.view(),
                    weights.key_norm.view(),
                    cache.layer(layer).expect("Tier B KV layer view"),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "{} position-{position} Layer-{layer} attention: {error}",
                        fixture.name
                    )
                });
                let residual = elementwise_add(hidden.view(), attention.output.view())
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} position-{position} Layer-{layer} attention residual: {error}",
                            fixture.name
                        )
                    });
                let post_norm = rms_norm(
                    residual.view(),
                    weights.post_norm.view(),
                    config.rms_norm_epsilon(),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "{} position-{position} Layer-{layer} post norm: {error}",
                        fixture.name
                    )
                });
                let router = route_tokens(post_norm.view(), weights.router.view(), config)
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} position-{position} Layer-{layer} router: {error}",
                            fixture.name
                        )
                    });
                if position + 1 == fixture.token_ids.len()
                    && GENERATION_GUARD_LAYERS.contains(&layer)
                {
                    final_guards.insert(layer, router.selected_experts.clone());
                }
                let moe = streaming_routed_experts_with_observer(
                    post_norm.view(),
                    &router,
                    config,
                    layer,
                    &mut store,
                    expert_layout,
                    |_, _, _, _| {},
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "{} position-{position} Layer-{layer} experts: {error}",
                        fixture.name
                    )
                });
                hidden = elementwise_add(residual.view(), moe.view()).unwrap_or_else(|error| {
                    panic!(
                        "{} position-{position} Layer-{layer} block: {error}",
                        fixture.name
                    )
                });
                updates.push((attention.key, attention.value));
            }
            let updates_view: Vec<_> = updates
                .iter()
                .map(|(key, value)| LayerKvUpdate { key, value })
                .collect();
            cache
                .append_token(&updates_view)
                .expect("transactional Tier B KV append");
            assert_eq!(cache.allocation_capacities(), allocation_capacities);
            current = Some(hidden);
        }

        let normalized = rms_norm(
            current.expect("non-empty Tier B fixture").view(),
            final_norm_weight.view(),
            config.rms_norm_epsilon(),
        )
        .unwrap_or_else(|error| panic!("{} final RMSNorm: {error}", fixture.name));
        let logits = streaming_language_model_head(
            &mut payload,
            &final_plan,
            &normalized,
            &mut dense_bytes_read,
        );
        let rust_top20 = deterministic_top_ids(&logits, 20);
        assert_eq!(rust_top20, fixture.top20_ids, "{} top-20 IDs", fixture.name);
        let rust_argmax = greedy_token(logits.view()).expect("finite Tier B logits");
        assert_eq!(rust_argmax, fixture.argmax, "{} argmax", fixture.name);
        for layer in GENERATION_GUARD_LAYERS {
            assert_eq!(
                final_guards[&layer], fixture.guard_ids[&layer],
                "{} Layer-{layer} guard IDs",
                fixture.name
            );
        }

        let norm_error = maximum_indexed_difference(
            normalized.data(),
            &fixture.final_norm_indices,
            &fixture.final_norm_values,
        );
        let norm_budget = tier_b_fixture_budget(&fixture.name, "final_norm");
        assert!(
            norm_error <= norm_budget,
            "{} fixed final-norm error {norm_error} exceeds fixture budget {norm_budget}",
            fixture.name
        );
        let fixed_logit_error = maximum_indexed_difference(
            logits.data(),
            &fixture.fixed_logit_indices,
            &fixture.fixed_logits,
        );
        let top20_logit_error =
            maximum_indexed_difference(logits.data(), &fixture.top20_ids, &fixture.top20_logits);
        let observed_logit_error = fixed_logit_error.max(top20_logit_error);
        let logit_budget = tier_b_fixture_budget(&fixture.name, "logits");
        if retain_diagnostic {
            atomic_diagnostic(
                &diagnostic_root.join(format!("{}-rust-final-norm-f32.bin", fixture.name)),
                &f32_little_endian_bytes(normalized.data()),
            );
            atomic_diagnostic(
                &diagnostic_root.join(format!("{}-rust-logits-f32.bin", fixture.name)),
                &f32_little_endian_bytes(logits.data()),
            );
        }
        assert!(
            observed_logit_error <= logit_budget,
            "{} compact logit error {observed_logit_error} exceeds fixture budget {logit_budget}",
            fixture.name
        );
        let required_margin = 2.0 * observed_logit_error;
        assert!(
            fixture.margin > required_margin,
            "{} top-1 margin is not safe for compact compared logits",
            fixture.name
        );
        assert_eq!(
            logits.data().iter().filter(|value| value.is_nan()).count(),
            fixture.nan_count,
            "{} NaN count",
            fixture.name
        );
        assert_eq!(
            logits
                .data()
                .iter()
                .filter(|value| value.is_infinite() && value.is_sign_positive())
                .count(),
            fixture.positive_infinity_count,
            "{} positive infinity count",
            fixture.name
        );
        assert_eq!(
            logits
                .data()
                .iter()
                .filter(|value| value.is_infinite() && value.is_sign_negative())
                .count(),
            fixture.negative_infinity_count,
            "{} negative infinity count",
            fixture.name
        );
        assert_eq!(cache.len(), fixture.token_ids.len());
        assert_eq!(cache.allocation_capacities(), allocation_capacities);
        let metrics = store.metrics();
        let expected_expert_occurrences = (fixture.token_ids.len() * 48 * 8) as u64;
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, expected_expert_occurrences);
        assert_eq!(metrics.loads, expected_expert_occurrences);
        assert_eq!(metrics.evictions, expected_expert_occurrences - 1);
        assert_eq!(metrics.peak_resident_bytes, 18_874_368);

        writeln!(
            evidence,
            "{}\t{}\t{}\t{}\t{norm_error:.17e}\t{norm_budget:.17e}\t{fixed_logit_error:.17e}\t{top20_logit_error:.17e}\t{logit_budget:.17e}\t{rust_argmax}\t{:.17e}\t{required_margin:.17e}\texact_match_safe_compact\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            fixture.name,
            comma_separated(&fixture.token_ids),
            fixture.final_norm_sha256,
            f32_little_endian_sha256(normalized.data()),
            fixture.margin,
            comma_separated(&rust_top20),
            comma_separated(&final_guards[&0]),
            comma_separated(&final_guards[&24]),
            comma_separated(&final_guards[&47]),
            cache.byte_size(),
            dense_bytes_read - fixture_dense_before,
            metrics.bytes_read,
            metrics.loads,
            metrics.evictions,
        )
        .expect("write Tier B evidence");
        println!(
            "tier_b fixture={} tokens={} argmax={} margin={} required_margin={} norm_error={} compact_logit_error={}",
            fixture.name,
            fixture.token_ids.len(),
            rust_argmax,
            fixture.margin,
            required_margin,
            norm_error,
            observed_logit_error,
        );
    }
    atomic_diagnostic(
        &diagnostic_root.join("m4.3-01-rust-tier-b-evidence-v1.tsv"),
        evidence.as_bytes(),
    );
}
