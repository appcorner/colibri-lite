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
use clr_storage::{
    ARTIFACT_FORMAT_VERSION, ArtifactManifest, ArtifactReader, ByteOrder, ExpertId, ExpertKey,
    ExpertRegistration, ExpertStore, TensorLocation, TensorMetadata,
};

use crate::{
    PINNED_QWEN3_30B_A3B_CONFIG,
    block::{ExpertMlpTrace, PreRouterOutput, pre_router_with_weights, route_tokens},
    streaming::{
        PackedExpertLayout, streaming_routed_experts_with_observer,
        streaming_routed_experts_with_trace_observer,
    },
};

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

fn read_exact_range(file: &mut File, record: &RangeRecord, bytes_read: &mut u64) -> Vec<u8> {
    file.seek(SeekFrom::Start(record.offset))
        .expect("seek artifact");
    let mut bytes = vec![0_u8; record.length];
    file.read_exact(&mut bytes).expect("read artifact range");
    *bytes_read += u64::try_from(bytes.len()).expect("range length fits u64");
    bytes
}

fn f32_tensor(bytes: &[u8], shape: &[usize]) -> Tensor {
    let data = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect();
    Tensor::new(TensorShape::new(shape.to_vec()), data).expect("valid f32 tensor")
}

fn artifact_tensor(
    file: &mut File,
    plan: &RuntimePlan,
    name: &str,
    bytes_read: &mut u64,
) -> Tensor {
    let record = plan
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing runtime tensor {name}"));
    f32_tensor(&read_exact_range(file, record, bytes_read), &record.shape)
}

fn embedding_rows(file: &mut File, plan: &RuntimePlan, bytes_read: &mut u64) -> Tensor {
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
        data.extend(f32_tensor(&read_exact_range(file, &row, bytes_read), &[2048]).into_data());
    }
    Tensor::new(TensorShape::new([4, 2048]), data).expect("embedding rows")
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
    file: &mut File,
    plan: &RuntimePlan,
    layer: usize,
    bytes_read: &mut u64,
) -> LayerWeights {
    let prefix = format!("model.layers.{layer}");
    let mut read =
        |suffix: &str| artifact_tensor(file, plan, &format!("{prefix}.{suffix}"), bytes_read);
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
    let reader = ArtifactReader::open(artifact_root.join("experts"), manifest)
        .expect("canonical selected expert reader");
    ExpertStore::new(reader, registrations, 18_874_368).expect("one-expert cache budget")
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
