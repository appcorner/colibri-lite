use std::env;

fn main() {
    println!("cargo:rerun-if-changed=native/r2_group32_avx2_fma.c");

    if env::var_os("CARGO_FEATURE_M6_3_R2_NATIVE").is_none() {
        return;
    }

    let target = env::var("TARGET").expect("Cargo TARGET");
    if target != "x86_64-pc-windows-msvc" {
        return;
    }

    cc::Build::new()
        .file("native/r2_group32_avx2_fma.c")
        .flag("/O2")
        .flag("/arch:AVX2")
        .flag("/fp:precise")
        .compile("clr_qwen3_moe_r2_native");
}
