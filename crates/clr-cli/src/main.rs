use clr_core::runtime_info;

fn main() {
    let info = runtime_info();

    println!("{} {}", info.name, info.version);
    println!("host: {}-{}", info.architecture, info.operating_system);
    println!("status: bootstrap ready");
}
