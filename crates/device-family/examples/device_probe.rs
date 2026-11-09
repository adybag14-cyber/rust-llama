//! Real, read-only driver enumeration; absence is an ordinary reported result.
fn main() {
    for report in llama_device_family::probe_all() {
        println!(
            "{}\t{:?}\tlibrary={:?}\tversion={:?}\tdevices={}\t{}",
            report.family,
            report.status,
            report.library,
            report.driver_version,
            report.devices.len(),
            report.detail
        );
        for device in report.devices {
            println!("  {device:?}");
        }
    }
    for family in [
        llama_device_family::Family::Hip,
        llama_device_family::Family::Musa,
    ] {
        match llama_device_family::runtime::RuntimeDriver::load(family) {
            Ok(_) => println!("{family} runtime resource symbols loaded; no allocation attempted"),
            Err(error) => println!("{family} runtime resource load: {error}"),
        }
    }
}
