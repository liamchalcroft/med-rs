//! Build script to ensure Python extension modules link correctly on macOS.

fn main() {
    let python_feature = std::env::var_os("CARGO_FEATURE_PYTHON").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if python_feature && target_os == "macos" {
        // Allow unresolved Python symbols for extension-module linking.
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
