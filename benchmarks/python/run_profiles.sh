#!/bin/bash
# Run profiling for medrs operations
#
# Usage:
#   ./profiling/run_profiles.sh              # Run all Rust profiles
#   ./profiling/run_profiles.sh python       # Run Python profiles
#   ./profiling/run_profiles.sh flamegraph   # Generate flamegraphs (requires cargo-flamegraph)

set -e

cd "$(dirname "$0")/.."

case "${1:-rust}" in
    rust)
        echo "Running Rust profiles..."
        echo ""

        echo "=== profile_all (32^3, 128^3, 256^3) ==="
        cargo run --example profile_all --release

        echo ""
        echo "=== Individual operation profiles at 256^3 ==="

        echo ""
        echo "--- Z-Normalization ---"
        cargo run --example profile_z_norm --release -- 256 20

        echo ""
        echo "--- Rescale Intensity ---"
        cargo run --example profile_rescale --release -- 256 20

        echo ""
        echo "--- Crop/Pad ---"
        cargo run --example profile_crop_pad --release -- 256 20

        echo ""
        echo "--- Flip ---"
        cargo run --example profile_flip --release -- 256 20

        echo ""
        echo "--- Load/Save ---"
        cargo run --example profile_load --release -- 256 20

        echo ""
        echo "--- Resample ---"
        cargo run --example profile_resample --release -- 128 10
        ;;

    python)
        echo "Running Python profiles..."
        uv run python profiling/profile_ops.py
        ;;

    flamegraph)
        # Generate flamegraphs for specific operations
        # Requires: cargo install flamegraph
        echo "Generating flamegraphs (this may take a while)..."

        mkdir -p profiling/flamegraphs

        echo "Profiling crop_or_pad..."
        cargo flamegraph --example profile_crop_pad --release -o profiling/flamegraphs/crop_pad_256.svg -- 256 50

        echo "Profiling flip..."
        cargo flamegraph --example profile_flip --release -o profiling/flamegraphs/flip_256.svg -- 256 50

        echo "Profiling z_norm..."
        cargo flamegraph --example profile_z_norm --release -o profiling/flamegraphs/z_norm_256.svg -- 256 50

        echo "Profiling rescale..."
        cargo flamegraph --example profile_rescale --release -o profiling/flamegraphs/rescale_256.svg -- 256 50

        echo ""
        echo "Flamegraphs saved to profiling/flamegraphs/"
        ;;

    samply)
        # Use samply for macOS profiling
        # Requires: cargo install samply
        OP="${2:-crop_pad}"
        SIZE="${3:-256}"
        echo "Profiling with samply: $OP at ${SIZE}^3"
        cargo build --example "profile_${OP}" --release
        samply record ./target/release/examples/profile_${OP} ${SIZE} 50
        ;;

    *)
        echo "Usage: $0 [rust|python|flamegraph|samply]"
        echo ""
        echo "  rust       - Run Rust profiles (default)"
        echo "  python     - Run Python profiles"
        echo "  flamegraph - Generate SVG flamegraphs"
        echo "  samply OP SIZE - Profile with samply (macOS)"
        exit 1
        ;;
esac
