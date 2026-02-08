#!/bin/bash

# Cross-platform validation script for scirs2-core
# This script validates the library across different platforms

set -e

echo "=== Cross-Platform Validation for scirs2-core ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

# Detect current platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux";;
        Darwin*)    PLATFORM="macos";;
        CYGWIN*|MINGW*|MSYS*) PLATFORM="windows";;
        *)          PLATFORM="unknown";;
    esac
    
    case "$(uname -m)" in
        x86_64)     ARCH="x86_64";;
        aarch64|arm64) ARCH="aarch64";;
        *)          ARCH="unknown";;
    esac
    
    echo "Detected platform: $PLATFORM-$ARCH"
    echo
}

# Check Rust toolchain
check_rust_toolchain() {
    echo "=== Rust Toolchain Check ==="
    
    # Check if Rust is installed
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version)
        print_status 0 "Rust installed: $RUST_VERSION"
    else
        print_status 1 "Rust not installed"
        exit 1
    fi
    
    # Check if cargo is available
    if command -v cargo &> /dev/null; then
        CARGO_VERSION=$(cargo --version)
        print_status 0 "Cargo available: $CARGO_VERSION"
    else
        print_status 1 "Cargo not available"
        exit 1
    fi
    
    # Check if nextest is installed
    if command -v cargo-nextest &> /dev/null; then
        print_status 0 "cargo-nextest installed"
    else
        print_status 1 "cargo-nextest not installed (required for testing)"
        echo "  Install with: cargo install cargo-nextest"
    fi
    
    echo
}

# Check system dependencies
check_system_dependencies() {
    echo "=== System Dependencies Check ==="
    
    # OxiBLAS (pure Rust) - no system BLAS dependencies required
    print_status 0 "OxiBLAS (pure Rust) - no system BLAS dependencies required"
    
    # Check for GPU support
    if command -v nvidia-smi &> /dev/null; then
        print_status 0 "NVIDIA GPU detected (CUDA support possible)"
    elif [ "$PLATFORM" = "macos" ]; then
        print_status 0 "Metal GPU support available"
    else
        print_status 1 "No GPU support detected"
    fi
    
    echo
}

# Build tests for different configurations
run_build_tests() {
    echo "=== Build Configuration Tests ==="
    
    # Test minimal build
    echo -n "Testing minimal build... "
    if cargo build --no-default-features &> /dev/null; then
        print_status 0 "Minimal build successful"
    else
        print_status 1 "Minimal build failed"
    fi
    
    # Test default build
    echo -n "Testing default build... "
    if cargo build &> /dev/null; then
        print_status 0 "Default build successful"
    else
        print_status 1 "Default build failed"
    fi
    
    # Test all features (excluding conflicting ones)
    echo -n "Testing all features build... "
    if cargo build --features all &> /dev/null; then
        print_status 0 "All features build successful"
    else
        print_status 1 "All features build failed"
    fi
    
    # Platform-specific builds
    case "$PLATFORM" in
        linux)
            echo -n "Testing Linux-specific features... "
            if cargo build --features "memory_efficient,simd,parallel" &> /dev/null; then
                print_status 0 "Linux features build successful"
            else
                print_status 1 "Linux features build failed"
            fi
            ;;
        macos)
            echo -n "Testing macOS-specific features... "
            if cargo build --features "memory_efficient,simd,parallel,accelerate" &> /dev/null; then
                print_status 0 "macOS features build successful"
            else
                print_status 1 "macOS features build failed"
            fi
            ;;
        windows)
            echo -n "Testing Windows-specific features... "
            if cargo build --features "memory_efficient,simd,parallel" &> /dev/null; then
                print_status 0 "Windows features build successful"
            else
                print_status 1 "Windows features build failed"
            fi
            ;;
    esac
    
    echo
}

# Run platform-specific tests
run_platform_tests() {
    echo "=== Platform-Specific Tests ==="
    
    # Memory mapping tests
    echo -n "Testing memory mapping... "
    if cargo test --features memory_efficient memory_mapped --lib &> /dev/null; then
        print_status 0 "Memory mapping tests passed"
    else
        print_status 1 "Memory mapping tests failed"
    fi
    
    # SIMD tests
    echo -n "Testing SIMD operations... "
    if cargo test --features simd simd --lib &> /dev/null; then
        print_status 0 "SIMD tests passed"
    else
        print_status 1 "SIMD tests failed"
    fi
    
    # Parallel processing tests
    echo -n "Testing parallel operations... "
    if cargo test --features parallel parallel --lib &> /dev/null; then
        print_status 0 "Parallel tests passed"
    else
        print_status 1 "Parallel tests failed"
    fi
    
    echo
}

# Check for WASM support
check_wasm_support() {
    echo "=== WebAssembly (WASM) Support Check ==="
    
    # Check if wasm target is installed
    if rustup target list --installed | grep -q wasm32-unknown-unknown; then
        print_status 0 "WASM target installed"
        
        # Try to build for WASM
        echo -n "Testing WASM build... "
        if cargo build --target wasm32-unknown-unknown --no-default-features &> /dev/null; then
            print_status 0 "WASM build successful"
        else
            print_status 1 "WASM build failed"
        fi
    else
        print_status 1 "WASM target not installed"
        echo "  Install with: rustup target add wasm32-unknown-unknown"
    fi
    
    echo
}

# Generate platform report
generate_report() {
    echo "=== Generating Platform Validation Report ==="
    
    REPORT_FILE="platform_validation_report_${PLATFORM}_${ARCH}.md"
    
    cat > "$REPORT_FILE" << EOF
# Platform Validation Report

**Platform**: $PLATFORM-$ARCH
**Date**: $(date)
**Rust Version**: $(rustc --version)

## Build Configurations Tested

- [x] Minimal build (no features)
- [x] Default build
- [x] All features build
- [x] Platform-specific features

## Platform-Specific Tests

- Memory mapping support
- SIMD operations
- Parallel processing
- GPU detection

## Recommendations

1. Ensure all dependencies are installed for your platform
2. Use platform-specific feature flags for optimal performance
3. Test your specific use cases with the validated configurations

EOF
    
    print_status 0 "Report generated: $REPORT_FILE"
}

# Main execution
main() {
    detect_platform
    check_rust_toolchain
    check_system_dependencies
    run_build_tests
    run_platform_tests
    check_wasm_support
    generate_report
    
    echo
    echo "=== Validation Complete ==="
    echo "See the generated report for details."
}

# Run the validation
main