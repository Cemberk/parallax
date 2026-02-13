.PHONY: all clean runtime pass differ test install help

BUILD_DIR := build

all: runtime pass differ

# Configure CMake
configure:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..

# Build runtime library
runtime: configure
	@echo "Building runtime library..."
	@cd $(BUILD_DIR) && cmake --build . --target gddbg_runtime -j$(shell nproc)

# Build LLVM pass
pass: configure
	@echo "Building LLVM pass..."
	@cd $(BUILD_DIR) && cmake --build . --target GpuDiffDbgPass -j$(shell nproc) || \
		echo "WARNING: LLVM pass build failed (LLVM 18+ required)"

# Build Rust differ tool
differ:
	@echo "Building differ tool..."
	@cd tools/gddbg-diff && cargo build --release
	@mkdir -p $(BUILD_DIR)/bin
	@cp tools/gddbg-diff/target/release/gddbg-diff $(BUILD_DIR)/bin/

# Build and run tests
test: runtime
	@echo "Building tests..."
	@cd $(BUILD_DIR) && cmake --build . --target manual_trace_test -j$(shell nproc)
	@echo "Running tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)
	@cd tools/gddbg-diff && cargo clean

# Install to system
install: all
	@cd $(BUILD_DIR) && cmake --install .
	@install -m 755 $(BUILD_DIR)/bin/gddbg-diff /usr/local/bin/

# Help
help:
	@echo "GPU DiffDbg Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build everything (default)"
	@echo "  runtime   - Build runtime library only"
	@echo "  pass      - Build LLVM pass only"
	@echo "  differ    - Build Rust differ tool only"
	@echo "  test      - Build and run tests"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to system (/usr/local)"
	@echo "  help      - Show this message"
	@echo ""
	@echo "Requirements:"
	@echo "  - LLVM 18+ (for pass)"
	@echo "  - CUDA Toolkit 12.0+"
	@echo "  - Rust (latest stable)"
	@echo "  - CMake 3.20+"
