# PRLX — GPU Differential Debugger
# Multi-stage build: builder + runtime

# ---- Stage 1: Build ----
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git wget ca-certificates python3 python3-pip python3-venv \
    lsb-release software-properties-common gnupg && \
    rm -rf /var/lib/apt/lists/*

# LLVM 20
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" \
        >> /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    llvm-20-dev clang-20 lld-20 && \
    rm -rf /var/lib/apt/lists/*

# Rust
RUN wget -qO- https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /src
COPY . /src

# Build C++/CUDA (pass + runtime)
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPRLX_LLVM_DIR=/usr/lib/llvm-20 \
    -DPRLX_CUDA_ARCHITECTURES="70;80;90" && \
    cmake --build build

# Build Rust differ
RUN cd differ && cargo build --release

# Build Python wheel
RUN python3 -m pip install --break-system-packages build && \
    python3 -m build --wheel --outdir /wheels

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy built artifacts
COPY --from=builder /src/build/lib/pass/libPrlxPass.so /usr/local/lib/
COPY --from=builder /src/build/lib/runtime/libprlx_runtime_shared.so /usr/local/lib/
COPY --from=builder /src/build/lib/runtime/prlx_runtime_nvptx.bc /usr/local/share/prlx/
COPY --from=builder /src/differ/target/release/prlx-diff /usr/local/bin/
COPY --from=builder /wheels/*.whl /tmp/wheels/

# Install Python package
RUN python3 -m pip install --break-system-packages /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

ENTRYPOINT ["prlx"]
