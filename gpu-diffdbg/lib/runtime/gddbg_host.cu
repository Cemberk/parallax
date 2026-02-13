#include "gddbg_runtime.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ctime>

// Global state
static void* d_trace_buffer = nullptr;
static size_t trace_buffer_size = 0;
static char output_path[256] = "trace.gddbg";
static bool initialized = false;
static bool enabled = true;

// Configuration from environment variables
static uint32_t events_per_warp = GDDBG_EVENTS_PER_WARP;

// FNV-1a hash for kernel name (64-bit)
static uint64_t fnv1a_hash(const char* str) {
    uint64_t hash = 14695981039346656037ULL;
    while (*str) {
        hash ^= (uint64_t)(*str++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Initialize tracing system (called once at startup)
extern "C" void gddbg_init(void) {
    if (initialized) return;

    // Read configuration from environment
    const char* trace_path = getenv("GDDBG_TRACE");
    if (trace_path) {
        strncpy(output_path, trace_path, sizeof(output_path) - 1);
        output_path[sizeof(output_path) - 1] = '\0';
    }

    const char* enabled_str = getenv("GDDBG_ENABLED");
    enabled = !enabled_str || atoi(enabled_str) != 0;

    const char* buffer_size_str = getenv("GDDBG_BUFFER_SIZE");
    if (buffer_size_str) {
        events_per_warp = atoi(buffer_size_str);
    }

    initialized = true;

    if (enabled) {
        fprintf(stderr, "[gddbg] Tracing enabled, output: %s\n", output_path);
    }
}

// Pre-launch hook: allocate and set up trace buffer
extern "C" void gddbg_pre_launch(const char* kernel_name, dim3 gridDim, dim3 blockDim) {
    if (!initialized) gddbg_init();
    if (!enabled) return;

    // Calculate number of warps
    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t warps_per_block = (threads_per_block + 31) / 32;
    uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
    uint32_t total_warps = total_blocks * warps_per_block;

    // Calculate buffer size
    size_t warp_buffer_size = sizeof(WarpBufferHeader) + events_per_warp * sizeof(TraceEvent);
    trace_buffer_size = sizeof(TraceFileHeader) + total_warps * warp_buffer_size;

    fprintf(stderr, "[gddbg] Allocating trace buffer: %zu MB for %u warps\n",
            trace_buffer_size / (1024*1024), total_warps);

    // Allocate device buffer
    cudaError_t err = cudaMalloc(&d_trace_buffer, trace_buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gddbg] ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        enabled = false;
        return;
    }

    // Zero the buffer
    cudaMemset(d_trace_buffer, 0, trace_buffer_size);

    // Prepare file header
    TraceFileHeader header = {};
    header.magic = GDDBG_MAGIC;
    header.version = GDDBG_VERSION;
    header.flags = 0;
    header.kernel_name_hash = fnv1a_hash(kernel_name);
    strncpy(header.kernel_name, kernel_name, sizeof(header.kernel_name) - 1);
    header.grid_dim[0] = gridDim.x;
    header.grid_dim[1] = gridDim.y;
    header.grid_dim[2] = gridDim.z;
    header.block_dim[0] = blockDim.x;
    header.block_dim[1] = blockDim.y;
    header.block_dim[2] = blockDim.z;
    header.num_warps_per_block = warps_per_block;
    header.total_warp_slots = total_warps;
    header.events_per_warp = events_per_warp;
    header.timestamp = time(nullptr);

    // Get CUDA arch (SM version)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    header.cuda_arch = prop.major * 10 + prop.minor;

    // Write header to device buffer
    cudaMemcpy(d_trace_buffer, &header, sizeof(header), cudaMemcpyHostToDevice);

    // Set the global device variable g_gddbg_buffer to point to this buffer
    // This is the key technique to avoid modifying kernel signatures (Death Valley 2)
    // Declare the device symbol so we can reference it
    extern __device__ TraceBuffer* g_gddbg_buffer;
    err = cudaMemcpyToSymbol(g_gddbg_buffer, &d_trace_buffer, sizeof(void*));
    if (err != cudaSuccess) {
        fprintf(stderr, "[gddbg] ERROR: cudaMemcpyToSymbol failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_trace_buffer);
        d_trace_buffer = nullptr;
        enabled = false;
        return;
    }

    fprintf(stderr, "[gddbg] Trace buffer ready for kernel: %s\n", kernel_name);
}

// Post-launch hook: copy buffer to host and write to file
extern "C" void gddbg_post_launch(void) {
    if (!enabled || !d_trace_buffer) return;

    fprintf(stderr, "[gddbg] Copying trace buffer from device...\n");

    // Allocate host buffer
    void* h_buffer = malloc(trace_buffer_size);
    if (!h_buffer) {
        fprintf(stderr, "[gddbg] ERROR: malloc failed\n");
        return;
    }

    // Copy from device to host
    cudaError_t err = cudaMemcpy(h_buffer, d_trace_buffer, trace_buffer_size,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gddbg] ERROR: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_buffer);
        return;
    }

    // Update warp buffer headers with actual event counts
    TraceFileHeader* header = (TraceFileHeader*)h_buffer;
    WarpBuffer* warps = (WarpBuffer*)((char*)h_buffer + sizeof(TraceFileHeader));

    uint64_t total_events = 0;
    uint64_t total_overflows = 0;

    for (uint32_t i = 0; i < header->total_warp_slots; i++) {
        uint32_t write_idx = warps[i].header.write_idx;
        warps[i].header.num_events = (write_idx < events_per_warp) ? write_idx : events_per_warp;
        total_events += warps[i].header.num_events;
        total_overflows += warps[i].header.overflow_count;
    }

    fprintf(stderr, "[gddbg] Recorded %lu events across %u warps (%lu overflows)\n",
            total_events, header->total_warp_slots, total_overflows);

    // Write to file
    FILE* f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "[gddbg] ERROR: cannot open output file: %s\n", output_path);
        free(h_buffer);
        return;
    }

    size_t written = fwrite(h_buffer, 1, trace_buffer_size, f);
    fclose(f);

    if (written != trace_buffer_size) {
        fprintf(stderr, "[gddbg] ERROR: incomplete write (%zu of %zu bytes)\n",
                written, trace_buffer_size);
    } else {
        fprintf(stderr, "[gddbg] Trace written to: %s (%zu MB)\n",
                output_path, trace_buffer_size / (1024*1024));
    }

    free(h_buffer);

    // Free device buffer
    cudaFree(d_trace_buffer);
    d_trace_buffer = nullptr;

    // Clear the global device pointer
    void* null_ptr = nullptr;
    cudaMemcpyToSymbol(g_gddbg_buffer, &null_ptr, sizeof(void*));
}

// Cleanup
extern "C" void gddbg_shutdown(void) {
    if (d_trace_buffer) {
        cudaFree(d_trace_buffer);
        d_trace_buffer = nullptr;
    }
}

// Constructor to auto-initialize
__attribute__((constructor))
static void auto_init() {
    gddbg_init();
}

// Destructor to auto-cleanup
__attribute__((destructor))
static void auto_shutdown() {
    gddbg_shutdown();
}
