#include "gddbg_runtime.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ctime>

// Global state
static void* d_trace_buffer = nullptr;
static size_t trace_buffer_size = 0;
static void* d_history_buffer = nullptr;
static size_t history_buffer_size = 0;
static uint32_t history_depth = 0;
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

    const char* history_depth_str = getenv("GDDBG_HISTORY_DEPTH");
    if (history_depth_str) {
        history_depth = atoi(history_depth_str);
    } else {
        history_depth = GDDBG_HISTORY_DEPTH_DEFAULT;
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
    header.flags = (history_depth > 0) ? GDDBG_FLAG_HISTORY : 0;
    header.history_depth = history_depth;
    header.history_section_offset = 0;  // 0 = immediately after event buffers
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

    // Allocate history ring buffer if enabled
    if (history_depth > 0) {
        size_t ring_size = sizeof(HistoryRingHeader) + history_depth * sizeof(HistoryEntry);
        history_buffer_size = total_warps * ring_size;

        err = cudaMalloc(&d_history_buffer, history_buffer_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[gddbg] WARNING: history buffer allocation failed: %s (disabling history)\n",
                    cudaGetErrorString(err));
            d_history_buffer = nullptr;
            history_buffer_size = 0;
            // Update header to remove history flag
            header.flags &= ~GDDBG_FLAG_HISTORY;
            header.history_depth = 0;
            cudaMemcpy(d_trace_buffer, &header, sizeof(header), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(d_history_buffer, 0, history_buffer_size);

            // Initialize each ring header with the depth
            void* h_ring_init = malloc(ring_size);
            memset(h_ring_init, 0, ring_size);
            ((HistoryRingHeader*)h_ring_init)->depth = history_depth;
            for (uint32_t w = 0; w < total_warps; w++) {
                cudaMemcpy((char*)d_history_buffer + w * ring_size, h_ring_init,
                           sizeof(HistoryRingHeader), cudaMemcpyHostToDevice);
            }
            free(h_ring_init);

            // Set device globals
            extern __device__ char* g_gddbg_history_buffer;
            extern __device__ uint32_t g_gddbg_history_depth;
            cudaMemcpyToSymbol(g_gddbg_history_buffer, &d_history_buffer, sizeof(void*));
            cudaMemcpyToSymbol(g_gddbg_history_depth, &history_depth, sizeof(uint32_t));

            fprintf(stderr, "[gddbg] History buffer: %zu KB (%u entries/warp, %u warps)\n",
                    history_buffer_size / 1024, history_depth, total_warps);
        }
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

    // Copy history buffer if present
    void* h_history = nullptr;
    if (d_history_buffer && history_buffer_size > 0) {
        h_history = malloc(history_buffer_size);
        if (h_history) {
            err = cudaMemcpy(h_history, d_history_buffer, history_buffer_size,
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "[gddbg] WARNING: history copy failed: %s\n",
                        cudaGetErrorString(err));
                free(h_history);
                h_history = nullptr;
            } else {
                uint64_t total_hist = 0;
                size_t ring_size = sizeof(HistoryRingHeader)
                                 + history_depth * sizeof(HistoryEntry);
                for (uint32_t i = 0; i < header->total_warp_slots; i++) {
                    HistoryRingHeader* ring = (HistoryRingHeader*)((char*)h_history + i * ring_size);
                    total_hist += (ring->total_writes < history_depth)
                                ? ring->total_writes : history_depth;
                }
                fprintf(stderr, "[gddbg] History: %lu entries across %u warps\n",
                        total_hist, header->total_warp_slots);
            }
        }
    }

    // Write to file
    FILE* f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "[gddbg] ERROR: cannot open output file: %s\n", output_path);
        free(h_buffer);
        if (h_history) free(h_history);
        return;
    }

    // Write main trace data
    size_t written = fwrite(h_buffer, 1, trace_buffer_size, f);
    size_t total_written = written;

    // Append history section
    if (h_history && written == trace_buffer_size) {
        size_t hist_written = fwrite(h_history, 1, history_buffer_size, f);
        total_written += hist_written;
        if (hist_written != history_buffer_size) {
            fprintf(stderr, "[gddbg] WARNING: incomplete history write (%zu of %zu bytes)\n",
                    hist_written, history_buffer_size);
        }
    }

    fclose(f);

    if (written != trace_buffer_size) {
        fprintf(stderr, "[gddbg] ERROR: incomplete write (%zu of %zu bytes)\n",
                written, trace_buffer_size);
    } else {
        fprintf(stderr, "[gddbg] Trace written to: %s (%zu MB)\n",
                output_path, total_written / (1024*1024));
    }

    free(h_buffer);
    if (h_history) free(h_history);

    // Free device buffers
    cudaFree(d_trace_buffer);
    d_trace_buffer = nullptr;

    if (d_history_buffer) {
        cudaFree(d_history_buffer);
        d_history_buffer = nullptr;
        history_buffer_size = 0;

        // Clear history device globals
        void* null_ptr = nullptr;
        uint32_t zero = 0;
        extern __device__ char* g_gddbg_history_buffer;
        extern __device__ uint32_t g_gddbg_history_depth;
        cudaMemcpyToSymbol(g_gddbg_history_buffer, &null_ptr, sizeof(void*));
        cudaMemcpyToSymbol(g_gddbg_history_depth, &zero, sizeof(uint32_t));
    }

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
    if (d_history_buffer) {
        cudaFree(d_history_buffer);
        d_history_buffer = nullptr;
        history_buffer_size = 0;
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
