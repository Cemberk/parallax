// NVBit binary instrumentation tool for prlx.
//
// This is the main NVBit tool. It registers callbacks for CUDA events,
// instruments SASS instructions, and produces .prlx trace files compatible
// with the existing Rust differ.
//
// Usage:
//   LD_PRELOAD=./build/lib/nvbit_tool/libprlx_nvbit.so \
//     PRLX_TRACE=trace.prlx ./my_cuda_app

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"

#include "common.h"
#include "site_table.h"
#include "trace_writer.h"

// Device-side globals (must match inject_funcs.cu declarations)
__managed__ ChannelDev* prlx_channel_dev_ptr = nullptr;
extern "C" __device__ __noinline__ ChannelDev* prlx_channel_dev;
extern "C" __device__ __noinline__ uint64_t prlx_grid_launch_id;
extern "C" __device__ __noinline__ int prlx_enabled;

// ---- Configuration (from environment variables) ----
static struct {
    std::string trace_path = "trace.prlx";
    std::string session_dir;         // Empty = single file mode
    std::string sites_path = "prlx-sites.json";
    std::string filter_pattern;      // Empty = instrument all
    uint32_t buffer_size = PRLX_NVBIT_DEFAULT_BUFFER_SIZE;
    uint32_t sample_rate = 1;
    bool enabled = true;
    bool compress = false;
} g_config;

// ---- Global state ----
static prlx::NvbitSiteTable g_site_table;
static prlx::TraceWriter* g_writer = nullptr;
static std::mutex g_writer_mutex;
static uint64_t g_grid_launch_id = 0;
static uint32_t g_launch_count = 0;
static std::unordered_set<CUfunction> g_instrumented;
static std::mutex g_instrument_mutex;

// Channel for device→host event transfer
static ChannelHost g_channel_host;
static ChannelDev* g_channel_dev = nullptr;
static volatile bool g_receiver_running = false;
static std::thread* g_receiver_thread = nullptr;

// ---- SASS opcode classification ----

static bool is_branch_opcode(const char* opcode) {
    // SASS branch/jump opcodes
    return (strcmp(opcode, "BRA") == 0 ||
            strcmp(opcode, "BRX") == 0 ||
            strcmp(opcode, "JMP") == 0 ||
            strcmp(opcode, "JMX") == 0 ||
            strcmp(opcode, "BREAK") == 0 ||
            strcmp(opcode, "CONT") == 0 ||
            strcmp(opcode, "BSYNC") == 0);
}

static bool is_shmem_store_opcode(const char* opcode) {
    return (strcmp(opcode, "STS") == 0 ||
            strncmp(opcode, "STS.", 4) == 0);
}

static bool is_atomic_opcode(const char* opcode) {
    return (strcmp(opcode, "ATOM") == 0 ||
            strncmp(opcode, "ATOM.", 5) == 0 ||
            strcmp(opcode, "ATOMS") == 0 ||
            strncmp(opcode, "ATOMS.", 6) == 0 ||
            strcmp(opcode, "RED") == 0 ||
            strncmp(opcode, "RED.", 4) == 0);
}

static bool is_func_exit_opcode(const char* opcode) {
    return (strcmp(opcode, "RET") == 0 ||
            strcmp(opcode, "EXIT") == 0);
}

// ---- Receiver thread ----
// Consumes channel events from device and feeds them to the trace writer.

static void receiver_thread_func() {
    char* recv_buffer = nullptr;
    size_t recv_buffer_size = 0;

    while (g_receiver_running) {
        uint32_t num_recv_bytes = g_channel_host.recv(recv_buffer, recv_buffer_size);

        if (num_recv_bytes > 0) {
            uint32_t num_events = num_recv_bytes / sizeof(prlx_channel_event_t);

            std::lock_guard<std::mutex> lock(g_writer_mutex);
            if (g_writer) {
                const prlx_channel_event_t* events =
                    reinterpret_cast<const prlx_channel_event_t*>(recv_buffer);
                for (uint32_t i = 0; i < num_events; i++) {
                    // Translate SASS PC to site_id using site table
                    prlx_channel_event_t translated = events[i];
                    translated.site_id = g_site_table.lookup(events[i].site_id);
                    g_writer->add_event(translated);
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    free(recv_buffer);
}

// ---- Kernel name filter ----
static bool matches_filter(const std::string& kernel_name) {
    if (g_config.filter_pattern.empty()) return true;

    // Substring match — glob patterns not supported yet
    return kernel_name.find(g_config.filter_pattern) != std::string::npos;
}

// ---- Instrumentation ----

static void instrument_function(CUcontext ctx, CUfunction func) {
    std::lock_guard<std::mutex> lock(g_instrument_mutex);

    if (g_instrumented.count(func)) return;
    g_instrumented.insert(func);

    const char* func_name = nvbit_get_func_name(ctx, func, true);
    if (!matches_filter(func_name)) return;

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);
    if (instrs.empty()) return;

    bool first_instr = true;

    for (auto* instr : instrs) {
        const char* opcode = instr->getOpcode();
        uint32_t sass_pc = instr->getOffset();

        const char* filename = "";
        uint32_t line = 0;
        // NVBit provides line info via the instruction's debug info
        // (available when the binary was compiled with -lineinfo or -g)

        uint8_t event_type = 255; // sentinel

        if (first_instr) {
            event_type = PRLX_EVENT_FUNC_ENTRY;
            uint32_t site_id = g_site_table.register_site(
                sass_pc, event_type, filename, func_name, line);

            nvbit_insert_call(instr, "prlx_instr_func_entry", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, sass_pc);
            first_instr = false;
        }

        if (is_branch_opcode(opcode)) {
            event_type = PRLX_EVENT_BRANCH;
            g_site_table.register_site(sass_pc, event_type, filename, func_name, line);

            nvbit_insert_call(instr, "prlx_instr_branch", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_pred_val(instr);      // branch_taken
            nvbit_add_call_arg_const_val32(instr, sass_pc);
            nvbit_add_call_arg_const_val64(instr, 0); // grid_id placeholder
        }
        else if (is_shmem_store_opcode(opcode)) {
            event_type = PRLX_EVENT_SHMEM_STORE;
            g_site_table.register_site(sass_pc, event_type, filename, func_name, line);

            nvbit_insert_call(instr, "prlx_instr_shmem_store", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, sass_pc);
            nvbit_add_call_arg_mref_addr64(instr, 0); // address
            nvbit_add_call_arg_const_val32(instr, 0);  // value: not extractable at SASS level
        }
        else if (is_atomic_opcode(opcode)) {
            event_type = PRLX_EVENT_ATOMIC;
            g_site_table.register_site(sass_pc, event_type, filename, func_name, line);

            nvbit_insert_call(instr, "prlx_instr_atomic", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, sass_pc);
            nvbit_add_call_arg_mref_addr64(instr, 0); // address
            nvbit_add_call_arg_const_val32(instr, 0);  // operand: not extractable at SASS level
        }
        else if (is_func_exit_opcode(opcode)) {
            event_type = PRLX_EVENT_FUNC_EXIT;
            g_site_table.register_site(sass_pc, event_type, filename, func_name, line);

            nvbit_insert_call(instr, "prlx_instr_func_exit", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, sass_pc);
        }
    }
}

// ---- NVBit callbacks ----

void nvbit_tool_init() {
    if (const char* v = getenv("PRLX_ENABLED")) {
        g_config.enabled = (atoi(v) != 0);
    }
    if (!g_config.enabled) {
        fprintf(stderr, "[prlx-nvbit] Disabled via PRLX_ENABLED=0\n");
        return;
    }

    if (const char* v = getenv("PRLX_TRACE")) {
        g_config.trace_path = v;
    }
    if (const char* v = getenv("PRLX_SESSION")) {
        g_config.session_dir = v;
    }
    if (const char* v = getenv("PRLX_SITES")) {
        g_config.sites_path = v;
    }
    if (const char* v = getenv("PRLX_FILTER")) {
        g_config.filter_pattern = v;
    }
    if (const char* v = getenv("PRLX_BUFFER_SIZE")) {
        g_config.buffer_size = atoi(v);
    }
    if (const char* v = getenv("PRLX_SAMPLE_RATE")) {
        g_config.sample_rate = atoi(v);
    }
    if (const char* v = getenv("PRLX_COMPRESS")) {
        g_config.compress = (atoi(v) != 0);
    }

    fprintf(stderr, "[prlx-nvbit] Initialized: trace=%s, buffer=%u, filter='%s'\n",
            g_config.trace_path.c_str(),
            g_config.buffer_size,
            g_config.filter_pattern.c_str());

    g_writer = new prlx::TraceWriter(g_config.buffer_size);
    g_channel_host.init(0, nullptr, sizeof(prlx_channel_event_t), 1 << 20);

    g_receiver_running = true;
    g_receiver_thread = new std::thread(receiver_thread_func);

    // Set device-side channel pointer
    // (Will be set per-context in nvbit_at_cuda_event)
}

void nvbit_tool_exit() {
    if (!g_config.enabled) return;

    g_receiver_running = false;
    if (g_receiver_thread) {
        g_receiver_thread->join();
        delete g_receiver_thread;
        g_receiver_thread = nullptr;
    }

    // Flush any remaining events
    g_channel_host.stop();

    if (g_writer && g_writer->total_events() > 0) {
        std::lock_guard<std::mutex> lock(g_writer_mutex);
        g_writer->write(g_config.trace_path, g_config.compress);
    }

    if (g_site_table.size() > 0) {
        g_site_table.export_json(g_config.sites_path);
    }

    delete g_writer;
    g_writer = nullptr;

    fprintf(stderr, "[prlx-nvbit] Shutdown: %u launches instrumented, %zu sites\n",
            g_launch_count, g_site_table.size());
}

void nvbit_at_cuda_event(
    CUcontext ctx,
    int is_exit,
    nvbit_api_cuda_t cbid,
    const char* name,
    void* params,
    CUresult* pStatus
) {
    if (!g_config.enabled) return;

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;

        if (!is_exit) {
            // Pre-launch: instrument the function and set up tracing
            instrument_function(ctx, p->f);

            const char* kernel_name = nvbit_get_func_name(ctx, p->f, true);

            int dev = 0;
            CUdevice cu_dev;
            cuCtxGetDevice(&cu_dev);
            int major = 0, minor = 0;
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev);
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev);
            uint32_t arch = major * 10 + minor;

            {
                std::lock_guard<std::mutex> lock(g_writer_mutex);
                if (g_writer) {
                    g_writer->set_kernel_info(
                        kernel_name,
                        p->gridDimX, p->gridDimY, p->gridDimZ,
                        p->blockDimX, p->blockDimY, p->blockDimZ,
                        arch);
                }
            }

            g_grid_launch_id++;

            int enabled_val = 1;
            cudaMemcpyToSymbol(prlx_enabled, &enabled_val, sizeof(int));
            cudaMemcpyToSymbol(prlx_grid_launch_id, &g_grid_launch_id, sizeof(uint64_t));

            g_launch_count++;
        } else {
            // Post-launch: drain the channel
            g_channel_host.stop();

            // If in session mode, write per-kernel file and clear
            if (!g_config.session_dir.empty()) {
                std::lock_guard<std::mutex> lock(g_writer_mutex);
                if (g_writer && g_writer->total_events() > 0) {
                    std::string kernel_file = g_config.session_dir + "/launch_" +
                        std::to_string(g_launch_count) + ".prlx";
                    g_writer->write(kernel_file, g_config.compress);
                    g_writer->clear();
                }
            }
        }
    }
}
