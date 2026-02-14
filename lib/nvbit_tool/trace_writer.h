#ifndef PRLX_NVBIT_TRACE_WRITER_H
#define PRLX_NVBIT_TRACE_WRITER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "site_table.h"

namespace prlx {

// Accumulated events for a single kernel launch, organized per-warp.
// The trace writer assembles these into the exact binary format that parser.rs expects.
class TraceWriter {
public:
    TraceWriter(uint32_t events_per_warp = PRLX_NVBIT_DEFAULT_BUFFER_SIZE)
        : events_per_warp_(events_per_warp) {}

    // Add a channel event (called from receiver thread)
    void add_event(const prlx_channel_event_t& evt);

    // Set kernel metadata before writing
    void set_kernel_info(const std::string& kernel_name,
                         uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                         uint32_t block_x, uint32_t block_y, uint32_t block_z,
                         uint32_t cuda_arch);

    // Write the trace file in .prlx binary format
    // Returns true on success
    bool write(const std::string& path, bool compress = false);

    // Clear accumulated events (reset for next kernel launch)
    void clear();

    // Statistics
    size_t total_events() const { return total_events_; }
    size_t num_warps_seen() const { return warp_events_.size(); }

private:
    // Per-warp event accumulator
    struct WarpBuffer {
        std::vector<prlx_channel_event_t> events;
        uint32_t overflow_count = 0;
    };

    // Kernel metadata
    std::string kernel_name_;
    uint32_t grid_dim_[3] = {0, 0, 0};
    uint32_t block_dim_[3] = {0, 0, 0};
    uint32_t cuda_arch_ = 0;

    uint32_t events_per_warp_;
    size_t total_events_ = 0;

    // warp_id → accumulated events
    std::unordered_map<uint32_t, WarpBuffer> warp_events_;
};

} // namespace prlx

#endif // PRLX_NVBIT_TRACE_WRITER_H
