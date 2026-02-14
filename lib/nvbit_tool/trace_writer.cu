// Trace writer: assembles channel events into the binary .prlx format
// compatible with the existing Rust parser (parser.rs).

#include "trace_writer.h"
#include "trace_format.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace prlx {

void TraceWriter::add_event(const prlx_channel_event_t& evt) {
    auto& buf = warp_events_[evt.warp_id];

    if (buf.events.size() < events_per_warp_) {
        buf.events.push_back(evt);
    } else {
        buf.overflow_count++;
    }
    total_events_++;
}

void TraceWriter::set_kernel_info(
    const std::string& kernel_name,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t cuda_arch
) {
    kernel_name_ = kernel_name;
    grid_dim_[0] = grid_x;
    grid_dim_[1] = grid_y;
    grid_dim_[2] = grid_z;
    block_dim_[0] = block_x;
    block_dim_[1] = block_y;
    block_dim_[2] = block_z;
    cuda_arch_ = cuda_arch;
}

bool TraceWriter::write(const std::string& path, bool compress) {
    uint32_t max_warp_id = 0;
    for (auto& [warp_id, _] : warp_events_) {
        if (warp_id > max_warp_id) max_warp_id = warp_id;
    }
    uint32_t total_warp_slots = warp_events_.empty() ? 0 : max_warp_id + 1;

    uint32_t threads_per_block = block_dim_[0] * block_dim_[1] * block_dim_[2];
    uint32_t warps_per_block = (threads_per_block + 31) / 32;

    TraceFileHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = PRLX_MAGIC;
    header.version = PRLX_VERSION;
    header.flags = 0;

    // Kernel name hash (FNV-1a, same as LLVM pass)
    uint32_t name_hash = 0x811c9dc5;
    for (char c : kernel_name_) {
        name_hash ^= (uint8_t)c;
        name_hash *= 0x01000193;
    }
    header.kernel_name_hash = name_hash;

    // Copy kernel name (truncate to 63 chars + null)
    size_t copy_len = std::min(kernel_name_.size(), (size_t)63);
    memcpy(header.kernel_name, kernel_name_.c_str(), copy_len);

    header.grid_dim[0] = grid_dim_[0];
    header.grid_dim[1] = grid_dim_[1];
    header.grid_dim[2] = grid_dim_[2];
    header.block_dim[0] = block_dim_[0];
    header.block_dim[1] = block_dim_[1];
    header.block_dim[2] = block_dim_[2];
    header.num_warps_per_block = warps_per_block;
    header.total_warp_slots = total_warp_slots;
    header.events_per_warp = events_per_warp_;

    auto now = std::chrono::system_clock::now();
    header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()
    ).count();

    header.cuda_arch = cuda_arch_;

    size_t warp_buffer_size = sizeof(WarpBufferHeader) + events_per_warp_ * sizeof(TraceEvent);
    size_t total_size = sizeof(TraceFileHeader) + total_warp_slots * warp_buffer_size;

    std::vector<uint8_t> buffer(total_size, 0);
    memcpy(buffer.data(), &header, sizeof(header));

    for (uint32_t w = 0; w < total_warp_slots; w++) {
        size_t warp_offset = sizeof(TraceFileHeader) + w * warp_buffer_size;

        WarpBufferHeader warp_hdr;
        memset(&warp_hdr, 0, sizeof(warp_hdr));

        auto it = warp_events_.find(w);
        if (it != warp_events_.end()) {
            auto& wb = it->second;
            uint32_t num_events = std::min((uint32_t)wb.events.size(), events_per_warp_);
            warp_hdr.write_idx = num_events;
            warp_hdr.overflow_count = wb.overflow_count;
            warp_hdr.num_events = num_events;
            warp_hdr.total_event_count = num_events + wb.overflow_count;

            memcpy(buffer.data() + warp_offset, &warp_hdr, sizeof(warp_hdr));

            for (uint32_t e = 0; e < num_events; e++) {
                const auto& ch_evt = wb.events[e];
                TraceEvent trace_evt;
                trace_evt.site_id = ch_evt.site_id;
                trace_evt.event_type = ch_evt.event_type;
                trace_evt.branch_dir = ch_evt.branch_dir;
                trace_evt._reserved = 0;
                trace_evt.active_mask = ch_evt.active_mask;
                trace_evt.value_a = ch_evt.value_a;

                size_t evt_offset = warp_offset + sizeof(WarpBufferHeader) + e * sizeof(TraceEvent);
                memcpy(buffer.data() + evt_offset, &trace_evt, sizeof(trace_evt));
            }
        } else {
            // Empty warp
            memcpy(buffer.data() + warp_offset, &warp_hdr, sizeof(warp_hdr));
        }
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[prlx-nvbit] Failed to open %s for writing\n", path.c_str());
        return false;
    }

    // NVBit trace writer does not support compression yet.
    // The LLVM pass runtime (prlx_host.cu) handles zstd compression.
    (void)compress;
    fwrite(buffer.data(), 1, buffer.size(), f);

    fclose(f);

    fprintf(stderr, "[prlx-nvbit] Wrote trace: %s (%zu bytes, %u warps, %zu events)\n",
            path.c_str(), buffer.size(), total_warp_slots, total_events_);

    return true;
}

void TraceWriter::clear() {
    warp_events_.clear();
    total_events_ = 0;
}

} // namespace prlx
