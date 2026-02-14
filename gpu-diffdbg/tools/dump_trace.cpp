#include "../lib/common/trace_format.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <trace.prlx>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", argv[1]);
        return 1;
    }

    // Read header
    TraceFileHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Failed to read header\n");
        fclose(f);
        return 1;
    }

    // Validate magic
    if (header.magic != PRLX_MAGIC) {
        fprintf(stderr, "Invalid magic: 0x%016lx (expected 0x%016lx)\n",
                header.magic, PRLX_MAGIC);
        fclose(f);
        return 1;
    }

    // Print header
    printf("=== PRLX Trace File ===\n");
    printf("Version: %u\n", header.version);
    printf("Kernel: %s (hash: 0x%016lx)\n", header.kernel_name, header.kernel_name_hash);
    printf("Grid: (%u, %u, %u)\n", header.grid_dim[0], header.grid_dim[1], header.grid_dim[2]);
    printf("Block: (%u, %u, %u)\n", header.block_dim[0], header.block_dim[1], header.block_dim[2]);
    printf("Warps per block: %u\n", header.num_warps_per_block);
    printf("Total warp slots: %u\n", header.total_warp_slots);
    printf("Events per warp: %u\n", header.events_per_warp);
    printf("Timestamp: %lu\n", header.timestamp);
    printf("CUDA Arch: SM_%u\n", header.cuda_arch);
    printf("\n");

    // Read per-warp data
    size_t warp_buffer_size = sizeof(WarpBufferHeader) + header.events_per_warp * sizeof(TraceEvent);
    char* warp_data = new char[warp_buffer_size];

    int total_events = 0;
    int total_overflow = 0;

    for (uint32_t warp = 0; warp < header.total_warp_slots; warp++) {
        if (fread(warp_data, warp_buffer_size, 1, f) != 1) {
            fprintf(stderr, "Failed to read warp %u data\n", warp);
            break;
        }

        WarpBufferHeader* warp_header = (WarpBufferHeader*)warp_data;
        TraceEvent* events = (TraceEvent*)(warp_data + sizeof(WarpBufferHeader));

        if (warp_header->num_events > 0) {
            printf("=== Warp %u ===\n", warp);
            printf("Events recorded: %u (overflow: %u)\n",
                   warp_header->num_events, warp_header->overflow_count);

            total_events += warp_header->num_events;
            total_overflow += warp_header->overflow_count;

            for (uint32_t i = 0; i < warp_header->num_events; i++) {
                TraceEvent* evt = &events[i];
                printf("  [%u] site=0x%08x type=%u branch=%u active_mask=0x%08x value_a=0x%08x\n",
                       i, evt->site_id, evt->event_type, evt->branch_dir,
                       evt->active_mask, evt->value_a);

                // Decode branch direction
                if (evt->event_type == EVENT_BRANCH) {
                    printf("       -> Branch %s (operand=%u)\n",
                           evt->branch_dir ? "TAKEN" : "NOT-TAKEN", evt->value_a);
                }
            }
            printf("\n");
        }
    }

    printf("=== Summary ===\n");
    printf("Total events: %d\n", total_events);
    printf("Total overflow: %d\n", total_overflow);

    delete[] warp_data;
    fclose(f);
    return 0;
}
