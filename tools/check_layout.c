#include "../lib/common/trace_format.h"
#include <stdio.h>
#include <stddef.h>

int main() {
    printf("=== Struct Sizes and Alignment ===\n\n");

    printf("TraceEvent:\n");
    printf("  sizeof: %zu bytes\n", sizeof(TraceEvent));
    printf("  alignof: %zu bytes\n", _Alignof(TraceEvent));
    printf("  Offsets:\n");
    printf("    site_id:     %zu\n", offsetof(TraceEvent, site_id));
    printf("    event_type:  %zu\n", offsetof(TraceEvent, event_type));
    printf("    branch_dir:  %zu\n", offsetof(TraceEvent, branch_dir));
    printf("    _reserved:   %zu\n", offsetof(TraceEvent, _reserved));
    printf("    active_mask: %zu\n", offsetof(TraceEvent, active_mask));
    printf("    value_a:     %zu\n", offsetof(TraceEvent, value_a));
    printf("\n");

    printf("WarpBufferHeader:\n");
    printf("  sizeof: %zu bytes\n", sizeof(WarpBufferHeader));
    printf("  alignof: %zu bytes\n", _Alignof(WarpBufferHeader));
    printf("  Offsets:\n");
    printf("    write_idx:       %zu\n", offsetof(WarpBufferHeader, write_idx));
    printf("    overflow_count:  %zu\n", offsetof(WarpBufferHeader, overflow_count));
    printf("    num_events:      %zu\n", offsetof(WarpBufferHeader, num_events));
    printf("    _reserved:       %zu\n", offsetof(WarpBufferHeader, _reserved));
    printf("\n");

    printf("TraceFileHeader:\n");
    printf("  sizeof: %zu bytes\n", sizeof(TraceFileHeader));
    printf("  alignof: %zu bytes\n", _Alignof(TraceFileHeader));
    printf("  Offsets:\n");
    printf("    magic:               %zu\n", offsetof(TraceFileHeader, magic));
    printf("    version:             %zu\n", offsetof(TraceFileHeader, version));
    printf("    flags:               %zu\n", offsetof(TraceFileHeader, flags));
    printf("    kernel_name_hash:    %zu\n", offsetof(TraceFileHeader, kernel_name_hash));
    printf("    kernel_name:         %zu\n", offsetof(TraceFileHeader, kernel_name));
    printf("    grid_dim:            %zu\n", offsetof(TraceFileHeader, grid_dim));
    printf("    block_dim:           %zu\n", offsetof(TraceFileHeader, block_dim));
    printf("    num_warps_per_block: %zu\n", offsetof(TraceFileHeader, num_warps_per_block));
    printf("    total_warp_slots:    %zu\n", offsetof(TraceFileHeader, total_warp_slots));
    printf("    events_per_warp:     %zu\n", offsetof(TraceFileHeader, events_per_warp));
    printf("    timestamp:           %zu\n", offsetof(TraceFileHeader, timestamp));
    printf("    cuda_arch:           %zu\n", offsetof(TraceFileHeader, cuda_arch));
    printf("    _reserved:           %zu\n", offsetof(TraceFileHeader, _reserved));
    printf("\n");

    printf("SiteTableEntry:\n");
    printf("  sizeof: %zu bytes\n", sizeof(SiteTableEntry));
    printf("  alignof: %zu bytes\n", _Alignof(SiteTableEntry));
    printf("\n");

    printf("=== Packing Strategy ===\n");
    printf("Natural alignment: YES\n");
    printf("Explicit alignment (TraceEvent): __attribute__((aligned(16)))\n");
    printf("Manual padding: YES (_reserved fields)\n");
    printf("#pragma pack(1): NO\n");

    return 0;
}
