// Host-only tests for TraceWriter (no GPU required).
// Compile: g++ -std=c++17 -I.. -I../../common test_trace_writer.cpp ../trace_writer.cu ../site_table.cu -o test_trace_writer

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "trace_writer.h"
#include "trace_format.h"

// ---- add_event accumulation ----

static prlx_channel_event_t make_event(uint32_t warp_id, uint8_t event_type, uint32_t site_id) {
    prlx_channel_event_t evt;
    memset(&evt, 0, sizeof(evt));
    evt.grid_launch_id = 1;
    evt.warp_id = warp_id;
    evt.site_id = site_id;
    evt.event_type = event_type;
    evt.branch_dir = 0;
    evt.active_mask = 0xFFFFFFFF;
    evt.value_a = 0;
    return evt;
}

static void test_add_event_basic() {
    prlx::TraceWriter writer(4096);
    auto evt = make_event(0, PRLX_EVENT_BRANCH, 0x100);
    writer.add_event(evt);

    assert(writer.total_events() == 1);
    assert(writer.num_warps_seen() == 1);
    printf("  PASS: add_event_basic\n");
}

static void test_add_event_multiple_warps() {
    prlx::TraceWriter writer(4096);
    writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100));
    writer.add_event(make_event(1, PRLX_EVENT_BRANCH, 0x100));
    writer.add_event(make_event(2, PRLX_EVENT_BRANCH, 0x100));

    assert(writer.total_events() == 3);
    assert(writer.num_warps_seen() == 3);
    printf("  PASS: add_event_multiple_warps\n");
}

static void test_add_event_same_warp() {
    prlx::TraceWriter writer(4096);
    for (int i = 0; i < 10; i++) {
        writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100 + i));
    }

    assert(writer.total_events() == 10);
    assert(writer.num_warps_seen() == 1);
    printf("  PASS: add_event_same_warp\n");
}

// ---- overflow counting ----

static void test_overflow_counting() {
    prlx::TraceWriter writer(4);  // Very small buffer
    for (int i = 0; i < 10; i++) {
        writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100));
    }

    assert(writer.total_events() == 10);
    assert(writer.total_overflow() == 6);  // 10 - 4 = 6 overflows
    printf("  PASS: overflow_counting\n");
}

static void test_overflow_per_warp() {
    prlx::TraceWriter writer(2);
    // Warp 0: 5 events → 3 overflows
    for (int i = 0; i < 5; i++) {
        writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100));
    }
    // Warp 1: 3 events → 1 overflow
    for (int i = 0; i < 3; i++) {
        writer.add_event(make_event(1, PRLX_EVENT_BRANCH, 0x200));
    }

    assert(writer.total_events() == 8);
    assert(writer.total_overflow() == 4);  // 3 + 1
    printf("  PASS: overflow_per_warp\n");
}

// ---- clear ----

static void test_clear() {
    prlx::TraceWriter writer(4096);
    writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100));
    writer.add_event(make_event(1, PRLX_EVENT_BRANCH, 0x200));

    writer.clear();
    assert(writer.total_events() == 0);
    assert(writer.num_warps_seen() == 0);
    printf("  PASS: clear\n");
}

// ---- sample_rate ----

static void test_sample_rate_default() {
    prlx::TraceWriter writer(4096);
    assert(writer.sample_rate() == 1);
    printf("  PASS: sample_rate_default\n");
}

static void test_sample_rate_set() {
    prlx::TraceWriter writer(4096);
    writer.set_sample_rate(100);
    assert(writer.sample_rate() == 100);
    printf("  PASS: sample_rate_set\n");
}

// ---- write() binary format validation ----

static void test_write_binary_format() {
    prlx::TraceWriter writer(4096);
    writer.set_kernel_info("test_kernel", 1, 1, 1, 32, 1, 1, 80);

    // Add events to two warps
    writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100));
    writer.add_event(make_event(0, PRLX_EVENT_SHMEM_STORE, 0x200));
    writer.add_event(make_event(1, PRLX_EVENT_ATOMIC, 0x300));

    std::string path = "/tmp/prlx_test_trace.prlx";
    bool ok = writer.write(path, false);
    assert(ok);

    // Read back the binary file
    std::ifstream f(path, std::ios::binary);
    assert(f.good());
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(file_size);
    f.read(reinterpret_cast<char*>(data.data()), file_size);

    // Verify header
    assert(file_size >= sizeof(TraceFileHeader));
    const TraceFileHeader* hdr = reinterpret_cast<const TraceFileHeader*>(data.data());

    assert(hdr->magic == PRLX_MAGIC);
    assert(hdr->version == PRLX_VERSION);
    assert(hdr->flags == 0);  // No compression, no sampling
    assert(hdr->total_warp_slots == 2);  // warps 0 and 1
    assert(hdr->events_per_warp == 4096);
    assert(hdr->grid_dim[0] == 1);
    assert(hdr->block_dim[0] == 32);
    assert(hdr->cuda_arch == 80);
    assert(strncmp(hdr->kernel_name, "test_kernel", 11) == 0);

    // Verify warp buffer layout
    size_t warp_buf_size = sizeof(WarpBufferHeader) + 4096 * sizeof(TraceEvent);
    size_t expected_size = sizeof(TraceFileHeader) + 2 * warp_buf_size;
    assert(file_size == expected_size);

    // Verify warp 0 header
    const WarpBufferHeader* w0 = reinterpret_cast<const WarpBufferHeader*>(
        data.data() + sizeof(TraceFileHeader));
    assert(w0->num_events == 2);
    assert(w0->overflow_count == 0);

    // Verify warp 0 events
    const TraceEvent* w0_events = reinterpret_cast<const TraceEvent*>(
        data.data() + sizeof(TraceFileHeader) + sizeof(WarpBufferHeader));
    assert(w0_events[0].site_id == 0x100);
    assert(w0_events[0].event_type == PRLX_EVENT_BRANCH);
    assert(w0_events[1].site_id == 0x200);
    assert(w0_events[1].event_type == PRLX_EVENT_SHMEM_STORE);

    // Verify warp 1 header
    const WarpBufferHeader* w1 = reinterpret_cast<const WarpBufferHeader*>(
        data.data() + sizeof(TraceFileHeader) + warp_buf_size);
    assert(w1->num_events == 1);

    // Verify warp 1 events
    const TraceEvent* w1_events = reinterpret_cast<const TraceEvent*>(
        data.data() + sizeof(TraceFileHeader) + warp_buf_size + sizeof(WarpBufferHeader));
    assert(w1_events[0].site_id == 0x300);
    assert(w1_events[0].event_type == PRLX_EVENT_ATOMIC);

    // Cleanup
    remove(path.c_str());

    printf("  PASS: write_binary_format\n");
}

static void test_write_sampled_flag() {
    prlx::TraceWriter writer(4096);
    writer.set_kernel_info("test_kernel", 1, 1, 1, 32, 1, 1, 80);
    writer.set_sample_rate(10);
    writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100));

    std::string path = "/tmp/prlx_test_sampled.prlx";
    bool ok = writer.write(path, false);
    assert(ok);

    std::ifstream f(path, std::ios::binary);
    TraceFileHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    assert(hdr.flags & PRLX_FLAG_SAMPLED);
    assert(hdr.sample_rate == 10);

    remove(path.c_str());
    printf("  PASS: write_sampled_flag\n");
}

static void test_write_empty_trace() {
    prlx::TraceWriter writer(4096);
    writer.set_kernel_info("empty_kernel", 1, 1, 1, 32, 1, 1, 80);

    std::string path = "/tmp/prlx_test_empty.prlx";
    bool ok = writer.write(path, false);
    assert(ok);

    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();

    // Empty trace = just the header
    assert(file_size == sizeof(TraceFileHeader));

    remove(path.c_str());
    printf("  PASS: write_empty_trace\n");
}

static void test_write_with_overflow() {
    prlx::TraceWriter writer(2);  // Tiny buffer
    writer.set_kernel_info("overflow_kernel", 1, 1, 1, 32, 1, 1, 80);

    // Add 5 events to warp 0 (buffer only holds 2)
    for (int i = 0; i < 5; i++) {
        writer.add_event(make_event(0, PRLX_EVENT_BRANCH, 0x100 + i));
    }

    std::string path = "/tmp/prlx_test_overflow.prlx";
    bool ok = writer.write(path, false);
    assert(ok);

    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> data(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());

    const WarpBufferHeader* w0 = reinterpret_cast<const WarpBufferHeader*>(
        data.data() + sizeof(TraceFileHeader));
    assert(w0->num_events == 2);
    assert(w0->overflow_count == 3);
    assert(w0->total_event_count == 5);

    remove(path.c_str());
    printf("  PASS: write_with_overflow\n");
}

int main() {
    printf("=== TraceWriter Tests ===\n");

    test_add_event_basic();
    test_add_event_multiple_warps();
    test_add_event_same_warp();
    test_overflow_counting();
    test_overflow_per_warp();
    test_clear();
    test_sample_rate_default();
    test_sample_rate_set();
    test_write_binary_format();
    test_write_sampled_flag();
    test_write_empty_trace();
    test_write_with_overflow();

    printf("\nAll trace writer tests passed!\n");
    return 0;
}
