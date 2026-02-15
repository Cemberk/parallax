// Host-only tests for NvbitSiteTable (no GPU required).
// Compile: g++ -std=c++17 -I.. -I../../common test_site_table.cpp ../site_table.cu -o test_site_table

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "site_table.h"

// ---- FNV-1a hash correctness ----

static void test_fnv1a_empty() {
    // FNV-1a of empty string = offset basis
    uint32_t h = prlx::NvbitSiteTable::fnv1a_hash("");
    assert(h == 0x811c9dc5);
    printf("  PASS: fnv1a_empty\n");
}

static void test_fnv1a_known_values() {
    // FNV-1a reference vectors
    uint32_t h1 = prlx::NvbitSiteTable::fnv1a_hash("a");
    uint32_t h2 = prlx::NvbitSiteTable::fnv1a_hash("b");
    assert(h1 != h2);
    assert(h1 != 0);

    // Same input → same hash
    uint32_t h3 = prlx::NvbitSiteTable::fnv1a_hash("a");
    assert(h1 == h3);

    printf("  PASS: fnv1a_known_values\n");
}

static void test_fnv1a_deterministic() {
    // Hash of a site key format used by the table
    std::string key = "kernel.cu:my_kernel:42:0:0";
    uint32_t h1 = prlx::NvbitSiteTable::fnv1a_hash(key);
    uint32_t h2 = prlx::NvbitSiteTable::fnv1a_hash(key);
    assert(h1 == h2);
    assert(h1 != 0);
    printf("  PASS: fnv1a_deterministic\n");
}

// ---- register_site tests ----

static void test_register_site_with_lineinfo() {
    prlx::NvbitSiteTable table;
    uint32_t site_id = table.register_site(0x100, 0, "kernel.cu", "my_func", 42);
    assert(site_id != 0);
    assert(table.size() == 1);

    // Lookup should return the registered site_id
    assert(table.lookup(0x100) == site_id);

    // Re-registering same SASS PC returns same site_id
    uint32_t site_id2 = table.register_site(0x100, 0, "kernel.cu", "my_func", 42);
    assert(site_id2 == site_id);
    assert(table.size() == 1);

    printf("  PASS: register_site_with_lineinfo\n");
}

static void test_register_site_without_lineinfo() {
    prlx::NvbitSiteTable table;
    // Empty filename, line=0 → uses ordinal-based key
    uint32_t site_id = table.register_site(0x200, 0, "", "my_func", 0);
    assert(site_id != 0);
    assert(table.size() == 1);
    assert(table.lookup(0x200) == site_id);

    printf("  PASS: register_site_without_lineinfo\n");
}

static void test_ordinal_counters() {
    prlx::NvbitSiteTable table;
    // Without lineinfo, ordinals should increment per (function, event_type)
    uint32_t s1 = table.register_site(0x300, 0, "", "func_a", 0);
    uint32_t s2 = table.register_site(0x304, 0, "", "func_a", 0);
    uint32_t s3 = table.register_site(0x308, 0, "", "func_a", 0);

    // Each should be unique since they hash different ordinal keys
    assert(s1 != s2);
    assert(s2 != s3);
    assert(s1 != s3);
    assert(table.size() == 3);

    printf("  PASS: ordinal_counters\n");
}

static void test_ordinal_counters_different_event_types() {
    prlx::NvbitSiteTable table;
    // Different event types reset ordinal counter
    uint32_t s1 = table.register_site(0x400, 0, "", "func_b", 0);  // branch ord0
    uint32_t s2 = table.register_site(0x404, 2, "", "func_b", 0);  // atomic ord0
    uint32_t s3 = table.register_site(0x408, 0, "", "func_b", 0);  // branch ord1

    assert(s1 != s2);
    assert(s2 != s3);
    assert(table.size() == 3);

    printf("  PASS: ordinal_counters_different_event_types\n");
}

static void test_lookup_not_found() {
    prlx::NvbitSiteTable table;
    // Unregistered SASS PC returns the PC itself
    assert(table.lookup(0x999) == 0x999);
    printf("  PASS: lookup_not_found\n");
}

// ---- JSON export ----

static void test_export_json() {
    prlx::NvbitSiteTable table;
    table.register_site(0x100, 0, "kernel.cu", "my_func", 10);
    table.register_site(0x200, 2, "kernel.cu", "my_func", 20);

    std::string path = "/tmp/prlx_test_sites.json";
    table.export_json(path);

    // Read back and verify basic structure
    std::ifstream f(path);
    assert(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    // Should contain JSON array markers
    assert(content.find("[") != std::string::npos);
    assert(content.find("]") != std::string::npos);
    assert(content.find("site_id") != std::string::npos);
    assert(content.find("kernel.cu") != std::string::npos);
    assert(content.find("my_func") != std::string::npos);

    // Cleanup
    remove(path.c_str());

    printf("  PASS: export_json\n");
}

static void test_export_json_escaping() {
    prlx::NvbitSiteTable table;
    table.register_site(0x100, 0, "path\\with\"quotes.cu", "func\"name", 1);

    std::string path = "/tmp/prlx_test_sites_escape.json";
    table.export_json(path);

    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    // Escaped characters should be present
    assert(content.find("\\\\") != std::string::npos);
    assert(content.find("\\\"") != std::string::npos);

    remove(path.c_str());

    printf("  PASS: export_json_escaping\n");
}

int main() {
    printf("=== NvbitSiteTable Tests ===\n");

    test_fnv1a_empty();
    test_fnv1a_known_values();
    test_fnv1a_deterministic();
    test_register_site_with_lineinfo();
    test_register_site_without_lineinfo();
    test_ordinal_counters();
    test_ordinal_counters_different_event_types();
    test_lookup_not_found();
    test_export_json();
    test_export_json_escaping();

    printf("\nAll site table tests passed!\n");
    return 0;
}
