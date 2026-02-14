#ifndef PRLX_NVBIT_SITE_TABLE_H
#define PRLX_NVBIT_SITE_TABLE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace prlx {

// Source location info from NVBit's line info API
struct NvbitSourceLoc {
    std::string filename;
    std::string function;
    uint32_t line;
    uint32_t column;
    uint8_t event_type;
    uint32_t ordinal;   // Per-(function, event_type) ordinal for cross-backend matching
};

// Maps SASS PC offsets to site_ids using FNV-1a hashing.
// The hashing algorithm matches the LLVM pass's SiteTable.cpp for compatibility.
class NvbitSiteTable {
public:
    // Register a site: maps sass_pc to a deterministic site_id
    // With lineinfo: hash "filename:function:line:0:event_type"
    // Without:       hash "function:event_type:ordN"
    uint32_t register_site(uint32_t sass_pc, uint8_t event_type,
                          const std::string& filename,
                          const std::string& function,
                          uint32_t line);

    // Look up site_id by SASS PC (returns 0 if not found)
    uint32_t lookup(uint32_t sass_pc) const;

    // Export to JSON file (same schema as LLVM pass's prlx-sites.json)
    void export_json(const std::string& path) const;

    // Number of registered sites
    size_t size() const { return sites_.size(); }

    // FNV-1a hash (public for testing)
    static uint32_t fnv1a_hash(const std::string& key);

private:
    struct SiteInfo {
        uint32_t site_id;
        uint32_t sass_pc;
        NvbitSourceLoc loc;
    };

    // Maps SASS PC → site_id for fast lookup during event processing
    std::unordered_map<uint32_t, uint32_t> pc_to_site_;

    // All registered sites
    std::vector<SiteInfo> sites_;

    // Per-(function, event_type) ordinal counter
    std::unordered_map<std::string, uint32_t> ordinal_counters_;
};

} // namespace prlx

#endif // PRLX_NVBIT_SITE_TABLE_H
