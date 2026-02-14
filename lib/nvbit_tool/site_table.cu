// Site table implementation: maps SASS PC offsets to deterministic site_ids
// using FNV-1a hashing that matches the LLVM pass's SiteTable.cpp.

#include "site_table.h"
#include <cstdio>
#include <sstream>

namespace prlx {

// FNV-1a hash constants (same as LLVM pass SiteTable.cpp)
static constexpr uint32_t FNV_OFFSET = 0x811c9dc5;
static constexpr uint32_t FNV_PRIME  = 0x01000193;

uint32_t NvbitSiteTable::fnv1a_hash(const std::string& key) {
    uint32_t hash = FNV_OFFSET;
    for (char c : key) {
        hash ^= static_cast<uint8_t>(c);
        hash *= FNV_PRIME;
    }
    return hash;
}

uint32_t NvbitSiteTable::register_site(
    uint32_t sass_pc, uint8_t event_type,
    const std::string& filename, const std::string& function,
    uint32_t line
) {
    auto it = pc_to_site_.find(sass_pc);
    if (it != pc_to_site_.end()) {
        return it->second;
    }

    // Build hash key (same format as LLVM pass)
    std::string hash_key;
    uint32_t ordinal = 0;

    if (!filename.empty() && line > 0) {
        // With lineinfo: "filename:function:line:0:event_type"
        // Column is always 0 for NVBit (SASS doesn't have column info)
        std::ostringstream oss;
        oss << filename << ":" << function << ":" << line << ":0:" << (int)event_type;
        hash_key = oss.str();
    } else {
        // Without lineinfo: "function:event_type:ordN"
        std::string ord_key = function + ":" + std::to_string(event_type);
        ordinal = ordinal_counters_[ord_key]++;
        std::ostringstream oss;
        oss << function << ":" << (int)event_type << ":ord" << ordinal;
        hash_key = oss.str();
    }

    uint32_t site_id = fnv1a_hash(hash_key);

    pc_to_site_[sass_pc] = site_id;

    NvbitSourceLoc loc;
    loc.filename = filename;
    loc.function = function;
    loc.line = line;
    loc.column = 0;
    loc.event_type = event_type;
    loc.ordinal = ordinal;

    sites_.push_back({site_id, sass_pc, loc});

    return site_id;
}

uint32_t NvbitSiteTable::lookup(uint32_t sass_pc) const {
    auto it = pc_to_site_.find(sass_pc);
    return (it != pc_to_site_.end()) ? it->second : sass_pc;
}

void NvbitSiteTable::export_json(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "[prlx-nvbit] Failed to write site table to %s\n", path.c_str());
        return;
    }

    fprintf(f, "[\n");
    for (size_t i = 0; i < sites_.size(); i++) {
        const auto& s = sites_[i];
        // Escape filename for JSON (basic: replace \ with \\, " with \")
        std::string escaped_filename;
        for (char c : s.loc.filename) {
            if (c == '\\') escaped_filename += "\\\\";
            else if (c == '"') escaped_filename += "\\\"";
            else escaped_filename += c;
        }
        std::string escaped_function;
        for (char c : s.loc.function) {
            if (c == '\\') escaped_function += "\\\\";
            else if (c == '"') escaped_function += "\\\"";
            else escaped_function += c;
        }

        fprintf(f, "  {\"site_id\": %u, \"filename\": \"%s\", \"function\": \"%s\", "
                   "\"line\": %u, \"column\": %u, \"event_type\": %u, \"ordinal\": %u}%s\n",
                s.site_id,
                escaped_filename.c_str(),
                escaped_function.c_str(),
                s.loc.line,
                s.loc.column,
                s.loc.event_type,
                s.loc.ordinal,
                (i + 1 < sites_.size()) ? "," : "");
    }
    fprintf(f, "]\n");
    fclose(f);

    fprintf(stderr, "[prlx-nvbit] Wrote %zu sites to %s\n", sites_.size(), path.c_str());
}

} // namespace prlx
