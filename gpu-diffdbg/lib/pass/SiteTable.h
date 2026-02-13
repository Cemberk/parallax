#ifndef GDDBG_SITE_TABLE_H
#define GDDBG_SITE_TABLE_H

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gddbg {

struct SourceLocation {
    std::string filename;
    std::string function_name;
    uint32_t line;
    uint32_t column;

    std::string toString() const;
};

struct SiteInfo {
    uint32_t site_id;
    SourceLocation location;
    uint8_t event_type;
};

// Manages site_id generation and source location mapping
class SiteTable {
public:
    SiteTable() = default;

    // Generate a unique, deterministic site_id for an instruction
    // Uses FNV-1a hash of (filename:function:line:column)
    // This ensures site IDs are stable across compilations (Death Valley critical!)
    uint32_t getSiteId(const llvm::Instruction* I, uint8_t event_type);

    // Get all recorded sites
    const std::vector<SiteInfo>& getSites() const { return sites_; }

    // Get site info by ID
    const SiteInfo* getSiteInfo(uint32_t site_id) const;

private:
    // Extract source location from an instruction
    SourceLocation getSourceLocation(const llvm::Instruction* I) const;

    // FNV-1a hash implementation (32-bit)
    static uint32_t fnv1a_hash(const std::string& str);

    // Map from site_id to index in sites_ vector
    std::unordered_map<uint32_t, size_t> site_map_;

    // All recorded sites
    std::vector<SiteInfo> sites_;
};

} // namespace gddbg

#endif // GDDBG_SITE_TABLE_H
