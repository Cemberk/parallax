#include "SiteTable.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include <sstream>

namespace gddbg {

std::string SourceLocation::toString() const {
    std::ostringstream oss;
    oss << filename << ":" << function_name << ":" << line << ":" << column;
    return oss.str();
}

SourceLocation SiteTable::getSourceLocation(const llvm::Instruction* I) const {
    SourceLocation loc;

    // Try to get debug location
    const llvm::DebugLoc& debugLoc = I->getDebugLoc();
    if (debugLoc) {
        loc.filename = debugLoc->getFilename().str();
        loc.line = debugLoc->getLine();
        loc.column = debugLoc->getColumn();
    } else {
        // Fallback: use "unknown" for missing debug info
        loc.filename = "unknown";
        loc.line = 0;
        loc.column = 0;
    }

    // Get function name
    if (I->getParent() && I->getParent()->getParent()) {
        loc.function_name = I->getParent()->getParent()->getName().str();
    } else {
        loc.function_name = "unknown";
    }

    return loc;
}

// FNV-1a hash (32-bit) - deterministic and fast
uint32_t SiteTable::fnv1a_hash(const std::string& str) {
    uint32_t hash = 2166136261U;  // FNV offset basis (32-bit)
    for (char c : str) {
        hash ^= static_cast<uint32_t>(c);
        hash *= 16777619U;  // FNV prime (32-bit)
    }
    return hash;
}

uint32_t SiteTable::getSiteId(const llvm::Instruction* I, uint8_t event_type) {
    SourceLocation loc = getSourceLocation(I);

    // Create deterministic hash of source location
    // Format: filename:function:line:column:event_type
    std::ostringstream oss;
    oss << loc.filename << ":" << loc.function_name << ":"
        << loc.line << ":" << loc.column << ":" << (int)event_type;

    std::string location_str = oss.str();
    uint32_t site_id = fnv1a_hash(location_str);

    // Check if we've already recorded this site
    auto it = site_map_.find(site_id);
    if (it != site_map_.end()) {
        return site_id;
    }

    // Record new site
    SiteInfo info;
    info.site_id = site_id;
    info.location = loc;
    info.event_type = event_type;

    size_t idx = sites_.size();
    sites_.push_back(info);
    site_map_[site_id] = idx;

    return site_id;
}

const SiteInfo* SiteTable::getSiteInfo(uint32_t site_id) const {
    auto it = site_map_.find(site_id);
    if (it != site_map_.end()) {
        return &sites_[it->second];
    }
    return nullptr;
}

} // namespace gddbg
