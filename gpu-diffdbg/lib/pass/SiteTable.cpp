#include "SiteTable.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
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

    // Compute per-(function, event_type) ordinal for cross-compilation stability.
    // This ensures that even without debug info, the Nth branch in function F
    // always gets the same ordinal regardless of optimization level or
    // recompilation, as long as the source code hasn't changed.
    std::string ordinal_key = loc.function_name + ":" + std::to_string(event_type);
    uint32_t ordinal = ordinal_counters_[ordinal_key]++;

    // Create deterministic hash of source location
    std::ostringstream oss;
    if (loc.line == 0 && loc.column == 0) {
        // No debug info: use function:event_type:ordinal for stability
        oss << loc.function_name << ":" << (int)event_type << ":ord" << ordinal;
    } else {
        // With debug info: use full source location
        oss << loc.filename << ":" << loc.function_name << ":"
            << loc.line << ":" << loc.column << ":" << (int)event_type;
    }

    std::string location_str = oss.str();
    uint32_t site_id = fnv1a_hash(location_str);

    // Record new site
    SiteInfo info;
    info.site_id = site_id;
    info.location = loc;
    info.event_type = event_type;
    info.ordinal = ordinal;

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

bool SiteTable::exportToJSON(const std::string& filename) const {
    std::error_code EC;
    llvm::raw_fd_ostream OS(filename, EC, llvm::sys::fs::OF_None);
    if (EC) {
        llvm::errs() << "[gddbg] Error opening site table file: " << EC.message() << "\n";
        return false;
    }

    // Write JSON array
    OS << "[\n";
    for (size_t i = 0; i < sites_.size(); ++i) {
        const SiteInfo& site = sites_[i];

        // Escape special characters in strings
        auto escapeJSON = [](const std::string& s) -> std::string {
            std::string result;
            for (char c : s) {
                switch (c) {
                    case '\\': result += "\\\\"; break;
                    case '"': result += "\\\""; break;
                    case '\n': result += "\\n"; break;
                    case '\r': result += "\\r"; break;
                    case '\t': result += "\\t"; break;
                    default: result += c; break;
                }
            }
            return result;
        };

        OS << "  {\n";
        OS << "    \"site_id\": " << site.site_id << ",\n";
        OS << "    \"filename\": \"" << escapeJSON(site.location.filename) << "\",\n";
        OS << "    \"function\": \"" << escapeJSON(site.location.function_name) << "\",\n";
        OS << "    \"line\": " << site.location.line << ",\n";
        OS << "    \"column\": " << site.location.column << ",\n";
        OS << "    \"event_type\": " << (int)site.event_type << ",\n";
        OS << "    \"ordinal\": " << site.ordinal << "\n";
        OS << "  }";
        if (i < sites_.size() - 1) {
            OS << ",";
        }
        OS << "\n";
    }
    OS << "]\n";

    return true;
}

} // namespace gddbg
