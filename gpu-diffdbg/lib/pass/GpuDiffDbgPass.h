#ifndef GDDBG_PASS_H
#define GDDBG_PASS_H

#include "SiteTable.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include <string>
#include <vector>

namespace gddbg {

class GpuDiffDbgPass : public llvm::PassInfoMixin<GpuDiffDbgPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);

private:
    // Check if this is an NVPTX module (device code)
    bool isNVPTXModule(const llvm::Module& M) const;

    // Check if a function is a device function
    bool isDeviceFunction(const llvm::Function& F) const;

    // Check if a function matches the user's filter pattern
    // Returns true if the function should be instrumented
    bool matchesFilter(const llvm::Function& F) const;

    // Simple glob matching: supports '*' wildcard
    static bool globMatch(const std::string& pattern, const std::string& text);

    // Load filter patterns from environment or command-line option
    void loadFilters();

    // Declare runtime recording functions in the module
    void declareRuntimeFunctions(llvm::Module& M);

    // Declare the global device variable for trace buffer pointer
    void declareTraceBufferGlobal(llvm::Module& M);

    // Instrument a conditional branch
    void instrumentBranch(llvm::BranchInst* BI, SiteTable& siteTable, llvm::Module& M);

    // Instrument a shared memory store
    void instrumentSharedMemStore(llvm::StoreInst* SI, SiteTable& siteTable, llvm::Module& M);

    // Instrument an atomic operation
    void instrumentAtomic(llvm::AtomicRMWInst* AI, SiteTable& siteTable, llvm::Module& M);

    // Instrument a compare-exchange atomic
    void instrumentCmpXchg(llvm::AtomicCmpXchgInst* CI, SiteTable& siteTable, llvm::Module& M);

    // Embed site table as global constant in the module
    void embedSiteTable(llvm::Module& M, const SiteTable& siteTable);

    // Runtime function declarations (cached)
    llvm::Function* record_branch_fn_ = nullptr;
    llvm::Function* record_shmem_store_fn_ = nullptr;
    llvm::Function* record_atomic_fn_ = nullptr;
    llvm::Function* record_func_fn_ = nullptr;

    // Selective instrumentation: function name filters
    // Empty = instrument everything (default)
    std::vector<std::string> filter_patterns_;
    bool filters_loaded_ = false;
};

} // namespace gddbg

#endif // GDDBG_PASS_H
