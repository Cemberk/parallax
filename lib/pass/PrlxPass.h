#ifndef PRLX_PASS_H
#define PRLX_PASS_H

#include "SiteTable.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include <string>
#include <vector>

namespace prlx {

class PrlxPass : public llvm::PassInfoMixin<PrlxPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);

private:
    bool isNVPTXModule(const llvm::Module& M) const;
    bool isDeviceFunction(const llvm::Function& F) const;
    bool matchesFilter(const llvm::Function& F) const;

    static bool globMatch(const std::string& pattern, const std::string& text);
    void loadFilters();

    void declareRuntimeFunctions(llvm::Module& M);
    void declareTraceBufferGlobal(llvm::Module& M);
    void instrumentBranch(llvm::BranchInst* BI, SiteTable& siteTable, llvm::Module& M);
    void instrumentSharedMemStore(llvm::StoreInst* SI, SiteTable& siteTable, llvm::Module& M);
    void instrumentGlobalMemStore(llvm::StoreInst* SI, SiteTable& siteTable, llvm::Module& M);
    void instrumentAtomic(llvm::AtomicRMWInst* AI, SiteTable& siteTable, llvm::Module& M);
    void instrumentCmpXchg(llvm::AtomicCmpXchgInst* CI, SiteTable& siteTable, llvm::Module& M);
    void instrumentValueCaptures(llvm::BranchInst* BI, SiteTable& siteTable, llvm::Module& M);

    // Triton's pattern: icmp/fcmp feeding inline asm predicates instead of
    // branches. Semantically equivalent to branch instrumentation.
    void instrumentPredicatedOps(std::vector<llvm::CmpInst*>& predicates,
                                 SiteTable& siteTable, llvm::Module& M);

    void instrumentSnapshot(llvm::CmpInst* CI, uint32_t site_id,
                           llvm::IRBuilder<>& Builder, llvm::Module& M);
    void embedSiteTable(llvm::Module& M, const SiteTable& siteTable);

    // Runtime function declarations (cached)
    llvm::Function* record_branch_fn_ = nullptr;
    llvm::Function* record_shmem_store_fn_ = nullptr;
    llvm::Function* record_global_store_fn_ = nullptr;
    llvm::Function* record_atomic_fn_ = nullptr;
    llvm::Function* record_func_fn_ = nullptr;
    llvm::Function* record_value_fn_ = nullptr;
    llvm::Function* record_snapshot_fn_ = nullptr;

    // Selective instrumentation: function name filters
    // Empty = instrument everything (default)
    std::vector<std::string> filter_patterns_;
    bool filters_loaded_ = false;
};

} // namespace prlx

#endif // PRLX_PASS_H
