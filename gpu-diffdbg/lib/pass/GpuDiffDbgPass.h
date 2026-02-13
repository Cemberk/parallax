#ifndef GDDBG_PASS_H
#define GDDBG_PASS_H

#include "SiteTable.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace gddbg {

class GpuDiffDbgPass : public llvm::PassInfoMixin<GpuDiffDbgPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);

private:
    // Check if this is an NVPTX module (device code)
    bool isNVPTXModule(const llvm::Module& M) const;

    // Check if a function is a device function
    bool isDeviceFunction(const llvm::Function& F) const;

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
};

} // namespace gddbg

#endif // GDDBG_PASS_H
