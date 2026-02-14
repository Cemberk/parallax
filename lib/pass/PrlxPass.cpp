#include "PrlxPass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "../common/trace_format.h"
#include <string>
#include <cstdlib>
#include <sstream>
#include <set>

using namespace llvm;

namespace prlx {

// ---- Selective Instrumentation: Compile-Time Filter ----

void PrlxPass::loadFilters() {
    if (filters_loaded_) return;
    filters_loaded_ = true;

    // Supports comma-separated patterns: "compute_attention,matmul_*"
    const char* env_filter = std::getenv("PRLX_FILTER");
    if (!env_filter) return;

    std::string filters(env_filter);
    if (filters.empty()) return;

    std::istringstream ss(filters);
    std::string pattern;
    while (std::getline(ss, pattern, ',')) {
        size_t start = pattern.find_first_not_of(" \t");
        size_t end = pattern.find_last_not_of(" \t");
        if (start != std::string::npos) {
            filter_patterns_.push_back(pattern.substr(start, end - start + 1));
        }
    }

    if (!filter_patterns_.empty()) {
        errs() << "[prlx] Selective instrumentation enabled. Filters:\n";
        for (const auto& p : filter_patterns_) {
            errs() << "[prlx]   - " << p << "\n";
        }
    }
}

bool PrlxPass::globMatch(const std::string& pattern, const std::string& text) {
    size_t pi = 0, ti = 0;
    size_t star_p = std::string::npos, star_t = 0;

    while (ti < text.size()) {
        if (pi < pattern.size() && (pattern[pi] == text[ti] || pattern[pi] == '?')) {
            pi++;
            ti++;
        } else if (pi < pattern.size() && pattern[pi] == '*') {
            star_p = pi++;
            star_t = ti;
        } else if (star_p != std::string::npos) {
            pi = star_p + 1;
            ti = ++star_t;
        } else {
            return false;
        }
    }

    while (pi < pattern.size() && pattern[pi] == '*') {
        pi++;
    }

    return pi == pattern.size();
}

bool PrlxPass::matchesFilter(const Function& F) const {
    if (filter_patterns_.empty()) return true;

    std::string fname = F.getName().str();

    for (const auto& pattern : filter_patterns_) {
        if (globMatch(pattern, fname)) return true;

        // CUDA mangles kernel names, so "my_kernel" might appear as
        // "_Z9my_kernelPiS_ii" -- also check as substring
        if (pattern.find('*') == std::string::npos &&
            pattern.find('?') == std::string::npos) {
            if (fname.find(pattern) != std::string::npos) return true;
        }
    }

    return false;
}

bool PrlxPass::isNVPTXModule(const Module& M) const {
    std::string triple = M.getTargetTriple();
    return triple.find("nvptx") != std::string::npos;
}

bool PrlxPass::isDeviceFunction(const Function& F) const {
    // In an NVPTX module, ALL non-declaration functions are device functions.
    // Clang-compiled CUDA uses the default C calling convention in IR,
    // and marks kernels via module-level !nvvm.annotations metadata (not
    // function-level metadata). So we check the module's target triple
    // instead of relying on calling conventions or function metadata.
    const Module* M = F.getParent();
    if (M && isNVPTXModule(*M)) {
        return true;
    }

    // Fallback: check PTX calling conventions (set by some frontends)
    CallingConv::ID CC = F.getCallingConv();
    if (CC == 71 || CC == 72) {  // PTX_Kernel / PTX_Device
        return true;
    }

    return false;
}

void PrlxPass::declareRuntimeFunctions(Module& M) {
    LLVMContext& Ctx = M.getContext();
    Type* VoidTy = Type::getVoidTy(Ctx);
    Type* I32Ty = Type::getInt32Ty(Ctx);
    Type* I8Ty = Type::getInt8Ty(Ctx);

    // Reuse existing function if already in the module (e.g., from runtime source
    // compiled in the same translation unit). Creating a duplicate would cause
    // LLVM to auto-rename it, producing unresolvable symbols like __prlx_record_branch3.
    auto getOrDeclare = [&](const char* name, FunctionType* FTy) -> Function* {
        if (Function* F = M.getFunction(name)) {
            return F;
        }
        return Function::Create(FTy, Function::ExternalLinkage, name, M);
    };

    // void __prlx_record_branch(uint32_t site_id, uint32_t condition, uint32_t operand_a)
    FunctionType* BranchFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty}, false);
    record_branch_fn_ = getOrDeclare("__prlx_record_branch", BranchFnTy);

    // void __prlx_record_shmem_store(uint32_t site_id, uint32_t address, uint32_t value)
    FunctionType* ShMemFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty}, false);
    record_shmem_store_fn_ = getOrDeclare("__prlx_record_shmem_store", ShMemFnTy);

    // void __prlx_record_atomic(uint32_t site_id, uint32_t address, uint32_t operand, uint32_t result)
    FunctionType* AtomicFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty, I32Ty}, false);
    record_atomic_fn_ = getOrDeclare("__prlx_record_atomic", AtomicFnTy);

    // void __prlx_record_func(uint32_t site_id, uint8_t is_entry, uint32_t arg0)
    FunctionType* FuncFnTy = FunctionType::get(VoidTy, {I32Ty, I8Ty, I32Ty}, false);
    record_func_fn_ = getOrDeclare("__prlx_record_func", FuncFnTy);

    // void __prlx_record_value(uint32_t site_id, uint32_t value) [time-travel]
    FunctionType* ValueFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty}, false);
    record_value_fn_ = getOrDeclare("__prlx_record_value", ValueFnTy);

    // void __prlx_record_snapshot(uint32_t site_id, uint32_t lhs, uint32_t rhs, uint32_t cmp_pred) [crash dump]
    FunctionType* SnapFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty, I32Ty}, false);
    record_snapshot_fn_ = getOrDeclare("__prlx_record_snapshot", SnapFnTy);
}

void PrlxPass::declareTraceBufferGlobal(Module& M) {
    // May already exist from runtime linkage
    if (M.getGlobalVariable("g_prlx_buffer")) {
        return;
    }

    LLVMContext& Ctx = M.getContext();
    Type* PtrTy = PointerType::getUnqual(Ctx);

    GlobalVariable* GV = new GlobalVariable(
        M, PtrTy, false, GlobalValue::ExternalLinkage, nullptr, "g_prlx_buffer"
    );
}

void PrlxPass::instrumentBranch(BranchInst* BI, SiteTable& siteTable, Module& M) {
    if (!BI->isConditional()) return;

    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    Value* Cond = BI->getCondition();
    uint32_t site_id = siteTable.getSiteId(BI, EVENT_BRANCH);

    Builder.SetInsertPoint(BI);

    Value* CondI32 = Builder.CreateZExt(Cond, Type::getInt32Ty(Ctx));
    Value* OperandA = ConstantInt::get(Type::getInt32Ty(Ctx), 0);

    if (auto* CmpI = dyn_cast<ICmpInst>(Cond)) {
        Value* LHS = CmpI->getOperand(0);
        if (LHS->getType()->isIntegerTy()) {
            if (LHS->getType()->getIntegerBitWidth() <= 32) {
                OperandA = Builder.CreateZExtOrTrunc(LHS, Type::getInt32Ty(Ctx));
            }
        }
    } else if (auto* FCmpI = dyn_cast<FCmpInst>(Cond)) {
        Value* LHS = FCmpI->getOperand(0);
        if (LHS->getType()->isFloatTy()) {
            OperandA = Builder.CreateBitCast(LHS, Type::getInt32Ty(Ctx));
        } else if (LHS->getType()->isDoubleTy()) {
            Value* AsI64 = Builder.CreateBitCast(LHS, Type::getInt64Ty(Ctx));
            OperandA = Builder.CreateTrunc(AsI64, Type::getInt32Ty(Ctx));
        }
    }

    CallInst* CI = Builder.CreateCall(record_branch_fn_, {
        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
        CondI32,
        OperandA
    });

    // CRITICAL: Mark as convergent so LLVM doesn't move it
    // __prlx_record_branch uses __activemask()/__ballot() which are convergent operations
    // Moving the call changes the active mask, breaking correctness
    CI->addFnAttr(Attribute::Convergent);

    if (auto* CmpI = dyn_cast<CmpInst>(Cond)) {
        instrumentSnapshot(CmpI, site_id, Builder, M);
    }
}

void PrlxPass::instrumentSnapshot(CmpInst* CI, uint32_t site_id,
                                         IRBuilder<>& Builder, Module& M) {
    LLVMContext& Ctx = M.getContext();

    Value* LHS = CI->getOperand(0);
    Value* RHS = CI->getOperand(1);

    Value* LhsI32 = ConstantInt::get(Type::getInt32Ty(Ctx), 0);
    if (LHS->getType()->isIntegerTy()) {
        unsigned BitW = LHS->getType()->getIntegerBitWidth();
        if (BitW <= 32) {
            LhsI32 = Builder.CreateZExtOrTrunc(LHS, Type::getInt32Ty(Ctx));
        } else {
            LhsI32 = Builder.CreateTrunc(LHS, Type::getInt32Ty(Ctx));
        }
    } else if (LHS->getType()->isFloatTy()) {
        LhsI32 = Builder.CreateBitCast(LHS, Type::getInt32Ty(Ctx));
    } else if (LHS->getType()->isDoubleTy()) {
        Value* AsI64 = Builder.CreateBitCast(LHS, Type::getInt64Ty(Ctx));
        LhsI32 = Builder.CreateTrunc(AsI64, Type::getInt32Ty(Ctx));
    }

    Value* RhsI32 = ConstantInt::get(Type::getInt32Ty(Ctx), 0);
    if (RHS->getType()->isIntegerTy()) {
        unsigned BitW = RHS->getType()->getIntegerBitWidth();
        if (BitW <= 32) {
            RhsI32 = Builder.CreateZExtOrTrunc(RHS, Type::getInt32Ty(Ctx));
        } else {
            RhsI32 = Builder.CreateTrunc(RHS, Type::getInt32Ty(Ctx));
        }
    } else if (RHS->getType()->isFloatTy()) {
        RhsI32 = Builder.CreateBitCast(RHS, Type::getInt32Ty(Ctx));
    } else if (RHS->getType()->isDoubleTy()) {
        Value* AsI64 = Builder.CreateBitCast(RHS, Type::getInt64Ty(Ctx));
        RhsI32 = Builder.CreateTrunc(AsI64, Type::getInt32Ty(Ctx));
    }

    uint32_t predicate = CI->getPredicate();

    CallInst* SnapCall = Builder.CreateCall(record_snapshot_fn_, {
        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
        LhsI32,
        RhsI32,
        ConstantInt::get(Type::getInt32Ty(Ctx), predicate)
    });

    // CRITICAL: ALL active lanes must execute __prlx_record_snapshot
    // because it uses __shfl_sync() internally to gather per-lane values
    SnapCall->addFnAttr(Attribute::Convergent);
}

void PrlxPass::instrumentSharedMemStore(StoreInst* SI, SiteTable& siteTable, Module& M) {
    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    Value* StoredVal = SI->getValueOperand();
    Value* Ptr = SI->getPointerOperand();
    uint32_t site_id = siteTable.getSiteId(SI, EVENT_SHMEM_STORE);

    Builder.SetInsertPoint(SI);

    Value* AddrInt = Builder.CreatePtrToInt(Ptr, Type::getInt32Ty(Ctx));
    Value* ValI32;
    if (StoredVal->getType()->isIntegerTy()) {
        ValI32 = Builder.CreateZExtOrTrunc(StoredVal, Type::getInt32Ty(Ctx));
    } else if (StoredVal->getType()->isFloatTy()) {
        ValI32 = Builder.CreateBitCast(StoredVal, Type::getInt32Ty(Ctx));
    } else if (StoredVal->getType()->isDoubleTy()) {
        Value* AsI64 = Builder.CreateBitCast(StoredVal, Type::getInt64Ty(Ctx));
        ValI32 = Builder.CreateTrunc(AsI64, Type::getInt32Ty(Ctx));
    } else {
        // Unsupported type - skip
        return;
    }

    CallInst* CI = Builder.CreateCall(record_shmem_store_fn_, {
        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
        AddrInt,
        ValI32
    });

    // CRITICAL: Mark as convergent (uses __activemask())
    CI->addFnAttr(Attribute::Convergent);
}

void PrlxPass::instrumentAtomic(AtomicRMWInst* AI, SiteTable& siteTable, Module& M) {
    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    uint32_t site_id = siteTable.getSiteId(AI, EVENT_ATOMIC);
    Builder.SetInsertPoint(AI);

    Value* Ptr = AI->getPointerOperand();
    Value* Val = AI->getValOperand();

    Value* AddrInt = Builder.CreatePtrToInt(Ptr, Type::getInt32Ty(Ctx));
    Value* ValI32;
    if (Val->getType()->isIntegerTy()) {
        ValI32 = Builder.CreateZExtOrTrunc(Val, Type::getInt32Ty(Ctx));
    } else {
        return;  // Skip non-integer atomics for now
    }

    // Result recording requires splitting the block; for now just record the operand
    CallInst* CI = Builder.CreateCall(record_atomic_fn_, {
        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
        AddrInt,
        ValI32,
        ConstantInt::get(Type::getInt32Ty(Ctx), 0)  // Result placeholder
    });

    // CRITICAL: Mark as convergent (uses __activemask())
    CI->addFnAttr(Attribute::Convergent);
}

void PrlxPass::instrumentCmpXchg(AtomicCmpXchgInst* CI, SiteTable& siteTable, Module& M) {
    // NOT YET IMPLEMENTED: CmpXchg (atomic compare-and-swap) instrumentation
    // requires splitting the basic block to capture the result after the
    // atomic completes. AtomicCmpXchg instructions will not appear in traces.
}

void PrlxPass::instrumentValueCaptures(BranchInst* BI, SiteTable& siteTable, Module& M) {
    // Time-travel: walk backward from branch condition to find contributing loads.
    // For each load that feeds the branch, insert __prlx_record_value() after it.
    // This captures the data that led to the branch decision.
    if (!BI->isConditional()) return;

    LLVMContext& Ctx = M.getContext();
    Value* Cond = BI->getCondition();

    std::vector<Value*> operands;

    if (auto* CmpI = dyn_cast<CmpInst>(Cond)) {
        operands.push_back(CmpI->getOperand(0));
        operands.push_back(CmpI->getOperand(1));
    } else {
        operands.push_back(Cond);
    }

    std::set<Instruction*> instrumented;
    for (Value* Op : operands) {
        Value* Current = Op;
        for (int hop = 0; hop < 2; hop++) {
            auto* Inst = dyn_cast<Instruction>(Current);
            if (!Inst) break;

            if (auto* LI = dyn_cast<LoadInst>(Inst)) {
                if (instrumented.count(LI)) break;
                instrumented.insert(LI);

                IRBuilder<> Builder(Ctx);
                Builder.SetInsertPoint(LI->getNextNonDebugInstruction());

                uint32_t site_id = siteTable.getSiteId(LI, EVENT_BRANCH);

                Value* ValI32 = nullptr;
                Type* LoadTy = LI->getType();
                if (LoadTy->isIntegerTy() && LoadTy->getIntegerBitWidth() <= 32) {
                    ValI32 = Builder.CreateZExtOrTrunc(LI, Type::getInt32Ty(Ctx));
                } else if (LoadTy->isFloatTy()) {
                    ValI32 = Builder.CreateBitCast(LI, Type::getInt32Ty(Ctx));
                } else if (LoadTy->isDoubleTy()) {
                    Value* AsI64 = Builder.CreateBitCast(LI, Type::getInt64Ty(Ctx));
                    ValI32 = Builder.CreateTrunc(AsI64, Type::getInt32Ty(Ctx));
                }

                if (ValI32) {
                    CallInst* CI = Builder.CreateCall(record_value_fn_, {
                        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
                        ValI32
                    });
                    CI->addFnAttr(Attribute::Convergent);
                }
                break;
            }

            if (Inst->getNumOperands() >= 1 && isa<CastInst>(Inst)) {
                Current = Inst->getOperand(0);
            } else {
                break;
            }
        }
    }
}

void PrlxPass::instrumentPredicatedOps(
    std::vector<CmpInst*>& predicates, SiteTable& siteTable, Module& M) {
    // Triton fully eliminates branches by lowering them to predicated PTX
    // instructions via inline asm. Example Triton IR:
    //
    //   %pred = icmp slt i32 %idx, %n_elements
    //   call i32 asm sideeffect "@$2 ld.global.b32 ...", "=r,l,b"(..., i1 %pred)
    //   call void asm sideeffect "@$2 st.global.b32 ...", "r,l,b"(..., i1 %pred)
    //
    // These predicated loads/stores are semantically equivalent to:
    //   if (idx < n_elements) { *out = *a + *b; }
    //
    // We instrument the comparison that produces the predicate, recording
    // the same EVENT_BRANCH data so the differ treats predicate divergence
    // identically to branch divergence.

    LLVMContext& Ctx = M.getContext();

    for (CmpInst* CI : predicates) {
        Instruction* InsertPt = CI->getNextNonDebugInstruction();
        if (!InsertPt) continue;

        IRBuilder<> Builder(InsertPt);

        uint32_t site_id = siteTable.getSiteId(CI, EVENT_BRANCH);

        Value* CondI32 = Builder.CreateZExt(CI, Type::getInt32Ty(Ctx));
        Value* OperandA = ConstantInt::get(Type::getInt32Ty(Ctx), 0);
        Value* LHS = CI->getOperand(0);
        if (LHS->getType()->isIntegerTy()) {
            if (LHS->getType()->getIntegerBitWidth() <= 32) {
                OperandA = Builder.CreateZExtOrTrunc(LHS, Type::getInt32Ty(Ctx));
            }
        } else if (LHS->getType()->isFloatTy()) {
            OperandA = Builder.CreateBitCast(LHS, Type::getInt32Ty(Ctx));
        } else if (LHS->getType()->isDoubleTy()) {
            Value* AsI64 = Builder.CreateBitCast(LHS, Type::getInt64Ty(Ctx));
            OperandA = Builder.CreateTrunc(AsI64, Type::getInt32Ty(Ctx));
        }

        CallInst* RecordCall = Builder.CreateCall(record_branch_fn_, {
            ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
            CondI32,
            OperandA
        });
        RecordCall->addFnAttr(Attribute::Convergent);

        instrumentSnapshot(CI, site_id, Builder, M);
    }
}

void PrlxPass::embedSiteTable(Module& M, const SiteTable& siteTable) {
    std::string filename = "prlx-sites.json";
    if (const char* env_path = std::getenv("PRLX_SITES")) {
        filename = env_path;
    }

    if (siteTable.exportToJSON(filename)) {
        errs() << "[prlx] Exported site table to: " << filename << "\n";
    } else {
        errs() << "[prlx] WARNING: Failed to export site table\n";
    }
}

PreservedAnalyses PrlxPass::run(Module& M, ModuleAnalysisManager& AM) {
    if (!isNVPTXModule(M)) {
        return PreservedAnalyses::all();
    }

    errs() << "[prlx] Instrumenting NVPTX module: " << M.getName() << "\n";

    loadFilters();
    declareRuntimeFunctions(M);
    declareTraceBufferGlobal(M);

    SiteTable siteTable;

    unsigned skipped = 0;

    for (Function& F : M) {
        if (F.isDeclaration()) continue;
        if (!isDeviceFunction(F)) continue;

        // Never instrument our own runtime functions or CUDA builtins used
        // by the runtime. Instrumenting atomicAdd/__activemask/etc. would
        // cause infinite recursion since __prlx_record_* calls them internally.
        StringRef Name = F.getName();
        if (Name.starts_with("__prlx_")) continue;

        // Skip CUDA builtin wrappers (mangled names contain these substrings)
        if (Name.contains("atomicAdd") || Name.contains("atomicSub") ||
            Name.contains("atomicExch") || Name.contains("atomicMin") ||
            Name.contains("atomicMax") || Name.contains("atomicCAS") ||
            Name.contains("atomicAnd") || Name.contains("atomicOr") ||
            Name.contains("atomicXor") || Name.contains("__activemask") ||
            Name.contains("__ballot") || Name.contains("__shfl") ||
            Name.contains("__syncthreads") || Name.contains("__threadfence") ||
            Name.contains("__ldg") || Name.contains("__stcg")) continue;

        if (!matchesFilter(F)) {
            skipped++;
            continue;
        }

        errs() << "[prlx]   Instrumenting function: " << F.getName() << " ("
               << F.size() << " blocks)\n";

        // Collect first, then instrument (avoids iterator invalidation)
        std::vector<BranchInst*> branches;
        std::vector<StoreInst*> shmem_stores;
        std::vector<AtomicRMWInst*> atomics;
        std::vector<AtomicCmpXchgInst*> cmpxchgs;
        std::vector<CmpInst*> predicates;

        // Avoid double-recording comparisons that already feed branches
        std::set<Value*> branchConditions;

        for (BasicBlock& BB : F) {
            if (auto* BI = dyn_cast<BranchInst>(BB.getTerminator())) {
                if (BI->isConditional()) {
                    branches.push_back(BI);
                    branchConditions.insert(BI->getCondition());
                }
            }

            for (Instruction& I : BB) {
                if (auto* SI = dyn_cast<StoreInst>(&I)) {
                    if (SI->getPointerAddressSpace() == 3) {  // Shared memory
                        shmem_stores.push_back(SI);
                    }
                }
                if (auto* AI = dyn_cast<AtomicRMWInst>(&I)) {
                    atomics.push_back(AI);
                }
                if (auto* CXI = dyn_cast<AtomicCmpXchgInst>(&I)) {
                    cmpxchgs.push_back(CXI);
                }

                // Triton predicated execution: comparisons feeding inline asm or select
                if (auto* CI = dyn_cast<CmpInst>(&I)) {
                    if (branchConditions.count(CI)) continue;

                    bool isPredicate = false;
                    for (User* U : CI->users()) {
                        if (auto* Call = dyn_cast<CallInst>(U)) {
                            if (Call->isInlineAsm()) {
                                isPredicate = true;
                                break;
                            }
                        }
                        if (isa<SelectInst>(U)) {
                            isPredicate = true;
                            break;
                        }
                    }
                    if (isPredicate) {
                        predicates.push_back(CI);
                    }
                }
            }
        }

        for (auto* BI : branches) {
            instrumentBranch(BI, siteTable, M);
            instrumentValueCaptures(BI, siteTable, M);  // Time-travel history
        }
        instrumentPredicatedOps(predicates, siteTable, M);
        for (auto* SI : shmem_stores) {
            instrumentSharedMemStore(SI, siteTable, M);
        }
        for (auto* AI : atomics) {
            instrumentAtomic(AI, siteTable, M);
        }
        for (auto* CXI : cmpxchgs) {
            instrumentCmpXchg(CXI, siteTable, M);
        }
    }

    errs() << "[prlx] Instrumented " << siteTable.getSites().size() << " sites\n";
    if (skipped > 0) {
        errs() << "[prlx] Skipped " << skipped << " functions (filtered out)\n";
    }

    embedSiteTable(M, siteTable);

    return PreservedAnalyses::none();
}

} // namespace prlx

// Plugin registration for new pass manager
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "PrlxPass", LLVM_VERSION_STRING,
        [](PassBuilder& PB) {
            // Register pass to run automatically in optimization pipeline
            PB.registerOptimizerEarlyEPCallback(
                [](ModulePassManager& MPM, OptimizationLevel OL
#if LLVM_VERSION_MAJOR >= 20
                   , ThinOrFullLTOPhase
#endif
                  ) {
                    MPM.addPass(prlx::PrlxPass());
                });

            // Also allow explicit invocation via -passes=prlx
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager& MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "prlx") {
                        MPM.addPass(prlx::PrlxPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
