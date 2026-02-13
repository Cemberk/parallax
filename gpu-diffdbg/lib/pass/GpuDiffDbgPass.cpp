#include "GpuDiffDbgPass.h"
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

using namespace llvm;

namespace gddbg {

bool GpuDiffDbgPass::isNVPTXModule(const Module& M) const {
    std::string triple = M.getTargetTriple();
    return triple.find("nvptx") != std::string::npos;
}

bool GpuDiffDbgPass::isDeviceFunction(const Function& F) const {
    // Check if function has kernel or device calling convention
    CallingConv::ID CC = F.getCallingConv();

    // NVPTX calling conventions:
    // CallingConv::PTX_Kernel = 71
    // CallingConv::PTX_Device = 72
    if (CC == 71 || CC == 72) {
        return true;
    }

    // Also check for nvvm annotations
    if (F.hasMetadata("nvvm.annotations")) {
        return true;
    }

    return false;
}

void GpuDiffDbgPass::declareRuntimeFunctions(Module& M) {
    LLVMContext& Ctx = M.getContext();
    Type* VoidTy = Type::getVoidTy(Ctx);
    Type* I32Ty = Type::getInt32Ty(Ctx);
    Type* I8Ty = Type::getInt8Ty(Ctx);

    // void __gddbg_record_branch(uint32_t site_id, uint32_t condition, uint32_t operand_a)
    FunctionType* BranchFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty}, false);
    record_branch_fn_ = Function::Create(BranchFnTy, Function::ExternalLinkage,
                                          "__gddbg_record_branch", M);

    // void __gddbg_record_shmem_store(uint32_t site_id, uint32_t address, uint32_t value)
    FunctionType* ShMemFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty}, false);
    record_shmem_store_fn_ = Function::Create(ShMemFnTy, Function::ExternalLinkage,
                                                "__gddbg_record_shmem_store", M);

    // void __gddbg_record_atomic(uint32_t site_id, uint32_t address, uint32_t operand, uint32_t result)
    FunctionType* AtomicFnTy = FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty, I32Ty}, false);
    record_atomic_fn_ = Function::Create(AtomicFnTy, Function::ExternalLinkage,
                                          "__gddbg_record_atomic", M);

    // void __gddbg_record_func(uint32_t site_id, uint8_t is_entry, uint32_t arg0)
    FunctionType* FuncFnTy = FunctionType::get(VoidTy, {I32Ty, I8Ty, I32Ty}, false);
    record_func_fn_ = Function::Create(FuncFnTy, Function::ExternalLinkage,
                                        "__gddbg_record_func", M);
}

void GpuDiffDbgPass::declareTraceBufferGlobal(Module& M) {
    // Check if g_gddbg_buffer already exists (from runtime linkage)
    if (M.getGlobalVariable("g_gddbg_buffer")) {
        return;
    }

    // Declare: __device__ TraceBuffer* g_gddbg_buffer
    // This is an opaque pointer type (LLVM 18 with opaque pointers)
    LLVMContext& Ctx = M.getContext();
    Type* PtrTy = PointerType::getUnqual(Ctx);  // Opaque pointer in LLVM 18

    GlobalVariable* GV = new GlobalVariable(
        M,
        PtrTy,                             // Type: ptr (opaque pointer to TraceBuffer)
        false,                             // Not constant
        GlobalValue::ExternalLinkage,      // External linkage (defined in runtime)
        nullptr,                           // No initializer (set by host runtime)
        "g_gddbg_buffer"
    );

    // Mark as device-side global (address space 0 for NVPTX is generic/global)
    // No special attributes needed - NVPTX will handle it correctly
}

void GpuDiffDbgPass::instrumentBranch(BranchInst* BI, SiteTable& siteTable, Module& M) {
    if (!BI->isConditional()) return;

    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    // Get the condition value
    Value* Cond = BI->getCondition();

    // Generate site ID
    uint32_t site_id = siteTable.getSiteId(BI, EVENT_BRANCH);

    // Insert instrumentation BEFORE the branch
    // Use SplitBlockAndInsertIfThen to safely handle phi nodes
    Builder.SetInsertPoint(BI);

    // Convert condition to i32 (0 or 1)
    Value* CondI32 = Builder.CreateZExt(Cond, Type::getInt32Ty(Ctx));

    // For operand_a, we want the value being tested
    // If the condition comes from a comparison, extract the operands
    Value* OperandA = ConstantInt::get(Type::getInt32Ty(Ctx), 0);

    if (auto* CmpI = dyn_cast<ICmpInst>(Cond)) {
        // Integer comparison - get the left operand
        Value* LHS = CmpI->getOperand(0);
        if (LHS->getType()->isIntegerTy()) {
            if (LHS->getType()->getIntegerBitWidth() <= 32) {
                OperandA = Builder.CreateZExtOrTrunc(LHS, Type::getInt32Ty(Ctx));
            }
        }
    } else if (auto* FCmpI = dyn_cast<FCmpInst>(Cond)) {
        // Float comparison - bitcast to i32
        Value* LHS = FCmpI->getOperand(0);
        if (LHS->getType()->isFloatTy()) {
            OperandA = Builder.CreateBitCast(LHS, Type::getInt32Ty(Ctx));
        } else if (LHS->getType()->isDoubleTy()) {
            // For double, just take lower 32 bits
            Value* AsI64 = Builder.CreateBitCast(LHS, Type::getInt64Ty(Ctx));
            OperandA = Builder.CreateTrunc(AsI64, Type::getInt32Ty(Ctx));
        }
    }

    // Call __gddbg_record_branch(site_id, condition, operand_a)
    CallInst* CI = Builder.CreateCall(record_branch_fn_, {
        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
        CondI32,
        OperandA
    });

    // CRITICAL: Mark as convergent so LLVM doesn't move it
    // __gddbg_record_branch uses __activemask()/__ballot() which are convergent operations
    // Moving the call changes the active mask, breaking correctness
    CI->addFnAttr(Attribute::Convergent);
}

void GpuDiffDbgPass::instrumentSharedMemStore(StoreInst* SI, SiteTable& siteTable, Module& M) {
    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    // Get the stored value
    Value* StoredVal = SI->getValueOperand();
    Value* Ptr = SI->getPointerOperand();

    // Generate site ID
    uint32_t site_id = siteTable.getSiteId(SI, EVENT_SHMEM_STORE);

    // Insert before the store
    Builder.SetInsertPoint(SI);

    // Convert pointer to integer (address offset)
    Value* AddrInt = Builder.CreatePtrToInt(Ptr, Type::getInt32Ty(Ctx));

    // Convert value to i32
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

    // Call __gddbg_record_shmem_store(site_id, address, value)
    CallInst* CI = Builder.CreateCall(record_shmem_store_fn_, {
        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
        AddrInt,
        ValI32
    });

    // CRITICAL: Mark as convergent (uses __activemask())
    CI->addFnAttr(Attribute::Convergent);
}

void GpuDiffDbgPass::instrumentAtomic(AtomicRMWInst* AI, SiteTable& siteTable, Module& M) {
    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    // Generate site ID
    uint32_t site_id = siteTable.getSiteId(AI, EVENT_ATOMIC);

    // Insert before the atomic
    Builder.SetInsertPoint(AI);

    Value* Ptr = AI->getPointerOperand();
    Value* Val = AI->getValOperand();

    // Convert pointer to i32
    Value* AddrInt = Builder.CreatePtrToInt(Ptr, Type::getInt32Ty(Ctx));

    // Convert operand to i32
    Value* ValI32;
    if (Val->getType()->isIntegerTy()) {
        ValI32 = Builder.CreateZExtOrTrunc(Val, Type::getInt32Ty(Ctx));
    } else {
        return;  // Skip non-integer atomics for now
    }

    // The result will be the return value of the atomic (the old value)
    // We need to insert instrumentation AFTER the atomic to record the result
    // For now, just record the operand (result recording requires splitting the block)

    // Call __gddbg_record_atomic(site_id, address, operand, 0)
    CallInst* CI = Builder.CreateCall(record_atomic_fn_, {
        ConstantInt::get(Type::getInt32Ty(Ctx), site_id),
        AddrInt,
        ValI32,
        ConstantInt::get(Type::getInt32Ty(Ctx), 0)  // Result placeholder
    });

    // CRITICAL: Mark as convergent (uses __activemask())
    CI->addFnAttr(Attribute::Convergent);
}

void GpuDiffDbgPass::instrumentCmpXchg(AtomicCmpXchgInst* CI, SiteTable& siteTable, Module& M) {
    // Similar to atomic RMW - omitted for brevity
    // TODO: implement
}

void GpuDiffDbgPass::embedSiteTable(Module& M, const SiteTable& siteTable) {
    // Export site table to JSON file
    // The differ will read this file to map site_ids to source locations
    std::string filename = "gddbg-sites.json";

    // Try to use GDDBG_SITES environment variable if set
    if (const char* env_path = std::getenv("GDDBG_SITES")) {
        filename = env_path;
    }

    if (siteTable.exportToJSON(filename)) {
        errs() << "[gddbg] Exported site table to: " << filename << "\n";
    } else {
        errs() << "[gddbg] WARNING: Failed to export site table\n";
    }
}

PreservedAnalyses GpuDiffDbgPass::run(Module& M, ModuleAnalysisManager& AM) {
    // 1. Check if this is an NVPTX module
    if (!isNVPTXModule(M)) {
        return PreservedAnalyses::all();
    }

    errs() << "[gddbg] Instrumenting NVPTX module: " << M.getName() << "\n";

    // 2. Declare runtime functions
    declareRuntimeFunctions(M);

    // 3. Declare trace buffer global variable
    declareTraceBufferGlobal(M);

    // 4. Build site table
    SiteTable siteTable;

    // 5. Iterate over all functions
    for (Function& F : M) {
        if (F.isDeclaration()) continue;
        if (!isDeviceFunction(F)) continue;

        errs() << "[gddbg]   Instrumenting function: " << F.getName() << "\n";

        // Collect instructions to instrument (to avoid iterator invalidation)
        std::vector<BranchInst*> branches;
        std::vector<StoreInst*> shmem_stores;
        std::vector<AtomicRMWInst*> atomics;
        std::vector<AtomicCmpXchgInst*> cmpxchgs;

        for (BasicBlock& BB : F) {
            // Collect branch instructions (terminators)
            if (auto* BI = dyn_cast<BranchInst>(BB.getTerminator())) {
                if (BI->isConditional()) {
                    branches.push_back(BI);
                }
            }

            // Collect shared memory stores and atomics
            for (Instruction& I : BB) {
                if (auto* SI = dyn_cast<StoreInst>(&I)) {
                    if (SI->getPointerAddressSpace() == 3) {  // Shared memory
                        shmem_stores.push_back(SI);
                    }
                }
                if (auto* AI = dyn_cast<AtomicRMWInst>(&I)) {
                    atomics.push_back(AI);
                }
                if (auto* CI = dyn_cast<AtomicCmpXchgInst>(&I)) {
                    cmpxchgs.push_back(CI);
                }
            }
        }

        // Instrument collected instructions
        for (auto* BI : branches) {
            instrumentBranch(BI, siteTable, M);
        }
        for (auto* SI : shmem_stores) {
            instrumentSharedMemStore(SI, siteTable, M);
        }
        for (auto* AI : atomics) {
            instrumentAtomic(AI, siteTable, M);
        }
        for (auto* CI : cmpxchgs) {
            instrumentCmpXchg(CI, siteTable, M);
        }
    }

    errs() << "[gddbg] Instrumented " << siteTable.getSites().size() << " sites\n";

    // 6. Embed site table
    embedSiteTable(M, siteTable);

    return PreservedAnalyses::none();
}

} // namespace gddbg

// Plugin registration for new pass manager
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "GpuDiffDbgPass", LLVM_VERSION_STRING,
        [](PassBuilder& PB) {
            // Register pass to run automatically in optimization pipeline
            PB.registerOptimizerEarlyEPCallback(
                [](ModulePassManager& MPM, OptimizationLevel OL) {
                    MPM.addPass(gddbg::GpuDiffDbgPass());
                });

            // Also allow explicit invocation via -passes=gpu-diffdbg
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager& MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "gpu-diffdbg") {
                        MPM.addPass(gddbg::GpuDiffDbgPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
