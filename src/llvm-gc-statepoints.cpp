// This file is a part of Julia. License is MIT: https://julialang.org/license

// Stackmap GC-root mode (--gc-roots=stackmap): the last step of the intrinsic
// lowering pipeline. LateLowerGCFrame has attached a `julia.gcroots` operand
// bundle to every safepoint call listing the tracked values that are live
// across it (and are not already kept in the memory frame). This pass turns
// each such call into an `llvm.experimental.gc.statepoint` with those values as
// deopt operands, so the backend records where they live at the call's return
// address in the `.llvm_stackmaps` section. The runtime (src/stackmaps.cpp)
// reads those records while unwinding a stopped thread or a suspended task.
//
// Julia's GC does not move objects, so no gc.relocate is needed and the
// "julia" GC strategy registered below requests neither RS4GC nor safepoint
// insertion.

#include "llvm-version.h"
#include "passes.h"

#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GCStrategy.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Statepoint.h>
#include <llvm/Support/Debug.h>

#include "julia.h"
#include "julia_assert.h"

#define DEBUG_TYPE "gc_statepoints"
STATISTIC(StatepointCount, "Number of safepoint calls converted to gc.statepoint");

using namespace llvm;

namespace {

class JuliaGCStrategy : public GCStrategy {
public:
    JuliaGCStrategy() {
        UseStatepoints = true;
        UseRS4GC = false;
        NeededSafePoints = false;
        UsesMetadata = false;
    }
    // By the time statepoints are emitted every pointer is in address space 0
    // and the roots are passed as deopt operands. Telling the backend that no
    // pointer is GC managed stops it from also treating them as (relocatable)
    // gc pointers, which would spill them to dedicated statepoint slots; deopt
    // operands may instead stay in callee-saved registers
    // (-use-registers-for-deopt-values).
    std::optional<bool> isGCManagedPointer(const Type *Ty) const override {
        (void)Ty;
        return false;
    }
};

GCRegistry::Add<JuliaGCStrategy> JuliaGCStrategyRegistration("julia", "Julia GC (stackmap roots)");

// Function attributes of the original call that remain meaningful on the
// statepoint wrapper. Memory effects and allocation attributes describe the
// wrapped callee, not the statepoint, and are dropped (as RS4GC does).
static bool keepFnAttrOnStatepoint(Attribute::AttrKind K)
{
    switch (K) {
    case Attribute::NoUnwind:
    case Attribute::NoReturn:
    case Attribute::WillReturn:
    case Attribute::Cold:
    case Attribute::NoInline:
    case Attribute::NoMerge:
    case Attribute::NoDuplicate:
        return true;
    default:
        return false;
    }
}

// Stackmap and statepoint frame-index operands must be addressed from the frame
// pointer (X86RegisterInfo::eliminateFrameIndex asserts "Expected the FP as
// base register"), but a function whose stack is dynamically realigned
// addresses its locals from the stack pointer. Realignment happens when any
// stack object needs more than the ABI stack alignment (an over-aligned alloca,
// or a spill slot for a wide vector register). Forbid it with
// "no-realign-stack" and clamp the IR alignment of over-aligned allocas, and of
// the accesses through them, to the stack alignment so the lowered code makes no
// alignment assumption the frame cannot honor.
static bool avoidStackRealignment(Function &F)
{
    const Align StackAlign(16); // x86-64 ABI stack alignment
    LLVMContext &ctx = F.getContext();
    bool Changed = false;
    if (!F.hasFnAttribute("no-realign-stack")) {
        F.addFnAttr("no-realign-stack");
        Changed = true;
    }
    SmallVector<AllocaInst *, 4> Over;
    for (Instruction &I : instructions(F)) {
        if (auto *AI = dyn_cast<AllocaInst>(&I))
            if (AI->getAlign() > StackAlign)
                Over.push_back(AI);
    }
    for (AllocaInst *AI : Over) {
        AI->setAlignment(StackAlign);
        Changed = true;
        SmallVector<Value *, 16> Worklist{AI};
        SmallPtrSet<Value *, 16> Visited{AI};
        while (!Worklist.empty()) {
            Value *V = Worklist.pop_back_val();
            for (User *U : V->users()) {
                if (auto *LI = dyn_cast<LoadInst>(U)) {
                    if (LI->getAlign() > StackAlign)
                        LI->setAlignment(StackAlign);
                }
                else if (auto *SI = dyn_cast<StoreInst>(U)) {
                    if (SI->getPointerOperand() == V && SI->getAlign() > StackAlign)
                        SI->setAlignment(StackAlign);
                }
                else if (auto *RMW = dyn_cast<AtomicRMWInst>(U)) {
                    if (RMW->getAlign() > StackAlign)
                        RMW->setAlignment(StackAlign);
                }
                else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(U)) {
                    if (CX->getAlign() > StackAlign)
                        CX->setAlignment(StackAlign);
                }
                else if (auto *MI = dyn_cast<MemIntrinsic>(U)) {
                    if (MI->getRawDest() == V && MI->getDestAlign() && *MI->getDestAlign() > StackAlign)
                        MI->setDestAlignment(StackAlign);
                    if (auto *MT = dyn_cast<MemTransferInst>(U))
                        if (MT->getRawSource() == V && MT->getSourceAlign() && *MT->getSourceAlign() > StackAlign)
                            MT->setSourceAlignment(StackAlign);
                }
                else if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U) || isa<AddrSpaceCastInst>(U) ||
                         isa<PHINode>(U) || isa<SelectInst>(U)) {
                    if (Visited.insert(U).second)
                        Worklist.push_back(U);
                }
                else if (auto *CB = dyn_cast<CallBase>(U)) {
                    for (unsigned i = 0; i < CB->arg_size(); i++) {
                        if (CB->getArgOperand(i) != V)
                            continue;
                        MaybeAlign A = CB->getParamAlign(i);
                        if (A && *A > StackAlign) {
                            CB->removeParamAttr(i, Attribute::Alignment);
                            CB->addParamAttr(i, Attribute::getWithAlignment(ctx, StackAlign));
                        }
                    }
                }
            }
        }
    }
    return Changed;
}

static bool emitGCStatepoints(Function &F)
{
    bool Changed = avoidStackRealignment(F);
    // Every function compiled in stackmap mode needs unwind tables: libunwind
    // walks through Julia frames and recovers their callee-saved registers
    // from CFI. (Codegen only requests them on Windows otherwise.)
    if (F.getUWTableKind() != UWTableKind::Async) {
        F.setUWTableKind(UWTableKind::Async);
        Changed = true;
    }

    SmallVector<CallInst *, 0> Calls;
    for (Instruction &I : instructions(F)) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
            if (CI->getOperandBundle("julia.gcroots"))
                Calls.push_back(CI);
        }
    }
    if (Calls.empty())
        return Changed;

    F.setGC("julia");
    Module *M = F.getParent();
    LLVMContext &ctx = F.getContext();
    Type *T_int32 = Type::getInt32Ty(ctx);
    Type *T_int64 = Type::getInt64Ty(ctx);
    uint64_t ID = 1; // 0 is the whole-frame record emitted by FinalLowerGC

    for (CallInst *CI : Calls) {
        assert(!CI->canReturnTwice() && "returns_twice calls must not carry julia.gcroots");
        assert(CI->getNumOperandBundles() == 1 && "unexpected extra operand bundle on safepoint call");
        auto Bundle = CI->getOperandBundle("julia.gcroots");
        SmallVector<Value *, 8> Deopt(Bundle->Inputs.begin(), Bundle->Inputs.end());

        FunctionType *FTy = CI->getFunctionType();
        Value *Callee = CI->getCalledOperand();
        SmallVector<Value *, 16> Args;
        Args.push_back(ConstantInt::get(T_int64, ID++));
        Args.push_back(ConstantInt::get(T_int32, 0)); // num patch bytes
        Args.push_back(Callee);
        Args.push_back(ConstantInt::get(T_int32, CI->arg_size()));
        Args.push_back(ConstantInt::get(T_int32, (uint64_t)StatepointFlags::None));
        Args.append(CI->arg_begin(), CI->arg_end());
        // deprecated inline transition/deopt argument counts (must be 0; the
        // deopt operands travel in the bundle)
        Args.push_back(ConstantInt::get(T_int32, 0));
        Args.push_back(ConstantInt::get(T_int32, 0));

        SmallVector<OperandBundleDef, 1> Bundles;
        Bundles.emplace_back("deopt", Deopt);

        Function *SPDecl = Intrinsic::getOrInsertDeclaration(
            M, Intrinsic::experimental_gc_statepoint, {Callee->getType()});
        CallInst *SP = CallInst::Create(SPDecl, Args, Bundles, "", CI->getIterator());
        SP->setCallingConv(CI->getCallingConv());
        SP->setDebugLoc(CI->getDebugLoc());

        AttributeList AL = CI->getAttributes();
        AttributeList SPAL = SP->getAttributes();
        AttrBuilder FnB(ctx);
        for (Attribute A : AL.getFnAttrs()) {
            if (A.isEnumAttribute() && keepFnAttrOnStatepoint(A.getKindAsEnum()))
                FnB.addAttribute(A);
        }
        SPAL = SPAL.addFnAttributes(ctx, FnB);
        // The verifier requires the callee operand to carry its function type.
        SPAL = SPAL.addParamAttribute(ctx, GCStatepointInst::CalledFunctionPos,
                                      Attribute::get(ctx, Attribute::ElementType, FTy));
        for (unsigned i = 0; i < CI->arg_size(); ++i) {
            AttributeSet PA = AL.getParamAttrs(i);
            if (PA.hasAttributes())
                SPAL = SPAL.addParamAttributes(ctx, GCStatepointInst::CallArgsBeginPos + i, AttrBuilder(ctx, PA));
        }
        SP->setAttributes(SPAL);

        if (!CI->getType()->isVoidTy()) {
            Function *ResDecl = Intrinsic::getOrInsertDeclaration(
                M, Intrinsic::experimental_gc_result, {CI->getType()});
            CallInst *Res = CallInst::Create(ResDecl, {SP}, "", CI->getIterator());
            Res->setDebugLoc(CI->getDebugLoc());
            AttributeSet RA = AL.getRetAttrs();
            if (RA.hasAttributes())
                Res->setAttributes(AttributeList::get(ctx, AttributeList::ReturnIndex, AttrBuilder(ctx, RA)));
            Res->takeName(CI);
            CI->replaceAllUsesWith(Res);
        }
        CI->eraseFromParent();
        ++StatepointCount;
        Changed = true;
    }
    return Changed;
}

} // anonymous namespace

PreservedAnalyses EmitGCStatepointsPass::run(Function &F, FunctionAnalysisManager &AM)
{
    if (emitGCStatepoints(F)) {
#ifdef JL_VERIFY_PASSES
        assert(!verifyLLVMIR(F));
#endif
        return PreservedAnalyses::none();
    }
    return PreservedAnalyses::all();
}
