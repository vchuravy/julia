// This file is a part of Julia. License is MIT: https://julialang.org/license

// This LLVM pass verifies invariants required for correct GC root placement.
// See the devdocs for a description of these invariants.

#include "llvm-version.h"

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/CallSite.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>

#include "codegen_shared.h"
#include "llvm-pass-helpers.h"
#include "julia.h"

#define DEBUG_TYPE "verify_jlir"
#undef DEBUG

using namespace llvm;

struct JLIRVerifier : public FunctionPass, public InstVisitor<JLIRVerifier>, private JuliaPassContext {
    static char ID;
    bool Broken = false;
    JLIRVerifier() : FunctionPass(ID) {}

private:
    void Check(bool Cond, const char *message, Value *Val) {
        if (!Cond) {
            dbgs() << message << "\n\t" << *Val << "\n";
            Broken = true;
        }
    }

public:
    void getAnalysisUsage(AnalysisUsage &AU) const override {
        FunctionPass::getAnalysisUsage(AU);
        AU.setPreservesAll();
    }

    bool doInitialization(Module &M) override;
    bool runOnFunction(Function &F) override;
    void visitCallInst(CallInst &CI);
};

void JLIRVerifier::visitCallInst(CallInst &CI) {
        auto callee = CI.getCalledValue();
        if (!callee)
            return;

        Check(!(callee == gc_preserve_begin_func || callee == gc_preserve_end_func),
              "Left over gc_preserve call", &CI);

        // Strip operand bundles
        for (unsigned I = 0, E = CI.getNumOperandBundles(); I != E; ++I) {
            auto bundle = CI.getOperandBundleAt(I);
            Check(bundle.getTagName() != "jl_roots", "jl_roots should no longer be an operand bundle", &CI);
        }
}

bool JLIRVerifier::doInitialization(Module &M) {
    // Initialize platform-agnostic references.
    initAll(M);
    return true;
}

bool JLIRVerifier::runOnFunction(Function &F) {
    visit(F);
    if (Broken) {
        abort();
    }
    return false;
}

char JLIRVerifier::ID = 0;
static RegisterPass<JLIRVerifier> X("JLIRVerifier", "Julia IR Output Verification Pass", false, false);

Pass *createJLIRVerifierPass() {
    return new JLIRVerifier();
}

extern "C" JL_DLLEXPORT void LLVMExtraAddJLIRVerifierPass(LLVMPassManagerRef PM)
{
    unwrap(PM)->add(createJLIRVerifierPass());
}
