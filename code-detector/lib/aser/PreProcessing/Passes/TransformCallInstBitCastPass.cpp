//
// Created by peiming on 8/13/20.
//

#include "aser/PreProcessing/Passes/TransformCallInstBitCastPass.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <thread>

using namespace llvm;

bool aser::TransformCallInstBitCastPass::runOnModule(llvm::Module &M) {
  //    return false;

  bool changed = false;
  IRBuilder<> builder(M.getContext());
  std::vector<Instruction *> removedInst;

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        // for every instruction in the Function, try
        if (auto call = dyn_cast<CallBase>(&I)) {
          if (auto CE = dyn_cast<ConstantExpr>(call->getCalledOperand())) {
            if (auto castCE = dyn_cast<BitCastOperator>(CE)) {
              // instead of typecasting the function type, cast the return type
              // and parameter type other wise it will be harder to analyze the
              // code fun_type1 bitcast fun_type_2
              std::vector<Value *> params;
              auto srcTy = dyn_cast<FunctionType>(
                  castCE->getSrcTy()->getPointerElementType());
              auto dstTy = cast<FunctionType>(
                  castCE->getDestTy()->getPointerElementType());
              if (srcTy != nullptr &&
                  srcTy->getNumParams() == dstTy->getNumParams()) {

                if (srcTy->getReturnType() != dstTy->getReturnType()) {
                  if (srcTy->getReturnType()->isStructTy() ||
                      dstTy->getReturnType()->isStructTy() ||
                      // !CastInst::isCastable(srcTy->getReturnType(),
                      // dstTy->getReturnType())) {
                      // For LLVM 12
                      !CastInst::isBitCastable(srcTy->getReturnType(),
                                               dstTy->getReturnType())) {
                    // TODO cast between two structure type is more complicated,
                    // do it later.
                    continue;
                  }
                }

                changed = true;
                builder.SetInsertPoint(&I);
                for (int i = 0; i < srcTy->getNumParams(); i++) {
                  auto arg = call->getArgOperand(i);
                  auto parTy = srcTy->getParamType(i);
                  if (arg->getType()->isIntegerTy() && parTy->isIntegerTy()) {
                    params.push_back(builder.CreateIntCast(arg, parTy, true));
                  } else {
                    params.push_back(
                        builder.CreateBitOrPointerCast(arg, parTy));
                  }
                }

                Value *newValue =
                    builder.CreateCall(srcTy, castCE->getOperand(0), params);
                if (newValue->getType() != I.getType()) {
                  newValue =
                      builder.CreateCast(CastInst::getCastOpcode(
                                             newValue, true, I.getType(), true),
                                         newValue, I.getType());
                  // newValue = builder.CreateBitOrPointerCast(newValue,
                  // I.getType());
                }

                if (auto invoke = dyn_cast<InvokeInst>(&I)) {
                  builder.CreateBr(invoke->getNormalDest());
                }

                I.replaceAllUsesWith(newValue);
                removedInst.push_back(&I);
              }
            }
          }
        }
      }
    }
  }

  for (auto I : removedInst) {
    I->eraseFromParent();
  }

  return changed;
}

char aser::TransformCallInstBitCastPass::ID = 0;
static RegisterPass<aser::TransformCallInstBitCastPass>
    TCIBCP("", "Eliminate bitcast constexpr used in callinst",
           false, /*CFG only*/
           false /*is analysis*/);
