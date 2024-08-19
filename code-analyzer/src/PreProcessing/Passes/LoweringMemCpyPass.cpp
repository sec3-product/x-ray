//
// Created by peiming on 1/22/20.
//

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Support/CommandLine.h>

#include <array>

#include "PreProcessing/Passes/LoweringMemCpyPass.h"
#include "Util/Log.h"

using namespace std;
using namespace aser;
using namespace llvm;

void LoweringMemCpyPass::lowerMemCpyForType(Type *type, Value *src, Value *dst,
                                            SmallVector<Value *, 5> &idx,
                                            IRBuilder<NoFolder> &builder) {
  switch (type->getTypeID()) {
  case llvm::Type::StructTyID: {
    auto structType = static_cast<const StructType *>(type);
    for (int i = 0; i < structType->getNumElements(); i++) {
      idx.push_back(ConstantInt::get(idxType, i));
      lowerMemCpyForType(structType->getElementType(i), src, dst, idx, builder);
      idx.pop_back();
    }
    break;
  }
  case llvm::Type::ArrayTyID: {
    auto arrayType = static_cast<const ArrayType *>(type);
    idx.push_back(ConstantInt::get(idxType, 0));
    lowerMemCpyForType(arrayType->getElementType(), src, dst, idx, builder);
    idx.pop_back();
    break;
  }
  case llvm::Type::PointerTyID: {
    auto srcGEP = builder.CreateGEP(nullptr, src, idx, "");
    auto dstGEP = builder.CreateGEP(nullptr, dst, idx, "");

    auto srcLoad =
        builder.CreateLoad(srcGEP->getType()->getPointerElementType(), srcGEP);
    builder.CreateStore(srcLoad, dstGEP, false);
    break;
  }
  case llvm::Type::FixedVectorTyID:
  case llvm::Type::ScalableVectorTyID: {
    // case llvm::Type::VectorTyID: {
    //  simply skip vector type
    LOG_TRACE("Unhandled Vector Type. type={}", static_cast<void *>(type));
    break;
  }
  default:
    // non-pointer scalar type, make no difference to pointer analysis.
    break;
  }
}

bool LoweringMemCpyPass::runOnModule(llvm::Module &M) {
  if (idxType == nullptr) {
    // use i32 to index getelementptr
    // TODO: does it matter to use i32 instead of i64?
    idxType = IntegerType::get(M.getContext(), 32);
  }

  bool changed = false;
  auto &DL = M.getDataLayout();
  IRBuilder<NoFolder> builder(M.getContext());

  array<StringRef, 2> MemCpys{"llvm.memcpy.p0i8.p0i8.i32",
                              "llvm.memcpy.p0i8.p0i8.i64"};
  for (StringRef MemCpyName : MemCpys) {
    Function *memcpy = M.getFunction(MemCpyName);
    if (memcpy != nullptr) {
      vector<Instruction *> instToRemove;
      for (auto user : memcpy->users()) {
        if (auto *callInst = dyn_cast<CallInst>(user)) {
          Value *dst = callInst->getArgOperand(0);
          Value *src = callInst->getArgOperand(1);
          Value *len = callInst->getArgOperand(2);

          auto constLen = dyn_cast<ConstantInt>(len);
          auto dstBitCast = dyn_cast<BitCastInst>(dst);
          auto srcBitCast = dyn_cast<BitCastInst>(src);

          if (constLen && dstBitCast && srcBitCast) {
            // we only lowering memcpy that uses
            // 1, constant length
            // 2, We can infer the type of dst and source, and theirs is the
            // same
            Type *dstType = dstBitCast->getSrcTy();
            Type *srcType = srcBitCast->getSrcTy();

            if (dstType == srcType) {
              Type *elemType = dstType->getPointerElementType();
              if (DL.getTypeAllocSize(elemType) == constLen->getSExtValue()) {
                changed = true;
                builder.SetInsertPoint(callInst);

                SmallVector<Value *, 5> idx;
                idx.push_back(ConstantInt::get(idxType, 0));

                lowerMemCpyForType(elemType, srcBitCast->getOperand(0),
                                   dstBitCast->getOperand(0), idx, builder);

                // do some simple cleanups
                // callInst->eraseFromParent();
                instToRemove.push_back(callInst);
                if (srcBitCast->getNumUses() == 0) {
                  // srcBitCast->eraseFromParent();
                  if (auto srcInst = llvm::dyn_cast<BitCastInst>(src)) {
                    instToRemove.push_back(srcInst);
                  }
                }
                if (dstBitCast->getNumUses() ==
                    0) { // src might be equal to dst
                  // dstBitCast->eraseFromParent();
                  if (auto dstInst = llvm::dyn_cast<BitCastInst>(dst)) {
                    if (dstInst->getParent() != nullptr) {
                      instToRemove.push_back(dstInst);
                    }
                  }
                }
              }
            }
          }
        }
      }

      for (auto inst : instToRemove) {
        inst->eraseFromParent();
      }

      instToRemove.clear();
    }
  }
  return changed;
}

char aser::LoweringMemCpyPass::ID = 0;
static RegisterPass<aser::LoweringMemCpyPass> LMCPY("", "Lowering MemCpy call",
                                                    false, /*CFG only*/
                                                    false /*is analysis*/);
