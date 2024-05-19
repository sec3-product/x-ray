//
// Created by peiming on 11/7/19.
//

//#include <llvm/IR/CallSite.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/CommandLine.h>

#include "aser/PointerAnalysis/Models/MemoryModel/FieldSensitive/Layout/ArrayLayout.h"
#include "aser/PointerAnalysis/Models/MemoryModel/FieldInsensitive/FICanonicalizer.h"
#include "aser/PointerAnalysis/Models/MemoryModel/FieldSensitive/FSCanonicalizer.h"
#include "aser/Util/Log.h"
#include "aser/Util/Util.h"

using namespace aser;
using namespace llvm;

extern cl::opt<bool> USE_MEMLAYOUT_FILTERING;
extern cl::opt<bool> CONFIG_USE_FI_MODE;

static const Value *stripNullOrUnDef(const Value *V) {
    // a uni ptr
    if (isa<UndefValue>(V)) {
        V = UndefValue::get(llvm::Type::getInt8PtrTy(V->getContext()));
    }

    // a null ptr
    if (auto C = dyn_cast<Constant>(V)) {
        if (C->isNullValue()) {
            V = ConstantPointerNull::get(Type::getInt8PtrTy(V->getContext()));
        }
    }

    if (Operator::getOpcode(V) == Instruction::IntToPtr) {
        if (auto ptrToInt = llvm::dyn_cast<PtrToIntOperator>(cast<Operator>(V)->getOperand(0))) {
            //handle (inttoptr (ptrtoint %ptr)) pattern
            V = ptrToInt->getOperand(0);
        } else {
            // inttoptr creates universal ptr, change it to
            V = UndefValue::get(Type::getInt8PtrTy(V->getContext()));
        }
    }
    return V;
}

// modified from llvm::stripPointerCastsAndOffsets
const Value *aser::stripPointerCastsAndOffsets(const Value *V) {
    // Even though we don't look through PHI nodes, we could be called on an
    // instruction in an unreachable block, which may be on a cycle.
    SmallPtrSet<const Value *, 4> Visited;
    Visited.insert(V);
    do {
        if (auto *GEP = dyn_cast<GEPOperator>(V)) {
            // skip even if GEP is not in_bound
            V = GEP->getPointerOperand();
        } else if (Operator::getOpcode(V) == Instruction::BitCast ||
                   Operator::getOpcode(V) == Instruction::AddrSpaceCast) {
            V = cast<Operator>(V)->getOperand(0);
        } else if (auto *GA = dyn_cast<GlobalAlias>(V)) {
            V = GA->getAliasee();
        } else {
            if (auto CB = dyn_cast<CallBase>(V)) {
                if (const Value *RV = CB->getReturnedArgOperand()) {
                    // the argument is also the return pointer,
                    // this can increase both performance and accuarcy if it is
                    // ever used but it seems no one use it
                    V = RV;
                    continue;
                }
                if (CB->getIntrinsicID() == Intrinsic::launder_invariant_group ||
                    CB->getIntrinsicID() == Intrinsic::strip_invariant_group) {
                    V = CB->getArgOperand(0);
                    continue;
                }
            }
            return V;
        }
        assert(V->getType()->isPointerTy() && "Unexpected operand type!");
    } while (Visited.insert(V).second);

    return V;
}

const Value *FICanonicalizer::canonicalize(const Value *V) {
    if (!V->getType()->isPointerTy()) return V;
    V = stripPointerCastsAndOffsets(V);

    return stripNullOrUnDef(V);
}

static const Value *stripPointerCasts(const Value *V) {
    // Even though we don't look through PHI nodes, we could be called on an
    // instruction in an unreachable block, which may be on a cycle.
    SmallPtrSet<const Value *, 4> Visited;
    Visited.insert(V);
    do {
        if (auto *GEP = dyn_cast<GEPOperator>(V)) {
            return V; // do not collapse any GEP even if it has all-zero offset
        } else if (Operator::getOpcode(V) == Instruction::BitCast ||
                   Operator::getOpcode(V) == Instruction::AddrSpaceCast) {
            V = cast<Operator>(V)->getOperand(0);
        } else if (auto *GA = dyn_cast<GlobalAlias>(V)) {
            V = GA->getAliasee();
        } else {
            if (auto CS = dyn_cast<CallBase>(V)) {
                if (const Value *RV = CS->getReturnedArgOperand()) {
                    V = RV;
                    continue;
                }
                if (CS->getIntrinsicID() == Intrinsic::launder_invariant_group ||
                    CS->getIntrinsicID() == Intrinsic::strip_invariant_group) {
                    V = CS->getArgOperand(0);
                    continue;
                }
            }
            return V;
        }
        assert(V->getType()->isPointerTy() && "Unexpected operand type!");
    } while (Visited.insert(V).second);
    return V;
}

/// Strip off pointer casts, all-zero GEPs, aliases and invariant group
/// info.
const Value *FSCanonicalizer::canonicalize(const llvm::Value *V) {
    if (CONFIG_USE_FI_MODE) {
        // use the field-insensitive canonicalizer instead
        return FICanonicalizer::canonicalize(V);
    }

    if (!V->getType()->isPointerTy()) return V;
    bool changed = true;

    while (changed) {
        V = V->stripPointerCastsAndInvariantGroups();
        if (auto gep = dyn_cast<GetElementPtrInst>(V)) {
            if (gep->getNumIndices() == 1 && gep->hasAllConstantIndices()) {
                if (auto idx = cast<ConstantInt>(gep->getOperand(1))) {
                    if (idx->isNegative()) {
                        V = canonicalize(gep->getOperand(0));
                    }
                }
            }
        }
        // does not strip GEP even it is zero offset, it will enable type filtering on zero-offset gep as well
        // V = stripPointerCasts(
	
	// strip away global alias
	if (auto GV = dyn_cast<GlobalAlias>(V)) {
            V = GV->getAliasee();
        }
	
        auto tmp = stripNullOrUnDef(V);
        if (V != tmp) {
            V = tmp;
            changed = true;
        } else {
            changed = false;
        }
    }

    return V;
}

namespace aser {

// get the step size of the getelementptr (which uses variable index)
// return MAX_size_t when it can not be resolved
size_t getGEPStepSize(const GetElementPtrInst *GEP, const DataLayout &DL) {
    assert(!GEP->hasAllConstantIndices());
    // since we canonicalized the getelementptr instruction before, the
    // getelementptr that uses variable to index object can only two different forms
    assert(GEP->getNumOperands() == 2 || GEP->getNumOperands() == 3);

    for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP); GTI != GTE; GTI++) {
        // 1st, the first idx is zero, and the second idx is a variable
        // getelementptr [type], [type *] %obj, 0, %var
        if (isa<UndefValue>(GTI.getOperand())) {
            return std::numeric_limits<size_t>::max();
        } else if (isa<Constant>(GTI.getOperand())) {
            assert(dyn_cast<Constant>(GTI.getOperand())->isZeroValue());
            continue;
        }

        // 2nd, the first idx is variable
        // getelementptr [type], [type *] %obj, %var
        assert(!isa<Constant>(GTI.getOperand()));
        return DL.getTypeAllocSize(GTI.getIndexedType());
    }

    // Should we show source location of the unexpected instruction to user?
    LOG_ERROR("Encountered unexpected Instruction");
    LOG_DEBUG("Encountered unexpected GEP Instruction. inst={}", *GEP);
    llvm_unreachable("bad gep format");
}

bool isArrayExistAtOffset(const std::map<size_t, ArrayLayout *> &arrayMap, size_t pOffset, size_t elementSize) {
    if (arrayMap.empty()) {
        return false;
    }

    auto it = arrayMap.find(pOffset);
    if (it != arrayMap.end()) {
        ArrayLayout *arrLayout = it->second;
        if (arrLayout->getElementSize() == elementSize) {
            return true;
        } else if (arrLayout->getElementSize() < elementSize) {
            // for flang
            if (arrLayout->getElementSize() == 1) {
                return true;
            }
            return false;
        }

        // the current layout is larger than the element size,
        // the underlying layout might be nested inside and is an array at zero offset
        return isArrayExistAtOffset(arrLayout->getSubArrayMap(), 0, elementSize);
    } else {
        for (auto it : arrayMap) {
            size_t arrOffset = it.first;
            ArrayLayout *arrLayout = it.second;

            if (arrOffset < pOffset && arrOffset + arrLayout->getArraySize() >= elementSize) {
                // might be nested here
                return isArrayExistAtOffset(arrLayout->getSubArrayMap(), pOffset - arrOffset, elementSize);
            }
        }
    }

    return false;
}

}  // namespace aser
