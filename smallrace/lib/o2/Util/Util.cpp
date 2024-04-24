#include "o2/Util/Util.h"

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/Local.h>

using namespace llvm;
using namespace o2;

static bool isCompatibleFunctionType(const FunctionType *FT1, const FunctionType *FT2);

static const Type *stripArrayType(const Type *type) {
    while (auto AT = llvm::dyn_cast<llvm::ArrayType>(type)) {
        // strip array
        type = AT->getArrayElementType();
    }
    return type;
}

inline static bool isPtrToEmptyStruct(const llvm::Type *T) {
    // {}* ptr
    if (StructType *ST = T->isPointerTy() ? dyn_cast<StructType>(T->getPointerElementType()) : nullptr) {
        if (ST->getNumElements() == 0) {
            return true;
        }
    }

    return false;
}

inline static bool isVoidPtr(const llvm::Type *T) {
    return T == PointerType::get(IntegerType::getInt8Ty(T->getContext()), 0) ||
           isPtrToEmptyStruct(T);
}

static bool isTypeEqual(const llvm::Type *T1, const llvm::Type *T2, const DataLayout &DL) {
    assert(T1 && T2);

    if (T1 == T2) {
        return true;
    }

    // IMPORTANT: treat i8* as equal type to every type as it is the void * in C/C++
    // i8* is used in some essential data structures like vtables to hold the alias of virtual function
    if (T1->isPointerTy() && T2->isPointerTy() &&
        (isVoidPtr(T1) || isVoidPtr(T2))) {
        return true;
    }

    if ((T1->isArrayTy() && T1->getArrayElementType() == IntegerType::getInt8Ty(T1->getContext())) ||
        (T2->isArrayTy() && T2->getArrayElementType() == IntegerType::getInt8Ty(T2->getContext()))) {
        if (DL.getTypeAllocSize(const_cast<Type *>(T1)) == DL.getTypeAllocSize(const_cast<Type *>(T2))) {
            return true;
        }
    }

    // looser function type filtering
    if (T1->isFunctionTy() && T2->isFunctionTy()) {
        return isCompatibleFunctionType(cast<FunctionType>(T1), cast<FunctionType>(T2));
    }

    if (T1->isStructTy() && T2->isStructTy()) {
        auto ST1 = llvm::cast<StructType>(T1);
        auto ST2 = llvm::cast<StructType>(T2);
        if (ST1->hasName() && ST2->hasName()) {
            auto N1 = stripNumberPostFix(ST1->getName());
            auto N2 = stripNumberPostFix(ST2->getName());
            if (N1 == N2 && !N1.equals("struct.anon")) { // has meaningful same name
                return ST1->getNumElements() == ST2->getNumElements() &&
                       DL.getTypeAllocSize(const_cast<Type *>(T1)) == DL.getTypeAllocSize(const_cast<Type *>(T2));
            }
        } else {
            // has a anonmyous structure
            // make sure every element is the same
            if (ST1->getNumElements() == ST2->getNumElements()) {
                for (int i = 0; i < ST1->getNumElements(); i++) {
                    if (!isTypeEqual(ST1->getElementType(i), ST2->getElementType(i), DL)) {
                        return false;
                    }
                }
                // all the fields are equal
                return true;
            }
        }
    }

    return false;
}

bool o2::isZeroOffsetTypeInRootType(const Type *rootType, const Type *elemType, const DataLayout &DL) {
    if (rootType == nullptr || elemType == nullptr) {
        // JEFF: should return false for null??
        return true;
    }
    
    // quick path
    if (isTypeEqual(rootType, elemType, DL)) {
        return true;
    }

    // if this is a type with flexible array element
    if (auto ST = dyn_cast<StructType>(elemType)) {
        if (auto flexibleArrayTy = isStructWithFlexibleArray(ST)) {
            if (getConvertedFlexibleArrayType(ST, flexibleArrayTy) == rootType) {
                return true;
            }
        }
    }

    auto stripedRootType = stripArrayType(rootType);
    if (stripedRootType != rootType &&
        isZeroOffsetTypeInRootType(stripedRootType, elemType, DL)) {
        return true;
    }

    auto stripedElemType = stripArrayType(elemType);
    if (stripedElemType != elemType && stripedRootType != rootType) {
        // if both are array, then unless their inner most element type is identical
        // they should not contain each others. we do not consider array length and dimension
        // e.g., [10 * struct.A] not equal to [10 * struct.B]
        return isTypeEqual(stripedRootType, stripedElemType, DL);
    }

    while (rootType->isAggregateType()) {
        // else go one level deeper into the type tree;
        if (rootType->getNumContainedTypes() == 0) {
            // zero-sized type
            // struct = type {}
            return false;
        }
        rootType = rootType->getContainedType(0);

        if (isTypeEqual(rootType, elemType, DL)) {
            return true;
        }
    }

    return isTypeEqual(rootType, elemType, DL);
}


const Type* o2::getTypeAtOffset(const Type *ctype, size_t offset,
                                  const DataLayout &DL, bool stripArray) {
    if (offset == 0) {
        if (stripArray) {
            return stripArrayType(ctype);
        }
        return ctype;
    }

    Type *type = const_cast<Type *>(ctype);
    if (auto ST = dyn_cast<StructType>(type)) {
        const StructLayout *SL = DL.getStructLayout(ST);
        for (int i = 0; i < ST->getNumElements(); i++) {
            auto elemType = ST->getElementType(i);
            size_t elemOff = SL->getElementOffset(i);

            if (DL.getTypeAllocSize(elemType) == 0 && offset == elemOff) {
                // 0 sized object
                return elemType;
            }
            if (offset >= elemOff && offset < elemOff + DL.getTypeAllocSize(elemType)) {
                return getTypeAtOffset(elemType, offset-elemOff, DL, stripArray);
            }
        }
    } else if (auto AT = dyn_cast<ArrayType>(type)) {
        auto elemSize = DL.getTypeAllocSize(AT->getArrayElementType());
        return getTypeAtOffset(AT->getArrayElementType(), offset % elemSize, DL, stripArray);
    }

    // wrong offset
    return nullptr;
}

// TODO. can llvm names has more than one number post fix?
// e.g., fun.123.456
StringRef o2::stripNumberPostFix(StringRef originalName) {
    auto splited = originalName.rsplit(".");
    if (!splited.second.empty()) {
        int tmp;
        if (!splited.second.getAsInteger(10, tmp)) {
            // end with .number
            originalName = splited.first;
        }
    }
    return originalName;
}

void o2::prettyFunctionPrinter(const Function *func, raw_ostream &os) {
    os << *func->getReturnType() << " @" << func->getName() << "(";
    auto funcType = func->getFunctionType();
    for (unsigned I = 0, E = funcType->getNumParams(); I != E; ++I) {
        if (I) os << ", ";
        os << *funcType->getParamType(I);
    }
    if (funcType->isVarArg()) {
        if (funcType->getNumParams()) os << ", ";
        os << "...";  // Output varargs portion of signature!
    }
    os << ")";
}


std::string o2::getDemangledName(StringRef mangledName) {
    ItaniumPartialDemangler demangler;
    mangledName = stripNumberPostFix(mangledName);

    if (demangler.partialDemangle(mangledName.begin())) {
        return "";
    }

    return demangler.finishDemangle(nullptr, nullptr);
}

// i8* is the void* in LLVM
static bool isVoidPointer(const Type *T) {
    static Type *VoidStarTy = nullptr;
    if (VoidStarTy == nullptr) {
        VoidStarTy = PointerType::getUnqual(IntegerType::get(T->getContext(), 8));
    }

    return VoidStarTy == T;
}


static bool isCompatibleType(const Type *T1, const Type *T2) {
    // fast path
    if (T1 == T2) {
        return true;
    } else if (isa<PointerType>(T1) && isa<PointerType>(T2)) {
        // is it too conservative?
        if (isVoidPointer(T1) || isVoidPointer(T2)) {
            return true;
        }

        if (isa<FunctionType>(T1->getPointerElementType()) &&
            isa<FunctionType>(T2->getPointerElementType())) {
            return isCompatibleFunctionType(cast<FunctionType>(T1->getPointerElementType()),
                                            cast<FunctionType>(T2->getPointerElementType()));
        }

        // llvm IR does not have struct parameter, they are expanded into fields before passing to functions
        // return false;p
        auto ST1 = dyn_cast<StructType>(T1->getPointerElementType());
        auto ST2 = dyn_cast<StructType>(T2->getPointerElementType());
        // TODO: maybe better to use memory layout to compare two different types
        if (ST1 && ST2) {
            if (!ST1->isLiteral() && !ST2->isLiteral()) {
                // sometimes LLVM has multiple definition on the same type
                // e.g., struct.test and struct.test.123
                auto splitedNamePair1 = ST1->getName().rsplit(".");
                auto splitedNamePair2 = ST2->getName().rsplit(".");

                StringRef properName1 = ST1->getName();
                StringRef properName2 = ST2->getName();

                int numberPostFix;
                if (!splitedNamePair1.second.getAsInteger(10, numberPostFix)) {
                    // ST1 end with a number
                    properName1 = splitedNamePair1.first;
                }
                if (!splitedNamePair2.second.getAsInteger(10, numberPostFix)) {
                    // ST2 ends with a number
                    properName2 = splitedNamePair2.first;
                }

                return properName2.equals(properName1);
            }
        } else {
            while (true) {
                T1 = T1->getPointerElementType();
                T2 = T2->getPointerElementType();

                if (!isa<PointerType>(T1) || !isa<PointerType>(T2)) {
                    break;
                }
            }
            // whether one of the type is something like void * (which is i8* in LLVM)
            if (!isa<PointerType>(T1) && !isa<PointerType>(T2)) {
                if (T1->isIntegerTy()) {
                    if (T1->getIntegerBitWidth() == 8) {
                        // i8* also used for void*
                        return true;
                    }
                }

                if (T2->isIntegerTy()) {
                    if (T2->getIntegerBitWidth() == 8) {
                        // i8* also used for void*
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

static bool isCompatibleFunctionType(const FunctionType *FT1, const FunctionType *FT2) {
    // fast path, the same type
    if (FT1 == FT2) {
        return true;
    }

    if (FT1->getReturnType() != FT2->getReturnType()) {
        return false;
    }

    if ((FT1->getNumParams() != FT2->getNumParams()) &&
        !FT2->isVarArg()) {
        // two non-vararg function should at have same number of parameters
        return false;
    }

    if (FT2->isVarArg() && FT2->getNumParams() > FT1->getNumParams()) {
        // calling a varargs function, the callsite should offer at least the
        // same number of parameters required by var-args
        return false;
    }

    // LLVM IR is strongly typed, so ensure every actually argument is of the
    // same type as the formal arguments.
    auto param1 = FT1->param_begin();
    for (auto param2: FT2->params()) {
        if (!isCompatibleType(*param1, param2)) {
            return false;
        }
        param1++;
    }

    return true;
}

static std::map<std::string, size_t> NodeDistribution;
void o2::recordCGNode(const llvm::Value *val) {
    auto dir = getSourceDir(val);
    if (NodeDistribution.find(dir) == NodeDistribution.end()) {
        NodeDistribution.insert(std::make_pair(dir, 1));
    } else {
        NodeDistribution[dir]++;
    }

}

void o2::dumpCGNodeDistribution() {
    llvm::outs() << "*************************************\n";
    for (auto it = NodeDistribution.begin(), ie = NodeDistribution.end();
         it != ie; it ++) {
        llvm::outs() << it->first << ": " << it->second << "\n";
    }
    llvm::outs() << "*************************************\n";
}

std::string o2::getSourceDir(const Value *val) {
    assert(val != nullptr);

    if (auto inst = llvm::dyn_cast<Instruction>(val)) {
        if (MDNode *N = inst->getMetadata("dbg")) {
            auto *loc = llvm::cast<llvm::DILocation>(N);
            return loc->getDirectory().str();
        } else if (isa<AllocaInst>(inst)) {
            // TODO: there must be other insts than AllocaInst
            // that can be a shared variable
            for (DbgInfoIntrinsic *DII : FindDbgAddrUses(const_cast<Instruction *>(inst))) {
                if (auto DDI = llvm::dyn_cast<llvm::DbgDeclareInst>(DII)) {
                    auto DIVar = llvm::cast<llvm::DIVariable>(DDI->getVariable());
                    return DIVar->getDirectory().str();
                }
            }
        }
    } else if (auto gvar = llvm::dyn_cast<llvm::GlobalVariable>(val)) {
        // find the debuggin information for global variables
        llvm::NamedMDNode *CU_Nodes = gvar->getParent()->getNamedMetadata("llvm.dbg.cu");
        if (CU_Nodes) {
            for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
                auto CUNode = llvm::cast<llvm::DICompileUnit>(CU_Nodes->getOperand(i));
                for (llvm::DIGlobalVariableExpression *GV : CUNode->getGlobalVariables()) {
                    llvm::DIGlobalVariable *DGV = GV->getVariable();
                    if (DGV->getName() == gvar->getName() || DGV->getLinkageName() == gvar->getName()) {
                        return DGV->getDirectory().str();
                    }
                }
            }
        }
    }

    return "";
}

static bool isLooslyCompatibleCall(const llvm::CallBase *CS, const llvm::Function *target) {
    assert(CS->isIndirectCall());

    // fast path, the same type
    if (CS->getCalledOperand()->getType() == target->getType()) {
        return true;
    }

    if (!CS->getType()->isPointerTy() && !target->getReturnType()->isPointerTy() &&
        CS->getType() != target->getReturnType()) {
        return false;
    }

    if (CS->getType()->isPointerTy() != target->getReturnType()->isPointerTy()) {
        return false;
    }

    if (CS->getNumArgOperands() != target->arg_size() && !target->isVarArg()) {
        // two non-vararg function should at have same number of parameters
        return false;
    }

    if (target->isVarArg() && target->arg_size() > CS->getNumArgOperands()) {
        // calling a varargs function, the callsite should offer at least the
        // same number of parameters required by var-args
        return false;
    }

    // LLVM IR is strongly typed, so ensure every actually argument is of the
    // same type as the formal arguments.
    auto fit = CS->arg_begin();
    for (const Argument &arg : target->args()) {
        const Value *param = *fit;
        if (!param->getType()->isPointerTy() && !arg.getType()->isPointerTy() &&
            param->getType() != arg.getType()) {
            // for non pointer type, they should be matched
            return false;
        } else if (param->getType()->isPointerTy() != arg.getType()->isPointerTy()) {
            return false;
        }
        fit++;
    }

    return true;
}

llvm::Type *o2::isStructWithFlexibleArray(const llvm::StructType *ST) {
    // C99, only structure type can have flexible array elements
    size_t numElem = ST->getNumElements();
    if (numElem) {
        auto lastElem = ST->elements()[numElem - 1];
        if (auto AT = llvm::dyn_cast<llvm::ArrayType>(lastElem);
            AT && AT->getNumElements() == 0) {
            return AT->getElementType();
        }
    }
    return nullptr;
}

llvm::StructType *o2::getConvertedFlexibleArrayType(const llvm::StructType *ST, llvm::Type *flexibleArrayElem) {
    assert(isStructWithFlexibleArray(ST));

    // if this is a structure type with flexible array element at the end
    // treat it differently
    // For example
    // %struct.SUdpConnSet = type { i32, i32, i32, i16, i8*, i32, [8 x i8], i8* (%struct.SRecvInfo*)*, [0 x %struct.SUdpConn] }
    // TODO: is it safe to assume that if the last element of the structure is a zero-sized array,
    // then it is a structure type with flexible array element at the end?
    // TODO: there also might be a optional padding at the end of the structure as well
    SmallVector<Type *, 8> fieldTy;
    for (int i = 0; i < ST->getNumElements() - 1; i++) {
        fieldTy.push_back(ST->getStructElementType(i));
    }
    fieldTy.push_back(getUnboundedArrayTy(flexibleArrayElem));
    return StructType::get(ST->getContext(), fieldTy, ST->isPacked());
}

extern cl::opt<bool> CONFIG_EXHAUST_MODE;

// simple type check fails when cases like
// call void (...) %ptr()
bool o2::isCompatibleCall(const llvm::Instruction *indirectCall, const llvm::Function *target) {
    auto call = cast<CallBase>(indirectCall);

    return isLooslyCompatibleCall(call, target);

//    TODO: maybe add a configuration for this?
//    if (!target->getFunctionType()->isVarArg()) {
//        // the resolved target should have the same number of argument as provided by the callsite if it is not var_args
//        if (target->getFunctionType()->getNumParams() != call->getNumArgOperands()) {
//            return false;
//        }
//    } else {
//        // the callsite should at list provide the minimal number of parameters to var_args function
//        if (call->getNumArgOperands() < target->getFunctionType()->getNumParams()) {
//            return false;
//        }
//    }
//
//    return isCompatibleFunctionType(call->getFunctionType(), target->getFunctionType());
}


