#include <Logger/Logger.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/Type.h>
#include <llvm/Transforms/Utils/Local.h>

#include "Util/Demangler.h"
#include "Util/TypeMetaData.h"
#include "Util/Util.h"

using namespace xray;
using namespace llvm;
using namespace std;

// anonymous namespace
namespace {

class DICompositeTypeCollector {
private:
  const Module *M = nullptr;
  std::vector<const DICompositeType *> typeDIVec;
  DenseMap<const llvm::Type *, const DICompositeType *> typeDIMap;

  void processFunctionMetadata(const Function &F,
                               DenseSet<const MDNode *> &mdnSet) {
    processGlobalObjectMetadata(F, mdnSet);
    for (auto &BB : F) {
      for (auto &I : BB)
        processInstructionMetadata(I, mdnSet);
    }
  }

  void processInstructionMetadata(const Instruction &I,
                                  DenseSet<const MDNode *> &mdnSet) {
    // Process metadata used directly by intrinsics.
    if (const CallInst *CI = dyn_cast<CallInst>(&I))
      if (Function *F = CI->getCalledFunction())
        if (F->isIntrinsic())
          for (auto &Op : I.operands())
            if (auto *V = dyn_cast_or_null<MetadataAsValue>(Op))
              if (MDNode *N = dyn_cast<MDNode>(V->getMetadata()))
                CreateMetadataSlot(N, mdnSet);

    // Process metadata attached to this instruction.
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    I.getAllMetadata(MDs);
    for (auto &MD : MDs)
      CreateMetadataSlot(MD.second, mdnSet);
  }

  /// CreateModuleSlot - Insert the specified MDNode* into the slot table.
  void CreateMetadataSlot(const MDNode *N, DenseSet<const MDNode *> &mdnSet) {
    assert(N && "Can't insert a null Value into SlotTracker!");

    // Don't make slots for DIExpressions. We just print them inline everywhere.
    if (isa<DIExpression>(N))
      return;

    if (!mdnSet.insert(N).second)
      return;

    if (auto DI = dyn_cast<DICompositeType>(N)) {
      if (DI->getSizeInBits() > 0 && !DI->getName().empty() &&
          (DI->getTag() == dwarf::DW_TAG_class_type ||
           DI->getTag() == dwarf::DW_TAG_structure_type)) {
        typeDIVec.push_back(DI);
      }
    }

    // Recursively add any MDNodes referenced by operands.
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      if (const MDNode *Op = dyn_cast_or_null<MDNode>(N->getOperand(i)))
        CreateMetadataSlot(Op, mdnSet);
  }

  void processGlobalObjectMetadata(const GlobalObject &GO,
                                   DenseSet<const MDNode *> &mdnSet) {
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    GO.getAllMetadata(MDs);
    for (auto &MD : MDs)
      CreateMetadataSlot(MD.second, mdnSet);
  }

  void processModule() {
    assert(M);

    DenseSet<const MDNode *> mdnSet;
    // Add all of the unnamed global variables to the value table.
    for (const GlobalVariable &Var : M->globals()) {
      processGlobalObjectMetadata(Var, mdnSet);
    }

    // Add metadata used by named metadata.
    for (const NamedMDNode &NMD : M->named_metadata()) {
      for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
        CreateMetadataSlot(NMD.getOperand(i), mdnSet);
    }

    for (const Function &F : *M) {
      processFunctionMetadata(F, mdnSet);
    }
  }

  void inline insertPair(const llvm::Type *T, const DICompositeType *DI) {
    typeDIMap.insert(make_pair(T, DI));
  }

  const inline DICompositeType *getIfExist(const llvm::Type *T) {
    // already cached
    if (auto it = typeDIMap.find(T); it != typeDIMap.end()) {
      return it->second;
    }
    return nullptr;
  }

  const DICompositeType *lookUpMDByName(const StringRef structName,
                                        StructType *T,
                                        bool useStructName = true) {
    // Structure type always looks like struct.XXX or class.XXX
    // TODO: is the observation always correct? it definitely only applicable to
    // C/C++
    auto tmp = structName.split(".");

    bool isClass = false;
    StringRef strippedName = structName;

    if (useStructName) {
      if (tmp.first.equals("struct")) {
        isClass = false;
      } else if (tmp.first.equals("class")) {
        isClass = true;
      } else {
        return nullptr;
      }
      strippedName = tmp.second;
      // usually the structure name looks as following
      // "class.std::ios_base::Init.xxx"
      tmp = strippedName.rsplit("::");
      if (!tmp.second.empty()) {
        // after this, stripped name = Init.xxx
        strippedName = tmp.second;
      }

      // after this, stripped name = Init
      strippedName = strippedName.rsplit(".").first;
    }

    auto DL = this->M->getDataLayout();
    auto SL = DL.getStructLayout(T);

    // look over all the DICompositeType
    for (auto DI : typeDIVec) {
      if (DI->getSizeInBits() == DL.getTypeAllocSizeInBits(T)) {
        // they are the same allocation site and has the same number of elements
        if (!useStructName ||
            ((isClass && DI->getTag() == dwarf::DW_TAG_class_type) ||
             (!isClass && DI->getTag() == dwarf::DW_TAG_structure_type))) {
          bool isNameEqual = false;
          if (useStructName) {
            isNameEqual = DI->getName().startswith(strippedName);
          } else {
            xray::Demangler demangler;
            if (!demangler.partialDemangle(DI->getIdentifier())) {
              if (demangler.isSpecialName()) { // is a special name; should be
                                               // typeinfo for XXX
                StringRef fullName = demangler.finishDemangle(nullptr, nullptr);
                if (fullName.endswith(structName)) {
                  isNameEqual = true;
                }
              }
            }
          }

          if (isNameEqual) {
            // a more expensive check
            // check each data member and their offset
            auto dataMembers = getNonStaticDataMember(DI);
            if (dataMembers.size() == T->getNumElements() ||
                (dataMembers.size() == T->getNumElements() - 1 &&
                 isEndedWithPadding(T))) {
              bool allSameOffset = true;
              for (int i = 0; i < dataMembers.size(); i++) {
                // the offset is the same
                if (dataMembers[i]->getOffsetInBits() !=
                    SL->getElementOffsetInBits(i)) {
                  allSameOffset = false;
                  break;
                }
              }
              if (allSameOffset) {
                // choose this one
                // cache the result and return
                typeDIMap.insert(make_pair(T, DI));
                return DI;
              }
            }
          }
        }
      }
    }
    return nullptr;
  }

public:
  static bool isEndedWithPadding(const StructType *T) {
    if (T->getNumElements() == 0) {
      return false;
    }

    // llvm generate [n * i8] for padding sometimes
    auto lastElemType = T->getElementType(T->getNumElements() - 1);
    if (auto AT = dyn_cast<ArrayType>(lastElemType)) {
      return AT->getArrayElementType()->isIntegerTy(8);
    }
    return false;
  }

  void collectMDIfNeeded(const Module *module) {
    if (this->M == module) {
      return;
    }
    // update the module
    this->M = module;

    // now collect all MD Nodes in the llvm::Module
    processModule();
  }

  inline const DataLayout &getDataLayout() { return this->M->getDataLayout(); }

  const DICompositeType *resolveTypeMetaData(const Value *allocSite,
                                             AllocKind T,
                                             const Type *allocType) {
    allocType = stripArray(allocType); // strip the array
    auto ST = dyn_cast<StructType>(allocType);
    if (ST == nullptr) {
      return nullptr;
    }

    //        if (auto cached = this->getIfExist(ST)) {
    //            return cached;
    //        }

    const DataLayout &DL = this->getDataLayout();

    switch (T) {
    case AllocKind::Stack: {
      for (auto user : FindDbgAddrUses(const_cast<Value *>(allocSite))) {
        // checkout the llvm.dbg.declare and retrieve the metadata;
        if (auto dbgDeclare = dyn_cast<DbgDeclareInst>(user)) {
          if (stripArray(dbgDeclare->getAddress()
                             ->getType()
                             ->getPointerElementType()) == ST) {
            auto objType =
                dyn_cast_or_null<DICompositeType>(xray::stripArrayAndTypeDefDI(
                    dbgDeclare->getVariable()->getType()));
            // alloca instruction create a pointer, we want the pointer base
            // type.
            if (objType) {
              if (objType->getFlags() & DINode::FlagFwdDecl) {
                this->insertPair(ST, nullptr);
                return nullptr;
              }
              // the object type should be the meta data we wanted
              // is it always true? seems to be a yes to me
              assert(DL.getTypeAllocSizeInBits(const_cast<StructType *>(ST)) ==
                     objType->getSizeInBits());

              // cached the type metadata mapping
              this->insertPair(ST, objType);
              return objType;
            }
          }

          // the dbg.declare does not match the supplied type
          break;
        }
      }
      // 2nd if failed, try to traverse all the metadata and find the right one
      return getTypeMetaData(ST);
    }

    case AllocKind::Globals: {
      // checkout dbg metadata along with the global variable

      // 2nd if failed
      return getTypeMetaData(ST);
    }
    case AllocKind::Heap: {
      // TODO:
      // 1st, checkout the llvm.dbg.value and retrieve the metadata

      // 2nd, try the to get the constructor followed by the allocation site,
      // which contains the full name for the type, compiler will generate the
      // following instructions.
      // TODO: handle _Znaw (new [] operator), which has a different pattern
      // %4 = tail call noalias i8* @_Znwm(i64 24) #19,
      // %5 = bitcast i8* %4 to %"struct.std::thread::_State_impl.7"*, !dbg
      // !2337 tail call void @CTOR(...) auto allocInst =
      // cast<CallBase>(allocSite);
      const BitCastInst *castInst = nullptr;
      const CallBase *ctorInst = nullptr;
      auto allocInst = cast<CallBase>(allocSite);

      if (auto callInst = dyn_cast<CallInst>(allocSite)) {
        castInst = dyn_cast_or_null<BitCastInst>(callInst->getNextNode());
        ctorInst = castInst != nullptr
                       ? dyn_cast<CallBase>(castInst->getNextNode())
                       : nullptr;
      } else if (auto invokeInst = dyn_cast<InvokeInst>(allocSite)) {
        castInst = dyn_cast_or_null<BitCastInst>(
            &*invokeInst->getNormalDest()->begin());
        ctorInst = castInst != nullptr
                       ? dyn_cast<CallBase>(castInst->getNextNode())
                       : nullptr;
      }

      if (castInst && ctorInst && ctorInst->getCalledFunction() &&
          castInst->getOperand(0) == allocInst &&
          ctorInst->getArgOperand(0) == castInst) {
        xray::Demangler demangler;
        if (!demangler.partialDemangle(
                ctorInst->getCalledFunction()->getName())) {
          if (demangler.isCtor()) {
            return lookUpMDByName(
                demangler.getFunctionDeclContextName(nullptr, nullptr),
                const_cast<StructType *>(ST), false);
          }
        }
      }
      // 3rd, if failed
      return getTypeMetaData(ST);
    }
    case AllocKind::Anonymous:
      return getTypeMetaData(ST);
    case AllocKind::Null:
    case AllocKind::Universal:
    case AllocKind::Functions:
      // not support
      return nullptr;
    }
    return nullptr;
  }

  const DICompositeType *lookUpMDForType(const StructType *ST) {
    assert(M && "Initialize collect");
    auto T = const_cast<StructType *>(ST);

    if (!T->hasName()) {
      return nullptr;
    }

    // already cached
    if (auto it = typeDIMap.find(T); it != typeDIMap.end()) {
      return it->second;
    }
    // T->getStructName() is not very precise as it does not contains the
    // template type information
    return lookUpMDByName(T->getStructName(), T);
  }
};

static DICompositeTypeCollector collector;

} // namespace

namespace xray {

DIType *stripTypeDefDI(DIType *DI) {
  while (DI->getTag() == dwarf::DW_TAG_typedef) {
    DI = cast<DIDerivedType>(DI)->getBaseType();
  }
  return DI;
}

DIType *stripArrayDI(DIType *DI) {
  while (DI->getTag() == dwarf::DW_TAG_array_type) {
    DI = cast<DICompositeType>(DI)->getBaseType();
  }

  return DI;
}

DIType *stripArrayAndTypeDefDI(DIType *DI) {
  while (DI->getTag() == dwarf::DW_TAG_array_type ||
         DI->getTag() == dwarf::DW_TAG_typedef) {

    if (DI->getTag() == dwarf::DW_TAG_array_type) {
      DI = cast<DICompositeType>(DI)->getBaseType();
    } else {
      DI = cast<DIDerivedType>(DI)->getBaseType();
    }
  }

  return DI;
}

void TypeMDinit(const llvm::Module *M) { collector.collectMDIfNeeded(M); }

// FIXME: is there any way that I can quickly get the type metadata with 100%
// accuracy???
const DICompositeType *getTypeMetaData(const StructType *T) {
  return collector.lookUpMDForType(T);
}

const DICompositeType *getTypeMetaData(const Module *M, const StructType *T) {
  collector.collectMDIfNeeded(M);
  return collector.lookUpMDForType(T);
}

const DICompositeType *getTypeMetaData(const Value *allocSite, AllocKind T,
                                       const Type *allocType) {
  return collector.resolveTypeMetaData(allocSite, T, allocType);
}

SmallVector<DIDerivedType *, 8>
getNonStaticDataMember(const DICompositeType *DI) {
  SmallVector<DIDerivedType *, 8> result;

  for (auto element : DI->getElements()) {
    if (auto derivedType = dyn_cast<DIDerivedType>(element)) {
      if (derivedType->getTag() == dwarf::DW_TAG_member) {
        if ((derivedType->getFlags() & llvm::DINode::FlagStaticMember) == 0) {
          // we do not count static members
          result.push_back(derivedType);
        }
      } else if (derivedType->getTag() == dwarf::DW_TAG_inheritance) {
        // TODO: how to handle virtual inheritance? I saw no use of virtual
        // inheritance in practice yet
        if ((derivedType->getFlags() & llvm::DINode::FlagVirtual) == 0) {
          result.push_back(derivedType);
        }
      }
    }
  }

  // sort based on the offset
  sort(result, [](DIDerivedType *lhs, DIDerivedType *rhs) -> bool {
    return lhs->getOffsetInBits() < rhs->getOffsetInBits();
  });

  return result;
}

// pOffset is in byte
void getFieldAccessPath(const DICompositeType *DI, size_t offsetInByte,
                        SmallVector<const DIType *, 8> &result) {
  auto members = getNonStaticDataMember(DI);
  size_t pOffset = offsetInByte * 8; // to bits
  result.push_back(DI);

  for (int i = 0; i < members.size(); i++) {
    auto member = members[i];
    size_t nextOffset;
    if (i + 1 < members.size()) {
      nextOffset = members[i + 1]->getOffsetInBits();
    } else {
      nextOffset = DI->getSizeInBits();
    }

    if (pOffset >= member->getOffsetInBits() && pOffset < nextOffset) {
      result.push_back(member);
      auto baseType = stripArrayAndTypeDefDI(member->getBaseType());

      if (auto compositeBase = dyn_cast<DICompositeType>(baseType)) {
        if (baseType->getTag() == dwarf::DW_TAG_structure_type ||
            baseType->getTag() == dwarf::DW_TAG_class_type) {
          // update the offset
          getFieldAccessPath(compositeBase, pOffset - member->getOffsetInBits(),
                             result);
        }
        // TODO: unhandled: DW_TAG_enumeration_type, DW_TAG_union_type
        return;
      } else {
        // TODO: inner type can be an array!!
        if (pOffset != member->getOffsetInBits()) {
          // will this happen??
          LOG_WARN("Find a InCompatible DICompositeType MD");
          result.clear();
          return;
        }
      }
      return;
    }
  }

  // pOffset > total size of the object
  // should be unreachable
  LOG_WARN("Find a InCompatible DICompositeType MD");
  result.clear();
}

std::size_t getDISize(llvm::DIDerivedType *T) {
  size_t size = T->getSizeInBits();
  if (size == 0 && T->getTag() == dwarf::DW_TAG_inheritance) {
    // parent class, the size can be retrieve from the parent type
    return stripTypeDefDI(getBaseType(T))->getSizeInBits();
  }
  return size;
}

} // namespace xray
