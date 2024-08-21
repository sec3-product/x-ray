// the basic framework for andersen-based algorithm, including common routines
// override neccessary ones, and the call will be STATICALLY redirected to it
#ifndef ASER_PTA_SOLVERBASE_H
#define ASER_PTA_SOLVERBASE_H

#define DEBUG_TYPE "pta"

#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

//#include "RDUtil.h"
#include "PointerAnalysis/Graph/CallGraph.h"
#include "PointerAnalysis/Graph/ConstraintGraph/ConstraintGraph.h"
#include "PointerAnalysis/Models/MemoryModel/MemModelTrait.h"
#include "PointerAnalysis/Solver/PointsTo/BitVectorPTS.h"
#include "Util/Log.h"
#include "Util/Statistics.h"

extern llvm::cl::opt<bool> ConfigPrintConstraintGraph;
extern llvm::cl::opt<bool> ConfigPrintCallGraph;
extern llvm::cl::opt<bool> ConfigDumpPointsToSet;

namespace xray {

template <typename ctx> class CallGraph;

template <typename LangModel, typename SubClass> class SolverBase {
private:
  struct Noop {
    template <typename... Args>
    __attribute__((always_inline)) void operator()(Args &&...) {}
  };
  std::unique_ptr<LangModel> langModel;

  LOCAL_STATISTIC(ProcessedCopy, "Number of Processed Copy Edges");
  LOCAL_STATISTIC(ProcessedLoad, "Number of Processed Load Edges");
  LOCAL_STATISTIC(ProcessedStore, "Number of Processed Store Edges");
  LOCAL_STATISTIC(ProcessedOffset, "Number of Processed Offset Edges");

  LOCAL_STATISTIC(EffectiveCopy, "Number of Effective Copy Edges");
  LOCAL_STATISTIC(EffectiveLoad, "Number of Effective Load Edges");
  LOCAL_STATISTIC(EffectiveStore, "Number of Effective Store Edges");
  LOCAL_STATISTIC(EffectiveOffset, "Number of Effective Offset Edges");

public:
  using LMT = LangModelTrait<LangModel>;
  using MemModel = typename LMT::MemModelTy;
  using MMT = MemModelTrait<MemModel>;
  using ctx = typename LangModelTrait<LangModel>::CtxTy;
  using CT = CtxTrait<ctx>;
  using ObjTy = typename MMT::ObjectTy;

protected:
  using PT = PTSTrait<typename LMT::PointsToTy>;
  using PtsTy = typename PT::PtsTy;
  using CallGraphTy = CallGraph<ctx>;
  using CallNodeTy = typename CallGraphTy::NodeType;
  using ConsGraphTy = ConstraintGraph<ctx>;
  using CGNodeTy = CGNodeBase<ctx>;
  using PtrNodeTy = CGPtrNode<ctx>;
  using ObjNodeTy = CGObjNode<ctx, ObjTy>;

  ConsGraphTy *consGraph;
  llvm::SparseBitVector<> updatedFunPtrs;

  // TODO: the intersection on pts should be done through PtsTrait for better
  // extensibility
  llvm::DenseMap<PtrNodeTy *, PtsTy> handledGEPMap;

  inline void updateFunPtr(NodeID indirectNode) {
    updatedFunPtrs.set(indirectNode);
  }

  inline bool resolveFunPtrs() {
    if (updatedFunPtrs.empty()) {
      return false;
    }

    bool reanalyze = LMT::updateFunPtrs(langModel.get(), updatedFunPtrs);
    updatedFunPtrs.clear();

    return reanalyze;
  }

  // some helper function that might be needed by subclasses
  constexpr inline bool processAddrOf(CGNodeTy *src, CGNodeTy *dst) const;
  inline bool processCopy(CGNodeTy *src, CGNodeTy *dst);

  // TODO: only process diff pts
  template <typename CallBack = Noop>

  inline bool processOffset(CGNodeTy *src, CGNodeTy *dst,
                            CallBack callBack = Noop{}) {
    if (src == dst) {
      // due to circle merge
      return false;
    }

    assert(!src->hasSuperNode() && !dst->hasSuperNode());
    ProcessedOffset++;

    // TODO: use llvm::cast in debugging build
    // gep for sure create a pointer node
    auto ptrNode = static_cast<CGPtrNode<ctx> *>(dst);
    // assert(ptrNode);

    // we must be handling a getelemntptr instruction if we are indexing a
    // object
    auto idx =
        llvm::cast<const llvm::Instruction>(ptrNode->getPointer()->getValue());
    // assert(gep);

    // TODO: the intersection on pts should be done through PtsTrait for better
    // extensibility
    PtsTy &handled = handledGEPMap.try_emplace(ptrNode).first->second;
    const PtsTy &curPts = PT::getPointsTo(src->getNodeID());

    PtsTy newGEPs;
    newGEPs.intersectWithComplement(curPts, handled);

    assert(!handled.intersects(newGEPs));
    assert(curPts.contains(newGEPs));
    handled |= newGEPs; // update handled gep

    bool changed = false;
    std::vector<ObjNodeTy *> nodeVec;
    size_t ptsSize = newGEPs.count();
    if (ptsSize == 0) {
      return false;
    }

    // there is new obj in src's pts
    nodeVec.reserve(ptsSize);
    // We need to cache all the node here because the PT might be modified and
    // the iterator might be invalid
    for (auto it = newGEPs.begin(), ie = newGEPs.end(); it != ie; ++it) {
      // TODO: use llvm::cast in debugging build
      auto objNode = static_cast<ObjNodeTy *>(consGraph->getObjectNode(*it));
      nodeVec.push_back(objNode);
    }

    // update the cached pts
    for (auto objNode : nodeVec) {
      // this might create new object, thus modify the points-to set
      auto *fieldObj = llvm::cast_or_null<ObjNodeTy>(
          LMT::indexObject(this->getLangModel(), objNode, idx));
      if (fieldObj == nullptr) {
        continue;
      }
      if (DEBUG_PTA)
        llvm::outs() << "processOffset: "
                     << "src: " << src->getNodeID()
                     << " fieldObj: " << fieldObj->getNodeID()
                     << " dst: " << dst->getNodeID() << "\n";

      if (!PT::has(ptrNode->getNodeID(), fieldObj->getObjectID())) {
#ifndef NO_ADDR_OF_FOR_OFFSET
        // insert an addr_of constraint if ptrNode does not points to field
        // object previous this is the major source for newly inserted
        // constraints remove this but relying on solver to handle it correctly
        // can improve both performance and memory effciency
        // but the visualization of the constraint graph will be affected.
        this->consGraph->addConstraints(fieldObj, ptrNode,
                                        Constraints::addr_of);
#endif

        // TODO: stop calling callback if the obj offset is non-pointer
        // auto ptrElemType =
        // ptrNode->getPointer()->getValue()->getType()->getPointerElementType();
        // auto offsetType =
        // fieldObj->getObject()->getOffsetType(idx->getModule()->getDataLayout());

        // auto offsetType =
        // fieldObj->getObject()->getOffsetType()->getPointerElementType();
        // offset is calculate by the size? offset / 4
        // auto offsetType = objElemType->getContainedType(loffset);  // JEFF
        // not sure (34860 is there a bug?)

        //                if (nodeVec.size() > 10) {
        //                    auto printOffsetType = offsetType;
        //                    if (offsetType == nullptr) {
        //                        printOffsetType = ptrElemType;
        //                        // for scalar object and the field-insensitive
        //                        object, the type can be null; llvm::outs() <<
        //                        "offsetType: null\n";
        //                    }
        //
        //                    llvm::outs() << "\nptrElemType: " << *ptrElemType
        //                    << "\nobjElemType: " << *objElemType << "\n"
        //                                 << "    poffset: " << poffset << "
        //                                 loffset: " << loffset
        //                                 << "  offsetType: " <<
        //                                 *printOffsetType << "\n";
        //
        //                    llvm::outs() << "PTA offset update size: " <<
        //                    nodeVec.size()
        //                                 << " ptr: " << ptrNode->toString()
        //                                 //<< " value: " << *ptr->getValue()
        //                                 << "\nfieldObj: " <<
        //                                 fieldObj->toString()  //<< " value: "
        //                                 << *fieldObj->getValue()
        //                                 << "\n"; // JEFF
        //                }
        // i8 is a special pointer?
        // if (ptrElemType->isIntegerTy(8) || ptrElemType->isPointerTy() ||
        // ptrElemType->isStructTy() ||
        //     ptrElemType->isFunctionTy())  // JEFF
        // {
        //                    if (nodeVec.size() > 10000)
        //                        llvm::outs() << "Large PTA offset update size:
        //                        " << nodeVec.size()
        //                                     << " ptr: " <<
        //                                     ptrNode->toString()         //<<
        //                                     " value: " << *ptr->getValue()
        //                                     << "\nfieldObj: " <<
        //                                     fieldObj->toString()  //<< "
        //                                     value: " << *fieldObj->getValue()
        //                                     << "\n"; // JEFF

        // JEFF TODO: get the type of fieldObj->offet
        // Peiming: We do not need to do type filtering here, because the type
        // filtering has already been conducted before indexing the object,
        // incompatible object will be filter out already
        //                    if (nodeVec.size() == 1 ||
        //                    isZeroOffsetTypeInRootType(offsetType,
        //                                                                          ptrElemType,
        //                                                                          idx->getModule()->getDataLayout())) {
        callBack(fieldObj, ptrNode);
        changed = true;
        //                    }
        //}
      }
    }

    //        // JEFF debug
    //        if (!changed) {
    //            llvm::outs() << "NO-MATCHED-OFFSET-TYPE: debugging (size=" <<
    //            nodeVec.size() << ")...\n";
    //            // something is wrong
    //            for (auto objNode : nodeVec) {
    //                // this might create new object, thus modify the points-to
    //                set auto *fieldObj =
    //                llvm::cast_or_null<ObjNodeTy>(LMT::indexObject(this->getLangModel(),
    //                objNode, idx)); if (fieldObj == nullptr) {
    //                    continue;
    //                }
    //                auto ptrElemType =
    //                ptrNode->getPointer()->getValue()->getType()->getPointerElementType();
    //                auto objElemType =
    //                fieldObj->getObject()->getValue()->getType()->getPointerElementType();
    //                auto poffset = fieldObj->getObject()->getPOffset();
    //                auto loffset = fieldObj->getObject()->getLOffset();
    //
    //                auto offsetType =
    //                fieldObj->getObject()->getOffsetType(idx->getModule()->getDataLayout());
    //                {
    //                    auto printOffsetType = offsetType;
    //                    if (offsetType == nullptr) {
    //                        printOffsetType = ptrElemType;
    //                        // for scalar object and the field-insensitive
    //                        object, the type can be null; llvm::outs() <<
    //                        "offsetType: null\n";
    //                    }
    //
    //                    llvm::outs() << "\nptrElemType: " << *ptrElemType <<
    //                    "\nobjElemType: " << *objElemType << "\n"
    //                                 << "    poffset: " << poffset << "
    //                                 loffset: " << loffset
    //                                 << "  offsetType: " << *printOffsetType
    //                                 << "\n";
    //
    //                    llvm::outs() << "PTA offset update size: " <<
    //                    nodeVec.size()
    //                                 << " ptr: " << ptrNode->toString() //<< "
    //                                 value: " << *ptr->getValue()
    //                                 << "\nfieldObj: " << fieldObj->toString()
    //                                 //<< " value: " << *fieldObj->getValue()
    //                                 << "\n"; // JEFF
    //                }
    //                break;  // print the first one only
    //            }
    //        }

    if (changed) {
      EffectiveOffset++;
    }
    return changed;
  }

  // TODO: only process diff pts
  // src --LOAD-->dst
  // for every node in pts(src):
  //     node --COPY--> dst
  template <typename CallBack = Noop>
  bool processLoad(CGNodeTy *src, CGNodeTy *dst, CallBack callBack = Noop{}) {
    assert(!src->hasSuperNode() && !dst->hasSuperNode());
    ProcessedLoad++;

    bool changed = false;
    for (auto it = PT::begin(src->getNodeID()), ie = PT::end(src->getNodeID());
         it != ie; it++) {
      auto node = consGraph->getObjectNode(*it);
      node = node->getSuperNode();
      if (consGraph->addConstraints(node, dst, Constraints::copy)) {
        changed = true;
        callBack(node, dst);
      }
    }

    if (changed) {
      EffectiveLoad++;
    }
    return changed;
  }

  template <typename CallBack = Noop>
  bool processSpecial(CGNodeTy *src, CGNodeTy *dst,
                      CallBack callBack = Noop{}) {
    // TODO: we should do a cache here as well! as handling special constraints
    // are expensive
    assert(!src->hasSuperNode() && !dst->hasSuperNode());

    struct OnNewConstraints : public ConsGraphTy::OnNewConstraintCallBack {
      CallBack &CB;
      explicit OnNewConstraints(CallBack &CB) : CB(CB) {}

      void onNewConstraint(CGNodeTy *src, CGNodeTy *dst,
                           Constraints constraint) override {
        assert(constraint == Constraints::copy &&
               "special constraints can only add new copy/addr_of constraints");
        CB(src, dst);
      }
    };

    OnNewConstraints cb(callBack);
    bool changed = false;
    this->consGraph->registerCallBack(&cb);
    for (auto it = PT::begin(src->getNodeID()), ie = PT::end(src->getNodeID());
         it != ie; it++) {
      auto node = llvm::cast<ObjNodeTy>(consGraph->getObjectNode(*it));
      changed = node->getObject()->processSpecial(src, dst);
    }
    this->consGraph->unregisterCallBack();
    return changed;
  }

  // TODO: only process diff pts
  // src --STORE-->dst
  // for every node in pts(dst):
  //      src --COPY--> node
  template <typename CallBack = Noop>
  bool processStore(CGNodeTy *src, CGNodeTy *dst, CallBack callBack = Noop{}) {
    assert(!src->hasSuperNode() && !dst->hasSuperNode());

    ProcessedStore++;
    bool changed = false;
    for (auto it = PT::begin(dst->getNodeID()), ie = PT::end(dst->getNodeID());
         it != ie; it++) {
      auto tmp = llvm::dyn_cast<ObjNodeTy>(consGraph->getCGNode(*it));
      auto node = consGraph->getObjectNode(*it);
      node = node->getSuperNode();

      if (consGraph->addConstraints(src, node, Constraints::copy)) {
        changed = true;
        callBack(src, node);
      }
    }

    if (changed) {
      EffectiveStore++;
    }
    return changed;
  }

  void solve() {
    bool reanalyze;
    // from here
    do {
      static_cast<SubClass *>(this)->runSolver(*langModel);
      // resolve indirect calls in language model
      reanalyze = resolveFunPtrs();
    } while (reanalyze);
  }

  [[nodiscard]] inline LangModel *getLangModel() const {
    return this->langModel.get();
  }

  void dumpPointsTo() {
    std::error_code ErrInfo;
    llvm::ToolOutputFile F("PTS.txt", ErrInfo, llvm::sys::fs::OF_None);
    if (!ErrInfo) {
      // dump the points to set

      // 1st, dump the Object Node Information
      for (auto it = this->getConsGraph()->begin(),
                ie = this->getConsGraph()->end();
           it != ie; it++) {
        CGNodeTy *node = *it;
        if (llvm::isa<ObjNodeTy>(node)) {
          // dump the information
          F.os() << "Object " << llvm::cast<ObjNodeTy>(node)->getObjectID()
                 << " : \n";
          F.os() << node->toString() << "\n";
        }
      }

      // 2nd, dump the points to set of every node
      for (auto it = this->getConsGraph()->begin(),
                ie = this->getConsGraph()->end();
           it != ie; it++) {
        CGNodeTy *node = *it;

        F.os() << node->toString() << " : ";
        F.os() << "{";
        bool isFirst = true;

        for (auto it = PT::begin(node->getSuperNode()->getNodeID()),
                  ie = PT::end(node->getSuperNode()->getNodeID());
             it != ie; it++) {
          if (isFirst) {
            F.os() << *it;
            isFirst = false;
          } else {
            F.os() << " ," << *it;
          }
        }
        F.os() << "}\n\n\n";
      }

      if (!F.os().has_error()) {
        F.os().flush();
        F.keep();
        return;
      }
    }
  }

public:
  // analyze the give module with specified entry function
  bool analyze(llvm::Module *module, llvm::StringRef entry) {
    assert(langModel == nullptr && "can not run pointer analysis twice");
    // ensure the points to set are cleaned.
    // TODO: support different point-to set instance for different PTA instance
    // new they all share a global vector to store it.
    PT::clearAll();

    // using language model to construct language model
    langModel.reset(LMT::buildInitModel(module, entry));
    LMT::constructConsGraph(langModel.get());

    consGraph = LMT::getConsGraph(langModel.get());

    //        if (ConfigPrintConstraintGraph) {
    //            WriteGraphToFile("ConstraintGraph_Initial",
    //            *this->getConsGraph());
    //        }

    LOG_INFO("Pointer Analysis Starting to Solve");

    // subclass might override solve() directly for more aggressive overriding
    static_cast<SubClass *>(this)->solve();

    LOG_INFO("Pointer Analysis Finished Solving");

    LOG_DEBUG("PTA constraint graph node number {}, callgraph node number {}",
              this->getConsGraph()->getNodeNum(),
              this->getCallGraph()->getNodeNum());

    if (ConfigPrintConstraintGraph) {
      WriteGraphToFile("ConstraintGraph_Final", *this->getConsGraph());
    }
    if (ConfigPrintCallGraph) {
      WriteGraphToFile("CallGraph_Final", *this->getCallGraph());
    }
    if (ConfigDumpPointsToSet) {
      // dump the points to set of every pointers
      dumpPointsTo();
    }

    return false;
  }

  CGNodeTy *getCGNode(const ctx *context, const llvm::Value *V) const {
    NodeID id = LMT::getSuperNodeIDForValue(langModel.get(), context, V);
    return (*consGraph)[id];
  }

  // FIXME: this should not be implemented in PTA
  // `lockStr` - a string representing the lock variable name
  // a mapping between lock variable name (whose pts is empty) and the speical
  // anon obj
  std::map<std::string, ObjNodeTy *> lockStrObjects;
  void getPointsToForSpecialLockPtr(const ctx *context,
                                    const llvm::Instruction *I,
                                    std::string lockStr,
                                    const llvm::Value *lockPtr,
                                    std::vector<const ObjTy *> &result) {
    // create annonymous object if it does not exist
    if (lockStrObjects.find(lockStr) == lockStrObjects.end()) {
      auto objNode = LMT::allocSpecialAnonObj(langModel.get(), I, lockPtr);
      lockStrObjects[lockStr] = objNode;
    }
    result.push_back(lockStrObjects.at(lockStr)->getObject());
  }

  void getPointsTo(const ctx *context, const llvm::Value *V,
                   std::vector<const ObjTy *> &result) const {
    assert(V->getType()->isPointerTy());

    // get the node value
    NodeID node = LMT::getSuperNodeIDForValue(langModel.get(), context, V);
    if (node == INVALID_NODE_ID) {
      return;
    }

    for (auto it = PT::begin(node), ie = PT::end(node); it != ie; it++) {
      auto objNode = llvm::dyn_cast<ObjNodeTy>(consGraph->getObjectNode(*it));
      assert(objNode);
      if (objNode->isSpecialNode()) {
        continue;
      }
      result.push_back(objNode->getObject());
    }
  }

  void getFSPointsTo(const ctx *context, const llvm::Value *V,
                     std::vector<const ObjTy *> &result) const {
    assert(V->getType()->isPointerTy());

    // get the node value
    NodeID node = LMT::getSuperNodeIDForValue(langModel.get(), context, V);
    if (node == INVALID_NODE_ID) {
      return;
    }

    for (auto it = PT::begin(node), ie = PT::end(node); it != ie; it++) {
      auto objNode = llvm::dyn_cast<ObjNodeTy>(consGraph->getObjectNode(*it));
      assert(objNode);
      if (objNode->isSpecialNode() || objNode->isFIObject()) {
        continue;
      }
      result.push_back(objNode->getObject());
    }
  }

  const llvm::Type *getPointedType(const ctx *context,
                                   const llvm::Value *V) const {
    std::vector<const ObjTy *> result;
    getPointsTo(context, V, result);

    if (result.size() == 1) {
      const llvm::Type *type = result[0]->getType();
      // the allocation site is a pointer type
      assert(type->isPointerTy());
      // get the actually allocated object type
      return type->getPointerElementType();
    }
    // do not know the type
    return nullptr;
  }

  [[nodiscard]] bool isObjectAllocInst(const llvm::Value *inst) {
    if (llvm::isa<llvm::AllocaInst>(inst)) {
      return true;
    } else if (auto call = llvm::dyn_cast<llvm::CallBase>(inst);
               call && call->getCalledFunction() != nullptr) {
      return LMT::isHeapAllocAPI(langModel.get(), call->getCalledFunction());
    }
    return false;
  }
  [[nodiscard]] bool alias(const ctx *c1, const llvm::Value *v1, const ctx *c2,
                           const llvm::Value *v2) const {
    assert(v1->getType()->isPointerTy() && v2->getType()->isPointerTy());

    NodeID n1 = LMT::getSuperNodeIDForValue(langModel.get(), c1, v1);
    NodeID n2 = LMT::getSuperNodeIDForValue(langModel.get(), c2, v2);

    assert(n1 != INVALID_NODE_ID && n2 != INVALID_NODE_ID &&
           "can not find node in constraint graph!");
    return PT::intersectWithNoSpecialNode(n1, n2);
  }

  [[nodiscard]] bool aliasIfExsit(const ctx *c1, const llvm::Value *v1,
                                  const ctx *c2, const llvm::Value *v2) const {
    assert(v1->getType()->isPointerTy() && v2->getType()->isPointerTy());

    NodeID n1 = LMT::getSuperNodeIDForValue(langModel.get(), c1, v1);
    NodeID n2 = LMT::getSuperNodeIDForValue(langModel.get(), c2, v2);

    if (n1 == INVALID_NODE_ID || n2 == INVALID_NODE_ID) {
      return false;
    }
    return PT::intersectWithNoSpecialNode(n1, n2);
  }

  [[nodiscard]] bool hasIdenticalPTS(const ctx *c1, const llvm::Value *v1,
                                     const ctx *c2,
                                     const llvm::Value *v2) const {
    assert(v1->getType()->isPointerTy() && v2->getType()->isPointerTy());

    NodeID n1 = LMT::getSuperNodeIDForValue(langModel.get(), c1, v1);
    NodeID n2 = LMT::getSuperNodeIDForValue(langModel.get(), c2, v2);

    assert(n1 != INVALID_NODE_ID && n2 != INVALID_NODE_ID &&
           "can not find node in constraint graph!");
    return PT::equal(n1, n2);
  }

  [[nodiscard]] bool containsPTS(const ctx *c1, const llvm::Value *v1,
                                 const ctx *c2, const llvm::Value *v2) const {
    assert(v1->getType()->isPointerTy() && v2->getType()->isPointerTy());

    NodeID n1 = LMT::getSuperNodeIDForValue(langModel.get(), c1, v1);
    NodeID n2 = LMT::getSuperNodeIDForValue(langModel.get(), c2, v2);

    assert(n1 != INVALID_NODE_ID && n2 != INVALID_NODE_ID &&
           "can not find node in constraint graph!");
    return PT::contains(n1, n2);
  }

  // Delegator of the language model
  [[nodiscard]] inline ConsGraphTy *getConsGraph() const {
    return LMT::getConsGraph(langModel.get());
  }

  [[nodiscard]] inline const CallGraphTy *getCallGraph() const {
    return LMT::getCallGraph(langModel.get());
  }

  [[nodiscard]] inline llvm::StringRef getEntryName() const {
    return LMT::getEntryName(this->getLangModel());
  }

  [[nodiscard]] inline const llvm::Module *getLLVMModule() const {
    return LMT::getLLVMModule(this->getLangModel());
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getDirectNode(const ctx *C, const llvm::Function *F) {
    return LMT::getDirectNode(this->getLangModel(), C,
                              F); //->getDirectNode(C, F);
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getDirectNodeOrNull(const ctx *C, const llvm::Function *F) {
    return LMT::getDirectNodeOrNull(this->getLangModel(), C, F);
  }

  [[nodiscard]] inline const InDirectCallSite<ctx> *
  getInDirectCallSite(const ctx *C, const llvm::Instruction *I) {
    return LMT::getInDirectCallSite(this->getLangModel(), C, I);
  }
};

template <typename LangModel, typename SubClass>
constexpr bool
SolverBase<LangModel, SubClass>::processAddrOf(CGNodeTy *src,
                                               CGNodeTy *dst) const {
#ifndef NDEBUG
  // should already been handled
  assert(!PT::insert(dst->getNodeID(), src->getNodeID()));
#endif
  return false;
}

// site. pts(dst) |= pts(src);
template <typename LangModel, typename SubClass>
bool SolverBase<LangModel, SubClass>::processCopy(CGNodeTy *src,
                                                  CGNodeTy *dst) {
  ProcessedCopy++;
  if (PT::unionWith(dst->getNodeID(), src->getNodeID())) {
    if (dst->isFunctionPtr()) {
      // node used for indirect call
      this->updateFunPtr(dst->getNodeID());
    }
    EffectiveCopy++;
    return true;
  }
  return false;
}

} // namespace xray

#undef DEBUG_TYPE

#endif
