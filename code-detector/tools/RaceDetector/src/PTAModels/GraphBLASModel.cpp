#include "PTAModels/GraphBLASModel.h"

#include <llvm/ADT/StringSet.h>

#include "CustomAPIRewriters/ThreadAPIRewriter.h"
#include "PTAModels/GraphBLASHeapModel.h"
#include "aser/PointerAnalysis/Context/KOrigin.h"

using namespace aser;
using namespace llvm;

extern bool DEBUG_RUST_API;

std::set<StringRef> GraphBLASHeapModel::USER_HEAP_API;

extern std::vector<std::string> CONFIG_INDIRECT_APIS;
extern std::map<std::string, std::string> CRITICAL_INDIRECT_TARGETS;

const std::string SOL_BUILT_IN_NAME = "sol.";  // for all the built-in sol function
const std::string SOL_BUILT_IN_MODEL_NAME = "sol.model.";

const std::set<llvm::StringRef> AUTHORITY_NAMES{"authority", "admin"};
const std::set<llvm::StringRef> USER_NAMES{"user", "payer"};

const std::set<llvm::StringRef> PREVILEDGE_NAMES{"authority", "owner", "manager",  "admin", "super",
                                                 "governor",  "exec",  "director", "chief"};  //"signer",
bool GraphBLASModel::isAuthorityAccount(llvm::StringRef accountName) {
    for (auto authority_name : AUTHORITY_NAMES) {
        if (accountName.contains(authority_name)) {
            return true;
        }
    }
    return false;
}
bool GraphBLASModel::isPreviledgeAccount(llvm::StringRef accountName) {
    for (auto previledge_name : PREVILEDGE_NAMES) {
        if (accountName.contains(previledge_name)) {
            return true;
        }
    }
    return false;
}
bool GraphBLASModel::isUserProvidedAccount(llvm::StringRef accountName) {
    for (auto user_name : USER_NAMES) {
        if (accountName.contains(user_name)) {
            return true;
        }
    }
    return false;
}

static int inline funcHasCallBack(const Function *fun) {
    for (int i = 0; i < fun->arg_size(); i++) {
        auto arg = fun->arg_begin() + i;
        if (arg->getType()->isPointerTy()) {
            if (arg->getType()->getPointerElementType()->isFunctionTy()) {
                return i;
            }
        }
    }

    return -1;
}

bool isConfiguredIndirectAPI(const llvm::Function *func) {
    auto it = std::find(CONFIG_INDIRECT_APIS.begin(), CONFIG_INDIRECT_APIS.end(), func->getName());
    if (it != CONFIG_INDIRECT_APIS.end()) {
        LOG_DEBUG("find configure indirect func: func={}, demangled={}", func->getName(),
                  llvm::demangle(func->getName().str()));
        // llvm::outs() << "find indirect func: " << llvm::demangle(func->getName()) << " - original: " <<
        // func->getName() << "\n";
        return true;
    } else {
        return false;
    }
}

// TODO: handle C++ mangled name
bool isCriticalIndirectTarget(const llvm::Instruction *callsite, const llvm::Function *func) {
    auto it = CRITICAL_INDIRECT_TARGETS.find(func->getName().str());
    if (it != CRITICAL_INDIRECT_TARGETS.end()) {
        std::string callerName = it->second;
        if (callsite->getFunction()->getName().equals(callerName)) {
            LOG_DEBUG("find configure critical indirect target: func={}, demangled={}", func->getName(),
                      llvm::demangle(func->getName().str()));
            return true;
        } else {
            return false;
        }
    }

    return false;
}

static const Value *getMatchedArg(const Argument *fArg, const CallBase *call) {
    assert(fArg->getType()->isPointerTy() && "only need to match pointer type");

    Value *matchedArg = nullptr;
    // else, checkout we can simply match the parameters
    for (auto &aArg : call->args()) {
        if (fArg->getType() == aArg->getType()) {
            if (!matchedArg) {
                matchedArg = cast<Value>(&aArg);
            } else {
                // multiple match, do not handle it
                return nullptr;
            }
        }
    }

    return matchedArg;
}

IndirectResolveOption GraphBLASModel::onNewIndirectTargetResolvation(const llvm::Function *target,
                                                                     const llvm::Instruction *callsite) {
    if (callsite->getFunction()->getName().equals("_ZZ9StartRESTRKN4util3RefEENK3$_1clEP11HTTPRequestRKNSt7__"
                                                  "cxx1112basic_stringIcSt11char_traitsIcESaIcEEE")) {
        return IndirectResolveOption::CRITICAL;
    }

    if (isConfiguredIndirectAPI(callsite->getFunction())) {
        return IndirectResolveOption::CRITICAL;
    }

    if (isCriticalIndirectTarget(callsite, target)) {
        return IndirectResolveOption::CRITICAL;
    }

    // insert if the limit is not exceeded.
    return IndirectResolveOption::WITH_LIMIT;
}

void GraphBLASModel::interceptHeapAllocSite(const aser::CtxFunction<aser::ctx> *caller,
                                            const aser::CtxFunction<aser::ctx> *callee,
                                            const llvm::Instruction *callsite) {
    // GrB_Matrix_new and GB_new take a pointer pointer of a Matrix to initialize
    // therefore we need to create a fake pointer node representing the pointer of a Matrix
    if (heapModel.isHeapInitFun(callee->getFunction())) {
        Type *type = callsite->getOperand(0)->getType()->getPointerElementType()->getPointerElementType();
        PtrNode *ptr = this->getPtrNode(caller->getContext(), callsite->getOperand(0));
        PtrNode *fakePtr = this->createAnonPtrNode();
        ObjNode *obj = this->allocHeapObj(caller->getContext(), callsite, type);

        this->consGraph->addConstraints(obj, fakePtr, Constraints::addr_of);
        this->consGraph->addConstraints(fakePtr, ptr, Constraints::store);
        return;
    } else if (heapModel.isHeapAllocFun(callee->getFunction())) {
        if (callee->getFunction()->getName().equals("f90_alloc04_chka_i8") ||
            callee->getFunction()->getName().equals("f90_ptr_alloc04a_i8")) {
            PtrNode *ptr = this->getPtrNode(caller->getContext(), llvm::cast<CallBase>(callsite)->getArgOperand(4));
            ObjNode *obj = this->allocHeapObj(caller->getContext(), callsite,
                                              getUnboundedArrayTy(IntegerType::get(callsite->getContext(), 8)));

            this->consGraph->addConstraints(obj->getAddrTakenNode(), ptr, Constraints::store);
            return;
        }

        Type *type = heapModel.inferHeapAllocType(callee->getFunction(), callsite);
        PtrNode *ptr = this->getPtrNode(caller->getContext(), callsite);
        ObjNode *obj = this->allocHeapObj(caller->getContext(), callsite, type);

        this->consGraph->addConstraints(obj, ptr, Constraints::addr_of);
        return;
    }
}

const StringRef GraphBLASModel::findGlobalString(const llvm::Value *value) {
    if (value) {
        if (auto user = dyn_cast<llvm::User>(value)) {
            auto globalVar = user->getOperand(0);
            if (auto gv = dyn_cast_or_null<GlobalVariable>(globalVar)) {
                // llvm::outs() << "global variable: " << *gv << "\n";
                if (auto globalData = dyn_cast_or_null<ConstantDataArray>(gv->getInitializer())) {
                    auto valueName = globalData->getAsString();
                    // llvm::outs() << "data: " << valueName << "\n";
                    return valueName;
                }
            }
        }
    }
    return "";
}

std::map<const llvm::Instruction *, const llvm::Function *> sigActionHandlerMap;
const llvm::Function *GraphBLASModel::findSignalHandlerFunc(const llvm::Instruction *callsite) {
    if (sigActionHandlerMap.find(callsite) != sigActionHandlerMap.end()) return sigActionHandlerMap.at(callsite);
    auto func = callsite->getFunction();
    aser::CallSite CS(callsite);
    const llvm::Value *v = CS.getArgOperand(1);  //%struct.sigaction*
    if (llvm::isa<llvm::AllocaInst>(v)) {
        LOG_TRACE("signal handle alloca ptr: {}", *v);

        // find all its uses
        for (const User *U : v->users()) {
            // TODO: find all the stores to handle alloc
            // for each store, get e from threadIDValueMap
            // Instruction *inst = dyn_cast<Instruction>(U->get());
            if (auto GEPI = llvm::dyn_cast<llvm::GetElementPtrInst>(U)) {
                LOG_TRACE("signal handle use. use={}", *U);
                // make sure it is the first element
                // LOG_DEBUG("gep number of operands: {}", GEPI->getNumOperands());
                if (GEPI->getNumOperands() > 2)
                    if (auto CONSI = llvm::dyn_cast<llvm::Constant>(GEPI->getOperand(2)))
                        if (CONSI->isZeroValue())
                            // heuristic: its next instructions are bitcast and store
                            if (auto BCI = llvm::dyn_cast<llvm::BitCastInst>(GEPI->getNextNode())) {
                                LOG_TRACE("signal handle next bitcast: {}", *BCI);

                                if (const StoreInst *SI = llvm::dyn_cast<llvm::StoreInst>(BCI->getNextNode())) {
                                    LOG_TRACE("signal handle next store: {}", *SI);
                                    if (auto sigFun =
                                            llvm::dyn_cast<llvm::Function>(SI->getOperand(0)->stripPointerCasts())) {
                                        sigActionHandlerMap[callsite] = sigFun;
                                        return sigFun;
                                    } else {
                                        // indirect function pointer?
                                        LOG_TRACE("find indirect signal handler: {}", *SI->getOperand(0));
                                        // try one more time
                                        {
                                            const llvm::Value *v2 = SI->getOperand(0)->stripPointerCasts();
                                            for (const User *U2 : v2->users()) {
                                                if (U2 == SI) continue;  // skip the same store
                                                LOG_TRACE("handle use2. use={}", *U2);
                                                if (auto SI2 = llvm::dyn_cast<llvm::StoreInst>(U2)) {
                                                    if (auto sigFun2 = llvm::dyn_cast<llvm::Function>(
                                                            SI2->getOperand(0)->stripPointerCasts())) {
                                                        sigActionHandlerMap[callsite] = sigFun2;
                                                        return sigFun2;
                                                    } else {
                                                        LOG_TRACE(
                                                            "indirect store2 again: {}",
                                                            *SI2->getOperand(
                                                                0));  // likely the func ptr is passed from the caller
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
            }
        }
    } else {
        // global variables?
        LOG_TRACE("signal handle global action ptr: {}", *v);
    }

    sigActionHandlerMap[callsite] = nullptr;
    return nullptr;
}
const llvm::Function *GraphBLASModel::findThreadStartFunc(PTA *pta, const ctx *context,
                                                          const llvm::Instruction *forkSite) {
    int callSiteOffset = 2;
    if (auto call = cast<CallBase>(forkSite)) {
        if (call->getCalledFunction()->getName().startswith(ThreadAPIRewriter::getCanonicalizedAPIPrefix())) {
            callSiteOffset = 1;
        } else if (call->getCalledFunction()->getName().equals("evhttp_set_gencb")) {
            callSiteOffset = 1;
        }
    }

    auto funptr = forkSite->getOperand(callSiteOffset)->stripPointerCasts();

    // case 1: directly passing the function pointer:
    // ```
    // pthread_create(&thread, NULL, PrintHello, NULL)
    // ```
    // where PrintHello is the function name;
    if (auto funStart = llvm::dyn_cast<llvm::Function>(funptr)) {
        return funStart;
    }
    // case 2: indirect passing the function pointer:
    // ```
    // void *(*callback) (void *) = PrintHello;
    // pthread_create(&thread, NULL, callback, NULL)
    // ```
    else {
        // FIXME: For indirect pthread callback, only traverse the first function ptr get resolved
        // eventually this should be fixed by having more precise pta results
        // such as adding context or inlining pthread wrapper
        auto cs = pta->getInDirectCallSite(context, forkSite);
        auto threadEntry = *cs->getResolvedNode().begin();
        if (threadEntry) return threadEntry->getTargetFun()->getFunction();
    }
    return nullptr;
}
// callbacks called by ConsGraphBuilder
InterceptResult GraphBLASModel::interceptFunction(const ctx *calleeCtx, const ctx *callerCtx, const llvm::Function *F,
                                                  const llvm::Instruction *callsite) {
    assert(!isHeapAllocAPI(F, callsite));

    if (F->hasName() && (F->getName().equals("raxFind") || F->getName().equals("raxInsert"))) {
        return {F, InterceptResult::Option::ONLY_CALLSITE};
    }

    if (F->hasName() && (F->getName().equals("moduleRegisterApi") || F->getName().equals("RM_GetApi"))) {
        // TESTING:
        // for map lookup and insertion
        return {F, InterceptResult::Option::ONLY_CALLSITE};
    }

    if (F->getName().startswith("_ZSt9call_once")) {
        // std::call_once
        // TODO: now only handle the simplest case when the Callable is passed as function pointer
        if (F->arg_size() >= 2) {
            auto callableTy = F->getArg(1)->getType();
            if (callableTy->isPointerTy() && callableTy->getPointerElementType()->isFunctionTy()) {
                // if the function is the a function
                aser::CallSite CS(callsite);
                const llvm::Value *v = CS.getArgOperand(1);

                if (auto threadFun = llvm::dyn_cast<llvm::Function>(v->stripPointerCasts())) {
                    // replace call to pthread_create to the thread starting
                    F = threadFun;
                } else {
                    // call back is indirect call.
                    return {v, InterceptResult::Option::EXPAND_BODY};
                }
            }
        }
    } else if (F->getName().equals("evhttp_set_gencb")) {
        aser::CallSite CS(callsite);
        assert(CS.isCallOrInvoke());
        const llvm::Value *v = CS.getArgOperand(1);

        if (auto threadFun = llvm::dyn_cast<llvm::Function>(v->stripPointerCasts())) {
            // replace call to pthread_create to the thread starting
            F = threadFun;
        } else {
            // call back is indirect call.
            return {v, InterceptResult::Option::EXPAND_BODY};
        }
    } else if (LangModel::isRegisterSignal(callsite)) {
        aser::CallSite CS(callsite);
        assert(CS.isCallOrInvoke());
        const llvm::Value *v = CS.getArgOperand(1);

        if (auto threadFun = llvm::dyn_cast<llvm::Function>(v->stripPointerCasts())) {
            F = threadFun;
        } else {
            // llvm_unreachable("need to handle more API");
            // this should never happen for signal handlers
            return {v, InterceptResult::Option::EXPAND_BODY};
        }
    } else if (LangModel::isRegisterSignalAction(callsite)) {
        aser::CallSite CS(callsite);
        assert(CS.isCallOrInvoke());
        const llvm::Value *v = CS.getArgOperand(1);
        // TODO: we need to find signal handler here
        // int sigaction(int signum, const struct sigaction *act,
        //              struct sigaction *oldact);
        //     If act is non-NULL, the new action for signal signum is installed
        //    from act.  If oldact is non-NULL, the previous action is saved in
        //    oldact.

        //    The sigaction structure is defined as something like:

        //        struct sigaction {
        //            void     (*sa_handler)(int);
        //            void     (*sa_sigaction)(int, siginfo_t *, void *);
        //            sigset_t   sa_mask;
        //            int        sa_flags;
        //            void     (*sa_restorer)(void);
        //        };

        if (auto threadFun = LangModel::findSignalHandlerFunc(callsite)) {
            F = threadFun;
        } else {
            // this should never happen for signal handlers
            return {nullptr, InterceptResult::Option::IGNORE_FUN};
        }
    } else if (F->isDeclaration()) {
        if (LangModel::isRustNormalCall(callsite)) {
            aser::CallSite CS(callsite);
            if (DEBUG_RUST_API) llvm::outs() << "TODO: intercepting sol library call: " << F->getName() << "\n";

            // TODO: find target of callsite
            return {nullptr, InterceptResult::Option::IGNORE_FUN};
        } else {
            // an extern api that uses call back
            // first get the callback function
            int cbIndex = funcHasCallBack(F);
            if (cbIndex >= 0) {
                auto call = llvm::cast<CallBase>(callsite);
                auto callbck = llvm::dyn_cast<llvm::Function>(call->getArgOperand(cbIndex));
                if (callbck == nullptr || callbck->isDeclaration()) {
                    // does not matter actually
                    return {nullptr, InterceptResult::Option::IGNORE_FUN};
                }

                for (auto &fArg : callbck->args()) {
                    if (fArg.getType()->isPointerTy()) {
                        if (getMatchedArg(&fArg, call) == nullptr) {
                            // can not find the matched arguments
                            return {nullptr, InterceptResult::Option::IGNORE_FUN};
                        }
                    }
                }

                // if all parameter can be matched...
                // replace it with call back (so that at least we have something to analyze).
                return {callbck, InterceptResult::Option::EXPAND_BODY};
            }
            // ignore empty function
            // return {nullptr, InterceptResult::Option::IGNORE };
        }
    }

    return {F, InterceptResult::Option::EXPAND_BODY};
}

// determine whether the resolved indirect call is compatible
bool GraphBLASModel::isCompatible(const llvm::Instruction *callsite, const llvm::Function *target) {
    auto call = llvm::cast<llvm::CallBase>(callsite);
    // only pthread will override to indirect call in default language model
    auto threadCreate = call->getCalledFunction();
    assert(threadCreate);

    if (threadCreate->getName().equals("pthread_create") ||
        threadCreate->getName().equals(ThreadAPIRewriter::getStandardCThreadCreateAPI())) {
        // pthread and thread library written in C
        // pthread call back type -> i8* (*) (i8*)
        if (target->arg_size() != 1) {
            return false;
        }
        // pthread's callback's return type does not matter.
        return target->arg_begin()->getType() == llvm::Type::getInt8PtrTy(callsite->getContext());
    } else if (threadCreate->getName().startswith(ThreadAPIRewriter::getCanonicalizedAPIPrefix())) {
        if (target->arg_size() != call->arg_size() - 2) {
            return false;
        }

        auto fit = call->arg_begin() + 2;
        for (const Argument &arg : target->args()) {
            const Value *param = *fit;
            if (param->getType()->isPointerTy() != arg.getType()->isPointerTy()) {
                // TODO: this is a loose check
                return false;
            }
            fit++;
        }
        return true;
    }

    llvm_unreachable("unrecognizable function");
}

static Optional<StringRef> getStringFromValue(const Value *V, const DataLayout &DL) {
    APInt offset(64, 0);
    auto baseObj = V->stripAndAccumulateConstantOffsets(DL, offset, false);

    if (auto GV = dyn_cast<GlobalVariable>(baseObj)) {
        auto strValue = llvm::dyn_cast_or_null<ConstantDataSequential>(GV->getInitializer());
        if (strValue && strValue->isCString()) {
            StringRef str = strValue->getAsCString();
            return str.substr(offset.getZExtValue());
        } else if (strValue && strValue->isString()) {
            StringRef str = strValue->getAsString();
            return str.substr(offset.getZExtValue());
        }
    }

    return {};
}

// static MapObject<StringRef, PT, >
bool GraphBLASModel::interceptCallSite(const CtxFunction<ctx> *caller, const CtxFunction<ctx> *callee,
                                       const llvm::Function *originalTarget, const llvm::Instruction *callsite) {
    assert(originalTarget != nullptr);
    assert(CT::contextEvolve(caller->getContext(), callsite) == callee->getContext());
    // the rule of context evolution should be obeyed.
    aser::CallSite CS(callsite);
    assert(CS.isCallOrInvoke());

    if (isRegisterSignal(callsite) || isRegisterSignalAction(callsite)) {
        return true;
    }

    if (callee->getFunction()->hasName() &&
        (callee->getFunction()->getName().equals("raxFind") || callee->getFunction()->getName().equals("raxInsert"))) {
        // TODO: choose another meaningful object for indexing for the map object
        auto mapObj = this->getOrAllocMapObj<StringRef>(CT::getGlobalCtx(), callsite);
        // key
        // auto strKey = getStringFromValue(CS.getArgOperand(0), this->getLLVMModule()->getDataLayout());

        // PtrNode of the value inserted into map
        // auto valNode = this->getPtrNode(caller->getContext(), CS.getArgOperand(1));

        //        if (callee )
        //        if (strKey.hasValue()) {
        //            // here get the key
        //            StringRef key = strKey.getValue();
        //
        //            // unknown key
        //            if (callee->getFunction()->getName().equals("moduleRegisterApi")) {
        //                // insert to map
        //                mapObj->insert(key, valNode);
        //            } else {
        //                // get from map
        //                auto node = mapObj->getElem(key);
        //                // int RM_GetApi(const char *funcname, void **targetPtrPtr)
        //                this->getConsGraph()->addConstraints(node, valNode, Constraints::store);
        //            }
        //        } else
        {
            // unknown key
            if (callee->getFunction()->getName().equals("raxInsert")) {
                auto valNode = this->getPtrNode(caller->getContext(), CS.getArgOperand(3));
                // insert to map
                mapObj->insertWithUnknownKey(valNode);
            } else {
                // get from map
                auto node = mapObj->getElemWithUnknownKey();
                auto dst = this->getPtrNode(caller->getContext(), callsite);
                // int RM_GetApi(const char *funcname, void **targetPtrPtr)
                this->getConsGraph()->addConstraints(node, dst, Constraints::copy);
            }
        }

        return true;
    }

    //    if (callee->getFunction()->hasName() && callee->getName().equals("realloc")) {
    //        if (auto call = dyn_cast<CallBase>(callsite)) {
    //            auto src = this->getPtrNode(caller->getContext(), callsite->getOperand(0));
    //            auto dst = this->getPtrNode(caller->getContext(), call);
    //
    //            this->consGraph->addConstraints(src, dst, Constraints::copy);
    //            return true;
    //        }
    //    }

    if (callee->getFunction()->hasName() && callee->getName().equals("dlsym")) {
        if (auto call = dyn_cast<CallBase>(callsite)) {
            // should be a global string
            auto symbolName = call->getArgOperand(1)->stripPointerCasts();
            auto result = getStringFromValue(symbolName, this->getLLVMModule()->getDataLayout());
            if (result.hasValue()) {
                StringRef funName = result.getValue();
                auto dlFun = callsite->getModule()->getFunction(funName);
                if (dlFun) {
                    // if the dlsym is looking up a function, treat it a an indirect call
                    auto symbolNode = this->getPtrNode(caller->getContext(), callsite);
                    auto funNode = this->getPtrNode(CT::getGlobalCtx(), dlFun);

                    this->consGraph->addConstraints(funNode, symbolNode, Constraints::copy);
                }
            }
        }
    }

    // a corner case: indirect call resolved to pthread_create, so we alway use
    // the original target to determine whether the callsite need to be intercepted
    // FIXME: should all other cases rely on the orignalTarget as well?
    // e.g., The resolved function need might need to be skipped as well
    auto fun = CS.getCalledFunction();
    if (fun && fun->getName().startswith("_ZSt9call_once")) {
        // call once
        if (fun->arg_size() >= 2) {
            auto callableTy = fun->getArg(1)->getType();
            if (callableTy->isPointerTy() && callableTy->getPointerElementType()->isFunctionTy()) {
                // if the function is the a function
                int argNum = callee->getFunction()->arg_size();
                for (int i = 0; i < argNum; i++) {
                    PtrNode *formal =
                        this->getPtrNodeOrNull(callee->getContext(), &*callee->getFunction()->arg_begin() + i);
                    if (formal == nullptr) {
                        // the argument is not a pointer
                        continue;
                    } else {
                        PtrNode *actual = this->getPtrNode(caller->getContext(), CS.getArgOperand(i + 2));
                        this->consGraph->addConstraints(actual, formal, Constraints::copy);
                    }
                }
                return true;
            }
        }
    } else if (fun && fun->getName().equals("evhttp_set_gencb")) {
        // handle libevent
        PtrNode *formal = this->getPtrNodeOrNull(callee->getContext(), callee->getFunction()->arg_begin());
        // 1st argument, an struct evhttp_request, fed by the framework, create an anonmyous object for it.
        auto actual = MMT::template allocateAnonObj<PT>(this->getMemModel(),  // memory model
                                                        caller->getContext(),
                                                        this->getLLVMModule()->getDataLayout(),  // data layout
                                                        callee->getFunction()->arg_begin()->getType(),
                                                        callee->getFunction()->arg_begin(),  // tag value
                                                        true);                               // do it recursively

        this->consGraph->addConstraints(actual, formal, Constraints::addr_of);

        PtrNode *formal2 = this->getPtrNode(callee->getContext(), callee->getFunction()->getArg(1));
        PtrNode *actual2 = this->getPtrNode(caller->getContext(), CS.getArgOperand(2));
        this->consGraph->addConstraints(formal2, actual2, Constraints::copy);

        return true;
    }

    // we need to use callsite to identify the thread creation
    // as the function was intercepted and replaced for thread creation
    // before
    if (auto fun = CS.getCalledFunction()) {
        if (fun->isDeclaration()) {
            // a extern api that uses call back
            // first get the callback function
            int cbIndex = funcHasCallBack(fun);
            if (cbIndex >= 0) {  // TODO: assert callback index is >= 0
                auto call = llvm::cast<CallBase>(callsite);
                auto callbck = llvm::dyn_cast<llvm::Function>(call->getArgOperand(cbIndex));

                if (callbck == nullptr) {
                    // it is actually unreachable, as the function is ignored
                    assert(false && "the function should be ignored already!");
                }

                for (auto &fArg : callbck->args()) {
                    if (fArg.getType()->isPointerTy()) {
                        auto aArg = getMatchedArg(&fArg, call);
                        if (aArg == nullptr) {
                            assert(false && "the function should be ignored already!");
                        }

                        PtrNode *formal = this->getPtrNode(callee->getContext(), &fArg);
                        PtrNode *actual = this->getPtrNode(caller->getContext(), aArg);  // First three arguemnts of

                        this->consGraph->addConstraints(actual, formal, Constraints::copy);
                    }
                }

                return true;
            }
        }
    }

    return false;
}

bool GraphBLASModel::isRegisterSignal(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("signal");
    }
}
bool GraphBLASModel::isRegisterSignalAction(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("sigaction");
    }
}

bool GraphBLASModel::isRead(const llvm::Instruction *inst) { return llvm::isa<llvm::LoadInst>(inst); }

bool GraphBLASModel::isWrite(const llvm::Instruction *inst) { return llvm::isa<llvm::StoreInst>(inst); }

bool GraphBLASModel::isReadORWrite(const llvm::Instruction *inst) { return isRead(inst) || isWrite(inst); }

bool GraphBLASModel::isRustNormalCall(const llvm::Function *func) { return isRustAPI(func) && !isRustModelAPI(func); }
bool GraphBLASModel::isRustNormalCall(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return GraphBLASModel::isRustNormalCall(CS.getCalledFunction());
    }
}
bool GraphBLASModel::isRustModelAPI(const llvm::Function *func) {
    return func->getName().startswith(SOL_BUILT_IN_MODEL_NAME);
}
bool GraphBLASModel::isRustAPI(const llvm::Function *func) { return func->getName().startswith(SOL_BUILT_IN_NAME); }
