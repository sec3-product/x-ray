#include "PTAModels/GraphBLASModel.h"

#include <llvm/ADT/StringSet.h>

#include "CustomAPIRewriters/ThreadAPIRewriter.h"
#include "PTAModels/GraphBLASHeapModel.h"
#include "aser/PointerAnalysis/Context/KOrigin.h"

using namespace aser;
using namespace llvm;

extern bool DEBUG_RUST_API;
extern bool CONFIG_EXPLORE_ALL_LOCK_PATHS;
bool isInCheckingMissedOmp = false;

std::set<StringRef> GraphBLASHeapModel::USER_HEAP_API;

const std::set<llvm::StringRef> StdVectorFunctions::VECTOR_READS{
    "operator[]", "begin()",    "end()",   "rbegin()", "rend()",  "crbegin()", "crend()", "size()",
    "max_size()", "capacity()", "empty()", "at(",      "front()", "back()",    "count("};

const std::set<llvm::StringRef> StdVectorFunctions::VECTOR_WRITES{
    "push_back(", "pop_back()",    "insert(",       "erase(",  "swap(",
    "emplace(",   "emplace_back(", "emplace_back<", "clear()", "resize("};

const llvm::StringRef StdStringFunctions::STRING_CTOR = "basic_string";

// NOTE: API not included:
// shrink_to_fit, reserve, get_allocator, c_str, operator basic_string_view
// operator<<, operator>>, operator</>/=>/=</<=>
// type convertion functions, hash, getline
const std::set<llvm::StringRef> StdStringFunctions::STRING_READS{"at",
                                                                 "operator[]",
                                                                 "front",
                                                                 "back",
                                                                 "data",
                                                                 "begin",
                                                                 "cbegin",
                                                                 "end",
                                                                 "cend",
                                                                 "rbegin",
                                                                 "crbegin",
                                                                 "rend",
                                                                 "crend",
                                                                 "empty",
                                                                 "size",
                                                                 "length",
                                                                 "max_size",
                                                                 "capacity",
                                                                 "compare",
                                                                 "starts_with",
                                                                 "ends_with",
                                                                 "substr",
                                                                 "copy",
                                                                 "find",
                                                                 "rfind",
                                                                 "find_first_of",
                                                                 "find_first_not_of",
                                                                 "find_last_of",
                                                                 "find_last not_of",
                                                                 "operator==",
                                                                 "operator!="};

// NOTE: swap seems like a special case, if a.swap(b), then we write to both a and b
const std::set<llvm::StringRef> StdStringFunctions::STRING_WRITES{
    "operator=", "assign",     "clear",   "insert", "erase", "push_back", "pop_back",
    "append",    "operator+=", "replace", "resize", "swap",  "operator+"};

const std::set<llvm::StringRef> OriginEntrys{
    "pthread_create",
    "signal",
    "sigaction",
    "GrB_Matrix_new",
    "__kmpc_fork_call",
    "__kmpc_fork_teams",
    "_ZNKSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info",
    "_ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EE7_M_swapERS2_"};  // for shared_ptr;

//_ZdlPv operator delete(void*)
//_ZdaPv operator delete[](void*)
const std::set<llvm::StringRef> MemoryFreeAPIs{"free", "redisFree", "je_free", "rax_free", "_ZdlPv", "_ZdaPv"};
const std::set<llvm::StringRef> ExtraLockAPIs{"std::mutex::lock()", "std::mutex::unlock()",
                                              "__gthread_mutex_lock(pthread_mutex_t*)",
                                              "__gthread_mutex_unlock(pthread_mutex_t*)"};
// make sure it is demangled
// const std::set<llvm::StringRef> CONFIG_MUST_EXPLORE_APIS{"_ExecutionPlan_FreeInternals"};
extern std::vector<std::string> CONFIG_MUST_EXPLORE_APIS;
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

bool GraphBLASModel::isInvokingAnOrigin(const ctx *prevCtx, const Instruction *I) {
    // if it is becomes too complex, maybe turn it into seperate class
    if (auto callBase = dyn_cast<CallBase>(I)) {
        // Peiming: use aser::CallSite here as it will do some simple target function resolution
        // such that
        // %fun = bitcast ...
        // call %fun
        // won't be considered as indirect call, and the origin will be inferred correctly
        // seems that there is always such pattern in fortran call to kmpc_fork_call
        auto CS = aser::CallSite(I);
        const llvm::Function *fun = CS.getCalledFunction();
        if (fun != nullptr) {
            if (OriginEntrys.find(fun->getName()) != OriginEntrys.end()) {
                return true;
            }
            if (fun->getName().startswith(ThreadAPIRewriter::getCanonicalizedAPIPrefix())) {
                return true;
            }

            std::string demangled = getDemangledName(fun->getName());

            if (demangled.rfind("std::thread::thread<", 0) == 0 ||
                demangled.rfind("public: __cdecl std::thread::thread<", 0) == 0) {
                return true;
            }

            // Heuristic for thread wrapper:
            // if a function has newed a object
            // and passes it to a thread creation API
            // then we should also consider this wrapper function as an origin
            for (auto &BB : *fun) {
                for (auto &inst : BB) {
                    if (auto call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
                        if (auto called = call->getCalledFunction()) {
                            // a heap allocator was called
                            if (heapModel.isHeapAllocFun(called)) {
                                // check if the pointer was passed to a thread creation API
                                for (auto user : call->users()) {
                                    if (auto callpthread = llvm::dyn_cast<llvm::CallBase>(user)) {
                                        if (auto pthread = callpthread->getCalledFunction()) {
                                            if (pthread->getName().equals("pthread_create") ||
                                                pthread->getName().equals("signal") ||
                                                pthread->getName().equals("sigaction")) {
                                                return true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

namespace {

static const char *tag = "pthread_specific";

}

extern cl::opt<bool> CONFIG_EXHAUST_MODE;

IndirectResolveOption GraphBLASModel::onNewIndirectTargetResolvation(const llvm::Function *target,
                                                                     const llvm::Instruction *callsite) {
    //    if (CONFIG_EXHAUST_MODE) {
    //        // no limit in exhaust mode
    //        return IndirectResolveOption::CRITICAL;
    //    }
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
    } else if (isThreadCreate(F)) {
        aser::CallSite CS(callsite);
        assert(CS.isCallOrInvoke());
        const llvm::Value *v;
        if (F->getName().startswith(ThreadAPIRewriter::getCanonicalizedAPIPrefix())) {
            v = CS.getArgOperand(1);  // the 2nd argument is the callback function
        } else {
            v = CS.getArgOperand(2);
        }

        if (auto threadFun = llvm::dyn_cast<llvm::Function>(v->stripPointerCasts())) {
            // replace call to pthread_create to the thread starting
            // routine
            F = threadFun;
        } else {
            // llvm_unreachable("need to handle more API");
            // pthread_create call back is indirect call.
            return {v, InterceptResult::Option::EXPAND_BODY};

            // llvm_unreachable("need to handle pthread_create using indirect call!");
            // return InterceptResult::IGNORE;
        }
        // thread creation site is intercepted, and the body of the
        // thread need to be handled.
        // return InterceptResult::EXPAND_BODY;
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

static bool isCXXInvokeMemFun(const CtxFunction<ctx> *callee) {
    ItaniumPartialDemangler demangler;
    if (!demangler.partialDemangle(callee->getFunction()->getName().begin())) {
        StringRef funName = demangler.getFunctionName(nullptr, nullptr);
        if (funName.startswith("std::__invoke_impl")) {
            StringRef funParams = demangler.getFunctionParameters(nullptr, nullptr);
            if (funParams.startswith("(std::__invoke_memfun_deref")) {
                return true;
            }
        }
    }

    return false;
}

static bool isCXXInvokeMemFunInStdThread(const CtxFunction<ctx> *callee) {
    if (isCXXInvokeMemFun(callee)) {
        // make sure it is called inside std::thread
        if (auto call = llvm::dyn_cast_or_null<CallBase>(callee->getContext()->getLast())) {
            if (auto fun = call->getCalledFunction();
                fun != nullptr && getDemangledName(fun->getName()).rfind("std::thread::thread<", 0) == 0) {
                // calling a std::thread
                return true;
            }
        }
    }
    return false;
}

static const Value *stripOffsetAndCast(const Value *V) {
    auto ret = V->stripInBoundsOffsets();
    while (auto gep = dyn_cast<GetElementPtrInst>(ret)) {
        ret = gep->getPointerOperand();
        ret = ret->stripInBoundsOffsets();
    }
    return ret;
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

    // TODO: this may be too aggressive, we need some benchmark on vector
    if (StdVectorFunctions::isVectorRead(callee->getFunction()) ||
        StdVectorFunctions::isVectorWrite(callee->getFunction())) {
        return true;
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
    } else if ((fun && isThreadCreate(fun)) || isThreadCreate(originalTarget)) {
        // dbg_os() << *callee->getFunction() << "\n";
        if (fun && fun->getName().startswith(ThreadAPIRewriter::getCanonicalizedAPIPrefix())) {
            int argNum = callee->getFunction()->arg_size();
            assert(argNum == fun->arg_size() - 2);
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
        } else {
            // We encounter case where coid casts a function with no args to a function
            // that takes void* so that it can be passed to pthread_create directly
            // assert(callee->getFunction()->arg_size() == 1);

            // If the callee has no args, nothing to link. Just return
            if (callee->getFunction()->arg_size() == 0) {
                return true;
            }

            // link the parameter
            PtrNode *formal = this->getPtrNodeOrNull(callee->getContext(), &*callee->getFunction()->arg_begin());
            if (formal == nullptr) {
                // some race cases where the callback function actually require a integer as the parameter
                // pthread_create(i8* (i8*)* bitcast (void (i32)* @myPartOfCalc to i8* (i8*)*), i8* %53)
                // and
                // void @myPartOfCalc(i32)
                return true;  // simply return
            }

            PtrNode *actual = this->getPtrNode(caller->getContext(), CS.getArgOperand(3));
            this->consGraph->addConstraints(actual, formal, Constraints::copy);
        }
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

    // FIXME: this is kinda dirty..
    // a dirty fix to std::thread which uses pointer to member function as the callable,
    // whose type is { i64, i64 } and thus is missed by pointer analysis
    // we manually resolved the points-to set here (if this is resolvable)
    if (caller->getContext() == callee->getContext() && isCXXInvokeMemFunInStdThread(callee)) {
        // 1st, find the indirect call
        PtrNode *indirectPtr = nullptr;
        Value *adjThis = nullptr;

        auto stdInvokeImpl = callee->getFunction();

        for (auto &BB : *stdInvokeImpl) {
            for (auto &I : BB) {
                if (auto call = dyn_cast<CallBase>(&I)) {
                    if (call->isIndirectCall()) {
                        indirectPtr = this->getPtrNode(callee->getContext(), call->getCalledOperand());
                        adjThis = call->getArgOperand(0);
                        break;
                    }
                }
            }
        }
        assert(indirectPtr && adjThis && "can not find the indirect call in std::__invoke_impl ??");

        // 2nd, resolve the target
        // get the std::thread creation site, which is the context's last item (if we are using origin-sensitive)
        // isCXXInvokeMemFunInStdThread guarantees the last context is call to std::threads
        // TODO: ctx insensitve, we need to find all the callsite to std::thread
        auto callStdThread = llvm::cast<CallBase>(callee->getContext()->getLast());
        auto callable = callStdThread->getArgOperand(1);

        if (callable->getType() == stdInvokeImpl->arg_begin()->getType()) {
            std::vector<const GetElementPtrInst *> geps;
            // now try to resolve the target if possible
            for (User *user : callable->users()) {
                if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
                    // find two geps that set the elements of the
                    geps.push_back(gep);
                }
            }

            if (geps.size() == 2) {
                auto gep1 = geps[0];
                auto gep2 = geps[1];
                // should looks like this
                // %.fca.1.gep2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %ref.tmp, i32 0, i32 1, !dbg !1596
                assert(gep1->getOperand(0) == callable && gep2->getOperand(0) == callable &&
                       gep1->getNumOperands() == 3 && gep2->getNumOperands() == 3);
                auto idx1 = cast<ConstantInt>(gep1->getOperand(1));
                auto idx2 = cast<ConstantInt>(gep2->getOperand(1));
                assert(idx1->getSExtValue() == 0 && idx2->getSExtValue() == 0);
                // we should have two geps to set { i64, i64 }
                // one is gep 0, 0
                // another is gep 0, 1
                idx1 = cast<ConstantInt>(gep1->getOperand(2));
                idx2 = cast<ConstantInt>(gep2->getOperand(2));
                assert((idx1->getSExtValue() == 0 && idx2->getSExtValue() == 1) ||
                       (idx1->getSExtValue() == 1 && idx2->getSExtValue() == 0));

                if (idx2->getSExtValue() == 0) {  //
                    std::swap(gep1, gep2);
                }

                const Function *target = nullptr;
                // here gep1 is the memptr.ptr, gep2 is the memptr.adj
                for (auto user : gep1->users()) {
                    // find a store
                    if (auto SI = dyn_cast<StoreInst>(user)) {
                        if (auto ptr2int = dyn_cast<PtrToIntInst>(SI->getValueOperand())) {
                            if (auto f = dyn_cast<Function>(ptr2int->getPointerOperand())) {
                                target = f;
                                break;
                            }
                        }
                    }
                }

                bool isAdjZero = false;
                for (auto user : gep2->users()) {
                    if (auto SI = dyn_cast<StoreInst>(user)) {
                        // find a store
                        if (auto adj = dyn_cast<ConstantInt>(SI->getValueOperand())) {
                            if (adj->getSExtValue() == 0) {
                                isAdjZero = true;
                            }
                        }
                    }
                }

                // TODO: what if the adjustment is non-zero?
                if (target != nullptr && isAdjZero) {
                    // we are sure that the callback is the resolved one, so the previous constraints are unnecessary
                    indirectPtr->clearConstraints();
                    PtrNode *targetNode = this->getPtrNode(CT::getGlobalCtx(), target);
                    this->consGraph->addConstraints(targetNode, indirectPtr, Constraints::copy);
                    this->consGraph->addConstraints(this->getPtrNode(callee->getContext(), stripOffsetAndCast(adjThis)),
                                                    this->getPtrNode(callee->getContext(), adjThis), Constraints::copy);
                }
            }
        }
    }

    return false;
}

bool GraphBLASModel::isStdThreadCreate(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return (llvm::demangle(CS.getCalledFunction()->getName().str())
                    .rfind("__coderrect_stub_thread_create_no_origin", 0) == 0);
        // from bitcoind: std::thread& std::vector<std::thread, std::allocator<std::thread> >::emplace_back
    }
}

bool GraphBLASModel::isStdThreadAssign(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return llvm::demangle(CS.getCalledFunction()->getName().str()) == "std::thread::operator=(std::thread&&)";
    }
}
bool GraphBLASModel::isStdThreadJoin(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return llvm::demangle(CS.getCalledFunction()->getName().str()) == "std::thread::join()" ||
               llvm::demangle(CS.getCalledFunction()->getName().str()) == "boost::thread::join()";
    }
}
bool GraphBLASModel::isLibEventSetCallBack(const Function *F) { return F->getName().equals("evhttp_set_gencb"); }
bool GraphBLASModel::isLibEventSetCallBack(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return isLibEventSetCallBack(CS.getCalledFunction());
    }
}
bool GraphBLASModel::isLibEventDispath(const Function *F) { return F->getName().equals("event_base_dispatch"); }
bool GraphBLASModel::isLibEventDispath(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return isLibEventDispath(CS.getCalledFunction());
    }
}
bool GraphBLASModel::isThreadCreate(const Function *F) {
    return F->getName().equals("pthread_create") || F->getName().equals("__coderrect_stub_thread_create_no_origin") ||

           F->getName().startswith(ThreadAPIRewriter::getCanonicalizedAPIPrefix());
}

bool GraphBLASModel::isThreadCreate(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return isThreadCreate(CS.getCalledFunction());  //|| isStdThreadCreate(inst)
    }
}

bool GraphBLASModel::isThreadJoin(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("pthread_join");
    }
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
bool GraphBLASModel::isCondWait(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("pthread_cond_wait");
    }
}

bool GraphBLASModel::isCondSignal(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("pthread_cond_signal");
    }
}
bool GraphBLASModel::isCondBroadcast(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("pthread_cond_broadcast");
    }
}
bool GraphBLASModel::isSyncCall(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return isSyncCall(CS.getCalledFunction());
    }
}
bool GraphBLASModel::isSyncCall(const Function *F) {
    return false;
}
bool GraphBLASModel::isSemaphoreWait(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("sem_wait") ||
               CS.getCalledFunction()->getName().startswith("tsem_wait");  //    tsem_wait(&pSub->sem);
    }
}

bool GraphBLASModel::isSemaphorePost(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("sem_post");
    }
}

bool GraphBLASModel::isMutexLock(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("pthread_mutex_lock") ||
               CS.getCalledFunction()->getName().equals("pthread_spin_lock") ||
               CS.getCalledFunction()->getName().equals(".coderrect.mutex.lock");
    }
}
bool GraphBLASModel::isMutexTryLock(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("pthread_mutex_trylock");
    }
}

bool GraphBLASModel::isMutexUnLock(const llvm::Instruction *inst) {
    aser::CallSite CS(inst);
    if (CS.isIndirectCall()) {
        return false;
    } else {
        return CS.getCalledFunction()->getName().equals("pthread_mutex_unlock") ||
               CS.getCalledFunction()->getName().equals("pthread_spin_unlock") ||
               CS.getCalledFunction()->getName().equals(".coderrect.mutex.unlock");
    }
}

bool GraphBLASModel::isRead(const llvm::Instruction *inst) { return llvm::isa<llvm::LoadInst>(inst); }

bool GraphBLASModel::isWrite(const llvm::Instruction *inst) { return llvm::isa<llvm::StoreInst>(inst); }

bool GraphBLASModel::isReadORWrite(const llvm::Instruction *inst) { return isRead(inst) || isWrite(inst); }

bool GraphBLASModel::isMemoryFree(const llvm::Function *func) {
    // auto name = llvm::demangle(func->getName());
    // llvm::outs() << "name: " << name << " - original: " << func->getName() << "\n";
    return MemoryFreeAPIs.find(func->getName()) != MemoryFreeAPIs.end();
}

bool GraphBLASModel::isMemoryFree(const llvm::Instruction *inst) {
    auto call = dyn_cast_or_null<CallBase>(inst);
    if (call == nullptr) {
        return false;
    }

    auto func = call->getCalledFunction();
    if (func == nullptr) {
        return false;
    }

    return isMemoryFree(func);
}

static std::set<const llvm::Function *> lockCallStackFunctions;
void GraphBLASModel::addLockCallStackFunction(std::vector<const Function *> &callStack) {
    // only consider the last five calls
    // int size = callStack.size();
    // int k = size-5;
    // if(k<0) k=0;
    // for (;k<size;k++){
    // auto f = callStack[k];
    if (CONFIG_EXPLORE_ALL_LOCK_PATHS)
        for (auto f : callStack) {
            auto result = lockCallStackFunctions.insert(f);
            if (result.second) {
                // std::cout << "Inserted a new lock call stack function: "<<llvm::demangle(f->getName().str())<<" size:
                // "<<lockCallStackFunctions.size()<<"\n";
            }
        }
}

bool GraphBLASModel::isCriticalAPI(const llvm::Function *func) {
    auto name = llvm::demangle(func->getName().str());
    // std::thread::thread<
    if (name.rfind("std::thread::", 0) == 0) return true;
    // llvm::outs() << "name: " << name << " - original: " << func->getName() << "\n";
    return (std::find(CONFIG_MUST_EXPLORE_APIS.begin(), CONFIG_MUST_EXPLORE_APIS.end(), name) !=
            CONFIG_MUST_EXPLORE_APIS.end()) ||
           ExtraLockAPIs.find(name) != ExtraLockAPIs.end() ||
           lockCallStackFunctions.find(func) != lockCallStackFunctions.end();
}

bool GraphBLASModel::isCriticalCallStack(std::vector<const llvm::Function *> &callStack) {
    for (auto f : callStack) {
        if (f->hasName()) {
            auto name = llvm::demangle(f->getName().str());
            if (std::find(CONFIG_MUST_EXPLORE_APIS.begin(), CONFIG_MUST_EXPLORE_APIS.end(), name) !=
                CONFIG_MUST_EXPLORE_APIS.end())
                return true;
        }
    }
    return false;
}

bool GraphBLASModel::isWriteAPI(const llvm::Function *func) {
    // model memory free as write as well
    return isMemoryFree(func) || StdVectorFunctions::isVectorWrite(func) || StdStringFunctions::isStringWrite(func) ||
           func->getName().equals("raxInsert") || func->getName().equals("raxRemove");
}
bool GraphBLASModel::isReadAPI(const llvm::Function *func) {
    return StdVectorFunctions::isVectorRead(func) || StdStringFunctions::isStringRead(func) ||
           func->getName().equals("raxFind");
}

bool GraphBLASModel::isReadWriteAPI(const llvm::Function *func) {
    return isWriteAPI(func) || isReadAPI(func) || StdVectorFunctions::isVectorAccess(func) ||
           StdStringFunctions::isStringAccess(func);
}

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
