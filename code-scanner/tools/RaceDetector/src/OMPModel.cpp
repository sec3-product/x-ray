#include "OMPModel.h"

#include "RDUtil.h"
#include "aser/PointerAnalysis/Program/CallSite.h"

using namespace llvm;

namespace aser {
namespace OMPModel {
llvm::SmallVector<llvm::Value *, 1> getLastIters(const llvm::Function *F) {
    // find last_iter variable
    SmallVector<Value *, 1> lastIters;
    for (auto const &BB : *F) {
        for (auto const &I : BB) {
            auto call = dyn_cast<CallBase>(&I);
            if (call == nullptr) {
                continue;
            }
            auto function = call->getCalledFunction();
            if (function == nullptr) {
                continue;
            }
            auto fname = function->getName();
            if (!isStaticForInit(fname)) {
                continue;
            }

            auto staticForInit = cast<StaticForInitCall>(call);
            auto pLastIter = staticForInit->getPLastIter();
            lastIters.push_back(pLastIter);

            // Find values lastIter is loaded into
            for (auto const &user : pLastIter->users()) {
                auto loadInst = dyn_cast<LoadInst>(user);
                if (loadInst == nullptr) {
                    continue;
                }
                lastIters.push_back(loadInst);
            }
        }
    }

    return lastIters;
}

std::map<const Instruction *const, const Instruction *const> getlastIterMap(const Function *F) {
    auto lastIters = getLastIters(F);

    std::map<const Instruction *const, const Instruction *const> result;

    // Quit early if empty
    if (lastIters.empty()) {
        return result;
    }

    // Search for branches guarded by lastIter flag
    for (auto const &BB : *F) {
        for (auto const &I : BB) {
            auto branch = dyn_cast<BranchInst>(&I);
            if (branch == nullptr || branch->isUnconditional()) {
                continue;
            }

            // Check if branch condition is lastIter
            auto condition = branch->getCondition();
            auto icmpInst = dyn_cast<ICmpInst>(condition);
            if (icmpInst == nullptr) {
                continue;
            }
            auto op1 = icmpInst->getOperand(0);

            auto match = std::find(lastIters.begin(), lastIters.end(), op1);
            if (match == lastIters.end()) continue;

            // Right now we assume clang will generate an instruction like:
            //    %10 = icmp eq i32 %.omp.is_last, 0
            //    br i1 %11, label %.omp.lastprivate.done, label
            //    %.omp.lastprivate.then
            auto op2 = cast<ConstantInt>(icmpInst->getOperand(1));
            assert(op2->isNullValue() && "lastiter cmp should be against 0");
            auto pred = icmpInst->getPredicate();
            assert((pred == CmpInst::Predicate::ICMP_EQ || pred == CmpInst::Predicate::ICMP_NE) &&
                   "We only handle EQ and NE");
            auto idx = (pred == CmpInst::Predicate::ICMP_EQ) ? 2 : 1;
            auto unguarded = cast<BasicBlock>(branch->getOperand(idx));
            auto p = std::make_pair(&I, unguarded->getFirstNonPHIOrDbgOrLifetime());
            result.insert(p);
        }
    }

    return result;
}

const Instruction *getEndOfMasterOrSingleSection(const aser::CallSite &CS) {
    auto func = CS.getCalledFunction();
    assert((isMaster(func) || isSingle(func)) && "Must be omp master or omp single");

    auto next = CS.getInstruction()->getNextNonDebugInstruction();

    // Expect an instruction in the format:
    //  `icmp ne %result, 0` or `icmp eq $result, 0`
    auto cmpInst = cast<ICmpInst>(next);
    auto pred = cmpInst->getPredicate();
    assert(pred == CmpInst::Predicate::ICMP_NE || pred == CmpInst::Predicate::ICMP_EQ);
    auto op2 = cast<ConstantInt>(cmpInst->getOperand(1));
    assert(op2->isNullValue() && "Second operand should be 0");

    // Next expect a branch where
    // one label goes to the start of the master region and
    // the other label goes to the end of the master region
    auto branchInst = cast<BranchInst>(cmpInst->getNextNonDebugInstruction());

    auto trueBlock = cast<BasicBlock>(branchInst->getOperand(1));
    auto falseBlock = cast<BasicBlock>(branchInst->getOperand(2));

    // Pick the end block based on the icmp predicate
    if (pred == CmpInst::Predicate::ICMP_NE) {
        return trueBlock->getFirstNonPHIOrDbgOrLifetime();
    }
    return falseBlock->getFirstNonPHIOrDbgOrLifetime();
}

const llvm::Instruction *getEndOfReduce(const aser::CallSite &CS) {
    // only for testing, can be removed
    auto func = CS.getCalledFunction();
    assert(isReduce(func) && "Cannot get end of non reduce instruction");

    // Clang generates code like:
    //     %14 = call i32 @__kmpc_reduce_nowait(...)
    //     switch i32 %14, label %.omp.reduction.default [
    //       i32 1, label %.omp.reduction.case1
    //       i32 2, label %.omp.reduction.case2
    //     ]
    // Where thedefault case '%.omp.reduction.default'
    // is the first block after the reduction code

    auto switchInst = cast<SwitchInst>(CS.getInstruction()->getNextNode());
    auto block = switchInst->getDefaultDest();
    return block->getFirstNonPHIOrDbgOrLifetime();
}

const llvm::Instruction *getEndOfOrdered(const aser::CallSite &CS) {
    auto func = CS.getCalledFunction();
    assert(isOrderedStart(func) && "Cannot get end of non ordered instruction");

    const llvm::Instruction *inst = CS.getInstruction()->getNextNode();
    while (true) {
        if (llvm::isa<llvm::CallBase>(inst)) {
            const CallSite CS2(inst);
            if (isOrderedEnd(CS2.getCalledFunction())) {
                break;
            }
        }

        inst = inst->getNextNode();
        if (!inst) return nullptr;  // return null if we do not find it
    }

    return inst;
}

const llvm::Function *getOutlinedFunction(const llvm::Instruction &forkCall) {
    assert(isFork(&forkCall) && "Cannot get outlined function from non omp fork call");

    // Offset to omp outlined funciton in fork call
    int callSiteOffset = 2;

    aser::CallSite CS(&forkCall);
    auto func = llvm::dyn_cast<llvm::Function>(CS.getArgOperand(callSiteOffset)->stripPointerCasts());
    assert(func && "Could not find forked function");

    return func;
}

const std::set<StringRef> ForkNames{
    "__kmpc_fork_call",
    "__kmpc_fork_teams",  // This is offloaded and we don't handle that yet
};

const std::set<StringRef> StaticForInitNames{
    "__kmpc_for_static_init_4",
    "__kmpc_for_static_init_4u",
    "__kmpc_for_static_init_8",
    "__kmpc_for_static_init_8u",
};

const std::set<StringRef> ReductionNames{
    "__kmpc_reduce_nowait", "__kmpc_reduce",
    // TODO: There are more but we need to make sure they are handled the same
};
const std::set<StringRef> OMPSetLockNames{
    "__omp_set_lock", "omp_set_lock", "omp_set_lock_", "omp_set_nest_lock_", "omp_set_nest_lock",
};
const std::set<StringRef> OMPUnsetLockNames{
    "__omp_unset_lock", "omp_unset_lock", "omp_unset_lock_", "omp_unset_nest_lock_", "omp_unset_nest_lock",
};

const std::set<StringRef> OMPReadAPINames{"f90io_fmt_reada", "f90io_fmt_read64_aa", "f90io_fmt_read_aa",
                                          "f90_str_copy_klen"};
const std::set<StringRef> OMPWriteAPINames{
    //"f90io_encode_fmta",
    "f90_str_copy_klen"
    //"f90io_sc_i_fmt_write",
    //"f90io_sc_fmt_write",
};

// Utility helper function
static inline bool containsAny(StringRef src, const std::set<StringRef> &list) { return list.find(src) != list.end(); }

bool isFork(const StringRef func_name) { return containsAny(func_name, ForkNames); }

bool isFork(const Function *func) {
    if (func != nullptr) {
        return containsAny(func->getName(), ForkNames);
    }
    return false;
}

bool isFork(const Instruction *inst) {
    if (isa<CallBase>(inst)) {
        aser::CallSite CS(inst);
        return isFork(CS.getCalledFunction());
    }
    return false;
}

bool isForkTeams(const Instruction *inst) {
    if (isa<CallBase>(inst)) {
        aser::CallSite CS(inst);
        return CS.getCalledFunction()->getName() == "__kmpc_fork_teams";
    }
    return false;
}
bool isPushNumThreads(const Function *func) { return func->getName() == "__kmpc_push_num_threads"; }
bool isPushNumTeams(const Function *func) { return func->getName() == "__kmpc_push_num_teams"; }

bool isGetThreadNum(const StringRef name) { return name == "omp_get_thread_num"; }
bool isGetMaxThreadsNum(const Function *func) { return func->getName() == "omp_get_max_threads"; }
bool isGetGlobalThreadNum(const StringRef name) { return name == "__kmpc_global_thread_num"; }

bool isGetGlobalThreadNum(const Function *func) { return isGetGlobalThreadNum(func->getName()); }

bool isDispatchNext(const llvm::StringRef name) { return name.startswith("__kmpc_dispatch_next_"); }
bool isDispatchNext(const llvm::Function *func) { return isDispatchNext(func->getName()); }
bool isDispatchInit(const llvm::StringRef name) { return name.startswith("__kmpc_dispatch_init_"); }
bool isDispatchInit(const llvm::Function *func) { return isDispatchInit(func->getName()); }

bool isStaticForInit(const StringRef name) { return containsAny(name, StaticForInitNames); }

bool isStaticForInit(const Function *func) { return containsAny(func->getName(), StaticForInitNames); }

bool isStaticForFini(const StringRef name) { return name == "__kmpc_for_static_fini"; }

bool isStaticForFini(const Function *func) { return isStaticForFini(func->getName()); }

bool isReduce(const StringRef name) { return containsAny(name, ReductionNames); }

bool isReduce(const Function *func) { return containsAny(func->getName(), ReductionNames); }

bool isReduceEnd(const StringRef name) { return name == "__kmpc_end_reduce"; }
bool isReduceEnd(const Function *func) { return isReduceEnd(func->getName()); }

bool isBarrier(const StringRef name) { return name == "__kmpc_barrier"; }

bool isBarrier(const Function *func) { return isBarrier(func->getName()); }

bool isSingle(const StringRef name) { return name == "__kmpc_single"; }

bool isSingle(const Function *func) { return isSingle(func->getName()); }

bool isSingleEnd(const StringRef name) { return name == "__kmpc_end_single"; }

bool isSingleEnd(const Function *func) { return isSingleEnd(func->getName()); }

bool isOrderedStart(const Function *func) { return func->getName() == "__kmpc_ordered"; }

bool isOrderedEnd(const Function *func) { return func->getName() == "__kmpc_end_ordered"; }

bool isCriticalStart(const StringRef name) { return name == "__kmpc_critical"; }

bool isCriticalStart(const Function *func) { return isCriticalStart(func->getName()); }

bool isCriticalEnd(const StringRef name) { return name == "__kmpc_end_critical"; }

bool isCriticalEnd(const Function *func) { return isCriticalEnd(func->getName()); }

bool isCritical(const StringRef name) { return isCriticalStart(name) || isCriticalEnd(name); }

bool isCritical(const Function *func) { return isCriticalStart(func) || isCriticalEnd(func); }

bool isMaster(const StringRef name) { return name == "__kmpc_master"; }

bool isMaster(const Function *func) { return isMaster(func->getName()); }

bool isMasterEnd(const StringRef name) { return name == "__kmpc_end_master"; }

bool isMasterEnd(const Function *func) { return isMasterEnd(func->getName()); }

bool isMasterEndOrSingleEnd(const Function *func) { return isMasterEnd(func) || isSingleEnd(func); }

bool isMasterOrSingle(const StringRef name) { return isMaster(name) || isSingle(name); }

bool isMasterOrSingle(const Function *func) { return isMasterOrSingle(func->getName()); }

bool isSetLock(const StringRef name) { return containsAny(name, OMPSetLockNames); }

bool isSetLock(const llvm::Function *func) { return isSetLock(func->getName()); }

bool isUnsetLock(const StringRef name) { return containsAny(name, OMPUnsetLockNames); }

bool isUnsetLock(const llvm::Function *func) { return isUnsetLock(func->getName()); }

bool isTask(const llvm::StringRef name) { return name == "__kmpc_omp_task"; }

bool isTask(const llvm::Function *func) { return isTask(func->getName()); }

bool isTaskAlloc(const llvm::Function *func) { return func->getName() == "__kmpc_omp_task_alloc"; }
bool isTaskDepend(const llvm::Function *func) { return func->getName() == "__kmpc_omp_task_with_deps"; }

bool isTaskWait(const StringRef name) { return name == "__kmpc_omp_taskwait"; }

bool isTaskWait(const Function *func) { return isTaskWait(func->getName()); }

bool isTaskGroupStart(const llvm::StringRef name) { return name == "__kmpc_taskgroup"; }

bool isTaskGroupStart(const llvm::Function *func) { return isTaskGroupStart(func->getName()); }

bool isTaskGroupEnd(const llvm::StringRef name) { return name == "__kmpc_end_taskgroup"; }

bool isTaskGroupEnd(const llvm::Function *func) { return isTaskGroupEnd(func->getName()); }

bool isAnyOpenMPCall(const Function *func) { return func->getName().startswith("__kmpc_"); }

bool isOmpDebug(const llvm::Function *func) { return func->getName().startswith(".omp_outlined._debug__"); }

bool isReadAPI(const llvm::Function *func) {
    return containsAny(func->getName(), OMPReadAPINames) || GraphBLASModel::isReadAPI(func);
}

bool isWriteAPI(const llvm::Function *func) {
    return containsAny(func->getName(), OMPWriteAPINames) || GraphBLASModel::isWriteAPI(func);
}

bool isReadWriteAPI(const llvm::Function *func) { return isWriteAPI(func) || isReadAPI(func); }

bool isGetThreadNum(const llvm::Function *func) {
    return func->getName() == "omp_get_thread_num" || func->getName() == "omp_get_thread_num_";
}

// may return nullptr if taskfunction cannot be found
const llvm::Function *getTaskFunction(const aser::CallSite &taskCallSite) {
    bool isTaskOrTaskloop = OMPModel::isTask(taskCallSite.getCalledFunction());
    if (isTaskOrTaskloop) {
        assert(isTaskOrTaskloop && "getTaskFunction called on non omp task instruction");

        // The signature of the task call is:
        //   kmp_int32 __kmpc_omp_task(ident_t *loc_ref, kmp_int32 gtid, kmp_task_t *new_task)
        // Or
        //   __kmpc_taskloop(ident_t *loc, int gtid, kmp_task_t *task, int if_val,
        //                 kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st, int nogroup,
        //                 int sched, kmp_uint64 grainsize, void *task_dup)
        //
        // Important Note: In both calls the kmp_task_t is the 3rd argument
        //
        // The structure of the kmp_task_t object is:
        //   typedef struct kmp_task {
        //     void *shareds; // pointer to block of pointers to shared vars
        //     kmp_routine_entry_t routine; // pointer to routine to call for executing task
        //     kmp_int32 part_id; // part id for the task
        //     kmp_cmplrdata_t data1; // Two known optional additions: destructors and priority
        //     kmp_cmplrdata_t data2; // Process destructors first, priority second
        //     /* future data */
        //     /*  private vars  */
        //   } kmp_task_t;
        //
        // However the kmp_task_t objects are created through a call to __kmpc_omp_task_alloc
        // The signature is:
        //   kmp_task_t *__kmpc_omp_task_alloc(
        //     ident_t *loc_ref,
        //     kmp_int32 gtid,
        //     kmp_int32 flags,
        //     size_t sizeof_kmp_task_t,
        //     size_t sizeof_shareds,
        //     kmp_routine_entry_t task_entry)

        // We need to get the call to __kmpc_omp_task_alloc from the __kmpc_omp_task call (Arg 2)
        // And then get the task_entry function pointer from the __kmpc_omp_task_alloc call (Arg 5)

        auto op = taskCallSite.getArgOperand(2)->stripPointerCasts();
        auto taskAlloc = dyn_cast<CallBase>(op);
        if (!taskAlloc || !isTaskAlloc(taskAlloc->getCalledFunction())) {
            // LOG_DEBUG("Failed to find task function. inst={}", *taskCallSite.getInstruction());
            return nullptr;
        }
        aser::CallSite taskAllocCall(cast<Instruction>(op));
        auto taskFunc = taskAllocCall.getArgOperand(5)->stripPointerCasts();

        return llvm::cast<llvm::Function>(taskFunc);
    } else {
        // if (OMPModel::isTaskAlloc(taskCallSite.getCalledFunction()))
        // task alloc
        auto taskFunc = taskCallSite.getArgOperand(5)->stripPointerCasts();
        return llvm::cast<llvm::Function>(taskFunc);
    }
}

enum class Type { None = 0, Fork, Reduce, Barrier, Master, Single, CritStart, CritEnd };

Type getType(const Function *func) {
    if (isFork(func)) {
        return Type::Fork;
    } else if (isReduce(func)) {
        return Type::Reduce;
    } else if (isBarrier(func)) {
        return Type::Barrier;
    } else if (isMaster(func)) {
        return Type::Master;
    } else if (isSingle(func)) {
        return Type::Single;
    } else if (isCriticalStart(func)) {
        return Type::CritStart;
    } else if (isCriticalEnd(func)) {
        return Type::CritEnd;
    }
    // TODO: addd the others

    return Type::None;
}

}  // namespace OMPModel
}  // namespace aser
