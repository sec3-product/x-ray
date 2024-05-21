//
// Created by peiming on 5/6/20.
//

#include "Analysis/InferProducerConsumerPass.h"

#include "Graph/Event.h"
#include "RDUtil.h"
using namespace aser;
using namespace llvm;
using CGNodeTy = PTA::CGNodeTy;
static bool DEBUG = false;

// instruction * :
// the Store/Load instruction
// Value *:
// the shared buffer pointer
// this is extensible if we recognize more than Store/Load Instruction (e.g. vector::push) as accesses
using AccessesSite = std::pair<const Instruction *, const Value *>;

// the instruction is the callsite of the access site,
// TODO: we can make it stack to do CtxEvolve
// NOT a good programming practice, refactor it!!
using CtxAccessesSite = std::pair<const Instruction *, AccessesSite>;

// return the buffer where the address of the object is stored into
// static const Value *getUniEscapeSiteIfExsit(const Value *ptr) {
//    assert(ptr->getType()->isPointerTy());
//
//    const Value *escapeSite = nullptr;
//    for (auto user : ptr->users()) {
//        // handle more cases, e.g., the user is passed to function
//        // or some common containers
//        if (auto SI = dyn_cast<StoreInst>(user)) {
//            if (escapeSite == nullptr) {
//                escapeSite = SI->getPointerOperand(); // the stored memory
//            } else {
//                // we have more than one escape site
//                // e.g., the pointer is stored into many other places.
//                return nullptr;
//            }
//        }
//    }
//
//    return escapeSite;
//}

// TODO: seems traverse Constraint Graph is not straightforward, as the node might be collapsed if
// they are in SCC, which makes it impossible to find the original node without any mapping

// static std::set<Value *> getProducerSites(Value *res) {
//    // traverse the constraint graph to find out where the
//    // res is stored into global/thread sharing objects
//
//}
//
// static std::set<CGNodeTy *> getConsumerSites(CGNodeTy *res) {
//
//}

static int getArgNo(const CallBase *call, const Value *v) {
    int argNo = 0;
    for (; argNo < call->getNumArgOperands(); argNo++) {
        auto arg = call->getArgOperand(argNo);
        if (arg == v) {
            break;
        }
    }

    return argNo;
}

static std::vector<const ReturnInst *> getReturnVec(const llvm::Function *F) {
    std::vector<const ReturnInst *> result;

    for (auto &BB : *F) {
        for (auto &I : BB) {
            if (auto RI = dyn_cast<ReturnInst>(&I)) {
                result.push_back(RI);
            }
        }
    }

    return result;
}

// use LLVM use def chain to find the produce site instead of path in constraint graph
static std::set<CtxAccessesSite> getProduceSites(const Value *ptr, const llvm::Function *F) {
    // use-def chain
    ptr = ptr->stripInBoundsOffsets();  // find the base object

    std::set<const Use *> visited;
    std::vector<std::pair<const Use *, const Instruction *>> userChain;

    std::set<CtxAccessesSite> result;

    for (auto &use : ptr->uses()) {
        if (visited.insert(&use).second) {
            userChain.push_back(std::make_pair(&use, nullptr));
        }
    }

    while (!userChain.empty()) {
        const Use *use = userChain.back().first;
        const Instruction *callsite = userChain.back().second;

        userChain.pop_back();

        if (auto call = dyn_cast<CallBase>(use->getUser())) {
            // producer should not pass the pointer to indirect call
            if (call->getCalledFunction() == nullptr) {
                continue;
            }
            int argNo = getArgNo(call, use->get());

            if (argNo < call->getCalledFunction()->arg_size()) {
                Argument *arg = call->getCalledFunction()->arg_begin() + argNo;
                if (!arg->hasNoCaptureAttr()) {
                    for (auto &use : arg->uses()) {
                        if (visited.insert(&use).second) {
                            if (callsite == nullptr) {
                                userChain.push_back(std::make_pair(&use, call));
                            } else {
                                userChain.push_back(std::make_pair(&use, callsite));
                            }
                        }
                    }
                }
            }
        } else if (auto SI = dyn_cast<StoreInst>(use->getUser())) {
            // TODO: any other instructions that should be considerred as produce site?
            if (SI->getPointerOperand() != use->get() && !hasNoAliasMD(SI)) {
                result.insert(std::make_pair(callsite, std::make_pair(SI, SI->getPointerOperand())));
            }

        } else if (isa<GetElementPtrInst>(use->getUser()) || isa<BitCastInst>(use->getUser())) {
            for (auto &alias : use->getUser()->uses()) {
                if (visited.insert(&alias).second) {
                    userChain.push_back(std::make_pair(&alias, callsite));
                }
            }

        } else if (auto phi = dyn_cast<PHINode>(use->getUser())) {
            for (int i = 0; i < phi->getNumOperands(); i++) {
                auto V = phi->getOperand(i);

                if (!llvm::isa<Constant>(V)) {
                    for (auto &alias : V->uses()) {
                        if (visited.insert(&alias).second) {
                            userChain.push_back(std::make_pair(&alias, callsite));
                        }
                    }
                }
            }
        } else {
            continue;  // does not matter
        }
    }

    return result;
}

static std::set<AccessesSite> getConsumeSite(const Value *ptr, std::vector<CallEvent *> trace2) {
    ptr = ptr->stripInBoundsOffsets();
    while (auto arg = dyn_cast<Argument>(ptr)) {
        if (trace2.empty()) {
            return {};
        }

        // there could be many callsite that calls the function,
        // we pick the one that is reported in race report
        auto callsite = dyn_cast<CallBase>(trace2.back()->getInst());
        trace2.pop_back();

        int argNo = arg->getArgNo();
        if (callsite == nullptr) {
            return {};
        }

        ptr = callsite->getArgOperand(argNo)->stripInBoundsOffsets();
    }

    std::set<const Value *> visited;
    std::vector<const Value *> defChain;
    std::set<AccessesSite> result;

    visited.insert(ptr);
    defChain.push_back(ptr);

    while (!defChain.empty()) {
        const Value *def = defChain.back();
        defChain.pop_back();

        if (auto call = dyn_cast<CallBase>(def)) {
            if (call->getCalledFunction() == nullptr) {
                return {};
            }
            const std::vector<const ReturnInst *> &rets = getReturnVec(call->getCalledFunction());
            for (auto RI : rets) {
                const Value *V = RI->getReturnValue()->stripInBoundsOffsets();
                if (visited.insert(V).second) {
                    defChain.push_back(V);
                }
            }
        } else if (auto phi = dyn_cast<PHINode>(def)) {
            for (int i = 0; i < phi->getNumOperands(); i++) {
                auto V = phi->getOperand(i)->stripInBoundsOffsets();
                if (llvm::isa<GlobalValue>(V)) {
                    return {};
                }

                if (!llvm::isa<Constant>(V)) {
                    if (visited.insert(V).second) {
                        defChain.push_back(V);
                    }
                }
            }
        } else if (auto LI = dyn_cast<LoadInst>(def)) {
            // found load site
            // TODO: is it possible that we can recognize some API? std::queue for example.
            result.insert(std::make_pair(LI, LI->getPointerOperand()));
        } else {
            // from something we do not understand
            return {};
        }
    }
    return result;
}

static bool protectedByLockPair(std::pair<const Instruction *, const Instruction *> lock, const Instruction *I,
                                const DominatorTree &DT) {
    if (lock.first && lock.second) {
        if (DT.dominates(lock.first, I) && !DT.dominates(lock.second, I)) {
            // access is protected by the lock
            return true;
        }
    }
    return false;
}

static std::pair<const Instruction *, const Instruction *> getSurroundedLockUnlock(const Instruction *I,
                                                                                   const DominatorTree &DT) {
    // try to figure out whether it is between the enqueuing
    const Instruction *lockSite = nullptr;
    const Instruction *unLockSite = nullptr;

    auto fun = I->getFunction();

    // collect lock and unlock site
    for (auto &BB : *fun) {
        for (auto &I : BB) {
            if (isa<CallBase>(I)) {
                if (GraphBLASModel::isMutexLock(&I)) {
                    if (lockSite == nullptr) {
                        lockSite = &I;
                    } else {
                        // TODO: what if there are multiple locks, less common
                        return std::make_pair(nullptr, nullptr);
                    }
                } else if (GraphBLASModel::isMutexUnLock(&I)) {
                    if (unLockSite == nullptr) {
                        unLockSite = &I;
                    } else {
                        // TODO: what if there are multiple locks, less common
                        return std::make_pair(nullptr, nullptr);
                    }
                }
            }
        }
    }

    if (protectedByLockPair(std::make_pair(lockSite, unLockSite), I, DT)) {
        return std::make_pair(lockSite, unLockSite);
    }

    return std::make_pair(nullptr, nullptr);
}

//  producer -->store--> buffer -->load--> consumer

bool ProducerConsumerDetect::isProducerConsumer(MemAccessEvent *e1, const std::vector<CallEvent *> &trace1,
                                                MemAccessEvent *e2, const std::vector<CallEvent *> &trace2) {
    Function *e1Fun = const_cast<Function *>(e1->getInst()->getFunction());
    DominatorTree producerDT = DominatorTree(*e1Fun);

    //    llvm::outs() << *e1->getInst() << getSourceLoc(e1->getInst()).sig() << "\n";
    //    llvm::outs() << *e2->getInst() << getSourceLoc(e2->getInst()).sig() << "\n";
    //
    //    if (e1->getInst()->getFunction()->getName().equals("storage_compact_thread") &&
    //        e2->getInst()->getFunction()->getName().equals("_storage_compact_cb")) {
    //        llvm::outs();
    //    }

    // Function *e2Fun = const_cast<Function *>(e2->getInst()->getFunction());

    const Value *ptr1 = e1->getPointerOperand();
    const Value *ptr2 = e2->getPointerOperand();

    // auto node1 =  pta->getCGNode(e1->getContext(), ptr1);
    // auto node2 =  pta->getCGNode(e1->getContext(), ptr2);

    // a quick filtering
    // if consumer only get things from producer, and producer put every thing through the shared object
    // then consumer should have exactly the same pts as the producer.

    // consumer should have every pointer provided by producer
    if (!pass.pta->containsPTS(e2->getContext(), ptr2, e1->getContext(), ptr1)) {
        return false;
    }

    // 1st, assume the the e1 is the producer

    const std::set<AccessesSite> &consumeSite =
        getConsumeSite(ptr2, trace2);  // inst: ptr2=q->tail, val: addrof(q->tail)
    if (consumeSite.empty()) {
        return false;
    }

    const std::set<CtxAccessesSite> &produceSite =
        getProduceSites(ptr1, e1->getInst()->getFunction());  // inst: q->head = ptr1, val: addrof(q->tail)

    // debug print all consumers and producers
    if (DEBUG) {
        llvm::outs() << "\nProducers: " << produceSite.size() << "\n";
        for (auto cons : produceSite) {
            if (cons.second.first)
                llvm::outs() << *cons.second.first << "\n"
                             << "    " << getSourceLoc(cons.second.first).sig() << "\n";
        }
        llvm::outs() << "\nConsumers: " << consumeSite.size() << "\n";
        for (auto cons : consumeSite) {
            if (cons.first)
                llvm::outs() << *cons.first << "\n"
                             << "    " << getSourceLoc(cons.first).sig() << "\n";
        }
    }

    if (produceSite.empty()) {
        // PEIMING, WHY the produceSite can be empty?

        for (auto &cons : consumeSite) {
            if (pass.graph->hasCommonLockProtectInsts(e1->getTID(), e1->getInst(), e2->getTID(), cons.first)) {
                // llvm::outs() << "\nempty produceSite! " << produceSite.size() << "\n";
                // llvm::outs() << *e1->getInst() << getSourceLoc(e1->getInst()).sig() << "\n";
                // llvm::outs() << *e2->getInst() << getSourceLoc(e2->getInst()).sig() << "\n";

                return true;
            }
        }

        // PEIMING: THE FOLLOWING IS NOT USED ANY MORE
        // auto result = getSurroundedLockUnlock(e1->getInst(), producerDT);
        // if (result.first && result.second) {
        //     // we can find the lock and unlock
        //     for (auto &BB : *e1->getInst()->getFunction()) {
        //         for (auto &I : BB) {
        //             if (auto SI = dyn_cast<StoreInst>(&I)) {
        //                 if (protectedByLockPair(result, SI, producerDT)) {
        //                     // if the there is a write to the one of the comsumeSite
        //                     // it should be an instruction in producer thread.
        //                     for (auto &cons : consumeSite) {
        //                         if (pass.pta->aliasIfExsit(e1->getContext(), SI->getPointerOperand(),
        //                         e2->getContext(),
        //                                                    cons.second)) {
        //                             return true;
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // can not find the lock and unlock
        return false;
    }

    for (auto &cons : consumeSite) {
        bool findProducerSite = false;

        // for every potential consumer sites, there should be AT LEAST ONE corresponding producer site
        for (auto &prod : produceSite) {
            // TODO: the context of the produce site and consumer site might change if they are in different origin
            // handle it!
            if (pass.pta->aliasIfExsit(e1->getContext(), prod.second.second, e2->getContext(), cons.second)) {
                findProducerSite = true;
                break;
            }
        }

        if (!findProducerSite) {
            return false;
        }
    }

    // 3rd, store should happens before ALL produceSite

    // 1st case, the memaccess should happens before ALL produce site
    bool isAccessHappensBeforeProduce = true;
    for (auto prod : produceSite) {
        if (prod.second.first->getFunction() == e1->getInst()->getFunction()) {
            assert(prod.first == nullptr);
            // fast and common cases: the produce site and accesses are in the same function as
            if (!producerDT.dominates(e1->getInst(), prod.second.first)) {
                // if it is dominates
                isAccessHappensBeforeProduce = false;
                break;
            }
        } else {
            if (prod.first != nullptr) {
                // Can we simply reuse HB graph?
                if (!producerDT.dominates(e1->getInst(), prod.first)) {
                    // if it is dominates
                    isAccessHappensBeforeProduce = false;
                    break;
                }
            }
        }
    }

    if (!isAccessHappensBeforeProduce) {
        // check if e1 is in the processing of the producing
        for (auto &cons : consumeSite) {
            if (pass.graph->hasCommonLockProtectInsts(e1->getTID(), e1->getInst(), e2->getTID(), cons.first)) {
                if (DEBUG)
                    llvm::outs() << "\nTrue producer-consumer pattern because: e1 " << *e1->getInst()
                                 << " and consumer site is protected by the same lock " << *cons.first << "\n";

                return true;
            }
        }
        // PEIMING: THE FOLLOWING IS NOT USED ANY MORE
        // auto result = getSurroundedLockUnlock(e1->getInst(), producerDT);
        // if (result.first && result.second) {
        //     // we can find the lock and unlock
        //     for (auto &BB : *e1->getInst()->getFunction()) {
        //         for (auto &I : BB) {
        //             if (auto SI = dyn_cast<StoreInst>(&I)) {
        //                 if (protectedByLockPair(result, SI, producerDT)) {
        //                     // if the there is a write to the one of the comsumeSite
        //                     // it should be an instruction in producer thread.
        //                     for (auto &cons : consumeSite) {
        //                         if (pta->aliasIfExsit(e1->getContext(), SI->getPointerOperand(), e2->getContext(),
        //                                               cons.second)) {
        //                             return true;
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // can not find the lock and unlock
        // return false;

        for (auto &prod : produceSite) {
            for (auto &cons : consumeSite) {
                if (!pass.graph->hasCommonLockProtectInsts(e1->getTID(), prod.second.first, e2->getTID(), cons.first)) {
                    if (DEBUG)
                        llvm::outs() << "\nFalse producer-consumer pattern because: producer " << *prod.second.first
                                     << " and consumer is not protected by the same lock " << *cons.first << "\n";

                    return false;
                }
            }
        }

        // if e1 is after the producer's lock region, than it may still race
        for (auto &prod : produceSite) {
            if (!pass.graph->hasCommonLockProtectInsts(e1->getTID(), e1->getInst(), e1->getTID(), prod.second.first)) {
                if (DEBUG)
                    llvm::outs() << "\nFalse producer-consumer pattern because: e1 " << *e1->getInst()
                                 << " and produce site is not protected by the same lock " << *prod.second.first
                                 << "\n";

                return false;
            }
        }
    }

    // if not, try whether the accessing is in the producing phase

    // TODO: 4th, make sure the ALL producesite and consumesite protected by the same lock

    return true;
}

bool ProducerConsumerDetect::isProducerConsumerPair(MemAccessEvent *e1, const std::vector<CallEvent *> &trace1,
                                                    MemAccessEvent *e2, const std::vector<CallEvent *> &trace2) {
    if(false)
    if (e1->getType() == EventType::APIWrite || e1->getType() == EventType::APIRead ||
        e2->getType() == EventType::APIWrite || e2->getType() == EventType::APIRead) {
        return false;
    }
    return isProducerConsumer(e1, trace1, e2, trace2) || isProducerConsumer(e2, trace2, e1, trace1);
}
