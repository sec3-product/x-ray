#include "Races.h"

#include <llvm/Support/Regex.h>

#include <fstream>

#include "RDUtil.h"
#include "ReportTemplate/DeadLock.h"

#define DEFAULT_BUDGET 25

using namespace aser;
using namespace std;
using namespace llvm;

// for cost quote
extern unsigned int NUM_OF_IR_LINES;
extern unsigned int NUM_OF_ATTACK_VECTORS;
extern unsigned int NUM_OF_FUNCTIONS;
extern unsigned int TOTAL_SOL_COST;
extern unsigned int TOTAL_SOL_TIME;
extern set<StringRef> SMART_CONTRACT_ADDRESSES;
extern llvm::cl::opt<std::string> ConfigOutputPath;
extern cl::opt<std::string> TargetModulePath;
extern bool CONFIG_NO_FILTER;

extern std::map<std::string, std::map<std::string, std::string>> SOLANA_SVE_DB;

extern map<string, string> NONE_PARALLEL_FUNCTIONS;
extern vector<string> IGNORED_FUN_ALL;
extern vector<string> IGNORED_VAR_ALL;
extern vector<string> IGNORED_LOCATION_ALL;
extern vector<string> LOW_PRIORITY_FILE_NAMES;
extern vector<string> LOW_PRIORITY_VAR_NAMES;
extern vector<string> HIGH_PRIORITY_FILE_NAMES;
extern vector<string> HIGH_PRIORITY_VAR_NAMES;
extern bool USE_FAKE_MAIN;
extern bool PRINT_IMMEDIATELY;
extern bool TERMINATE_IMMEDIATELY;
extern std::string CR_FUNC_NAME;
extern std::string CR_ALLOC_OBJ_RECUR;
// Not to limit the number of bugs we collected
// by default we only collect at most 25 cases for each type of bug
static bool nolimit = false;

extern bool CONFIG_INTEGRATE_DYNAMIC;
extern set<const Function *> CR_UNExploredFunctions;
static set<const Function *> CR_RacyFunctions;

static set<std::string> RacyLocations;
static set<std::string> RacyLocationPairs;  // fun_name@line|
static set<std::string> IgnoreRacyLocations;

/* --------------------------------

            Data Race

----------------------------------- */

// DateRace related static fields
vector<DataRace> aser::DataRace::races;
// race signatures: based on source code information
set<std::string> aser::DataRace::raceSigs;
set<std::string> aser::DataRace::rawLineSigs;
map<std::string, bool> aser::DataRace::methodPairs;

uint aser::DataRace::budget = DEFAULT_BUDGET;
uint aser::DataRace::omp_budget = DEFAULT_BUDGET;

aser::DataRace::DataRace(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, bool isOmpRace, int P)
    : p(P),
      access1(srcInfo1),
      access2(srcInfo2),
      objInfo(objInfo),
      objName(objInfo.getName()),
      objTy(objInfo.getType()),
      objField(objInfo.getAccessPath()),
      objLine(objInfo.getLine()),
      objDir(objInfo.getDir()),
      objFilename(objInfo.getFilename()),
      objSrcLine(objInfo.getSourceLine()),
      isOmpRace(isOmpRace) {}

aser::DataRace::DataRace(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, int P)
    : DataRace(srcInfo1, srcInfo2, objInfo, false, P) {}

aser::DataRace::DataRace(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, const string &ompFilename,
                         const string &ompDir, const string &ompSnippet, const vector<string> &callingCtx, int P)
    : DataRace(srcInfo1, srcInfo2, objInfo, true, P) {
    this->ompFilename = ompFilename;
    this->ompDir = ompDir;
    this->ompSnippet = ompSnippet;
    this->callingCtx = callingCtx;
}

void aser::DataRace::print() {
    outs() << "\n==== Found a race between: \n"
           << "line " << access1.getLine() << ", column " << access1.getCol() << " in " << access1.getFilename()
           << " AND "
           << "line " << access2.getLine() << ", column " << access2.getCol() << " in " << access2.getFilename()
           << "\n";
    if (objInfo.isGlobalValue())
        error("Static variable: ");
    else
        error("Shared variable: ");
    info(objName, false);
    outs() << " at line " << objLine << " of " << objFilename << "\n";
    outs() << objSrcLine;
    if (!objField.empty()) {
        info("shared field: ", false);
        info(objField);
    }
    if (access1.isWrite())
        highlight("Thread 1 (write): ");
    else
        highlight("Thread 1 (read): ");
    outs() << access1.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(access1.getStackTrace());

    if (access2.isWrite())
        highlight("Thread 2 (write): ");
    else
        highlight("Thread 2 (read): ");
    outs() << access2.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(access2.getStackTrace());

    if (isOmpRace) {
        info("The OpenMP region this bug occurs:", true);
        if (ompFilename.size() > 0 && ompFilename.front() == '/')
            outs() << ompFilename << "\n";
        else
            outs() << ompDir + "/" + ompFilename << "\n";

        outs() << ompSnippet;
        info("Gets called from:", true);
        printStackTrace(callingCtx);
    }
}

json DataRace::to_json() {
    json sharedObj;
    sharedObj["line"] = objLine;
    sharedObj["name"] = objName;
    sharedObj["tyStr"] = objTy;
    sharedObj["field"] = objField;
    sharedObj["filename"] = objFilename;
    sharedObj["dir"] = objDir;
    sharedObj["sourceLine"] = objSrcLine;
    json race;
    race["priority"] = p;
    race["access1"] = access1;
    race["access2"] = access2;
    race["sharedObj"] = sharedObj;
    race["isOmpRace"] = isOmpRace;
    if (isOmpRace) {
        json ompInfo;
        ompInfo["filename"] = ompFilename;
        ompInfo["dir"] = ompDir;
        ompInfo["snippet"] = ompSnippet;
        ompInfo["callingCtx"] = callingCtx;
        race["ompInfo"] = ompInfo;
    }
    return race;
}

void aser::DataRace::outputJSON() {
    json rs;
    std::string path;
    if (!ConfigOutputPath.empty()) {
        path = ConfigOutputPath;
    } else {
        info("writing detection results to ./races.json");
        path = "races.json";
    }

    std::vector<json> raceJsons;
    for (auto &r : races) {
        raceJsons.emplace_back(r.to_json());
    }

    std::ofstream output(path, std::ofstream::out);
    rs["races"] = raceJsons;
    rs["version"] = 1;
    rs["generatedAt"] = getCurrentTimeStr();
    rs["bcfile"] = TargetModulePath.getValue();
    output << rs;
    output.close();
}

void aser::DataRace::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        omp_budget = configReportLimit;
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

bool aser::DataRace::filter(Event *e1, Event *e2, const ObjTy *obj, SourceInfo &srcInfo1, SourceInfo &srcInfo2,
                            SourceInfo &sharedObj) {
    // All filtering schemes under this branch
    // are heuristic-based (unsound)

    // LOG_DEBUG("debug filter for races. e1={}, e2={}", srcInfo1.sig(), srcInfo2.sig());

    if (!CONFIG_NO_FILTER &&  // do not filter double-free races
        (e1->getType() != EventType::APIWrite && e2->getType() != EventType::APIWrite)) {
        // FIXME: the two if-else filtering below will fail some DRB cases

        // TODO: the following seems to not work with cr_main - need to double check

        // if (srcInfo1.getFilename().empty() || srcInfo2.getFilename().empty()) {
        //     LOG_DEBUG("Skipping race because the source file info cannot be found. e1={}, e2={}", *e1->getInst(),
        //               *e2->getInst());
        //     return true;
        // }
        if (srcInfo1.getSourceLine().empty() && srcInfo2.getSourceLine().empty()) {
            LOG_DEBUG("Skipping race because source line info cannot be found. e1={}, e2={}", srcInfo1.sig(),
                      srcInfo2.sig());
            return true;
        }
        // if (sharedObj.getLine() == 0) {
        //     LOG_DEBUG("Skipping race because the shared object info cannot be found. e1={}, e2={}", *e1->getInst(),
        //               *e2->getInst());
        //     return true;
        // }

        {
            auto field = obj->getFieldAccessPath();
            auto sourceLine1 = srcInfo1.getSourceLine();
            auto sourceLine2 = srcInfo1.getSourceLine();
            if (!field.empty() && !sourceLine1.empty() && !sourceLine2.empty()) {
                std::string delimiter = "->";
                field.erase(0, field.find_last_of(delimiter) + delimiter.length());
                if (sourceLine1.find(field) == string::npos || sourceLine2.find(field) == string::npos) {
                    LOG_DEBUG(
                        "Skipping potential race because shared variable does not appear in source lines. e1={}, e2={}",
                        *e1->getInst(), *e2->getInst());
                    return true;
                }
            }
        }
        // if the source line is the same, but they reside in different file
        // we only report one
        std::string rawLineSig = getRaceRawLineSig(srcInfo1, srcInfo2);
        if (filterStrPattern(rawLineSig)) {
            return true;
        }
        if (!rawLineSig.empty()) {
            if (rawLineSigs.find(rawLineSig) == rawLineSigs.end()) {
#pragma omp critical(rawLineSigs)
                { rawLineSigs.insert(rawLineSig); }
            } else {
                return true;
            }
        }
    }

    // this is not a heuristic based filtering
    // and will only filter out identical races (due to our race detection algorithm)
    // therefore this should always be executed
    // (although this should have no effect with filtering turned on)
    std::string sig = getRaceSrcSig(srcInfo1, srcInfo2);
    if (raceSigs.find(sig) != raceSigs.end()) {
        return true;
    }
#pragma omp critical(raceSigs)
    { raceSigs.insert(sig); }
    return false;
}

// NOTE: for api callback, remove coderrect_cb. from stack trace
void cleanCallStackTrace(std::vector<std::string> &st1, std::vector<std::string> &st2) {
    if (st1.size() > 2 && st1[1].find(CR_FUNC_NAME) == 0) {
        st1.erase(st1.begin(), st1.begin() + 2);
    }
    if (st2.size() > 2 && st2[1].find(CR_FUNC_NAME) == 0) {
        st2.erase(st2.begin(), st2.begin() + 2);
    }
}

static Function *getOpenLibAPIFor(const ObjTy *anonObj) {
    if (auto call = dyn_cast<CallBase>(anonObj->getValue())) {
        if (call->getFunction()->getName().equals("cr_main") && call->getCalledFunction() != nullptr &&
            call->getCalledFunction()->getName().equals(CR_ALLOC_OBJ_RECUR)) {
            // this is a openlib anonymous object
            // find the caller function that used the object

            for (auto &BB : *call->getFunction()) {
                for (auto &I : BB) {
                    auto fakePthread = dyn_cast<CallBase>(&I);
                    if (fakePthread && fakePthread->getCalledFunction() &&
                        fakePthread->getCalledFunction()->getName().equals("pthread_create")) {
                        if (fakePthread->getArgOperand(3)->stripPointerCasts() != call) {
                            continue;
                        }

                        if (auto coderrectCB = dyn_cast<Function>(fakePthread->getArgOperand(2)->stripPointerCasts())) {
                            if (coderrectCB->hasName() && coderrectCB->getName().startswith(CR_FUNC_NAME)) {
                                // find the call to the openAPI
                                CallBase *lastCall = nullptr;
                                for (auto &BB : *coderrectCB) {
                                    for (auto &I : BB) {
                                        if (auto openAPICall = dyn_cast<CallBase>(&I)) {
                                            lastCall = openAPICall;
                                        }
                                    }
                                }

                                return lastCall->getCalledFunction();
                            }
                        }
                    }
                }
            }
            //            // 1st it is a bitcast
            //            if (call->user_empty()) {
            //                return nullptr;
            //            }
            //            auto bitCast = dyn_cast<BitCastInst>(*call->user_begin());
            //            if (bitCast == nullptr) {
            //                return nullptr;
            //            }
            //            if (bitCast->user_empty()) {
            //                return nullptr;
            //            }
            //            auto fakePthread = dyn_cast<CallBase>(*bitCast->user_begin());
            //            if (fakePthread == nullptr) {
            //                llvm::outs();
            //            }
            //
            //            if (fakePthread != nullptr) {
            //                if (fakePthread->getCalledFunction() &&
            //                    fakePthread->getCalledFunction()->getName().equals("pthread_create")) {
            //                    // if the callback function is coderrect_cb.XXX
            //                    if (auto coderrectCB =
            //                    dyn_cast<Function>(fakePthread->getArgOperand(2)->stripPointerCasts())) {
            //                        if (coderrectCB->hasName() && coderrectCB->getName().startswith(CR_FUNC_NAME)) {
            //                            // find the call to the openAPI
            //                            CallBase *lastCall = nullptr;
            //                            for (auto &BB : *coderrectCB) {
            //                                for (auto &I : BB) {
            //                                    if (auto openAPICall = dyn_cast<CallBase>(&I)) {
            //                                        lastCall = openAPICall;
            //                                    }
            //                                }
            //                            }
            //
            //                            return lastCall->getCalledFunction();
            //                        }
            //                    }
            //                }
            //            }
        }
    }

    return nullptr;
}

bool isArrayIndexRace(Event *e1, Event *e2) {
    if (auto storeInst = dyn_cast_or_null<StoreInst>(e1->getInst())) {  // e1 is memory write always
        return llvm::isa<llvm::GetElementPtrInst>(storeInst->getPointerOperand()->stripPointerCasts());
    } else if (auto loadInst = dyn_cast_or_null<LoadInst>(e1->getInst())) {  // e1 is memory read never
        return llvm::isa<llvm::GetElementPtrInst>(loadInst->getPointerOperand()->stripPointerCasts());
    }
    return false;
}
// collect race from raw data structure into JSON
// NOTE: also include filtering and terminal print-out
// NOTE: this is the public API to use
void aser::DataRace::collect(Event *e1, Event *e2, const ObjTy *obj,
                             std::map<TID, std::vector<CallEvent *>> &callEventTraces, int P) {
    SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
    SourceInfo srcInfo2 = getSourceLoc(e2->getInst());
    SourceInfo sharedObjLoc = getSourceLoc(obj->getValue());

    if (sharedObjLoc.getLine() == 0) {
        if (!obj->isAnonObj()) {
            LOG_DEBUG("Race shared object line 0: value={}", *obj->getValue());
            // llvm::outs() <<"Shared object line 0: "<<*obj->getValue()<<sharedObjLoc.str()<<"\n";
            // Function Attrs: nounwind uwtable
            // define noalias %struct.ProcedureCtx.8707* @Proc_LabelsCtx() #2 !dbg !1179091 {
            // return;  // JEFF: skip likely false positives
        } else {
            Function *openAPI = getOpenLibAPIFor(obj);
            if (openAPI != nullptr) {
                sharedObjLoc = getSourceLoc(openAPI);
            }
        }
    }
    // if the race get filterd out, no need to collect this race
    if (filter(e1, e2, obj, srcInfo1, srcInfo2, sharedObjLoc)) return;

    auto st1 = getStackTrace(e1, callEventTraces, srcInfo1.isCpp());
    auto st2 = getStackTrace(e2, callEventTraces, srcInfo2.isCpp());

    // user customized filtering (by pattern)
    if (isNotParallelFunctions(st1, st2, NONE_PARALLEL_FUNCTIONS)) return;

    if (customizedFilterIgnoreFunctions(e1, e2, IGNORED_FUN_ALL)) return;
    if (customizedFilterIgnoreVariables(sharedObjLoc.getName(), IGNORED_VAR_ALL)) return;
    if (customizedFilterIgnoreLocations(e1, e2, IGNORED_LOCATION_ALL)) return;

    // filter by method pairs
    auto msig = getRaceMethodSig(e1, e2, obj);
    bool isGep = isArrayIndexRace(e1, e2);
    if (methodPairs.find(msig) != methodPairs.end()) {
        // if it is GEP or an non-GEP race already exists
        if (isGep) return;
        bool nonGepExist = false;
#pragma omp critical(methodPairs)
        { nonGepExist = methodPairs.at(msig); }
        if (nonGepExist) return;
    }
#pragma omp critical(methodPairs)
    {
        methodPairs[msig] = !isGep;  // non-gep
    }
    P = customizedPriorityAdjust(P, sharedObjLoc.getName(), srcInfo1, srcInfo2, st1, st2, LOW_PRIORITY_FILE_NAMES,
                                 HIGH_PRIORITY_FILE_NAMES, LOW_PRIORITY_VAR_NAMES, HIGH_PRIORITY_VAR_NAMES);
    cleanCallStackTrace(st1, st2);
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            // TODO: show the entire call chain of the thread until main
            st1 = getCallStackUntilMain(e1, callEventTraces, srcInfo1.isCpp());
            st2 = getCallStackUntilMain(e2, callEventTraces, srcInfo2.isCpp());
            srcInfo1.setStackTrace(st1);
            srcInfo2.setStackTrace(st2);
            sharedObjLoc.setAccessPath(obj->getFieldAccessPath());
            if (sharedObjLoc.isGlobalValue()) P++;  // P = Priority::LEVEL6;

            // tryCorrectSourceInfo(srcInfo1, srcInfo2, sharedObjLoc);
            if (e1->getType() == EventType::Write || e1->getType() == EventType::APIWrite) srcInfo1.setWrite();
            if (e2->getType() == EventType::Write || e2->getType() == EventType::APIWrite) srcInfo2.setWrite();
            races.emplace_back(srcInfo1, srcInfo2, sharedObjLoc, P);
            --budget;
            // JEFF: insert racy locations
            if (CONFIG_INTEGRATE_DYNAMIC) {
                auto loc1 = e1->getInst()->getFunction()->getName().str() + "@" + std::to_string(srcInfo1.getLine());
                auto loc2 = e2->getInst()->getFunction()->getName().str() + "@" + std::to_string(srcInfo2.getLine());
                RacyLocations.insert(loc1);
                RacyLocations.insert(loc2);

                // insert racy functions
                CR_RacyFunctions.insert(e1->getInst()->getFunction());
                CR_RacyFunctions.insert(e2->getInst()->getFunction());

                // need to demangle for RacyLocationPairs
                loc1 =
                    demangle(e1->getInst()->getFunction()->getName().str()) + "@" + std::to_string(srcInfo1.getLine());
                loc2 =
                    demangle(e2->getInst()->getFunction()->getName().str()) + "@" + std::to_string(srcInfo2.getLine());
                if (srcInfo1.getLine() < srcInfo2.getLine()) {
                    RacyLocationPairs.insert(loc1 + "|" + loc2);
                } else {
                    RacyLocationPairs.insert(loc2 + "|" + loc1);
                }
            }

            if (PRINT_IMMEDIATELY) races.back().print();
            if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}
void aser::DataRace::collectOMP(Event *e1, Event *e2, const ObjTy *obj,
                                std::map<TID, std::vector<CallEvent *>> &callEventTraces, CallingCtx &callingCtx,
                                const Instruction *ompRegion, int P) {
    SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
    SourceInfo srcInfo2 = getSourceLoc(e2->getInst());
    auto sharedObjLoc = getSourceLoc(obj->getValue());
    if (sharedObjLoc.getLine() == 0) {
        // if (!obj->isAnonObj())
        {
            LOG_DEBUG("Race shared object line 0: value={}", *obj->getValue());
            // llvm::outs() << "Shared object line 0: " << *obj->getValue() << sharedObjLoc.str() << "\n";
            if (auto *val = llvm::dyn_cast<llvm::Argument>(obj->getValue())) {
                // llvm::outs() << "Argument object skipped: " << *val << "\n";
                return;  // JEFF: skip likely false positives
            }
            // Function Attrs: nounwind uwtable
            // define noalias %struct.ProcedureCtx.8707* @Proc_LabelsCtx() #2 !dbg !1179091 {
        }
    }
    // if the race get filterd out, no need to collect this race
    if (filter(e1, e2, obj, srcInfo1, srcInfo2, sharedObjLoc)) return;

    auto st1 = getStackTrace(e1, callEventTraces, srcInfo1.isCpp());
    auto st2 = getStackTrace(e2, callEventTraces, srcInfo2.isCpp());

    bool isCpp = srcInfo1.isCpp() & srcInfo2.isCpp();
    const auto cc = getCallingCtx(callingCtx, isCpp);

    // user customized filtering (by pattern)
    if (isNotParallelFunctions(st1, st2, NONE_PARALLEL_FUNCTIONS)) return;

    if (customizedFilterIgnoreFunctions(e1, e2, IGNORED_FUN_ALL)) return;
    if (customizedFilterIgnoreVariables(sharedObjLoc.getName(), IGNORED_VAR_ALL)) return;
    if (customizedFilterIgnoreLocations(e1, e2, IGNORED_LOCATION_ALL)) return;

    // filter by method pairs
    auto msig = getRaceMethodSig(e1, e2, obj);
    bool isGep = isArrayIndexRace(e1, e2);
    if (methodPairs.find(msig) != methodPairs.end()) {
        // if it is GEP or an non-GEP race already exists
        if (isGep) return;
        bool nonGepExist = false;
#pragma omp critical(methodPairs)
        { nonGepExist = methodPairs.at(msig); }
        if (nonGepExist) return;
    }
#pragma omp critical(methodPairs)
    {
        methodPairs[msig] = !isGep;  // non-gep
    }

    P = customizedPriorityAdjust(P, sharedObjLoc.getName(), srcInfo1, srcInfo2, st1, st2, LOW_PRIORITY_FILE_NAMES,
                                 HIGH_PRIORITY_FILE_NAMES, LOW_PRIORITY_VAR_NAMES, HIGH_PRIORITY_VAR_NAMES);
    cleanCallStackTrace(st1, st2);
    SourceInfo ompSrc = getSourceLoc(ompRegion);
    const std::string ompSnippet = getCodeSnippet(ompSrc, 0, 5);

    // for omp private only
    // fortran IR: PRIVATE(index)
    // auto found1 = ompSrc.getSourceLine().find("PRIVATE(");
    // if (found1 != string::npos) {
    //     auto line1 = ompSrc.getSourceLine().substr(found1);
    //                 llvm::outs() << "\n============== line1: " << line1<< "\n";

    //     auto found2 = line1.find(")");
    //     auto privateVars = line1.substr(0, found2);
    //                 llvm::outs() << "\n============== privateVars: " << privateVars<< "\n";

    //     auto found3 = srcInfo1.getSourceLine().find("=");
    //     auto var = srcInfo1.getSourceLine().substr(0, found3);
    //                         llvm::outs() << "\n============== private var: " << var<< "\n";

    //     if(privateVars.find(var)!= string::npos){
    //         //skip this race
    //                     llvm::outs() << "\n============== skipped fortran race: " << ompSnippet;
    //         return;
    //     }
    // }

    if (nolimit || omp_budget > 0) {
        srcInfo1.setStackTrace(st1);
        srcInfo2.setStackTrace(st2);
        sharedObjLoc.setAccessPath(obj->getFieldAccessPath());
        if (sharedObjLoc.isGlobalValue()) P += 2;  //= Priority::LEVEL6;

        if (srcInfo1.isInAccurate() && !srcInfo2.isInAccurate()) {
            srcInfo1 = srcInfo2;
        } else if (srcInfo2.isInAccurate() && !srcInfo1.isInAccurate()) {
            srcInfo2 = srcInfo1;
        }
        if (e1->getType() == EventType::Write || e1->getType() == EventType::APIWrite) srcInfo1.setWrite();
        if (e2->getType() == EventType::Write || e2->getType() == EventType::APIWrite || e1->getInst() == e2->getInst())
            srcInfo2.setWrite();
        races.emplace_back(srcInfo1, srcInfo2, sharedObjLoc, ompSrc.getFilename(), ompSrc.getDir(), ompSnippet, cc, P);
        --omp_budget;
        // JEFF: insert racy locations
        if (CONFIG_INTEGRATE_DYNAMIC) {
            auto loc1 = e1->getInst()->getFunction()->getName().str() + "@" + std::to_string(srcInfo1.getLine());
            auto loc2 = e2->getInst()->getFunction()->getName().str() + "@" + std::to_string(srcInfo2.getLine());
            RacyLocations.insert(loc1);
            RacyLocations.insert(loc2);

            // insert racy functions
            CR_RacyFunctions.insert(e1->getInst()->getFunction());
            CR_RacyFunctions.insert(e2->getInst()->getFunction());

            // need to demangle for RacyLocationPairs
            loc1 = demangle(e1->getInst()->getFunction()->getName().str()) + "@" + std::to_string(srcInfo1.getLine());
            loc2 = demangle(e2->getInst()->getFunction()->getName().str()) + "@" + std::to_string(srcInfo2.getLine());
            if (srcInfo1.getLine() < srcInfo2.getLine()) {
                RacyLocationPairs.insert(loc1 + "|" + loc2);
            } else {
                RacyLocationPairs.insert(loc2 + "|" + loc1);
            }
        }

        if (PRINT_IMMEDIATELY) races.back().print();
        if (TERMINATE_IMMEDIATELY) exit(1);
    }
}

void aser::DataRace::printAll() {
    // TODO: also consider Priority
    // TODO: should there also be an order between openmp races vs. normal races?
    std::sort(races.begin(), races.end());
    for (auto r : races) {
        r.print();
    }
}

void aser::DataRace::printSummary() { info("detected " + to_string(races.size()) + " races in total."); }

/* --------------------------------

            Order Violation

----------------------------------- */

std::vector<aser::OrderViolation> aser::OrderViolation::ovs;
uint aser::OrderViolation::budget = DEFAULT_BUDGET;

void aser::OrderViolation::collect(const MemAccessEvent *e1, const MemAccessEvent *e2, const ObjTy *obj,
                                   std::map<TID, std::vector<CallEvent *>> &callEventTraces, int P) {
    SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
    SourceInfo srcInfo2 = getSourceLoc(e2->getInst());
    SourceInfo sharedObjLoc = getSourceLoc(obj->getValue());

    // TODO: real ov filter
    static std::set<std::string> ovfilter;
    std::string sig1 = getRaceRawLineSig(srcInfo1);
    std::string sig2 = getRaceRawLineSig(srcInfo2);
    if (filterStrPattern(sig1) || filterStrPattern(sig2)) {
        return;
    }
    if (sig1.empty() || sig2.empty()) {
        return;
    }
    if (ovfilter.find(sig1) != ovfilter.end() || ovfilter.find(sig2) != ovfilter.end()) {
        return;
    }

    auto st1 = getStackTrace(e1, callEventTraces, srcInfo1.isCpp());
    auto st2 = getStackTrace(e2, callEventTraces, srcInfo2.isCpp());

    // user customized filtering (by pattern)
    if (isNotParallelFunctions(st1, st2, NONE_PARALLEL_FUNCTIONS)) return;

    if (customizedFilterIgnoreFunctions(e1, e2, IGNORED_FUN_ALL)) return;
    if (customizedFilterIgnoreVariables(sharedObjLoc.getName(), IGNORED_VAR_ALL)) return;
    if (customizedFilterIgnoreLocations(e1, e2, IGNORED_LOCATION_ALL)) return;
    P = customizedPriorityAdjust(P, sharedObjLoc.getName(), srcInfo1, srcInfo2, st1, st2, LOW_PRIORITY_FILE_NAMES,
                                 HIGH_PRIORITY_FILE_NAMES, LOW_PRIORITY_VAR_NAMES, HIGH_PRIORITY_VAR_NAMES);
    cleanCallStackTrace(st1, st2);
#pragma omp critical(ovfilter)
    {
        ovfilter.insert(sig1);
        ovfilter.insert(sig2);
    }
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            srcInfo1.setStackTrace(st1);
            srcInfo2.setStackTrace(st2);
            sharedObjLoc.setAccessPath(obj->getFieldAccessPath());
            if (sharedObjLoc.isGlobalValue()) P++;  // P = Priority::LEVEL6;

            if (e1->getType() == EventType::Write || e1->getType() == EventType::APIWrite) srcInfo1.setWrite();
            if (e2->getType() == EventType::Write || e2->getType() == EventType::APIWrite) srcInfo2.setWrite();

            ovs.emplace_back(srcInfo1, srcInfo2, sharedObjLoc, P);
            --budget;
            if (PRINT_IMMEDIATELY) ovs.back().print();
            if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}

void aser::OrderViolation::print() const {
    outs() << "\n==== Found an Order Violation between: \n"
           << "line " << access1.getLine() << ", column " << access1.getCol() << " in " << access1.getFilename()
           << " AND "
           << "line " << access2.getLine() << ", column " << access2.getCol() << " in " << access2.getFilename()
           << "\n";

    if (access1.isWrite())
        highlight("Thread 1 (write): ");
    else
        highlight("Thread 1 (read): ");
    outs() << access1.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(access1.getStackTrace());
    if (access2.isWrite())
        highlight("Thread 2 (write): ");
    else
        highlight("Thread 2 (read): ");
    outs() << access2.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(access2.getStackTrace());
}

json aser::OrderViolation::to_json() const {
    json ov;
    ov["priority"] = priority;
    ov["access1"] = access1;
    ov["access2"] = access2;
    return ov;
}

void aser::OrderViolation::printAll() {
    for (auto const &ov : OrderViolation::getOvs()) {
        ov.print();
    }
}

void aser::OrderViolation::printSummary() {
    info("detected " + to_string(OrderViolation::getNumOvs()) + " order violations in total.");
}

void aser::OrderViolation::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

/* --------------------------------

            Dead Lock

----------------------------------- */

// static fields
uint aser::DeadLock::budget = DEFAULT_BUDGET;
vector<DeadLock> aser::DeadLock::deadlocks;

void aser::DeadLock::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

aser::DeadLock::DeadLock(std::vector<SourceInfo> &locks, std::vector<std::vector<SourceInfo>> &traces, int P)
    : lockNum(locks.size()), locks(locks), dlTraces(traces), p(P) {}

// TODO: how to filter deadlock
// TODO: does deadlock need stacktrace?
void aser::DeadLock::collect(std::vector<const ObjTy *> locks, std::vector<std::vector<const LockEvent *>> dlTraces,
                             int P) {
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            auto size = dlTraces.size();
            std::vector<std::vector<SourceInfo>> dlInfo(size);
            // sort the acquiring events into their happening order
            // and get their source info
            for (int idx = 0; idx < size; idx++) {
                auto &trace = dlTraces[idx];
                std::sort(trace.begin(), trace.end(),
                          [](const LockEvent *e1, const LockEvent *e2) -> bool { return (e1->getID() < e2->getID()); });
                for (auto e : trace) {
                    dlInfo[idx].push_back(aser::getSourceLoc(e->getInst()));
                }
            }
            std::vector<SourceInfo> lockInfo;
            for (auto o : locks) {
                lockInfo.push_back(aser::getSourceLoc(o->getValue()));
            }
            // collect the serializable deadlock info info static field
            deadlocks.emplace_back(lockInfo, dlInfo, P);
            if (PRINT_IMMEDIATELY) deadlocks.back().print();
            if (TERMINATE_IMMEDIATELY) exit(1);
            --budget;
        }
    }
}

void aser::DeadLock::printAll() {
    for (auto dl : deadlocks) {
        dl.print();
    }
}

void aser::DeadLock::printSummary() { info("detected " + to_string(deadlocks.size()) + " deadlocks in total."); }

void aser::DeadLock::print() {
    // tplPrint(DLTemplate, to_json());
    // do not use the template due to crashes
    // dlTraces std::vector<std::vector<SourceInfo>> &traces
    // locks std::vector<SourceInfo> &locks,

    // TODO show full call stack
    outs() << "\nFound a potential deadlock:\n";
    // int lockId=1;
    // for(auto l: locks){
    //     outs()<<"  lock"<<lockId++<<": "<<l.str();
    // }
    int threadId = 1;
    for (auto trace : dlTraces) {
        outs() << "====thread" << threadId++ << " lock history====\n";
        for (auto e : trace) {
            outs() << e.str();
        }
    }
}

json &aser::DeadLock::to_json() {
    if (j.empty()) {
        j["acquires"] = dlTraces;
        j = json{{"priority", p}, {"number", lockNum}, {"locks", locks}, {"acquires", dlTraces}};
    }
    return j;
}

/* --------------------------------

           Mismatched API

----------------------------------- */

// static fields
uint aser::MismatchedAPI::budget = DEFAULT_BUDGET;
vector<MismatchedAPI> aser::MismatchedAPI::mismatchedAPIs;
// used for filtering
set<string> aser::MismatchedAPI::apiSigs;
std::set<std::vector<std::string>> aser::MismatchedAPI::callStackSigs;

void aser::MismatchedAPI::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

string aser::MismatchedAPI::getErrorMsg(MismatchedAPI::Type type) {
    string msg;
    switch (type) {
        case Type::REENTRANT_LOCK:
            msg =
                "Found a potential re-entrant lock:\n"
                "   You might have missed an unlock before this call,\n"
                "   which may cause a deadlock.";
            break;
        case Type::MISS_LOCK:
            msg =
                "Found a potential mismatched lock/unlock:\n"
                "   You might have missed a lock before this unlock call.\n";
            break;
        case Type::MISS_UNLOCK:
            msg =
                "Found a potential mismatched lock/unlock:\n"
                "   You might have forgotten to release the lock.\n";
            break;
        case Type::MISS_SIGNAL:
            msg =
                "Found a potential missing pthread_cond_signal:\n"
                "    This pthread_cond_wait() may block forever.";
            break;
        case Type::UNCHECKED_COND_WAIT:
            msg =
                "Found a potential misuse of pthread_cond_wait:\n"
                "    You may have missed a while(...) check.\n";
            break;
        default:
            assert(false && "unhandled mismatched API");
            break;
    }
    return msg;
}

// we report at most 1 MismatchedAPI bug for each function call
bool aser::MismatchedAPI::filter(SourceInfo &srcInfo) {
    if (apiSigs.find(srcInfo.sig()) != apiSigs.end()) {
        return true;
    }
#pragma omp critical(apiSigs)
    { apiSigs.insert(srcInfo.sig()); }
    return false;
}

// we report at most 1 MismatchedAPI bug for each call stack
bool aser::MismatchedAPI::filterByCallStack(std::vector<std::string> &st0) {
    auto st = st0;
    // this one is special: keep only the last few entries
    if (st0.size() > 1) {
        std::vector<std::string> vect(st0.begin() + st0.size() - 1, st0.end());
        st = vect;
    }

    if (callStackSigs.find(st) != callStackSigs.end()) {
        return true;
    }
#pragma omp critical(callStackSigs)
    { callStackSigs.insert(st); }
    return false;
}

static bool isInsideMultiPurposeAPI(const Event *e) {
    auto prevInst = e->getInst()->getPrevNode();
    if (auto prevCall = dyn_cast_or_null<CallBase>(prevInst)) {
        if (prevCall->getCalledFunction() && prevCall->getCalledFunction()->getName().startswith(".coderrect")) {
            // caused by multiple purpose API
            return true;
        }
    }
    auto nextInst = e->getInst()->getNextNode();
    if (auto nextCall = dyn_cast_or_null<CallBase>(nextInst)) {
        if (nextCall->getCalledFunction() && nextCall->getCalledFunction()->getName().startswith(".coderrect")) {
            // caused by multiple purpose API
            return true;
        }
    }

    return false;
}

void aser::MismatchedAPI::collect(const Event *e, map<TID, vector<CallEvent *>> &callEventTraces, Type type, int P) {
    if (isInsideMultiPurposeAPI(e)) {
        return;
    }

    SourceInfo srcInfo = getSourceLoc(e->getInst());
    if (filter(srcInfo)) return;

    std::vector<std::string> st = getStackTrace(e, callEventTraces, srcInfo.isCpp());
    cleanCallStackTrace(st, st);
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            srcInfo.setStackTrace(st);
            auto msg = getErrorMsg(type);  // TODO: add source location
            mismatchedAPIs.emplace_back(srcInfo, msg, P);
            --budget;
            if (PRINT_IMMEDIATELY) mismatchedAPIs.back().print();
            // intentionally commented out since mismatchedAPIs needs improvement
            // if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}

// know FP in fortran mode due to flang IR eg in Clover_Leaf
//  %71 = call i32 (...) @_mp_bcs_nest_red(), !dbg !2906
//  %72 = call i32 (...) @_mp_bcs_nest_red(), !dbg !2906
void aser::MismatchedAPI::collect(const Event *e, const Event *e_last, map<TID, vector<CallEvent *>> &callEventTraces,
                                  Type type, int P) {
    if (isInsideMultiPurposeAPI(e) || isInsideMultiPurposeAPI(e_last)) {
        return;
    }

    // ignore re-entrant lock in ignored functions
    if (customizedFilterIgnoreFunctions(e, e_last, IGNORED_FUN_ALL)) return;

    SourceInfo srcInfo = getSourceLoc(e->getInst());

    // if (filter(srcInfo)) return;//TODO filter by callstack

    std::vector<std::string> st = getStackTrace(e, callEventTraces, srcInfo.isCpp());
    cleanCallStackTrace(st, st);

    if (filterByCallStack(st)) return;  // filter by callstack

#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            // TODO: show the entire call chain of the thread until main
            st = getCallStackUntilMain(e, callEventTraces, srcInfo.isCpp());
            srcInfo.setStackTrace(st);
            auto msg = getErrorMsg(type);  // TODO: add source location
            SourceInfo srcInfo2 = getSourceLoc(e_last->getInst());
            auto st2 = getCallStackUntilMain(e_last, callEventTraces, srcInfo2.isCpp());
            srcInfo2.setStackTrace(st2);
            auto I = e_last->getInst();
            std::string msg2 = "Previously locked at " + getSourceLoc(I).sig() +
                               " in func: " + demangle(I->getFunction()->getName().str());
            mismatchedAPIs.emplace_back(srcInfo, srcInfo2, msg, msg2, P);
            --budget;
            if (PRINT_IMMEDIATELY) {
                mismatchedAPIs.back().print();
            }
            // intentionally commented out since mismatchedAPIs needs improvement
            // if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}

aser::MismatchedAPI::MismatchedAPI(SourceInfo &srcInfo, std::string msg, int P)
    : apiInst(srcInfo), errorMsg(msg), p(P) {}

aser::MismatchedAPI::MismatchedAPI(SourceInfo &srcInfo, SourceInfo &srcInfo2, std::string msg, std::string msg2, int P)
    : apiInst(srcInfo), apiInst2(srcInfo2), errorMsg(msg), errorMsg2(msg2), p(P) {}

json aser::MismatchedAPI::to_json() {
    if (!errorMsg2.empty()) {
        json j({{"priority", p},
                {"inst", apiInst},
                {"inst2", apiInst2},
                {"errorMsg", errorMsg},
                {"errorMsg2", errorMsg2}});
        return j;
    }
    json j({{"priority", p}, {"inst", apiInst}, {"errorMsg", errorMsg}});
    return j;
}
void aser::MismatchedAPI::print() {
    outs() << "\n" << errorMsg << "\n";
    outs() << apiInst.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(apiInst.getStackTrace());
    if (!errorMsg2.empty()) {
        outs() << "\n" << errorMsg2 << "\n";
        outs() << apiInst2.getSnippet();
        outs() << ">>>Stack Trace:\n";
        printStackTrace(apiInst2.getStackTrace());
    }
}
void aser::MismatchedAPI::printAll() {
    std::sort(mismatchedAPIs.begin(), mismatchedAPIs.end());
    for (auto r : mismatchedAPIs) {
        r.print();
    }
}

void aser::MismatchedAPI::printSummary() {
    info("detected " + to_string(mismatchedAPIs.size()) + " mismatched apis in total.");
}

/* --------------------------------

           SVE Types

----------------------------------- */
std::map<SVE::Type, std::string> aser::SVE::sveTypeIdMap;
void SVE::addTypeID(std::string ID, SVE::Type type) { sveTypeIdMap[type] = ID; }
std::string SVE::getTypeID(SVE::Type type) {
    if (sveTypeIdMap.find(type) != sveTypeIdMap.end())
        return sveTypeIdMap.at(type);
    else
        return "";
}
std::set<std::string> aser::SVE::disabledCheckers;
void SVE::addDisabledChecker(std::string ID) { disabledCheckers.insert(ID); }
bool SVE::isCheckerDisabled(SVE::Type type) {
    auto id = SVE::getTypeID(type);
    if (disabledCheckers.find(id) != disabledCheckers.end())
        return true;
    else
        return false;
}
/* --------------------------------

           CosplayAccounts

----------------------------------- */

// static fields
uint aser::CosplayAccounts::budget = DEFAULT_BUDGET;
vector<CosplayAccounts> aser::CosplayAccounts::cosplayAccounts;
// used for filtering
set<string> aser::CosplayAccounts::apiSigs;

void aser::CosplayAccounts::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

string aser::CosplayAccounts::getErrorMsg(SVE::Type type) {
    string msg;
    switch (type) {
        case SVE::Type::COSPLAY_FULL:
            msg = "These two data types have the same layout:\n";
            break;
        case SVE::Type::COSPLAY_PARTIAL:
            msg = "These two data types are partially compatible:\n";
            break;
        case SVE::Type::ACCOUNT_DUPLICATE:
            msg = "The data type may contain duplicated mutable accounts:";
            break;
        case SVE::Type::PDA_SEEDS_COLLISIONS:
            msg = "These two PDA accounts may have the same seeds, which may lead to PDA collisions:";
            break;
        case SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ORDER:
            msg =
                "These two accounts are reordered in the instruction and may break the ABI of the deployed on-chain "
                "program, according to the IDL available on Anchor:";
            break;
        default:
            assert(false && "unhandled CosplayAccounts");
            break;
    }
    return msg;
}

// we report at most 1 UntrustfulAccount bug for each function call
bool aser::CosplayAccounts::filter(SourceInfo &srcInfo) {
    if (apiSigs.find(srcInfo.sig()) != apiSigs.end()) {
        return true;
    }
#pragma omp critical(apiSigs)
    { apiSigs.insert(srcInfo.sig()); }
    return false;
}

void aser::CosplayAccounts::collect(const Event *e1, const Event *e2, map<TID, vector<CallEvent *>> &callEventTraces,
                                    SVE::Type type, int P) {
    SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
    SourceInfo srcInfo2 = getSourceLoc(e2->getInst());

    std::string sig = getRaceSrcSig(srcInfo1, srcInfo2);
    if (apiSigs.find(sig) != apiSigs.end()) {
        return;
    } else {
        apiSigs.insert(sig);
    }

    if (customizedFilterIgnoreFunctions(e1, e2, IGNORED_FUN_ALL)) return;
    bool isIgnored = false;
    bool isHidden = false;
    if (SVE::isCheckerDisabled(type)) isHidden = true;
    if (customizedFilterIgnoreLocations(e1, e2, IGNORED_LOCATION_ALL)) isIgnored = true;

    // std::vector<std::string> st = getStackTrace(e, callEventTraces, srcInfo.isCpp());
    std::vector<std::string> st1;
    std::vector<std::string> st2;

    TID tid1 = e1->getTID();
    TID tid2 = e1->getTID();
    EventID id1 = e1->getID();
    EventID id2 = e2->getID();

    auto &callTrace1 = callEventTraces[tid1];
    for (CallEvent *ce : callTrace1) {
        if (ce->getID() > id1) break;
        if (ce->getEndID() == 0 || ce->getEndID() > id1) {
            st1.push_back(ce->getCallSiteString(true));
        }
    }
    auto &callTrace2 = callEventTraces[tid2];
    for (CallEvent *ce : callTrace2) {
        if (ce->getID() > id2) break;
        if (ce->getEndID() == 0 || ce->getEndID() > id2) {
            st2.push_back(ce->getCallSiteString(true));
        }
    }
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            srcInfo1.setStackTrace(st1);
            srcInfo2.setStackTrace(st1);
            auto msg = getErrorMsg(type);  // TODO: add source location
            if (srcInfo1.getLine() < srcInfo2.getLine())
                cosplayAccounts.emplace_back(srcInfo1, srcInfo2, msg, type, P, isIgnored, isHidden);
            else
                cosplayAccounts.emplace_back(srcInfo2, srcInfo1, msg, type, P, isIgnored, isHidden);
            --budget;
            if (PRINT_IMMEDIATELY) cosplayAccounts.back().print();
            // intentionally commented out since UntrustfulAccount needs improvement
            // if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}

aser::CosplayAccounts::CosplayAccounts(SourceInfo &srcInfo1, SourceInfo &srcInfo2, std::string msg, SVE::Type t, int P,
                                       bool isIgnored, bool isHidden)
    : apiInst1(srcInfo1), apiInst2(srcInfo2), errorMsg(msg), type(t), p(P), ignore(isIgnored), hide(isHidden) {
    id = SVE::getTypeID(t);
    name = SOLANA_SVE_DB[id]["name"];
    description = SOLANA_SVE_DB[id]["description"];
    url = SOLANA_SVE_DB[id]["url"];
}

json aser::CosplayAccounts::to_json() {
    json j({{"priority", p},
            {"inst1", apiInst1},
            {"inst2", apiInst2},
            {"errorMsg", errorMsg},
            {"id", id},
            {"hide", hide},
            {"ignore", ignore},
            {"description", description},
            {"url", url}});
    return j;
}
void aser::CosplayAccounts::print() {
    // llvm::errs() << "==============VULNERABLE: Possible Accounts Cosplay Attacks!============\n";
    // outs() << errorMsg;
    // outs() << " Data Type1 defined at line " << apiInst1.getLine() << ", column " << apiInst1.getCol() << " in "
    //        << apiInst1.getFilename() << "\n";
    // outs() << apiInst1.getSnippet();
    // outs() << ">>>Stack Trace:\n";
    // printStackTrace(apiInst1.getStackTrace());
    // outs() << " Data Type2 defined at line " << apiInst2.getLine() << ", column " << apiInst2.getCol() << " in "
    //        << apiInst2.getFilename() << "\n";
    // outs() << apiInst2.getSnippet();
    // outs() << ">>>Stack Trace:\n";
    // printStackTrace(apiInst2.getStackTrace());
    // outs() << "\n";
    outs() << "ignored: " << ignore << "\n";
    llvm::errs() << "==============VULNERABLE: " << name << "!============\n";
    outs() << description << ":\n";
    auto desc_type = "Data type";
    if (type == SVE::Type::PDA_SEEDS_COLLISIONS)
        desc_type = "PDA account";
    else if (type == SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ORDER)
        desc_type = "Account";
    outs() << " " << desc_type << "1 defined at line " << apiInst1.getLine() << ", column " << apiInst1.getCol()
           << " in " << apiInst1.getFilename() << "\n";
    outs() << apiInst1.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(apiInst1.getStackTrace());
    outs() << " " << desc_type << "2 defined at line " << apiInst2.getLine() << ", column " << apiInst2.getCol()
           << " in " << apiInst2.getFilename() << "\n";
    outs() << apiInst2.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(apiInst2.getStackTrace());
    outs() << "\n";
    outs() << "For more info, see " << url << "\n\n\n";
}
void aser::CosplayAccounts::printAll() {
    std::sort(cosplayAccounts.begin(), cosplayAccounts.end());
    for (auto r : cosplayAccounts) {
        r.print();
    }
}

void aser::CosplayAccounts::printSummary() {
    info("detected " + to_string(cosplayAccounts.size()) + " accounts cosplay issues in total.");
}

/* --------------------------------

           UntrustfulAccount

----------------------------------- */

// static fields
uint aser::UntrustfulAccount::budget = DEFAULT_BUDGET;
vector<UntrustfulAccount> aser::UntrustfulAccount::untrustfulAccounts;
// used for filtering
// std::map<SVE::Type, set<const llvm::Value *>> aser::UntrustfulAccount::apiSigsMap;
std::set<const llvm::Value *> aser::UntrustfulAccount::apiSigs;
std::set<std::string> aser::UntrustfulAccount::cpiSigs;

std::set<std::vector<std::string>> aser::UntrustfulAccount::callStackSigs;

void aser::UntrustfulAccount::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

string aser::UntrustfulAccount::getErrorMsg(SVE::Type type) {
    string msg;
    switch (type) {
        case SVE::Type::ACCOUNT_UNVALIDATED_BORROWED:
            msg = "The account is not validated before parsing its data:";
            break;
        case SVE::Type::ACCOUNT_UNVALIDATED_OTHER:
            msg = "The account is not properly validated and may be untrustful:";
            // msg = "The account info is not trustful:\n";
            break;
        case SVE::Type::ACCOUNT_UNVALIDATED_PDA:
            msg = "The PDA account is not properly validated and may be untrustful:";
            break;
        case SVE::Type::ACCOUNT_UNVALIDATED_DESTINATION:
            msg =
                "The account is used as destination in token transfer without validation and it could be the same as "
                "the transfer source account:";
            break;
        case SVE::Type::ACCOUNT_INCORRECT_AUTHORITY:
            msg =
                "The PDA account may be incorrectly used as a shared authority and may allow any account to transfer "
                "or burn tokens:";
            break;
        case SVE::Type::INSECURE_INIT_IF_NEEDED:
            msg = "The `init_if_needed` keyword in anchor-lang prior to v0.24.x has a critical security bug:";
            break;
        case SVE::Type::MISS_OWNER:
            msg = "The account info is missing owner check:";
            break;
        case SVE::Type::MISS_SIGNER:
            msg = "The account info is missing signer check:";
            break;
        case SVE::Type::MISS_CPI_RELOAD:
            msg = "The token account is missing reload after CPI:";
            break;
        case SVE::Type::MISS_ACCESS_CONTROL_UNSTAKE:
            msg = "The unstake instruction may be missing an access_control account validation:";
            break;
        case SVE::Type::ARBITRARY_CPI:
            msg = "The CPI may invoke an arbitrary program:";
            break;
        case SVE::Type::INSECURE_SPL_TOKEN_CPI:
            msg = "The spl_token account may be arbitrary:";
            break;
        case SVE::Type::INSECURE_ACCOUNT_REALLOC:
            msg = "The account realloc in solana_program prior to v1.10.29 may cause programs to malfunction:";
            break;
        case SVE::Type::INSECURE_ASSOCIATED_TOKEN:
            msg = "The associated token account may be faked:";
            break;
        case SVE::Type::MALICIOUS_SIMULATION:
            msg = "The program may be malicious:";
            break;
        case SVE::Type::UNSAFE_SYSVAR_API:
            msg = "The sysvar::instructions API is unsafe and deprecated:";
            break;
        case SVE::Type::DIV_BY_ZERO:
            msg = "The arithmetic operation may result in a div-by-zero error:";
            break;
        case SVE::Type::REINIT:
            msg = "The account is vulnerable to program re-initialization:";
            break;
        case SVE::Type::BUMP_SEED:
            msg = "The account's bump seed is not validated and may be vulnerable to seed canonicalization attacks:";
            break;
        case SVE::Type::INSECURE_PDA_SHARING:
            msg = "The PDA sharing with these seeds is insecure:";
            break;
        case SVE::Type::ACCOUNT_CLOSE:
            msg = "The account closing is insecure:";
            break;
        default:
            assert(false && "unhandled untrustful account:");
            break;
    }
    return msg;
}

// we report at most 1 UntrustfulAccount bug for each function call
bool aser::UntrustfulAccount::filter(SVE::Type type, SourceInfo &srcInfo) {
    // for CPI
    if (SVE::Type::ARBITRARY_CPI == type || SVE::Type::ACCOUNT_CLOSE == type) {
        auto sig = srcInfo.sig();
        if (cpiSigs.find(sig) != cpiSigs.end()) {
            // llvm::outs() << "filter true:" << srcInfo.sig() << "\n";
            return true;
        } else {
            cpiSigs.insert(sig);
            // llvm::outs() << "filter false:" << srcInfo.sig() << "\n";
            return false;
        }
    }
    const llvm::Value *v = srcInfo.getValue();
    if (apiSigs.find(v) != apiSigs.end()) {
        // llvm::outs() << "filter true:" << srcInfo.sig() << "\n";
        return true;
    } else {
        apiSigs.insert(v);
        // llvm::outs() << "filter false:" << srcInfo.sig() << "\n";
        return false;
    }
}

// we report at most 1 UntrustfulAccount bug for each call stack
bool aser::UntrustfulAccount::filterByCallStack(std::vector<std::string> &st0) {
    auto st = st0;
    // this one is special: keep only the last few entries
    if (st0.size() > 1) {
        std::vector<std::string> vect(st0.begin() + st0.size() - 1, st0.end());
        st = vect;
    }

    if (callStackSigs.find(st) != callStackSigs.end()) {
        return true;
    }
#pragma omp critical(callStackSigs)
    { callStackSigs.insert(st); }
    return false;
}

void aser::UntrustfulAccount::collect(llvm::StringRef accountName, const Event *e,
                                      map<TID, vector<CallEvent *>> &callEventTraces, SVE::Type type, int P) {
    SourceInfo srcInfo = getSourceLoc(e->getInst());
    if (filter(type, srcInfo)) return;

    if (SVE::Type::MISS_SIGNER == type) {
        // SKIP PDA is_signer
        if (getSourceLinesForSoteria(srcInfo, 1).find(" PDA") != std::string::npos ||
            getSourceLinesForSoteria(srcInfo, 2).find(" PDA") != std::string::npos)
            return;
    }

    if (customizedFilterIgnoreFunctions(e, e, IGNORED_FUN_ALL)) return;
    // for Anchor accounts, _ is ignored by default
    bool isIgnored = accountName.startswith("_");
    bool isHidden = false;
    if (accountName.contains("_no_check")) isIgnored = true;  // skip no_check
    if (SVE::isCheckerDisabled(type)) isHidden = true;
    if (customizedFilterIgnoreLocations(e, e, IGNORED_LOCATION_ALL)) isIgnored = true;

    if (SVE::Type::ACCOUNT_UNVALIDATED_DESTINATION == type) {
        bool isDestinationIgnored = customizedFilterSoteriaIgnoreSymbol(e, "dest");
        if (isDestinationIgnored) isIgnored = true;
    } else if (SVE::Type::MISS_SIGNER == type) {
        bool isSignerIgnored = customizedFilterSoteriaIgnoreSymbol(e, "signer");
        if (isSignerIgnored) isIgnored = true;
    } else if (SVE::Type::ACCOUNT_UNVALIDATED_OTHER == type) {
        bool isUntrustfulIgnored = customizedFilterSoteriaIgnoreSymbol(e, "untrust");
        if (isUntrustfulIgnored) isIgnored = true;
    }
    // std::vector<std::string> st = getStackTrace(e, callEventTraces, srcInfo.isCpp());
    std::vector<std::string> st;
    TID tid = e->getTID();
    EventID id = e->getID();

    auto &callTrace = callEventTraces[tid];
    std::string last_str = "";
    for (CallEvent *ce : callTrace) {
        if (ce->getID() > id) break;
        if (ce->getEndID() == 0 || ce->getEndID() > id) {
            auto call_str = ce->getCallSiteString(true);
            if (last_str != call_str) st.push_back(call_str);
            last_str = call_str;
        }
    }
    // st.erase(st.begin(), st.begin() + 2);
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            srcInfo.setStackTrace(st);
            auto msg = getErrorMsg(type);  // TODO: add source location
            untrustfulAccounts.emplace_back(accountName.str(), srcInfo, msg, type, P, isIgnored, isHidden);
            --budget;
            if (PRINT_IMMEDIATELY) untrustfulAccounts.back().print();
            // intentionally commented out since UntrustfulAccount needs improvement
            // if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}

aser::UntrustfulAccount::UntrustfulAccount(std::string account, SourceInfo &srcInfo, std::string msg, SVE::Type t,
                                           int P, bool isIgnored, bool isHidden)
    : apiInst(srcInfo), errorMsg(msg), type(t), accountName(account), p(P), ignore(isIgnored), hide(isHidden) {
    id = SVE::getTypeID(t);
    name = SOLANA_SVE_DB[id]["name"];
    description = SOLANA_SVE_DB[id]["description"];
    url = SOLANA_SVE_DB[id]["url"];
}

json aser::UntrustfulAccount::to_json() {
    json j({{"priority", p},
            {"account", accountName},
            {"inst", apiInst},
            {"errorMsg", errorMsg},
            {"id", id},
            {"hide", hide},
            {"ignore", ignore},
            {"description", description},
            {"url", url}});
    return j;
}
void aser::UntrustfulAccount::print() {
    outs() << "ignored: " << ignore << "\n";
    // llvm::outs() << "=============This account may be UNTRUSTFUL!================\n";
    llvm::errs() << "==============VULNERABLE: " << name << "!============\n";
    outs() << "Found a potential vulnerability at line " << apiInst.getLine() << ", column " << apiInst.getCol()
           << " in " << apiInst.getFilename() << "\n";
    // outs() << errorMsg << "\n";
    outs() << description << ":\n";
    outs() << apiInst.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(apiInst.getStackTrace());
    outs() << "\n";
    outs() << "For more info, see " << url << "\n\n\n";
}
void aser::UntrustfulAccount::printAll() {
    std::sort(untrustfulAccounts.begin(), untrustfulAccounts.end());
    for (auto r : untrustfulAccounts) {
        r.print();
    }
}

void aser::UntrustfulAccount::printSummary() {
    info("detected " + to_string(untrustfulAccounts.size()) + " untrustful accounts in total.");
}

/* --------------------------------

           UnSafeOperation

----------------------------------- */

// static fields
uint aser::UnSafeOperation::budget = DEFAULT_BUDGET;
vector<UnSafeOperation> aser::UnSafeOperation::unsafeOperations;
// used for filtering
set<string> aser::UnSafeOperation::apiSigs;
std::set<std::vector<std::string>> aser::UnSafeOperation::callStackSigs;

void aser::UnSafeOperation::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

string aser::UnSafeOperation::getErrorMsg(SVE::Type type) {
    string msg;
    switch (type) {
        case SVE::Type::OVERFLOW_ADD:
            msg = "The add operation may result in overflows:\n";
            break;
        case SVE::Type::OVERFLOW_SUB:
            msg = "The sub operation may result in underflows:\n";
            break;
        case SVE::Type::OVERFLOW_MUL:
            msg = "The mul operation may result in overflows:\n";
            break;
        case SVE::Type::OVERFLOW_DIV:
            msg = "The div operation may result in divide-by-zero errors or overflows:\n";
            break;
        case SVE::Type::INCORRECT_BREAK_LOGIC:
            msg = "Loop break instead of continue (jet-v1 exploit):\n";
            break;
        case SVE::Type::BIDIRECTIONAL_ROUNDING:
            msg = "The arithmetics here may suffer from inconsistent rounding errors:\n";
            break;
        case SVE::Type::CAST_TRUNCATE:
            msg = "The cast operation here may lose precision due to truncation:\n";
            break;
        case SVE::Type::CRITICAL_REDUNDANT_CODE:
            msg = "The code may be redundant or unused, but appears critical:";
            break;
        case SVE::Type::ORDER_RACE_CONDITION:
            msg =
                "The instruction may suffer from a race condition between order cancellation and order recreation by "
                "an attacker:";
            break;
        case SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ADD:
            msg =
                "The account may break the ABI of the deployed on-chain program as it does not exist in the IDL "
                "available on Anchor:";
            break;
        case SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_MUT:
            msg =
                "The mutable account may break the ABI of the deployed on-chain program as it is immutable according "
                "to the IDL available on Anchor:";
            break;
        case SVE::Type::REENTRANCY_ETHER:
            msg =
                "The function may suffer from reentrancy attacks due to the use of call.value, which can invoke an "
                "external contract's fallback function:";
            break;
        case SVE::Type::ARBITRARY_SEND_ERC20:
            msg =
                "The function may allow an attacker to send from an arbitrary address, instead of from the msg.sender:";
            break;
        case SVE::Type::SUISIDE_SELFDESTRUCT:
            msg = "The function may allow an attacker to destruct the contract:";
            break;
        case SVE::Type::MISS_INIT_UNIQUE_ADMIN_CHECK:
            msg =
                "The init function misses checking admin uniqueness and may allow an attacker to call the init "
                "function more than once:";
            break;
        case SVE::Type::BIT_SHIFT_OVERFLOW:
            msg = "The bit shift operation may result in overflows:";
            break;
        case SVE::Type::DIV_PRECISION_LOSS:
            msg = "The division operation here may lose precision:\n";
            break;
        case SVE::Type::VULNERABLE_SIGNED_INTEGER_I128:
            msg = "The I128 signed integer implementation in Move may be vulnerable and is not recommended:\n";
            break;
        case SVE::Type::INCORRECT_TOKEN_CALCULATION:
            msg =
                "The token amount calculation may be incorrect. Consider using the reserves instead of the balances:\n";
            break;

        default:
            assert(false && "Unhandled UnSafeOperation");
            break;
    }
    return msg;
}

bool aser::UnSafeOperation::filter(SourceInfo &srcInfo) {
    if (apiSigs.find(srcInfo.sig()) != apiSigs.end()) {
        return true;
    }
#pragma omp critical(apiSigs)
    { apiSigs.insert(srcInfo.sig()); }
    return false;
}

bool aser::UnSafeOperation::filterByCallStack(std::vector<std::string> &st0) {
    auto st = st0;
    // this one is special: keep only the last few entries
    if (st0.size() > 1) {
        std::vector<std::string> vect(st0.begin() + st0.size() - 1, st0.end());
        st = vect;
    }

    if (callStackSigs.find(st) != callStackSigs.end()) {
        return true;
    }
#pragma omp critical(callStackSigs)
    { callStackSigs.insert(st); }
    return false;
}

extern bool hasOverFlowChecks;
void aser::UnSafeOperation::collect(const Event *e, map<TID, vector<CallEvent *>> &callEventTraces, SVE::Type type,
                                    int P) {
    if (hasOverFlowChecks && type != SVE::Type::CAST_TRUNCATE) return;

    SourceInfo srcInfo = getSourceLoc(e->getInst());
    if (filter(srcInfo)) return;
    if (customizedFilterIgnoreFunctions(e, e, IGNORED_FUN_ALL)) return;
    bool isHidden = false;
    bool isIgnored = false;
    if (SVE::isCheckerDisabled(type)) isHidden = true;
    if (customizedFilterIgnoreLocations(e, e, IGNORED_LOCATION_ALL)) isIgnored = true;

    if (type == SVE::Type::CRITICAL_REDUNDANT_CODE) {
        bool isRedundantIgnored = customizedFilterSoteriaIgnoreSymbol(e, "redundant");
        if (isRedundantIgnored) isIgnored = true;
    }

    // std::vector<std::string> st = getStackTrace(e, callEventTraces, srcInfo.isCpp());
    std::vector<std::string> st;
    TID tid = e->getTID();
    EventID id = e->getID();

    auto &callTrace = callEventTraces[tid];
    std::string last_str = "";
    for (CallEvent *ce : callTrace) {
        if (ce->getID() > id) break;
        if (ce->getEndID() == 0 || ce->getEndID() > id) {
            auto call_str = ce->getCallSiteString(true);
            if (last_str != call_str) st.push_back(call_str);
            last_str = call_str;
        }
    }
    // st.erase(st.begin(), st.begin() + 2);
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            srcInfo.setStackTrace(st);
            auto msg = getErrorMsg(type);  // TODO: add source location
            unsafeOperations.emplace_back(srcInfo, msg, type, P, isIgnored, isHidden);
            --budget;
            if (PRINT_IMMEDIATELY) unsafeOperations.back().print();
            // intentionally commented out since unsafeOperations needs improvement
            // if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}

aser::UnSafeOperation::UnSafeOperation(SourceInfo &srcInfo, std::string msg, SVE::Type t, int P, bool isIgnored,
                                       bool isHidden)
    : apiInst(srcInfo), errorMsg(msg), type(t), p(P), ignore(isIgnored), hide(isHidden) {
    id = SVE::getTypeID(t);
    name = SOLANA_SVE_DB[id]["name"];
    description = SOLANA_SVE_DB[id]["description"];
    url = SOLANA_SVE_DB[id]["url"];
}

json aser::UnSafeOperation::to_json() {
    json j({{"priority", p},
            {"inst", apiInst},
            {"errorMsg", errorMsg},
            {"id", id},
            {"hide", hide},
            {"ignore", ignore},
            {"description", description},
            {"url", url}});
    return j;
}
void aser::UnSafeOperation::print() {
    // llvm::outs() << "=============This arithmetic operation may be UNSAFE!================\n";
    // outs() << "Found a potential vulnerability at line " << apiInst.getLine() << ", column " << apiInst.getCol()
    //        << " in " << apiInst.getFilename() << "\n";
    // outs() << errorMsg << "\n";
    // outs() << apiInst.getSnippet();
    // outs() << ">>>Stack Trace:\n";
    // printStackTrace(apiInst.getStackTrace());
    // outs() << "\n";
    outs() << "ignored: " << ignore << "\n";
    llvm::errs() << "==============VULNERABLE: " << name << "!============\n";
    outs() << "Found a potential vulnerability at line " << apiInst.getLine() << ", column " << apiInst.getCol()
           << " in " << apiInst.getFilename() << "\n";
    // outs() << errorMsg << "\n";
    outs() << description << ":\n";
    outs() << apiInst.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(apiInst.getStackTrace());
    outs() << "\n";
    outs() << "For more info, see " << url << "\n\n\n";
}
void aser::UnSafeOperation::printAll() {
    std::sort(unsafeOperations.begin(), unsafeOperations.end());
    for (auto r : unsafeOperations) {
        r.print();
    }
}

void aser::UnSafeOperation::printSummary() {
    info("detected " + to_string(unsafeOperations.size()) + " unsafe operations in total.");
}

/* --------------------------------

            Data Race

----------------------------------- */

// DateRace related static fields
vector<TOCTOU> aser::TOCTOU::races;
// race signatures: based on source code information
set<std::string> aser::TOCTOU::raceSigs;
set<std::string> aser::TOCTOU::rawLineSigs;
set<std::string> aser::TOCTOU::methodPairs;

uint aser::TOCTOU::budget = DEFAULT_BUDGET;
uint aser::TOCTOU::omp_budget = DEFAULT_BUDGET;

aser::TOCTOU::TOCTOU(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, bool isOmpRace, int P)
    : p(P),
      access1(srcInfo1),
      access2(srcInfo2),
      objInfo(objInfo),
      objName(objInfo.getName()),
      objField(objInfo.getAccessPath()),
      objLine(objInfo.getLine()),
      objDir(objInfo.getDir()),
      objFilename(objInfo.getFilename()),
      objSrcLine(objInfo.getSourceLine()),
      isOmpRace(isOmpRace) {}

aser::TOCTOU::TOCTOU(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, int P)
    : TOCTOU(srcInfo1, srcInfo2, objInfo, false, P) {}

aser::TOCTOU::TOCTOU(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, const string &ompFilename,
                     const string &ompDir, const string &ompSnippet, const vector<string> &callingCtx, int P)
    : TOCTOU(srcInfo1, srcInfo2, objInfo, true, P) {
    this->ompFilename = ompFilename;
    this->ompDir = ompDir;
    this->ompSnippet = ompSnippet;
    this->callingCtx = callingCtx;
}

void aser::TOCTOU::print() {
    outs() << "\n==== Found a TOCTOU Vulnerability: \n"
           << "line " << access1.getLine() << ", column " << access1.getCol() << " in " << access1.getFilename()
           << " AND "
           << "line " << access2.getLine() << ", column " << access2.getCol() << " in " << access2.getFilename()
           << "\n";
    if (objInfo.isGlobalValue())
        error("Static variable: ");
    else
        error("Shared variable: ");
    info(objName, false);
    outs() << " at line " << objLine << " of " << objFilename << "\n";
    outs() << objSrcLine;
    if (!objField.empty()) {
        info("shared field: ", false);
        info(objField);
    }

    if (access1.isWrite())
        highlight("Thread 1 (write): ");
    else
        highlight("Thread 1 (read): ");
    outs() << access1.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(access1.getStackTrace());

    if (access2.isWrite())
        highlight("Thread 2 (write): ");
    else
        highlight("Thread 2 (read): ");
    outs() << access2.getSnippet();
    outs() << ">>>Stack Trace:\n";
    printStackTrace(access2.getStackTrace());

    if (isOmpRace) {
        info("The OpenMP region this bug occurs:", true);
        if (ompFilename.size() > 0 && ompFilename.front() == '/')
            outs() << ompFilename << "\n";
        else
            outs() << ompDir + "/" + ompFilename << "\n";
        outs() << ompSnippet;
        info("Gets called from:", true);
        printStackTrace(callingCtx);
    }
}

json TOCTOU::to_json() {
    json sharedObj;
    sharedObj["line"] = objLine;
    sharedObj["name"] = objName;
    sharedObj["field"] = objField;
    sharedObj["filename"] = objFilename;
    sharedObj["dir"] = objDir;
    sharedObj["sourceLine"] = objSrcLine;
    json race;
    race["priority"] = p;
    race["access1"] = access1;
    race["access2"] = access2;
    race["sharedObj"] = sharedObj;
    race["isOmpRace"] = isOmpRace;
    if (isOmpRace) {
        json ompInfo;
        ompInfo["filename"] = ompFilename;
        ompInfo["dir"] = ompDir;
        ompInfo["snippet"] = ompSnippet;
        ompInfo["callingCtx"] = callingCtx;
        race["ompInfo"] = ompInfo;
    }
    return race;
}

void aser::TOCTOU::outputJSON() {
    json rs;
    std::string path;
    if (!ConfigOutputPath.empty()) {
        path = ConfigOutputPath;
    } else {
        info("writing detection results to ./races.json");
        path = "races.json";
    }

    std::vector<json> raceJsons;
    for (auto &r : races) {
        raceJsons.emplace_back(r.to_json());
    }

    std::ofstream output(path, std::ofstream::out);
    rs["races"] = raceJsons;
    rs["version"] = 1;
    rs["generatedAt"] = getCurrentTimeStr();
    rs["bcfile"] = TargetModulePath.getValue();
    output << rs;
    output.close();
}

void aser::TOCTOU::init(int configReportLimit, bool configNoReportLimit) {
    if (configReportLimit != -1) {
        omp_budget = configReportLimit;
        budget = configReportLimit;
    }
    nolimit = configNoReportLimit;
}

bool aser::TOCTOU::filter(Event *e1, Event *e2, const ObjTy *obj, SourceInfo &srcInfo1, SourceInfo &srcInfo2,
                          SourceInfo &sharedObj) {
    // All filtering schemes under this branch
    // are heuristic-based (unsound)

    // LOG_DEBUG("debug filter for races. e1={}, e2={}", srcInfo1.sig(), srcInfo2.sig());

    if (!CONFIG_NO_FILTER &&  // do not filter double-free races
        (e1->getType() != EventType::APIWrite && e2->getType() != EventType::APIWrite)) {
        // FIXME: the two if-else filtering below will fail some DRB cases

        // TODO: the following seems to not work with cr_main - need to double check

        // if (srcInfo1.getFilename().empty() || srcInfo2.getFilename().empty()) {
        //     LOG_DEBUG("Skipping race because the source file info cannot be found. e1={}, e2={}", *e1->getInst(),
        //               *e2->getInst());
        //     return true;
        // }
        if (srcInfo1.getSourceLine().empty() && srcInfo2.getSourceLine().empty()) {
            LOG_DEBUG("Skipping race because source line info cannot be found. e1={}, e2={}", srcInfo1.sig(),
                      srcInfo2.sig());
            return true;
        }
        // if (sharedObj.getLine() == 0) {
        //     LOG_DEBUG("Skipping race because the shared object info cannot be found. e1={}, e2={}", *e1->getInst(),
        //               *e2->getInst());
        //     return true;
        // }
        {
            auto field = obj->getFieldAccessPath();
            auto sourceLine1 = srcInfo1.getSourceLine();
            auto sourceLine2 = srcInfo1.getSourceLine();
            if (!field.empty() && !sourceLine1.empty() && !sourceLine2.empty()) {
                std::string delimiter = "->";
                field.erase(0, field.find_last_of(delimiter) + delimiter.length());
                if (sourceLine1.find(field) == string::npos || sourceLine2.find(field) == string::npos) {
                    LOG_DEBUG(
                        "Skipping potential race because shared variable does not appear in source lines. e1={}, e2={}",
                        *e1->getInst(), *e2->getInst());
                    return true;
                }
            }
        }
        auto msig = getRaceMethodSig(e1, e2, obj);

        if (methodPairs.find(msig) == methodPairs.end()) {
#pragma omp critical(methodPairs)
            { methodPairs.insert(msig); }
        } else {
            return true;
        }

        // if the source line is the same, but they reside in different file
        // we only report one
        std::string rawLineSig = getRaceRawLineSig(srcInfo1, srcInfo2);
        if (filterStrPattern(rawLineSig)) {
            return true;
        }
        if (!rawLineSig.empty()) {
            if (rawLineSigs.find(rawLineSig) == rawLineSigs.end()) {
#pragma omp critical(rawLineSigs)
                { rawLineSigs.insert(rawLineSig); }
            } else {
                return true;
            }
        }
    }

    // this is not a heuristic based filtering
    // and will only filter out identical races (due to our race detection algorithm)
    // therefore this should always be executed
    // (although this should have no effect with filtering turned on)
    std::string sig = getRaceSrcSig(srcInfo1, srcInfo2);
    if (raceSigs.find(sig) != raceSigs.end()) {
        return true;
    }
#pragma omp critical(raceSigs)
    { raceSigs.insert(sig); }
    return false;
}

// collect race from raw data structure into JSON
// NOTE: also include filtering and terminal print-out
// NOTE: this is the public API to use
void aser::TOCTOU::collect(Event *e1, Event *e2, const ObjTy *obj,
                           std::map<TID, std::vector<CallEvent *>> &callEventTraces, int P) {
    SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
    SourceInfo srcInfo2 = getSourceLoc(e2->getInst());
    SourceInfo sharedObjLoc = getSourceLoc(obj->getValue());
    if (sharedObjLoc.getLine() == 0) {
        if (!obj->isAnonObj()) {
            LOG_DEBUG("Race shared object line 0: value={}", *obj->getValue());
            // llvm::outs() <<"Shared object line 0: "<<*obj->getValue()<<sharedObjLoc.str()<<"\n";
            // Function Attrs: nounwind uwtable
            // define noalias %struct.ProcedureCtx.8707* @Proc_LabelsCtx() #2 !dbg !1179091 {
            // return;  // JEFF: skip likely false positives
        } else {
            Function *openAPI = getOpenLibAPIFor(obj);
            if (openAPI != nullptr) {
                sharedObjLoc = getSourceLoc(openAPI);
            }
        }
    }

    // if the race get filterd out, no need to collect this race
    if (filter(e1, e2, obj, srcInfo1, srcInfo2, sharedObjLoc)) return;

    auto st1 = getStackTrace(e1, callEventTraces, srcInfo1.isCpp());
    auto st2 = getStackTrace(e2, callEventTraces, srcInfo2.isCpp());

    // for (auto &str : st1) LOG_DEBUG("TOCTOU NotParallelFunctionPairs: str1={}", str);
    // for (auto &str : st2) LOG_DEBUG("TOCTOU NotParallelFunctionPairs: str2={}", str);

    // user customized filtering (by pattern)
    if (isNotParallelFunctions(st1, st2, NONE_PARALLEL_FUNCTIONS)) return;

    if (customizedFilterIgnoreFunctions(e1, e2, IGNORED_FUN_ALL)) return;
    if (customizedFilterIgnoreVariables(sharedObjLoc.getName(), IGNORED_VAR_ALL)) return;
    if (customizedFilterIgnoreLocations(e1, e2, IGNORED_LOCATION_ALL)) return;
    P = customizedPriorityAdjust(P, sharedObjLoc.getName(), srcInfo1, srcInfo2, st1, st2, LOW_PRIORITY_FILE_NAMES,
                                 HIGH_PRIORITY_FILE_NAMES, LOW_PRIORITY_VAR_NAMES, HIGH_PRIORITY_VAR_NAMES);
    cleanCallStackTrace(st1, st2);
#pragma omp critical(budget)
    {
        if (nolimit || budget > 0) {
            st1 = getCallStackUntilMain(e1, callEventTraces, srcInfo1.isCpp());
            st2 = getCallStackUntilMain(e2, callEventTraces, srcInfo2.isCpp());
            srcInfo1.setStackTrace(st1);
            srcInfo2.setStackTrace(st2);
            sharedObjLoc.setAccessPath(obj->getFieldAccessPath());
            if (sharedObjLoc.isGlobalValue()) P++;  // P = Priority::LEVEL6;

            if (e1->getType() == EventType::Write || e1->getType() == EventType::APIWrite) srcInfo1.setWrite();
            if (e2->getType() == EventType::Write || e2->getType() == EventType::APIWrite) srcInfo2.setWrite();

            races.emplace_back(srcInfo1, srcInfo2, sharedObjLoc, P);
            --budget;
            if (PRINT_IMMEDIATELY) races.back().print();
            if (TERMINATE_IMMEDIATELY) exit(1);
        }
    }
}

void aser::TOCTOU::printAll() {
    // TODO: also consider Priority
    // TODO: should there also be an order between openmp races vs. normal races?
    std::sort(races.begin(), races.end());
    for (auto r : races) {
        r.print();
    }
}

void aser::TOCTOU::printSummary() { info("detected " + to_string(races.size()) + " TOCTOU in total."); }

/* --------------------------------

                Utils

----------------------------------- */
extern llvm::cl::opt<std::string> ConfigOutputPath;
void outputRaceList() {
    std::string path = "cr_racelist";
    if (!ConfigOutputPath.empty()) {
        std::size_t pos = ConfigOutputPath.find_last_of("/");
        if (pos != std::string::npos) path = ConfigOutputPath.substr(0, pos + 1) + path;
    }

    if (RacyLocationPairs.size() > 0) {
        info("saving race pairs to " + path);
        std::ofstream output(path, std::ofstream::out);
        for (auto &r : RacyLocationPairs) {
            output << r << "\n";
        }
        output << "#total_races:" << RacyLocationPairs.size() << "\n";
        output.close();
    }
}

void outputIncludeList() {
    std::string path = "cr_includelist";
    if (!ConfigOutputPath.empty()) {
        std::size_t pos = ConfigOutputPath.find_last_of("/");
        if (pos != std::string::npos) path = ConfigOutputPath.substr(0, pos + 1) + path;
    }

    info("saving racy locations to " + path);
    std::ofstream output(path, std::ofstream::out);

    // if(RacyLocations.size()>0)
    {
        // adding racy  functions to includelist
        for (auto f : CR_RacyFunctions) output << "func:" << f->getName().str() << "\n";
        output << "#total_racy_functions:" << CR_RacyFunctions.size() << "\n";

        // adding unexplored function to includelist
        for (auto f : CR_UNExploredFunctions) output << "func:" << f->getName().str() << "\n";
        output << "#total_unexplored_functions:" << CR_UNExploredFunctions.size() << "\n";

        for (auto &r : RacyLocations) {
            output << "loc:" << r << "\n";
        }
        output << "#total_locations:" << RacyLocations.size() << "\n";
    }

    output.close();
}
void outputIgnoreList() {
    std::string path = "cr_ignorelist";
    if (!ConfigOutputPath.empty()) {
        std::size_t pos = ConfigOutputPath.find_last_of("/");
        if (pos != std::string::npos) path = ConfigOutputPath.substr(0, pos + 1) + path;
    }

    if (IgnoreRacyLocations.size() > 0) {
        info("saving non-racy locations to " + path);
        std::ofstream output(path, std::ofstream::out);
        for (auto &r : IgnoreRacyLocations) {
            output << r << "\n";
        }
        output << "#total:" << IgnoreRacyLocations.size() << "\n";
        output.close();
    }
}
void aser::ignoreRaceLocations(Event *e1, Event *e2) {
    SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
    SourceInfo srcInfo2 = getSourceLoc(e2->getInst());
    auto loc1 = e1->getInst()->getFunction()->getName().str() + "@" + std::to_string(srcInfo1.getLine());
    auto loc2 = e2->getInst()->getFunction()->getName().str() + "@" + std::to_string(srcInfo2.getLine());
#pragma omp critical(IgnoreRacyLocations)
    {
        if (srcInfo1.getLine() < srcInfo2.getLine()) {
            IgnoreRacyLocations.insert(loc1 + "|" + loc2);
        } else {
            IgnoreRacyLocations.insert(loc2 + "|" + loc1);
        }
    }
}
void aser::outputJSON() {
    if (CONFIG_INTEGRATE_DYNAMIC) {
        outputRaceList();
        outputIncludeList();
        outputIgnoreList();
    }
    std::string path;
    if (!ConfigOutputPath.empty()) {
        path = ConfigOutputPath;
    } else {
        info("writing detection results to ./races.json");
        path = "races.json";
    }

    // this part can be optimized
    // by utilizing void to_json(json &, DataRace &)
    std::vector<json> drJsons;
    // rank races
    std::sort(DataRace::races.begin(), DataRace::races.end());
    for (auto &r : DataRace::races) {
        drJsons.emplace_back(r.to_json());
    }

    std::vector<json> ovJsons;
    for (auto const &ov : OrderViolation::getOvs()) {
        ovJsons.emplace_back(ov.to_json());
    }

    std::vector<json> dlJsons;
    for (auto &r : DeadLock::deadlocks) {
        dlJsons.emplace_back(r.to_json());
    }

    std::vector<json> mapiJsons;
    std::sort(MismatchedAPI::mismatchedAPIs.begin(), MismatchedAPI::mismatchedAPIs.end());
    for (auto &r : MismatchedAPI::mismatchedAPIs) {
        mapiJsons.emplace_back(r.to_json());
    }
    std::vector<json> uaccountsJsons;
    std::sort(UntrustfulAccount::untrustfulAccounts.begin(), UntrustfulAccount::untrustfulAccounts.end());
    for (auto &r : UntrustfulAccount::untrustfulAccounts) {
        uaccountsJsons.emplace_back(r.to_json());
    }
    std::vector<json> usafeOperationsJsons;
    std::sort(UnSafeOperation::unsafeOperations.begin(), UnSafeOperation::unsafeOperations.end());
    for (auto &r : UnSafeOperation::unsafeOperations) {
        usafeOperationsJsons.emplace_back(r.to_json());
    }
    std::vector<json> cosplayAccountsJsons;
    std::sort(CosplayAccounts::cosplayAccounts.begin(), CosplayAccounts::cosplayAccounts.end());
    for (auto &r : CosplayAccounts::cosplayAccounts) {
        cosplayAccountsJsons.emplace_back(r.to_json());
    }

    std::vector<json> toctouJsons;
    // rank races
    std::sort(TOCTOU::races.begin(), TOCTOU::races.end());
    for (auto &r : TOCTOU::races) {
        toctouJsons.emplace_back(r.to_json());
    }

    json rs;
    rs["dataRaces"] = drJsons;
    rs["raceConditions"] = std::vector<json>();
    rs["orderViolations"] = ovJsons;
    rs["deadLocks"] = dlJsons;
    rs["mismatchedAPIs"] = mapiJsons;
    rs["toctou"] = toctouJsons;
    rs["untrustfulAccounts"] = uaccountsJsons;
    rs["unsafeOperations"] = usafeOperationsJsons;
    rs["cosplayAccounts"] = cosplayAccountsJsons;
    rs["version"] = 1;
    rs["generatedAt"] = getCurrentTimeStr();
    rs["bcfile"] = TargetModulePath.getValue();
    rs["numOfIRLines"] = NUM_OF_IR_LINES;
    rs["numOfAttackVectors"] = NUM_OF_ATTACK_VECTORS;
    rs["numOfFunctions"] = NUM_OF_FUNCTIONS;
    rs["addresses"] = SMART_CONTRACT_ADDRESSES;
    std::ofstream output(path, std::ofstream::out);
    output << rs;
    output.close();
}