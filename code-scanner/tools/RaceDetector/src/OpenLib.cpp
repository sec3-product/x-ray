#include "OpenLib.h"

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/GlobPattern.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>

#include "PTAModels/GraphBLASModel.h"
using namespace llvm;
using namespace aser;
using namespace std;
using namespace nlohmann;
using namespace openlib;

extern bool CONFIG_ENTRY_POINT_SINGLE_TIME;
std::string CR_ALLOC_OBJ_RECUR = ".coderrect.recursive.allocation";
std::string CR_FUNC_NAME = "coderrect_cb.";
static unsigned int crcount = 0;
std::set<llvm::Value *> crFunctions;
extern bool DEBUG_API;
static bool PRINT_API = false;
static uint32_t api_limit_count = 1000;
// TODO: avoid recursion
// be simple: set a depth
uint8_t OBJECT_MAX_DEPTH = 200;
uint8_t object_depth;

static const string ENTRY_RUN_ONCE = "once";
static const string ENTRY_ARG_SHARED = "arg_shared";

llvm::Function *crAllocRecFun = nullptr;
std::map<llvm::Function *, EntryPointT> matchedEntryFunctions;

llvm::Value *createNewObject(llvm::IRBuilder<> &builder, llvm::PointerType *type) {
    // PEIMING: recursive initialize a object of certain type is already implemented by pointer analysis
    // simply use a special function and delegate the new object initialization to pointer analysis
    auto ptr = builder.CreateCall(crAllocRecFun->getFunctionType(), crAllocRecFun);
    return builder.CreateBitCast(ptr, type);

    //    // create a shared object
    //    llvm::AllocaInst *AllocSharedObj = builder.CreateAlloca(type->getPointerElementType(), 0, "");
    //    if (object_depth > OBJECT_MAX_DEPTH) return AllocSharedObj;
    //    // let's just initialize this obj's field
    //    object_depth++;
    //    if (auto ty = dyn_cast<StructType>(type->getPointerElementType())) {
    //        for (unsigned int i = 0; i < ty->getStructNumElements(); i++) {
    //            auto ty2 = ty->getStructElementType(i);
    //            if (ty2->isPtrOrPtrVectorTy()) {
    //                //
    //                // llvm::AllocaInst *AllocSharedObj1 = builder.CreateAlloca(ty2->getPointerElementType(), 0, "");
    //                if (auto ptrTy = dyn_cast<llvm::PointerType>(ty2)) {
    //                    if (auto ST = dyn_cast<StructType>(ptrTy->getPointerElementType())) {
    //                        if (ST->isOpaque()) {
    //                            // skip the opaque type
    //                            continue;
    //                        }
    //                    } else if (ptrTy->getPointerElementType()->isFunctionTy()) {
    //                        // alloca a function is not allowed in IR
    //                        continue;
    //                    }
    //                }
    //                llvm::Value *AllocSharedObj1 = createNewObject(builder, dyn_cast<llvm::PointerType>(ty2));
    //                llvm::APInt zero(32, 0);
    //                llvm::APInt index_i(32, i);
    //                std::vector<llvm::Value *> values;
    //                values.push_back(llvm::Constant::getIntegerValue(builder.getInt32Ty(), zero));
    //                values.push_back(llvm::Constant::getIntegerValue(builder.getInt32Ty(), index_i));
    //
    //                auto gep = builder.CreateGEP(AllocSharedObj, values);
    //                builder.CreateStore(AllocSharedObj1, gep, false);
    //
    //            } else if (ty2->isAggregateType()) {
    //            }
    //        }
    //    } else if (auto ty = dyn_cast<ArrayType>(type->getPointerElementType())) {
    //        auto ty2 = ty->getArrayElementType();
    //        if (ty2->isPtrOrPtrVectorTy()) {
    //            llvm::Value *AllocSharedObj1 = createNewObject(builder, dyn_cast<llvm::PointerType>(ty2));
    //            llvm::APInt zero(32, 0);
    //            llvm::APInt index_i(32, 0);  // initialize the first element?
    //            std::vector<llvm::Value *> values;
    //            values.push_back(llvm::Constant::getIntegerValue(builder.getInt32Ty(), zero));
    //            values.push_back(llvm::Constant::getIntegerValue(builder.getInt32Ty(), index_i));
    //
    //            auto gep = builder.CreateGEP(AllocSharedObj, values);
    //            builder.CreateStore(AllocSharedObj1, gep, false);
    //        } else if (ty2->isAggregateType()) {
    //            // recursive?
    //        }
    //    }
    //    // object_depth--;
    //    return AllocSharedObj;
}
std::map<PointerType *, llvm::Value *> allocatedObjects;
llvm::Value *createNewObjectWrapper(llvm::IRBuilder<> &builder, llvm::PointerType *type, bool reuse) {
    if (reuse && allocatedObjects.find(type) != allocatedObjects.end()) {
        auto obj = allocatedObjects.at(type);
        // llvm::outs() << "1type: " << type << " " << *type << " obj: " << obj << " " << *obj << "\n";
        return obj;
    }
    object_depth = 0;
    auto obj = createNewObject(builder, type);
    if (reuse) allocatedObjects[type] = obj;
    // llvm::outs() << "2type: " << type << " " << *type << " obj: " << obj << " " << *obj << "\n";
    return obj;
}
llvm::Value *getOrAllocateNewObject(llvm::IRBuilder<> &builder, PointerType *type) {
    if (allocatedObjects.find(type) == allocatedObjects.end()) {
        llvm::Value *AllocSharedObj = createNewObjectWrapper(builder, type, true);
        allocatedObjects[type] = AllocSharedObj;
        return AllocSharedObj;
    } else {
        return allocatedObjects.at(type);
    }
}
std::map<const llvm::Function *, llvm::Value *> allocatedSmallTalkInitializeObjects;
llvm::Value *createSmallTalkInitializeObjectWrapper(llvm::IRBuilder<> &builder, const llvm::Function *callee,
                                                    llvm::PointerType *type, bool reuse) {
    if (reuse && allocatedSmallTalkInitializeObjects.find(callee) != allocatedSmallTalkInitializeObjects.end()) {
        auto obj = allocatedSmallTalkInitializeObjects.at(callee);
        return obj;
    }

    auto para = createNewObjectWrapper(builder, type, reuse);
    std::vector<llvm::Value *> Args;
    Args.push_back(para);
    llvm::ArrayRef<llvm::Value *> argsRef(Args);
    auto obj = builder.CreateCall((llvm::Function *)callee, argsRef, "");
    if (reuse) allocatedSmallTalkInitializeObjects[callee] = obj;
    return obj;
}

void addFunctionCallInCRCallBack(Module *module, llvm::Function *caller, const llvm::Function *callee) {
    llvm::BasicBlock *entryBB = llvm::BasicBlock::Create(module->getContext(), "entrypoint", caller);
    llvm::IRBuilder<> builder(module->getContext());
    builder.SetInsertPoint(entryBB);

    llvm::Value *ret;
    std::vector<llvm::Value *> Args;

    int size = callee->getFunctionType()->getFunctionNumParams();
    for (unsigned int i = 0; i < size; i++) {
        Type *type = callee->getFunctionType()->getFunctionParamType(i);
        if (i == 0 && type == caller->getFunctionType()->getFunctionParamType(0)) {
            // for argument 0
            //%3 = alloca %"class.pthreadrace::RingBuffer"*, align 8
            // store i8* %0, i8** %2, align 8, !tbaa !1002
            for (auto &ag : caller->args()) {
                Args.push_back(&ag);  // i64* %3
                break;
            }
            continue;
        }

        {
            if (type->isPtrOrPtrVectorTy()) {
                // type = type->getPointerElementType();
                // llvm::AllocaInst *Alloc = builder.CreateAlloca(type, 0, "");

                llvm::Value *Alloc = createNewObjectWrapper(builder, dyn_cast<llvm::PointerType>(type), false);

                Args.push_back(Alloc);  // i8* %11

            } else {
                // TODO: use type

                Args.push_back(Constant::getNullValue(type));  // 0
            }
        }
    }

    llvm::ArrayRef<llvm::Value *> argsRef(Args);
    builder.CreateCall((llvm::Function *)callee, argsRef, "");
    // builder.CreateRet(builder.getInt32(0));
    builder.CreateRetVoid();
}
llvm::Value *createNewCRCallBack(Module *module, llvm::Type *type) {
    crcount++;
    string funcName = CR_FUNC_NAME + std::to_string(crcount);
    llvm::Value *coderrectCall = module->getFunction(funcName);
    if (!coderrectCall) {
        llvm::IRBuilder<> builder(module->getContext());
        // create one
        Type *argsPTC[] = {type};
        FunctionType *coderrectCallTy = FunctionType::get(builder.getVoidTy(), ArrayRef<Type *>(argsPTC, 1), false);
        auto callee = module->getOrInsertFunction(funcName, coderrectCallTy);
        coderrectCall = callee.getCallee();
    }

    crFunctions.insert(coderrectCall);
    return coderrectCall;
}
llvm::Value *getOrCreatePthreadCreateFunction(Module *module, llvm::Type *type) {
    llvm::Value *pthreadCreateFun = module->getFunction("pthread_create");
    if (!pthreadCreateFun) {
        // create one
        llvm::IRBuilder<> builder(module->getContext());
        Type *argsPTC[] = {builder.getInt64Ty()->getPointerTo(), builder.getInt8Ty()->getPointerTo(),
                           static_cast<Type *>(FunctionType::get(builder.getVoidTy(), type, false))->getPointerTo(),
                           type};
        // Type *argsPTC[] = {
        //     builder.getInt64Ty()->getPointerTo(), builder.getVoidTy()->getPointerTo(),
        //     static_cast<Type *>(FunctionType::get(builder.getVoidTy(), builder.getVoidTy()->getPointerTo(), false))
        //         ->getPointerTo(),
        //     builder.getVoidTy()->getPointerTo()};

        FunctionType *pthreadCreateTy = FunctionType::get(builder.getInt32Ty(), ArrayRef<Type *>(argsPTC, 4), false);
        auto callee = module->getOrInsertFunction("pthread_create", pthreadCreateTy);
        pthreadCreateFun = callee.getCallee();
    }

    return pthreadCreateFun;
}
bool hasSpecialFunctionStrings(std::string &funcName) {
    if (funcName.find("st.anonfun.") == 0 || funcName.find("__cxx11::") == 0 || funcName.find("__cxx_") == 0 ||
        funcName.find("__gnu_cxx::") == 0 || funcName.find("__gthread") == 0 || funcName.find("_GLOBAL__") == 0 ||
        funcName.find("std::") != string::npos || funcName.find("::$_") != string::npos ||
        funcName.find(".omp") == 0)  //.omp_task_entry. .omp_outlined_.
        return true;

    // skip sol.model.
    if (funcName.find("sol.model.") == 0) return true;

    // heuristics: skip those funcs with names init/alloc
    std::transform(funcName.begin(), funcName.end(), funcName.begin(), ::tolower);
    if (funcName == "main" || funcName == "cr_main" || funcName.find("init") != string::npos ||
        funcName.find("alloc") != string::npos || funcName.find("coderrect_") != string::npos) {
        return true;
    }
    return false;
}

set<const Function *> interestingFunctions;        // functions of interest
set<const Function *> interestingFunctionCallers;  //
set<const Function *> interestingPublicAPIs;       //
set<const Function *> exploredFunctions;           //
map<const Function *, set<const Function *>> calleeCallerMap;
set<const Function *> excludedConstructors;

void aser::openlib::computeInterestingAPIs(set<const Function *> &interestingFuncs) {
    // find public APIs that call interestingFunctions
    for (auto interest : interestingFuncs) {
        if (exploredFunctions.count(interest)) continue;
        exploredFunctions.insert(interest);

        if (calleeCallerMap.find(interest) != calleeCallerMap.end()) {
            if (interestingFunctionCallers.count(interest)) continue;
            auto interestingFuncs2 = calleeCallerMap.at(interest);
            computeInterestingAPIs(interestingFuncs2);
            interestingFunctionCallers.insert(interest);
        } else {
            auto funcName = demangle(interest->getName().str());
            if (hasSpecialFunctionStrings(funcName)) continue;  // skip cr_main

            if (excludedConstructors.find(interest) == excludedConstructors.end()) {
                if (DEBUG_API) llvm::outs() << "public api (interesting): " << funcName << "\n";
                interestingPublicAPIs.insert(interest);
            }
        }
    }
}

set<const Function *> apiFuncs;
map<const Function *, const Function *> openLibCallbackCalleeMap;

void aser::openlib::computeCandidateAPIs(Module *module, int mode) {
    // return if already computed
    if (!apiFuncs.empty()) return;

    map<Type *, set<Function *>> candidateClassesMap;
    map<Type *, set<Function *>> candidateMethodsMap;

    // 1. get all C++ classes with constructors
    auto &functionList = module->getFunctionList();
    for (auto &function : functionList) {
        if (function.isIntrinsic() || function.isDeclaration()) continue;
        string funcName = demangle(function.getName().str());
        // llvm::outs() << "func: "<<<<"\n";
        bool isCPP = false;
        if (funcName.find("::") != string::npos) isCPP = true;
        size_t size = function.getFunctionType()->getNumParams();

        if (isCPP && size > 0) {
            auto *type = function.getFunctionType()->getFunctionParamType(0);
            if (type->isPointerTy()) {
                if (type->getPointerElementType()->isStructTy()) {
                    // llvm::outs() << "class type: "<<*type<<"\n";
                    candidateClassesMap[type].insert(&function);
                }
            }
        } else {
            for (int i = 0; i < size; i++) {
                auto *type = function.getFunctionType()->getFunctionParamType(i);
                if (type->isStructTy() || (type->isPtrOrPtrVectorTy() && type->getPointerElementType()->isStructTy()))
                    candidateClassesMap[type].insert(&function);
            }
        }
    }

    // traverse and find out constructor, destructor, and apis
    for (auto [type, funcs] : candidateClassesMap) {
        if (DEBUG_API) llvm::outs() << "\nstruct type: " << *type << "\n";
        for (auto *f : funcs) {
            string funcName = demangle(f->getName().str());

            if (funcName.find("::") != funcName.npos) {  // C++
                // c++ names
                std::size_t pos1 = funcName.find_last_of("(");
                std::size_t pos2 = funcName.substr(0, pos1).find_last_of("::");
                string methodname = funcName.substr(pos2 + 1, pos1 - pos2 - 1);
                string classname = funcName.substr(0, pos2 - 1);
                std::size_t pos3 = classname.find_last_of("::");
                if (pos3) classname = classname.substr(pos3 + 1, pos2 - pos3 + 1);

                if (methodname.find("~") != methodname.npos) {
                    // destructor
                    if (DEBUG_API) llvm::outs() << "destructor : " << funcName << "\n";
                    excludedConstructors.insert(f);
                } else if (methodname == classname) {
                    // constructor
                    if (DEBUG_API) llvm::outs() << "constructor : " << funcName << "\n";
                    excludedConstructors.insert(f);

                } else {
                    // other methods
                    if (DEBUG_API) llvm::outs() << "API : " << funcName << "\n";
                    if (funcName.find("std::") == string::npos) candidateMethodsMap[type].insert(f);
                }
            } else {  // C
                if (DEBUG_API) llvm::outs() << "API : " << funcName << "\n";
                if (funcName.find("std::") == string::npos) candidateMethodsMap[type].insert(f);
            }
        }
    }

    for (auto &function : functionList) {
        auto &basicBlockList = function.getBasicBlockList();
        for (auto &BB : basicBlockList) {
            for (BasicBlock::const_iterator BI = BB.begin(), BE = BB.end(); BI != BE; ++BI) {
                const Instruction *inst = llvm::dyn_cast<Instruction>(BI);
                if (auto callInst = llvm::dyn_cast<llvm::CallBase>(inst)) {
                    auto *callee = callInst->getCalledFunction();
                    if (callee)  // make sure callee is resolved
                    {
                        if ((mode == 2 && LangModel::isSmalltalkForkCall(callee)) ||
                            (mode == 1 && LangModel::isSyncCall(callee))) {
                            interestingFunctions.insert(&function);
                        } else {
                            calleeCallerMap[callee].insert(&function);
                        }
                    } else {
                        // llvm::outs() << "indirect call : " << *callInst << " in " <<
                        // demangle(function.getName().str()) << "\n";
                    }
                }
            }
        }
    }

    if (DEBUG_API) {
        for (auto f : interestingFunctions) llvm::outs() << "sync func : " << demangle(f->getName().str()) << "\n";
    }
    if (mode > 0) {
        // find public APIs that call interestingFunctions
        computeInterestingAPIs(interestingFunctions);

        // clear maps
        interestingFunctions.clear();
        interestingFunctionCallers.clear();
        exploredFunctions.clear();
        calleeCallerMap.clear();
        candidateClassesMap.clear();
        candidateMethodsMap.clear();
        apiFuncs.clear();

        // assign apiFuncs in optimal mode
        apiFuncs = interestingPublicAPIs;
    } else {
        // get public APIs with no callers
        for (auto &function : functionList) {
            if (function.isIntrinsic() || function.isDeclaration()) continue;
            if (calleeCallerMap.find(&function) == calleeCallerMap.end()) {
                string funcName = demangle(function.getName().str());

                // skip standard libraries
                //__gthread __cxx11:: std:: __cxx_ __gnu_cxx:: _GLOBAL__sub_I_ __invoke .omp_outlined.
                // TestDirectory()::$_4::__invoke(void*, char const*, unsigned int)

                if (hasSpecialFunctionStrings(funcName)) continue;

                // TODO: get rid of constructors and destructors
                if (excludedConstructors.find(&function) != excludedConstructors.end()) continue;

                // consider func with name starting with "Graph_" only
                // if (function.getName().startswith("Graph_Query")
                // //|| function.getName().startswith("Graph_Explain")
                // //  ||   function.getName().startswith("Graph_Profile")
                //     )
                if (DEBUG_API) llvm::outs() << "public api: " << demangle(function.getName().str()) << "\n";
                apiFuncs.insert(&function);
            }
        }
    }
}
std::string aser::openlib::getCleanFunctionName(const llvm::Function *f) {
    std::string funcName = demangle(f->getName().str());
    // llvm::outs() << "public api before: " << name << "\n";
    // c++ names
    // TODO: we need a full fledged parser for this task
    if (funcName.find("::") != funcName.npos) {
        // strip away parameters and anonymous namespace
        std::regex r("\\(anonymous\\ namespace\\)::");
        funcName = std::regex_replace(funcName, r, "");

        // viz::mojom::LayeredWindowUpdaterInterceptorForTesting::Draw(base::OnceCallback<void ()>)

        std::size_t pos1 = funcName.find_last_of("(");  // first or last?
        std::size_t pos1_a = funcName.find("(");
        if (pos1 != pos1_a) {
            // TODO: there exist template paras <()>
            std::size_t pos1_b = funcName.substr(0, pos1_a).find("<");
            if (pos1_b == string::npos) pos1 = pos1_a;
        }
        std::size_t pos2 = funcName.substr(0, pos1).find_last_of("::");
        string methodname = funcName.substr(pos2 + 1, pos1 - pos2 - 1);

        // get rid of methodname now
        funcName = funcName.substr(0, pos1);
        pos2 = funcName.find_last_of("::");
        size_t pos0 = funcName.find('<');
        if (pos0 != string::npos) {
            funcName = funcName.substr(0, pos0);
            pos2 = funcName.find_last_of("::");
        }
        if (pos2) {
            string classname = funcName.substr(0, pos2 - 1);
            std::size_t pos3 = classname.find_last_of("::");
            if (pos3) classname = classname.substr(pos3 + 1, pos2 - pos3 + 1);
            // strip away any template class parameter
            size_t pos4 = classname.find('<');
            if (pos4 != string::npos) {
                classname = classname.substr(0, pos4);
            }
            funcName = classname + "::" + methodname;
        } else {
            funcName = methodname;
        }
    }
    size_t found = funcName.find('(');
    if (found != string::npos) {
        funcName = funcName.substr(0, found);
    }
    found = funcName.find('[');
    if (found != string::npos) {
        //// NumberToString[ab::cxx11]
        // get rid of [..]
        funcName = funcName.substr(0, found);
    }
    // non-virtual thunk to HandleManager::Evict
    // void rocksdb::(anonymous namespace)::DeleteCachedEntry<rocksdb::BlockContents>(rocksdb::Slice const&,
    // void*)
    // non-virtual thunk to chrome_pdf::OutOfProcessInstance::Beep()

    found = funcName.find_last_of(' ');
    if (found != string::npos) {
        funcName = funcName.substr(found + 1);
    }
    // authnone_marshal.1354
    found = funcName.find('.');
    if (found != string::npos && !f->getName().startswith("st.")) {
        funcName = funcName.substr(0, found);
    }
    if (DEBUG_API) {
        llvm::outs() << "public api: " << funcName << "\n";
        if (funcName.empty() || funcName.find(">") != string::npos) {
            llvm::outs() << "public api before: " << demangle(f->getName().str()) << "\n";
        }
    }

    return funcName;
}
void generateSortedAPINames(std::vector<std::string> &apiNames) {
    std::set<std::string> names;
    for (auto *f : apiFuncs) {
        auto funcName = getCleanFunctionName(f);
        if (funcName.find(">") != string::npos) continue;  // this may be a constructor
        if (!funcName.empty()) names.insert(funcName);
    }
    apiNames.assign(names.begin(), names.end());
    std::sort(apiNames.begin(), apiNames.end());
}
void printAPIToJson() {
    // Jie: generate a json file containing all these public apis
    // $CODERRECT_TMPDIR/api.json
    // {
    //     apis: [
    //        "SharedLRUCache::Insert",
    //        "SharedLRUCache::Put",
    //        ... ...
    //    ]
    // }
    json apis;
    std::vector<std::string> apiNames;
    generateSortedAPINames(apiNames);

    std::string path = "api.json";
    if (const char *env_p = std::getenv("CODERRECT_TMPDIR")) {
        path = "/" + path;  // unix separator
        path = env_p + path;
    }
    std::ofstream output(path, std::ofstream::out);
    apis["apis"] = apiNames;
    output << apis;
    output.close();
}
void createThreadCallBack(Module *module, llvm::IRBuilder<> &builder, const llvm::Function *f,
                          llvm::Value *sharedObjPtr) {
    llvm::Type *type = sharedObjPtr->getType();  //->getAllocatedType();
    llvm::Value *pthreadCreateFun = getOrCreatePthreadCreateFunction(module, type);
    auto funTy = cast<FunctionType>(pthreadCreateFun->getType()->getPointerElementType());
    // sharedObject AllocInst
    //%1 = alloca %"class.pthreadrace::RingBuffer", align 8

    std::vector<llvm::Value *> Args;
    llvm::AllocaInst *AllocThreadId = builder.CreateAlloca(builder.getInt64Ty(), 0, "");
    llvm::AllocaInst *AllocThreadAttr = builder.CreateAlloca(builder.getInt8Ty(), 0, "");
    Args.push_back(AllocThreadId);  // 0
    // Args.push_back(AllocThreadAttr); //1
    Args.push_back(builder.CreateBitCast(AllocThreadAttr, funTy->getParamType(1)));  // 1
    llvm::Value *callback = createNewCRCallBack(module, type);
    // Args.push_back(callback);
    Args.push_back(builder.CreateBitCast(callback, funTy->getParamType(2)));  // 2
    // Args.push_back(AllocSharedObj);
    Args.push_back(builder.CreateBitCast(sharedObjPtr, funTy->getParamType(3)));  // 3

    llvm::ArrayRef<llvm::Value *> argsRef(Args);
    builder.CreateCall(cast<Function>(pthreadCreateFun), argsRef, "");
    addFunctionCallInCRCallBack(module, (llvm::Function *)callback, f);
    openLibCallbackCalleeMap[(const llvm::Function *)callback] = f;
}

std::set<const llvm::Function *> addedCallBackFuntions;
void createThreadCallBackWrapper(Module *module, llvm::IRBuilder<> &builder, const llvm::Function *f,
                                 llvm::Value *sharedObjPtr, bool duplicate) {
    if (addedCallBackFuntions.find(f) == addedCallBackFuntions.end()) {
        createThreadCallBack(module, builder, f, sharedObjPtr);
        if (duplicate) createThreadCallBack(module, builder, f, sharedObjPtr);
        addedCallBackFuntions.insert(f);
    }
}

void createBuilderCallFunction(llvm::IRBuilder<> &builder, llvm::Function *f) {
    std::vector<llvm::Value *> Args;
    auto it = f->arg_begin();
    auto ie = f->arg_end();

    for (; it != ie; it++) {
        if (it->getType()->isPointerTy()) {
            llvm::AllocaInst *allocaInst =
                builder.CreateAlloca(dyn_cast<PointerType>(it->getType())->getPointerElementType(), 0, "");
            Args.push_back(allocaInst);
        } else {
            llvm::APInt zero(32, 0);
            Args.push_back(llvm::Constant::getIntegerValue(builder.getInt32Ty(), zero));
        }
    }
    llvm::ArrayRef<llvm::Value *> argsRef(Args);
    builder.CreateCall(f, argsRef, "");
}
void refineAPIFunctions() {
    set<const llvm::Function *> apiFuncs2;
    map<char, int> charCountMap;
    for (auto *f : apiFuncs) {
        auto name = getCleanFunctionName(f);
        auto c = name.front();
        if (!charCountMap.count(c)) {
            charCountMap[c] = 0;
        } else {
            if (charCountMap[c] > 9) {
                continue;
            }
        }
        charCountMap[c]++;
        apiFuncs2.insert(f);
    }
    apiFuncs.clear();
    apiFuncs.insert(apiFuncs2.begin(), apiFuncs2.end());
}
extern std::set<std::string> SmallTalkCTRemoteClasses;
void keepOnlyCTRemoteAPIFunctions() {
    set<const llvm::Function *> apiFuncs2;
    for (auto *f : apiFuncs) {
        auto fname = getCleanFunctionName(f);
        auto className = fname.substr(fname.find_last_of('$') + 1);
        auto pos = className.find(" class");
        if (pos != string::npos) className = className.substr(0, pos);
        if (SmallTalkCTRemoteClasses.find(className) != SmallTalkCTRemoteClasses.end()) {
            // llvm::outs() << "st.CTRemote: " << className << "\n";
            apiFuncs2.insert(f);
        }
    }
    if (apiFuncs2.size() > 0) {
        // only trigger if there exists any SmallTalkCTRemoteClasses in the api functions
        apiFuncs.clear();
        apiFuncs.insert(apiFuncs2.begin(), apiFuncs2.end());
    }
}
void doAlgorithm2(Module *module, llvm::IRBuilder<> &builder, bool once) {
    // st.CTRemote
    if (SmallTalkCTRemoteClasses.size() > 0) {
        // llvm::outs() << "SmallTalkCTRemoteClasses size:" << SmallTalkCTRemoteClasses.size() << "\n";
        keepOnlyCTRemoteAPIFunctions();
    }
    if (apiFuncs.empty())
        return;
    else if (apiFuncs.size() > api_limit_count) {
        // if too many api functions, then pick a subset of them
        refineAPIFunctions();
    }
    // llvm::outs() << "\npublic api limit: " << api_limit_count << "\n";
    llvm::outs() << "automatically inferred " << apiFuncs.size() << " apis as entry points (limit=" << api_limit_count
                 << "):\n";

    // std::vector<std::string> apiNames;
    // generateSortedAPINames(apiNames);
    // llvm::outs() << "automatically inferred " << apiNames.size() << " apis as entry points:\n";
    // for (auto name : apiNames) llvm::outs() << "public api: " << name << "\n";
    Type *mytype = builder.getInt8PtrTy();
    // init
    crAllocRecFun = llvm::cast<Function>(
        module->getOrInsertFunction(CR_ALLOC_OBJ_RECUR, FunctionType::get(builder.getInt8PtrTy(), false)).getCallee());

    uint32_t count = 0;
    bool repeated = !once;
    for (auto *f : apiFuncs) {
        if (count < 99)
            llvm::outs() << "public api: " << demangle(f->getName().str()) << "\n";
        else if (count == 99)
            llvm::outs() << "public api: " << (apiFuncs.size() - 99) << " more apis ...\n\n";

        bool stHandled = false;

        // for smalltalk - %3 = call i8* @"st.initialize$LamAlarmManager"(i8* %0)
        auto fname = f->getName();
        if (fname.startswith("st.")) {
            auto className = fname.substr(fname.find_last_of('$'));
            auto initializeFuncName = "st.initialize" + className.str();
            llvm::Function *initializeFunc = module->getFunction(initializeFuncName);
            if (initializeFunc) {
                // llvm::outs() << "initializeFunc: " << initializeFuncName << "\n";
                llvm::Value *AllocSharedObj = createSmallTalkInitializeObjectWrapper(
                    builder, initializeFunc, dyn_cast<llvm::PointerType>(mytype), true);
                createThreadCallBackWrapper(module, builder, f, AllocSharedObj, repeated);  // call twice if repeated
                stHandled = true;
            }
            auto initializeFromFuncName = "st.initializeFrom:" + className.str();
            llvm::Function *initializeFromFunc = module->getFunction(initializeFromFuncName);
            if (initializeFromFunc) {
                // llvm::outs() << "initializeFromFunc: " << initializeFromFuncName << "\n";
                llvm::Value *AllocSharedObj = createSmallTalkInitializeObjectWrapper(
                    builder, initializeFromFunc, dyn_cast<llvm::PointerType>(mytype), true);
                createThreadCallBackWrapper(module, builder, f, AllocSharedObj, repeated);  // call twice if repeated
                stHandled = true;
            }
        }
        if (!stHandled) {
            int size = f->getFunctionType()->getFunctionNumParams();
            for (unsigned int i = 0; i < size; i++) {
                Type *type = f->getFunctionType()->getFunctionParamType(i);
                if (type->isPtrOrPtrVectorTy()) {
                    mytype = type;
                    break;
                }
            }
            llvm::Value *AllocSharedObj = createNewObjectWrapper(builder, dyn_cast<llvm::PointerType>(mytype), true);
            createThreadCallBackWrapper(module, builder, f, AllocSharedObj, repeated);  // call twice if repeated
        }
        if (count++ > api_limit_count) break;
    }
}
void exploreOpenLibraryAPIs(Module *module, llvm::IRBuilder<> &builder, int mode, bool once) {
    computeCandidateAPIs(module, mode);
    doAlgorithm2(module, builder, once);
}

llvm::Value *allocObjectForFunction(const llvm::Function *function, IRBuilder<> &builder, bool reuse = true) {
    uint32_t size = function->getFunctionType()->getNumParams();
    llvm::Value *Alloc;
    if (size > 0 && function->getFunctionType()->getFunctionParamType(0)->isPtrOrPtrVectorTy()) {
        Type *type = function->getFunctionType()->getFunctionParamType(0);
        if (type == PointerType::get(IntegerType::get(function->getContext(), 8), 0)) {
            // an void * type
            // here try to infer the type by the first bitcast in the function
            auto arg = function->arg_begin();
            const BitCastInst *bitCastUser = nullptr;
            for (auto user : arg->users()) {
                if (auto bitCast = dyn_cast<BitCastInst>(user)) {
                    bitCastUser = bitCast;
                } else if (!isa<IntrinsicInst>(user)) {
                    bitCastUser = nullptr;
                    break;
                }
            }
            if (bitCastUser && bitCastUser->getDestTy()->isPointerTy()) {
                Alloc = llvm::cast<Instruction>(builder.CreateBitCast(
                    createNewObjectWrapper(builder, dyn_cast<llvm::PointerType>(bitCastUser->getDestTy()), reuse),
                    type));
            } else {
                Alloc = createNewObjectWrapper(builder, dyn_cast<llvm::PointerType>(type), reuse);
            }
        } else {
            Alloc = createNewObjectWrapper(builder, dyn_cast<llvm::PointerType>(type), reuse);
        }
    } else {
        Alloc = createNewObjectWrapper(builder, builder.getInt8PtrTy(), reuse);
    }

    return Alloc;
}

void handleEntryPoints(Module *module, llvm::IRBuilder<> &builder, std::vector<EntryPointT> &entrys) {
    llvm::outs() << "\nEntryPoints:\n";
    std::set<EntryPointT> entryPoints;

    // Filter empty name and duplicate entries
    for (auto const &entry : entrys) {
        if (entry.entryName.empty()) {
            continue;
        }

        llvm::outs() << entry.entryName;
        if (!entryPoints.insert(entry).second) {
            llvm::outs() << "  <- this entry point is specified multiple times!";
        }
        llvm::outs() << "\n";
    }

    // Stop here if there are no entry points
    if (entryPoints.empty()) {
        return;
    }

    // Set up special recusrsive create PTA function
    crAllocRecFun = llvm::cast<Function>(
        module->getOrInsertFunction(CR_ALLOC_OBJ_RECUR, FunctionType::get(builder.getInt8PtrTy(), false)).getCallee());

    // Match entry names to functions
    std::set<EntryPointT> unmatchedEntries;
    for (auto const &entry : entryPoints) {
        unmatchedEntries.insert(entry);
    }

    for (auto &function : module->getFunctionList()) {
        if (function.isIntrinsic() || function.isDeclaration()) {
            continue;
        }

        auto funcName = getCleanFunctionName(&function);
        if (funcName.empty() || funcName.find(">") != std::string::npos) {
            funcName = demangle(function.getName().str());
        }

        for (auto const &entry : entryPoints) {
            string entryName = "*" + entry.entryName;  // match any prefix
            if (entryName == "**") entryName = "*";    // special "*" to match all apis
            Expected<GlobPattern> pattern = llvm::GlobPattern::create(entryName);
            if (pattern.takeError()) {
                // llvm::outs() << "Unable to search for: " << entry.entryName << "\n";
                continue;
            }

            if (pattern->match(funcName)) {
                // llvm::outs() << "Matched: " << entry.entryName << " to " << funcName << "\n";
                matchedEntryFunctions.insert(std::make_pair(&function, entry));
                unmatchedEntries.erase(entry);
                break;
            } else {
                // llvm::outs() << "Not Matched: " << entry.entryName << " to " << funcName << "\n";
            }
        }
    }

    for (auto const &entry : unmatchedEntries) {
        llvm::outs() << " -- Could not find: " << entry.entryName << "\n";
    }

    if (matchedEntryFunctions.empty()) {
        llvm::outs() << "No entry points are matched. You may have mistyped the entry names.\n";
        return;
    }

    if (matchedEntryFunctions.size() > 100) {
        llvm::outs() << "Too many APIs are selected as entry points (" << matchedEntryFunctions.size()
                     << " in total). This may incur large memory consumption!\n";
        for (auto &entryFunction : matchedEntryFunctions)
            llvm::outs() << demangle(entryFunction.first->getName().str()) << "\n";
    }

    // First insert function calls for each non-parallel entry
    for (auto &matchpair : matchedEntryFunctions) {
        auto func = matchpair.first;
        auto const &entryInfo = matchpair.second;

        // Only handling non-prallel in first pass
        if (entryInfo.isParallel) {
            continue;
        }

        // Create fake args
        std::vector<llvm::Value *> args;
        for (auto const &arg : func->args()) {
            auto const type = arg.getType();
            if (type->isPtrOrPtrVectorTy()) {
                auto alloc = createNewObjectWrapper(builder, dyn_cast<PointerType>(type), false);
                args.push_back(alloc);
            } else {
                args.push_back(Constant::getNullValue(type));
            }
        }
        auto call = builder.CreateCall(func, args);
        assert(call && "failed to create call inst");
    }

    // Now insert pthread calls for each parallel entry
    for (auto const &matchpair : matchedEntryFunctions) {
        auto func = matchpair.first;
        auto const &entryInfo = matchpair.second;

        // Already handled non-prallel in first pass
        if (!entryInfo.isParallel) {
            continue;
        }

        auto Alloc = allocObjectForFunction(func, builder);
        if (entryInfo.runOnce) {
            createThreadCallBack(module, builder, func, Alloc);
        }
        // Run twice with shared arg
        else if (entryInfo.argShared) {
            createThreadCallBack(module, builder, func, Alloc);
            createThreadCallBack(module, builder, func, Alloc);
        }
        // Run twice but arg is not shared
        else {
            createThreadCallBack(module, builder, func, Alloc);
            llvm::Value *Alloc2 = allocObjectForFunction(func, builder, false);
            createThreadCallBack(module, builder, func, Alloc2);
        }
    }
}
void aser::openlib::createFakeMain(OpenLibConfig &config) {
    if (config.printAPI) {
        // after printing apis we will exit
        computeCandidateAPIs(config.module, config.mode);
        printAPIToJson();
        exit(0);
    }

    Module *module = config.module;
    auto entryPoints = config.entryPoints;

    PRINT_API = config.printAPI;
    api_limit_count = config.apiLimit;
    // let's create a fake main func here and add it to the module IR
    // in the fake main, call each entry point func
    // TODO: need to deal with demangled function names
    // for example:     //test => _Z4testv
    llvm::IRBuilder<> builder(module->getContext());
    // create fake main with type int(i32 argc, i8** argv)
    auto functionType = llvm::FunctionType::get(builder.getInt32Ty(),
                                                {builder.getInt32Ty(), builder.getInt8PtrTy()->getPointerTo()}, false);
    llvm::Function *mainFunction =
        llvm::Function::Create(functionType, llvm::Function::ExternalLinkage, "cr_main", module);
    llvm::BasicBlock *entryBB = llvm::BasicBlock::Create(module->getContext(), "entrypoint", mainFunction);
    builder.SetInsertPoint(entryBB);

    // to handle signals, let's also call real main from a different thread?
    // or we call signal handling before real main?
    // now signals are automatically handled, so real main can be moved to before entry points
    // this avoid FPs between main and other entry apis

    llvm::Function *realMainFun = module->getFunction("main");
    if (realMainFun && !realMainFun->isDeclaration()) {
        if (realMainFun->getFunctionType() == functionType) {
            // create a call to real main using fake main's argc, argv if possible
            llvm::SmallVector<Value *, 2> args;
            for (auto &arg : mainFunction->args()) {
                args.push_back(&arg);
            }
            builder.CreateCall(realMainFun, args, "");
        } else {
            createBuilderCallFunction(builder, realMainFun);
        }
    } else {
        // for fortran code generated by flang
        llvm::Function *fortranMainFun = module->getFunction("MAIN_");
        if (fortranMainFun && !fortranMainFun->isDeclaration()) {
            if (fortranMainFun->getFunctionType() == functionType) {
                // create a call to fortran main using fake main's argc, argv if possible
                llvm::SmallVector<Value *, 2> args;
                for (auto &arg : mainFunction->args()) {
                    args.push_back(&arg);
                }
                builder.CreateCall(fortranMainFun, args, "");
            } else {
                createBuilderCallFunction(builder, fortranMainFun);
            }
        }
    }

    // TODO: handle entry points
    if (!entryPoints.empty()) {
        handleEntryPoints(module, builder, entryPoints);
    }
    if (config.explorePublicAPIs) {
        exploreOpenLibraryAPIs(module, builder, config.mode, config.onceOnly);
    }

    // cr_main return
    builder.CreateRet(llvm::ConstantInt::get(builder.getInt32Ty(), 0));

    // print newly added callback functions
    if (DEBUG_API) {
        // print cr_main
        mainFunction->print(llvm::errs(), nullptr);
        // module->print(llvm::errs(), nullptr);
        for (auto *f : crFunctions) {
            ((Function *)f)->print(llvm::errs(), nullptr);
            llvm::verifyFunction(*(Function *)f);
        }
    }
}
bool aser::openlib::isInferredPublicAPI(const llvm::Function *f) {
    if (apiFuncs.find(f) != apiFuncs.end())
        return true;
    else
        return false;
}
