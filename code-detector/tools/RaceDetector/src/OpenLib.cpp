#include "OpenLib.h"

#include <fstream>
#include <regex>
#include <string>

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/GlobPattern.h>

#include <nlohmann/json.hpp>

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

map<const Function *, const Function *> openLibCallbackCalleeMap;

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
    Module *module = config.module;
    auto entryPoints = config.entryPoints;

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
    }

    // TODO: handle entry points
    if (!entryPoints.empty()) {
        handleEntryPoints(module, builder, entryPoints);
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
