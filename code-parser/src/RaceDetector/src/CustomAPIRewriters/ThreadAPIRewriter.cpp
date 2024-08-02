#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/Regex.h>

#include "o2/Util/Log.h"
#include "o2/Util/Demangler.h"
#include "conflib/conflib.h"
#include "CustomAPIRewriters/ThreadAPIRewriter.h"
#include "CustomAPIRewriters/ThreadProfileRewriter.h"

using namespace std;
using namespace llvm;
using namespace o2;

namespace {

using StrVector = SmallVector<StringRef, 8>;
const StringRef MEM_PTR_REGEX = R"(^.+\((.+)::\*\)(\(.*\))$)";
const StringRef FUN_PTR_REGEX = R"(^.+\(\*\)(\(.*\))$)";
const StringRef LAMBDA_REGEX = R"(^(.+::\$_[[:digit:]]+)\&{0,2}$)";
const StringRef FUNCTOR_REGEX = R"(^([a-zA-Z]+)\&{0,2}$)";

inline PointerType* getVoidPtrType(LLVMContext &C) {
    return PointerType::getUnqual(IntegerType::get(C, 8));
}

inline StrVector getParamVector(StringRef paramStr) {
    StrVector result;

    paramStr = paramStr.substr(1, paramStr.size()-2); // get rid of parenthesis
//    paramStr.split(result, ',', -1, false);

    int nestedLevel = 0;
    int subStrStart = 0;
    int subStrLen = 0;
    for (char c : paramStr) {
        if (c == ',' && nestedLevel == 0) {
            result.push_back(paramStr.substr(subStrStart, subStrLen));
            subStrStart = subStrStart + subStrLen + 1;
            subStrLen = -1;
        } else if (c == '(' || c == '<') {
            nestedLevel ++;
        } else if (c == ')' || c == '>') {
            nestedLevel --;
        }
        subStrLen ++;
    }
    if (subStrLen != 0) {
        result.push_back(paramStr.substr(subStrStart, subStrLen));
    }
    return result;
}

inline void assertTypeStrEqual(StringRef expected, StringRef actual) {
    if (!expected.trim(' ').equals(actual.trim(' '))) {
#ifndef NDEBUG
        llvm_unreachable("Unmatched parameter");
#else
        LOG_ERROR("Unmatched parameter, expected={}, actual={}", expected, actual);
#endif
    }
}

//inline FunctionType* getStandardCallBackType(LLVMContext &C) {
//    return FunctionType::get(Type::getVoidTy(C), { getVoidPtrType(C), getVoidPtrType(C) },false);
//}

enum class CallableKind {
    MEM_PTR, // use member pointer to function as the callable
    FUN_PTR, // use the function pointer
    LAMBDA,  // use the lambda as the callable
    FUNCTOR, // use the object that overrides the operator() as the callable

    UNRESOLVABLE, // is there any thing that we missed here?
};

class ThreadAPIInfo;

struct OpParenthesis {
private:
    int vtableOff;
    Function *fun;

public:
    OpParenthesis() : vtableOff(-1), fun(nullptr) {}

    bool isVirtual() const {
        return vtableOff >= 0;
    }

    Function *getOpParenthesisFun() const{
        return fun;
    }

    friend ThreadAPIInfo;
};

class ThreadAPIInfo {
    bool isStatic;
    // we can learn how the functor is divided by looking at the callsite of the functor
    CallBase *functorReferee = nullptr;

    size_t entryIdx;
    size_t argStartIdx;
    size_t functorCount = 1;

    CallableKind kind;
    SmallVector<Type *, 8> args;
    Function *F;

    map<string, SmallVector<Function *, 2>> &opParenthesisMap;
    OpParenthesis opParenthesis; // operator(), for lambda and functor
private:
    void initialize(Demangler &demangler);
    void initArgTypeVector(StringRef baseClass, const StrVector &paramVec, const StrVector &callableParamVec);

public:
//    ThreadAPIInfo()
//        : F(nullptr), isStatic(false), entryIdx(0), argStartIdx(0), kind(CallableKind::UNRESOLVABLE) {}

    ThreadAPIInfo(Function *F, Demangler &demangler, size_t entryIdx, size_t argStartIdx, bool isNonStatic,
                  map<string, SmallVector<Function *, 2>> &map)
        : F(F), entryIdx(entryIdx), argStartIdx(argStartIdx), kind(CallableKind::UNRESOLVABLE),
          opParenthesisMap(map), isStatic(!isNonStatic) {
        // 1st, determine the callable kind
        initialize(demangler);
    }

    inline Function *getAPI() const { return F; }
    inline bool isStaticAPI() const { return isStatic; }
    inline size_t getEntryIdx() const { return entryIdx;}
    inline size_t getArgStartIdx() const { return argStartIdx; }
    inline size_t getArgNum() const { return this->args.size(); }
    inline CallableKind getCallableKind() const { return kind; }

    inline const SmallVector<Type *, 8>& getArgsType() const {
        return args;
    }

    inline const OpParenthesis& getOpParenthesis() const {
        assert(this->kind == CallableKind::LAMBDA || this->kind == CallableKind::FUNCTOR);
        return opParenthesis;
    }

    void resolveOpParenthesis(StringRef baseClass, const StrVector &paramVec, bool byVal);

    inline FunctionType *getCallableType() const {
        if (kind == CallableKind::LAMBDA || kind == CallableKind::FUNCTOR) {
            return opParenthesis.fun->getFunctionType();
        }
        return FunctionType::get(Type::getVoidTy(F->getContext()), args, false);
    }

    inline size_t getFunctorCount() const {
        assert(kind == CallableKind::LAMBDA || kind == CallableKind::FUNCTOR);
        return functorCount;
    }

    inline CallBase *getFunctorReferee() const {
        assert(kind == CallableKind::LAMBDA || kind == CallableKind::FUNCTOR);
        return functorReferee;
    }

    // covert the argument index in source code to the argument index in llvm ir
    inline size_t getArgStartIdxInIRFunc() const {
        return getArgIdxInIRFunc(getArgStartIdx());
    }

    inline size_t getEntryIdxInIRFunc() const {
        return getArgIdxInIRFunc(getEntryIdx());
    }

    // covert the argument index in source code to the argument index in llvm ir
    inline size_t getArgIdxInIRFunc(size_t argIdx) const {
        if (kind == CallableKind::MEM_PTR && entryIdx < argIdx) {
            // a memptr is implemented as two integers, so if it occurs before the argument,
            // then the argument start index need to be forwarded again
            argIdx ++;
        } else if ((kind == CallableKind::FUNCTOR || kind == CallableKind::LAMBDA) && entryIdx < argIdx) {
            argIdx += functorCount - 1;
        }

        // if it is not static, then the first one is always this
        if (!isStatic) {
            argIdx ++;
        }

        return argIdx;
    }

    inline Argument *getArgAtIdx(size_t argIdx) const {
        return F->arg_begin() + this->getArgIdxInIRFunc(argIdx);
    }

    inline Argument *getArgAtIdx(Function *fun, size_t argIdx) const {
        return fun->arg_begin() + this->getArgIdxInIRFunc(argIdx);
    }
};


// NOTE: entry number is the number of the parameter in the C++ source code, not in LLVM-IR
// e.g., ctor(entry), entryNum for entry is 0, that is, the implicit `this` does not count
void ThreadAPIInfo::initArgTypeVector(StringRef baseClass, const StrVector& paramVec,
                                      const StrVector& callableParamVec) {
    assert(callableParamVec.size() < paramVec.size());

    // this is member pointer used as a callable
    // the first argument is `this`, which should be equal to the base class type
    size_t argIdx = this->getArgStartIdx();

    if (!baseClass.empty()) {
        assertTypeStrEqual(paramVec[argIdx], baseClass.str() += "*");
        // push the type of *this* as the first argument type
        args.push_back(getArgAtIdx(argIdx)->getType());
        argIdx++;
    }

    // handle the rest arguments
    for (auto callableParam : callableParamVec) {
        assertTypeStrEqual(paramVec[argIdx], callableParam);

        args.push_back(getArgAtIdx(argIdx)->getType());
        argIdx ++;
    }
}

void ThreadAPIInfo::resolveOpParenthesis(StringRef baseClass, const StrVector &paramVec, bool passByVal) {
    assert(this->kind == CallableKind::LAMBDA || this->kind == CallableKind::FUNCTOR);

    Demangler demangler;
    if (opParenthesisMap.empty()) {
        // initialize map between class base name ==> the operator() overrided by the class
        for (auto &fun : *(this->getAPI()->getParent())) {
            if (!demangler.partialDemangle(fun.getName())) {
                StringRef baseName = demangler.getFunctionBaseName(nullptr, nullptr);
                if (baseName.equals("operator()")) {
                    StringRef declCtxName = demangler.getFunctionDeclContextName(nullptr, nullptr);
                    auto it = opParenthesisMap.find(baseName.str());
                    if (it == opParenthesisMap.end()) {
                        SmallVector<Function *, 2> tmp{&fun};
                        opParenthesisMap.insert(make_pair(declCtxName.str(), std::move(tmp)));
                    } else {
                        // has more than one overload for operator()
                        it->second.push_back(&fun);
                    }
                }
            }
        }
    }

    // resolved
    auto it = opParenthesisMap.find(baseClass.str());
    if (it == opParenthesisMap.end()) {
        LOG_ERROR("unable to find the overriden operator() for class. ClassName={}", baseClass);
        kind = CallableKind::UNRESOLVABLE;
        return;
    }

    auto &candidates = it->second;

    int maxMatch = -1;
    Function *target = nullptr;
    for (Function *candidate : candidates) {
        if (!demangler.partialDemangle(candidate->getName())) {
            StringRef callParamStr = demangler.getFunctionParameters(nullptr, nullptr);
            StrVector callParamVec = getParamVector(callParamStr);

            if (paramVec.size() <= this->getArgStartIdx()) {
                if (callParamVec.empty()) {
                    //maxMatch = 0;
                    target = candidate;
                    break;
                }
            } else {
                // find the longest match
                int idx = this->getArgStartIdx();

                bool fullyMatched = true;
                for (auto callParam : callParamVec) {
                    if (idx >= paramVec.size() ||
                        !callParam.trim(' ').equals(paramVec[idx].trim(' '))) {
                        fullyMatched = false;
                        break;
                    }
                    idx++;
                }

                if (fullyMatched && (int)(callParamVec.size()) > maxMatch) {
                    maxMatch = callParamVec.size();
                    target = candidate;
                }
            }
        }
    }

    if (target != nullptr) {
        if (passByVal) {
            // this is passed by value
            Argument *entryInAPI = this->getArgAtIdx(this->getEntryIdx());
            if (entryInAPI->getType() != target->arg_begin()->getType()) {
                // pass by val, the structure is expanded in the argument list.
                // need to figure out how the structure is lowered

                // 1st, find one of callsite of the API
                CallBase *callsite = nullptr;
                for (auto user : this->getAPI()->users()) {
                    if (auto call = dyn_cast<CallBase>(user)) {
                        if (call->getCalledFunction() == this->getAPI()) {
                            callsite = call;
                            break;
                        }
                    }
                }
                if (callsite == nullptr) {
                    kind = CallableKind::UNRESOLVABLE;
                    return;
                }
                int idx = this->getArgIdxInIRFunc(this->getEntryIdx());

                Value *actualArg = callsite->getArgOperand(idx);
                Value *loadTarget = nullptr;
                if (auto LI = dyn_cast<LoadInst>(actualArg)) {
                    auto operand = LI->getPointerOperand()->stripInBoundsConstantOffsets();
                    if (operand->getType() == target->arg_begin()->getType()) {
                        loadTarget = operand;
                    }
                }

                // find the following argument that is loaded from the same target
                if (loadTarget == nullptr) {
                    kind = CallableKind::UNRESOLVABLE;
                    return;
                }

                int count = 1;
                for (int i = idx + 1; i < callsite->arg_size(); i++) {
                    auto LI = dyn_cast<LoadInst>(callsite->getArgOperand(i));
                    if (LI && LI->getPointerOperand()->stripInBoundsConstantOffsets() == loadTarget) {
                        count ++;
                    } else {
                        break;
                    }
                }
                this->functorCount = count;
                this->functorReferee = callsite;
            }
        } else {
            // the entry functor/lambda should has the same type as the type of *this* in operator()
            assert(this->getArgAtIdx(this->getEntryIdx())->getType() == target->arg_begin()->getType());

        }

        this->opParenthesis.fun = target;
        for (auto& arg : target->args()) {
            this->args.push_back(arg.getType());
        }
        // TODO: should we handle virtual operator() differetly?
    } else {
        LOG_ERROR("unable to find the overriden operator() for class. ClassName={}", baseClass);
        kind = CallableKind::UNRESOLVABLE;
    }
}

// the argIdx is the *starting point* of the argument list
void ThreadAPIInfo::initialize(Demangler &demangler) {
    StringRef paramStr = demangler.getFunctionParameters(nullptr, nullptr);
    StrVector paramVec = getParamVector(paramStr);
    if (paramStr.equals("()")) {
        kind = CallableKind::UNRESOLVABLE;
        return;
    }

    StringRef entryTypeStr = paramVec[entryIdx];

    StrVector matched;
    Regex memptrRegex(MEM_PTR_REGEX);
    if (memptrRegex.match(entryTypeStr, &matched)) {
        assert(matched.size() == 3);

        // void (ClassName::*)(args1, args2)
        // 0 -- the whole match
        // 1 -- the matched `ClassName`
        // 2 -- the matched parameter list `(args1, args2)`
        StrVector callableParamVec = getParamVector(matched[2]);

        kind = CallableKind::MEM_PTR;
        this->initArgTypeVector(matched[1], paramVec, callableParamVec);
        return;
    }

    Regex funptrRegex(FUN_PTR_REGEX);
    if (funptrRegex.match(entryTypeStr, &matched)) {
        assert(matched.size() == 2);

        // void (ClassName::*)(args1, args2)
        // 0 -- the whole match
        // 2 -- the matched parameter list `(args1, args2)`
        StrVector callableParamVec = getParamVector(matched[1]);

        kind = CallableKind::FUN_PTR;
        this->initArgTypeVector("", paramVec, callableParamVec);
        return;
    }

    Regex lambdaRegex(LAMBDA_REGEX);
    if (lambdaRegex.match(entryTypeStr, &matched)) {
        assert(matched.size() == 2);
        // 0 -- the whole match
        // 1 -- the base class
        // find the operator() for the lambda
        kind = CallableKind::LAMBDA;
        // if there is not "&&" at the end of the
        this->resolveOpParenthesis(matched[1], paramVec, matched[1].equals(matched[0]));

        return;
    }

    Regex functorRegex(FUNCTOR_REGEX);
    if (functorRegex.match(entryTypeStr, &matched)) {
        assert(matched.size() == 2);

        kind = CallableKind::FUNCTOR;
        this->resolveOpParenthesis(matched[1], paramVec, matched[1].equals(matched[0]));
        return;
    }
}

// convert the customized API into a form that o2 can understand
class ThreadAPIConverter {
private:
    Module *M;
    IRBuilder<> builder;
    Function *getConvertedMemPtrAPI(const ThreadAPIInfo &APIInfo, bool isVirtual);
    static int curID;

    FunctionCallee getNextThreadEntryWrapper(const ThreadAPIInfo &APIInfo) {
        string name = ThreadAPIRewriter::getCanonicalizedAPIPrefix().str() + to_string(curID++);

        SmallVector<Type *, 8> params;
        params.push_back(getVoidPtrType(getContext())); // thread handler
        params.push_back(PointerType::getUnqual(APIInfo.getCallableType())); // call back function pointer
        params.append(APIInfo.getArgsType().begin(), APIInfo.getArgsType().end()); // the argument to callback function

        FunctionCallee entryWrapper = M->getOrInsertFunction(name, FunctionType::get(Type::getVoidTy(getContext()),
                                                                                     params, false));
        return entryWrapper;
    }

    /// core logic
    void convertMemPtrAsEntryThreadAPI(const ThreadAPIInfo &APIinfo);
    void convertFunPtrAsEntryThreadAPI(const ThreadAPIInfo &APIInfo);
    void convertOperatorAsEntryThreadAPI(const ThreadAPIInfo &APIInfo);

//    Function *getConvertedThreadAPI(CallableKind kind, const ThreadAPIInfo &APIInfo,
//                                    bool isVirtual = false) {
//        switch (kind) {
//            case CallableKind::MEM_PTR:
//                return getConvertedMemPtrAPI(APIInfo, isVirtual);
//            case CallableKind::FUN_PTR:
//                break;
//            case CallableKind::LAMBDA:
//                break;
//            case CallableKind::FUNCTOR:
//                break;
//            case CallableKind::UNRESOLVABLE:
//                break;
//        }
//
//        return nullptr;
//    }
public:
    explicit ThreadAPIConverter(Module *M) : M(M), builder(M->getContext()) {}

    inline LLVMContext& getContext() { return M->getContext(); }
    void convertThreadAPI(const ThreadAPIInfo &APIInfo);
};

int ThreadAPIConverter::curID = 0;

/// In the Itanium and ARM ABIs, method pointers have the form:
///   struct { ptrdiff_t ptr; ptrdiff_t adj; } memptr;
///
/// In the Itanium ABI:
///  - method pointers are virtual if (memptr.ptr & 1) is nonzero
///  - the this-adjustment is (memptr.adj)
///  - the virtual offset is (memptr.ptr - 1)
///
/// In the ARM ABI:
///  - method pointers are virtual if (memptr.adj & 1) is nonzero
///  - the this-adjustment is (memptr.adj >> 1)
///  - the virtual offset is (memptr.ptr)
/// ARM uses 'adj' for the virtual flag because Thumb functions
/// may be only single-byte aligned.
///
/// If the member is virtual, the adjusted 'this' pointer points
/// to a vtable pointer from which the virtual offset is applied.
///
/// If the member is non-virtual, memptr.ptr is the address of
/// the function to call.

/// Now, only handle Itanium ABI
void ThreadAPIConverter::convertMemPtrAsEntryThreadAPI(const ThreadAPIInfo &APIInfo) {
    Function *theAPI = APIInfo.getAPI();

    // when using memptr as entry, we need to manually resolve the target for different callsites
    for (auto user : theAPI->users()) {
        if (auto call = dyn_cast<CallBase>(user)) {
            // find a callsite
            // resolve the target of the member pointer
            size_t ptrIdx = APIInfo.getArgIdxInIRFunc(APIInfo.getEntryIdx());
            size_t adjIdx = ptrIdx + 1;
            /* store { i64, i64 } { i64 ptrtoint @func to i64), i64 0 }, { i64, i64 }* %7, align 8, !dbg !7528, !tbaa !7529
               %16 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 0, !dbg !7528
               %17 = load i64, i64* %16, align 8, !dbg !7528
               %18 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 1, !dbg !7528
               %19 = load i64, i64* %18, align 8, !dbg !7528*/
            auto ptr = cast<LoadInst>(call->getArgOperand(ptrIdx));
            auto adj = cast<LoadInst>(call->getArgOperand(adjIdx));
            assert(ptr->getPointerOperand()->stripPointerCasts() ==
                   adj->getPointerOperand()->stripInBoundsConstantOffsets());

            StoreInst *memPtrStore = nullptr;
            for (auto u : ptr->getPointerOperand()->stripPointerCasts()->users()) {
                if (auto SI = dyn_cast<StoreInst>(u)) {
                    if (memPtrStore == nullptr) {
                        memPtrStore = SI;
                    } else {
                        llvm_unreachable("multiple store for one member function definition are found");
                    }
                }
            }

            assert(memPtrStore);
            auto memPtrValue = cast<ConstantStruct>(memPtrStore->getValueOperand());
            auto ptrValue = memPtrValue->getOperand(0);

            if (auto ptrToInt = dyn_cast<PtrToIntOperator>(ptrValue)) {
                // this is a member pointer to non-virtual function, the target is the actual entry function
                auto target = llvm::cast<Function>(ptrToInt->getPointerOperand());
                // TODO, or should we call isCompatibableFunctionType? it should be okay, as template function
                // should guarantee the type equivalence statically
                assert(target->getFunctionType() == APIInfo.getCallableType());

                // Types are matched, converted the thread api to a form that is easier to analyze
                auto fun = this->getConvertedMemPtrAPI(APIInfo, false);
                call->setCalledFunction(fun);
            } else {
                // this is a member pointer to virtual function
                auto fun = this->getConvertedMemPtrAPI(APIInfo, true);
                call->setCalledFunction(fun);
            }
        }
    }
}

void ThreadAPIConverter::convertFunPtrAsEntryThreadAPI(const ThreadAPIInfo &APIInfo) {
    Function *theAPI = APIInfo.getAPI();
    // reimplement the API to call o2 special routine
    APIInfo.getAPI()->deleteBody();
    APIInfo.getAPI()->addFnAttr(Attribute::AlwaysInline);

    auto entry = BasicBlock::Create(getContext(), "entry", theAPI);
    builder.SetInsertPoint(entry);

    // 1st check whether the parameters are matched
    Argument *callback = APIInfo.getArgAtIdx(APIInfo.getEntryIdx());
    assert(callback->getType()->getPointerElementType() == APIInfo.getCallableType());

    // now create o2.thread.create function
    FunctionCallee threadWrapper = this->getNextThreadEntryWrapper(APIInfo);
    SmallVector<Value *, 8> args;
    if (!APIInfo.isStaticAPI()) {
        // if it is not static API, push *this* as the thread handler
        args.push_back(builder.CreateBitCast(theAPI->arg_begin(), builder.getInt8PtrTy()));
    } else {
        // not handled yet!
        llvm_unreachable("need to handle!");
    }

    // push back the call back
    args.push_back(callback);

    // push back the rest parameters.
    int argStartIdx = APIInfo.getArgStartIdxInIRFunc();
    for (int i = argStartIdx; i < argStartIdx + APIInfo.getArgNum(); i++) {
        args.push_back(theAPI->arg_begin() + i);
    }

    builder.CreateCall(threadWrapper, args);
    // 3rd, create call to o2.thread.create
    if (theAPI->getReturnType() != builder.getVoidTy()) {
        builder.CreateRet(UndefValue::get(theAPI->getReturnType()));
    } else {
        builder.CreateRetVoid();
    }
    //builder.CreateRetVoid();
}

void ThreadAPIConverter::convertOperatorAsEntryThreadAPI(const ThreadAPIInfo &APIInfo) {
    Function *theAPI = APIInfo.getAPI();
    Function *callback = APIInfo.getOpParenthesis().getOpParenthesisFun();


    size_t idx = APIInfo.getArgStartIdxInIRFunc(); // -1 will give include
    for (int i = 1; i < callback->arg_size(); i ++) { // start from 1 to skip *this*
        if ((callback->arg_begin() + i)->getType() != (theAPI->arg_begin() + idx)->getType()) {
            LOG_ERROR("Unmatched API, Lambda Type:{}, API Type:{}",
                      static_cast<void *>(callback->getFunctionType()),
                      static_cast<void *>(theAPI->getFunctionType()));
            return;
        }
        idx ++;
    }

    // reimplement the API to call o2 special routine
    theAPI->deleteBody();
    theAPI->addFnAttr(Attribute::AlwaysInline);

    auto entry = BasicBlock::Create(getContext(), "entry", theAPI);
    builder.SetInsertPoint(entry);

    Value *functor = APIInfo.getArgAtIdx(APIInfo.getEntryIdx());

    if (APIInfo.getArgAtIdx(APIInfo.getEntryIdx())->getType() != callback->arg_begin()->getType()) {
        // 1st the functor structure is lowered
        //

        functor = builder.CreateAlloca(callback->arg_begin()->getType()->getPointerElementType(), 0, nullptr);
        auto casted = builder.CreateBitCast(functor, builder.getInt8PtrTy());

        CallBase *referee = APIInfo.getFunctorReferee();
        assert(referee);

        // rebuild the functor according to referee, where the functor is decoupled.
        size_t functorStart = APIInfo.getEntryIdxInIRFunc();
        size_t functorEnd = APIInfo.getFunctorCount() + functorStart;

        for (size_t i = functorStart; i < functorEnd; i++) {
            auto op = cast<LoadInst>(referee->getArgOperand(i))->getPointerOperand();
            auto off = llvm::APInt(64, 0);
            op->stripAndAccumulateConstantOffsets(M->getDataLayout(), off, true);

            auto curArg = theAPI->arg_begin() + i;
            auto ptr = builder.CreateConstInBoundsGEP1_64(nullptr, casted, off.getSExtValue());
            auto addr = builder.CreateBitCast(ptr, PointerType::getUnqual(curArg->getType()));
            builder.CreateStore(curArg, addr);
        }
    }

    // now create o2.thread.create function
    FunctionCallee threadWrapper = this->getNextThreadEntryWrapper(APIInfo);
    SmallVector<Value *, 8> args;
    if (!APIInfo.isStaticAPI()) {
        // if it is not static API, push *this* as the thread handler
        args.push_back(builder.CreateBitCast(theAPI->arg_begin(), builder.getInt8PtrTy()));
    } else {
        // not handled yet!
        llvm_unreachable("need to handle!");
    }

    // push back the call back
    args.push_back(callback);
    args.push_back(functor);

    // push back the rest parameters.
    int argStartIdx = APIInfo.getArgStartIdxInIRFunc();
    for (int i = argStartIdx; i < argStartIdx + APIInfo.getArgNum() - 1; i++) {
        args.push_back(theAPI->arg_begin() + i);
    }

    builder.CreateCall(threadWrapper, args);
    // 3rd, create call to o2.thread.create
    if (theAPI->getReturnType() != builder.getVoidTy()) {
        builder.CreateRet(UndefValue::get(theAPI->getReturnType()));
    } else {
        builder.CreateRetVoid();
    }
}

void ThreadAPIConverter::convertThreadAPI(const ThreadAPIInfo &APIInfo) {
    switch (APIInfo.getCallableKind()) {
        case CallableKind::MEM_PTR:
            return this->convertMemPtrAsEntryThreadAPI(APIInfo);
        case CallableKind::FUN_PTR:
            return this->convertFunPtrAsEntryThreadAPI(APIInfo);
        case CallableKind::LAMBDA:
        case CallableKind::FUNCTOR:
            return this->convertOperatorAsEntryThreadAPI(APIInfo);
        case CallableKind::UNRESOLVABLE:
            break;
    }
}

// ASSUMPTION: the type equivalence should be guaranteed by the callee already
Function *ThreadAPIConverter::getConvertedMemPtrAPI(const ThreadAPIInfo &APIInfo, bool isVirtual) {
    Function *originalAPI = APIInfo.getAPI();
    string convertedName;
    if (isVirtual) {
        convertedName = originalAPI->getName().str();
        convertedName += ".memptr.virtual";
    } else {
        convertedName = originalAPI->getName().str();
        convertedName += ".memptr.non_virtual";
    }

    // if we already handle the case
    if (auto result = M->getFunction(convertedName)) {
        assert(result->getFunctionType() == originalAPI->getFunctionType());
        return result;
    }

    auto funCallee = M->getOrInsertFunction(convertedName, originalAPI->getFunctionType());
    auto fun = cast<Function>(funCallee.getCallee());

    fun->addFnAttr(Attribute::AlwaysInline);
    assert(fun->isDeclaration());

    auto entry = BasicBlock::Create(getContext(), "entry", fun);
    builder.SetInsertPoint(entry);
    // 1st, calculate the member pointer value
    Argument *ptr = APIInfo.getArgAtIdx(fun, APIInfo.getEntryIdx());
    Argument *adj = ptr + 1;
    assert(ptr->getType()->isIntegerTy() && adj->getType()->isIntegerTy());

    // adjust the offset for *this*
    llvm::Value *memptrThis = APIInfo.getArgAtIdx(fun, APIInfo.getArgStartIdx());
    llvm::Value *casted = builder.CreateBitCast(memptrThis, builder.getInt8PtrTy());
    casted = builder.CreateInBoundsGEP(nullptr, casted, adj);
    memptrThis = builder.CreateBitCast(casted, memptrThis->getType(), "this.adjusted");

    // 2nd, create unpack argument thread entry
    FunctionCallee threadWrapper = this->getNextThreadEntryWrapper(APIInfo);

    // memptr to function type, for non virtual member pointer to function, simply converted it back
    Value *callback;
    if (isVirtual) {
        // memptr convert to function pointer
        // here covert it to load from vtable
        // 1st, adjust this should points to the vtable
        auto vtableType = builder.getInt8PtrTy();
        auto vtablePtr = builder.CreateBitCast(memptrThis, PointerType::get(vtableType, 0));
        auto vtable = builder.CreateLoad(vtablePtr->getType()->getPointerElementType(), vtablePtr);

        Constant *ptrdiff_1 = llvm::ConstantInt::get(ptr->getType(), 1);
        auto vtableOff = builder.CreateSub(ptr, ptrdiff_1);
        auto vfunPtr = builder.CreateGEP(nullptr, vtable, vtableOff);
        vfunPtr = builder.CreateBitCast(vfunPtr,
                                        PointerType::getUnqual(PointerType::getUnqual(APIInfo.getCallableType())));
        callback = builder.CreateLoad(vfunPtr->getType()->getPointerElementType(), vfunPtr);
    } else {
        callback = builder.CreateIntToPtr(ptr, PointerType::getUnqual(APIInfo.getCallableType()));
    }

    SmallVector<Value *, 8> args;
    if (!APIInfo.isStaticAPI()) {
        // if it is not static API,
        args.push_back(builder.CreateBitCast(fun->arg_begin(), builder.getInt8PtrTy()));
    } else {
        // no handled yet?
        llvm_unreachable("need to handle!");
    }

    args.push_back(callback); // push back the call back
    // then push back the adjusted *this*
    args.push_back(memptrThis);

    // push back the rest parameters.
    int argStartIdx = APIInfo.getArgStartIdxInIRFunc();
    for (int i = argStartIdx + 1; i < argStartIdx + APIInfo.getArgNum(); i++) {
        args.push_back(fun->arg_begin() + i);
    }

    builder.CreateCall(threadWrapper, args);
    // 3rd, create call to o2.thread.create
    // TODO: handle case when API return non-void
    builder.CreateRetVoid();
    return fun;
}

FunctionType *getPthreadCallBackType(LLVMContext &C) {
    return FunctionType::get(getVoidPtrType(C), {getVoidPtrType(C)}, false);
}

FunctionCallee getStandardCThreadCreateWrapper(Module *M) {
    string name = ThreadAPIRewriter::getStandardCThreadCreateAPI().str();

    SmallVector<Type *, 8> params;
    // thread handler
    params.push_back(getVoidPtrType(M->getContext()));
    // call back function pointer
    params.push_back(PointerType::getUnqual(getPthreadCallBackType(M->getContext())));
    // the argument to callback function
    params.push_back(getVoidPtrType(M->getContext()));
    FunctionCallee entryWrapper = M->getOrInsertFunction(name,FunctionType::get(Type::getVoidTy(M->getContext()),
                                                                                params, false));
    return entryWrapper;
}

FunctionCallee getSpecialCThreadCreateWrapper(Module *M, FunctionType *callbackTy, Type *retTy) {
    static int postFix = 0;
    string name = ThreadAPIRewriter::getCanonicalizedAPIPrefix().str() + ".Special.C."+ to_string(postFix++);

    SmallVector<Type *, 8> params;
    // thread handler
    params.push_back(getVoidPtrType(M->getContext()));
    // call back function pointer
    params.push_back(PointerType::getUnqual(callbackTy));
    // the argument to callback function
    params.append(callbackTy->params().begin(), callbackTy->params().end());
    FunctionCallee entryWrapper = M->getOrInsertFunction(name,FunctionType::get(retTy, params, false));
    return entryWrapper;
}

void rewriteCThreadCreateAPI(Module *M, Function *fun, const ThreadCreateAPI &api) {
    int entryIdx = api.getEntryIdx() - 1; // start from 1
    Argument *callback = fun->arg_begin() + entryIdx;

    if (!callback->getType()->isPointerTy() ||
        !callback->getType()->getPointerElementType()->isFunctionTy()) {
        // callback is not a function type?
        LOG_ERROR("Function has bad signature: callback for F: {} is not a function pointer", fun->getName());
        return;
    }

    auto callBackType = cast<FunctionType>(callback->getType()->getPointerElementType());
    if (callBackType->isVarArg()) {
        LOG_ERROR("Does not support var_arg function passed as callback!");
        return;
    }
    // clear previous implementation
    fun->deleteBody();
    fun->addFnAttr(Attribute::AlwaysInline);

    IRBuilder<> builder(M->getContext());
    auto entryBB = BasicBlock::Create(M->getContext(), "o2.lock", fun);
    builder.SetInsertPoint(entryBB);

    SmallVector<Value *, 4> params;

    // FIXME: This is an assumption that might be wrong
    // 1st argument is a pointer ==> thread handler
    if (fun->arg_begin()->getType()->isPointerTy() && api.getEntryIdx() != 1 && api.getArgIdx() != 1) {
        auto handle = builder.CreateBitCast(fun->arg_begin(), builder.getInt8PtrTy());
        params.push_back(handle);
    } else {
        // push a nullptr
        params.push_back(ConstantPointerNull::get(builder.getInt8PtrTy()));
    }

    if (callBackType->getNumParams() == 1) {
        auto casted = builder.CreateBitCast(callback, PointerType::getUnqual(getPthreadCallBackType(M->getContext())));
        params.push_back(casted);

        int argIdx = api.getArgIdx() - 1;  // start from 1
        auto argument = builder.CreateBitCast(fun->arg_begin() + argIdx, builder.getInt8PtrTy());
        params.push_back(argument);

        FunctionCallee wrapper = getStandardCThreadCreateWrapper(M);
        builder.CreateCall(wrapper, params);

        if (fun->getReturnType() != builder.getVoidTy()) {
            builder.CreateRet(UndefValue::get(fun->getReturnType())); // does not care about return?
        } else {
            builder.CreateRetVoid();
        }
    } else {
        // the callback has multiple arguments, need to handle it specially
        params.push_back(callback);
        int callbackArgIdx = api.getcallBackArgIdx() - 1;

        for (int i = 0; i < callBackType->params().size(); i ++) {
            if (callbackArgIdx == i) {
                int argIdx = api.getArgIdx() - 1;  // start from 1
                params.push_back(fun->arg_begin() + argIdx);
            } else {
                params.push_back(UndefValue::get(callBackType->getParamType(i)));
            }
        }

        FunctionCallee wrapper = getSpecialCThreadCreateWrapper(M, callBackType, fun->getReturnType());
        auto ret = builder.CreateCall(wrapper, params);
        if (!fun->getReturnType()->isVoidTy()) {
            builder.CreateRet(ret);
        } else {
            builder.CreateRetVoid();
        }
    }
}

}

void ThreadAPIRewriter::rewriteModule(llvm::Module *M, const std::map<string, ThreadProfile> &profiles) {
    Demangler demangler;
    ThreadAPIConverter converter(M);

    vector<Function*> funVec;
    funVec.reserve(M->getFunctionList().size());
    // as well will insert new function, cached the original function here
    auto entrys = conflib::Get<vector<string>>("XthreadCreateAPI", {});

    map<string, SmallVector<Function *, 2>> OpParenthesisMap;

    for (auto &F : *M) {
        // find the boost::thread::ctor
        for (const auto &it : profiles) {
            if (it.first == "pthread") {
                continue;
            }

            const ThreadProfile &profile = it.second;
            if (profile.isCXXProfile()) {
                // CXX profile need to be demangled
                if (!demangler.partialDemangle(F.getName())) {
                    for (const ThreadCreateAPI &threadCreate : profile.getThreadAPI().getCreateAPIs()) {
                        if (threadCreate.isCtor() && demangler.isCtor()) {
                            StringRef funDeclCtx = demangler.getFunctionDeclContextName(nullptr, nullptr);
                            if (funDeclCtx.equals(threadCreate.getFunctionName())) {
                                // boost::thread<> is a template, determine the callable object
                                ThreadAPIInfo APIInfo(&F, demangler,
                                                      threadCreate.getEntryIdx() - 1 ,
                                                      threadCreate.getArgIdx() - 1,
                                                      threadCreate.isNonStaticAPI(),
                                                      OpParenthesisMap);
                                converter.convertThreadAPI(APIInfo);
                            }
                        } else {
                            StringRef funName = demangler.getFunctionName(nullptr, nullptr);
                            if (funName.equals(threadCreate.getFunctionName())) {
                                ThreadAPIInfo APIInfo(&F, demangler,
                                                      threadCreate.getEntryIdx() - 1 ,
                                                      threadCreate.getArgIdx() - 1,
                                                      threadCreate.isNonStaticAPI(),
                                                      OpParenthesisMap);
                                converter.convertThreadAPI(APIInfo);
                            }
                        }
                    }
                }
                continue;
            } else {

                for (const ThreadCreateAPI &threadCreate : profile.getThreadAPI().getCreateAPIs()) {
                    if (!demangler.partialDemangle(F.getName())) {
                        StringRef name = demangler.getFunctionName(nullptr, nullptr);
                        // find a customized thread create
                        if (name.equals(threadCreate.getFunctionName())) {
                            rewriteCThreadCreateAPI(M, &F, threadCreate);
                        }
                    } else if (stripNumberPostFix(F.getName()).equals(threadCreate.getFunctionName())) {
                        rewriteCThreadCreateAPI(M, &F, threadCreate);
                    }
                }
            }
        }

//        if (!demangler.partialDemangle(F.getName())) {
//            // if can be demangled
//            if (demangler.isCtor()) {
//                StringRef funDeclCtx = demangler.getFunctionDeclContextName(nullptr, nullptr);
//                // this is the boost thread constructor
//
//                if (funDeclCtx.equals("boost::thread")) {
//                    // boost::thread<> is a template, determine the callable object
//                    ThreadAPIInfo APIInfo(&F, demangler, 0, 1, false, OpParenthesisMap);
//                    converter.convertThreadAPI(APIInfo);
//                }
//            }
//            for (const auto& entry : entrys) {
//                StringRef funName = demangler.getFunctionName(nullptr, nullptr);
//                if (funName.startswith(entry)) {
//                    ThreadAPIInfo APIInfo(&F, demangler, 0, 1, OpParenthesisMap);
//                    converter.convertThreadAPI(APIInfo);
//                }
//            }
//
//            // check the thread profiles
//        }
    }
}
