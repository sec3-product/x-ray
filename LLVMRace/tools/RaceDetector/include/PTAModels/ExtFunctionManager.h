//
// Created by peiming on 3/5/20.
//
#ifndef ASER_PTA_EXTFUNCTIONMANAGER_H
#define ASER_PTA_EXTFUNCTIONMANAGER_H

#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Function.h>

#include <set>

// external functions Manager
class ExtFunctionsManager {
private:
    // static std::set<llvm::StringRef> HEAP_ALLOCATIONS;
    // different from HEAP_ALLOCAIONS
    // HEAP_INITS is the kind of APIs that takes a pointer as an argument
    // and point it to the newly create memory
    // static std::set<llvm::StringRef> HEAP_INITS;

    static std::set<std::string> THREAD_CREATIONS;
    static std::set<std::string> SKIP;
    static std::set<std::string> ONLY_CALLSITE;
    static std::set<std::string> SKIP_PATTERN;
    static std::vector<std::string> SKIP_FUNCTIONS;
    static std::vector<std::string> IGNORE_RACE_FUNCTIONS;
    static std::vector<std::string> HEAP_APIS;

public:
    static void init(std::vector<std::string> &skipFun);

    static bool isSkipped(const llvm::Function *F);
    static bool onlyKeepCallSite(const llvm::Function *F);
    static bool isIgnoredRacesOrSkippedFunction(const llvm::Function *F);

    static bool isPthreadGetSpecific(const llvm::Function *F);
    static bool isPthreadSetSpecific(const llvm::Function *F);
};

#endif  // ASER_PTA_EXTFUNCTIONMANAGER_H
