//
// Created by peiming on 3/5/20.
//
#include <PTAModels/ExtFunctionManager.h>
#include <conflib/conflib.h>
#include <llvm/Support/GlobPattern.h>
#include <llvm/Support/Regex.h>

using namespace std;
using namespace llvm;

std::set<string> ExtFunctionsManager::THREAD_CREATIONS{"pthread_create", "__coderrect_stub_thread_create_no_origin"};

std::set<string> ExtFunctionsManager::SKIP_PATTERN{"llvm."};

// excluding GB_builder, GB_selector may be an aggressive optimization
std::set<string> ExtFunctionsManager::ONLY_CALLSITE{"GrB_Matrix_new", "GB_new",           "GrB_init",
                                                    "GB_cast_array",  "GB_hyper_realloc", "GB_selector",
                                                    "GB_builder",     "GB_memcpy",        "GB_realloc_memory"};

std::set<string> ExtFunctionsManager::SKIP{"Curl_strcasecompare", "freecookie",           "dprintf_DollarString",
                                           "curl_msnprintf",      "Curl_strncasecompare", "curl_mvsnprintf",
                                           "Curl_infof",          "Curl_strerror",        "dprintf_formatf"};

std::vector<string> ExtFunctionsManager::SKIP_FUNCTIONS;
std::vector<string> ExtFunctionsManager::IGNORE_RACE_FUNCTIONS;

void ExtFunctionsManager::init(std::vector<std::string> &skipFuns) {
    SKIP_FUNCTIONS = skipFuns;
    IGNORE_RACE_FUNCTIONS = conflib::Get<std::vector<std::string>>("ignoreRacesInFunctions", {});
}

bool ExtFunctionsManager::onlyKeepCallSite(const llvm::Function *F) {
    std::string funName = llvm::demangle(F->getName().str());

    if (ONLY_CALLSITE.find(F->getName().str()) != ONLY_CALLSITE.end()) {
        return true;
    }

    return false;
}

// user specified functions to ignore
bool ExtFunctionsManager::isSkipped(const llvm::Function *fun) {
    // FIXME: can we always do a demangle?
    std::string funName = llvm::demangle(fun->getName().str());

    if (SKIP.find(funName) != SKIP.end()) {
        return true;
    }
    for (auto pat : SKIP_PATTERN) {
        llvm::StringRef funNameRef(funName);
        if (funNameRef.startswith(pat)) {
            return true;
        }
    }
    for (auto pat : SKIP_FUNCTIONS) {
        auto rg = llvm::GlobPattern::create(pat);

        // llvm::Regex rg(pat);
        // debug
        // llvm::outs() << "funcName: " << funName << " pattern: " << pat << "\n";
        if (rg && rg->match(funName)) {
            return true;
        }
    }
    return false;
}
bool ExtFunctionsManager::isIgnoredRacesOrSkippedFunction(const llvm::Function *fun) {
    std::string funName = llvm::demangle(fun->getName().str());
    if (funName.rfind("std::thread::_Invoke") != string::npos || funName.rfind("std::invoke") != string::npos) {
        return false;
    }

    for (auto pat : IGNORE_RACE_FUNCTIONS) {
        auto rg = llvm::GlobPattern::create(pat);
        // llvm::Regex rg(pat);
        // debug
        // llvm::outs() << "funcName: " << funName << " ignored pattern: " << pat << "\n";
        if (rg && rg->match(funName)) {
            return true;
        }
    }
    for (auto pat : SKIP_FUNCTIONS) {
        auto rg = llvm::GlobPattern::create(pat);
        // llvm::Regex rg(pat);
        // debug
        // llvm::outs() << "funcName: " << funName << " skipped pattern: " << pat << "\n";
        if (rg && rg->match(funName)) {
            return true;
        }
    }
    return false;
}

bool ExtFunctionsManager::isPthreadGetSpecific(const llvm::Function *F) {
    return F->getName().equals("pthread_getspecific");
}

bool ExtFunctionsManager::isPthreadSetSpecific(const llvm::Function *F) {
    return F->getName().equals("pthread_setspecific");
}
