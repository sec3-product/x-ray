//
// Created by peiming on 10/5/20.
//
#include "conflib/conflib.h"
#include "CustomAPIRewriters/ThreadProfileRewriter.h"
#include "CustomAPIRewriters/ThreadAPIRewriter.h"

using namespace std;
using namespace aser;
using namespace llvm;


void ThreadProfileRewriter::rewriteModule(llvm::Module *M) {
    // get all the thread profile entry in the json, which is a map
    auto profiles = conflib::Get<std::map<string, ThreadProfile>>("threadAPIProfiles", {});

    ThreadAPIRewriter::rewriteModule(M, profiles);
}