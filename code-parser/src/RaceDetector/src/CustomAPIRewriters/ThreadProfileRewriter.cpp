#include "conflib/conflib.h"
#include "CustomAPIRewriters/ThreadProfileRewriter.h"
#include "CustomAPIRewriters/ThreadAPIRewriter.h"
#include "CustomAPIRewriters/LockUnlockRewriter.h"

using namespace std;
using namespace o2;
using namespace llvm;


void ThreadProfileRewriter::rewriteModule(llvm::Module *M) {
    // get all the thread profile entry in the json, which is a map
    auto profiles = conflib::Get<std::map<string, ThreadProfile>>("threadAPIProfiles", {});

    LockUnlockRewriter::rewriteModule(M, profiles);
    ThreadAPIRewriter::rewriteModule(M, profiles);
}