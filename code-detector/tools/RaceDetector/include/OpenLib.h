#ifndef RACEDETECTOR_OPENLIB_H
#define RACEDETECTOR_OPENLIB_H

#include <llvm/IR/Module.h>

#include <jsoncons/json.hpp>
#include <set>
namespace aser {
namespace openlib {

struct EntryPointT {
    std::string entryName;
    bool runOnce;
    bool argShared;
    bool isParallel;

    EntryPointT(const std::string &name, bool runOnce = false, bool argShared = true, bool isParallel = true)
        : entryName(name), runOnce(runOnce), argShared(argShared), isParallel(isParallel) {}

    EntryPointT(const EntryPointT &) = default;
    EntryPointT(EntryPointT &&) = default;
    EntryPointT &operator=(EntryPointT &&) = default;
    EntryPointT &operator=(const EntryPointT &) = default;

    bool operator<(const EntryPointT &rhs) const { return this->entryName < rhs.entryName; }
};

// using EntryPointT = std::map<std::string, std::map<std::string, bool>>;

struct OpenLibConfig {
    llvm::Module *module;
    std::vector<EntryPointT> entryPoints;
    bool explorePublicAPIs;
    // bool optimal;
    // bool fork;
    int mode;
    bool printAPI;
    bool onceOnly;
    uint32_t apiLimit;
};

void createFakeMain(OpenLibConfig &config);
std::string getCleanFunctionName(const llvm::Function *f);

}  // namespace openlib
}  // namespace aser

#endif
