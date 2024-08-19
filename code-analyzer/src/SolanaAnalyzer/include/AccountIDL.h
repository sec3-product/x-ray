#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <jsoncons/json.hpp>
#include <llvm/IR/Module.h>

namespace aser {

class AccountIDL {
public:
  const std::string name;
  bool isMut;
  bool isSigner;
  bool isNested;
  AccountIDL(const std::string name, bool isMut, bool isSigner,
             bool isNested = false)
      : name(name), isMut(isMut), isSigner(isSigner), isNested(isNested) {}
};

extern std::set<llvm::StringRef> SMART_CONTRACT_ADDRESSES;

extern void loadIDLInstructionAccounts(std::string api_name, jsoncons::json j);
extern void computeDeclareIdAddresses(llvm::Module *module);

} // namespace aser
