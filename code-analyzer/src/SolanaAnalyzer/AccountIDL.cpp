#include "AccountIDL.h"

#include <stdio.h>

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <PointerAnalysis/Program/CallSite.h>
#include <jsoncons/json.hpp>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Module.h>

#include "PTAModels/GraphBLASModel.h"

namespace aser {

std::map<std::string, std::vector<AccountIDL>> IDL_INSTRUCTION_ACCOUNTS;

void loadIDLInstructionAccounts(std::string api_name, jsoncons::json j) {
  // llvm::outs() << "accounts: " << j.as_string() << "\n";
  if (j.is_array()) {
    // iterate the array
    for (const auto &item : j.array_range()) {
      auto account_name = item["name"].as<std::string>();
      // nested accounts
      if (item.contains("accounts")) {
        auto j2 = item.at("accounts");
        // llvm::outs() << "nested accounts: " << account_name << "\n";
        IDL_INSTRUCTION_ACCOUNTS[api_name].emplace_back(account_name, false,
                                                        false, true);
        loadIDLInstructionAccounts(api_name, j2);
      } else if (item.contains("isMut") && item.contains("isSigner")) {
        auto isMut = item["isMut"].as<bool>();
        auto isSigner = item["isSigner"].as<bool>();
        // llvm::outs() << "account_name: " << account_name << " isMut: " <<
        // isMut << " isSigner: " << isSigner<< "\n";
        IDL_INSTRUCTION_ACCOUNTS[api_name].emplace_back(account_name, isMut,
                                                        isSigner);
      }
    }
  }
}

static std::string exec(std::string command) {
  char buffer[128];
  std::string result = "";

  // Open pipe to file
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return "popen failed!";
  }

  // read till end of process:
  while (!feof(pipe)) {
    // use buffer to read and add to result
    if (fgets(buffer, 128, pipe) != NULL)
      result += buffer;
  }

  pclose(pipe);
  return result;
}

std::set<llvm::StringRef> SMART_CONTRACT_ADDRESSES;

void computeDeclareIdAddresses(llvm::Module *module) {
  // st.class.metadata
  auto f = module->getFunction("sol.model.declare_id.address");
  if (f) {
    for (auto &BB : *f) {
      for (auto &I : BB) {
        if (llvm::isa<llvm::CallBase>(&I)) {
          aser::CallSite CS(&I);
          if (CS.getNumArgOperands() < 1) {
            continue;
          }

          auto v1 = CS.getArgOperand(0);
          auto valueName1 = LangModel::findGlobalString(v1);
          llvm::outs() << "contract address: " << valueName1 << "\n";
          SMART_CONTRACT_ADDRESSES.insert(valueName1);

          if (!IDL_INSTRUCTION_ACCOUNTS.empty()) {
            continue;
          }

          // mainnet only
          // anchor --provider.cluster mainnet idl fetch
          // nosRB8DUV67oLNrL45bo2pFLrmsWPiewe2Lk2DRNYCp -o
          // nosRB8DUV67oLNrL45bo2pFLrmsWPiewe2Lk2DRNYCp.json
          auto address_str = valueName1.str();
          auto path = address_str + "-idl.json";
          if (const char *env_p = std::getenv("CODERRECT_TMPDIR")) {
            path = "/" + path; // unix separator
            path = env_p + path;
          }
          auto cmd = "/usr/bin/anchor --provider.cluster mainnet idl fetch " +
                     address_str + +" -o " + path;
          auto result = exec(cmd);
          // llvm::outs() << "anchor idl : " << result << "\n";
          LOG_INFO("Anchor IDL result: {}", result);
          std::ifstream ifile;
          ifile.open(path);
          if (ifile) {
            LOG_INFO("Find Anchor IDL File: {}", path);
            auto j0 = jsoncons::json::parse(ifile);
            std::string instructions = "instructions";
            if (j0.contains(instructions)) {
              auto j = j0.at(instructions);
              if (j.is_array()) {
                // iterate the array
                for (const auto &item : j.array_range()) {
                  auto api_name = item["name"].as<std::string>();
                  // llvm::outs() << "api_name: " << api_name << "\n";
                  auto j2 = item["accounts"];
                  loadIDLInstructionAccounts(api_name, j2);
                }
              }
            }
          }
        }
      }
    }
  }
}

} // namespace aser
