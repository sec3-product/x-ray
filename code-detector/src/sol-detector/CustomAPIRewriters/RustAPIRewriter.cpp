#include "CustomAPIRewriters/RustAPIRewriter.h"

#include <map>
#include <set>
#include <string>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "LogColor.h"
#include "PTAModels/GraphBLASModel.h"
#include "aser/Util/Util.h"

extern bool DEBUG_RUST_API;

namespace aser {

llvm::StringRef stripSolFileName(llvm::StringRef originalName, bool isDeclare) {
  // oracles.refresh_oracle_price.3$1
  //=> refresh_oracle_price.3
  auto splited = originalName.find(".");
  auto splited1 = originalName.find("::");
  if (splited < splited1) {
    originalName = originalName.substr(splited + 1);
  } else {
    // for create_cdp_vault::handler.2
    if (!isDeclare && !originalName.contains("::handler."))
      originalName = originalName.substr(splited1 + 2);
  }
  auto splited2 = originalName.find("$");
  if (splited2 != std::string::npos) {
    originalName = originalName.substr(0, splited2);
  }
  // Self::borrow.4
  auto splited3 = originalName.find("Self::");
  if (splited3 != std::string::npos) {
    originalName = originalName.substr(splited3 + 6);
  }

  // strip Processor::<{TOKEN_COUNT}>::process.3
  auto found_ = originalName.find(">::");
  if (found_ != std::string::npos) {
    if (DEBUG_RUST_API)
      llvm::outs() << "originalName >:: : " << originalName << "\n";
    originalName = originalName.substr(found_ + 3);
    if (DEBUG_RUST_API)
      llvm::outs() << "originalName after >:: : " << originalName << "\n";
  }

  if (!originalName.startswith("handler_") &&
      !originalName.endswith("::from.1") &&
      !originalName.endswith("::try_from.1") &&
      !originalName.endswith("::of_token.2")) // skip sol.handler_...::process
  {                                           // Pool::swap.2
    auto found = originalName.find("::");
    if (found != std::string::npos) {
      originalName = originalName.substr(found + 2);
    }
  }
  return originalName;
}

void rewriteIndirectTargets(llvm::Module *M) {
  llvm::IRBuilder<> builder(M->getContext());
  // 1. find all the rust functions, put them into three maps
  std::map<llvm::StringRef, std::set<llvm::Function *>> solDeclaredFunctionsMap;
  std::map<llvm::StringRef, std::set<llvm::Function *>> solIndexedFunctionsMap;
  for (llvm::Function &F : *M) {
    if (F.isDeclaration()) {
      if (aser::LangModel::isRustNormalCall(&F)) {
        // llvm::outs() << "declare: " << F.getName() << "\n";
        auto name = stripSolFileName(F.getName(), true);
        // function_name
        // sol.instructions::create_cdp_vault::handler.2
        // create_cdp_vault::handler.2
        auto splited = name.find("::handler.");
        if (splited != std::string::npos) {
          name = name.substr(splited + 2);
        }
        solDeclaredFunctionsMap[name].insert(&F);
        if (DEBUG_RUST_API)
          if (F.getName().contains("handler.2"))
            llvm::outs() << "decl: " << F.getName() << " -> " << name << "\n";
      }
    } else {
      // create_cdp_vault::handler.2
      auto name = stripSolFileName(F.getName(), false);
      if (name.startswith("handler.")) {
        name = F.getName();
      }
      solIndexedFunctionsMap[name].insert(&F);
      solIndexedFunctionsMap[F.getName()].insert(&F);
      if (DEBUG_RUST_API)
        if (F.getName().contains("handler.2"))
          llvm::outs() << "impl: " << F.getName() << " -> " << name << "\n";
    }
  }
  for (auto [name, declareFuncs] : solDeclaredFunctionsMap) {
    for (auto declareF : declareFuncs) {
      // llvm::outs() << "declared function: " << name << "\n";
      std::set<llvm::Function *> targetFunctions;

      if (solIndexedFunctionsMap.find(name) != solIndexedFunctionsMap.end() ||
          solIndexedFunctionsMap.find(name.lower()) !=
              solIndexedFunctionsMap.end() ||
          solIndexedFunctionsMap.find(name.upper()) !=
              solIndexedFunctionsMap.end()) {
        // llvm::outs() << "solIndexedFunctionsMap: name: " << name << "\n";

        if (solIndexedFunctionsMap.find(name) != solIndexedFunctionsMap.end()) {
          auto set = solIndexedFunctionsMap.at(name);
          targetFunctions.insert(set.begin(), set.end());
        }
        if (solIndexedFunctionsMap.find(name.lower()) !=
            solIndexedFunctionsMap.end()) {
          auto set = solIndexedFunctionsMap.at(name.lower());
          targetFunctions.insert(set.begin(), set.end());
        }
        if (solIndexedFunctionsMap.find(name.upper()) !=
            solIndexedFunctionsMap.end()) {
          auto set = solIndexedFunctionsMap.at(name.upper());
          targetFunctions.insert(set.begin(), set.end());
        }
      }
      if (targetFunctions.empty()) {
        // check partial lower and upper cases
        //  || name.endswith(namex) || namex.endswith(name)
        for (auto [namex, set] : solIndexedFunctionsMap) {
          if (namex.lower() == name.lower()) {
            targetFunctions.insert(set.begin(), set.end());
            break;
          } else {
            bool potentialMatch = false;
            if (namex.size() > name.size()) {
              if (namex.endswith("::" + name.str()))
                potentialMatch = true;
            } else if (namex.size() < name.size()) {
              if (name.endswith("::" + namex.str()))
                potentialMatch = true;
            }

            if (potentialMatch) {
              if (DEBUG_RUST_API)
                llvm::outs()
                    << "potential match: " << name << " -> " << namex << "\n";
              targetFunctions.insert(set.begin(), set.end());
              if (name.contains("transfer_context.") ||
                  name.contains("validate.") || name.contains("process.") ||
                  name.contains("swap.") ||
                  name.contains("note_burn_context.") ||
                  name.contains("validate_account_program_owners.") ||
                  name.contains("assert_") ||
                  name.contains("validate_vault_account.") ||
                  name.contains("handler.")) {
                continue; // for common keep adding
              } else
                break;
            }
          }
        }
      }
      if (!targetFunctions.empty()) {
        auto entryBB =
            llvm::BasicBlock::Create(M->getContext(), "sol.call", declareF);
        builder.SetInsertPoint(entryBB);
        llvm::CallInst *replaced = nullptr;
        for (auto targetF : targetFunctions) {
          // F.addFnAttr(Attribute::AlwaysInline);
          if (DEBUG_RUST_API)
            llvm::outs() << "Resolved call: " << declareF->getName()
                         << " -> target function: " << targetF->getName()
                         << "\n";

          if (declareF->arg_size() == targetF->arg_size()) {
            SmallVector<Value *, 8> args;
            for (int i = 0; i < declareF->arg_size(); i++) {
              llvm::Value *actual = declareF->arg_begin() + i;
              llvm::Value *formal = targetF->arg_begin() + i;

              if (actual->getType() != formal->getType()) {
                args.push_back(
                    builder.CreateBitCast(actual, formal->getType()));
              } else {
                args.push_back(actual);
              }
            }

            replaced = builder.CreateCall(
                targetF->getFunctionType(), targetF, args, "",
                targetF->getMetadata(
                    llvm::Metadata::MetadataKind::DILocationKind));
          } else {
            // not compatible
            if (DEBUG_RUST_API)
              llvm::outs() << "Imcompatible: " << declareF->arg_size()
                           << " -> : " << targetF->arg_size() << "\n";
          }
        }
        if (replaced != nullptr) {
          builder.CreateRet(replaced);
        }
      } else {
        if (DEBUG_RUST_API)
          llvm::outs() << "Unresolved call: " << declareF->getName() << "\n";
      }
    }
  }
}

void RustAPIRewriter::rewriteModule(llvm::Module *M) {
  rewriteIndirectTargets(M);
}

} // namespace aser
