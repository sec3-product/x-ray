#include "StaticThread.h"

#include "Graph/Event.h"

using namespace aser;
using namespace std;

TID StaticThread::curID = 0;
std::map<TID, StaticThread*> StaticThread::tidToThread;
std::map<TID, std::map<uint8_t, const llvm::Constant*>> StaticThread::threadArgs;

void StaticThread::setThreadArg(TID tid, std::map<uint8_t, const llvm::Constant*>& argMap) {
    threadArgs.insert(std::make_pair(tid, argMap));
};

std::map<uint8_t, const llvm::Constant*>* StaticThread::getThreadArg(TID tid) {
    if (threadArgs.count(tid)) {
        return &(threadArgs.at(tid));
    }
    return nullptr;
};

extern std::map<std::string, std::vector<AccountIDL>> IDL_INSTRUCTION_ACCOUNTS;
bool StaticThread::isAccountCompatibleOrder(llvm::StringRef& accountName1, llvm::StringRef& accountName2) {
    if (IDL_INSTRUCTION_ACCOUNTS.find(anchor_idl_instruction_name) != IDL_INSTRUCTION_ACCOUNTS.end()) {
        auto accounts = IDL_INSTRUCTION_ACCOUNTS.at(anchor_idl_instruction_name);
        auto size1 = accounts.size();
        auto size2 = anchorStructFunctionFields.size();
        if (size1 != size2) {
            // TODO: should report account changes
            if (DEBUG_RUST_API)
                llvm::outs() << "Imcompatible accounts size - "
                             << "idl: " << size1 << " this: " << size2 << "\n";
        }
        // if this size less than idl size, it means some accounts have been removed...
        auto size = size1;
        if (size1 > size2) {
            size = size2;
        }
        for (uint i = 0; i < size; i++) {
            auto item = accounts[i];
            // llvm::outs() << "idl account_name: " << item.name << " isMut: " << item.isMut
            //              << " isSigner: " << item.isSigner << " isNested: " << item.isNested << "\n";
            if (item.isNested) {
            } else {
                auto pair = anchorStructFunctionFields[i];
                // llvm::outs() << "current account_name: " << pair.first << " type: " << pair.second << "\n";
                auto accountName_anchor = convertToAnchorString(pair.first);
                if (item.name != accountName_anchor) {
                    // find the next account
                    for (uint j = i + 1; j < size; j++) {
                        auto pair2 = anchorStructFunctionFields[j];
                        // llvm::outs() << "next account_name: " << pair2.first << " type: " << pair2.second << "\n";
                        auto accountName2_anchor = convertToAnchorString(pair2.first);
                        if (item.name == accountName2_anchor) {
                            if (DEBUG_RUST_API)
                                llvm::outs()
                                    << "reordered account1: " << pair.first << " account2: " << pair2.first << "\n";
                            accountName1 = pair.first;
                            accountName2 = pair2.first;
                            return false;
                            // i = j+1;
                            // break;
                        }
                    }
                }
            }
        }
    }

    return true;
}

int StaticThread::isAccountCompatibleAddOrMut(llvm::StringRef accountName) {
    if (DEBUG_RUST_API)
        llvm::outs() << "isAccountCompatibleAddOrMut anchor_idl_instruction_name: " << anchor_idl_instruction_name
                     << " accountName: " << accountName << "\n";
    // for (auto& [api_name, items] : IDL_INSTRUCTION_ACCOUNTS) {
    //     llvm::outs() << "api_name: " << api_name << " items: " << items.size() << "\n";
    // }

    if (IDL_INSTRUCTION_ACCOUNTS.find(anchor_idl_instruction_name) != IDL_INSTRUCTION_ACCOUNTS.end()) {
        auto accounts = IDL_INSTRUCTION_ACCOUNTS.at(anchor_idl_instruction_name);
        // rule 1: add a new account
        // rule 2: account reorder
        // rule 3: changing immutable to mutable
        auto accountName_anchor = convertToAnchorString(accountName);
        bool isMut = accountsMutMap[accountName];
        if (DEBUG_RUST_API) llvm::outs() << "accountName_anchor: " << accountName_anchor << " isMut: " << isMut << "\n";
        {
            for (const auto& item : accounts) {
                // llvm::outs() << "account_name: " << item.name << " isMut: " << item.isMut << " isSigner: " <<
                // item.isSigner << "\n";
                if (item.name == accountName_anchor) {
                    // llvm::outs() << "IDL has_account: " << accountName << "\n";
                    if (isMut && !item.isMut) {
                        if (DEBUG_RUST_API) llvm::outs() << "IDL isMut changed to: " << isMut << "\n";
                        return 1;
                    }
                    return 0;
                }
            }
        }
        return -1;
    }
    return 0;
}
