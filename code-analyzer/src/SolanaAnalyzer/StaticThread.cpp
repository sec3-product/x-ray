#include "StaticThread.h"

#include "AccountIDL.h"
#include "Graph/Event.h"

using namespace xray;

TID StaticThread::curID = 0;
std::map<TID, StaticThread *> StaticThread::tidToThread;
std::map<TID, std::map<uint8_t, const llvm::Constant *>>
    StaticThread::threadArgs;

void StaticThread::setThreadArg(
    TID tid, std::map<uint8_t, const llvm::Constant *> &argMap) {
  threadArgs.insert(std::make_pair(tid, argMap));
};

std::map<uint8_t, const llvm::Constant *> *StaticThread::getThreadArg(TID tid) {
  if (threadArgs.count(tid)) {
    return &(threadArgs.at(tid));
  }
  return nullptr;
};

bool StaticThread::initIDLInstructionName() {
  if (startFunc) {
    llvm::SmallVector<StringRef, 3> sol_name_vec;
    startFunc->getName().split(sol_name_vec, '.', -1, false);
    if (sol_name_vec.size() == 3) {
      auto idl_instruction_name = sol_name_vec[1];
      // init_market => initMarket
      anchor_idl_instruction_name = convertToAnchorString(idl_instruction_name);
      if (DEBUG_RUST_API)
        llvm::outs() << "anchor_idl_instruction_name: "
                     << anchor_idl_instruction_name << "\n";
      return true;
    }
  }
  return false;
}

void StaticThread::updateMostRecentFuncReturn(const llvm::Function *func,
                                              llvm::StringRef valueName) {
  if (DEBUG_RUST_API)
    llvm::outs() << "updateMostRecentFuncReturn func: " << func->getName()
                 << " value: " << valueName << "\n";
  mostRecentFuncReturnMap[func] = valueName;
}

