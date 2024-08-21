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
