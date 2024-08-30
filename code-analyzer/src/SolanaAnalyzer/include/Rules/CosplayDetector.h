#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/Function.h>

#include "Graph/Event.h"

namespace xray {

using FunctionFieldsMap =
    std::map<const llvm::Function *,
             std::vector<std::pair<llvm::StringRef, llvm::StringRef>>>;

class ReachGraph;

class CosplayDetector {
public:
  CosplayDetector(const FunctionFieldsMap &normalStructFunctionFieldsMap,
                  ReachGraph *graph,
                  std::map<TID, std::vector<CallEvent *>> &callEventTraces)
      : normalStructFunctionFieldsMap(normalStructFunctionFieldsMap),
        graph(graph), callEventTraces(callEventTraces) {}

  void detectCosplay(const xray::ctx *ctx, TID tid);

private:
  const FunctionFieldsMap &normalStructFunctionFieldsMap;
  ReachGraph *graph;
  std::map<TID, std::vector<CallEvent *>> &callEventTraces;
};

} // namespace xray
