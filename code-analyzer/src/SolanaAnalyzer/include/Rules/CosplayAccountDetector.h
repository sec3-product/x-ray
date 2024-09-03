#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/Function.h>

#include "Graph/Event.h"
#include "Rules/Rule.h"

namespace xray {

class ReachGraph;

class CosplayAccountDetector {
public:
  CosplayAccountDetector(
      const FunctionFieldsMap &normalStructFunctionFieldsMap, ReachGraph *graph,
      std::map<TID, std::vector<CallEvent *>> &callEventTraces)
      : normalStructFunctionFieldsMap(normalStructFunctionFieldsMap),
        graph(graph), callEventTraces(callEventTraces) {}

  void detect(const xray::ctx *ctx, TID tid);

private:
  const FunctionFieldsMap &normalStructFunctionFieldsMap;
  ReachGraph *graph;
  std::map<TID, std::vector<CallEvent *>> &callEventTraces;
};

} // namespace xray
