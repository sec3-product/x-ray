#include "Rules/CosplayAccountDetector.h"

#include <llvm/Support/raw_ostream.h>

#include "Collectors/CosplayAccount.h"
#include "DebugFlags.h"
#include "Graph/ReachGraph.h"
#include "SourceInfo.h"

namespace xray {

void CosplayAccountDetector::detect(const xray::ctx *ctx, TID tid) {
  for (auto &[func1, fieldTypes1] : normalStructFunctionFieldsMap) {
    for (auto &[func2, fieldTypes2] : normalStructFunctionFieldsMap) {
      if (func1 == func2) {
        continue;
      }

      auto size = fieldTypes1.size();
      auto size2 = fieldTypes2.size();
      if (size > size2) {
        continue;
      }

      bool mayCosplay = true;
      bool hasPubkey = false;
      for (size_t i = 0; i < size; i++) {
        auto type1 = fieldTypes1[i].second.substr(1);
        auto type2 = fieldTypes2[i].second.substr(1);
        if (!type1.equals(type2) || (fieldTypes1[i].first.contains("crimi") ||
                                     fieldTypes2[i].first.contains("crimi"))) {
          mayCosplay = false;
          break;
        }
        if (type1.contains("ubkey")) {
          hasPubkey = true;
        }
      }
      if (!mayCosplay || !hasPubkey) {
        continue;
      }

      auto inst1 = func1->getEntryBlock().getFirstNonPHI();
      auto inst2 = func2->getEntryBlock().getFirstNonPHI();
      auto e1 = graph->createApiReadEvent(ctx, inst1, tid);
      auto e2 = graph->createApiReadEvent(ctx, inst2, tid);
      auto file1 = getSourceLoc(e1->getInst()).getFilename();
      auto file2 = getSourceLoc(e2->getInst()).getFilename();
      if (DEBUG_RUST_API) {
        llvm::errs() << "==============file1: " << file1 << " file2: " << file2
                     << "============\n";
      }
      if (file1 == file2) {
        if (size == size2) {
          CosplayAccount::collect(e1, e2, callEventTraces,
                                  SVE::Type::COSPLAY_FULL, 6);
        } else {
          CosplayAccount::collect(e1, e2, callEventTraces,
                                  SVE::Type::COSPLAY_PARTIAL, 5);
        }
      }
    }
  }
}

} // namespace xray
