//
// Created by peiming on 7/21/20.
//

#ifndef ASER_PTA_INTERCEPTRESULT_H
#define ASER_PTA_INTERCEPTRESULT_H

// forward declaration
namespace llvm {
class Value;
}

namespace xray {

struct InterceptResult {
  enum class Option {
    EXPAND_BODY,   // analyze and expand the body of the function
    ONLY_CALLSITE, // do not analyze into the function body, but keep the
                   // callsite
    IGNORE_FUN,    // ignore the function completely (no callnode in the
                   // callgraph).
  };

  const llvm::Value *redirectTo;
  Option option;

  InterceptResult(const llvm::Value *target, Option opt)
      : redirectTo(target), option(opt) {}
};

} // namespace xray
#endif // ASER_PTA_INTERCEPTRESULT_H
