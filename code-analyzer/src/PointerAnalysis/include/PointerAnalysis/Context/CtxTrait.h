//
// Created by peiming on 8/14/19.
//
#ifndef ASER_PTA_CTXTRAIT_H
#define ASER_PTA_CTXTRAIT_H

namespace xray {

template <typename ctx> class CtxTrait {
  using unknownTypeError = typename ctx::unknownTypeErrorType;
};

} // namespace xray

#endif
