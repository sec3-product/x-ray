//
// Created by peiming on 8/14/19.
//
#ifndef ASER_PTA_CTXTRAIT_H
#define ASER_PTA_CTXTRAIT_H

namespace aser {

template <typename ctx>
class CtxTrait {
    using unknownTypeError = typename ctx::unknownTypeErrorType;
};

}  // namespace aser

#endif
