//
// Created by peiming on 3/30/20.
//

#ifndef ASER_PTA_STATISTICS_H
#define ASER_PTA_STATISTICS_H

#include <llvm/ADT/Statistic.h>

#define LOCAL_STATISTIC(VARNAME, DESC)                                               \
  llvm::Statistic VARNAME {DEBUG_TYPE, #VARNAME, DESC}

#endif  // ASER_PTA_STATISTICS_H
