//
// Created by peiming on 10/23/19.
//

#include "PointerAnalysis/Solver/PointsTo/BitVectorPTS.h"
#include "PointerAnalysis/Solver/PointsTo/PointedByPts.h"

namespace aser {
uint32_t BitVectorPTS::PTS_SIZE_LIMIT =
    std::numeric_limits<uint32_t>::max(); // no limit
std::vector<BitVectorPTS::PtsTy> BitVectorPTS::ptsVec;

uint32_t PointedByPts::PTS_SIZE_LIMIT =
    std::numeric_limits<uint32_t>::max(); // no limit
std::vector<PointedByPts::PtsTy> PointedByPts::pointsTo;
std::vector<PointedByPts::PtsTy> PointedByPts::pointedBy;
} // namespace aser