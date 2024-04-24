//
// Created by peiming on 5/6/20.
//

#ifndef ASER_PTA_INFERPRODUCERCONSUMERPASS_H
#define ASER_PTA_INFERPRODUCERCONSUMERPASS_H

#include <llvm/IR/Dominators.h>
#include <llvm/Pass.h>

#include "Graph/Event.h"
#include "Graph/ReachGraph.h"
#include "PTAModels/GraphBLASModel.h"
#include "RaceDetectionPass.h"

namespace aser {

class MemAccessEvent;
class CallEvent;

class ProducerConsumerDetect {
private:
    // const PTA *pta;           // the pointer analysis instance
    // const ReachGraph *graph;  // the shb graph
    RaceDetectionPass &pass;
    bool isProducerConsumer(MemAccessEvent *ptr1, const std::vector<CallEvent *> &trace1, MemAccessEvent *ptr2,
                            const std::vector<CallEvent *> &trace2);

public:
    ProducerConsumerDetect(RaceDetectionPass &pass) : pass(pass) {}

    bool isProducerConsumerPair(MemAccessEvent *ptr1, const std::vector<CallEvent *> &trace1, MemAccessEvent *ptr2,
                                const std::vector<CallEvent *> &trace2);
};

}  // namespace aser

#endif  // ASER_PTA_INFERPRODUCERCONSUMERPASS_H
