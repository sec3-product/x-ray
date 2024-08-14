//
// Created by peiming on 8/14/19.
//
//#include "aser/PointerAnalysis/Context/Context.h"
//#include "aser/PointerAnalysis/Context/CtxTrait.h"
//#include "aser/PointerAnalysis/Program/Program.h"
//
// llvm::cl::opt<uint32_t> K_Limiting("k", llvm::cl::init(3),
// llvm::cl::desc("k-limiting for k-CFA"));
//
// namespace aser {
//
////bool operator<(const K_CallSite &lhs, const K_CallSite &rhs) {
////    // 1st, order by depth
////    if (lhs.getDepth() == rhs.getDepth()) {
////        // 2nd, simply order by the pointer value stored in the vector
////        for (auto iter1 = lhs.cbegin(), iter2 = rhs.cbegin(); iter1 !=
/// lhs.cend(); iter1 ++, iter2++) { /            const llvm::Instruction *
/// const pp1 = *iter1; /            const llvm::Instruction * const pp2 =
/// *iter2;
////
////            if (pp1 == pp2) {
////                //same value, compare next
////                continue;
////            }
////            return pp1 < pp2;
////        }
////        //every element is equal, consider as equal context
////        return false;
////    }
////    return lhs.getDepth() < rhs.getDepth();
////}
//
//}