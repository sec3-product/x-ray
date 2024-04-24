#include "ValueFlowAnalysis/SVFPass.h"

#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/IRBuilder.h>

#include <regex>

#include "CUDAModel.h"
#include "OMPModel.h"
#include "z3++.h"
using namespace z3;
using namespace std;
using namespace aser;
using namespace llvm;

extern bool FORTRAN_IR_MODE;
// static variables for scev
cl::opt<bool> DEBUG_SCEV("debug-svf", cl::desc("Turn on debug svf analysis"));
cl::opt<bool> DEBUG_SCEV_BOUND("debug-svf-bound", cl::desc("Turn on debug svf bound analysis"));
// cl::opt<bool> DEBUG_SCEV_SIM("debug-scev-sim", cl::desc("Print scev simplification debug info"));
cl::opt<bool> SHOW_SCEV_MSG("show-scev", cl::desc("Print scev debug info"));

cl::opt<bool> DO_NOT_USE_Z3("Xno-z3", cl::desc("Turn off static bound checking with Z3"));
static int z3_invoke_times = 0;
static const int Z3_TIMES_LIMIT = 10;
bool DEBUG_SCEV_SIM = false;
void setDebugAllSCEV(bool flag) {
    DEBUG_SCEV = flag;
    SHOW_SCEV_MSG = flag;
    DEBUG_SCEV_BOUND = flag;
}
const int CALL_LIMIT_DEPTH = 100;
static int cur_depth = 0;

// std::atomic<int> cur_region_id;
static std::regex regex_integer("(^|\\s)([\\+-]?([0-9]+\\.?[0-9]*|\\.?[0-9]+))(\\s|$)");

std::string DEBUGSTRING = "";
// store temporal result for scev
std::string SCEV_TEMP_RESULT = "";
bool USE_YACC_PARSER = true;
extern int test_scev_parser(const char *yy_str);

// cache for alias query results
static map<const llvm::Value *, std::string> valueKeyStringResultCache;
static map<const llvm::SCEV *, std::string> scevKeyStringResultCache;

std::string findStringFromLLVMValue(const Value *v) {
    if (valueKeyStringResultCache.find(v) != valueKeyStringResultCache.end()) {
        return valueKeyStringResultCache.at(v);
    } else {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        os << *v;
        auto result = os.str();
        valueKeyStringResultCache[v] = result;
        return result;
    }
}
std::string findStringFromSCEVValue(const SCEV *scev) {
    if (scevKeyStringResultCache.find(scev) != scevKeyStringResultCache.end()) {
        return scevKeyStringResultCache.at(scev);
    } else {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        os << *scev;
        auto result = os.str();
        scevKeyStringResultCache[scev] = result;  // never use scev cache
        return result;
    }
}
bool balanceParentheses(string &expr) {
    size_t count_left = std::count(expr.begin(), expr.end(), '(');
    size_t count_right = std::count(expr.begin(), expr.end(), ')');
    if (count_left == count_right)
        return true;
    else
        return false;
}

size_t findBalanceParenthese(string &expr, size_t start) {
    int size = expr.size();
    if (start >= size) return string::npos;
    for (size_t i = start; i < size; i++) {
        char c = expr[i];
        if (c == ')') {
            if (DEBUG_SCEV_SIM) llvm::outs() << "found balance index ): " << i << "\n";
            return i;
        } else if (c == '(') {
            if (DEBUG_SCEV_SIM) llvm::outs() << "found more index (: " << i << "\n";
            i = findBalanceParenthese(expr, i + 1);
        }
    }

    return string::npos;
}

string findNextToken(string &expr) {
    while (expr.front() == ' ') expr = expr.substr(1);

    for (size_t i = 0; i < expr.size(); i++) {
        if (expr[i] == ' ' || expr[i] == ')') {
            auto token = expr.substr(0, i);
            if (DEBUG_SCEV_SIM) llvm::outs() << "token: " << token << "\n\n";
            return token;
        }
    }

    string token = expr.substr(0, expr.size());
    if (DEBUG_SCEV_SIM) llvm::outs() << "token: " << token << "\n\n";

    return token;
}

// find all unknowns in a given SCEV expr
class SCEVUnknownHandler : public llvm::SCEVRewriteVisitor<SCEVUnknownHandler> {
private:
    std::set<llvm::Value *> unknowns;
    SVFPass *svf;
    using super = SCEVRewriteVisitor<SCEVUnknownHandler>;

public:
    SCEVUnknownHandler(llvm::ScalarEvolution &SE, SVFPass *svf) : super(SE), svf(svf) {}

    const llvm::SCEV *visitUnknown(const llvm::SCEVUnknown *Expr) {
        // it could be a function argument

        unknowns.insert(Expr->getValue());

        return Expr;
    }
    std::set<llvm::Value *> &findAllUnknowns(const llvm::SCEV *scev) {
        super::visit(scev);
        return unknowns;
    }
};

class SCEVUnknownRewriter : public SCEVRewriteVisitor<SCEVUnknownRewriter> {
public:
    static const SCEV *rewrite(const SCEV *Scev, ScalarEvolution &SE,
                               DenseMap<const Value *, const llvm::SCEV *> &Map) {
        SCEVUnknownRewriter Rewriter(SE, Map);
        return Rewriter.visit(Scev);
    }

    SCEVUnknownRewriter(ScalarEvolution &SE, DenseMap<const Value *, const llvm::SCEV *> &M)
        : SCEVRewriteVisitor(SE), Map(M) {}

    const SCEV *visitUnknown(const SCEVUnknown *Expr) {
        Value *V = Expr->getValue();
        if (Map.count(V)) {
            // Value *NV = Map[V];
            // return SE.getUnknown(NV);
            return Map[V];
        }
        return Expr;
    }

private:
    DenseMap<const Value *, const llvm::SCEV *> &Map;
};

class SCEVSimplifyRewriter : public SCEVRewriteVisitor<SCEVSimplifyRewriter> {
public:
    static const SCEV *rewrite(const SCEV *Scev, ScalarEvolution &SE) {
        SCEVSimplifyRewriter Rewriter(SE);
        return Rewriter.visit(Scev);
    }

    SCEVSimplifyRewriter(ScalarEvolution &SE) : SCEVRewriteVisitor(SE) {}

    const SCEV *visitAddRecExpr(const SCEVAddRecExpr *Expr) {
        auto *start = Expr->getStart();

        // if (auto *op = dyn_cast<SCEVNAryExpr>(start)) start = op->getOperand(0);

        if (DEBUG_SCEV_SIM)
            llvm::outs() << "simplifying SCEVAddRecExpr: " << *Expr << "      -------to------->      " << *start
                         << "\n";
        start = this->visit(start);
        return start;
    }

    const llvm::SCEV *visitSignExtendExpr2(const llvm::SCEVSignExtendExpr *Expr) {
        auto *Operand = Expr->getOperand();

        if (DEBUG_SCEV_SIM)
            llvm::outs() << "simplifying SignExtendExpr: " << *Expr << "      -------to------->      " << *Operand
                         << "\n";
        Operand = this->visit(Operand);

        return Operand;
    }
};

// move add operation out the (sext) SCEV
class SCEVBoundAnalyzer : public llvm::SCEVRewriteVisitor<SCEVBoundAnalyzer> {
private:
    using super = SCEVRewriteVisitor<SCEVBoundAnalyzer>;

public:
    static const SCEV *analyze(const SCEV *Scev, ScalarEvolution &SE) {
        SCEVBoundAnalyzer Rewriter(SE);
        return Rewriter.visit(Scev);
    }
    SCEVBoundAnalyzer(llvm::ScalarEvolution &SE) : super(SE) {}

    const llvm::SCEV *visitAddRecExpr(const llvm::SCEVAddRecExpr *Expr) {
        // if (Expr->isAffine())
        {
            auto op = Expr->getOperand(0);
            // if (llvm::isa<SCEVUnknown>(op))
            {
                llvm::outs() << "--------- loop bound analysis for scev -----------      " << *Expr << "\n";

                auto loop = Expr->getLoop();
                auto bounds = loop->getBounds(SE);
                // auto initValue = bounds->getInitialIVValue();
                // auto finalValue = bounds->getFinalIVValue();
                llvm::outs() << "Loop initial value: " << bounds->getInitialIVValue()
                             << "\nLoop final value: " << bounds->getFinalIVValue() << "\n";

                auto step = Expr->getOperand(1);
                if (DEBUG_SCEV) llvm::outs() << "Loop step: " << *step << "\n";

                auto maxTripCount = SE.getSmallConstantMaxTripCount(Expr->getLoop());
                if (DEBUG_SCEV) llvm::outs() << "Loop max trip count: " << maxTripCount << "\n";

                auto backEdgeCount = SE.getBackedgeTakenCount(loop);

                if (DEBUG_SCEV) llvm::outs() << "Loop backEdge count: " << *backEdgeCount << "\n";

                return backEdgeCount;
            }
        }

        return Expr;
    }
};

void SCEVDataItem::dump() {
    for (auto [v, data] : unknownMap) {
        auto scev = data->getSCEV();
        // ***COULDNOTCOMPUTE*** - means top-level arguments
        // if (scev->isZero()) continue;

        llvm::outs() << "unknown " << *v << "      ---------------------->      " << *scev << "\n";
        data->dump();
    }
}

static bool isAnyExternalInferface(const llvm::Function *func) {
    return CUDAModel::isAnyCUDACall(func) || OMPModel::isStaticForInit(func) || OMPModel::isStaticForFini(func) ||
           OMPModel::isGetThreadNum(func) || OMPModel::isGetGlobalThreadNum(func) || OMPModel::isDispatchNext(func) ||
           OMPModel::isDispatchInit(func) || OMPModel::isFork(func);  // also exclude recursive __kmpc_fork_call
}

void SVFModel::simplifySCEVExpression(string &expr) {
    const char *input = expr.c_str();
    if (DEBUG_SCEV_SIM) llvm::outs() << "parse: " << input << "\n\n";
    test_scev_parser(input);
    return;
}
std::map<std::string, std::string> globalNameConstantValueMap;
z3::expr getZ3ExprFromString(context &c, std::string scev, std::map<std::string, z3::expr> &symbolMap) {
    // expr scev1 = (8 * tid1) + (160 * xi) + xa;
    // expr scev2 = (8 * tid2) + (160 * (1 + xi)) + xa;
    if (DEBUG_SCEV_SIM) llvm::outs() << "\n\nz3scev: " << scev << "\n\n";

    while (scev.front() == ' ') scev = scev.substr(1);

    auto left_expr = c.int_const("xyz");

    if (scev.front() == '(') {
        // find the next balance "d"
        size_t pos = findBalanceParenthese(scev, 1);
        if (pos != string::npos) {
            left_expr = getZ3ExprFromString(c, scev.substr(1, pos - 1), symbolMap);
            if (pos == scev.size()) return left_expr;
            scev = scev.substr(pos + 1);
        }
    }

    auto token = findNextToken(scev);
    if (token.front() == '*') {
        size_t found_times = scev.find("* ");  // return the pos of the FIRST character that matches
        string right_part = scev.substr(found_times + 2);
        return left_expr * getZ3ExprFromString(c, right_part, symbolMap);

    } else if (token.front() == '/') {
        size_t found_div = scev.find("/u ");  // return the pos of the FIRST character that matches
        string right_part = scev.substr(found_div + 3);
        return left_expr * getZ3ExprFromString(c, right_part, symbolMap);

    } else if (token.front() == '+') {
        size_t found_plus = scev.find("+ ");  // return the pos of the FIRST character that matches
        string right_part = scev.substr(found_plus + 2);
        return left_expr + getZ3ExprFromString(c, right_part, symbolMap);

    } else {
        if (token.empty()) return left_expr;  // empty token
        auto token_size = token.size();

        if (token.front() == '@') {
            if (globalNameConstantValueMap.find(token) != globalNameConstantValueMap.end()) {
                auto value = globalNameConstantValueMap.at(token);
                if (DEBUG_SCEV_SIM) llvm::outs() << "found global token: " << token << " value: " << value << "\n\n";
                token = value;  // token size may change here
            }
        }

        if (token == "undef") token = "%" + token;
        // if (token == "%.uplevelArgPack0001_397") token = "x397";

        //% or constant
        if (symbolMap.find(token) == symbolMap.end()) {
            if (token.front() == '%' || token.front() == '@') {
                auto xi = c.int_const(token.c_str());
                symbolMap.insert({token, xi});
            } else if (regex_match(token, regex_integer)) {
                auto xi = c.int_val(token.c_str());
                symbolMap.insert({token, xi});

            } else {
                // something is wrong
                if (DEBUG_SCEV_SIM) llvm::outs() << "wrong parsing token: " << token << "\n\n";
                auto xi = c.int_const(token.c_str());
                symbolMap.insert({token, xi});
            }
        }
        left_expr = symbolMap.at(token);

        if (scev.size() == token_size) return left_expr;

        scev = scev.substr(token_size);

        token = findNextToken(scev);
        // must be * or +
        if (token.front() == '*') {
            size_t found_times = scev.find("* ");  // return the pos of the FIRST character that matches
            string right_part = scev.substr(found_times + 2);
            return left_expr * getZ3ExprFromString(c, right_part, symbolMap);

        } else if (token.front() == '/') {
            size_t found_div = scev.find("/u ");  // return the pos of the FIRST character that matches
            string right_part = scev.substr(found_div + 3);
            return left_expr * getZ3ExprFromString(c, right_part, symbolMap);

        } else if (token.front() == '+') {
            size_t found_plus = scev.find("+ ");  // return the pos of the FIRST character that matches
            string right_part = scev.substr(found_plus + 2);
            return left_expr + getZ3ExprFromString(c, right_part, symbolMap);

        } else {
            // something is wrong
            if (DEBUG_SCEV_SIM) llvm::outs() << "wrong parsing expr: " << scev << " token: " << token << "\n\n";
        }
    }

    return left_expr;
}
// assumption: make sure F has function body, not a declaration
llvm::ScalarEvolution *SVFPass::getOrCreateFunctionSCEV(const llvm::Function *F) {
    bool OPT = false;

    // it seems ScalarEvolution segfault across multiple functions
    // if (!OPT) {
    //     auto se = &this->getAnalysis<ScalarEvolutionWrapperPass>(const_cast<llvm::Function &>(*F)).getSE();  //
    //     // scalar evolution
    //     return se;
    // }

    if (OPT && seMap.find(F) != seMap.end()) {
        // if (DEBUG_SCEV) llvm::outs() << "GOT cached se for function : " << F->getName() << "\n";

        return seMap.at(F);

    } else {
        Function &F2 = const_cast<llvm::Function &>(*F);
        // DominatorTree domTree = DominatorTree(F2);
        // LoopInfo loopInfo = LoopInfo();
        // if (DEBUG_SCEV) llvm::outs() << "CREATing new se for function : " << F->getName() << "\n";

        ScalarEvolution *se = new ScalarEvolution(F2, getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F2),
                                                  getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F2),
                                                  getAnalysis<DominatorTreeWrapperPass>(F2).getDomTree(),
                                                  getAnalysis<LoopInfoWrapperPass>(F2).getLoopInfo());
        seMap[F] = se;

        return se;
    }
}

// maintain a cache: syntacticalCache

bool SVFPass::hasSyntacticalSignal(const llvm::Function *F, const llvm::Value *v) {
    if (syntacticalCache.find(v) != syntacticalCache.end()) return syntacticalCache.at(v);

    auto vsourceLoc = getSourceLoc(v);
    // fortran flang tends to generate memory access on array indices
    //>108|    DO k=y_min-depth,y_max+y_inc+depth
    if (FORTRAN_IR_MODE) {
        auto line = vsourceLoc.getSourceLine();
        // line.find("DO ") != string::npos ||
        if (line.find("ENDDO") != string::npos) {
            // llvm::outs() << "\n============== skipped fortran DO source line: " << line;
            syntacticalCache[v] = true;
            return true;
        }
    }
    {
        // check branch
        // AMG: my_thread = hypre_GetThreadNum();
        // if (my_thread == 0)
        // regular expression
        // std::regex regex_ifthreadzero("(if([\\s]*)\\(([\\s]*)([a-zA-Z_0-9]*)thread([\\s]*)==([\\s]*)0([\\s]*)\\))");
        // if (regex_match(vsourceLoc.getSnippet(), regex_ifthreadzero)) {
        //     llvm::outs() << "filter out regex_ifthreadzero: " << vsourceLoc.getSnippet() << "\n";
        //     syntacticalCache[v] = true;
        //     return true;
        // }
    }

    auto vSource = vsourceLoc.getSnippet();

    auto se = getOrCreateFunctionSCEV(F);
    auto scev = se->getSCEV((llvm::Value *)v);

    if (SHOW_SCEV_MSG) {
        llvm::outs() << "\n============== func: " << demangle(F->getName().str());
        llvm::outs() << "\n============== v: " << *v;
        llvm::outs() << "\n============== v's scev: " << *scev;
        llvm::outs() << "\n============== v's source code: \n" << vSource << "\n";
    }
    SCEVUnknownHandler unknownHandler(*se, this);
    for (auto *un : unknownHandler.findAllUnknowns(scev)) {
        // print out source code line for debug
        // if (auto *inst = dyn_cast<Instruction>(un))

        if (SHOW_SCEV_MSG) llvm::outs() << "\n\nSyntacticalSignal checking unknown: " << *un << "\n";
        if (un->hasName())
            if (un->getName().contains("first") || un->getName().contains("start") || un->getName().contains("lower")) {
                syntacticalCache[v] = true;
                return true;
            }
        // 1. value name
        // 2. line source
        // 3. snippet

        // TODO: skip races that have scev unknowns, whose source code line contains start,end...
        auto sourceLoc = getSourceLoc(un);
        if (SHOW_SCEV_MSG) llvm::outs() << "at source code line: \n" << sourceLoc.getSourceLine();
        if (sourceLoc.getSourceLine().find("first") != string::npos ||
            sourceLoc.getSourceLine().find("start") != string::npos ||
            sourceLoc.getSourceLine().find("lower") != string::npos) {
            syntacticalCache[v] = true;
            return true;
        }
        if (SHOW_SCEV_MSG) llvm::outs() << "in source code snippet: \n" << sourceLoc.getSnippet() << "\n";
        if ((sourceLoc.getSnippet().find("first") != string::npos &&
             sourceLoc.getSnippet().find("last") != string::npos) ||
            (sourceLoc.getSnippet().find("start") != string::npos &&
             sourceLoc.getSnippet().find("end") != string::npos) ||
            (sourceLoc.getSnippet().find("lower") != string::npos &&
             sourceLoc.getSnippet().find("upper") != string::npos)) {
            syntacticalCache[v] = true;
            return true;
        }
    }

    syntacticalCache[v] = false;
    return false;
}

int compareSCEVStringMatch(std::string &SCEV_RESULT_1, std::string &SCEV_RESULT_2, const Value *v1, const Value *v2) {
    //     inst:   %49 = bitcast %struct_drb159_0_* @_drb159_0_ to i32*, !dbg !62
    // SCEV_RESULT_1: (28 + (4 * %tid40) + @_drb159_0_) i: 0 j1: 32
    // SCEV_RESULT_2: @_drb159_0_ i: 0 j2: 10
    // ============== scev1 substr: 28 + 4 * %tid40 + @_drb159_0_
    // ============== scev2 substr: @_drb159_0_
    // avoid the fortran IR problem: if bitcast then return no race
    if (FORTRAN_IR_MODE) {
        if (!isa<BitCastInst>(v1) && isa<BitCastInst>(v2)) {
            if (auto *bitcast = dyn_cast<BitCastInst>(v2)) {
                if (bitcast->getSrcTy()->isPointerTy())
                    if (bitcast->getSrcTy()->getPointerElementType()->isStructTy()) return 0;
            }
        }

        if (isa<BitCastInst>(v1) && !isa<BitCastInst>(v2)) {
            if (auto *bitcast = dyn_cast<BitCastInst>(v1)) {
                if (bitcast->getSrcTy()->isPointerTy())
                    if (bitcast->getSrcTy()->getPointerElementType()->isStructTy()) return 0;
            }
        }
        // to avoid PTA issues -- BIG
        //         ============== scev1 substr: 8 * %tid + -8 * 48 + %.uplevelArgPack0001_567 + 16 +
        //         %.uplevelArgPack0001_567
        // ============== scev2 substr: %.uplevelArgPack0001_567
        if (SCEV_RESULT_1.find("%tid") == string::npos && SCEV_RESULT_2.find("%tid") != string::npos &&
            SCEV_RESULT_2.find("%.uplevelArgPack") != string::npos)
            return 0;

        if (SCEV_RESULT_2.find("%tid") == string::npos && SCEV_RESULT_1.find("%tid") != string::npos &&
            SCEV_RESULT_1.find("%.uplevelArgPack") != string::npos)
            return 0;
    }

    //__nv_MAIN_F1L26_3Arg2
    // 16 + %__nv_MAIN_F1L26_3Arg2
    // if substring, then no alias
    int size1 = SCEV_RESULT_1.size();
    int size2 = SCEV_RESULT_2.size();
    int size_min = std::min(size1, size2);  // pick min

    if (size1 < size2) {
        if (SCEV_RESULT_2.find(SCEV_RESULT_1) != string::npos && SCEV_RESULT_2.find("%tid") == string::npos) return 0;

    } else if (size1 > size2) {
        if (SCEV_RESULT_1.find(SCEV_RESULT_2) != string::npos && SCEV_RESULT_1.find("%tid") == string::npos) return 0;
    }
    //(-4 + (4 * %i_320.0) + @.BSS1)
    //(-8 + (4 * %i_320.0) + @.BSS1)

    // heuristics: most scev the same?
    // start from the left
    int i = 0;
    while (i < size_min) {
        auto substr1 = SCEV_RESULT_1.substr(i);
        auto substr2 = SCEV_RESULT_2.substr(i);
        auto token1 = findNextToken(substr1);
        auto token2 = findNextToken(substr2);
        if (token1 == token2 && !token1.empty())
            i = i + token1.size();
        else
            break;
        // if (SCEV_RESULT_1.at(i) != SCEV_RESULT_2.at(i)) break;
    }
    int j1 = size1 - 1;
    int j2 = size2 - 1;
    for (; j1 > 0 && j2 > 0; j1--, j2--) {
        if (SCEV_RESULT_1.at(j1) != SCEV_RESULT_2.at(j2)) break;
    }

    auto substr1 = SCEV_RESULT_1.substr(i, j1 + 1);
    auto substr2 = SCEV_RESULT_2.substr(i, j2 + 1);

    // get rid of "(" and ")"
    std::string chars = "()";
    for (char c : chars) {
        substr1.erase(std::remove(substr1.begin(), substr1.end(), c), substr1.end());
        substr2.erase(std::remove(substr2.begin(), substr2.end(), c), substr2.end());
    }
    if (SHOW_SCEV_MSG) {
        llvm::outs() << "SCEV_RESULT_1: " << SCEV_RESULT_1 << " i: " << i << " j1: " << j1 << "\n";
        llvm::outs() << "SCEV_RESULT_2: " << SCEV_RESULT_2 << " i: " << i << " j2: " << j2 << "\n";
        llvm::outs() << "============== scev1 substr: " << substr1 << "\n";
        llvm::outs() << "============== scev2 substr: " << substr2 << "\n";
        // if both are constant, then return no race
    }
    // std::regex regex_float("-?[0-9]+\\.[0-9]+");
    if (regex_match(substr1, regex_integer) && regex_match(substr2, regex_integer)) {
        // equal size and both contain tid, return potential race
        if (SCEV_RESULT_1.find("%tid") != string::npos && SCEV_RESULT_2.find("%tid") != string::npos)
            return 2;
        else
            return 0;
    }

    // size_t pos1 = SCEV_RESULT_1.find("+");
    // size_t pos2 = SCEV_RESULT_2.find("+");
    // if (pos1 == pos2) return Priority::NORACE;

    return 2;

    // TODO: generate constraints and solve them

    // if (SCEV_RESULT_1.find("tid") != string::npos && SCEV_RESULT_2.find("tid") != string::npos)
    //     return Priority::NORACE;

    // return Priority::NORACE;  // not alias

    // any parameters from omp_outlined should be considered as global
    // func: .omp_outlined., var: %2
}
// TODO: maintain a cache
int SVFPass::mayAlias(const llvm::Function *ompEntryFunc, const ctx *ctx1, const llvm::Instruction *caller1_inst,
                      const llvm::Instruction *inst1, const Value *v1, const ctx *ctx2,
                      const llvm::Instruction *caller2_inst, const llvm::Instruction *inst2, const Value *v2) {
    // if (true) return Priority::NORACE;
    if (SHOW_SCEV_MSG) llvm::outs() << "\n\n***********  checking alias constraints  *********\n";
    const llvm::Function *f1 = inst1->getFunction();
    const llvm::Function *f2 = inst2->getFunction();
    // const llvm::Instruction *caller_inst = nullptr;  // calling context?
    // TODO: if there are the same gep, no need to recompute twice, but just to use the tid
    std::string SCEV_RESULT_1;
    if (scevStringResultCache.find(v1) != scevStringResultCache.end()) {
        SCEV_RESULT_1 = scevStringResultCache.at(v1);
    } else {
        auto se1 = getOrCreateFunctionSCEV(f1);
        auto scev1 = se1->getSCEV((llvm::Value *)v1);

        // todo: bound analysis is complex
        //        auto bound = SCEVBoundAnalyzer::analyze(scev1, *se1);
        //        SCEV_TEMP_RESULT = findStringFromSCEVValue(bound);
        //        cur_depth = 0;  // reset current limit to 0
        //        auto scev1_data_bound = getGlobalSCEVInternal(ctx1, caller1_inst, f1, v1);
        //        SVFPass::printSCEVItem(scev1_data_bound->f, scev1_data_bound->getUnknownMap());
        //        if (SHOW_SCEV_MSG)
        //            llvm::outs() << "============== bound after replacing unknowns: " << SCEV_TEMP_RESULT << "\n";
        //        SVFModel::simplifySCEVExpression(SCEV_TEMP_RESULT);
        //        if (SHOW_SCEV_MSG) llvm::outs() << "\n============== simplified bound: " << SCEV_TEMP_RESULT << "\n";

        //        for (const auto &entry: boundsMap){
        //            auto inst = entry.first;
        //            auto pair = entry.second;
        //
        //            auto lb = pair.first;
        //            auto ub = pair.second;
        //            llvm::outs() << "static init inst: "<< *inst<<"\n";
        //            llvm::outs() << "lower bound: "<< *lb<<"\n";
        //            llvm::outs() << "upper bound: "<< *ub<<"\n";
        //
        //            auto se = getOrCreateFunctionSCEV(inst->getFunction());
        //            auto scev_lb = se->getSCEV((llvm::Value *)lb);
        //            auto scev_ub = se->getSCEV((llvm::Value *)ub);
        //            llvm::outs() << "lower bound scev: "<< *scev_lb<<"\n";
        //            llvm::outs() << "upper bound scev: "<< *scev_ub<<"\n";
        //        }

        scev1 = SCEVSimplifyRewriter::rewrite(scev1, *se1);

        // std::string buffer;
        // llvm::raw_string_ostream os(buffer);
        // os << *scev1;
        // TESTSCEV = os.str();
        SCEV_TEMP_RESULT = findStringFromSCEVValue(scev1);
        if (SHOW_SCEV_MSG) llvm::outs() << "\n============== e1's top-level scev: " << *scev1 << "\n";
        cur_depth = 0;  // reset current limit to 0
        auto scev1_data = getGlobalSCEVInternal(ctx1, caller1_inst, f1, v1);
        SVFPass::printSCEVItem(scev1_data->f, scev1_data->getUnknownMap());
        if (SHOW_SCEV_MSG)
            llvm::outs() << "============== e1's scev after replacing unknowns: " << SCEV_TEMP_RESULT << "\n";
        SVFModel::simplifySCEVExpression(SCEV_TEMP_RESULT);
        if (SHOW_SCEV_MSG)
            llvm::outs() << "============== e1's final scev after simplification: " << SCEV_TEMP_RESULT << "\n";

        SCEV_RESULT_1 = SCEV_TEMP_RESULT;
        scevStringResultCache[v1] = SCEV_RESULT_1;
    }
    // if v1 and v2 are the same, then only check the scev result contains %tid or not
    if (v1 == v2) {
        // llvm::outs() << "\n======v1 == v2======== scev: " << SCEV_RESULT_1 << "\n";

        if (SCEV_RESULT_1.find("%tid") != string::npos)
            return 0;
        else {
            // TODO: handle indirect (derived from array indexed by tid)
            // heuristic: typically if there is %, then no race
            auto tmp_scev = SCEV_RESULT_1;
            auto index = tmp_scev.find("%this");
            if (index != std::string::npos) tmp_scev.replace(index, 5, "");
            if (tmp_scev.find("+") != string::npos) return 0;

            if (DEBUG_SCEV)
                llvm::outs() << "\n============== FINAL scev after simplification: " << SCEV_RESULT_1 << "\n";
            return 2;
        }
    }
    std::string SCEV_RESULT_2;
    if (scevStringResultCache.find(v2) != scevStringResultCache.end()) {
        SCEV_RESULT_2 = scevStringResultCache.at(v2);
    } else {
        auto se2 = getOrCreateFunctionSCEV(f2);
        auto scev2 = se2->getSCEV((llvm::Value *)v2);
        // auto bound2 = SCEVBoundAnalyzer::analyze(scev2, *se2);

        scev2 = SCEVSimplifyRewriter::rewrite(scev2, *se2);

        SCEV_TEMP_RESULT = findStringFromSCEVValue(scev2);
        if (SHOW_SCEV_MSG) llvm::outs() << "\n============== e2's top-level scev: " << *scev2 << "\n";
        cur_depth = 0;  // reset current limit to 0
        auto scev2_data = getGlobalSCEVInternal(ctx2, caller2_inst, f2, v2);
        SVFPass::printSCEVItem(scev2_data->f, scev2_data->getUnknownMap());
        if (SHOW_SCEV_MSG)
            llvm::outs() << "============== e2's scev after replacing unknowns: " << SCEV_TEMP_RESULT << "\n";
        SVFModel::simplifySCEVExpression(SCEV_TEMP_RESULT);
        if (SHOW_SCEV_MSG)
            llvm::outs() << "============== e2's final scev after simplification: " << SCEV_TEMP_RESULT << "\n";

        SCEV_RESULT_2 = SCEV_TEMP_RESULT;
        scevStringResultCache[v2] = SCEV_RESULT_2;
    }
    // check if scev1_data and scev2_data are equivalent
    // TODO: generate constraints and solve them
    if (SCEV_RESULT_1 == SCEV_RESULT_2 && SCEV_RESULT_2.find("%tid") != string::npos) {
        return 0;
    }

    // if same inst and both contain %tid, return no race
    if (inst1 == inst2 && SCEV_RESULT_1.find("%tid") != string::npos && SCEV_RESULT_2.find("%tid") != string::npos)
        return 0;
    else {
        // llvm::outs() << "scev1: " << SCEV_RESULT_1 << "\nscev2: " << SCEV_RESULT_2 << "\n";
    }
    // turn this on to use static bound checking
    if (!DO_NOT_USE_Z3)                                  //&& !FORTRAN_IR_MODE
        if (SCEV_RESULT_2.find("%tid") != string::npos)  // contains tid and are different
        {
            if (SHOW_SCEV_MSG) llvm::outs() << "scev1: " << SCEV_RESULT_1 << "\nscev2: " << SCEV_RESULT_2 << "\n";

            // when we get here, we are being conservative, so we may report a false positive race, e.g. DRB053
            // scev1 = (8 * %tid) + (160 * %i) + %a
            // scev2 = (8 * %tid) + (160 * (1 + %i)) + %a

            context c;
            solver s(c);
            // params p(c);
            // p.set(":timeout", 5000u);
            // s.set(p);
            expr tid1 = c.int_const("%tid1");
            expr tid2 = c.int_const("%tid2");

            /*
                    expr xi = c.int_const("%i");
                    expr xa = c.int_const("%a");
                    //expr scev1 = (8 * tid1) + (160 * xi) + xa;
                    //expr scev2 = (8 * tid2) + (160 * (1 + xi)) + xa;
                    expr const_1 = c.int_val("8");
                    expr const_2 = c.int_val("160");
                    expr const_3 = c.int_val("1");

                    expr scev1 = (const_1 * tid1) + (const_2 * xi) + xa;
                    expr scev2 = (const_1 * tid2) + (const_2 * (const_3 + xi)) + xa;
            */
            // todo write our own parser
            std::map<std::string, z3::expr> symbolMap;
            symbolMap.insert({"%tid1", tid1});
            symbolMap.insert({"%tid2", tid2});

            if (SCEV_RESULT_1.find("%tid") == string::npos) {
                if (SHOW_SCEV_MSG) llvm::outs() << "CONSERVATIVE race!\n";
                return 2;  // return potential race
            }
            // for large program, the scev may contain multiple tids
            // scev1: (8 * %tid15) + %vnewc
            // scev2: (8 * %tid17) + %vnewc
            auto index1 = SCEV_RESULT_1.find("tid");
            size_t i = index1 + 3;
            for (; i < SCEV_RESULT_1.size(); i++) {
                if (SCEV_RESULT_1[i] == ' ' || SCEV_RESULT_1[i] == ')') {
                    break;
                }
            }
            SCEV_RESULT_1.replace(index1, i - index1, "tid1");

            auto index2 = SCEV_RESULT_2.find("tid");
            size_t j = index2 + 3;
            for (; j < SCEV_RESULT_2.size(); j++) {
                if (SCEV_RESULT_2[j] == ' ' || SCEV_RESULT_2[j] == ')') {
                    break;
                }
            }
            SCEV_RESULT_2.replace(index2, j - index2, "tid2");

            // let's be conservative: if % count does not match, no race
            auto count1 = std::count(SCEV_RESULT_1.begin(), SCEV_RESULT_1.end(), '%');
            auto count2 = std::count(SCEV_RESULT_2.begin(), SCEV_RESULT_2.end(), '%');
            // llvm::outs() << "count1: " << count1 << " count2: " << count2 << "\n";
            if (count1 > 1 && count2 > 1 && count2 != count1) return 0;

            if (z3_invoke_times > Z3_TIMES_LIMIT) return compareSCEVStringMatch(SCEV_RESULT_1, SCEV_RESULT_2, v1, v2);

            expr scev1 = getZ3ExprFromString(c, SCEV_RESULT_1, symbolMap);
            expr scev2 = getZ3ExprFromString(c, SCEV_RESULT_2, symbolMap);

            expr conjecture = scev1 == scev2;

            // the following does not work
            //        auto result = Z3_eval_smtlib2_string(c,"(and (= (8 * tid1) (8 * tid2)) (not (= tid1 tid2)))");
            //        std::cout << "result: "<<result<<"\n";

            s.add(conjecture);
            s.add(tid1 != tid2);
            // todo get the bounds
            // expr xtid = abs(tid2-tid1);//abs crashes in z3
            if (ompEntryFunc && boundsFunctionMap.find(ompEntryFunc) != boundsFunctionMap.end()) {
                if (DEBUG_SCEV_BOUND) {
                    llvm::outs() << "\nchecking bounds for ompEntryFunc: " << ompEntryFunc->getName() << "\n";
                    llvm::outs() << "scev1: " << SCEV_RESULT_1 << "\nscev2: " << SCEV_RESULT_2 << "\n";
                }
                auto pair = boundsFunctionMap[ompEntryFunc];
                auto inst = boundsFunctionStaticInitInstMap[ompEntryFunc];

                auto lb = pair.first;
                auto ub = pair.second;

                if (DEBUG_SCEV_BOUND) {
                    llvm::outs() << "lower bound value: " << *lb << "\n";
                    llvm::outs() << "upper bound value: " << *ub << "\n";
                }

                // todo retrieved precomputed scev fro lb and ub directly
                auto pair_data = boundsFunctionScevDataMap[ompEntryFunc];
                auto scev_lb_data = pair_data.first;
                auto scev_ub_data = pair_data.second;

                auto se_lb = getOrCreateFunctionSCEV(inst->getFunction());
                auto scev_lb = se_lb->getSCEV((llvm::Value *)lb);
                SCEV_TEMP_RESULT = findStringFromSCEVValue(scev_lb);
                SVFPass::printSCEVItem(scev_lb_data->f, scev_lb_data->getUnknownMap());
                SVFModel::simplifySCEVExpression(SCEV_TEMP_RESULT);
                auto lb_str = SCEV_TEMP_RESULT;

                auto se_ub = getOrCreateFunctionSCEV(inst->getFunction());
                auto scev_ub = se_ub->getSCEV((llvm::Value *)ub);
                SCEV_TEMP_RESULT = findStringFromSCEVValue(scev_ub);
                SVFPass::printSCEVItem(scev_ub_data->f, scev_ub_data->getUnknownMap());
                SVFModel::simplifySCEVExpression(SCEV_TEMP_RESULT);

                auto ub_str = SCEV_TEMP_RESULT;

                if (lb_str.size() > 0 && ub_str.size() > 0) {
                    if (DEBUG_SCEV_BOUND) {
                        llvm::outs() << "lower bound scev: " << lb_str << "\n";
                        llvm::outs() << "upper bound scev: " << ub_str << "\n";
                    }

                    if (FORTRAN_IR_MODE) {
                        // avoid the fortran IR problem:
                        // for fortran we skip bounds with unknown
                        if (lb_str.front() == '%' || ub_str.front() == '%' ||
                            (lb_str.find("%") != string::npos && ub_str.find("%") != string::npos) /*||
                            SCEV_RESULT_1.find("uplevelArgPack0001_397") != string::npos*/) {  // avoid crash in Z3
                            return compareSCEVStringMatch(SCEV_RESULT_1, SCEV_RESULT_2, v1, v2);
                        }
                    }

                    expr lower = getZ3ExprFromString(c, lb_str, symbolMap);
                    expr upper = getZ3ExprFromString(c, ub_str, symbolMap);
                    expr bound = upper - lower + 1;
                    s.add(tid2 - tid1 < bound);
                    s.add(tid1 - tid2 < bound);

                    // print out model of z3 s
                    if (DEBUG_SCEV_BOUND) llvm::outs() << "z3 smt constraints: \n" << s.to_smt2() << "\n";
                    auto result = s.check();  // OH BAD, Z3 CRASHES ON VALID CONSTRAINTS DRB053 FORTRAN
                    z3_invoke_times++;
                    if (DEBUG_SCEV_BOUND) llvm::outs() << "z3 times: " << (z3_invoke_times) << "\n";
                    if (DEBUG_SCEV_BOUND) llvm::outs() << "z3 result: " << result << "\n";

                    switch (result) {
                        case z3::check_result::unsat:
                            if (DEBUG_SCEV_BOUND) llvm::outs() << "not race!\n\n";
                            return 0;
                        case z3::check_result::sat:
                            if (DEBUG_SCEV_BOUND) llvm::outs() << "race!\n\n";
                            // llvm::outs() << "counterexample:\n" << s.get_model() << "\n";//crash on DRB033
                            return 2;
                        case z3::check_result::unknown:
                            if (DEBUG_SCEV_BOUND) llvm::outs() << "unknown\n\n";
                            break;
                    }
                } else {
                    if (llvm::isa<ConstantInt>(lb) && llvm::isa<ConstantInt>(ub)) {
                        auto lb_str = findStringFromSCEVValue(scev_lb);
                        auto ub_str = findStringFromSCEVValue(scev_ub);

                        expr lower = c.int_val(lb_str.c_str());
                        expr upper = c.int_val(ub_str.c_str());
                        expr bound = upper - lower + 1;
                        s.add(tid2 - tid1 < bound);
                        s.add(tid1 - tid2 < bound);

                        switch (s.check()) {
                            case unsat:
                                if (DEBUG_SCEV_BOUND) llvm::outs() << "not race!\n\n";
                                return 0;
                            case sat:
                                if (DEBUG_SCEV_BOUND) llvm::outs() << "race!\n\n";
                                // llvm::outs() << "counterexample:\n" << s.get_model() << "\n";//crash on DRB033
                                return 2;
                            case unknown:
                                if (DEBUG_SCEV_BOUND) llvm::outs() << "unknown\n\n";
                                break;
                        }
                    }
                }
            } else {
                // let's print out for debug
                if (DEBUG_SCEV_BOUND) {
                    llvm::outs() << "\nNOT checking bounds for ompEntryFunc: " << ompEntryFunc->getName() << "\n";
                    llvm::outs() << "scev1: " << SCEV_RESULT_1 << "\nscev2: " << SCEV_RESULT_2 << "\n";
                }
            }

            // todo solution: track bounds or use dynamic validation
            if (SHOW_SCEV_MSG) llvm::outs() << "CONSERVATIVE race!\n";
            return 1;  // return potential race
        }
    // TODO： need a constraint solving engine
    //    SCEV_TEMP_RESULT = "( - ("+SCEV_RESULT_1 + ") + " +SCEV_RESULT_2 + ")";
    //    SVFModel::simplifySCEVExpression(SCEV_TEMP_RESULT);
    //    if (SHOW_SCEV_MSG)
    //        llvm::outs() << "============== final scev1-scev2: " << SCEV_TEMP_RESULT << "\n";

    return compareSCEVStringMatch(SCEV_RESULT_1, SCEV_RESULT_2, v1, v2);
}
void SVFPass::printSCEVItem(const llvm::Function *F, DenseMap<const Value *, SCEVDataItem *> &unknownMap) {
    for (auto [v, data] : unknownMap) {
        // auto scev = data->getSCEV();
        // ***COULDNOTCOMPUTE*** - means top-level arguments
        // if (scev->isZero()) continue;
        auto se = getOrCreateFunctionSCEV(data->f);
        auto scev = se->getSCEV((llvm::Value *)data->v);
        // todo: bound does not work
        //        auto bound = SCEVBoundAnalyzer::analyze(scev, *se);
        //        if (DEBUG_SCEV) {
        //            llvm::outs() <<"bound of "<< *v<<" : "<< *bound<<"\n";
        //        }

        scev = SCEVSimplifyRewriter::rewrite(scev, *se);

        if (SHOW_SCEV_MSG)
            llvm::outs() << "unknown [func: " << F->getName() << ", var: " << *v
                         << "]      ---------------------->     [func: " << data->f->getName() << ", var: " << *scev
                         << "]\n";
        std::string k_str = findStringFromLLVMValue(v);

        std::string key_str;

        size_t found_percent = k_str.find('%');  // must exist
        if (found_percent != string::npos)
            key_str = k_str.substr(found_percent);
        else {
            // something must be wrong
        }

        size_t found_assign = k_str.find('=');
        if (found_assign != string::npos) {
            key_str = key_str.substr(0, found_assign - 3);
        }

        std::regex e(key_str + "([^0-9]|$)");

        std::string v_str = findStringFromSCEVValue(scev);

        if (key_str != v_str) SCEV_TEMP_RESULT = std::regex_replace(SCEV_TEMP_RESULT, e, v_str + "$1");

        if (DEBUG_SCEV) {
            llvm::outs() << "\nreplacing key_str: " << key_str << " with " << v_str << "\n";
            llvm::outs() << "after one round replacement: " << SCEV_TEMP_RESULT << "\n\n";
        }
        printSCEVItem(data->f, data->getUnknownMap());
    }
}
const llvm::Value *trackBounds(const DominatorTree *DT, const llvm::Value *val, const llvm::Instruction *val_inst) {
    val = val->stripPointerCasts();
    for (auto U : val->users()) {
        // make sure U dominates val
        if (auto SI = llvm::dyn_cast<llvm::StoreInst>(U))  // store inst
        {
            if (DT->dominates(SI, val_inst)) {
                auto *value = SI->getValueOperand();
                if (auto cons_int_value = llvm::dyn_cast<llvm::ConstantInt>(value)) {
                    if (DEBUG_SCEV) llvm::outs() << "bound of " << *val << " is: " << *cons_int_value << "\n";
                    return cons_int_value;
                } else {
                    // TODO
                    if (DEBUG_SCEV) llvm::outs() << "indirect bound of " << *val << " is: " << *value << "\n";
                    return value;
                }
            }
        } else if (auto *bitcast = dyn_cast<BitCastInst>(U))  // handle fortran
        {
            // llvm::outs() << "track bound debug: " << *val << " inst: " << *val_inst << "\n";
        }
    }

    // fortran
    // ub: %8 -> de0003p.copy_478 -> %3 -> de0003p_366 -> 8
    // lb: %7 -> dl0003p.copy_477 -> %2 -> dl0003p_370 -> 1
    //   call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10,
    //   i32 1), !dbg !99
    // store i32 8, i32* %.de0003p_366, align 4, !dbg !99
    // 343   store i32 1, i32* %.di0003p_367, align 4, !dbg !99
    // 344   %1 = load i32, i32* %.di0003p_367, align 4, !dbg !99
    // 345   store i32 %1, i32* %.ds0003p_368, align 4, !dbg !99
    // 346   store i32 1, i32* %.dl0003p_370, align 4, !dbg !99
    // 347   %2 = load i32, i32* %.dl0003p_370, align 4, !dbg !99
    // 348   store i32 %2, i32* %.dl0003p.copy_477, align 4, !dbg !99
    // 349   %3 = load i32, i32* %.de0003p_366, align 4, !dbg !99
    // 350   store i32 %3, i32* %.de0003p.copy_478, align 4, !dbg !99
    // 351   %4 = load i32, i32* %.ds0003p_368, align 4, !dbg !99
    // 352   store i32 %4, i32* %.ds0003p.copy_479, align 4, !dbg !99
    // 353   %5 = load i32, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !99
    // 354   %6 = bitcast i32* %.i0000p_334 to i64*, !dbg !99
    // 355   %7 = bitcast i32* %.dl0003p.copy_477 to i64*, !dbg !99
    // 356   %8 = bitcast i32* %.de0003p.copy_478 to i64*, !dbg !99
    // 357   %9 = bitcast i32* %.ds0003p.copy_479 to i64*, !dbg !99
    // 358   %10 = load i32, i32* %.ds0003p.copy_479, align 4, !dbg !99
    // 359   call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32
    // %10, i32 1), !dbg !99

    return nullptr;
}

// quick hack:
static const llvm::Function *svf_ompEntryFunc = nullptr;
// todo: make svf pass per omp engine
void SVFPass::connectSCEVOMPEntryFunctionArgs(const ctx *context, const llvm::Instruction *const ompForkCall,
                                              const llvm::Function *ompEntryFunc) {
    svf_ompEntryFunc = ompEntryFunc;
    //  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t*
    //  %.kmpc_loc.addr, i32 5, void (i32*, i32*, ...)* %11,
    //               i32* %m, i64 100, i64 100, double* %vla, i32* %i), !dbg !64
    /*__kmpc_fork_call
     * arg1: loc source location information
     * arg2: argc total number of arguments in the ellipsis
     * arg3: microtask pointer to callback routine consisting of outlined parallel construct
     * ...:  pointers to shared variables that aren’t global
     *
     * omp_outlined.(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid.,
     *              i32* nocapture readonly dereferenceable(4) %m, i64 %vla, i64 %vla1, double* nocapture
     * dereferenceable(8) %b, i32* nocapture readonly dereferenceable(4) %i) arg1: %.global_tid., arg2: %.bound_tid
     */
    int start_omp_entry = 2;
    int start_omp_fork_call = 3;

    if (DEBUG_SCEV) llvm::outs() << "connectSCEVOMPEntryFunctionArgs ompForkCall: " << *ompForkCall << "\n";
    CallSite CS(ompForkCall);

    int i = start_omp_fork_call;
    for (Function::const_arg_iterator I = ompEntryFunc->arg_begin() + start_omp_entry, E = ompEntryFunc->arg_end();
         I != E; ++I) {
        const Argument *Arg = I;

        auto call_arg = CS.getArgOperand(i);
        if (call_arg->getType()->isIntegerTy() || call_arg->getType()->isPointerTy() ||
            call_arg->getType()->isArrayTy()) {  // only care about integer type

            if (DEBUG_SCEV) {
                llvm::outs() << "connecting  arg: " << *call_arg << "   to  " << *Arg << "\n";
            }
            // now, get SCEV of the call_arg
            // this can be expensive
            cur_depth = 0;  // reset current limit to 0
            auto scev_data = getGlobalSCEVInternal(context, nullptr, ompForkCall->getFunction(), call_arg);

            // if (callArgScevMap.find(ompForkCall) != callArgScevMap.end()) {
            //     auto &argmap = callArgScevMap[ompForkCall];
            //     argmap[Arg] = scev_data;
            // } else {
            //     DenseMap<const Value *, SCEVDataItem *> argmap;
            //     argmap[Arg] = scev_data;
            //     callArgScevMap[ompForkCall] = argmap;
            // }

            callArgOmpForkScevMap[Arg] = scev_data;

            // JEFF: DEBUG
            // if (func->getName() == "_Z10DoCriticalii") {
            //     llvm::outs() << "connecting call inst: " << *inst << "  arg: " << *call_arg << "   to  " << *Arg
            //                  << "\n";
            //     setDebugAllSCEV(true);
            //     SVFPass::printSCEVItem(scev_data->f, scev_data->getUnknownMap());
            // }
            // setDebugAllSCEV(false);

            // let's search if there is any previously unresolved unknown relevant to Arg
            if (unresolvedUnknowns.find(ompForkCall) != unresolvedUnknowns.end()) {
                auto &argmap = unresolvedUnknowns[ompForkCall];
                if (argmap.find(Arg) != argmap.end()) {
                    auto *data = argmap[Arg];
                    data->addUnknownValue(Arg, scev_data);
                }
            }
        }
        i++;
    }
}
void SVFPass::connectSCEVFunctionArgs(const ctx *context, const llvm::Instruction *caller_inst,
                                      const llvm::Function *F_caller, const llvm::Instruction *inst) {
    if (callInstCache.find(inst) != callInstCache.end()) {
        return;
    } else {
        callInstCache.insert(inst);  // cache it and proceed
    }
    CallSite CS(inst);
    auto func = CS.getTargetFunction();
    if (!func) return;

    // for CUDA
    // i32 @llvm.nvvm.read.ptx.sreg.tid.{x,y,z}	threadIdx.{x,y,z}
    // i32 @llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}	blockIdx.{x,y,z}
    // i32 @llvm.nvvm.read.ptx.sreg.ntid.{x,y,z}	blockDim.{x,y,z}
    // i32 @llvm.nvvm.read.ptx.sreg.nctaid.{x,y,z}	gridDim.{x,y,z}
    // void @llvm.nvvm.barrier0()	__syncthreads()

    if (func->isIntrinsic() && !func->getName().startswith("llvm.nvvm.")) return;  // skip llvm.debug/llvm.lifetime

    if (DEBUG_SCEV)
        llvm::outs() << "connecting args in func: " << F_caller->getName() << " for call: " << *inst << "\n";

    if (CS.isIndirectCall()) {
        return;
    }  // TODO: handle indirect

    if (isAnyExternalInferface(func)) {
        // TODO: replacing it with a special TID
        if (OMPModel::isStaticForInit(func)) {
            //__kmpc_for_static_init_4
            // 1 gtid Global thread id of this thread
            // 4 Pointer to the lower bound
            if (auto *call = dyn_cast<llvm::CallBase>(inst)) {
                // llvm::outs() << "*** setting call arg " << *inst->getOperand(1) << " to gtid: " <<
                // func->getName()<<
                // "\n"; llvm::outs() << "*** setting call arg " << *inst->getOperand(4) << " to tid: " <<
                // func->getName()<< "\n";

                // track bounds
                // call void @__kmpc_for_static_init_4(%struct.ident_t* %.kmpc_loc.addr, i32 %12, i32 34, i32*
                // %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1), !dbg !80 store i32
                // 19, i32* %.omp.ub, align 4, !dbg !81, !tbaa !24 store i32 0, i32* %.omp.lb, align 4, !dbg !81,
                // !tbaa !24
                auto lb = call->getOperand(4);  //%.omp.lb
                auto ub = call->getOperand(5);  //%.omp.ub

                auto DT =
                    &this->getAnalysis<DominatorTreeWrapperPass>(const_cast<llvm::Function &>(*F_caller)).getDomTree();

                auto lower_bound = trackBounds(DT, lb, inst);
                auto upper_bound = trackBounds(DT, ub, inst);
                if (lower_bound && upper_bound) {
                    // save to bounds map
                    // auto id = cur_region_id++;
                    auto ompEntryFunc = caller_inst->getFunction();
                    if (svf_ompEntryFunc) ompEntryFunc = svf_ompEntryFunc;
                    // is it possible for the same func to have more than one static-for-init?
                    if (boundsFunctionMap.find(ompEntryFunc) != boundsFunctionMap.end()) {
                        if (DEBUG_SCEV)
                            llvm::outs() << "!!function contains multiple static-for-init: " << ompEntryFunc->getName()
                                         << "\n";
                        // one function may contain multiple __kmpc_for_static_init_4
                        // for such cases we may fail to infer the correct bounds,so be conservative
                    } else {
                        if (DEBUG_SCEV)
                            llvm::outs() << "saving bounds for ompEntryFunc: " << ompEntryFunc->getName() << "\n";
                        boundsFunctionMap[ompEntryFunc] = std::make_pair(lower_bound, upper_bound);
                        boundsFunctionStaticInitInstMap[ompEntryFunc] = inst;

                        // compute scev for lower_bound and upper_bound
                        cur_depth = 0;  // reset current limit to 0
                        auto scev_lb_data = getGlobalSCEVInternal(context, caller_inst, F_caller, lower_bound);

                        cur_depth = 0;  // reset current limit to 0
                        auto scev_ub_data = getGlobalSCEVInternal(context, caller_inst, F_caller, upper_bound);
                        boundsFunctionScevDataMap[ompEntryFunc] = std::make_pair(scev_lb_data, scev_ub_data);

                        // FOR DEBUG
                        // auto se_lb = getOrCreateFunctionSCEV(F_caller);
                        // auto scev_lb = se_lb->getSCEV((llvm::Value *)lower_bound);
                        // SCEV_TEMP_RESULT = findStringFromSCEVValue(scev_lb);
                        // SVFPass::printSCEVItem(scev_lb_data->f, scev_lb_data->getUnknownMap());
                        // llvm::outs() << "============== lb_str lower bound scev after replacing unknowns: "
                        //              << SCEV_TEMP_RESULT << "\n";

                        // auto se_ub = getOrCreateFunctionSCEV(F_caller);
                        // auto scev_ub = se_ub->getSCEV((llvm::Value *)upper_bound);
                        // SCEV_TEMP_RESULT = findStringFromSCEVValue(scev_ub);
                        // SVFPass::printSCEVItem(scev_ub_data->f, scev_ub_data->getUnknownMap());
                        // llvm::outs() << "============== ub_str upper bound scev after replacing unknowns: "
                        //              << SCEV_TEMP_RESULT << "\n";
                    }
                }

                if (auto *bitcast = dyn_cast<BitCastInst>(call->getOperand(1))) {
                    bitcast->getOperand(0)->setName("gtid");
                    // llvm::outs() << "setting var to gtid: " << *bitcast << "\n";
                } else {
                    call->getOperand(1)->setName("gtid");
                }
                if (auto *bitcast = dyn_cast<BitCastInst>(call->getOperand(4))) {
                    bitcast->getOperand(0)->setName("tid");
                    // llvm::outs() << "setting var to tid: " << *bitcast << "\n";
                } else {
                    call->getOperand(4)->setName("tid");
                }
            }
        } else if (OMPModel::isDispatchNext(func) || OMPModel::isDispatchInit(func)) {
            //__kmpc_dispatch_next_4
            // 1 gtid Global thread id of this thread
            // 3 Pointer to the lower bound
            if (auto *call = dyn_cast<llvm::CallBase>(inst)) {
                // call->getOperand(1)->setName("gtid");
                // call->getOperand(3)->setName("tid");

                if (auto *bitcast = dyn_cast<BitCastInst>(call->getOperand(1))) {
                    bitcast->getOperand(0)->setName("gtid");
                    // llvm::outs() << "setting var to gtid: " << *bitcast << "\n";
                } else {
                    call->getOperand(1)->setName("gtid");
                }

                if (auto *bitcast = dyn_cast<BitCastInst>(call->getOperand(3))) {
                    bitcast->getOperand(0)->setName("tid");
                    // llvm::outs() << "setting var to tid: " << *bitcast << "\n";
                } else {
                    call->getOperand(3)->setName("tid");
                }
            }
        } else if (OMPModel::isGetThreadNum(func)) {
            // if (auto *call = dyn_cast<llvm::InvokeInst>(inst))
            {
                // llvm::outs() << "*** setting call arg " << *inst->getOperand(0) << " to tid: " <<
                // func->getName()<<
                // "\n";
                // inst->getOperand(0)->setName("tid");
            }((llvm::Value *)inst)
                ->setName("tid");

        } else if (OMPModel::isGetGlobalThreadNum(func)) {
            // llvm::outs() << "*** setting call arg " << *inst->getOperand(0) << " to gtid: " << func->getName() <<
            // "\n";
            inst->getOperand(0)->setName("gtid");
        } else if (CUDAModel::isGetThreadIdX(func)) {
            ((llvm::Value *)inst)->setName("tid.x");
        } else if (CUDAModel::isGetThreadIdY(func)) {
            ((llvm::Value *)inst)->setName("tid.y");
        } else if (CUDAModel::isGetThreadIdZ(func)) {
            ((llvm::Value *)inst)->setName("tid.z");
        } else if (CUDAModel::isGetBlockIdX(func)) {
            ((llvm::Value *)inst)->setName("bid.x");
        } else if (CUDAModel::isGetBlockIdY(func)) {
            ((llvm::Value *)inst)->setName("bid.y");
        } else if (CUDAModel::isGetBlockIdZ(func)) {
            ((llvm::Value *)inst)->setName("bid.z");
        } else if (CUDAModel::isGetBlockDimX(func)) {
            ((llvm::Value *)inst)->setName("dim.x");
        } else if (CUDAModel::isGetBlockDimY(func)) {
            ((llvm::Value *)inst)->setName("dim.y");
        } else if (CUDAModel::isGetBlockDimZ(func)) {
            ((llvm::Value *)inst)->setName("dim.z");
        } else if (CUDAModel::isGetGridDimX(func)) {
            ((llvm::Value *)inst)->setName("gid.x");
        } else if (CUDAModel::isGetGridDimX(func)) {
            ((llvm::Value *)inst)->setName("gid.y");
        } else if (CUDAModel::isGetGridDimX(func)) {
            ((llvm::Value *)inst)->setName("gid.z");
        }
        // set the return of these func as special
        // auto se = getOrCreateFunctionSCEV(func);
        // auto nullItem = new SCEVDataItem(se->getCouldNotCompute());
        // callRetScevMap[inst] = nullItem;

        return;  // no need to further process these external functions
    }

    if (func->isDeclaration()) {
        // golden place to mark external input
        if (DEBUG_SCEV) llvm::outs() << "TODO: golden place to consider external func: " << func->getName() << "\n";
        return;
    }
    // DEBUG_SCEV = false;
    // if (func->getName() == "_Z3geti") DEBUG_SCEV = true;
    // basic assumption: caller's arguments have been resolved
    // now connect arg with call arg
    if (CS.getNumArgOperands() > 0) {
        int i = 0;
        for (Function::const_arg_iterator I = func->arg_begin(), E = func->arg_end(); I != E; ++I) {
            const Argument *Arg = I;

            if (CS.getNumArgOperands() <= i) {
                break;
            }
            auto call_arg = CS.getArgOperand(i);
            if (call_arg->getType()->isIntegerTy() || call_arg->getType()->isPointerTy() ||
                call_arg->getType()->isArrayTy()) {  // only care about integer type

                if (DEBUG_SCEV) {
                    llvm::outs() << "connecting call inst: " << *inst << "  arg: " << *call_arg << "   to  " << *Arg
                                 << "\n";
                    if (caller_inst) llvm::outs() << "caller_inst is: " << *caller_inst << "\n";
                }
                // now, get SCEV of the call_arg
                // this can be expensive
                cur_depth = 0;  // reset current limit to 0
                auto scev_data = getGlobalSCEVInternal(context, caller_inst, F_caller, call_arg);

                if (callArgScevMap.find(inst) != callArgScevMap.end()) {
                    auto &argmap = callArgScevMap[inst];
                    argmap[Arg] = scev_data;
                } else {
                    DenseMap<const Value *, SCEVDataItem *> argmap;
                    argmap[Arg] = scev_data;
                    callArgScevMap[inst] = argmap;
                }

                // if caller_inst is __kmpc_fork_call, add
                if (caller_inst && OMPModel::isFork(caller_inst)) {
                    callArgOmpForkScevMap[Arg] = scev_data;
                    // llvm::outs() << "callArgOmpForkScevMap caller_inst: " << *caller_inst << "  arg: " << *call_arg
                    // << "   to  " << *Arg
                    //              << "\n";
                }

                // JEFF: DEBUG
                // if (func->getName() == "_Z10DoCriticalii") {
                //     llvm::outs() << "connecting call inst: " << *inst << "  arg: " << *call_arg << "   to  " <<
                //     *Arg
                //                  << "\n";
                //     setDebugAllSCEV(true);
                //     SVFPass::printSCEVItem(scev_data->f, scev_data->getUnknownMap());
                // }
                // setDebugAllSCEV(false);

                // let's search if there is any previously unresolved unknown relevant to Arg
                if (unresolvedUnknowns.find(inst) != unresolvedUnknowns.end()) {
                    auto &argmap = unresolvedUnknowns[inst];
                    if (argmap.find(Arg) != argmap.end()) {
                        auto *data = argmap[Arg];
                        data->addUnknownValue(Arg, scev_data);
                    }
                }
            }
            i++;
        }
    }

    // DEBUG_SCEV = false;

    // return values
    // if (!func->getReturnType()->isVoidTy()) {  // only care about integer type
    if (func->getReturnType()->isIntegerTy()) {  // only care about integer type

        for (const BasicBlock &BB : func->getBasicBlockList()) {
            if (isa<ReturnInst>(BB.getTerminator())) {
                auto v2 = BB.getTerminator()->getOperand(0);
                cur_depth = 0;  // reset current limit to 0
                auto scev2_data = getGlobalSCEVInternal(context, inst, func, v2);

                callRetScevMap[inst] = scev2_data;
                // consider return only once
                break;
            }
        }
    }
}
const llvm::SCEV *SVFPass::getGlobalSCEV(const ctx *context, const llvm::Instruction *caller_inst,
                                         const llvm::Function *F, const llvm::Value *v) {
    // interesting, once you rename an instruction, the ids of other instructions of will
    // be automatically changed
    if (false) {
        auto se = getOrCreateFunctionSCEV(F);
        auto scev = se->getSCEV((llvm::Value *)v);
        llvm::outs() << "inst: " << *v << " scev: " << *scev << "\n";
        return scev;
    }

    llvm::outs() << "\n\n============== function " << F->getName() << " inst: " << *v << "\n";

    cur_depth = 0;  // reset current limit to 0
    SCEVDataItem *data = getGlobalSCEVInternal(context, caller_inst, F, v);
    auto se = getOrCreateFunctionSCEV(data->f);
    auto scev = se->getSCEV((llvm::Value *)data->v);
    llvm::outs() << "\n============== final scev: " << *scev << " =============\n";
    // llvm::outs() << "\n============== final scev: " << *data->getSCEV() << " =============\n";
    // data->dump();
    SVFPass::printSCEVItem(data->f, data->getUnknownMap());
    llvm::outs() << "=======================================================================\n\n\n\n";

    return data->getSCEV();
}

bool SVFPass::flowsFromAny(llvm::Instruction *inst, std::vector<std::string> keys) {
    // auto s = getSourceLoc(inst).getSourceLine();
    // if (s.find("zetot[z] += dwork") != std::string::npos) {
    //     llvm::outs() << s << "\n";
    // }

    llvm::outs() << "===BASE " << *inst << "\n";
    // TODO: dont repeat all this work
    auto se = getOrCreateFunctionSCEV(inst->getFunction());
    auto scev = se->getSCEV(inst);
    SCEVUnknownHandler unknownHandler(*se, this);
    for (auto const val : unknownHandler.findAllUnknowns(scev)) {
        auto source = getSourceLoc(val).getSnippet();
        llvm::outs() << "SCEV BRAD " << *val << "\n" << source;
        for (auto const &key : keys) {
            if (source.find(key) != std::string::npos) {
                return true;
            }
            if (val->getName().find(key) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

// caller_inst: the caller instruction of F
// F: the function containing v
SCEVDataItem *SVFPass::getGlobalSCEVInternal(const ctx *context, const llvm::Instruction *caller_inst,
                                             const llvm::Function *F, const llvm::Value *v) {
    // if (F->getName() == ".omp_outlined._debug__") DEBUG_SCEV = true;
    if (DEBUG_SCEV) llvm::outs() << DEBUGSTRING << "SCEV func " << F->getName() << " pointer inst: " << *v << "\n";

    if (F->isDeclaration()) {
        llvm::outs() << "DECLARATION ONLY FUNC: " << F->getName() << "\n";
    }
    if (scevCache.find(v) != scevCache.end()) return scevCache.at(v);
    if (DEBUG_SCEV) DEBUGSTRING += "    ";

    auto se = getOrCreateFunctionSCEV(F);
    // make sure v is from F
    if (auto *inst_tmp = dyn_cast<llvm::Instruction>(v)) {
        if (inst_tmp->getFunction() != F)
            if (DEBUG_SCEV)
                llvm::outs() << "ERROR ERROR ERROR: v->func: " << inst_tmp->getFunction()->getName() << " v: " << *v
                             << "\n";
    } else {
    }

    auto scev = se->getSCEV((llvm::Value *)v);
    if (DEBUG_SCEV) llvm::outs() << DEBUGSTRING << "        scev: " << *scev << "\n";

    SCEVDataItem *data = new SCEVDataItem(scev, F, (llvm::Value *)v);
    // FIX a stack over flow bug: limit the depth of getGlobalSCEVInternal call
    if (cur_depth++ > CALL_LIMIT_DEPTH) return data;

    // TODO: disjunc all unknowns and merge each into the current scev

    // logger::endPhase();

    // llvm::outs() << "VAL: " << *v << "\n" << getSourceLoc(v).getSourceLine();
    SCEVUnknownHandler unknownHandler(*se, this);
    for (auto *un : unknownHandler.findAllUnknowns(scev)) {
        if (DEBUG_SCEV) {
            llvm::outs() << DEBUGSTRING << "              scev unknown: " << *un << "\n";
        }

        // if (F->getName() == "_Z10DoCriticalii")
        //     llvm::outs() << "TEST scev unknown: " << *un << "  in scev: " << *scev << "   v:  " << *v << "\n";

        if (auto *argu = dyn_cast<Argument>(un)) {
            // TODO: only need to handle top-level function
            if (caller_inst == nullptr) {
                // connect with external input interfaces such as get_thread_id, etc

                if (DEBUG_SCEV)
                    llvm::outs() << "AHA caller_inst is nullptr! you need more context sensitivity for caller of func: "
                                 << F->getName() << "\n";
                // make sure the parameters are set to unknown
                // let's just set it to notcomputable
                auto nullItem = new SCEVDataItem(se->getCouldNotCompute(), F, un);  // do not use un
                data->addUnknownValue(un, nullItem);
            } else {
                if (DEBUG_SCEV)
                    llvm::outs() << "trying to match unknown arg: " << *un << "  with inst: " << *caller_inst << "\n";
                // make sure the caller_inst is correct
                if (callArgScevMap[caller_inst].find(un) != callArgScevMap[caller_inst].end()) {
                    auto scev_data_cached = callArgScevMap[caller_inst][un];
                    data->addUnknownValue(un, scev_data_cached);

                    if (DEBUG_SCEV) {
                        llvm::outs() << "matched cached results for unknown: " << *un << "\n";
                        // SVFPass::printSCEVItem(scev_data_cached->f, scev_data_cached->getUnknownMap());
                    }

                } else {
                    if (DEBUG_SCEV)
                        llvm::outs() << "does not match any cached result for unknown: " << *un << " try more\n";
                    // the caller might be wrong?
                    // try one more time to get it
                    if (callArgOmpForkScevMap.find(un) != callArgOmpForkScevMap.end()) {
                        auto scev_data_cached = callArgOmpForkScevMap[un];
                        data->addUnknownValue(un, scev_data_cached);

                        if (DEBUG_SCEV) {
                            llvm::outs() << "matched cached results for unknown in callArgOmpForkScevMap: " << *un
                                         << "\n";
                            // SVFPass::printSCEVItem(scev_data_cached->f, scev_data_cached->getUnknownMap());
                        }

                    } else {
                        // TODO: special parameter for unknowns
                        // auto nullItem = new SCEVDataItem(se->getZero(un->getType()));

                        // IRBuilder
                        // llvm::IRBuilder<> Builder(un->getContext());
                        // auto newValue = Builder.CreateGlobalString(StringRef("GlobalXYZ"), "xyz");
                        // auto nullItem = new SCEVDataItem(se->getUnknown(newValue), F, un);
                        if (DEBUG_SCEV) llvm::outs() << "finally did not find a cached result\n";
                        // if we are here, it means we come across unknowns that should be resolved by future calls
                        // to do this correctly, we may need a redesign or a fixpoint computation...

                        // save it to unresolvedUnknowns
                        unresolvedUnknowns[caller_inst][un] = data;
                    }
                }
            }

        } else if (llvm::isa<llvm::CallBase>(un)) {
            llvm::Instruction *call_inst = (llvm::Instruction *)un;

            if (DEBUG_SCEV)
                llvm::outs() << DEBUGSTRING << "-----------------  resolving unknown ret: " << *call_inst << "\n";

            CallSite CS(call_inst);
            // compute the scev of callee return
            auto func = CS.getTargetFunction();
            if (!func || CS.isIndirectCall()) continue;  // TODO: handle indirect call

            // make sure the func retun type is primitive?

            if (func->getReturnType()->isIntegerTy()) {
                // now, go to get the scev2 for the return value v2
                if (callRetScevMap.find(call_inst) != callRetScevMap.end()) {
                    data->addUnknownValue(un, callRetScevMap[call_inst]);
                    continue;
                } else if (func->isDeclaration() || isAnyExternalInferface(func)) {
                    auto nullItem = new SCEVDataItem(se->getCouldNotCompute(), F, un);
                    data->addUnknownValue(un, nullItem);
                    callRetScevMap[call_inst] = nullItem;
                    continue;
                }
                if (DEBUG_SCEV)
                    llvm::outs() << DEBUGSTRING << "-----------------  resolving additional call: " << *call_inst
                                 << "\n";

                for (const BasicBlock &BB : func->getBasicBlockList()) {
                    if (isa<ReturnInst>(BB.getTerminator())) {
                        // this call must return non-void
                        auto v2 = BB.getTerminator()->getOperand(0);
                        auto scev2_data = getGlobalSCEVInternal(context, call_inst, func, v2);
                        // suppose scev2 only has unknown from caller CS2
                        // replace those unknown with the scevs of parameters

                        // let's handle parameters here
                        if (CS.getNumArgOperands() > 0) {
                            int i = 0;
                            for (Function::const_arg_iterator I = func->arg_begin(), E = func->arg_end(); I != E; ++I) {
                                const Argument *Arg = I;

                                auto call_arg = CS.getArgOperand(i);

                                if (DEBUG_SCEV)
                                    llvm::outs()
                                        << DEBUGSTRING << "-----------------  matching call arg: " << *call_arg << "\n";

                                // IMPORTANT: get the global scev of the call arg
                                //
                                if (!call_arg->getType()->isIntOrPtrTy()) {
                                    continue;
                                }
                                auto scev_arg_item = getGlobalSCEVInternal(context, caller_inst, F, call_arg);

                                if (DEBUG_SCEV)
                                    llvm::outs() << DEBUGSTRING << "-----------------  matched func arg: " << *Arg
                                                 << " --> call arg scev: " << *scev_arg_item->getSCEV() << "\n";
                                i++;

                                // instead: add to the SCEVDataItem
                                // we add only for unknown values
                                // llvm::outs() << "checking arg in unknowns of scev2_data: " << *Arg << "\n";
                                if (scev2_data->hasUnknown((llvm::Value *)Arg)) {
                                    // llvm::outs() << "adding arg to unknowns of scev2_data: " << *Arg << "\n";
                                    scev2_data->addUnknownValue(Arg, scev_arg_item);
                                }
                            }
                            // it seems rewrite does not work across multiple functions - need another solution
                            // auto call_scev = SCEVUnknownRewriter::rewrite(scev2, *se2, vsMap);

                            // at this point, we get the call inst return's scev
                            // now let's replace the old unknown with the scev
                            // vsMap.clear();
                            // vsMap[un] = call_scev;
                            // scev = SCEVUnknownRewriter::rewrite(scev, *se, vsMap);
                            data->addUnknownValue(un, scev2_data);

                            // llvm::outs() << "        scev after resolving call: " << *scev << "\n";
                            // if the are multiple return, we only handle the first one for now
                            break;
                        }
                    }
                }
            }

        } else if (auto *load = dyn_cast<LoadInst>(un)) {
            // if (true) continue;
            auto DT = &this->getAnalysis<DominatorTreeWrapperPass>(const_cast<llvm::Function &>(*F)).getDomTree();

            auto loadPtr = load->getPointerOperand();
            // llvm::outs() << DEBUGSTRING << "                resolved points to: " << *loadPtr << "\n";

            // TODO: find all uses of this pointer: stores and passed as func parameters
            // if loadPtr has only one use (which is un), then we set un to unknown
            if (loadPtr->hasName() && (loadPtr->getName().find("tid") != string::npos)) {
                // call void @__kmpc_for_static_init_4(
                // connect with thread id
                // llvm::outs() << "AHA catch a terminal tid: " << *loadPtr << " in func: " << F->getName() << "\n";
                data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, loadPtr));

            } else if (auto *bitcast = dyn_cast<BitCastInst>(loadPtr)) {
                data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, bitcast));
            } else if (loadPtr->hasOneUse())

                data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, loadPtr));

            else {
                llvm::Instruction *prevDominator = nullptr;
                int count = 0;
                for (auto U : loadPtr->users()) {
                    if (U == un) continue;  // dominate itself
                    if (prevDominator && isa<Instruction>(U)) {
                        if (DT->dominates((llvm::Instruction *)U, prevDominator))
                            continue;  // TODO: remove non-denominating stores
                    }

                    if (DEBUG_SCEV) llvm::outs() << DEBUGSTRING << "                resolved use: " << *U << "\n";
                    // resolve at most five uses
                    if (count++ > 5) break;
                    // tid
                    if (U->hasName() && (U->getName() == "tid")) {
                        // call void @__kmpc_for_static_init_4(
                        // connect with thread id
                        if (DEBUG_SCEV)
                            llvm::outs() << DEBUGSTRING << "                    find a terminal tid: " << *U << "\n";
                        data->addUnknownValue(un, new SCEVDataItem(se->getUnknown(U), F, U));
                        // un->setName("tid");
                        break;
                        // data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, U));
                    } else if (auto SI = llvm::dyn_cast<llvm::StoreInst>(U)) {  // store
                        //  make sure the load may read from the store
                        if (DT->dominates(SI, load)) {
                            if (DEBUG_SCEV)
                                llvm::outs()
                                    << DEBUGSTRING << "                    find a dominator store : " << *SI << "\n";
                            Value *value = SI->getValueOperand();
                            if (auto cons_int_value = llvm::dyn_cast<llvm::ConstantInt>(value)) {
                                if (DEBUG_SCEV)
                                    llvm::outs()
                                        << DEBUGSTRING << "                    stored constant: " << *value << "\n";
                                // since it is in the current func and it is a constant, we use rewrite
                                // DenseMap<const Value *, const llvm::SCEV *> vsMap;
                                // vsMap[un] = se->getConstant(cons_int_value);
                                // SE.getConstant(const_cast<llvm::ConstantInt *>(it->getSecond()));

                                // hopefully this will rewrite successfully
                                // scev = SCEVUnknownRewriter::rewrite(scev, *se, vsMap);
                                // data->updateSCEV(scev);
                                // llvm::outs() << "                    after rewrite constant: " << *scev << "\n";
                                // llvm::outs() << DEBUGSTRING << "                    after rewrite constant
                                // data->scev:
                                // "<< *data->getSCEV() << "\n";
                                auto store_const = se->getConstant(cons_int_value);
                                data->addUnknownValue(un, new SCEVDataItem(store_const, F, value));

                            } else {
                                // other types - can be pointers or unknown?
                                if (DEBUG_SCEV)
                                    llvm::outs()
                                        << DEBUGSTRING << "                    stored other TODO: " << *value << "\n";
                                auto scev2_data = getGlobalSCEVInternal(context, caller_inst, F, value);
                                data->addUnknownValue(un, scev2_data);
                            }
                            prevDominator = SI;
                        }
                        // TODO: handle multiple stores, and find the most immediately dominator store
                    } else if (llvm::isa<llvm::CallBase>(U)) {
                        llvm::Instruction *call_inst2 = (llvm::Instruction *)U;

                        CallSite CS2(call_inst2);
                        auto func2 = CS2.getTargetFunction();
                        // auto call_inst2 = llvm::cast<llvm::CallBase>(U);
                        // auto func2 = call_inst2->getCalledFunction();
                        if (func2 == nullptr) {
                            continue;
                        }
                        // if (func2->getReturnType()->isIntegerTy())  // consider only integer return type?
                        {
                            // now, go to get the scev2 for the return value v2
                            if (callRetScevMap.find(call_inst2) != callRetScevMap.end()) {
                                data->addUnknownValue(un, callRetScevMap[call_inst2]);
                                prevDominator = call_inst2;

                                continue;
                            } else if (isAnyExternalInferface(func2)) {
                                auto nullItem2 = new SCEVDataItem(se->getCouldNotCompute(), F, loadPtr);  // loadPtr
                                callRetScevMap[call_inst2] = nullItem2;
                                data->addUnknownValue(un, nullItem2);
                                prevDominator = call_inst2;

                                continue;
                            }

                            // the following may not be needed
                            // auto tid_unknown = se->getUnknown(un);

                            // DenseMap<const Value *, const llvm::SCEV *> vsMap;
                            // vsMap[un] = tid_unknown;
                            // scev = SCEVUnknownRewriter::rewrite(scev, *se, vsMap);

                            // llvm::outs() << "                    after renaming tid_unknown: " << *tid_unknown <<
                            // "\n";
                            // when we get here, claim that this unknown is linked to tid, and break the for loop
                            // llvm::outs() << DEBUGSTRING << "                    after renaming unknow to tid: "
                            // << *scev<< "\n"; data->addUnknownValue(un, new SCEVDataItem(tid_unknown));

                            else {
                                // TODO define other calls
                                if (DEBUG_SCEV)
                                    llvm::outs() << DEBUGSTRING
                                                 << "                    *** TODO: this unknown is from other call: "
                                                 << *call_inst2 << "\n";

                                // there are two cases:
                                // 1. loadPtr is defined by call_inst2
                                // 2. loadPtr is passed as para to call_inst2
                                if (!func2->getReturnType()->isVoidTy() && loadPtr == call_inst2) {
                                    if (DEBUG_SCEV)
                                        llvm::outs()
                                            << "TODO impossible here: via ret from call: " << *call_inst2 << "\n";
                                } else {
                                    // F_caller2 should not be null
                                    connectSCEVFunctionArgs(context, caller_inst, call_inst2->getFunction(),
                                                            call_inst2);  // missed call when traversing
                                }
                            }
                        }
                    } else if (auto *load2 = dyn_cast<LoadInst>(U)) {
                        // LULESH %12 = load double*, double** %m_dxx.i, align 8, !dbg !3042, !tbaa !3043
                        data->addUnknownValue(
                            un, getGlobalSCEVInternal(context, caller_inst, F, load2->getPointerOperand()));
                    } else if (auto BCI = llvm::dyn_cast<llvm::BitCastInst>(U)) {
                        // bitcat to another pointer
                        // %9 = bitcast i32* %2 to i8*, !dbg !35
                    }
                }

                // if prevDominator is still null, then loadPtr could be a func argu
                if (!prevDominator) {
                    if (isa<Argument>(loadPtr)) {
                        // TODO we should use pointer analysis here
                        data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, loadPtr));
                    }
                }
            }
            // llvm::outs() << "\n";
        } else if (auto *alloc = dyn_cast<AllocaInst>(un)) {
            // this is useful for connecting args in DRB054
            // connecting  arg:   %m = alloca i32, align 4   to  i32* %m
            // find all uses of this pointer
            if (alloc->getName().find("tid") == string::npos) {
                for (auto U : alloc->users()) {
                    if (auto SI = llvm::dyn_cast<llvm::StoreInst>(U))  // store inst
                    {
                        // llvm::outs() << "store use of alloc : " << *SI << "\n";

                        auto *value = SI->getValueOperand();
                        if (auto cons_int_value = llvm::dyn_cast<llvm::ConstantInt>(value)) {
                            if (DEBUG_SCEV) llvm::outs() << "const value of " << *un << ": " << *cons_int_value << "\n";
                            auto store_const = se->getConstant(cons_int_value);
                            data->addUnknownValue(un, new SCEVDataItem(store_const, F, value));
                            // data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F,
                            // cons_int_value));
                            break;
                        }
                    }
                }
            }
        } else if (auto *trunc = dyn_cast<TruncInst>(un)) {
            //%54 = trunc i64 %indvars.iv19 to i32, !dbg !1399

            data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, trunc->getOperand(0)));
        } else if (auto *addrspacecast = dyn_cast<AddrSpaceCastInst>(un)) {
            //%14 = addrspacecast [2 x i32] addrspace(3)* @smem_second to [2 x i32]*, !dbg !637
            data->addUnknownValue(un,
                                  getGlobalSCEVInternal(context, caller_inst, F, addrspacecast->getPointerOperand()));
        } else if (auto *phi = dyn_cast<PHINode>(un)) {
            // TODO: handle PHI -- this is difficult to deal with (crashes in ScalarEvolution::createNodeForPHI)
            bool analyzePHI = true;
            // 65646   %14 = phi i32 [ %add173, %omp.dispatch.inc ], [ %13, %omp.inner.for.cond.preheader.preheader
            // ] 66019   store i32 %add173, i32* %.omp.lb, align 4, !dbg !34047, !tbaa !3157, !alias.scope !2624
            // 65637   %13 = load i32, i32* %.omp.lb, align 4, !dbg !34048, !tbaa !3157, !alias.scope !2624

            // slightly slower performance if analyzePHI is true: it reduces several false positives in covid-sim
            // and graphblas
            if (analyzePHI) {
                // to improve performance, we only deal with phi with the first value (in case it is related to
                // %tid)
                auto phi0 = phi->getIncomingValue(0);

                Value *possible = nullptr;
                for (auto U : phi0->users()) {
                    if (auto SI = llvm::dyn_cast<llvm::StoreInst>(U)) {
                        auto storePtr = SI->getPointerOperand();
                        if (storePtr->hasName() && (storePtr->getName().find(".omp.lb") != string::npos ||
                                                    storePtr->getName().find("tid") != string::npos)) {
                            if (DEBUG_SCEV)
                                llvm::outs() << DEBUGSTRING << "INTERESTING PHI: " << *phi0
                                             << " storePtr: " << *storePtr << "\n";
                        }
                        possible = storePtr;
                    } else if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
                        auto loadPtr = LI->getPointerOperand();
                        if (loadPtr->hasName() && (loadPtr->getName().find(".omp.lb") != string::npos ||
                                                   loadPtr->getName().find("tid") != string::npos)) {
                            if (DEBUG_SCEV)
                                llvm::outs()
                                    << DEBUGSTRING << "INTERESTING PHI: " << *phi0 << " loadPtr: " << *loadPtr << "\n";
                        }
                        possible = loadPtr;
                    }
                }
                if (possible)
                    data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, possible));
                else {
                    // try it for the second
                    //%54 = phi i32 [ %41, %.lr.ph ], [ %88, %87 ]
                    unsigned size = phi->getNumIncomingValues();
                    if (size > 1) phi0 = phi->getIncomingValue(size - 1);  // last one
                    for (auto U : phi0->users()) {
                        if (auto SI = llvm::dyn_cast<llvm::StoreInst>(U)) {
                            auto storePtr = SI->getPointerOperand();
                            if (storePtr->hasName() && (storePtr->getName().find(".omp.lb") != string::npos ||
                                                        storePtr->getName().find("tid") != string::npos)) {
                                if (DEBUG_SCEV)
                                    llvm::outs() << DEBUGSTRING << "INTERESTING PHI: " << *phi0
                                                 << " storePtr: " << *storePtr << "\n";
                            }
                            possible = storePtr;
                        } else if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
                            auto loadPtr = LI->getPointerOperand();
                            if (loadPtr->hasName() && (loadPtr->getName().find(".omp.lb") != string::npos ||
                                                       loadPtr->getName().find("tid") != string::npos)) {
                                if (DEBUG_SCEV)
                                    llvm::outs() << DEBUGSTRING << "INTERESTING PHI: " << *phi0
                                                 << " loadPtr: " << *loadPtr << "\n";
                            }
                            possible = loadPtr;
                        }
                    }
                    if (possible)
                        data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, possible));
                    else {
                        // last try
                        if (auto LI = llvm::dyn_cast<llvm::LoadInst>(phi0)) {
                            auto loadPtr = LI->getPointerOperand();
                            if (loadPtr->hasName() && (loadPtr->getName().find(".omp.lb") != string::npos ||
                                                       loadPtr->getName().find("tid") != string::npos)) {
                                if (DEBUG_SCEV)
                                    llvm::outs() << DEBUGSTRING << "INTERESTING PHI: " << *phi0
                                                 << " loadPtr: " << *loadPtr << "\n";
                            }
                            possible = loadPtr;
                        }
                        if (possible)
                            data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, possible));
                    }
                }
            }
            // to improve performance, we only deal with phi with two values
            // if (size == 2) {
            //     data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F,
            //     phi->getIncomingValue(0))); data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst,
            //     F, phi->getIncomingValue(1)));
            // }
        } else if (auto *meta = dyn_cast<MetadataAsValue>(un)) {
        } else if (auto *global = dyn_cast<GlobalVariable>(un)) {
            // assume global is constant?
            if (global->hasInitializer()) {
                auto value = global->getInitializer();
                if (auto cons_int_value = llvm::dyn_cast<llvm::ConstantInt>(value)) {
                    auto name = "@" + global->getName().str();
                    auto val = std::to_string(cons_int_value->getSExtValue());

                    globalNameConstantValueMap[name] = val;
                    if (DEBUG_SCEV) llvm::outs() << "const value of global name: " << name << " val: " << val << "\n";
                    // auto store_const = se->getConstant(cons_int_value);
                    // data->addUnknownValue(un, new SCEVDataItem(store_const, F, value));
                }
            }
        } else if (auto *const_expr = dyn_cast<ConstantExpr>(un)) {
            // constant expression such as fptosi,
        } else if (auto *const_data = dyn_cast<ConstantData>(un)) {
            // constant data such as undef, nullptr
        } else if (auto *oper = dyn_cast<Operator>(un)) {
            // Unary and BinaryOperations such as: srem sdiv

            // TODO: need to handle divide
            //%12 = sdiv i32 %tid.x, 32, !dbg !636
            if (auto *sdiv = dyn_cast<SDivOperator>(oper)) {
                data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, sdiv->getOperand(0)));
            } else if (auto *ashr = dyn_cast<AShrOperator>(oper)) {
                data->addUnknownValue(un, getGlobalSCEVInternal(context, caller_inst, F, ashr->getOperand(0)));
            }
            if (DEBUG_SCEV) llvm::outs() << "TODO: other operator inst: " << *un << "\n";

        } else if (auto *select = dyn_cast<SelectInst>(un)) {
            // select: add a join of multiple values?
        } else {
            // other types of instructions
            if (DEBUG_SCEV) llvm::outs() << "TODO: other types: " << *un << "\n";
            //%class.Domain* undef
            // i32 undef
        }
    }

    if (DEBUG_SCEV) {
        DEBUGSTRING.pop_back();
        DEBUGSTRING.pop_back();
        DEBUGSTRING.pop_back();
        DEBUGSTRING.pop_back();
    }

    // enable cache will produce segfault on lulesh, not sure why
    scevCache[v] = data;
    return data;
}
char SVFPass::ID = 0;
