#pragma once

#include "antlr4-runtime.h"

class RustParser : public antlr4::Parser
{
    bool next(char expect)
    {
        // size_t _la = _input->LA(1);
        // printf("RustParser::next: %d expect: %s\n", _la, expect);
        // only handle '<' and '>'
        size_t _la = _input->LA(1);
        if (expect == '<' && _la == RustParser::LT)
            return true;
        if (expect == '>' && _la == RustParser::GT)
            return true;
        return _input->LA(1) == expect;
    }
}