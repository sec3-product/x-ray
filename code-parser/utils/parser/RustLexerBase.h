#pragma once

#include "antlr4-runtime.h"
using namespace antlr4;
class RustLexer : public antlr4::Lexer
{
    antlr4::Token *lt1;
    antlr4::Token *lt2;
    bool floatDotPossible()
    {
        size_t next = _input->LA(1);
        // printf("RustLexer::floatDotPossible: %d\n", next);

        // only block . _ identifier after float
        if (next == '.' || next == '_')
            return false;
        if (next == 'f')
        {
            // 1.f32
            if (_input->LA(2) == '3' && _input->LA(3) == '2')
                return true;
            // 1.f64
            if (_input->LA(2) == '6' && _input->LA(3) == '4')
                return true;
            return false;
        }
        if (next >= 'a' && next <= 'z')
            return false;
        if (next >= 'A' && next <= 'Z')
            return false;
        return true;
    }
    bool floatLiteralPossible()
    {
        if (lt1 == nullptr)
        {
            return true;
        }
        if (lt2 == nullptr)
        {
            return true;
        }
        // always return true to avoid crash
        return true;
        // crash here
        if (lt1 && lt1->getType() != RustLexer::DOT)
        {
            return true;
        }
        // printf("RustLexer::floatLiteralPossible: %s\n", this->lt2->getText());

        switch (this->lt2->getType())
        {
        case RustLexer::CHAR_LITERAL:
        case RustLexer::STRING_LITERAL:
        case RustLexer::RAW_STRING_LITERAL:
        case RustLexer::BYTE_LITERAL:
        case RustLexer::BYTE_STRING_LITERAL:
        case RustLexer::RAW_BYTE_STRING_LITERAL:
        case RustLexer::INTEGER_LITERAL:
        case RustLexer::DEC_LITERAL:
        case RustLexer::HEX_LITERAL:
        case RustLexer::OCT_LITERAL:
        case RustLexer::BIN_LITERAL:

        case RustLexer::KW_SUPER:
        case RustLexer::KW_SELFVALUE:
        case RustLexer::KW_SELFTYPE:
        case RustLexer::KW_CRATE:
        case RustLexer::KW_DOLLARCRATE:

        case RustLexer::GT:
        case RustLexer::RCURLYBRACE:
        case RustLexer::RSQUAREBRACKET:
        case RustLexer::RPAREN:

        case RustLexer::KW_AWAIT:

        case RustLexer::NON_KEYWORD_IDENTIFIER:
        case RustLexer::RAW_IDENTIFIER:
        case RustLexer::KW_MACRORULES:
            return false;
        default:
            return true;
        }
    }
    bool SOF()
    {
        size_t _la = _input->LA(-1);
        // printf("RustLexer::SOF: %d\n", _la);

        // if (_la < 0) {
        //   printf("RustLexer::SOF <0: %d\n", _la);
        // } else {
        //   printf("RustLexer::SOF >=0: %d\n", _la);
        // }
        if (_la == 0 || _la == IntStream::EOF)
        {
            // printf("RustLexer::SOF ==EOF: %d\n", IntStream::EOF);
            return true;
        }
        else
        {
            // printf("RustLexer::SOF not==EOF: %d\n", IntStream::EOF);
            return false;
        }
        // return _la <= 0;
    }
    bool next(char expect) { return _input->LA(1) == expect; }
    std::unique_ptr<antlr4::Token> nextToken()
    {
        std::unique_ptr<Token> next = Lexer::nextToken();
        // printf("RustLexer::nextToken: %s\n", next->toString());

        if (next->getChannel() == Token::DEFAULT_CHANNEL)
        {
            // Keep track of the last token on the default channel.
            this->lt2 = this->lt1;
            this->lt1 = next.get();
        }

        return std::move(next);
    }