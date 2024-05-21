/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    ConstantInt = 258,
    FLOAT = 259,
    STRING = 260,
    END = 261,
    ID = 262,
    AddRec = 263,
    Plus = 264,
    Mul = 265,
    Div = 266,
    UNDEF = 267,
    MinMax = 268,
    NoWrapNW = 269,
    NoWrapNSW = 270,
    NoWrapNUW = 271,
    ZExtX = 272,
    SExtX = 273,
    TruncX = 274,
    ToX = 275,
    BLeft = 276,
    BRight = 277,
    CBLeft = 278,
    CBRight = 279,
    CHLeft = 280,
    CHRight = 281
  };
#endif
/* Tokens.  */
#define ConstantInt 258
#define FLOAT 259
#define STRING 260
#define END 261
#define ID 262
#define AddRec 263
#define Plus 264
#define Mul 265
#define Div 266
#define UNDEF 267
#define MinMax 268
#define NoWrapNW 269
#define NoWrapNSW 270
#define NoWrapNUW 271
#define ZExtX 272
#define SExtX 273
#define TruncX 274
#define ToX 275
#define BLeft 276
#define BRight 277
#define CBLeft 278
#define CBRight 279
#define CHLeft 280
#define CHRight 281

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 29 "scev.y" /* yacc.c:1909  */

  int ival;
  float fval;
  char *sval;

#line 112 "y.tab.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */
