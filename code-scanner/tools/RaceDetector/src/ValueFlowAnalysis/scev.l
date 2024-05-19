%{
#include <stdio.h>
  #include "y.tab.h"  // to get the token types from


typedef struct tnode{
 int val; //value of the expression tree
 char *op; //indicates the opertor
 struct tnode *left,*right; //left and right branches
 }tnode;
	
/*Make a leaf tnode and set the value of val field*/
struct tnode* makeLeafNodeUnknown(char* id);
struct tnode* makeLeafNodeConstant(int n);
	
/*Make a tnode with opertor, left and right branches set*/
struct tnode* makeOperatorNode(char c,struct tnode *l,struct tnode *r);
	
/*To evaluate an expression tree*/
int evaluate(struct tnode *t);
/*[\0\n]	  {return END;}*/


%}
%%
[ \t\n\0]           ;
\+        { yylval.sval = strdup(yytext); return Plus;}         
\*        { yylval.sval = strdup(yytext); return Mul;}         
,\+,        { yylval.sval = strdup(yytext); return AddRec;}         
\(        { yylval.sval = strdup(yytext); return BLeft;}         
\)        { yylval.sval = strdup(yytext); return BRight;}         
\{        { yylval.sval = strdup(yytext); return CBLeft;}         
\}        { yylval.sval = strdup(yytext); return CBRight;}         
\<        { yylval.sval = strdup(yytext); return CHLeft;}         
\>        { yylval.sval = strdup(yytext); return CHRight;}         
undef        { yylval.sval = strdup(yytext); return UNDEF;}         
(s|u)(min|max)        { yylval.sval = strdup(yytext); return MinMax;}         
\/(s|u)        { yylval.sval = strdup(yytext); return Div;}         
zext\ i[0-9]+        { yylval.sval = strdup(yytext); return ZExtX;}         
sext\ i[0-9]+        { yylval.sval = strdup(yytext); return SExtX;}         
trunc\ i[0-9]+        { yylval.sval = strdup(yytext); return TruncX;}         
to\ i[0-9]+        { yylval.sval = strdup(yytext); return ToX;}         
\<nw\>        { yylval.sval = strdup(yytext); return NoWrapNW;}         
\<nsw\>        { yylval.sval = strdup(yytext); return NoWrapNSW;}         
\<nuw\>        { yylval.sval = strdup(yytext); return NoWrapNUW;}         
[%|@]([\.]?[a-zA-Z_0-9]*)+        { yylval.sval = strdup(yytext); return ID;}         
[0-9]+\.[0-9]+    { yylval.sval = strdup(yytext); return FLOAT; }
0|(-?[1-9][0-9]*) {yylval.sval = strdup(yytext); 
//yylval.no = makeLeafNode(number);

return ConstantInt;}
[a-zA-Z0-9]+      {
  // We have to strdup yytext because Flex will change it for the next token.
  // Note that this memory must be freed somewhere, so that's why we call
  // free() above in the Bison section.  (Aside: we use free() instead of
  // delete because strdup is a C function that uses malloc, not a C++
  // function that uses new.)
  yylval.sval = strdup(yytext);
  return STRING;
}
%%
int yywrap(void){
	return 1;
}
