%{
#include <stdio.h>
#include <string.h>
  #include <iostream>
  #include <regex>
 
using namespace std;

  // Declare stuff from Flex that Bison needs to know about:

extern string SCEV_TEMP_RESULT;
extern bool DEBUG_SCEV_SIM;
extern int yy_scan_string(const char *yy_str);
extern int yylex();
extern int yyparse();
extern FILE *yyin;
  void yyerror(const char *s);
%}


// Bison fundamentally works by asking flex to get the next token, which it
// returns as an object of type "yystype".  Initially (by default), yystype
// is merely a typedef of "int", but for non-trivial projects, tokens could
// be of any arbitrary data type.  So, to deal with that, the idea is to
// override yystype's default typedef to be a C union instead.  Unions can
// hold all of the types of tokens that Flex could return, and this this means
// we can return ints or floats or strings cleanly.  Bison implements this
// mechanism with the %union directive:
%union {
  int ival;
  float fval;
  char *sval;
}

%start fullexpr

%type <sval> scev
%type <sval> SCEVAddRec
%type <sval> SCEVAddRec2
%type <sval> LoopHeader
%type <sval> NoWrap

// Define the "terminal symbol" token types I'm going to use (in CAPS
// by convention), and associate each with a field of the %union:
%token <sval> ConstantInt
%token <sval> FLOAT
%token <sval> STRING
%token <sval> END
%token <sval> ID
%token <sval> AddRec
%token <sval> Plus 
%token <sval> Mul
%token <sval> Div
%token <sval> UNDEF
%token <sval> MinMax
%token <sval> NoWrapNW
%token <sval> NoWrapNSW
%token <sval> NoWrapNUW
//zero/sign extentions 
%token <sval> ZExtX
%token <sval> SExtX
//trunc extentions 
%token <sval> TruncX
%token <sval> ToX
//brackets () 
%token <sval> BLeft
%token <sval> BRight
//curly brackets {}
%token <sval> CBLeft
%token <sval> CBRight
//chevron brackets <>
%token <sval> CHLeft
%token <sval> CHRight

%%

fullexpr: scev {
SCEV_TEMP_RESULT = $1;
     if(DEBUG_SCEV_SIM) cout << "matched a full scev: "<<$1<< endl;
};

scev: 
   BLeft scev BRight {
char s = $2[0];

if(s=='('|| s=='%' || s == '@' || s == '0')	$$=$2;
else {	
	string tmp=$2;
	tmp= "("+tmp+ ")";
	char* result = strdup(tmp.c_str());
	$$ = strdup(result);}
     if(DEBUG_SCEV_SIM) cout << "matched a scev: "<<$$ << endl;
  }         
  | scev Mul scev {
string tmp1=$1; 
string tmp2=$3; 
if(tmp1=="0" || tmp2=="0")
$$ = (char*)"0";
else if (tmp1=="1") $$ = $3;
else if (tmp2=="1") $$ = $1;
else {
string tmp3= tmp1+ " * " +tmp2;
char* result = strdup(tmp3.c_str());
$$ = strdup(result);
}
      if(DEBUG_SCEV_SIM)cout << "matched a multiplication of scev: "<<$$ << endl;
	}
  | scev Div scev {
  
        string tmp1=$1;
        string tmp2=$2;
        string tmp3=$3;
        string tmp= tmp1+" "+tmp2+" "+tmp3;
char* result = strdup(tmp.c_str());
$$ = strdup(result);
        
        if(DEBUG_SCEV_SIM)cout << "matched a div of scev: "<<$$ << endl;

	}
  | scev Plus scev {

	string tmp1=$1;
        if(tmp1=="0") $$=$3;
        else {
		string tmp2=$3;
 		if(tmp2=="0") $$=$1;
		else {
	string tmp3= tmp1+ " + " +tmp2;
char* result = strdup(tmp3.c_str());
$$ = strdup(result);}
      	}
	if(DEBUG_SCEV_SIM)cout << "matched a sum of scev: "<<$$ << endl;
	}
  | scev MinMax scev {
  
        string tmp1=$1;
        string tmp2=$2;
        string tmp3=$3;
        string tmp= tmp1+" "+tmp2+" "+tmp3;
char* result = strdup(tmp.c_str());
$$ = strdup(result);
        
        if(DEBUG_SCEV_SIM)cout << "matched a min/max of scev: "<<$$ << endl;

	}
  | scev NoWrap {
      	$$=$1;
    }
  | SCEVAddRec {
$$= $1;  
if(DEBUG_SCEV_SIM) cout << "matched an SCEVAddRec: "<<$$<< endl;
  }
  | ZExtX scev ToX {
	$$=$2;
      if(DEBUG_SCEV_SIM)cout << "matched a zero extension of scev: "<<$$ <<endl;
	}
  | SExtX scev ToX {
	$$=$2;
      if(DEBUG_SCEV_SIM)cout << "matched a sign extension of scev: "<<$$ <<endl;
	}
  | TruncX scev ToX {
	$$=$2;
      if(DEBUG_SCEV_SIM)cout << "matched a trunc of scev: "<<$$ <<endl;
	}
  | ID {
     $$ = $1;
	if(DEBUG_SCEV_SIM)cout << "found an scev identifier: " << $1 << endl;           
    }
  | ConstantInt           {
     $$ = $1;    
	if(DEBUG_SCEV_SIM)cout << "found a constant int: " << $1 << endl;      
    }
  | UNDEF           {
     $$ = $1;    
	if(DEBUG_SCEV_SIM)cout << "found an undef value: " << $1 << endl;      
    }
;

SCEVAddRec: CBLeft scev AddRec scev CBRight SCEVAddRec2 {

$$=$2;
};

SCEVAddRec2: LoopHeader {
$$=$1;	   
           }
	   | NoWrap SCEVAddRec2 {
            $$=$2;
           };

LoopHeader: CHLeft ID CHRight	{
$$=$2;
 if(DEBUG_SCEV_SIM)cout << "recognizing a loop header: "<<$$<< endl;   
	}
;

NoWrap: NoWrapNW {
	$$ = $1;
	if(DEBUG_SCEV_SIM)cout << "found an <nw>: " << $1 << endl;      
	}
	| NoWrapNSW {
	$$ = $1;
	if(DEBUG_SCEV_SIM)cout << "found an <nsw>: " << $1 << endl;      
	}	
	       | NoWrapNUW {
	$$ = $1;
	if(DEBUG_SCEV_SIM)cout << "found an <nuw>: " << $1 << endl;      
	}
;

%%

int main(int argc, char** argv) {
  // Open a file handle to a particular file:

const char* filename;
  if(argc>1) {
      filename = argv[1];
    } else {

  filename = "test.txt";
}
  FILE *myfile = fopen(filename, "r");
  // Make sure it is valid:
  if (!myfile) {
    cout << "I can't open "<<filename<<"!" << endl;
    return -1;
  }
  cout << "reading input from "<<filename<<endl << endl;

  // Set Flex to read from it instead of defaulting to STDIN:
  yyin = myfile;
  
  // Parse through the input:
  yyparse();
  return 0; 
}

string INPUT_STRING;
int test_scev_parser(const char *input) {
    std::string inputStr(input);
    INPUT_STRING = inputStr;

    yy_scan_string(input);

    // Parse through the input:
    yyparse();
    return 0;
}

void yyerror(const char *s) {
if(DEBUG_SCEV_SIM)
  cout << "\nParse error!  Message: " << INPUT_STRING << endl;
  // might as well halt now:
  // exit(-1);
}
