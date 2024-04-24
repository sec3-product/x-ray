//
// Created by peiming on 2/6/20.
//
#include "../intercept.h"

#include <set>
#include <iostream>
#include <malloc.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cxxabi.h>
#include "Interceptor.h"

using namespace std;

#define CODERRECT_AR "CODERRECT_AR"
#define CODERRECT_LD "CODERRECT_LD"
#define CODERRECT_CLANG "CODERRECT_CLANG"
#define CODERRECT_CLANGXX "CODERRECT_CLANGXX"
#define CODERRECT_FORTRAN "CODERRECT_FORTRAN"
#define CODERRECT_CP "CODERRECT_CP"

#define ENV_C_COMPILERS "CR_INTERCEPT_C_COMPILERS"
#define ENV_CXX_COMPILERS "CR_INTERCEPT_CXX_COMPILERS"
#define ENV_FORTRAN_COMPILERS "CR_INTERCEPT_FORTRAN_COMPILERS"
#define ENV_AR_LINKERS "CR_INTERCEPT_AR_LINKERS"
#define ENV_LD_LINKERS "CR_INTERCEPT_LD_LINKERS"
#define ENV_FILE_OPERATORS "CR_INTERCEPT_FILE_OPERATORS"

static set<string> C_Compilers{
    "cc",
    "gcc",
    "clang",
    "cr-clang",
    "icc" // intel c compiler
};

static set<string> CXX_Compilers{
    "c++",
    "g++",
    "clang++",
    "cr-clang++",
    "icpc" // intel c++ compiler
};

static set<string> FORTRAN_Compilers{
    "mpif90",
    "mpifort",
    "flang",
    "gfortran",
    "g77",
    "f95",
    "ifort" // intel Fortran compiler
};

static set<string> ar_linkers{
    "ar",
    "llvm-ar",
    "llvm-ar-11",
};

static set<string> ld_linkers{
    "ld",
    "gold",
    "lld",
    "ld.lld",
    "ld.gold",
};

static set<string> FILE_OPERATOR{
    "cp",
    "mv",
};

set<string> read_user_config(const char *key)
{
    auto data = std::getenv(key);
    if (!data)
    {
        return set<string>();
    }

    set<string> values;
    string sdata(data);
    string current = "";
    constexpr char delim = ' ';
    for (auto c : sdata)
    {
        if (c == delim)
        {
            if (!current.empty())
            {
                values.insert(current);
                current.clear();
            }
        }
        else
        {
            current += c;
        }
    }
    if (!current.empty())
    {
        values.insert(current);
    }

    return values;
}

void read_user_configs()
{
    auto user_c_compilers = read_user_config(ENV_C_COMPILERS);
    C_Compilers.insert(user_c_compilers.begin(), user_c_compilers.end());

    auto user_cxx_compilers = read_user_config(ENV_CXX_COMPILERS);
    CXX_Compilers.insert(user_cxx_compilers.begin(), user_cxx_compilers.end());

    auto user_fortran_compilers = read_user_config(ENV_FORTRAN_COMPILERS);
    FORTRAN_Compilers.insert(user_fortran_compilers.begin(), user_fortran_compilers.end());

    auto user_ar_linkers = read_user_config(ENV_AR_LINKERS);
    ar_linkers.insert(user_ar_linkers.begin(), user_ar_linkers.end());

    auto user_ld_linkers = read_user_config(ENV_LD_LINKERS);
    ld_linkers.insert(user_ld_linkers.begin(), user_ld_linkers.end());

    auto user_file_operators = read_user_config(ENV_FILE_OPERATORS);
    FILE_OPERATOR.insert(user_file_operators.begin(), user_file_operators.end());
}

result intercept_call(const char *file, char const *const argv[])
{
    read_user_configs();
    Interceptor interceptor(file, argv);
    auto r = interceptor.getInterceptedArgs();
    return {strdup(interceptor.file.c_str()), r};
}

static string getCmdName(const string &fullPath)
{
    auto pos = fullPath.find_last_of('/');
    if (pos != string::npos)
        return fullPath.substr(pos + 1, fullPath.size());

    return fullPath;
}

static char *captureEnv(const char *envName)
{
    char *envValue = getenv(envName);
    if (nullptr == envValue)
    {
        std::cerr << "fail to get environment variable " << envName << endl;
        abort();
    }
    return envValue;
}

Interceptor::Interceptor(const char *file, char const *const *argv) : file(file)
{
    // 1st, copy all the execve arguments
    for (char const *const *it = argv; (it) && (*it); ++it)
    {
        this->args.emplace_back(*it);
    }

    // 2nd, get the environment variable
    this->coderrectAR = captureEnv(CODERRECT_AR);
    this->coderrectLD = captureEnv(CODERRECT_LD);
    this->coderrectClang = captureEnv(CODERRECT_CLANG);
    this->coderrectClangXX = captureEnv(CODERRECT_CLANGXX);
    this->coderrectFortran = captureEnv(CODERRECT_FORTRAN);

    this->coderrectCp = captureEnv(CODERRECT_CP);
}

char **Interceptor::getInterceptedArgs()
{
    string cmd = getCmdName(file);

    if (C_Compilers.find(cmd) != C_Compilers.end())
    {
        args[0] = strdup(coderrectClang.c_str());
        file = args[0];
    }
    else if (CXX_Compilers.find((cmd)) != CXX_Compilers.end())
    {
        args[0] = strdup(coderrectClangXX.c_str());
        ;
        file = args[0];
    }
    else if (ar_linkers.find(cmd) != ar_linkers.end())
    {
        args[0] = strdup(coderrectAR.c_str());
        file = args[0];
    }
    else if (ld_linkers.find(cmd) != ld_linkers.end())
    {
        args[0] = strdup(coderrectLD.c_str());
        file = args[0];
    }
    else if (FORTRAN_Compilers.find(cmd) != FORTRAN_Compilers.end())
    {
        args[0] = strdup(coderrectFortran.c_str());
        file = args[0];
    }
    else if (FILE_OPERATOR.find(cmd) != FILE_OPERATOR.end())
    {
        args.insert(args.begin(), cmd);
        args[0] = strdup(coderrectCp.c_str());
        file = args[0];
    }

    shouldInterceptFollow = 0;
    if (file == coderrectClang || file == coderrectClangXX || file == coderrectFortran)
    {
        shouldInterceptFollow = 1;
    }

    /*
  cerr << "executing ..: ";
  for (auto str : args) {
     cerr << str << " ";
  }
  cerr << "\n";
*/

    // dump the modified args
    char **result = (char **)malloc((args.size() + 1) * sizeof(char *));
    char **out_it = result;
    for (auto &str : args)
    {
        *out_it = strdup(str.c_str());
        out_it++;
    }
    *out_it = 0;

    return result;
}
