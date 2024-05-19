//
// Created by peiming on 2/6/20.
//

#ifndef BEAR_INTERCEPTOR_H
#define BEAR_INTERCEPTOR_H

#include <vector>
#include <string>

class Interceptor
{
private:
    std::string coderrectAR;
    std::string coderrectLD;
    std::string coderrectClang;
    std::string coderrectClangXX;
    std::string coderrectFortran;
    std::string coderrectCp;

public:
    std::vector<std::string> args;
    std::string file;

    explicit Interceptor(const char *c_file, char const *const *argv);
    char **getInterceptedArgs();
};

#endif //BEAR_INTERCEPTOR_H
