#ifndef RACEDETECTOR_AV_TPL_H
#define RACEDETECTOR_AV_TPL_H

#include <string>

namespace aser {

static std::string AVTemplate =
    "     Thread1                      Thread2"
    " ------------------                              \n"
    " |                |                              \n"
    " ------------------                              \n"
    "                         \\                       \n"
    "                               ------------------\n"
    "                               |                |\n"
    "                               ------------------\n"
    "                         /                       \n"
    " ------------------                              \n"
    " |                |                              \n"
    " ------------------                              \n";

}  // namespace aser

#endif