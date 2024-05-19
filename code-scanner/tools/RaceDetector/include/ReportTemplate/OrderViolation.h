#ifndef RACEDETECTOR_OV_TPL_H
#define RACEDETECTOR_OV_TPL_H

#include <string>

namespace aser {

static std::string OVTemplate =
    "     Thread1                      Thread2"
    " ------------------                              \n"
    " |                |                              \n"
    " ------------------                              \n"
    "                         \\                       \n"
    "                               ------------------\n"
    "                               |                |\n"
    "                               ------------------\n";

}  // namespace aser

#endif