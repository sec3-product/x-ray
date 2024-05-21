#ifndef RACEDETECTOR_DL_TPL_H
#define RACEDETECTOR_DL_TPL_H

#include <string>

namespace aser {

static std::string DLTemplate =
    "{% for trace in acquires  %}"
    "========================================\n"
    "|                                      |\n"
    "|               Thread {{loop.index}}:              |\n"
    "|                                      |\n"
    "========================================\n"
    "Acquire lock1\n"
    "{{trace.0.snippet}}"
    "----------------------------------------\n"
    "                  ||           \n"
    "                  ||           \n"
    "                  \\/           \n"
    "----------------------------------------\n"
    "Acquire lock2\n"
    "{{trace.1.snippet}}"
    "{% endfor %}";

static std::string DLTemplate2 =
    "{% for trace in acquires  %}"
    "Thread {{loop.index}}: \n"
    "--------------------------------                              ------------------------------------\n"
    "|    {{trace.0.snippet}}       |           ----->             |       {{trace.1.snippet}}        |\n"
    "--------------------------------                              ------------------------------------\n"
    "{% endfor %}";

}  // namespace aser

#endif