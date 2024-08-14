#pragma once

#include <string>

namespace aser {

void highlight(std::string msg);

void info(std::string msg);
void info(std::string msg, bool newline);

void error(std::string msg);

std::string getCurrentTimeStr();

} // namespace aser
