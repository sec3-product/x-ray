#pragma once

#include <spdlog/spdlog.h>

namespace o2 {
namespace logger {

struct LoggingConfig {
  // Enable Progress Spinners
  bool enableProgress;

  // Enable printing logs to terminal
  bool enableTerminal;

  // Enable printing logs to file
  bool enableFile;

  // Directory to store logfile
  std::string logFolder;

  // File to store logs
  std::string logFile;

  // Filter all log messages below this level
  // Note: This setting has a higher precedent than file/terminal levels
  spdlog::level::level_enum level;

  // minimum log level for log file
  spdlog::level::level_enum fileLevel;

  // minimum log level for terminal
  spdlog::level::level_enum terminalLevel;

  // Specify some defaults
  LoggingConfig()
      : enableProgress(true), enableTerminal(true), enableFile(true),
        logFolder("."), logFile("log.current"), level(spdlog::level::debug),
        fileLevel(level), terminalLevel(spdlog::level::info) {}
};

// Must call this before using logging
void init(LoggingConfig cfg);

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__);
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__);
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__);
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__);
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__);

} // namespace logger
} // namespace o2
