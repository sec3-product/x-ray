#ifndef O2_LOGGING_H
#define O2_LOGGING_H

#include "spdlog/spdlog.h"

namespace o2 {
namespace logger {

#ifndef USE_DEFAULT_SPDLOG

// Globally shared logger
// Do not use directly
// Use the Macros below instead
extern std::shared_ptr<spdlog::logger> logger;

template <typename... Args>
inline void trimmed_log(spdlog::source_loc source, spdlog::level::level_enum lvl, spdlog::string_view_t fmt,
                        const Args&... args) {
    // Only print the filename, not the full path
    source.filename += std::string(source.filename).rfind("/") + 1;
    logger->log(source, lvl, fmt, args...);
}

#define LOG_INTERNAL(level, ...) \
    o2::logger::trimmed_log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__)

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
        : enableProgress(true),
          enableTerminal(true),
          enableFile(true),
          logFolder("."),
          logFile("log.current"),
          level(spdlog::level::debug),
          fileLevel(level),
          terminalLevel(spdlog::level::info) {}
};

// Must call this before using logging
void init(LoggingConfig cfg);

// Use macros to log at different levels. Same API as SPDLOG macros
#define LOG_TRACE(...) LOG_INTERNAL(spdlog::level::trace, __VA_ARGS__)
#define LOG_DEBUG(...) LOG_INTERNAL(spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...) LOG_INTERNAL(spdlog::level::info, __VA_ARGS__)
#define LOG_WARN(...) LOG_INTERNAL(spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...) LOG_INTERNAL(spdlog::level::err, __VA_ARGS__)

// Example usage
// namespace {
// void example() {
//     LOG_DEBUG("Parsing the source code. file={}", "somefile.c");
//     LOG_ERROR("Unexpected file type. file={}", "program.exe");
// }
//
// }  // namespace

#else

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__);
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__);
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__);
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__);
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__);

#endif

}  // namespace logger
}  // namespace o2

#endif
