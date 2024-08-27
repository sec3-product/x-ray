#include "o2/Util/Log.h"

#include <filesystem>

#include "spdlog/async.h"
#include "spdlog/fmt/ostr.h" // must be included
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace o2 {
namespace logger {

void init(LoggingConfig config) {
  std::vector<spdlog::sink_ptr> sinks;

  // Console sink prints messages to the console.
  if (config.enableTerminal) {
    auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    console_sink->set_level(config.terminalLevel);
    // https://github.com/gabime/spdlog/wiki/3.-Custom-formatting#pattern-flags
    console_sink->set_pattern("%H:%M:%S [%^%=7l%$] [%n] [%s:%#] %v");
    sinks.push_back(console_sink);
  }

  // File sink logs to a file.
  if (config.enableFile) {
    auto log_path = std::filesystem::path(config.logFolder) / config.logFile;
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        log_path.string(), false);
    file_sink->set_level(config.fileLevel);
    file_sink->set_pattern("%H:%M:%S [%l] [%n] [%s:%#] %v");
    sinks.push_back(file_sink);
  }

  auto logger =
      std::make_shared<spdlog::logger>("xray", sinks.begin(), sinks.end());
  logger->set_level(config.level);
  spdlog::set_default_logger(logger);

  LOG_DEBUG("Set logging configs: level={}, logFolder={}, logFile={}, "
            "enableTerminal={}, enableFile={}",
            config.level, config.logFolder, config.logFile,
            config.enableTerminal, config.enableFile);
}

} // namespace logger
} // namespace o2
