#include "o2/Util/Log.h"

#include "spdlog/async.h"
#include "spdlog/fmt/ostr.h"  // must be included
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace o2 {
namespace logger {

std::shared_ptr<spdlog::logger> logger;

static bool progressEnabled;

void init(LoggingConfig config) {
    // Console sink prints messages to the console
    auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    console_sink->set_level(config.terminalLevel);
    console_sink->set_pattern("%H:%M:%S [%^%=7l%$] [%n] [%@] %v");

    // File sink will log messages to a file
    auto log_file = config.logFolder + "/" + config.logFile;  // TODO: make sure the path is formed correctly
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, false);
    file_sink->set_level(config.fileLevel);
    file_sink->set_pattern("%H:%M:%S [%l] [%n] [%@] %v");

    std::vector<spdlog::sink_ptr> sinks;
    if (config.enableTerminal) {
        sinks.push_back(console_sink);
    }
    if (config.enableFile) {
        sinks.push_back(file_sink);
    }

    logger = std::make_shared<spdlog::logger>("racedetect", sinks.begin(), sinks.end());
    logger->set_level(config.level);

    progressEnabled = config.enableProgress;

    LOG_DEBUG("Logging configs. level={}, logFolder={}, logFile={}, enableTerminal={}, enableFile={}", config.level,
              config.logFolder, config.logFile, config.enableTerminal, config.enableFile);
}

}  // namespace logger
}  // namespace o2