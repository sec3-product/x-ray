#include "Logger/Logger.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <indicators/color.hpp>
#include <indicators/progress_spinner.hpp>
#include <indicators/setting.hpp>
#include <spdlog/async.h>
#include <spdlog/fmt/ostr.h> // must be included
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xray {
namespace logger {

static bool progressEnabled = false;

class Spinner {
public:
  Spinner(std::string beginMsg, std::string endMsg, int tickIntervalms = 200);
  Spinner(std::string msg, int tickIntervalms = 200)
      : Spinner(msg, msg, tickIntervalms) {}
  ~Spinner();

  void begin();
  void end();

private:
  indicators::ProgressSpinner spinner;
  const std::chrono::milliseconds interval;
  std::thread ticker;
  std::string endMsg;
};

Spinner::Spinner(std::string beginMsg, std::string endMsg, int tickIntervalms)
    : spinner(indicators::option::PrefixText{" - "},
              indicators::option::PostfixText{beginMsg},
              indicators::option::ForegroundColor{indicators::Color::white},
              indicators::option::FontStyles{std::vector<indicators::FontStyle>{
                  indicators::FontStyle::bold}},
              indicators::option::ShowPercentage{false},
#ifdef __MINGW32__
              indicators::option::SpinnerStates{
                  std::vector<std::string>{"/", "\\"}},
              indicators::option::ShowElapsedTime{true}
#else
              indicators::option::SpinnerStates{
                  std::vector<std::string>{"▖", "▘", "▝", "▗"}},
              indicators::option::ShowElapsedTime{true}
#endif
              ),
      interval(tickIntervalms), ticker([this]() {
        while (!spinner.is_completed()) {
          spinner.tick();
          std::this_thread::sleep_for(interval);
        }
      }),
      endMsg(endMsg) {
}

Spinner::~Spinner() {
  if (ticker.joinable()) {
    end();
    ticker.join();
  }
}

void Spinner::begin() {
  // Spinner auto starts in the constructor.
}

void Spinner::end() {
  if (spinner.is_completed()) {
    return;
  }
  spinner.set_option(
      indicators::option::ForegroundColor{indicators::Color::green});
#ifdef __MINGW32__
  spinner.set_option(indicators::option::PrefixText{" - *"});
#else
  spinner.set_option(indicators::option::PrefixText{" - ✔"});
#endif
  spinner.set_option(indicators::option::PostfixText{endMsg});
  spinner.set_option(indicators::option::ShowSpinner{false});
  spinner.mark_as_completed();
}

static std::unique_ptr<Spinner> currentSpinner = nullptr;

void newPhaseSpinner(std::string beginMsg, std::string endMsg,
                     int tickIntervalms) {
  if (!progressEnabled) {
    return;
  }

  if (currentSpinner != nullptr) {
    currentSpinner->end();
  }

  currentSpinner = std::make_unique<Spinner>(beginMsg, endMsg, tickIntervalms);
  currentSpinner->begin();
}

void endPhase() {
  if (!progressEnabled) {
    return;
  }

  if (currentSpinner != nullptr) {
    currentSpinner->end();
    currentSpinner.reset();
  }
}

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

  // Used by the progress spinner.
  progressEnabled = config.enableProgress;

  LOG_DEBUG("Set logging configs: level={}, logFolder={}, logFile={}, "
            "enableTerminal={}, enableFile={}",
            config.level, config.logFolder, config.logFile,
            config.enableTerminal, config.enableFile);
}

} // namespace logger
} // namespace xray
