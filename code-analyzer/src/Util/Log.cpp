#include "Util/Log.h"

#include "indicators/color.hpp"
#include "indicators/progress_spinner.hpp"
#include "indicators/setting.hpp"
#include "spdlog/async.h"
#include "spdlog/fmt/ostr.h" // must be included
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace xray {
namespace logger {

std::shared_ptr<spdlog::logger> logger;

class Spinner;
static Spinner *currentSpinner;
static bool progressEnabled;

void init(LoggingConfig config) {
  // Console sink prints messages to the console
  auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
  console_sink->set_level(config.terminalLevel);
  console_sink->set_pattern("%H:%M:%S [%^%=7l%$] [%n] [%@] %v");

  // File sink will log messages to a file
  auto log_file =
      config.logFolder + "/" +
      config.logFile; // TODO: make sure the path is formed correctly
  auto file_sink =
      std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, false);
  file_sink->set_level(config.fileLevel);
  file_sink->set_pattern("%H:%M:%S [%l] [%n] [%@] %v");

  std::vector<spdlog::sink_ptr> sinks;
  if (config.enableTerminal) {
    sinks.push_back(console_sink);
  }
  if (config.enableFile) {
    sinks.push_back(file_sink);
  }

  logger = std::make_shared<spdlog::logger>("racedetect", sinks.begin(),
                                            sinks.end());
  logger->set_level(config.level);

  // Used by progress spinner
  currentSpinner = nullptr;
  progressEnabled = config.enableProgress;

  LOG_DEBUG("Logging configs. level={}, logFolder={}, logFile={}, "
            "enableTerminal={}, enableFile={}",
            config.level, config.logFolder, config.logFile,
            config.enableTerminal, config.enableFile);
}

class Spinner {
  indicators::ProgressSpinner spinner;
  const std::chrono::milliseconds interval;
  std::atomic<bool> done;
  std::thread *ticker;
  std::string endMsg;

public:
  Spinner(std::string beginMsg, std::string endMsg, int tickIntervalms = 200);
  Spinner(std::string msg, int tickIntervalms = 200)
      : Spinner(msg, msg, tickIntervalms) {}
  ~Spinner();

  void begin();
  void end();
  void set_progress(float) {}
};

Spinner::Spinner(std::string beginMsg, std::string endMsg, int tickIntervalms)
    : interval(tickIntervalms), ticker(nullptr), done(false), endMsg(endMsg),
      spinner(indicators::option::PrefixText{" - "},
              indicators::option::PostfixText{beginMsg},
              indicators::option::ForegroundColor{indicators::Color::white},
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
      ) {
}

Spinner::~Spinner() {
  if (ticker != nullptr) {
    done.store(true);
    ticker->join();
    delete ticker;
  }
}

void Spinner::begin() {
  ticker = new std::thread([this]() {
    while (!this->done.load()) {
      this->spinner.tick();
      auto c = this->spinner.current();
      if (c > 90) {
        this->spinner.set_progress(1);
      }
      std::this_thread::sleep_for(interval);
    }
  });
}

void Spinner::end() {
  if (ticker) {
    done.store(true);
    ticker->join();
    delete ticker;
    ticker = nullptr;
  }
  spinner.set_progress(1);
  spinner.set_option(
      indicators::option::ForegroundColor{indicators::Color::green});
#ifdef __MINGW32__
  spinner.set_option(indicators::option::PrefixText{" - *"});
#else
  spinner.set_option(indicators::option::PrefixText{" - ✔"});
#endif
  spinner.set_option(indicators::option::PostfixText{endMsg});
  spinner.set_option(indicators::option::ShowSpinner{false});
  spinner.tick();
  std::cout << std::endl;
}

void newPhaseSpinner(std::string beginMsg, std::string endMsg,
                     int tickIntervalms) {
  if (!progressEnabled)
    return;

  if (currentSpinner != nullptr) {
    currentSpinner->end();
    delete currentSpinner;
  }

  currentSpinner = new Spinner(beginMsg, endMsg, tickIntervalms);
  currentSpinner->begin();
}

void endPhase() {
  if (!progressEnabled)
    return;

  if (currentSpinner != nullptr) {
    currentSpinner->end();
    delete currentSpinner;
    currentSpinner = nullptr;
  }
}

} // namespace logger
} // namespace xray
