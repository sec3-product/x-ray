#pragma once

#include <memory>
#include <ostream>
#include <string>

#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_ostream.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

namespace xray {
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

// Must call this before using logging.
void init(LoggingConfig cfg);

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__);
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__);
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__);
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__);
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__);

// Example usage
namespace {
[[maybe_unused]] void example() {
  LOG_DEBUG("Parsing the source code. file={}", "somefile.c");
  LOG_ERROR("Unexpected file type. file={}", "program.exe");
}

} // namespace

// Call newPhaseSpinner to start a new Progress Spinner.
// newPhaseSpinner automatically completes a previous spinner if one exists
// Begin text is the message to display while the spinner is in progress
// End text is the message to display once the spinner had completed
// To complete the spinner either call newPhaseSpinner again to end this spiiner
// and start a new one
//  or call endPhase, which will end the current spinner explictly
// These functions do nothing if logging to terminal is enabled
void newPhaseSpinner(std::string beginMsg, std::string endMsg,
                     int tickIntervalms = 200);
inline void newPhaseSpinner(std::string msg, int tickIntervalms = 200) {
  newPhaseSpinner(msg, msg, tickIntervalms);
}
void endPhase();

} // namespace logger
} // namespace xray

// Define the formatter for LLVM types.
namespace fmt {
inline namespace v9 {

template <typename T> struct llvm_formatter {
  constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
    return ctx.end();
  }

  template <typename FormatContext>
  auto format(const T &value, FormatContext &ctx) -> decltype(ctx.out()) {
    std::string str;
    llvm::raw_string_ostream rso(str);
    value.print(rso);
    return format_to(ctx.out(), "{}", rso.str());
  }
};

#define DEFINE_LLVM_FORMATTER(Type)                                            \
  template <> struct formatter<Type> : llvm_formatter<Type> {};

DEFINE_LLVM_FORMATTER(llvm::Instruction)
DEFINE_LLVM_FORMATTER(llvm::Type)
DEFINE_LLVM_FORMATTER(llvm::User)
DEFINE_LLVM_FORMATTER(llvm::Value)
DEFINE_LLVM_FORMATTER(llvm::GetElementPtrInst)
DEFINE_LLVM_FORMATTER(llvm::GlobalAlias)
DEFINE_LLVM_FORMATTER(llvm::ExtractValueInst)
DEFINE_LLVM_FORMATTER(llvm::MemCpyInst)
DEFINE_LLVM_FORMATTER(llvm::AllocaInst)
DEFINE_LLVM_FORMATTER(llvm::DICompositeType)
DEFINE_LLVM_FORMATTER(llvm::GlobalVariable)
DEFINE_LLVM_FORMATTER(llvm::StoreInst)
DEFINE_LLVM_FORMATTER(llvm::BitCastInst)

} // namespace v9
} // namespace fmt

// Define a custom formatter for spdlog::level::level_enum.
template <>
struct fmt::formatter<spdlog::level::level_enum> : fmt::formatter<std::string> {
  template <typename FormatContext>
  auto format(spdlog::level::level_enum level, FormatContext &ctx) {
    // Convert the level_enum to string and format it.
    std::string level_str = spdlog::level::to_string_view(level).data();
    return fmt::formatter<std::string>::format(level_str, ctx);
  }
};
