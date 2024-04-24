// logger.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "logger.h"

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>

void InitLogger()
{
#ifndef PROD
    /// Strange build error when you use colored console logging
    static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    //static plog::RollingFileAppender<plog::TxtFormatter> fileAppender("log.interception", 8000, 3); 
    plog::init(plog::debug, &consoleAppender)
        //.addAppender(&fileAppender)
        ;
    //plog::get()->setMaxSeverity(plog::debug);
    printf("debug");
#endif
}