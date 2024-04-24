#ifndef LOGGER_H
#define LOGGER_H

#pragma once

#include <string>
#include <iostream>
#include <plog/Log.h>

template<typename T, typename... Args>
void Debug(const T& fmt, Args...args)
{
	PLOGD.printf(fmt, args...);
}

template<typename T, typename... Args>
void Info(const T& fmt, Args...args)
{
    PLOGI.printf(fmt, args...);
}

template<typename T, typename... Args>
void Warn(const T& fmt, Args...args)
{
    PLOGW.printf(fmt, args...);
}

template<typename T, typename... Args>
void Error(const T& fmt, Args...args)
{
    PLOGE.printf(fmt, args...);
}

template<typename T, typename... Args>
void Critical(const T& fmt, Args...args)
{
    PLOGF.printf(fmt, args...);
}

void InitLogger();

#endif // LOGGER_H
