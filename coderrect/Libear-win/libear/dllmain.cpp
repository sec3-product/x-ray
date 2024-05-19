// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

#include <stdio.h>
#include <windows.h>
#include <detours.h>
#include <limits.h>
#include <strsafe.h>
#include <filesystem>
#include <stdlib.h>
#include <set>
#include <string>
#include <memory>
#include <sstream>
#include <algorithm>
#include <logger.h>

using namespace std;

static HMODULE s_hInst = NULL;
static CHAR dllPathA[MAX_PATH];
static WCHAR dllPathW[MAX_PATH];

static STARTUPINFOA siA;
static STARTUPINFOW siW;
static PROCESS_INFORMATION pi;

HANDLE ghMutex;

static const set<wstring> MS_BUILD_NAME{
    L"cl.exe",
    L"link.exe",
};

wstring ModuleDir;

const DWORD INTERCEPTION_CONTINUE = 0;
const DWORD INTERCEPTION_RETURN = -1;
const LPCWSTR INTERCEPTION_EXE_NAME = L"coderrect-dispatcher.exe";

// true pointers to CreateProcessRelated WINAPI
static BOOL (WINAPI *TrueCreateProcessA)(
     LPCSTR lpApplicationName,
     LPSTR lpCommandLine,
     LPSECURITY_ATTRIBUTES lpProcessAttributes,
     LPSECURITY_ATTRIBUTES lpThreadAttributes,
     BOOL bInheritHandles,
     DWORD dwCreationFlags,
     LPVOID lpEnvironment,
     LPCSTR lpCurrentDirectory,
     LPSTARTUPINFOA lpStartupInfo,
     LPPROCESS_INFORMATION lpProcessInformation
) = CreateProcessA; // true pointer to CreateProcessA

static BOOL (WINAPI *TrueCreateProcessW)(
     LPCWSTR lpApplicationName,
     LPWSTR lpCommandLine,
     LPSECURITY_ATTRIBUTES lpProcessAttributes,
     LPSECURITY_ATTRIBUTES lpThreadAttributes,
     BOOL bInheritHandles,
     DWORD dwCreationFlags,
     LPVOID lpEnvironment,
     LPCWSTR lpCurrentDirectory,
     LPSTARTUPINFOW lpStartupInfo,
     LPPROCESS_INFORMATION lpProcessInformation
) = CreateProcessW;

//__declspec(dllexport) 
BOOL WINAPI UnInterceptedCreateProcessA(
    LPCSTR lpApplicationName,
    LPSTR lpCommandLine,
    LPSECURITY_ATTRIBUTES lpProcessAttributes,
    LPSECURITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpEnvironment,
    LPCSTR lpCurrentDirectory,
    LPSTARTUPINFOA lpStartupInfo,
    LPPROCESS_INFORMATION lpProcessInformation) {

    //printf("2. in unintercepted create processA, %s -- %s\n", lpApplicationName, lpCommandLine);
    //fflush(stdout);
    BOOL ret = TrueCreateProcessA(
        lpApplicationName, lpCommandLine, lpProcessAttributes,
        lpThreadAttributes, bInheritHandles, dwCreationFlags,
        lpEnvironment, lpCurrentDirectory, lpStartupInfo, lpProcessInformation);
    //printf("3. finished!, %s -- %s \n", lpApplicationName, lpCommandLine);
    //fflush(stdout);

    return ret;
}

//__declspec(dllexport) 
BOOL WINAPI UnInterceptedCreateProcessW(
    LPCWSTR lpApplicationName,
    LPWSTR lpCommandLine,
    LPSECURITY_ATTRIBUTES lpProcessAttributes,
    LPSECURITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpEnvironment,
    LPCWSTR lpCurrentDirectory,
    LPSTARTUPINFOW lpStartupInfo,
    LPPROCESS_INFORMATION lpProcessInformation) {

    //printf("2. in unintercepted create processW, %ls -- %ls\n", lpApplicationName, lpCommandLine);
    //fflush(stdout);
    BOOL ret = TrueCreateProcessW(
        lpApplicationName, lpCommandLine, lpProcessAttributes,
        lpThreadAttributes, bInheritHandles, dwCreationFlags,
        lpEnvironment, lpCurrentDirectory, lpStartupInfo, lpProcessInformation);
    //printf("3. finished!, %ls -- %ls\n", lpApplicationName, lpCommandLine);
    //fflush(stdout);

    return ret;
}


void ShowMessage(LPCTSTR lpszFunction, DWORD dwErrorCode)
{
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dwErrorCode,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0, NULL);

    // Display the error message and exit the process

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
        (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));
    StringCchPrintf((LPTSTR)lpDisplayBuf,
        LocalSize(lpDisplayBuf) / sizeof(TCHAR),
        TEXT("%s failed with error %d: %s"),
        lpszFunction, dwErrorCode, lpMsgBuf);
    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
}

void ErrorExit(LPCTSTR lpszFunction, DWORD dwErrorCode)
{
    ShowMessage(lpszFunction, dwErrorCode);
    ExitProcess(dwErrorCode);
}

std::wstring s2ws(const std::string& str)
{
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

wstring& ltrim(wstring& str, const wstring& chars = L"\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

wstring& rtrim(wstring& str, const wstring& chars = L"\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

wstring& trim(wstring& str, const wstring& chars = L"\t\n\v\f\r ")
{
    return ltrim(rtrim(str, chars), chars);
}

DWORD OnInterception(wstring applicationName,
    wstring commandLine) {

    Debug(L"OnInterception: pid=%d, application=%ls, cmdline=%ls\n", GetCurrentProcessId(), applicationName.c_str(), commandLine.c_str());

    if (!applicationName.empty()) {
        wstring appName = L"";
        size_t idx = applicationName.find_last_of(L"\\");
        if (idx != string::npos) {
            appName = applicationName.substr(idx+1);
        }

        trim(appName, L"\"");
        transform(appName.begin(), appName.end(), appName.begin(),
            [](unsigned char c) { return std::tolower(c); });
        if (MS_BUILD_NAME.find(appName) == MS_BUILD_NAME.end()) {
            Debug(L"skip, the command is not concerned: appName=%ls\n", appName.c_str());
            return INTERCEPTION_RETURN;
        }
    }
    else {
        Debug(L"skip, the command is not concerned: commandline=%ls\n", commandLine.c_str());
        return INTERCEPTION_RETURN;
    }

    wostringstream ss;
    ss.str(L"");
    ss.clear();
    ss << ModuleDir << "\\" << INTERCEPTION_EXE_NAME;
    wstring newApplicationName = ss.str();

    ss.str(L"");
    ss.clear();
    ss << INTERCEPTION_EXE_NAME << " " << commandLine;
    wstring newCommandLine = ss.str();

    LPWSTR newClBuffer = new WCHAR[newCommandLine.length() + 1];
    lstrcpyW(newClBuffer, newCommandLine.c_str());

    Debug(L"interception execution: pid=%d, application=%ls, commandline=%ls\n", GetCurrentProcessId(), newApplicationName.c_str(), newClBuffer);

    DWORD dwFlags = CREATE_DEFAULT_ERROR_MODE;
    WaitForSingleObject(ghMutex, INFINITE);
    ZeroMemory(&siW, sizeof(siW));
    ZeroMemory(&siA, sizeof(siA));
    ZeroMemory(&pi, sizeof(pi));
    siA.cb = sizeof(siA);
    siW.cb = sizeof(siW);
    if (TrueCreateProcessW(newApplicationName.c_str(), newClBuffer, NULL, NULL, TRUE, dwFlags, NULL, NULL, &siW, &pi)) {
        Debug(L"interception create ok: pid=%d, application=%ls, commandline=%ls\n", GetCurrentProcessId(), newApplicationName.c_str(), newClBuffer);
        WaitForSingleObject(pi.hProcess, INFINITE);
        ReleaseMutex(ghMutex);
    }
    else {
        DWORD dwError = GetLastError();
        ss.str(L"");
        ss.clear();
        ss << newClBuffer << ", appliname=" << newApplicationName << ", lasterror"<<dwError;
        printf("interception failed: lastError=%ld, application=%ls, commandline=%ls, pid : %ld\n", dwError, applicationName.c_str(), newClBuffer, GetCurrentProcessId());
        ReleaseMutex(ghMutex);
    }

    delete []newClBuffer;
    return INTERCEPTION_CONTINUE;
}


// the interception routines
BOOL WINAPI CoderrectCreateProcessA(
    LPCSTR lpApplicationName,
    LPSTR lpCommandLine,
    LPSECURITY_ATTRIBUTES lpProcessAttributes,
    LPSECURITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpEnvironment,
    LPCSTR lpCurrentDirectory,
    LPSTARTUPINFOA lpStartupInfo,
    LPPROCESS_INFORMATION lpProcessInformation) {

    wstring appName = lpApplicationName ? s2ws(lpApplicationName) : L"";
    wstring cmd = lpCommandLine ? s2ws(lpCommandLine) : L"";
    Debug("CreateProcessA pid=%d, appName=%ls, cmd=%ls\n", GetCurrentProcessId(), appName.c_str(), cmd.c_str());
    DWORD dwRet = OnInterception(appName, cmd);
    if (INTERCEPTION_RETURN == dwRet) {
        goto _exit;
    }

    _exit:
    BOOL ret= DetourCreateProcessWithDllExA(
        lpApplicationName, lpCommandLine, lpProcessAttributes,
        lpThreadAttributes, bInheritHandles, dwCreationFlags,
        lpEnvironment, lpCurrentDirectory, lpStartupInfo, lpProcessInformation,
        dllPathA, UnInterceptedCreateProcessA);
    return ret;
}

BOOL WINAPI CoderrectCreateProcessW(
    LPCWSTR lpApplicationName,
    LPWSTR lpCommandLine,
    LPSECURITY_ATTRIBUTES lpProcessAttributes,
    LPSECURITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpEnvironment,
    LPCWSTR lpCurrentDirectory,
    LPSTARTUPINFOW lpStartupInfo,
    LPPROCESS_INFORMATION lpProcessInformation) {

    wstring appName = lpApplicationName ? lpApplicationName : L"";
    wstring cmd = lpCommandLine ? lpCommandLine : L"";
    Debug("CreateProcessW pid=%d, appName=%ls, cmd=%ls\n", GetCurrentProcessId(), appName.c_str(), cmd.c_str());
    DWORD dwRet = OnInterception(appName, cmd);
    if (INTERCEPTION_RETURN == dwRet) {
        goto _exit;
    }

    _exit:
    BOOL ret = DetourCreateProcessWithDllExW(
        lpApplicationName, lpCommandLine, lpProcessAttributes,
        lpThreadAttributes, bInheritHandles, dwCreationFlags,
        lpEnvironment, lpCurrentDirectory, lpStartupInfo, lpProcessInformation,
        dllPathA, UnInterceptedCreateProcessW);
    return ret;
 }


static VOID Dump(PBYTE pbBytes, LONG nBytes, PBYTE pbTarget)
{
    for (LONG n = 0; n < nBytes; n += 16) {
        printf("    %p: ", pbBytes + n);
        for (LONG m = n; m < n + 16; m++) {
            if (m >= nBytes) {
                printf("  ");
            }
            else {
                printf("%02x", pbBytes[m]);
            }
            if (m % 4 == 3) {
                printf(" ");
            }
        }
        if (n == 0 && pbTarget != DETOUR_INSTRUCTION_TARGET_NONE) {
            printf(" [%p]", pbTarget);
        }
        printf("\n");
    }
}

static VOID Decode(PCSTR pszDesc, PBYTE pbCode, PBYTE pbOther, PBYTE pbPointer, LONG nInst)
{
    if (pbCode != pbPointer) {
        printf("  %s = %p [%p]\n", pszDesc, pbCode, pbPointer);
    }
    else {
        printf("  %s = %p\n", pszDesc, pbCode);
    }

    if (pbCode == pbOther) {
        printf("    ... unchanged ...\n");
        return;
    }

    PBYTE pbSrc = pbCode;
    PBYTE pbEnd;
    PVOID pbTarget;
    for (LONG n = 0; n < nInst; n++) {
        pbEnd = (PBYTE)DetourCopyInstruction(NULL, NULL, pbSrc, &pbTarget, NULL);
        Dump(pbSrc, (int)(pbEnd - pbSrc), (PBYTE)pbTarget);
        pbSrc = pbEnd;
    }
}

static VOID WINAPI Verify(PCHAR pszFunc, PVOID pvPointer)
{
    PVOID pvCode = DetourCodeFromPointer(pvPointer, NULL);

    Decode(pszFunc, (PBYTE)pvCode, NULL, (PBYTE)pvPointer, 3);
}

static VOID WINAPI VerifyEx(PCHAR pszFunc, PVOID pvPointer, LONG nInst)
{
    PVOID pvCode = DetourCodeFromPointer(pvPointer, NULL);

    Decode(pszFunc, (PBYTE)pvCode, NULL, (PBYTE)pvPointer, nInst);
}

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  dwReason,
                       LPVOID reserved) {
    LONG error;

    if (DetourIsHelperProcess()) {
        return TRUE;
    }
    
    if (dwReason == DLL_PROCESS_ATTACH) {
        ghMutex = CreateMutex(
            NULL,              // default security attributes
            FALSE,             // initially not owned
            NULL);             // unnamed mutex

        DetourRestoreAfterWith();
        InitLogger();

        s_hInst = hModule;
        GetModuleFileNameA(s_hInst, dllPathA, ARRAYSIZE(dllPathA));
        GetModuleFileNameW(s_hInst, dllPathW, ARRAYSIZE(dllPathW));
        ModuleDir = dllPathW;
        size_t idx = ModuleDir.find_last_of(L"\\");        
        Debug(L"modulepath=%ls, idx=%d, length=%d\n", ModuleDir.c_str(), idx, ModuleDir.length());
        if (idx != string::npos) {
            ModuleDir = ModuleDir.substr(0, idx);
        }
        Debug(L"moduleDir=%ls, dllpath=%ls", ModuleDir.c_str(), dllPathW);

        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourAttach(&(PVOID&)TrueCreateProcessA, CoderrectCreateProcessA);
        DetourAttach(&(PVOID&)TrueCreateProcessW, CoderrectCreateProcessW);

        error = DetourTransactionCommit();
        if (error != NO_ERROR) {
            printf("libear" DETOURS_STRINGIFY(DETOURS_BITS) ".dll: "
                " Error detouring CreateProcess(): %ld\n", error);
        }
    }
    else if (dwReason == DLL_PROCESS_DETACH) {
        CloseHandle(ghMutex);

        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourDetach(&(PVOID&)TrueCreateProcessA, CoderrectCreateProcessA);
        DetourDetach(&(PVOID&)TrueCreateProcessW, CoderrectCreateProcessW);

        error = DetourTransactionCommit();
    }

    return TRUE;
}
