@REM Script to build LLVM on Windows
@REM NOTE: only used in coderrect windows installer docker container

@echo off
SET CURRENT="%CD%"

@REM if -d is passed, download prebuilt llvm
IF NOT "%1"=="" (
    IF "%1"=="-d" (
        GOTO :DownloadPrebuiltLLVM
    )
)

@REM if llvm repo doesn't exist, download prebuilt llvm
IF NOT EXIST C:\code\classic-flang-llvm-project (
    ECHO class-flang-llvm-project doesn't exist in C:\code.
    GOTO :DownloadPrebuiltLLVM
    EXIT /B 0
)

:BuildLLVM
cd C:\code\classic-flang-llvm-project
IF NOT EXIST build (
    mkdir build
) ELSE (
    ECHO Found `build` folder, LLVM may already be built.
    ECHO If not, please remove the `build` folder
)
cd build
cmake -G "MinGW Makefiles" -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang ../llvm
SET flag=%1
IF "%flag%"=="" (
    SET flag=-j8
)
@REM echo %flag%
make %flag%
cd %CURRENT%
EXIT /B 0

@REM Download a prebuilt llvm instead of building it
@REM NOTE: use with caution, the prebuilt llvm may be outdated
:DownloadPrebuiltLLVM
cd C:\code
IF EXIST C:\code\classic-flang-llvm-project.zip (
    SET /P REDOWNLOAD="Existing prebuilt package found, do you want to re-download it? (Y/[N], default: N)?"
    IF /I "%REDOWNLOAD%" NEQ "Y" (
        ECHO Unzipping the existing classic-flang-llvm-project.zip...
        GOTO :UnzipLLVM
    )
)
ECHO Downloading the prebuilt package classic-flang-llvm-project.zip...
ECHO The progress bar is turned off due to its significant slowdown. Please be patient :).
powershell -command "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri https://public-installer-pkg.s3.us-east-2.amazonaws.com/windows/classic-flang-llvm-project.zip -OutFile classic-flang-llvm-project.zip"

@REM Unzip the prebuilt llvm
:UnzipLLVM
ECHO Deleting old classic-flang-llvm-project folder...
rmdir /s /q classic-flang-llvm-project
ECHO Unzipping classic-flang-llvm-project.zip...
tar -xf classic-flang-llvm-project.zip

cd %CURRENT%
EXIT /B 0