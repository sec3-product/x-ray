@REM Build all project on windows
@REM NOTE: This script requires a ton of dependencies and a specific folder structure
@REM 1. Use it within coderrect windows container (Recommendeded but not necessary)
@REM 2. Make sure all dependencies below are installed and under PATH (except vs2019);
@REM    and put all github repos under c:\code\
@REM 
@REM Dependencies (use chocolatey to install them):
@REM 1. MinGW 8.1.0 (gcc/g++)
@REM 2. cmake (use cmake.portable package in choco)
@REM 3. make
@REM 4. python3 (needed for LLVM)
@REM 5. golang
@REM 6. nsis3 + advanced logging + EnVar Plugin (used for create windows installer)
@REM 7. visual studio 2019 (libear-win)
@REM All above should already been installed in the windows installer docker image
@REM 
@REM External Dependencies:
@REM 1. LLVM must be built beforehand using MinGW and mounted to docker container
@REM 2. All coderrect's repositories (should all be placed under c:\code)


@REM ---------------------------------------
@REM Main()
@REM ---------------------------------------

@echo off
set workingDir="%CD%"
set installDir="%CD%\build"
set help=0
set build=0
set develop=0
set branchVersion=
set pkgVersion=dev

@REM parse arguments
:loop
IF NOT "%1"=="" (
    IF "%1"=="-h" (
        SET help=1        
        GOTO :optional
    )
    IF "%1"=="-d" (
        SET develop=1
        SET build=1
        GOTO :optional
    )
    IF "%1"=="-r" (
        SET branchVersion=%2
        SET pkgVersion=%2
        SET build=1
        SHIFT
        GOTO :optional
    )
    goto :positional

    :optional
    SHIFT
    GOTO :loop
    
    :positional
    SET pkgVersion=%1
    SHIFT
    GOTO:loop 
)
@REM end of parse argument 

echo help=%help%
echo d=%develop%
echo branchVersion=%branchVersion%
echo packageVersion=%pkgVersion%

set showhelp=0
if %help% == 1 SET showhelp=1
if %build% == 0 SET showhelp=1
if %showhelp% == 1 (
    :usage
    echo build and package all the binary files into a single installer
    echo[
    echo build-all [options] version [effective_version]
    echo[
    echo     -h print this message
    echo     -d build from the develop branches
    echo     -r build from the release branches
    echo  1. build the package v0.0.1 from release branches
    echo  build-all -r 0.0.1
    echo[
    echo  2. build the package from the develop branch
    echo  build-pkg -d
    echo[
    echo  3. build the package from the release-v0.8.0 branch 
    echo  but set VERSION file to be 0.8.1
    echo  build-pkg -r 0.8.0 0.8.1
    echo[
    EXIT /B 0
)

call :MakeLayout
call :MakeVersion
call :CopyExample

call :BuildGosrc
call :BuildRaced
call :BuildLLVM
call :BuildLibear
call :DownloadDLLs
call :Package

@REM return to workding dir
cd %workingDir%
EXIT /B 0


@REM ---------------------------------------
@REM Make build directory layout
@REM ---------------------------------------
:MakeLayout
@REM clean previous battlefield
rmdir /s /q build

@REM copy package files
mkdir build
mkdir build\bin
mkdir build\clang
mkdir build\clang\bin
mkdir build\clang\include
mkdir build\clang\include\clang-c
mkdir build\clang\include\llvm-c
mkdir build\clang\lib
mkdir build\clang\lib\clang
mkdir build\examples

xcopy /e /y ..\package\* build\
EXIT /B 0

@REM ---------------------------------------
@REM Get Utc Second
@REM ---------------------------------------
:GetUtcSecond
FOR /F "tokens=* USEBACKQ" %%F IN (`powershell -command "[int32](New-TimeSpan -Start (Get-Date "01/01/1970") -End (Get-Date)).TotalSeconds"`) DO (
    SET utcsecond=%%F
)
set "%~1=%utcsecond%"
EXIT /B 0


@REM ---------------------------------------
@REM Make version
@REM ---------------------------------------
:MakeVersion
call :GetUtcSecond buildNumber
echo "package (system) %pkgVersion% build %buildNumber%" > build\VERSION
EXIT /B 0


@REM ---------------------------------------
@REM build libear
@REM ---------------------------------------
:BuildLibear
cd %workingDir%
cd c:\code\coderrect\Libear-win

@REM Load VS 2019 command line env
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"

@REM Build x64 libear-win and libear64.dll
@REM libear-win is similar to coderrect-exec
devenv.com .\Libear-win.sln /Build "Release|x64"
move x64\Release\libear-win.exe %installDir%\bin\
move x64\Release\libear64.dll %installDir%\bin\

devenv.com .\Libear-win.sln /Build "Release|x86"
move Win32\Release\libear32.dll %installDir%\bin\

EXIT /B 0


@REM ---------------------------------------
@REM build golang executables
@REM ---------------------------------------
:BuildGosrc
cd %workingDir%
cd c:\code\coderrect\gosrc
make windows
move bin\*.* %installDir%\bin\
EXIT /B 0


@REM ---------------------------------------
@REM copy example
@REM ---------------------------------------
:CopyExample
cd %workingDir%
xcopy /e /y c:\code\coderrect\benchmark\windows\* build\examples\
EXIT /B 0


@REM ---------------------------------------
@REM Copy LLVM
@REM This function assumes LLVM is already built
@REM Because building LLVM is time consuming and LLVM changes infrequently
@REM ---------------------------------------
:BuildLLVM
cd %workingDir%
xcopy /e /y c:\code\classic-flang-llvm-project\build\bin\llvm-link.exe build\clang\bin\
xcopy /e /y c:\code\classic-flang-llvm-project\build\bin\clang-cl.exe build\clang\bin\
xcopy /s /e /y c:\code\classic-flang-llvm-project\build\lib\clang build\clang\lib\clang
xcopy /s /y c:\code\classic-flang-llvm-project\clang\include\clang-c build\clang\include\clang-c
xcopy /s /y c:\code\classic-flang-llvm-project\llvm\include\llvm-c build\clang\include\llvm-c
EXIT /B 0


@REM ---------------------------------------
@REM Build Racedetect binary
@REM ---------------------------------------
:BuildRaced
cd c:\code\LLVMRace
if not exist build (
    mkdir build
    cd build
    cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=c:\code\classic-flang-llvm-project\build\lib\cmake\llvm ..
    ) else (cd build)
make
cd %workingDir%
xcopy /e /y c:\code\LLVMRace\build\bin\* build\bin
EXIT /B 0


@REM ---------------------------------------
@REM Download extra dlls (This should be fixed in the future)
@REM ---------------------------------------
:DownloadDLLs
cd %workingDir%
cd build\bin
powershell -command "Invoke-WebRequest -Uri https://public-installer-pkg.s3.us-east-2.amazonaws.com/libgcc_s_seh-1.dll -OutFile libgcc_s_seh-1.dll"
powershell -command "Invoke-WebRequest -Uri https://public-installer-pkg.s3.us-east-2.amazonaws.com/libstdc%%2B%%2B-6.dll -OutFile libstdc++-6.dll"
powershell -command "Invoke-WebRequest -Uri https://public-installer-pkg.s3.us-east-2.amazonaws.com/libwinpthread-1.dll -OutFile libwinpthread-1.dll"
EXIT /B 0


@REM ---------------------------------------
@REM Package in setup.exe
@REM ---------------------------------------
:Package
cd %workingDir%
makensis.exe /DPRODUCT_VERSION=%pkgVersion% package.nsi 
if %develop%==1 (
    move Setup.exe ..\coderrect-win64-develop.exe
) else (
    move Setup.exe ..\coderrect-win64-%pkgVersion%.exe
)
EXIT /B 0
