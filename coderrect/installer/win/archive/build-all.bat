@REM build all project on windows
@REM 
@REM make sure the installation requirement as following:
@REM visual studio 2019 (libear)
@REM visual studio 2017 (llvm)
@REM golang sdk: (gosrc)
@REM python 3: (llvm)
@REM nsis 3: https://nsis.sourceforge.io/Download (installer)
@REM 


@REM ---------------------------------------
@REM Main()
@REM ---------------------------------------

@echo off
set workingDir="%CD%"
set installDir="%CD%\build"
set help=0
set develop=0
set branchVersion=
set pkgVersion=dev

@REM pase arguments
:loop
IF NOT "%1"=="" (
    SET hit=0
    IF "%1"=="-h" (
        SET help=1        
        GOTO :optional
    ) 
    IF "%1"=="-d" (
        SET develop=1
        GOTO :optional
    ) 
    IF "%1"=="-r" (
        SET branchVersion=%2
        SET pkgVersion=%2
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

if %help% == 1 (
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
)

call :MakeLayout
call :MakeVersion
call :CopyExample

call :BuildLibear
call :BuildGosrc
call :BuildRaced
call :BuildLLVM
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
mkdir build\examples

xcopy /e /y ..\package\* build\
EXIT /B 0

@REM ---------------------------------------
@REM Get Utc Second
@REM ---------------------------------------
:GetUtcSecond2
@REM LocalDateTime=20201224133908.385000+480, YYYYMMDDHHMMSS.milliseconds+GMT_Offset_in_minutes
for /f "usebackq tokens=1,2 delims==.+ " %%i in (`wmic os get LocalDateTime /value`) do @if %%i==LocalDateTime (
     set datetime=%%j
)

set month=%datetime:~0,4%-%datetime:~4,2%
set "%~1=%datetime%"
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
cd ..\..\Libear-win

@REM for 64bit
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\devenv" Libear-win.sln /Rebuild "Release|x64"
move x64\Release\libear-win.exe %installDir%\bin\
move x64\Release\libear64.dll %installDir%\bin\

@REM for 32bit?
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\devenv" Libear-win.sln /Rebuild "Release|x86"
move win32\Release\libear32.dll %installDir%\bin\
EXIT /B 0


@REM ---------------------------------------
@REM build golang project
@REM ---------------------------------------
:BuildGosrc
cd %workingDir%
cd ..\..\gosrc
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64\nmake" -f makefile.win all
move bin\*.* %installDir%\bin\
EXIT /B 0


@REM ---------------------------------------
@REM copy example
@REM ---------------------------------------
:CopyExample
cd %workingDir%
xcopy /e /y ..\..\benchmark\windows\* build\examples\
EXIT /B 0


@REM ---------------------------------------
@REM Get LLVM binary
@REM ---------------------------------------
:BuildLLVM
cd %workingDir%
cd %workingDir%
powershell -Command "Invoke-WebRequest https://custom-clang.s3-us-west-2.amazonaws.com/llvm-win64-dev.zip -OutFile llvm.zip"
powershell -command "Expand-Archive -Force 'llvm.zip' '%~dp0build\clang'"
del llvm.zip
EXIT /B 0


@REM ---------------------------------------
@REM Get Racedetect binary
@REM ---------------------------------------
:BuildRaced
cd %workingDir%
powershell -Command "Invoke-WebRequest https://llvmrace-binary.s3-us-west-2.amazonaws.com/racedetect-win64-dev.zip -OutFile raced.zip"
powershell -command "Expand-Archive -Force 'raced.zip' '%~dp0build\bin\'"
del raced.zip
EXIT /B 0


@REM ---------------------------------------
@REM Package in setup.exe
@REM ---------------------------------------
:Package
cd %workingDir%
"C:\Program Files (x86)\NSIS\makensis.exe" /DPRODUCT_VERSION=%pkgVersion% package.nsi 
if %develop%==1 (
    move Setup.exe ..\coderrect-win64-develop.exe
) else (
    move Setup.exe ..\coderrect-win64-%pkgVersion%.exe
)

EXIT /B 0

