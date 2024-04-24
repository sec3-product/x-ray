@REM clean all intermediate files in benchmark
@REM 


@REM ---------------------------------------
@REM Main()
@REM ---------------------------------------

@echo off
set workingDir=%CD%
set installDir=%CD%\build

call :CleanBuildGarbage win32sync
call :CleanBuildGarbage std
call :CleanBuildGarbage hellovs

@REM return to workding dir
cd %workingDir%
EXIT /B 0

@REM ---------------------------------------
@REM Clean visual studio build temporary files
@REM ---------------------------------------
:CleanBuildGarbage
cd %workingDir%
cd ..\..\benchmark\windows\%~1

echo cleanning the dir: %~1

FOR /D %%i IN (*) DO (
    if exist %%i\x64 (        
        rmdir /q /s %%i\x64
    )

    if exist %%i\.vs (        
        rmdir /q /s %%i\.vs
    )

    if exist %%i\Debug (        
        rmdir /q /s %%i\Debug
    )
)

@REM delete root
if exist x64 (        
    rmdir /q /s x64
)
if exist .vs (        
    rmdir /q /s .vs
)
if exist Debug (        
    rmdir /q /s Debug
)


EXIT /B 0
