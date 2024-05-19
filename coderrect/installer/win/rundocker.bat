@REM check if coderrect docker image has been installed
@REM same as `docker image ls | grep coderrect/windows | wc -l`
@echo off
FOR /F %%i IN ('docker image ls ^| findstr coderrect/windows-installer ^| findstr 1.1 ^| find /c /v ""') DO SET IMAGE=%%i
IF %IMAGE% NEQ 1 (
    IF NOT "%1"=="" (
        IF "%1"=="-d" (
            GOTO :PullDockerImage
        )
    )
    ECHO Build Coderrect Docker Image
    docker build -f .\Dockerfile.win -t coderrect/windows-installer:1.1 .
    GOTO :StartContainer

    :PullDockerImage
    docker pull coderrect/windows-installer:1.1
    )

:StartContainer
ECHO Start Docker Container
docker run --rm -it -v c:\code:c:\code coderrect/windows-installer:1.1 powershell