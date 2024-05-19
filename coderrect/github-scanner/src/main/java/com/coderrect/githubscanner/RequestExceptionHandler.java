package com.coderrect.githubscanner;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@ControllerAdvice
public class RequestExceptionHandler {

    @ExceptionHandler
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    @ResponseBody
    public ApiResponse unknownException(Exception ex) {
        log.error("Fail to handle request: {}", ex.getMessage(), ex);
        return ApiResponse.fail(1, ex.getMessage());
    }

    @ExceptionHandler(StorageFileNotFoundException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    @ResponseBody
    public ApiResponse<Object> handleStorageFileNotFound(StorageFileNotFoundException exc) {
        return ApiResponse.fail(1, exc.getMessage());
    }
}
