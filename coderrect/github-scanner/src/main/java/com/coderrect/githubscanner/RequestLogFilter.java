package com.coderrect.githubscanner;

import java.io.IOException;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebFilter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.web.filter.OncePerRequestFilter;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@WebFilter("/api/*")
public class RequestLogFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        StringBuilder builder = new StringBuilder(1024);
        builder.append(request.getRemoteAddr()).append(" ");
        builder.append(request.getMethod()).append(" ").append(request.getRequestURI()).append(" ")
                .append(request.getProtocol()).append(" ");
        if (request.getQueryString() != null) {
            builder.append("?").append(request.getQueryString());
        }
        filterChain.doFilter(request, response);

        builder.append(response.getStatus()).append(" ");
        log.info(builder.toString());
    }

}
