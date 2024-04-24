package com.coderrect.githubscanner;

import java.time.LocalDateTime;

import lombok.Data;

@Data
public class ReportItem {
    private String repoUrl;

    private LocalDateTime reportTime;

    private String reportId;
}
