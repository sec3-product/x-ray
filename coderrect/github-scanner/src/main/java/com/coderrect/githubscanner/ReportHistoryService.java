package com.coderrect.githubscanner;

import java.time.LocalDateTime;
import java.util.List;

public interface ReportHistoryService {

    List<ReportItem> get(LocalDateTime startTime, LocalDateTime endTime);

    void add(String repoUrl, String reportId);
}
