package com.coderrect.githubscanner;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/v1/report")
public class ReportController {

    private final String TIME_PATTERN = "yyyy-MM-dd HH:mm:ss";

    private final DateTimeFormatter dtFormatter = DateTimeFormatter.ofPattern(TIME_PATTERN);

    private final StorageService storageService;

    private final ReportHistoryService reportHistoryService;

    @Autowired
    public ReportController(StorageService storageService, ReportHistoryService reportHistoryService) {
        this.storageService = storageService;
        this.reportHistoryService = reportHistoryService;
    }

    @PostMapping("upload")
    public ApiResponse<Object> handleFileUpload(@RequestParam("reportId") String reportId,
            @RequestParam("repoUrl") String repoUrl, @RequestParam("compressedFile") MultipartFile compressedFile) {
        storageService.decompression(compressedFile, reportId);
        reportHistoryService.add(repoUrl, reportId);
        return ApiResponse.success(null);
    }

    @GetMapping("/history")
    public ApiResponse<List<ReportItem>> getHistory(
            @RequestParam(value = "gmtStartTime", required = false) String gmtStartTime,
            @RequestParam(value = "gmtEndTime", required = false) String gmtEndTime) {

        LocalDateTime startTime, endTime;
        if (!StringUtils.hasText(gmtStartTime)) {
            startTime = LocalDateTime.now(ZoneId.of("UTC")).withHour(0).withMinute(0).withSecond(0).withNano(0);
        } else {
            startTime = LocalDateTime.parse(gmtStartTime, dtFormatter);
        }

        if (!StringUtils.hasText(gmtEndTime)) {
            endTime = LocalDateTime.now(ZoneId.of("UTC")).withHour(23).withMinute(59).withSecond(59).withNano(0);
        } else {
            endTime = LocalDateTime.parse(gmtEndTime, dtFormatter).atZone(ZoneId.of("UTC")).toLocalDateTime();
        }

        if (startTime.isAfter(endTime)) {
            throw new BizException(String.format("The end time is before start time: %s, %s", startTime, endTime));
        }
        return ApiResponse.success(reportHistoryService.get(startTime, endTime));
    }
}
