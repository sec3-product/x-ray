package com.coderrect.githubscanner;

import java.io.BufferedWriter;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.QuoteMode;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class ReportHistoryServiceImpl implements ReportHistoryService {
    private final Path historyFilePath;

    private final String FILE_NAME_HISTORY = "history.db";

    private final String TIME_PATTERN = "yyyy-MM-dd HH:mm:ss";

    private final DateTimeFormatter dtFormatter = DateTimeFormatter.ofPattern(TIME_PATTERN);

    private final String HEADER_REPORT_ID = "reportId";
    private final String HREADER_REPO_URL = "repoUrl";
    private final String HEADER_REPORT_DATE = "reportDate";
    String[] HEADERS = { HEADER_REPORT_ID, HREADER_REPO_URL, HEADER_REPORT_DATE };

    @Autowired
    public ReportHistoryServiceImpl(StorageProperties properties) {
        this.historyFilePath = Paths.get(properties.getHistoryLocation(), FILE_NAME_HISTORY);
        if (!this.historyFilePath.getParent().toFile().exists()) {
            throw new StorageException(
                    String.format("The history storage location does not exist: %s", historyFilePath.toString()));
        }

        if (!historyFilePath.toFile().exists()) {
            try (BufferedWriter writer = Files.newBufferedWriter(this.historyFilePath, StandardOpenOption.CREATE);
                    CSVPrinter csvPrinter = new CSVPrinter(writer,
                            CSVFormat.DEFAULT.withHeader(HEADERS).withQuoteMode(QuoteMode.ALL));) {
            } catch (Exception e) {
                throw new StorageException(String.format("Create history file error: %s", historyFilePath.toString()));
            }
        }
    }

    @Override
    public List<ReportItem> get(LocalDateTime startTime, LocalDateTime endTime) {
        LinkedList<ReportItem> reportItems = new LinkedList<>();
        log.debug("the report is between: {} to {}", startTime, endTime);
        try (Reader reader = Files.newBufferedReader(this.historyFilePath);
                CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader());) {
            for (CSVRecord csvRecord : csvParser) {

                LocalDateTime reportTime = LocalDateTime.parse(csvRecord.get(HEADER_REPORT_DATE), dtFormatter);
                log.debug("the report time is {}", reportTime);

                long diffStart = ChronoUnit.SECONDS.between(startTime, reportTime);
                long diffEnd = ChronoUnit.SECONDS.between(reportTime, endTime);
                if (diffStart >= 0 && diffEnd >= 0) {
                    ReportItem reportItem = new ReportItem();
                    reportItem.setRepoUrl(csvRecord.get(HREADER_REPO_URL));
                    reportItem.setReportId(csvRecord.get(HEADER_REPORT_ID));
                    reportItem.setReportTime(reportTime);
                    reportItems.addFirst(reportItem);
                }
            }
        } catch (NoSuchFileException noe) {
            log.warn("History db not created now");
        } catch (Exception e) {
            throw new StorageException(String.format("Parse history file error: %s", historyFilePath.toString()));
        }
        return reportItems;
    }

    @Override
    public void add(String repoUrl, String reportId) {
        try (BufferedWriter writer = Files.newBufferedWriter(this.historyFilePath, StandardOpenOption.APPEND,
                StandardOpenOption.CREATE);
                CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT.withQuoteMode(QuoteMode.ALL));) {
            csvPrinter.print(reportId);
            csvPrinter.print(repoUrl);
            csvPrinter.print(LocalDateTime.now(ZoneId.of("UTC")).format(dtFormatter));
            csvPrinter.println();
            csvPrinter.flush();
        } catch (Exception e) {
            throw new StorageException(String.format("Add history item error: %s", historyFilePath.toString()));
        }
    }
}
