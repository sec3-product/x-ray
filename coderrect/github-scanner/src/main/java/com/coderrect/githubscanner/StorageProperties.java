package com.coderrect.githubscanner;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("storage")
public class StorageProperties {
    /**
     * Folder location for storing files
     */
    private String location = "upload-dir";

    private String historyLocation = "data";

    public String getLocation() {
        int idx = location.indexOf("file:");
        if (idx != -1) {
            return location.substring("file:".length());
        }
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public String getHistoryLocation() {
        return historyLocation;
    }

    public void setHistoryLocation(String location) {
        this.historyLocation = location;
    }
}
