package com.coderrect.githubscanner;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.web.servlet.ServletComponentScan;

@SpringBootApplication
@EnableConfigurationProperties(StorageProperties.class)
@ServletComponentScan
public class GithubScannerApplication {

	public static void main(String[] args) {
		SpringApplication.run(GithubScannerApplication.class, args);
	}

}