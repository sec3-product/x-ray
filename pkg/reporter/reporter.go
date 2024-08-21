package reporter

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/buger/jsonparser"

	"github.com/sec3-product/x-ray/pkg/conflib"
	"github.com/sec3-product/x-ray/pkg/logger"
)

func isDirectory(path string) bool {
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}

	return fi.IsDir()
}

func getPackageVersion() (string, error) {
	coderrectHome, err := conflib.GetCoderrectHome()
	if err != nil {
		return "", err
	}

	content, err := ioutil.ReadFile(filepath.Join(coderrectHome, "VERSION"))
	if err != nil {
		return "", err
	}

	s := strings.Split(string(content), "\n")
	for _, line := range s {
		if strings.HasPrefix(line, "package ") {
			pos := strings.Index(line, " ")
			return line[pos+1:], nil
		}
	}

	return "v0.0.1", nil
}

func showSummary(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	numOfUntrustfulAccounts := 0
	numOfUnsafeOperations := 0
	numOfCosplayAccounts := 0

	// for untrustfulAccounts
	jsonparser.ArrayEach(data, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfUntrustfulAccounts++
	}, "untrustfulAccounts")
	// for unsafeOperations
	jsonparser.ArrayEach(data, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfUnsafeOperations++
	}, "unsafeOperations")

	// for cosplayAccounts
	jsonparser.ArrayEach(data, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfCosplayAccounts++
	}, "cosplayAccounts")

	fmt.Println("")
	if numOfUntrustfulAccounts == 0 && numOfUnsafeOperations == 0 && numOfCosplayAccounts == 0 {
		fmt.Printf("\tNo issues detected\n")
		return nil
	}

	if numOfUntrustfulAccounts > 0 {
		fmt.Printf("\t%2d untrustful account issues\n", numOfUntrustfulAccounts)
	}
	if numOfUnsafeOperations > 0 {
		fmt.Printf("\t%2d unsafe operation issues\n", numOfUnsafeOperations)
	}
	if numOfCosplayAccounts > 0 {
		fmt.Printf("\t%2d cosplay account issues\n", numOfCosplayAccounts)
	}
	return nil
}

func GenerateReport(rawJSONDir string) error {
	reportJSON, err := filepath.Abs(rawJSONDir)
	if err != nil {
		return fmt.Errorf("Fail to get the absolute path of %q: %w", rawJSONDir, err)
	}

	loadIndex := 1
	if !isDirectory(rawJSONDir) {
		loadIndex = 0
		rawJSONDir = filepath.Dir(rawJSONDir)
	}

	_, err = logger.Init("reporter")
	if err != nil {
		return fmt.Errorf("unable to initialize logger: %w", err)
	}

	logger.Infof("Check the cmdlineenv. cmdlineEnv=%v", os.Getenv("CODERRECT_CMDLINE_OPTS"))

	// Parse index.json and create a struct to store the information of
	// conent of index.json.
	var indexInfo IndexInfo
	if loadIndex == 1 {
		if err := ReadIndexJSON(&indexInfo, rawJSONDir); err != nil {
			return fmt.Errorf("unable to read index.json: %w", err)
		}
	} else {
		var execInfo ExecutableInfo
		execInfo.Name = filepath.Base(reportJSON)
		execInfo.Name = strings.TrimSuffix(execInfo.Name, filepath.Ext(execInfo.Name))
		execInfo.ReportJSON = reportJSON
		indexInfo = IndexInfo{Executables: []ExecutableInfo{execInfo}}
		indexInfo.CoderrectVer, err = getPackageVersion()
		if err != nil {
			return fmt.Errorf("unable to get package version: %w", err)
		}
	}

	// --- Generate report for each file.
	for _, exec := range indexInfo.Executables {
		fmt.Printf("\n--------The summary of potential vulnerabilities in %s--------\n", exec.Name)
		if err := showSummary(exec.ReportJSON); err != nil {
			return fmt.Errorf("unable to show summary: %w", err)
		}
		fmt.Print("\n\n")
	}

	// Write index.json to disk.
	WriteIndexJSON(indexInfo, rawJSONDir)

	logger.Infof("Reporter exits gracefully")

	return nil
}
