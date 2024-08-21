package reporter

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/buger/jsonparser"

	"github.com/coderrect-inc/coderrect/pkg/util/conflib"
	"github.com/coderrect-inc/coderrect/pkg/util/logger"
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

func showSummary(raceJsonPath string) bool {
	numOfOMP := 0
	numOfNormalDataRaces := 0
	numOfDeadlocks := 0
	numOfMismatchedAPIs := 0
	numOfUntrustfulAccounts := 0
	numOfUnsafeOperations := 0
	numOfCosplayAccounts := 0
	numOfOrderViolations := 0
	numOfTOCTOU := 0
	numOfRaceConditions := 0
	total := 0

	type DataRace struct {
		IsOmpRace bool `json:"isOmpRace"`
	}

	raceJsonBytes, err := ioutil.ReadFile(raceJsonPath)
	if err != nil {
		//return err
		return true
	}

	// for normal data races
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		var dataRace DataRace
		if json.Unmarshal(value, &dataRace) == nil {
			if dataRace.IsOmpRace {
				numOfOMP++
			} else {
				numOfNormalDataRaces++
			}
			total++
		}
	}, "dataRaces")

	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		total++
		numOfOrderViolations++
	}, "orderViolations")

	// for atomicity violations
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		total++
		numOfRaceConditions++
	}, "raceConditions")

	// for deadlocks
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfDeadlocks++
		total++
	}, "deadLocks")

	// for mismatchedAPIs
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfMismatchedAPIs++
		total++
	}, "mismatchedAPIs")

	// for untrustfulAccounts
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfUntrustfulAccounts++
		total++
	}, "untrustfulAccounts")
	// for unsafeOperations
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfUnsafeOperations++
		total++
	}, "unsafeOperations")
	// for cosplayAccounts
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfCosplayAccounts++
		total++
	}, "cosplayAccounts")

	// for TOCTOU
	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		numOfTOCTOU++
		total++
	}, "toctou")

	fmt.Println("")

	if total == 0 {
		//fmt.Println("No vulnerabilities detected.")
		os.Stdout.WriteString(fmt.Sprintf("\tNo vulnerabilities detected\n"))
		return false
	}

	if numOfNormalDataRaces > 0 {
		os.Stdout.WriteString(fmt.Sprintf("\t%2d shared data races\n", numOfNormalDataRaces))
	}
	if numOfOMP > 0 {
		os.Stdout.WriteString(fmt.Sprintf("\t%2d OpenMP races\n", numOfOMP))
	}
	if numOfRaceConditions > 0 {
		os.Stdout.WriteString(fmt.Sprintf("\t%2d atomicity violations\n", numOfRaceConditions))
	}
	// FIXME: we do not report deadlock in our report for now
	// if numOfDeadlocks > 0 {
	// 	fmt.Printf("\t%2d deadlocks\n", numOfDeadlocks)
	// }
	if numOfMismatchedAPIs > 0 {
		fmt.Printf("\t%2d mis-matched API issues\n", numOfMismatchedAPIs)
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

	// FIXME: we do not report OV in our report for now
	// if numOfOrderViolations > 0 {
	// 	fmt.Printf("\t%2d order violations\n", numOfOrderViolations)
	// }
	if numOfTOCTOU > 0 {
		fmt.Printf("\t%2d TOCTOU\n", numOfTOCTOU)
	}

	return true
}

func getNumberOfDataRace(raceJsonPath string) int {
	total := 0

	type DataRace struct {
		IsOmpRace bool `json:"isOmpRace"`
	}

	raceJsonBytes, err := ioutil.ReadFile(raceJsonPath)
	if err != nil {
		return 0
	}

	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		var dataRace DataRace
		if json.Unmarshal(value, &dataRace) == nil {
			total++
		}
	}, "dataRaces")

	return total
}

func getNumberOfRaceConditions(raceJsonPath string) int {
	total := 0

	raceJsonBytes, err := ioutil.ReadFile(raceJsonPath)
	if err != nil {
		return 0
	}

	jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		total += 1
	}, "raceConditions")

	return total
}

type ReportInfo struct {
	Name string
	Num  int
}

func GenerateReport(rawJSONDir string) error {
	raceJSONPath, err := filepath.Abs(rawJSONDir)
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
		os.Stderr.WriteString(fmt.Sprintf("Fail to inititalize logger. err=%v", err))
		os.Exit(1)
	}

	logger.Infof("Check the cmdlineenv. cmdlineEnv=%v", os.Getenv("CODERRECT_CMDLINE_OPTS"))

	// parse index.json and create a struct to store the information of conent of index.json
	var indexInfo IndexInfo
	if loadIndex == 1 {
		if err := ReadIndexJSON(&indexInfo, rawJSONDir); err != nil {
			logger.Errorf("Failed to read index.json %q: %v", rawJSONDir, err)
			os.Exit(1)
		}
	} else {
		var execInfo ExecutableInfo
		execInfo.Name = filepath.Base(raceJSONPath)
		execInfo.Name = strings.TrimSuffix(execInfo.Name, filepath.Ext(execInfo.Name))

		execInfo.DataRaces = getNumberOfDataRace(raceJSONPath)
		execInfo.RaceConditions = getNumberOfRaceConditions(raceJSONPath)
		execInfo.RaceJSON = raceJSONPath
		indexInfo = IndexInfo{Executables: []ExecutableInfo{execInfo}}
		indexInfo.CoderrectVer, _ = getPackageVersion()
	}

	// --- Filter races
	// the executableList will maintain a summary of filtered race report for later display
	var executableList []ReportInfo
	var executablesWithNoRace []string
	for i := 0; i < len(indexInfo.Executables); i++ {
		executable := indexInfo.Executables[i]
		if executable.DataRaces == 0 && executable.RaceConditions == 0 {
			executablesWithNoRace = append(executablesWithNoRace, executable.Name)
			continue
		}
		switch num := Rewrite(executable.RaceJSON); num {
		case 0:
			executablesWithNoRace = append(executablesWithNoRace, executable.Name)
		case -1:
			num = executable.DataRaces + executable.RaceConditions
		default:
			executableList = append(executableList, ReportInfo{executable.Name, num})
		}
	}

	// --- Generate report for each executable
	// For each executable with races, we create a report
	for i := 0; i < len(indexInfo.Executables); i++ {
		executable := indexInfo.Executables[i]
		rawRaceJSONPath := executable.RaceJSON
		fmt.Printf("\n--------The summary of potential vulnerabilities in %s--------\n", executable.Name)
		_ = showSummary(rawRaceJSONPath)
		fmt.Print("\n\n")
	}

	// update index.json with the information "if any new race was found"
	WriteIndexJSON(indexInfo, rawJSONDir)

	logger.Infof("Reporter exit gracefully")

	return nil
}
