package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/buger/jsonparser"

	"github.com/coderrect-inc/coderrect/pkg/reporter"
	"github.com/coderrect-inc/coderrect/pkg/util/conflib"
	"github.com/coderrect-inc/coderrect/pkg/util/logger"
	"github.com/coderrect-inc/coderrect/pkg/util/platform"
)

const (
	VERSION = "v0.0.1"

	baseDirName    = "basereport"
	gitLogFileName = "git-log"
)

var fileMode os.FileMode

// type aliases
type ExecutableInfo = reporter.ExecutableInfo
type IndexInfo = reporter.IndexInfo

func usage() {
	msg := `
usage: reporter [options] path_to_racedetect_json path_to_report_folder

Options:
  -h, -help                  Display this information.
  -v, -version               Display versions of all components.
`
	fmt.Printf("%s", msg)
}

/**
 * Checks if "path" is a file.
 */
func isFile(path string) (bool, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return false, err
	}

	return fi.Mode().IsRegular(), nil
}

func doesExist(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

func isDirectory(path string) bool {
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}

	return fi.IsDir()
}

func copyFile(src, dst string) error {
	var err error
	var srcfd *os.File
	var dstfd *os.File
	var srcinfo os.FileInfo

	if srcfd, err = os.Open(src); err != nil {
		return err
	}
	defer srcfd.Close()

	if dstfd, err = os.Create(dst); err != nil {
		return err
	}
	defer dstfd.Close()

	if _, err = io.Copy(dstfd, srcfd); err != nil {
		return err
	}
	if srcinfo, err = os.Stat(src); err != nil {
		return err
	}
	return os.Chmod(dst, srcinfo.Mode())
}

func copyDir(src, dst string) error {
	var err error
	var fds []os.FileInfo
	var srcinfo os.FileInfo

	if srcinfo, err = os.Stat(src); err != nil {
		return err
	}

	if err = os.MkdirAll(dst, srcinfo.Mode()); err != nil {
		return err
	}

	if fds, err = ioutil.ReadDir(src); err != nil {
		return err
	}
	for _, fd := range fds {
		srcfp := path.Join(src, fd.Name())
		dstfp := path.Join(dst, fd.Name())

		if fd.IsDir() {
			if err = copyDir(srcfp, dstfp); err != nil {
				return err
			}
		} else {
			if err = copyFile(srcfp, dstfp); err != nil {
				return err
			}
		}
	}
	return nil
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

/**
 * index.html wants to import "src" json as a javascript like
 *   <script src="race.json"></script>
 * because coderrect.js can't load race.json from the local disk using
 *   $.getJSON()
 * for security reason.
 *
 * But we can't include the plain json file directly because it is just
 * a JSON object without a name.
 *
 * So, createRaceJson create a file 'dst' by prepending "raceData = " to
 * "src". Instead of import the original "src", index.html imports "dst"
 * as a javascript file so that it defines a variable "raceData". Other
 * javascript code access data in "src" using this name.
 */
func createRaceJson(src, dst string) error {
	var err error
	var srcfd *os.File
	var dstfd *os.File
	var srcinfo os.FileInfo
	var nread int

	if srcfd, err = os.Open(src); err != nil {
		return err
	}
	defer srcfd.Close()

	if dstfd, err = os.Create(dst); err != nil {
		return err
	}
	defer dstfd.Close()

	if _, err = dstfd.Write([]byte("raceData = \n")); err != nil {
		return err
	}

	buf := make([]byte, 4096)
	if nread, err = srcfd.Read(buf); err != nil {
		return err
	}
	for nread > 0 {
		if _, err = dstfd.Write(buf[:nread]); err != nil {
			return err
		}

		if nread, err = srcfd.Read(buf); err != nil {
			if err == io.EOF {
				break
			}

			return err
		}
	}
	if _, err = dstfd.Write([]byte(";\n")); err != nil {
		return err
	}

	coderrectVer, _ := getPackageVersion()
	type Version struct {
		Version              string
		EnableMismatchedAPIs bool
	}
	version := Version{
		Version:              coderrectVer,
		EnableMismatchedAPIs: conflib.GetBool("enableMismatchedAPI", false),
	}
	versionBytes, err := json.Marshal(version)
	if err != nil {
		panic(err)
	}
	versionStr := "verData=" + string(versionBytes) + ";\n"
	if _, err = dstfd.Write([]byte(versionStr)); err != nil {
		panic(err)
	}

	projectDir := "projectDir=" + "\"" + conflib.GetString("cwd", "") + "\";\n\n"
	if _, err = dstfd.Write([]byte(projectDir)); err != nil {
		panic(err)
	}

	if srcinfo, err = os.Stat(src); err != nil {
		return err
	}
	return os.Chmod(dst, srcinfo.Mode())
}

func createRacesRealJson(src, dst string) error {
	var err error
	var srcfd *os.File
	var dstfd *os.File
	var srcinfo os.FileInfo
	var nread int

	if srcfd, err = os.Open(src); err != nil {
		return err
	}
	defer srcfd.Close()

	if dstfd, err = os.Create(dst); err != nil {
		return err
	}
	defer dstfd.Close()

	if _, err = dstfd.Write([]byte("{ \"raceData\" : \n")); err != nil {
		return err
	}

	buf := make([]byte, 4096)
	if nread, err = srcfd.Read(buf); err != nil {
		return err
	}
	for nread > 0 {
		if _, err = dstfd.Write(buf[:nread]); err != nil {
			return err
		}

		if nread, err = srcfd.Read(buf); err != nil {
			if err == io.EOF {
				break
			}

			return err
		}
	}
	if _, err = dstfd.Write([]byte(",\n")); err != nil {
		return err
	}

	coderrectVer, _ := getPackageVersion()
	type Version struct {
		Version              string
		EnableMismatchedAPIs bool
	}
	version := Version{
		Version:              coderrectVer,
		EnableMismatchedAPIs: conflib.GetBool("enableMismatchedAPI", false),
	}
	versionBytes, err := json.Marshal(version)
	if err != nil {
		panic(err)
	}
	versionStr := "\"verData\":" + string(versionBytes) + ",\n"
	if _, err = dstfd.Write([]byte(versionStr)); err != nil {
		panic(err)
	}

	projectDir := "\"projectDir\":" + "\"" + conflib.GetString("cwd", "") + "\",\n\n"
	if _, err = dstfd.Write([]byte(projectDir)); err != nil {
		panic(err)
	}

	if srcinfo, err = os.Stat(src); err != nil {
		return err
	}
	return os.Chmod(dst, srcinfo.Mode())
}

/**
 * Determine if a command (e.g. browse) exists using "which"
 */
func commandExists(cmd string) bool {
	_, err := platform.Command(fmt.Sprintf("which %s", cmd)).CombinedOutput()
	return err == nil
}

func showThread(access reporter.ThreadAccess) {
	fmt.Print(access.Snippet)
	fmt.Println(">>>Stacktrace:")
	spaces := ""
	for _, s := range access.Stacktrace {
		fmt.Printf(">>>%s%s\n", spaces, s)
		spaces = spaces + "  "
	}
}

func showTerminalReport(raceJsonPath string) {
	raceJsonBytes, err := ioutil.ReadFile(raceJsonPath)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to read the data file - %v\n", err))
		return
	}

	tmplDataRace := `
==== Found a data race between: 
line %d, column %d in %s AND line %d, column %d in %s
`
	var raceReport reporter.RaceReport
	if err = json.Unmarshal(raceJsonBytes, &raceReport); err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to parse the data file - %v\n", err))
		return
	}

	for _, race := range raceReport.DataRaces {
		so := race.SharedObj
		a1 := race.Access1
		a2 := race.Access2
		fmt.Printf(tmplDataRace, a1.Line, a1.Col, a1.Filename, a2.Line, a2.Col, a2.Filename)

		ErrorColor := "\033[1;31m%s\033[0m\n"
		fmt.Printf(ErrorColor, "Shared variable:")
		fmt.Printf(" at line %d of %s\n", so.Line, so.Filename)
		fmt.Print(so.SourceLine)

		WarningColor := "\033[1;33m%s\033[0m\n"
		fmt.Printf(WarningColor, "Thread 1:")
		showThread(a1)
		fmt.Printf(WarningColor, "Thread 2:")
		showThread(a2)
	}

	tmplAtomicityViolation := `
==== Found a data race between: 
line %d, column %d in %s AND line %d, column %d in %s
`
	for _, av := range raceReport.RaceConditions {
		so := av.SharedObj
		a1 := av.Access1
		a2 := av.Access2
		a3 := av.Access3
		fmt.Printf(tmplAtomicityViolation, a1.Line, a1.Col, a1.Filename,
			a2.Line, a2.Col, a2.Filename, a3.Line, a3.Col, a3.Filename)

		ErrorColor := "\033[1;31m%s\033[0m\n"
		fmt.Printf(ErrorColor, "Shared variable:")
		fmt.Printf(" at line %d of %s\n", so.Line, so.Filename)
		fmt.Print(so.SourceLine)

		WarningColor := "\033[1;33m%s\033[0m\n"
		fmt.Printf(WarningColor, "Thread 1 (load):")
		showThread(a1)
		fmt.Printf(WarningColor, "Thread 2 (store):")
		showThread(a2)
		fmt.Printf(WarningColor, "Thread 1 (store):")
		showThread(a3)
	}

	tmplTocTou := `
==== Found a TOCTOU between: 
line %d, column %d in %s AND line %d, column %d in %s
`

	for _, race := range raceReport.TocTous {
		so := race.SharedObj
		a1 := race.Access1
		a2 := race.Access2
		fmt.Printf(tmplTocTou, a1.Line, a1.Col, a1.Filename, a2.Line, a2.Col, a2.Filename)

		ErrorColor := "\033[1;31m%s\033[0m\n"
		fmt.Printf(ErrorColor, "Shared variable:")
		fmt.Printf(" at line %d of %s\n", so.Line, so.Filename)
		fmt.Print(so.SourceLine)

		WarningColor := "\033[1;33m%s\033[0m\n"
		fmt.Printf(WarningColor, "Thread 1:")
		showThread(a1)
		fmt.Printf(WarningColor, "Thread 2:")
		showThread(a2)
	}

	// todo show other high-level races

	fmt.Println("\033[0m")
}

// check if there is any race for given executable file
func raceExist(raceJsonPath string, reportDir string, executableName string) bool {
	numOfOMP := 0
	numOfNormalDataRaces := 0
	numOfDeadlocks := 0
	numOfMissingPairAPIs := 0
	numOfUntrustfulAccounts := 0
	numOfUnsafeOperations := 0
	numOfCosplayAccounts := 0
	numOfOrderViolations := 0
	numOfAtomicityViolations := 0
	total := 0

	type DataRace struct {
		IsOmpRace bool `json:"isOmpRace"`
	}

	raceJsonBytes, err := ioutil.ReadFile(raceJsonPath)
	if err != nil {
		return false
	}

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

	// for race conditions
	type Violation struct {
		IsAV bool `json:"isAV"`
	}

	if conflib.GetBool("reporter.showHighLevelRaces", false) {
		jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
			var violationRace Violation
			if json.Unmarshal(value, &violationRace) == nil {
				if violationRace.IsAV {
					numOfAtomicityViolations++
				} else {
					numOfOrderViolations++
				}
				total++
			}
		}, "raceConditions")

		// for deadlocks
		jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
			numOfDeadlocks++
			total++
		}, "deadLocks")

		// for mismatchedAPIs
		jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
			numOfMissingPairAPIs++
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

	}

	if total == 0 {
		return false
	}

	return true
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

func getBaseDirPath() string {
	return filepath.Join(os.Getenv("CODERRECT_CWD"), baseDirName)
}

func getWorkingDirPath() string {
	return filepath.Join(os.Getenv("CODERRECT_CWD"))
}

func getGitLogPath(parentPath string) string {
	return filepath.Join(parentPath, gitLogFileName)
}

func getBaseGitLog() string {
	content, _ := ioutil.ReadFile(getGitLogPath(getBaseDirPath()))
	return string(content)
}

// the first return value is a bool indicating whether any new bugs is detected
func diffRaceJSON(baseRawJSONPath string, currentRawJSONPath string, appendFilePath string, appendRealJsonFilePath string) (bool, error) {
	logger.Debugf("Diff file. base=%s, path=%s", baseRawJSONPath, currentRawJSONPath)
	baseParser := reporter.NewRaceJSONParser()
	currentParser := reporter.NewRaceJSONParser()
	baseError := baseParser.LoadRaceJSON(baseRawJSONPath)
	currentError := currentParser.LoadRaceJSON(currentRawJSONPath)

	comparedWithBase := true
	if baseError != nil {
		logger.Warnf("Open base file error. path=%s, err=%v", baseRawJSONPath, baseError)
		comparedWithBase = false
	}

	if currentError != nil {
		logger.Errorf("Open current file error. path=%s, err=%v", currentRawJSONPath, baseError)
		return false, currentError
	}

	currentParser.ComputeSignature()
	baseParser.ComputeSignature()
	var newDataRace []int = make([]int, 0)
	for idx, v := range currentParser.DataRaceSignatureList {
		//fmt.Println("newDataRace: ", v)
		_, ok := baseParser.DataRaceSignatureMap[v]
		if !ok {
			newDataRace = append(newDataRace, idx)
		}
	}

	var newCosplayAccount []int = make([]int, 0)
	for idx, v := range currentParser.CosplayAccountSignatureList {
		_, ok := baseParser.CosplayAccountSignatureMap[v]
		if !ok {
			newCosplayAccount = append(newCosplayAccount, idx)
		}
	}

	var newUnsafeOperation []int = make([]int, 0)
	for idx, v := range currentParser.UnsafeOperationSignatureList {
		_, ok := baseParser.UnsafeOperationSignatureMap[v]
		if !ok {
			newUnsafeOperation = append(newUnsafeOperation, idx)
		}
	}

	var newUntrustfulAccount []int = make([]int, 0)
	for idx, v := range currentParser.UntrustfulAccountSignatureList {
		_, ok := baseParser.UntrustfulAccountSignatureMap[v]
		if !ok {
			newUntrustfulAccount = append(newUntrustfulAccount, idx)
		}
	}

	foundNewBugs := false
	if len(newDataRace) != 0 || len(newCosplayAccount) != 0 || len(newUnsafeOperation) != 0 || len(newUntrustfulAccount) != 0 {
		foundNewBugs = true
	}

	var err error
	var fd *os.File
	if fd, err = os.OpenFile(appendFilePath, os.O_APPEND|os.O_WRONLY, 0644); err != nil {
		return foundNewBugs, err
	}
	defer fd.Close()

	var err2 error
	var fd2 *os.File
	if fd2, err2 = os.OpenFile(appendRealJsonFilePath, os.O_APPEND|os.O_WRONLY, 0644); err2 != nil {
		return foundNewBugs, err2
	}
	defer fd2.Close()

	if _, err = fd.Write([]byte("// --------------- below are new data --------------------------- //\n")); err != nil {
		return foundNewBugs, err
	}

	if _, err = fd.Write([]byte(fmt.Sprintf("comparedWithBase=%t\n", comparedWithBase))); err != nil {
		return foundNewBugs, err
	}
	if _, err2 = fd2.Write([]byte(fmt.Sprintf("\"comparedWithBase\":%t,\n", comparedWithBase))); err2 != nil {
		return foundNewBugs, err2
	}

	jsonBytes, err := json.Marshal(newDataRace)
	str := "newDataRaces = " + string(jsonBytes) + ";\n"
	if _, err = fd.Write([]byte(str)); err != nil {
		return foundNewBugs, err
	}

	str2 := "\"newDataRaces\" : " + string(jsonBytes) + ",\n"
	if _, err2 = fd2.Write([]byte(str2)); err2 != nil {
		return foundNewBugs, err2
	}

	jsonBytes, err = json.Marshal(newCosplayAccount)
	str = "newCosplayAccounts = " + string(jsonBytes) + ";\n"
	if _, err = fd.Write([]byte(str)); err != nil {
		return foundNewBugs, err
	}
	str2 = "\"newCosplayAccounts\" : " + string(jsonBytes) + ",\n"
	if _, err2 = fd2.Write([]byte(str2)); err2 != nil {
		return foundNewBugs, err2
	}

	jsonBytes, err = json.Marshal(newUnsafeOperation)
	str = "newUnsafeOperations = " + string(jsonBytes) + ";\n"
	if _, err = fd.Write([]byte(str)); err != nil {
		return foundNewBugs, err
	}
	str2 = "\"newUnsafeOperations\": " + string(jsonBytes) + ",\n"
	if _, err2 = fd2.Write([]byte(str2)); err2 != nil {
		return foundNewBugs, err2
	}

	jsonBytes, err = json.Marshal(newUntrustfulAccount)
	str = "newUntrustfulAccounts = " + string(jsonBytes) + ";\n"
	if _, err = fd.Write([]byte(str)); err != nil {
		return foundNewBugs, err
	}
	str2 = "\"newUntrustfulAccounts\" :" + string(jsonBytes) + ",\n"
	if _, err2 = fd2.Write([]byte(str2)); err2 != nil {
		return foundNewBugs, err2
	}

	str = "baseGitCommitLog = \"" + getBaseGitLog() + "\";\n"
	if _, err = fd.Write([]byte(str)); err != nil {
		return foundNewBugs, err
	}
	str2 = "\"baseGitCommitLog\" :\"" + getBaseGitLog() + "\",\n"
	if _, err2 = fd2.Write([]byte(str2)); err2 != nil {
		return foundNewBugs, err2
	}

	str = "currGitCommitLog  = \"" + getGitLog() + "\";\n"
	if _, err = fd.Write([]byte(str)); err != nil {
		return foundNewBugs, err
	}
	str2 = "\"currGitCommitLog\"  : \"" + getGitLog() + "\",\n"
	if _, err2 = fd2.Write([]byte(str2)); err2 != nil {
		return foundNewBugs, err2
	}

	str = "currentGitUrl = \"" + getGitFullPath() + "\";"
	if _, err = fd.Write([]byte(str)); err != nil {
		return foundNewBugs, err
	}
	str2 = "\"currentGitUrl\" : \"" + getGitFullPath() + "\"}"
	if _, err2 = fd2.Write([]byte(str2)); err2 != nil {
		return foundNewBugs, err2
	}
	return foundNewBugs, nil
}

// Get the full name and commit hash of the git repo,
// so that we can access its files from HTTP request
func getGitFullPath() string {
	// First check if this is a git repo
	cmd := platform.Command("git", "rev-parse", "HEAD")
	output, err := cmd.Output()
	// if not, we use time stamp as commit info
	if err != nil || len(output) == 0 { // SHA could be empty if it's a new repo
		logger.Warnf("Not a git repository, fail to get Git log")
		return ""
	}
	commitHash := strings.TrimSpace(string(output))
	// using command "git --no-pager log -1 --oneline --pretty='%h %s'"
	cmd = platform.Command("git", "config", "--get", "remote.origin.url")
	cmdExeOutput, err := cmd.Output()
	if err != nil {
		logger.Warnf("Fail to get Git log. err=%v", err)
		return ""
	}
	gitUrl := string(cmdExeOutput)
	if !strings.HasSuffix(gitUrl, ".git") {
		// In rare cases, the url may not include .git, such as LocustDB repo
		gitUrl += ".git"
	}
	re := regexp.MustCompile(`https://github\.com/(?P<user>.*)/(?P<repo>.*)\.git`)
	match := re.FindStringSubmatch(gitUrl)
	if len(match) != 3 || re.SubexpNames()[1] != "user" || re.SubexpNames()[2] != "repo" {
		logger.Warnf("Not a valid git URL: %s", gitUrl)
		return ""
	}
	gitUser := match[1]
	gitRepo := match[2]
	fullUrl := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s/", gitUser, gitRepo, commitHash)
	return fullUrl
}

func getGitLog() string {
	// First check if this is a git repo
	cmd := platform.Command("git", "rev-parse", "--git-dir")
	_, err := cmd.Output()
	// if not, we use time stamp as commit info
	if err != nil {
		logger.Warnf("Not a git repository, fail to get Git log")
		return "Scan at -- " + time.Now().Format(time.UnixDate)
	}

	// using command "git --no-pager log -1 --oneline --pretty='%h %s'"
	cmd = platform.Command("git", "--no-pager", "log", "-1", "--oneline", "--pretty='%h %s'")
	// cmd := platform.Command(cmdline)
	// cmd.Dir = os.Getenv("CODERRECT_CWD")
	cmdExeOutput, err := cmd.Output()
	if err != nil {
		logger.Warnf("Fail to get Git log. err=%v", err)
		//panic(err)
		//do not panic: it is possible to get empty git log for a project just after git init
		return ""
	}
	log := string(cmdExeOutput)
	log = strings.TrimSpace(log)
	log = strings.Trim(log, "\"")
	log = strings.TrimSpace(log)
	log = strings.ReplaceAll(log, "\"", "\\\"")
	// fmt.Println("==========GIT LOG=========")
	// fmt.Println(log)
	return log
}

// clean old base, create new base
func buildBase(execInfo []ExecutableInfo) (isRebuilt bool, err error) {
	var tmpBaseDirPath string
	tmpBaseDirPath, err = ioutil.TempDir(getWorkingDirPath(), "report-tmp")
	if err != nil {
		logger.Errorf("Create temporary base directory error=%v", err)
		panic(err)
	}
	defer os.RemoveAll(tmpBaseDirPath)

	isRebuilt = true
	baseDirPath := getBaseDirPath()
	logger.Infof("read report.keepBase. value=%v", conflib.GetBool("report.keepBase", false))
	// if specified as `keepBase` and the base directory exist, then direct return
	if conflib.GetBool("report.keepBase", false) {
		if doesExist(baseDirPath) {
			isRebuilt = false
			logger.Debugf("Base directory found")
			return
		}
		// first create
		logger.Debugf("Base directory not found")
	}

	logger.Debugf("Building base report, src=%s, base=%s", tmpBaseDirPath, baseDirPath)
	for i := 0; i < len(execInfo); i++ {
		srcPath := execInfo[i].RaceJSON
		dstPath := filepath.Join(tmpBaseDirPath, filepath.Base(srcPath))
		logger.Debugf("Copying to base from file: %s", srcPath)
		if err = copyFile(srcPath, dstPath); err != nil {
			logger.Errorf("Fail to copy raw json to base. sourceDir=%s, destDir=%s, err=%v",
				srcPath, dstPath, err)
			panic(err)
		}
	}

	logger.Debugf("Get and set GIT log to file")
	cmdExecOutput := getGitLog()
	err = ioutil.WriteFile(getGitLogPath(tmpBaseDirPath), ([]byte)(cmdExecOutput), 0644)
	if err != nil {
		logger.Errorf("Fail to write GIT log. err=%v", err)
	}

	// clean old base, rename temp dir to base
	if isRebuilt {
		if err = os.RemoveAll(baseDirPath); err != nil {
			logger.Errorf("Fail to clean old base directory err=%v", err)
			return
		}
	}

	// FIXME: sometimes rename will cause "invalid cross-device link"
	if err = os.Rename(tmpBaseDirPath, baseDirPath); err != nil {
		logger.Errorf("Fail to move temp to base, tmp=%s, base=%s, err=%v", tmpBaseDirPath, baseDirPath, err)
		return
	}
	return
}

type ReportInfo struct {
	Name string
	Num  int
}

type indexMenuBuildParam struct {
	ReportList []ReportInfo
	NoRace     []string
	Version    string
}

func (idx *indexMenuBuildParam) Len() int {
	return len(idx.ReportList)
}

func (idx *indexMenuBuildParam) Swap(i, j int) {
	idx.ReportList[i], idx.ReportList[j] = idx.ReportList[j], idx.ReportList[i]
}

func (idx *indexMenuBuildParam) Less(i, j int) bool {
	return idx.ReportList[i].Num > idx.ReportList[j].Num
}

// createIndexMenu Implememnt menu in index.html using go template
func createIndexMenu(menuBuildParam *indexMenuBuildParam, reportDir string) {
	if !isCreateHTMLReport(reportDir) {
		return
	}

	t, err := template.ParseFiles(filepath.Join(reportDir, "index.html"))
	if err != nil {
		logger.Errorf("Fail to using template to parse index.html. err=%v", err)
	}
	f, err := os.Create(filepath.Join(reportDir, "index.html"))
	if err != nil {
		logger.Errorf("Fail to create index.html. err=%v", err)
		os.Exit(1)
	}

	sort.Sort(menuBuildParam)
	if err = t.Execute(f, menuBuildParam); err != nil {
		logger.Errorf("Fail to execute template data into index.html. err=%v", err)
		os.Exit(1)
	}
	f.Close()
}

// showHelpInfo helpful information to tell user how to view generated report
func showHelpInfo(reportDir string) {
	if isCreateHTMLReport(reportDir) {
		//show helpful information to tell user how to view generated report
		fmt.Println("")
		if commandExists("browse") {
			fmt.Printf("To check the full report in your browser, run \"browse %s/index.html\"\n",
				reportDir)
		} else if commandExists("xdg-open") {
			fmt.Printf("To check the full report in your browser, run \"xdg-open %s/index.html\"\n",
				reportDir)
		} else {
			if runtime.GOOS == "windows" {
				fmt.Printf("To check the full report, open '%s\\index.html' in your browser\n",
					reportDir)
			} else {
				fmt.Printf("To check the full report, open '%s/index.html' in your browser\n",
					reportDir)
			}
		}
	}

	if false && !conflib.GetBool("report.enableTerminal", false) {
		fmt.Println("")
		fmt.Println("If you want coderrect to show results in the terminal, please add '-t' to the command line like:")
		fmt.Println("")
		fmt.Println("\tcoderrect -t make")
	}
	if false {
		fmt.Println("\nAny feedback? please send them to feedback@coderrect.com, thank you!")
	}
}

func isCreateHTMLReport(reportDir string) bool {
	return len(reportDir) > 0 && reportDir != "/dev/null"
}

/**
 * Reporter reads index.json, configuration.json, and a list of raw json file under /path/to/rawJsonPath
 * and generate report to /path/to/reportDir
 *
 * $ reporter /path/to/reportDir /path/to/rawJsonPath
 * $ reporter /path/to/reportDir /path/to/race.json
 *
 * The report is a directory containing multiple files such as css/js/html and images.
 */
func main() {
	// initialize the configurations
	//
	argAlias := map[string]string{
		"v": "version",
		"h": "help",
	}
	_, err := conflib.InitializeWithEnvCmdline(argAlias)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// show help info
	//
	if conflib.GetBool("help", false) {
		usage()
		return
	}

	// show version info
	//
	if conflib.GetBool("version", false) {
		fmt.Println(VERSION)
		return
	}

	args := os.Args[1:]
	if len(args) != 2 {
		fmt.Printf("Missing arguments\n")
		usage()
		os.Exit(2)
	}

	reportDir, _ := filepath.Abs(args[0])
	rawJSONDir, _ := filepath.Abs(args[1])
	raceJSONPath := rawJSONDir

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
		res := reporter.ReadIndexJSON(&indexInfo, rawJSONDir)
		if res == 0 {
			logger.Errorf("Fail to find index.json. rawJSONDir=%s", rawJSONDir)
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

	// use go template to generate index.html
	//
	coderrectHome, err := conflib.GetCoderrectHome()
	if err != nil {
		logger.Errorf("Fail to obtain CODERRECT_HOME. err=%v", err)
		os.Exit(1)
	}

	isCreateHTMLReport := isCreateHTMLReport(reportDir)
	if isCreateHTMLReport {
		// we use the perm mode of the current working directory
		// for all new created files
		workingDir, _ := os.Getwd()
		fi, _ := os.Stat(workingDir)
		fileMode = fi.Mode()

		if doesExist(reportDir) {
			logger.Debugf("The report folder already existed, clean it. reportDir=%s", reportDir)
			if errRemove := os.RemoveAll(reportDir); errRemove != nil {
				panic(errRemove)
			}
		}
		if err = os.MkdirAll(reportDir, fileMode); err != nil {
			logger.Errorf("Fail to create the report folder. reportDir=%s, err=%v",
				reportDir, err)
			os.Exit(1)
		}

		// Copy /artfacts from $CODERREETC_HOME/data/reporter to reportDir
		artifactsPath := filepath.Join(coderrectHome, "data", "reporter", "artifacts")
		if err = copyDir(artifactsPath, filepath.Join(reportDir, "artifacts")); err != nil {
			logger.Errorf("Fail to copy /artifacts to reportDir. sourceDir=%s, destDir=%s, err=%v",
				artifactsPath, filepath.Join(reportDir, "artifacts"), err)
			os.Exit(1)
		}

		// Copy index.html from $CODERREETC_HOME/data/reporter to reportDir
		indexHtmlPath := filepath.Join(coderrectHome, "data", "reporter", "index.html")
		if err = copyFile(indexHtmlPath, filepath.Join(reportDir, "index.html")); err != nil {
			logger.Errorf("Fail to copy index.html to reportDir. sourceDir=%s. destDir=%s, err=%v",
				indexHtmlPath, filepath.Join(reportDir, "index.html"), err)
			os.Exit(1)
		}

		// Copy configuration.json from $CODERREETC_HOME/data/reporter to reportDir
		configurationPath := filepath.Join(rawJSONDir, "configuration.json")
		if err = copyFile(configurationPath, filepath.Join(reportDir, "configuration.json")); err != nil {
			logger.Warnf("Fail to copy configuration.json to reportDir. sourceDir=%s, destDir=%s, err=%v",
				configurationPath, filepath.Join(reportDir, "configuration.json"), err)
		}
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
		// Use filters defined in config file to rewrite the raw race JSON
		rawRaceJSONPath := executable.RaceJSON
		filterDef := conflib.GetStrings("report.filters")
		num := reporter.Rewrite(rawRaceJSONPath, filterDef)
		if num == 0 {
			executablesWithNoRace = append(executablesWithNoRace, executable.Name)
			continue
		}
		if num == -1 {
			num = executable.DataRaces + executable.RaceConditions
		}
		executableList = append(executableList, ReportInfo{executable.Name, num})
	}

	var foundError bool
	// --- Generate report for each executable
	// For each executable with races, we create a report
	for i := 0; i < len(indexInfo.Executables); i++ {
		executable := indexInfo.Executables[i]
		rawRaceJSONPath := executable.RaceJSON

		// Create /$executable directory for storing executable.html and race.json
		if isCreateHTMLReport {
			executbalePath := filepath.Join(reportDir, executable.Name)
			if err = os.MkdirAll(executbalePath, fileMode); err != nil {
				logger.Errorf("Fail to create executbale path. pathName=%s, err=%s",
					executbalePath, err)
				os.Exit(1)
			}

			// Copy report.html from $CODERREETC_HOME/data/reporter to reportDir
			// and rename it as $executable.html
			reportHTML := filepath.Join(coderrectHome, "data", "reporter", "report.html")
			executableHTML := executable.Name + ".html"
			if err = copyFile(reportHTML, filepath.Join(reportDir, executable.Name, executableHTML)); err != nil {
				logger.Errorf("Fail to copy report.html to reportDir. sourceDir=%s, destDir=%s, err=%v",
					reportHTML, filepath.Join(reportDir, executable.Name, executableHTML), err)
				os.Exit(1)
			}
		}

		// Copy executable json file from rawJsonPath to reportDir
		if isCreateHTMLReport {
			raceJSONPath := filepath.Join(reportDir, executable.Name, "race.json")
			if err = createRaceJson(rawRaceJSONPath, raceJSONPath); err != nil {
				logger.Errorf("Fail to create race data json, json=%s, err=%v", raceJSONPath, err)
				os.Exit(1)
			}

			raceRealJSONPath := filepath.Join(reportDir, executable.Name, "races.json")
			if err = createRacesRealJson(rawRaceJSONPath, raceRealJSONPath); err != nil {
				logger.Errorf("Fail to create races json, json=%s, err=%v", raceJSONPath, err)
				os.Exit(1)
			}

			// --- Generate incremental view by diffing the base report with the current report
			baseDirPath := getBaseDirPath()
			baseRawJSONPath := filepath.Join(baseDirPath, filepath.Base(executable.RaceJSON))
			foundNewBugs, err := diffRaceJSON(baseRawJSONPath, rawRaceJSONPath, raceJSONPath, raceRealJSONPath)
			if err != nil {
				logger.Errorf("Fail to diff base vs current race json, json=%s, err=%v", raceJSONPath, err)
				os.Exit(1)
			}

			indexInfo.Executables[i].NewBugs = foundNewBugs
		}

		fmt.Printf("\n--------The summary of potential vulnerabilities in %s--------\n", executable.Name)
		// show the summary of analysis
		foundError = showSummary(rawRaceJSONPath)
		fmt.Print("\n\n")
	} // for each executable

	// update index.json with the information "if any new race was found"
	reporter.WriteIndexJSON(indexInfo, rawJSONDir)

	// --- Save base report for incremental view
	if _, err = buildBase(indexInfo.Executables); err != nil {
		logger.Errorf("Fail to build base report")
	}

	if len(executableList) > 0 {
		buildParam := indexMenuBuildParam{executableList, executablesWithNoRace, indexInfo.CoderrectVer}
		createIndexMenu(&buildParam, reportDir)
		// NOTE: for plugin, it's easier to just use ONE command to finish detection + uploading
		ext := conflib.GetBool("publish.ext", false)
		if ext {
			reporter.PublishReport(reportDir, "DEFAULT")
		} else {
			// help info doesn't make too much sense in plugin context
			showHelpInfo(reportDir)
		}
	} else {
		//fmt.Println("No vulnerabilities detected")
	}

	logger.Infof("Reporter exit gracefully")

	if foundError {
		os.Exit(9)
	} else {
		os.Exit(0)
	}
}
