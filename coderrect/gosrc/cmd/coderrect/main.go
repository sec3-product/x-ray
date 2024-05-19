package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"io/ioutil"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"syscall"

	"github.com/coderrect-inc/coderrect/gllvm/shared"
	"github.com/coderrect-inc/coderrect/reporter"
	"github.com/coderrect-inc/coderrect/util"
	"github.com/coderrect-inc/coderrect/util/platform"
	"github.com/gookit/color"

	//	"log"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/coderrect-inc/coderrect/util/conflib"
	"github.com/coderrect-inc/coderrect/util/jsonparser"
	"github.com/coderrect-inc/coderrect/util/logger"
	toml "github.com/pelletier/go-toml"
	bolt "go.etcd.io/bbolt"
)

const (
	colorReset = "\033[0m"
)

var usage = `
usage: soteria [option]... <your build command line>

Options:
  -h, -help              
		Display this information.

  -v, -version           
		Display versions of all components.

  -showconf              
		Display effective value of all configurations.

  -o <report_dir>, -report.outputDir=<report_dir>        
		Specify the report folder.

  -e <executable1,executable2,...>, -analyzeBinaries=<executable1,executable2,...>
		Specify a list of executable names you want to analyze.

  -analyzeAll
        Let soteria analyze all qualified projects under a given path.

  -t, -report.enableTerminal
		Generate the terminal-based report.

  -c, -cleanBuild
		Delete all intermediate files. 

  -plan=free|build|scale
		Specify the plan. "free" analyzes a subset of vulnerabilities; 
		"scale" aims to all vulnerabilities; and the default plan "build"
		tries to detect all vulnerabilities while balancing speed and accuracy.

  -mode=fast|normal|exhaust
        Specify the detection mode. "fast" prefers analyzing speed; 
        "exhaust" prefers the accuracy; and the default mode "normal"
        tries to achieve a balance between speed and accuracy.

  -continueIfBuildErrorPresents
        By default, soteria terminates immediately when build encounters
        errors. This flag lets soteria continue the vulnerability detection.

Examples:

1. Detect vulnerabilities in helloworld
$ cd path/to/examples/helloworld && soteria .

2. Detect vulnerabilities in all solana-program-library projects
$ soteria -analyzeAll path/to/solana-program-library
`

// -solPaths=path1,path2
// Specify a list of comma separated directories where soteria searchs
// for projects containing Xargo.toml files.

var multipleExecutableMsg = `
The project creates multiple executables. Please select one from the list below
to detect races. 

In the future, you can specify the executable using the option "-e" if you know 
which ones you want to analyze. 

    coderrect -e executable_name1,executable_name2 your_build_command_line
`

var openAPIMsg = `
To automate the API analysis, please read the tutorial https://coderrect.com/analyzing-static-dynamic-library-code/.	
`

// type aliases
type ExecutableInfo = reporter.ExecutableInfo
type IndexInfo = reporter.IndexInfo

/**
 * Checks if "path" is a file.
 */
func isFile(path string) bool {
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}

	return fi.Mode().IsRegular()
}

func isDir(path string) bool {
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}

	return fi.Mode().IsDir()
}

func isSourceFile(sourceFile string) bool {
	lcSourceFile := strings.ToLower(sourceFile)
	sourceCodeSuffix := [...]string{
		".c", ".cpp", ".cc", ".cxx", ".c++", ".cu",
		".h", ".hxx", ".hh", ".hpp",
		".swift", ".m",
		".f", ".f90", ".f03", ".f68", ".f77", ".f95",
	}

	for _, s := range sourceCodeSuffix {
		if strings.HasSuffix(lcSourceFile, s) {
			return true
		}
	}

	return false
}

func normalizeCmdlineArg(arg string) string {
	tokens := strings.SplitN(arg, "=", 2)
	if len(tokens) == 1 {
		return arg
	}

	if strings.Contains(tokens[1], "\"") {
		return fmt.Sprintf("%s='%s'", tokens[0], tokens[1])
	} else {
		if strings.Contains(tokens[1], "'") || strings.Contains(tokens[1], " ") {
			return fmt.Sprintf("%s=\"%s\"", tokens[0], tokens[1])
		} else {
			return arg
		}
	}
}

/**
 * we call the custom build command to build project rather than CLANG for the single-file
 * project and 'make' for a makefile-based project.
 *
 * the custom command line is stored in args. args[0] is the command, and args[1:] are
 * command-line arguments
 */
func doCustomBuild(args []string, isCargoBuild bool) (*exec.Cmd, string) {
	coderrectHome := os.Getenv("CODERRECT_HOME")
	if runtime.GOOS == "windows" {
		exePath := filepath.Join(coderrectHome, "bin", "libear-win.exe")
		libPath := filepath.Join(coderrectHome, "bin", "libear64.dll")
		libOption := fmt.Sprintf("-d:%s ", libPath)

		cmdline := fmt.Sprintf("%s %s ", exePath, libOption)
		for _, s := range args {
			cmdline = cmdline + normalizeCmdlineArg(s) + " "
		}

		data := append([]string{libOption}, args...)
		cmd := exec.Command(exePath, data...)
		return cmd, cmdline
	}

	// Set these ENV variables not matter if it's a cargo build
	// This is to handle cargo command called inside custom script
	//-Zsymbol-mangling-version=v0
	os.Setenv("RUSTFLAGS", "--emit=llvm-ir -g -C opt-level=0")
	os.Setenv("CARGO_PROFILE_DEV_LTO", "true")
	os.Setenv("CARGO_TARGET_DIR", os.Getenv("CODERRECT_BUILD_DIR"))
	// For Rust build, we do not need those libear stuff
	// if isCargoBuild {
	// 	exePath := filepath.Join(coderrectHome, "bin", "coderrect-exec")
	// 	cmdLine := exePath + " "
	// 	for _, s := range args {
	// 		cmdLine = cmdLine + normalizeCmdlineArg(s) + " "
	// 	}
	// 	cmd := platform.Command(cmdLine)
	// 	return cmd, cmdLine
	// }

	exePath := filepath.Join(coderrectHome, "bin", "coderrect-exec")
	libPath := filepath.Join(coderrectHome, "bin", "libear.so")
	cmdline := fmt.Sprintf("export LD_PRELOAD=%s; %s ", libPath, exePath)
	for _, s := range args {
		cmdline = cmdline + normalizeCmdlineArg(s) + " "
	}

	// support multiple command in oneline,
	// coderrect "./autogen.sh;./configure;make"
	//
	// sub process, env var inherited from parent, but new env var not affected parent
	// https://stackoverflow.com/questions/14024798/setting-environment-variables-for-multiple-commands-in-bash-one-liner
	cmdline = "(" + cmdline + ")"
	cmd := platform.Command(cmdline)
	return cmd, cmdline
}

func sortFoundExecutables(executables []string) []string {
	sl := []string{}
	testsl := []string{}

	var m map[string]int
	m = make(map[string]int)

	i := len(executables) - 1
	for i >= 0 {
		// _, filename := path.Split(executables[i])
		// tmp := strings.ToLower(filename)
		tmp := strings.ToLower(executables[i])
		if _, ok := m[tmp]; ok {
			i--
			continue
		}
		m[tmp] = 1
		if strings.Contains(tmp, "test") || strings.Contains(tmp, "util") || strings.HasSuffix(tmp, "a.out") || strings.Contains(tmp, "/lib") {
			// if strings.HasSuffix(tmp, "test") || strings.HasPrefix(tmp, "test") {
			testsl = append(testsl, executables[i])
		} else {
			sl = append(sl, executables[i])
		}
		i--
	}
	logger.Infof("Generate the executable list. sl=%v, testsl=%v", sl, testsl)

	maxCount := int(conflib.GetInt("maximumExecutablesCount", 99)) //default 99

	if len(sl) < maxCount {
		diff := maxCount - len(sl)
		i = 0
		for diff > 0 && i < len(testsl) {
			sl = append(sl, testsl[i])
			diff--
			i++
		}
	} else {
		// we pickup executable first, then .so file; ".a" files have lowest priorities.
		sort.Slice(sl, func(i, j int) bool {
			_, filename1 := filepath.Split(sl[i])
			_, filename2 := filepath.Split(sl[j])

			if strings.HasSuffix(filename1, ".a") {
				if strings.HasSuffix(filename2, ".a") {
					return filename1 <= filename2
				} else {
					return false
				}
			} else if strings.HasSuffix(filename1, ".so") {
				if strings.HasSuffix(filename2, ".a") {
					return true
				} else if strings.HasSuffix(filename2, ".so") {
					return filename1 <= filename2
				} else {
					return false
				}
			} else {
				if strings.HasSuffix(filename2, ".a") || strings.HasSuffix(filename2, ".so") {
					return true
				} else {
					return filename1 <= filename2
				}
			}
		})
		sl = sl[0:maxCount]
	}

	return sl
}

func getRelativePath(cwd string, myFilepath string) string {
	cwdTokens := strings.Split(cwd, string(filepath.Separator))
	filepathTokens := strings.Split(myFilepath, string(filepath.Separator))

	cwdTokensIdx := 0
	filepathTokensIdx := 0
	for cwdTokensIdx < len(cwdTokens) {
		if cwdTokens[cwdTokensIdx] == filepathTokens[filepathTokensIdx] {
			cwdTokensIdx++
			filepathTokensIdx++
		} else {
			break
		}
	}

	remainingFilepath := strings.Join(filepathTokens[filepathTokensIdx:], string(filepath.Separator))
	if cwdTokensIdx == len(cwdTokens) {
		return remainingFilepath
	} else {
		parentPath := ""
		for cwdTokensIdx < len(cwdTokens) {
			parentPath += (".." + string(filepath.Separator))
			cwdTokensIdx++
		}
		return parentPath + remainingFilepath
	}
}

//
// binaryPath is the absolute or relative path of a binary like "/home/jsong/project/libmath.a". binaryNameList
// is a list of single-stage names such as ["libmath.a", "memcached-server", "memcached-benchmark"].
//
// if the filename of binaryPath belongs to binaryNameList (case-insensitive), returns true.
//
func belongsToList(binaryPath string, binaryNameList []string) bool {
	_, filename := filepath.Split(binaryPath)
	for i := 0; i < len(binaryNameList); i++ {
		if strings.ToLower(filename) == strings.ToLower(binaryNameList[i]) {
			return true
		}
	}
	return false
}

func getAllExecFromDB() []string {
	logger.Debugf("Find all the executable file path from bolt db.")

	var execPathList []string
	boltDbPath := os.Getenv("CODERRECT_BOLD_DB_PATH")
	boltDb, err := bolt.Open(boltDbPath, 0666, nil)
	if err != nil {
		logger.Warnf("Failed to open bolt db. boltDbPath=%s, err=%v", boltDbPath, err)
		return execPathList
	}

	boltDb.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(shared.BucketNameExecFile))
		c := b.Cursor()

		for k, _ := c.First(); k != nil; k, _ = c.Next() {
			execPathList = append(execPathList, string(k))
		}
		return nil
	})
	boltDb.Close()

	return execPathList
}

func getAllExecFromCargoBuild(buildDir string) []string {
	logger.Debugf("Find all the executable file path from cargo build.")

	var execPathList []string
	// binDir := filepath.Join(buildDir, "debug")
	// UPDATE: exetuables generated for tests won't be under debug library,
	// so we have directly go to deps.
	//
	binDir := filepath.Join(buildDir, "bpfel-unknown-unknown", "release", "deps")
	files, err := ioutil.ReadDir(binDir)
	if err != nil {
		logger.Warnf("Failed to open build directory. binDir=%s, err=%v", binDir, err)
		// return execPathList
	}
	var bcFiles []string
	for _, file := range files {
		// if file.Mode().IsRegular() &&
		// 	!strings.HasPrefix(file.Name(), ".") &&
		// 	!strings.HasSuffix(file.Name(), ".d") &&
		// 	!strings.HasSuffix(file.Name(), ".rlib") {
		// 	execPathList = append(execPathList, filepath.Join(binDir, file.Name()))
		// }
		if strings.HasSuffix(file.Name(), ".ll") {
			fullName := filepath.Join(binDir, file.Name())
			bcFiles = append(bcFiles, fullName)
			execPathList = append(execPathList, fullName)
			//fmt.Println("DEBUG adding ll file: %", fullName)
		}
	}
	if conflib.GetBool("analyzeAll", false) {
		releaseDir := filepath.Join(buildDir, "bpfel-unknown-unknown", "release")
		outputFile := filepath.Join(releaseDir, "all.ll")
		var linkArgs []string
		linkArgs = append(linkArgs, "-o", outputFile)
		linkArgs = append(linkArgs, bcFiles...)

		success := shared.LinkSolanaBCFiles(linkArgs)
		if !success {
			return execPathList
		}
		var execPathList0 []string
		execPathList0 = append(execPathList0, outputFile)
		return execPathList0
	}

	exampleDir := filepath.Join(buildDir, "bpfel-unknown-unknown", "release")
	files, err = ioutil.ReadDir(exampleDir)
	if err != nil {
		logger.Warnf("Failed to open build directory. exampleDir=%s, err=%v", exampleDir, err)
		//return execPathList
	}
	// Binary in Rust has versions alias
	// e.g., crossbeam_tail_index and crossbeam_tail_index-ce4dd1fbc9d180b5
	// the latter has corresponding crossbeam_tail_index-ce4dd1fbc9d180b5.ll
	// we use that heuristics to pick only the versioned binary
	cacheNames := make(map[string]struct{})
	var exampleBins []string
	for _, file := range files {
		if file.Mode().IsRegular() &&
			!strings.HasPrefix(file.Name(), ".") &&
			!strings.HasSuffix(file.Name(), ".bc") &&
			!strings.HasSuffix(file.Name(), ".d") &&
			!strings.HasSuffix(file.Name(), ".o") {
			fullName := filepath.Join(exampleDir, file.Name())
			if strings.HasSuffix(file.Name(), ".ll") {
				cacheNames[fullName] = struct{}{}
			} else {
				exampleBins = append(exampleBins, fullName)

			}
		}
	}
	re := regexp.MustCompile("(?P<exe>.*)-[0-9a-z]+$")
	for _, exampleBin := range exampleBins {
		correspondingIRName := exampleBin + ".ll"
		_, ok := cacheNames[correspondingIRName]
		if ok {
			match := re.FindStringSubmatch(exampleBin)
			if len(match) != 2 || re.SubexpNames()[1] != "exe" {
				logger.Warnf("Failed to find unversioned example binary for: %s", exampleBin)
				continue
			}
			exeName := match[1]
			execPathList = append(execPathList, exeName)
		}
	}

	return execPathList
}

func searchFile(targetDir string, g *regexp.Regexp) (bool, string) {
	files, err := ioutil.ReadDir(targetDir)
	if err != nil {
		logger.Warnf("Failed to open build directory. targetDir=%s, err=%v", targetDir, err)
		return false, ""
	}
	for _, file := range files {
		if file.Mode().IsRegular() && strings.HasSuffix(file.Name(), ".ll") {
			if g.MatchString(file.Name()) {
				return true, filepath.Join(targetDir, file.Name())
			}
		}
	}
	return false, ""
}

func getCargoTargetBcFile(buildDir string, exeName string) string {
	_, exeName = filepath.Split(exeName)
	// Rust will replace - to _ for the name of the IR file
	// ignore the last - since it's used for the version hash
	cnt := strings.Count(exeName, "-")
	if cnt > 1 {
		exeName = strings.Replace(exeName, "-", "_", cnt-1)
	}
	binPattern := fmt.Sprintf("^%s(-[0-9a-z]+)?.ll$", exeName)
	// examplePattern := fmt.Sprintf("%s/examples/%s.ll", buildDir, exeName)
	g := regexp.MustCompile(binPattern)
	matched, llPath := searchFile(filepath.Join(buildDir, "bpfel-unknown-unknown", "release", "deps"), g)
	if matched {
		return llPath
	}
	matched, llPath = searchFile(filepath.Join(buildDir, "bpfel-unknown-unknown", "release"), g)
	if matched {
		return llPath
	}
	logger.Warnf("Failed to find IR file for binary: %s in directory %s", exeName, buildDir)
	return ""
}

/**
 * Returns a list of executable/library generated by a project and a boolean vairable
 * that determines if user input contains executable name that does not exist, return
 * true, if there is input executables that don't exist, otherwise return false.
 *
 * If the user specifies the configuration of "executable" that is a comma-separated list,
 * getExecutablePathList() returns the list of executables whose name belong to "executable" list
 * (note that we compare filename case-insensitively).
 *
 * If the project generates only one executable/library this function return it immediately.
 *
 * Otherwise, getExecutablePathList() asks the user to pick up several executables interactively.
 *
 * Only the binary whose names statisfy the following criteria will be considered
 *   1. the name isn't end with ".o"
 *   2. the file is executable
 *   3. the file isn't a hidden file
 *
 */
func getExecutablePathList(token string, buildDir string, isCargoBuild bool) ([]string, bool) {
	var execPathList []string
	// FIXME: For custom build that contains cargo, it may contain both C++ targets and rust targets
	if isCargoBuild {
		execPathList = getAllExecFromCargoBuild(buildDir)
	} else {
		execPathList = getAllExecFromDB()
	}

	inputExecutableNames := conflib.GetString("analyzeBinaries", "")
	inputExecutableNameList := strings.Split(inputExecutableNames, ",")
	var foundExecutableList []string
	var inputFoundExecutaleList []string
	for i := 0; i < len(execPathList); i++ {
		exePath := execPathList[i]
		if strings.HasSuffix(exePath, ".ll") { //for ll file directly
			//foundExecutableList = append(foundExecutableList, exePath)
		} else {
			if strings.HasSuffix(exePath, ".o") ||
				strings.HasSuffix(exePath, "conftest") ||
				strings.HasSuffix(exePath, "libgtest_main.a") ||
				strings.HasSuffix(exePath, "libgmock_main.a") ||
				strings.HasSuffix(exePath, "libgtest.a") ||
				strings.HasSuffix(exePath, "libgmock.a") ||
				(!platform.IsExecutable(exePath) && !platform.IsLibrary(exePath)) {
				continue
			}
		}
		// check if user has specified exe to analyze
		if len(inputExecutableNames) > 0 && belongsToList(exePath, inputExecutableNameList) {
			inputFoundExecutaleList = append(inputFoundExecutaleList, exePath)
		} else {
			_, filename := path.Split(exePath)
			// ignore all hidden files
			if !strings.HasPrefix(filename, ".") {
				foundExecutableList = append(foundExecutableList, exePath)
			}
		}
	}
	// if user has specified exes to analyze, return the list of exes found
	if len(inputExecutableNames) > 0 {
		if len(inputFoundExecutaleList) > 0 {
			if len(inputExecutableNameList) == len(inputFoundExecutaleList) {
				return inputFoundExecutaleList, false
			}
		}
		// failed to find some specified input executables, return the boolean as true.
		return inputFoundExecutaleList, true
	}

	// If user doesn't specify exe names to analyze
	// Use interactive mode for user to select exes they want to analyze
	foundExecutableCnt := len(foundExecutableList)
	if foundExecutableCnt == 0 {
		return nil, false
	} else if foundExecutableCnt == 1 {
		// it's the only choice, let's return it immediately
		return foundExecutableList, false
	}

	// the project does generate multiple executables or libaries.
	if conflib.GetBool("analyzeAll", false) {
		return foundExecutableList, false
	}

	// we show the list to screen and allow the user to pick up.
	// in case there are too many executables, the following rules are
	// used to display executables
	//
	//   1. sort executables in the reverse order when they were generated
	//   2. name with "test" ad suffix or prefix will have lowest priority
	//   3. display at most 20
	//
	var selectedExe []string

	// print the executable list with ord numbers
	cwd, _ := os.Getwd()
	foundExecutableList = sortFoundExecutables(foundExecutableList)
	fmt.Println(multipleExecutableMsg)
	for i, s := range foundExecutableList {
		ord := i + 1
		relativePath := getRelativePath(cwd, s)
		fmt.Printf("%2d) %s\n", ord, relativePath)
	}
	fmt.Println("")

	// parse user's input
	reader := bufio.NewReader(os.Stdin)
	for true {
		selectedExe = nil
		fmt.Print("Please select binaries by entering their ordinal numbers (e.g. 1,2,6):")
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(strings.Replace(text, "\n", "", -1))
		if len(text) == 0 {
			continue
		}

		tmpTokens := strings.Split(text, ",")
		validInput := true
		for i := 0; i < len(tmpTokens); i++ {
			if ord, err := strconv.Atoi(tmpTokens[i]); err != nil || ord <= 0 || ord > foundExecutableCnt {
				fmt.Println("Invalid input")
				validInput = false
				break
			} else {
				selectedExe = append(selectedExe, foundExecutableList[ord-1])
			}
		}
		if !validInput {
			continue
		}

		return selectedExe, false
	}

	panic("Unreachable code")
}

// Find all exetuable paths
// Used for `-XbcOnly`
func getAllExecutablePath(token string) []string {
	execPaths := getAllExecFromDB()

	var foundExes []string
	for i := 0; i < len(execPaths); i++ {
		exePath := execPaths[i]
		if strings.HasSuffix(exePath, ".o") ||
			strings.HasSuffix(exePath, "conftest") ||
			strings.HasSuffix(exePath, "libgtest_main.a") ||
			strings.HasSuffix(exePath, "libgmock_main.a") ||
			strings.HasSuffix(exePath, "libgtest.a") ||
			strings.HasSuffix(exePath, "libgmock.a") ||
			(!platform.IsExecutable(exePath) && !platform.IsLibrary(exePath)) {
			continue
		}

		_, filename := path.Split(exePath)
		// ignore all hidden files
		if !strings.HasPrefix(filename, ".") {
			foundExes = append(foundExes, exePath)
		}
	}
	return foundExes
}

func panicGracefully(msg string, err error, tmpDir string) {
	logger.Errorf("%s err=%v", msg, err)
	//os.Stderr.WriteString(fmt.Sprintf("%v\n", err))
	exitGracefully(1, tmpDir)
}

func exitGracefully(exitCode int, tmpDir string) {
	if len(tmpDir) > 0 && util.FileExists(tmpDir) {
		os.RemoveAll(tmpDir)
	}

	logger.Infof("End of the session. exitCode=%d", exitCode)
	os.Exit(exitCode)
}

/**
 *
 * It checks index.json under $cwd/.coderrect/build/index.json. It returns 0
 * if .coderrect/build/index.json doesn't exist or none of executables tracked by
 * index.json has races. Otherwise, it returns 1.
 *
 * Panic if there are other errors.
 *
 * If there are races found, it also calls reporter to generate
 * a terminal-based report
 */
func checkResults() (bool, bool) {
	var indexInfo IndexInfo
	// NOTE: Now only read from default path, maybe this need to change in the future
	res := reporter.ReadIndexJSON(&indexInfo, "")
	// index.json not found
	if res == 0 {
		return false, false
	}

	foundRaces := false
	for _, executable := range indexInfo.Executables {
		if executable.DataRaces > 0 {
			foundRaces = true
			break
		}
	}

	if foundRaces {
		// If there are any race found,
		// then check if there are any NEW race found.
		foundNewRaces := false
		for _, executable := range indexInfo.Executables {
			if executable.NewBugs {
				foundNewRaces = true
				break
			}
		}
		if foundNewRaces {
			return true, true
		}
		return true, false
	}
	return false, false
}

/**
 * See https://coderrect.atlassian.net/browse/COD-558
 *
 * We check if "msg" contains the keyword "libtinfo.so.5" and show more user-friendly
 * messages if there is.
 */
func onLibtinfoIssue(msg []byte) bool {
	msgstr := string(msg)
	matched, err := regexp.MatchString(`libtinfo.so.5:\s+cannot open shared objec`, msgstr)
	if err != nil || !matched {
		return false
	}

	fmt.Printf("\nClang 9 requires libtinfo.so.5 that isn't found in your system. Please check %s to fix this issue.\n\n",
		"http://docs.coderrect.com/fix-libtinfo-so-5-issue/")
	return true
}

type BufferedStderr struct {
	buffer *bytes.Buffer
}

func (bs BufferedStderr) Write(data []byte) (n int, err error) {
	max := 32 * 1024
	nleft := max - bs.buffer.Len()
	if len(data) > nleft {
		bs.buffer.Reset()
		nleft = max
	} // no choice, just returns the only one

	if len(data) <= nleft {
		bs.buffer.Write(data)
	}

	return os.Stderr.Write(data)
}

// func addOrReplaceLogFolder(jsonOptStr string, logFolder string) string {
// 	type CmdlineOpts struct {
// 		Opts []string
// 	}
// 	var cmdlineOpts CmdlineOpts
// 	if err := json.Unmarshal([]byte(jsonOptStr), &cmdlineOpts); err != nil {
// 		logger.Warnf("Failed to unmarshal cmdlineopt json. str=%s, err=%v", jsonOptStr, err)
// 		return jsonOptStr
// 	}

// 	tmp := cmdlineOpts.Opts[:0]
// 	replaced := false
// 	for _, item := range cmdlineOpts.Opts {
// 		if strings.HasPrefix(item, "-logger.logFolder=") {
// 			replaced = true
// 			tmp = append(tmp, "-logger.logFolder="+logFolder)
// 		} else {
// 			tmp = append(tmp, item)
// 		}
// 	}

// 	if !replaced {
// 		tmp = append(tmp, "-logger.logFolder="+logFolder)
// 	}

// 	cmdlineOpts.Opts = tmp
// 	if b, err := json.Marshal(&cmdlineOpts); err != nil {
// 		logger.Warnf("Failed to marshal json. str=%s, err=%v", jsonOptStr, err)
// 		return jsonOptStr
// 	} else {
// 		return string(b)
// 	}
// }

func addOrReplaceCommandline(jsonOptStr string, key string, value string) string {
	type CmdlineOpts struct {
		Opts []string
	}
	var cmdlineOpts CmdlineOpts
	if err := json.Unmarshal([]byte(jsonOptStr), &cmdlineOpts); err != nil {
		logger.Warnf("Failed to unmarshal cmdlineopt json. str=%s, err=%v", jsonOptStr, err)
		return jsonOptStr
	}

	tmp := cmdlineOpts.Opts[:0]
	option := fmt.Sprintf("-%s=%s", key, value)
	replaced := false
	for _, item := range cmdlineOpts.Opts {
		if strings.HasPrefix(item, "-"+key) {
			replaced = true
			tmp = append(tmp, option)
		} else {
			tmp = append(tmp, item)
		}
	}

	if !replaced {
		tmp = append(tmp, option)
	}

	cmdlineOpts.Opts = tmp
	if b, err := json.Marshal(&cmdlineOpts); err != nil {
		logger.Warnf("Failed to marshal json. str=%s, err=%v", jsonOptStr, err)
		return jsonOptStr
	} else {
		return string(b)
	}
}

func addEntryPoints(jsonOptStr string, entryPoints []string) string {
	type CmdlineOpts struct {
		Opts []string
	}
	var cmdlineOpts CmdlineOpts
	if err := json.Unmarshal([]byte(jsonOptStr), &cmdlineOpts); err != nil {
		logger.Warnf("Failed to unmarshal cmdlineopt json. str=%s, err=%v", jsonOptStr, err)
		return jsonOptStr
	}

	cmdlineOpts.Opts = append(cmdlineOpts.Opts, "@openlib.entryPoints="+strings.Join(entryPoints, ","))

	if b, err := json.Marshal(&cmdlineOpts); err != nil {
		logger.Warnf("Failed to marshal json. str=%s, err=%v", jsonOptStr, err)
		return jsonOptStr
	} else {
		return string(b)
	}
}

// func addArg(jsonOptStr string, arg string) string {
// 	type CmdlineOpts struct {
// 		Opts []string
// 	}
// 	var cmdlineOpts CmdlineOpts
// 	if err := json.Unmarshal([]byte(jsonOptStr), &cmdlineOpts); err != nil {
// 		logger.Warnf("Failed to unmarshal cmdlineopt json. str=%s, err=%v", jsonOptStr, err)
// 		return jsonOptStr
// 	}

// 	cmdlineOpts.Opts = append(cmdlineOpts.Opts, arg)

// 	if b, err := json.Marshal(&cmdlineOpts); err != nil {
// 		logger.Warnf("Failed to marshal json. str=%s, err=%v", jsonOptStr, err)
// 		return jsonOptStr
// 	} else {
// 		return string(b)
// 	}
// }
//TODO find bc file from db
func getBCFilePath(execFileName string, coderrectBuildDir string) string {
	bcFile, _ := shared.FindBCFilePathMapping(execFileName+".bc", coderrectBuildDir, false)
	// check if bcFile doesn't exist.
	if !util.FileExists(bcFile) && !util.LinkFileExists(bcFile) {
		dir, file := filepath.Split(bcFile)
		if !strings.HasPrefix(file, ".") {
			file = "." + file
		}
		bcFile = filepath.Join(dir, file)
	}
	return bcFile
}

func getPackageVersion() string {
	coderrectHome, _ := conflib.GetCoderrectHome()
	versionFilepath := path.Join(coderrectHome, "VERSION")
	if buf, err := ioutil.ReadFile(versionFilepath); err != nil {
		return "Unknown"
	} else {
		versionStr := string(buf)
		return versionStr[len("Package "):]
	}
}

/**
 * Calls racedetect to generate a API file under $CODERRECT_TMPDIR/api.json and
 * show a selected list of APIs to the screen and let the user choose from.
 */
func getAPIList(bcFilePath string) ([]string, error) {
	// run racedetect --print-api to generate the api list
	coderrectHome, _ := conflib.GetCoderrectHome()
	racedetect := filepath.Join(coderrectHome, "bin", "racedetect")
	noprogress := conflib.GetBool("noprogress", false)

	var cmdArgument []string
	cmdArgument = append(cmdArgument, "--print-api")
	if noprogress {
		cmdArgument = append(cmdArgument, "--no-progress")
	}
	cmdArgument = append(cmdArgument, bcFilePath)
	omplibPath := filepath.Join(coderrectHome, "bin", "libomp.so")
	cmdline := fmt.Sprintf("export LD_PRELOAD=%s; %s ", omplibPath, racedetect)
	for _, s := range cmdArgument {
		cmdline = cmdline + normalizeCmdlineArg(s) + " "
	}
	cmdline = "(" + cmdline + ")"
	cmd := platform.Command(cmdline)
	//cmd := exec.Command(racedetect, cmdArgument...)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		logger.Errorf("Failed to generate the api list. cmd=%s %s, err=%v", racedetect, cmdArgument, err)
		return nil, err
	}

	// unmarshall the json to get the complete api list
	tmpDir := os.Getenv("CODERRECT_TMPDIR")
	apiJsonPath := filepath.Join(tmpDir, "api.json")
	jsonBytes, err := ioutil.ReadFile(apiJsonPath)
	if err != nil {
		logger.Errorf("Failed to read api.json. apiJsonPath=%s, err=%v", apiJsonPath, err)
		return nil, err
	}

	type ApiList struct {
		Apis []string
	}
	var apiListJson ApiList
	if err := json.Unmarshal(jsonBytes, &apiListJson); err != nil {
		logger.Errorf("Failed to unmarshal cmdlineopt json. str=%s, err=%v", string(jsonBytes), err)
		return nil, err
	}
	if len(apiListJson.Apis) == 0 {
		return nil, nil
	}

	// show the list and list the customer to pick up
	maxWidth := 0
	for _, name := range apiListJson.Apis {
		if len(name) > maxWidth {
			maxWidth = len(name)
		}
	}
	maxWidth += 10

	col := 3
	var startIdx []int
	if len(apiListJson.Apis) >= 3 {
		for i := 0; i < col; i++ {
			startIdx = append(startIdx, int(len(apiListJson.Apis)*i/col))
		}
	} else {
		startIdx = append(startIdx, 0)
		startIdx = append(startIdx, 1)
		startIdx = append(startIdx, 0)
	}
	maxLoop := startIdx[1]
	for i := 0; i < maxLoop; i++ {
		for c := 0; c < col && c < len(apiListJson.Apis); c++ {
			pos := startIdx[c]
			if pos < len(apiListJson.Apis) {
				fmt.Printf("%4d) %s", pos+1, apiListJson.Apis[pos])
				spaces := maxWidth - (len(apiListJson.Apis[pos]) + 6)
				for k := 0; k < spaces; k++ {
					fmt.Print(" ")
				}
				startIdx[c]++
			}
		}
		fmt.Println("")
	}
	fmt.Println("")

	reader := bufio.NewReader(os.Stdin)
	for true {
		fmt.Print("Please select APIs by entering their numbers or names (e.g. 1,2,leveldb::SharedLRUCache,22): ")
		text, _ := reader.ReadString('\n')
		text = strings.Replace(text, "\n", "", -1)

		selectedList, err := convertToList(apiListJson.Apis, text)
		if err != nil {
			fmt.Printf("Invalid input - %v. \n", err)
			continue
		}

		return selectedList, nil
	}

	return nil, nil
}

func convertToList(list []string, input string) ([]string, error) {
	var rs []string
	tokens := strings.Split(input, ",")
	for _, token := range tokens {
		token = strings.Trim(token, " ")
		i, err := strconv.Atoi(token)
		if err == nil {
			i--
			if i >= len(list) {
				return nil, errors.New("Index is out of range")
			}
			rs = append(rs, list[i])
		} else {
			rs = append(rs, token)
		}
	}

	if len(rs) == 0 {
		return nil, errors.New("Empty result list")
	}
	return rs, nil
}

var skip_over_flow_checks bool

func findAllXargoTomlDirectory(rootPath string) []string {

	//targetName := "Xargo.toml";
	var pathList []string
	var ignorePaths []string
	err := filepath.WalkDir(rootPath,
		func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				//fmt.Println(err)
				return err
			}
			//check malicious path
			if strings.Contains(path, ";") || strings.Contains(path, "`") || strings.Contains(path, "<") {
				logger.Infof("skip malicious path: %s", path)
				return filepath.SkipDir
			}

			if strings.HasSuffix(path, "soteria.ignore") || strings.HasSuffix(path, "sec3.ignore") {
				//ignorePaths
				bytesRead, _ := ioutil.ReadFile(path)
				file_content := string(bytesRead)
				lines := strings.Split(file_content, "\n")
				for _, line := range lines {
					if len(line) > 0 {
						ignorePaths = append(ignorePaths, line)
					}
				}
			}
			{ //check folder
				file, err := os.Open(path)
				if err != nil {
					//fmt.Println(err)
					return err
				}
				fileInfo, err := file.Stat()
				if err != nil {
					//fmt.Println(err)
					return err
				}
				if fileInfo.IsDir() {
					if strings.Contains(path, "/.") {
						return filepath.SkipDir
					}
					parent, _ := filepath.Split(path)
					if strings.Contains(parent, "/proptest") || strings.Contains(parent, "/logs") || strings.Contains(parent, "/target") || strings.Contains(parent, "/docs") || strings.Contains(parent, "/migrations") || strings.Contains(parent, "/services") || strings.Contains(parent, "/js") || strings.HasSuffix(parent, "/test") {
						//fmt.Println("skip path: ", path)
						logger.Infof("skip path: %s", path)

						return filepath.SkipDir

						// if strings.HasPrefix(name, ".") || name == "target" {
						// 	return filepath.SkipDir
						// }
					}

					//if path contains ignore path
					for _, ignore_path := range ignorePaths {
						path1 := path + "/"
						ignore_path1 := ignore_path + "/"
						if strings.HasSuffix(path, ignore_path) || strings.HasSuffix(path1, ignore_path) || strings.HasSuffix(path, ignore_path1) {
							fmt.Println("ignored path: ", path)
							return filepath.SkipDir
						}
					}
				}
			}

			if strings.HasSuffix(path, "argo.toml") {
				//check section [dependencies] solana-program
				config, err := toml.LoadFile(path)
				if err == nil {
					//fmt.Println("Checking Path: ", path)
					solana := config.Get("dependencies.solana-program")
					anchor := config.Get("dependencies.anchor-lang")
					if solana != nil || anchor != nil {
						srcFolder, _ := filepath.Split(path)
						pathList = append(pathList, strings.TrimSuffix(srcFolder, "/"))
						fmt.Println("Program found in: ", srcFolder)
						//fmt.Println("solana-program: ", solana)
					}
					overflow := config.Get("profile.release.overflow-checks")
					if overflow != nil {
						if overflow == true {
							//fmt.Println("overflow-checks true: ", overflow)
							skip_over_flow_checks = true
						} else {
							//fmt.Println("overflow-checks false: ", overflow)
						}
					}
				}
			}
			//fmt.Println("path: ", path)
			return nil
		})
	if err != nil {
		fmt.Println(err)
	}
	//if path contains ignore path
	if len(ignorePaths) > 0 {
		var pathList2 []string
		for _, path := range pathList {
			ignored := false
			for _, ignore_path := range ignorePaths {
				path1 := path + "/"
				if strings.Contains(path1, ignore_path) {
					fmt.Println("ignored path: ", path)
					ignored = true
					break
				}
			}

			if !ignored {
				pathList2 = append(pathList2, path)
			}
		}
		pathList = pathList2
	}
	//fmt.Println("pathList: ", pathList)
	return pathList
}
func generateSolanaIR(args []string, srcFilePath string, tmpDir string) string {
	coderrectHome := os.Getenv("CODERRECT_HOME")
	exePath := filepath.Join(coderrectHome, "bin", "sol-racedetect")
	srcFolder, irFileName := filepath.Split(srcFilePath)
	if len(irFileName) == 0 {
		srcFolder, irFileName = filepath.Split(strings.TrimSuffix(srcFolder, "/"))
	}
	//if irFileName == "program"
	{
		//add more info
		_, name := filepath.Split(strings.TrimSuffix(srcFolder, "/"))
		//srcFolder2, _ := filepath.Split(strings.TrimSuffix(srcFolder2, "/"))
		if len(name) > 0 {
			irFileName = name + "_" + irFileName
		}
	}
	//fmt.Printf("srcFilePath: %v\n", srcFilePath)

	fileInfo, err := os.Stat(srcFilePath)
	if err != nil {
		// error handling
	}

	// IsDir is short for fileInfo.Mode().IsDir()
	if fileInfo.IsDir() {
		// file is a directory
		srcFolder = srcFilePath
	} else {
		// file is not a directory
	}

	//fmt.Printf("path: %v ir: %v.ll\n", srcFolder, irFileName)

	// cwd, err := os.Getwd()
	// if err != nil {
	// 	panicGracefully("Failed to obtain the current working directory for smalltalk IR", err, "")
	// }
	irFilePath := filepath.Join(srcFolder, irFileName+".ll")
	//irFilePath := "t.ll"

	cmdline := fmt.Sprintf("%s --dump-ir -o %s ", exePath, irFilePath)
	cmdline = cmdline + normalizeCmdlineArg(srcFilePath) + " "
	for _, s := range args {
		cmdline = cmdline + normalizeCmdlineArg(s) + " "
	}

	//fmt.Printf("cmdline: %v\n", cmdline)

	//cmd = exec.Command(racedetect, cmdArgument...)
	cmdline = "(" + cmdline + ")"
	cmd := platform.Command(cmdline)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		panicGracefully(fmt.Sprintf("Failed to parse the BC file. cmdline=%s", cmdline), err, tmpDir)
	}

	return irFilePath
}
func checkCRSoteriaL() int {
	var key1 string
	var key2 string
	var key3 string
	key1 = "2zCz"
	key2 = "2GgSoSS68eNJENWrYB48dMM1zmH8SZkgYneVDv2G4g"
	key3 = "D6Es"
	key := key1 + key2 + "RsVfwu5rNXtK5BKFxn7fSqX9BvrBc1rdPAeBEc" + key3
	//fmt.Printf("key: %s\n", key)
	input := os.Getenv("SOTERIA_LICENSE_KEY")
	//fmt.Printf("input: %s\n", input)
	if key != input {
		return 1
	} else {
		time_now := time.Now().Unix()
		//fmt.Printf("time_now: %s\n", time_now)
		if time_now > 1646779189 {
			return 2
		}
	}
	return 0
}
func main() {
	error_code := checkCRSoteriaL()
	error_code = 0
	if error_code > 0 {
		if error_code == 1 {
			fmt.Printf("ERROR: Invalid License Key.\nPlease set correct environment variable with your license key by:\nexport SOTERIA_LICENSE_KEY=your_valid_license_key\n")
		} else {
			fmt.Printf("ERROR: License Expired. Please contact soteria.dev\n")
		}
		os.Exit(error_code)
	}

	if len(os.Args) == 1 {
		fmt.Printf("%s\n", usage)
		return
	}

	argAlias := map[string]string{
		"v":  "version",
		"h":  "help",
		"o":  "report.outputDir",
		"e":  "analyzeBinaries",
		"p":  "bcPaths",
		"t:": "report.enableTerminal",
		"c:": "cleanBuild",
		"s:": "startServer",
	}

	remainingArgs, err := conflib.Initialize(argAlias)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	coderrectHome, err := conflib.GetCoderrectHome()
	if err != nil {
		fmt.Printf("Unable to find coderrect home - %v\n", err)
		os.Exit(2)
	}

	// show versions of all components
	if conflib.GetBool("version", false) || conflib.GetBool("-version", false) {
		version, err := ioutil.ReadFile(filepath.Join(coderrectHome, "VERSION"))
		if err != nil {
			fmt.Printf("Unable to read %s/VERSION - %v\n", coderrectHome, err)
			os.Exit(3)
		} else {
			fmt.Printf("%s\n", version)
			os.Exit(0)
		}
	}

	// show usage info
	if conflib.GetBool("help", false) || conflib.GetBool("-help", false) {
		fmt.Printf("%s\n", usage)
		os.Exit(0)
	}

	// show effective value of all configurations
	if conflib.GetBool("showconf", false) {
		fmt.Printf("%s\n", conflib.Dump())
		os.Exit(0)
	}

	_, err = logger.InitWithLogRotation("coderrect")
	if err != nil {
		exitGracefully(1, "")
	}

	version := getPackageVersion()

	logger.Infof("Start a new session. version=%s, args=%v", version, os.Args)
	os.Stdout.WriteString(fmt.Sprintf("Coderrect %s\n", version))

	cwd, err := os.Getwd()
	if err != nil {
		panicGracefully("Failed to obtain the current working directory", err, "")
	}

	directBCFile := conflib.GetString("Xbc", "")

	// create the coderrect working directory
	fi, _ := os.Stat(cwd)
	coderrectWorkingDir := filepath.Join(cwd, ".coderrect")

	// create the build directory
	cleanBuild := conflib.GetBool("cleanBuild", false)
	logger.Infof("Check cleanBuild flag. cleanBuild=%v", cleanBuild)
	coderrectBuildDir := filepath.Join(coderrectWorkingDir, "build")

	// Find the bold db path
	boldDbFilePath := filepath.Join(coderrectBuildDir, "bold.db")

	// set the following environment variables
	//   - PATH - make sure coderrect_home/bin in the front of other folders
	//   - CODERRECT_HOME
	//   - CODERRECT_BUILD_DIR
	//   - CODERRECT_CWD
	//   - CODERRECT_TMPDIR
	//   - CODERRECT_TOKEN - the token of this run. it's used by all components
	//                       as the key (or part of the key) to retrieve diskv db
	//   - CODERRECT_CMDLINE_OPTS - command-line opts passed to child components

	os.Setenv("CODERRECT_HOME", coderrectHome)
	os.Setenv("PATH", filepath.Join(coderrectHome, "bin")+":"+os.Getenv("PATH"))

	// Because of COD-1456, tmp directory for racedetect have to create
	token := fmt.Sprintf("%d", time.Now().Unix())
	tmpDir := filepath.Join(coderrectWorkingDir, "tmp", "coderrect-"+token)
	os.Setenv("CODERRECT_TMPDIR", tmpDir)
	if _, serr := os.Stat(tmpDir); serr != nil {
		merr := os.MkdirAll(tmpDir, os.ModePerm)
		if merr != nil {
			panicGracefully("Failed to create tmp directory", merr, "")
		}
	}

	os.Setenv("CODERRECT_TOKEN", token)
	os.Setenv("CODERRECT_BOLD_DB_PATH", boldDbFilePath)
	os.Setenv("CODERRECT_BUILD_DIR", coderrectBuildDir)
	os.Setenv("CODERRECT_CWD", coderrectWorkingDir)

	bcPaths := conflib.GetString("bcPaths", "")
	if len(bcPaths) > 0 {
		os.Setenv("CODERRECT_BC_PATHS", bcPaths)
	}

	// set env variables for user specified libear intercepts
	//   - CR_INTERCEPT_C_COMPILERS
	//   - CR_INTERCEPT_CXX_COMPILERS
	//   - CR_INTERCEPT_FORTRAN_COMPILERS
	//   - CR_INTERCEPT_AR_LINKERS
	//   - CR_INTERCEPT_LD_LINKERS
	//   - CR_INTERCEPT_FILE_OPERATORS

	os.Setenv("CR_INTERCEPT_C_COMPILERS", strings.Join(conflib.GetStrings("buildTools.cCompiler"), " "))
	os.Setenv("CR_INTERCEPT_CXX_COMPILERS", strings.Join(conflib.GetStrings("buildTools.cxxCompiler"), " "))
	os.Setenv("CR_INTERCEPT_FORTRAN_COMPILERS", strings.Join(conflib.GetStrings("buildTools.fortranCompiler"), " "))
	os.Setenv("CR_INTERCEPT_AR_LINKERS", strings.Join(conflib.GetStrings("buildTools.archiver"), " "))
	os.Setenv("CR_INTERCEPT_LD_LINKERS", strings.Join(conflib.GetStrings("buildTools.linker"), " "))
	os.Setenv("CR_INTERCEPT_FILE_OPERATORS", strings.Join(conflib.GetStrings("buildTools.fileOp"), " "))

	// Per COD-665, we add (or replace) "-logger.logFolder" in the cmdline opts
	//
	cmdlineOpts := addOrReplaceCommandline(conflib.GetCmdlineOpts(), "logger.logFolder", logger.GetLogFolder())
	// FIXME: the below addOrReplaceCommandline seems redundant
	// if conflib.GetBool("publish.jenkins", false) {
	// 	cmdlineOpts = addOrReplaceCommandline(cmdlineOpts, "reporter.terminalReportOnly", "true")
	// }
	customConfigPath := conflib.GetString("conf", "")
	if len(customConfigPath) > 0 {
		// cod1974, replace relative path to abs, and pass to next
		customConfigPath, _ = filepath.Abs(customConfigPath)
		cmdlineOpts = addOrReplaceCommandline(cmdlineOpts, "conf", customConfigPath)
	}

	// start the code server for reporter
	if conflib.GetBool("startServer", false) || conflib.GetBool("-startServer", false) {
		cmdlineOpts = addOrReplaceCommandline(cmdlineOpts, "startServer", "true")
		// Only set this to let racedetect output full path for project source code
		// when we do use the code server
		// cmdlineOpts = addOrReplaceCommandline(cmdlineOpts, "cwd", cwd)
	}
	cmdlineOpts = addOrReplaceCommandline(cmdlineOpts, "cwd", cwd)

	os.Setenv("CODERRECT_CMDLINE_OPTS", cmdlineOpts)
	logger.Infof("Passed cmdline args via env. cmdlineOpts=%v", os.Getenv("CODERRECT_CMDLINE_OPTS"))

	homeBinDirPath := filepath.Join(coderrectHome, "bin")
	os.Setenv("CODERRECT_CLANG", filepath.Join(homeBinDirPath, "coderrect-clang"))
	os.Setenv("CODERRECT_CLANGXX", filepath.Join(homeBinDirPath, "coderrect-clang++"))
	os.Setenv("CODERRECT_AR", filepath.Join(homeBinDirPath, "coderrect-ar"))
	os.Setenv("CODERRECT_LD", filepath.Join(homeBinDirPath, "coderrect-ld"))
	os.Setenv("CODERRECT_FORTRAN", filepath.Join(homeBinDirPath, "coderrect-fortran"))
	os.Setenv("CODERRECT_CP", filepath.Join(homeBinDirPath, "coderrect-cp"))

	/**
	 * there are list of environment variables used by gllvm.compiler.go
	 *
	 * envpath    = "LLVM_COMPILER_PATH"
	 * envcc      = "LLVM_CC_NAME"
	 * envcxx     = "LLVM_CXX_NAME"
	 * envar      = "LLVM_AR_NAME"
	 * envlnk     = "LLVM_LINK_NAME"
	 */
	clangBinDirPath := filepath.Join(coderrectHome, "clang", "bin")
	os.Setenv("LLVM_COMPILER_PATH", clangBinDirPath)
	os.Setenv("LLVM_CC_NAME", filepath.Join(clangBinDirPath, "clang"))
	os.Setenv("LLVM_CXX_NAME", filepath.Join(clangBinDirPath, "clang++"))
	os.Setenv("LLVM_LINK_NAME", filepath.Join(clangBinDirPath, "llvm-link"))
	os.Setenv("LLVM_CL_NAME", filepath.Join(clangBinDirPath, "clang-cl"))
	os.Setenv("LLD_LINK_NAME", filepath.Join(clangBinDirPath, "lld-link"))

	os.Setenv("LLVM_FORTRAN_COMPILER_PATH", filepath.Join(coderrectHome, "fortran", "bin"))
	os.Setenv("LLVM_FORTRAN_NAME", filepath.Join(clangBinDirPath, "flang"))
	os.Setenv("FORTRAN_LLVM_LINK_NAME", filepath.Join(clangBinDirPath, "llvm-link"))

	if !util.FileExists(coderrectWorkingDir) {
		if err = os.Mkdir(coderrectWorkingDir, fi.Mode()); err != nil {
			panicGracefully("Failed to create the working directory.", err, "")
		}
	}

	if util.FileExists(coderrectBuildDir) {
		if cleanBuild {
			logger.Infof("Delete the build directory. dir=%s", coderrectBuildDir)
			if err := os.RemoveAll(coderrectBuildDir); err != nil {
				panicGracefully("Failed to delete old build directory.", err, "")
			}
		}
	}

	if !util.FileExists(coderrectBuildDir) {
		logger.Infof("Create the build directory. dir=%s", coderrectBuildDir)
		if err = os.Mkdir(coderrectBuildDir, fi.Mode()); err != nil {
			panicGracefully("Failed to create build directory.", err, "")
		}
	}

	// FIXME: now we always keep boltdb, will this cause problem?
	// remove last boltdb, creat new boltdb
	if !util.FileExists(boldDbFilePath) {
		// os.Remove(boldDbFilePath)
		boltDb, err := bolt.Open(boldDbFilePath, 0666, nil)
		if err != nil {
			panicGracefully("Failed to to create db.", err, "")
		}
		err = boltDb.Update(func(tx *bolt.Tx) error {
			_, err = tx.CreateBucket([]byte(shared.BucketNameBcFile))
			if err != nil {
				return err
			}
			_, err = tx.CreateBucket([]byte(shared.BucketNameExecFile))
			_, err = tx.CreateBucket([]byte(shared.BucketNameMoveFile))
			return err
		})
		if err != nil {
			panicGracefully("Failed to create a bucket in the db.", err, "")
		}
		boltDb.Close()
	}

	// upload report to remote host, used for CI/CD integration
	host := conflib.GetString("publish.cloud", "")
	if conflib.GetBool("publish", false) {
		host = "DEFAULT"
	}
	if conflib.GetBool("publish.jenkins", false) || host != "" {
		foundRace, foundNewRace := checkResults()

		// Force push the report to cloud server
		// This is used in some special cases like we have a manually generated report from race.json
		// therefore `checkResults` fail to return correct result
		force := conflib.GetBool("publish.force", false)

		// if any race is detected
		if foundRace || force {
			// only upload report to cloud if cloud host is set
			if host != "" {
				logger.Infof("Upload report to remote server")
				cwd, err := os.Getwd()
				if err != nil {
					panicGracefully("Failed to obtain the current working directory", err, "")
				}

				reportDir := conflib.GetString("publish.src", "")
				if reportDir == "" {
					// use default report path if `publish.src` is not specified
					reportDir = filepath.Join(cwd, ".coderrect", "report")
				}

				url := reporter.PublishReport(reportDir, host)

				// dump the url for the uploaded report to a file
				// use for our CI/CD tasks
				dump := conflib.GetBool("publish.dumpLink", false)
				if dump {
					dumpPath := filepath.Join(cwd, "coderrect-publish-result")
					dumpContent := url
					if !foundNewRace {
						dumpContent = dumpContent + "\nNo New Race Detected"
					}
					if err = ioutil.WriteFile(dumpPath, []byte(dumpContent), fi.Mode()); err != nil {
						logger.Errorf("Fail to create index.json. err=%v", err)
						panicGracefully("Failed to create index.json", err, tmpDir)
					}
				}
			}
			// check whether user want to suppress coderrect fomr exit(1) when found race
			exit0 := conflib.GetBool("publish.exit0", false)
			if exit0 {
				os.Exit(0)
			}
			os.Exit(1)
		}
		fmt.Println("No vulnerabilities detected, no report to publish")
		os.Exit(0)
	}

	//fmt.Println("remainingArgs: ", remainingArgs)
	//logger.Infof("remainingArgs =%v", remainingArgs)
	isCargoBuild := false // This is c direct Cargo Build
	var executablePathList []string
	var executablesInCmdlineNotExist bool
	cmdline := ""

	if len(directBCFile) == 0 {
		if conflib.GetBool("solana.on", false) && len(remainingArgs) > 0 {
			//for solana, invoke sol-racedetect to generate IR first
			logger.Infof("Execute sol-racedetect command to generate IR files. cmdline=%v", remainingArgs)
			srcFilePath, _ := filepath.Abs(remainingArgs[0])
			var args []string
			if len(remainingArgs) > 1 {
				args = remainingArgs[1:]
			}
			//find all directories with the Xargo.toml file
			solanaProgramDirectories := findAllXargoTomlDirectory(srcFilePath)
			if len(solanaProgramDirectories) == 0 {
				fmt.Println("No Xargo.toml or Cargo.toml file found in:", srcFilePath)
				os.Exit(0)
			}
			for _, programPath := range solanaProgramDirectories {
				directBCFile = generateSolanaIR(args, programPath, tmpDir)
				executablePathList = append(executablePathList, directBCFile)
			}
			//then invoke coderrect -Xbc=path-to-ir
		}
	}
	if len(executablePathList) == 0 {
		//something is wrong - need to exit
		if len(directBCFile) == 0 {
			// step 1. Compile the project to generate executables
			//
			logger.Infof("Execute the build command to generate BC files. cmdline=%v", remainingArgs)
			var cmd, cmdline = doCustomBuild(remainingArgs, isCargoBuild)
			var outBuf bytes.Buffer
			cmd.Stderr = BufferedStderr{
				buffer: &outBuf,
			}

			cmd.Stdout = os.Stdout
			logger.Infof("Execute command line with coderrect-exec: %s", cmdline)
			path := os.Getenv("PATH")
			logger.Infof("Path env: %s", path)
			if err := cmd.Run(); err != nil {
				// cargo build does not requite such error message check
				if !isCargoBuild && !onLibtinfoIssue(outBuf.Bytes()) {
					os.Stderr.WriteString("Failed to build the project\n")
					logger.Errorf("Failed to build the project. cmdline=%s", cmdline)
				}
				//if users add -continueIfBuildErrorPresents in cmd, we let the program continue to analyze
				//otherwise, we terminate the process
				if !conflib.GetBool("continueIfBuildErrorPresents", false) {
					exitGracefully(1, tmpDir)
				}
				logger.Warnf("Fail to build the project but still conitnue analyzing. cmdline=%v, err=%v", cmdline, err)
			}
			logger.Infof("Finish the build of the project. cmdline=%v", cmdline)

			// Check if the custom build contains any cargo command
			if !isCargoBuild {
				// For now we only check bins and examples
				cargoTargetDirs := [2]string{
					filepath.Join(coderrectBuildDir, "target", "debug", "deps"),
					filepath.Join(coderrectBuildDir, "target", "debug", "examples"),
				}
				for _, cargoTargetDir := range cargoTargetDirs {
					files, err := ioutil.ReadDir(cargoTargetDir)
					if err == nil && len(files) != 0 {
						// isCargoCustomBuild = true
						isCargoBuild = true
						break
					}
				}
			}

			// only generate bc files, not doing any analysis
			if conflib.GetBool("XbcOnly", false) {
				allExes := getAllExecutablePath(token)
				fmt.Println("Generate the BC file only.")

				for _, exe := range allExes {
					fmt.Printf("Generating bitcode for %s ...\n", exe)
					logger.Infof("On-demand link BC file. bcFile=%s.bc", exe)
					// on-demand linking
					shared.LinkBitcode(exe, coderrectBuildDir)
				}

				logger.Infof("End the session with BC files only")
				// No need to proceed to analysis phase
				exitGracefully(0, tmpDir)
			}

			executablePathList, executablesInCmdlineNotExist = getExecutablePathList(token, coderrectBuildDir, isCargoBuild)
			if executablesInCmdlineNotExist {
				fmt.Println("Unable to find executables specified in the command line.")
				exitGracefully(1, tmpDir)
			}

			if len(executablePathList) == 0 {
				fmt.Println("")
				fmt.Println("Unable to find any executable or library to analyze.")
				exitGracefully(0, tmpDir)
			}
		} else { // if len(directBCFile) > 0
			executablePathList = append(executablePathList, directBCFile)
		}
	}
	//
	// Assume the user picks up 3 binaries "prog1", "prog2", and "prog3". The code below
	// run racedetect against them one by one. For each binary, we generate a json file containing
	// race data. For example, we genarete prog1.json for "prog1" under .coderrect/build.
	//
	// We also generate index.json under .coderrect/build. Below is an example
	//
	// {
	//   "Executables": [
	//     {
	//        "Name": "prog1",
	//        "RaceJson": "path/to/coderrect_prog1_report.json",
	//        "DataRaces": 2  // number of data races
	//     },
	//     ... ...
	//   ]
	// }
	//
	var rawJsonPathList []string
	var outputDir string
	var interactiveOpenAPI bool
	var cmd *exec.Cmd

	var executableInfoList []ExecutableInfo
	totalRaceCnt := 0
	time_start := time.Now()
	//default 480min = 8h
	//support --h--m--s
	timeout, _ := time.ParseDuration(conflib.GetString("timeout", "8h"))

	for i := 0; i < len(executablePathList); i++ {
		var executableInfo ExecutableInfo

		bcFile := ""
		fmt.Printf("\nAnalyzing %s ...\n", executablePathList[i])
		directBCFile = executablePathList[i]
		if len(directBCFile) == 0 {
			if isCargoBuild {
				bcFile = executablePathList[i]
				if !strings.HasSuffix(bcFile, ".ll") {
					bcFile = getCargoTargetBcFile(coderrectBuildDir, executablePathList[i])
				}
			} else {
				logger.Infof("On-demand link BC file. bcFile=%s.bc", executablePathList[i])
				// on-demand linking
				bcFile, _ = shared.LinkBitcode(executablePathList[i], coderrectBuildDir)
			}
		} else {
			bcFile, _ = filepath.Abs(directBCFile)
		}
		if bcFile == "" || !util.FileExists(bcFile) {
			continue //skip if BC file does not exist
		}
		logger.Infof("Analyzing BC file to detect races. bcFile=%s.bc", bcFile)

		// Step 3. If the executable is a library and configuration doesn't specify entry points,
		// and user does not specify analyzeAll or openlib.analyzeAPI or openlib.optimal
		// we let the user choose APIs
		interactiveOpenAPI = false
		if platform.IsLibrary(executablePathList[i]) {
			apiList := conflib.GetStrings("openlib.entryPoints")
			logger.Infof("Get entry points from conf. apiList=%v", apiList)

			if len(apiList) == 0 && !conflib.GetBool("analyzeAll", false) && !conflib.GetBool("openlib.analyzeAPI", false) && !conflib.GetBool("openlib.optimal", false) {
				if apiList, err = getAPIList(bcFile); err != nil {
					logger.Errorf("Failed to get a list of APIs for analysis. err=%v", err)
					panicGracefully("Failed to get a list of APIs for analysis", err, tmpDir)
				}
				logger.Infof("The user chooses a list of APIs. apiList=%v", apiList)
				if apiList == nil {
					fmt.Println("We didn't detect public APIs for the given library.")
					continue
					//exitGracefully(1, tmpDir)
				}

				envCmdlineOpts := os.Getenv("CODERRECT_CMDLINE_OPTS")
				logger.Infof("Prepare to add entry points into env cmdline. envCmdlineOpts=%s", envCmdlineOpts)
				os.Setenv("CODERRECT_CMDLINE_OPTS", addEntryPoints(envCmdlineOpts, apiList))
				interactiveOpenAPI = true
			}
		}

		// Step 4. Call racedetect
		executablePath := executablePathList[i]
		_, executableName := filepath.Split(executablePath)
		executableInfo.Name = executableName
		tmpJsonPath := filepath.Join(tmpDir, fmt.Sprintf("coderrect_%s_report.json", executableName))
		logger.Infof("Generating the json file. jsonPath=%s", tmpJsonPath)
		racedetect := filepath.Join(coderrectHome, "bin", "racedetect")
		analyzeAPIFlag := ""
		if conflib.GetBool("openlib.analyzeAPI", false) {
			analyzeAPIFlag = "--analyze-api"
		}
		analyzeAPIOnceFlag := ""
		if conflib.GetBool("openlib.once", false) {
			analyzeAPIOnceFlag = "--openlib-once"
		}
		analyzeAPIOptimalFlag := ""
		if conflib.GetBool("openlib.optimal", false) {
			analyzeAPIOptimalFlag = "--openlib-optimal"
		}
		detectModeFlag := ""
		switch mode := conflib.GetString("mode", "normal"); mode {
		case "fast":
			detectModeFlag = "--fast"
		case "exhaust":
			detectModeFlag = "--full"
		case "normal":
			detectModeFlag = ""
		default:
			if len(mode) != 0 {
				fmt.Println("Unrecognized mode - " + mode)
				os.Exit(1)
			}
		}

		detectPlanFlag := ""
		switch plan := strings.ToLower(conflib.GetString("plan", "build")); plan {
		case "free":
			detectPlanFlag = "-plan=free"
		case "build":
			detectPlanFlag = ""
		case "scale":
			detectPlanFlag = "-plan=scale"
		default:
			if len(plan) != 0 {
				fmt.Println("Unrecognized plan - " + plan)
				os.Exit(1)
			}
		}
		//fmt.Println("detectPlanFlag:", detectPlanFlag)

		noprogress := conflib.GetBool("noprogress", false)
		displayFlag := ""
		if noprogress {
			displayFlag += "--no-progress"
		}

		var cmdArgument []string
		if len(analyzeAPIFlag) > 0 {
			cmdArgument = append(cmdArgument, analyzeAPIFlag)
		}
		if len(analyzeAPIOnceFlag) > 0 {
			cmdArgument = append(cmdArgument, analyzeAPIOnceFlag)
		}
		if len(analyzeAPIOptimalFlag) > 0 {
			cmdArgument = append(cmdArgument, analyzeAPIOptimalFlag)
		}
		if len(detectModeFlag) > 0 {
			cmdArgument = append(cmdArgument, detectModeFlag)
		}
		if len(detectPlanFlag) > 0 {
			cmdArgument = append(cmdArgument, detectPlanFlag)
		}
		if skip_over_flow_checks {
			cmdArgument = append(cmdArgument, "--skip-overflow")
		}
		if len(displayFlag) > 0 {
			cmdArgument = append(cmdArgument, displayFlag)
		}
		if isCargoBuild {
			// enable atomicity violation by default for Rust
			cmdArgument = append(cmdArgument, "-include-atomics")
		}
		cmdArgument = append(cmdArgument, "-o", tmpJsonPath, bcFile)
		omplibPath := filepath.Join(coderrectHome, "bin", "libomp.so")
		cmdline := fmt.Sprintf("export LD_PRELOAD=%s; %s ", omplibPath, racedetect)
		for _, s := range cmdArgument {
			cmdline = cmdline + normalizeCmdlineArg(s) + " "
		}
		//cmd = exec.Command(racedetect, cmdArgument...)
		cmdline = "(" + cmdline + ")"
		cmd := platform.Command(cmdline)
		cmd.Stderr = os.Stderr
		cmd.Stdout = os.Stdout
		if err := cmd.Run(); err != nil {
			panicGracefully(fmt.Sprintf("Failed to parse the BC file. cmdline=%s %s", racedetect, cmdArgument), err, tmpDir)
		}

		//For each executable, generate raw_$executable.json under .coderrect/build
		raceJsonBytes, err := ioutil.ReadFile(tmpJsonPath)
		if err != nil {
			panicGracefully("Unable to read json file", err, tmpDir)
		}
		// copy the raw json file to build
		rawJsonPath := filepath.Join(coderrectBuildDir, fmt.Sprintf("raw_%s.json", executableName))
		ioutil.WriteFile(rawJsonPath, raceJsonBytes, fi.Mode())
		executableInfo.RaceJSON = rawJsonPath

		//check number of race detected
		currentRaces := 0
		// FIXME: below is the fully list of bugs we can detect
		// not reporting deadlocks and OV because they are not production ready
		// raceTypes := []string{"dataRaces", "deadLocks", "mismatchedAPIs", "orderViolations", "toctou"}
		raceTypes := []string{"dataRaces", "raceConditions", "mismatchedAPIs", "toctou", "untrustfulAccounts", "unsafeOperations", "cosplayAccounts"}
		for _, raceType := range raceTypes {
			jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
				totalRaceCnt++
				currentRaces++
			}, raceType)
		}
		executableInfo.DataRaces = currentRaces
		executableInfo.NewBugs = false
		executableInfoList = append(executableInfoList, executableInfo)
		time_duration := time.Since(time_start)
		if time_duration > timeout {
			//fmt.Printf("Taking so far %s\n", time_duration)
			fmt.Printf("Timeout (exceeded %v), skipping the rest %v executables\n", timeout, (len(executablePathList) - i))
			break
		}

	}

	// reset the color of terminal
	color.Reset()

	// Generate index.json
	indexInfo := IndexInfo{
		Executables:  executableInfoList,
		CoderrectVer: version,
	}
	reporter.WriteIndexJSON(indexInfo, coderrectBuildDir)

	// create a configuration.json file which contains all configurations
	configRawString := []byte(conflib.Dump())
	configJSONPath := filepath.Join(coderrectBuildDir, "configuration.json")
	if err = ioutil.WriteFile(configJSONPath, configRawString, fi.Mode()); err != nil {
		logger.Errorf("Fail to create configuration.json. err=%v", err)
		panicGracefully("Failed to create configuration.json", err, tmpDir)
	}

	// step 5. call reporter to generate the report if there are races
	reporterBin := filepath.Join(coderrectHome, "bin", "reporter")
	outputDir = conflib.GetString("report.outputDir", filepath.Join(coderrectWorkingDir, "report"))
	os.RemoveAll(outputDir)
	cmd = exec.Command(reporterBin, outputDir, coderrectBuildDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	var exitCode int
	if err = cmd.Run(); err != nil {
		cmdline = fmt.Sprintf("%s %s %s", reporterBin, outputDir, coderrectBuildDir)
		// try to get the exit code
		if exitError, ok := err.(*exec.ExitError); ok {
			ws := exitError.Sys().(syscall.WaitStatus)
			exitCode = ws.ExitStatus()
		}
		if exitCode < 9 {
			panicGracefully(fmt.Sprintf("Failed to parse race detection result. cmdline=%s", cmdline), err, tmpDir)
		}
	} else {
		exitCode = 0
	}

	if interactiveOpenAPI {
		fmt.Println(openAPIMsg)
	}

	for i := 0; i < len(rawJsonPathList); i++ {
		os.Remove(rawJsonPathList[i])
	}
	os.RemoveAll(tmpDir)

	logger.Infof("End the session")
	//fmt.Println("Exit code: ", exitCode)
	os.Exit(exitCode)
}
