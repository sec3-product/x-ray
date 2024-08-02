package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/gookit/color"
	toml "github.com/pelletier/go-toml"

	"github.com/coderrect-inc/coderrect/pkg/reporter"
	"github.com/coderrect-inc/coderrect/pkg/util"
	"github.com/coderrect-inc/coderrect/pkg/util/conflib"
	"github.com/coderrect-inc/coderrect/pkg/util/jsonparser"
	"github.com/coderrect-inc/coderrect/pkg/util/logger"
	"github.com/coderrect-inc/coderrect/pkg/util/platform"
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
$ soteria path/to/solana-program-library
`

type ExecutableInfo = reporter.ExecutableInfo
type IndexInfo = reporter.IndexInfo

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
	{
		_, name := filepath.Split(strings.TrimSuffix(srcFolder, "/"))
		if len(name) > 0 {
			irFileName = name + "_" + irFileName
		}
	}

	fileInfo, err := os.Stat(srcFilePath)
	if err != nil {
		// error handling
	}

	if fileInfo.IsDir() {
		srcFolder = srcFilePath
	}
	irFilePath := filepath.Join(srcFolder, irFileName+".ll")

	cmdline := fmt.Sprintf("%s --dump-ir -o %s ", exePath, irFilePath)
	cmdline = cmdline + normalizeCmdlineArg(srcFilePath) + " "
	for _, s := range args {
		cmdline = cmdline + normalizeCmdlineArg(s) + " "
	}

	cmdline = "(" + cmdline + ")"
	cmd := platform.Command(cmdline)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		panicGracefully(fmt.Sprintf("Failed to parse the BC file. cmdline=%s", cmdline), err, tmpDir)
	}

	return irFilePath
}

func main() {
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

	// create the coderrect working directory
	fi, _ := os.Stat(cwd)
	coderrectWorkingDir := filepath.Join(cwd, ".coderrect")

	// create the build directory
	cleanBuild := conflib.GetBool("cleanBuild", false)
	logger.Infof("Check cleanBuild flag. cleanBuild=%v", cleanBuild)
	coderrectBuildDir := filepath.Join(coderrectWorkingDir, "build")

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
	os.Setenv("CODERRECT_BUILD_DIR", coderrectBuildDir)
	os.Setenv("CODERRECT_CWD", coderrectWorkingDir)

	bcPaths := conflib.GetString("bcPaths", "")
	if len(bcPaths) > 0 {
		os.Setenv("CODERRECT_BC_PATHS", bcPaths)
	}

	// Per COD-665, we add (or replace) "-logger.logFolder" in the cmdline opts
	//
	cmdlineOpts := addOrReplaceCommandline(conflib.GetCmdlineOpts(), "logger.logFolder", logger.GetLogFolder())
	if customConfigPath := conflib.GetString("conf", ""); len(customConfigPath) > 0 {
		// cod1974, replace relative path to abs, and pass to next
		customConfigPath, _ = filepath.Abs(customConfigPath)
		cmdlineOpts = addOrReplaceCommandline(cmdlineOpts, "conf", customConfigPath)
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
	var executablePathList []string
	cmdline := ""

	if conflib.GetBool("solana.on", false) && len(remainingArgs) > 0 {
		//for solana, invoke sol-racedetect to generate IR first
		logger.Infof("Execute sol-racedetect command to generate IR files. cmdline=%v", remainingArgs)
		srcFilePath, _ := filepath.Abs(remainingArgs[0])
		var args []string
		if len(remainingArgs) > 1 {
			args = remainingArgs[1:]
		}
		// Step 1. Find all directories with the Xargo.toml file
		solanaProgramDirectories := findAllXargoTomlDirectory(srcFilePath)
		if len(solanaProgramDirectories) == 0 {
			fmt.Println("No Xargo.toml or Cargo.toml file found in:", srcFilePath)
			os.Exit(0)
		}
		// Step 2. Generate IR files for each program
		for _, programPath := range solanaProgramDirectories {
			generated := generateSolanaIR(args, programPath, tmpDir)
			executablePathList = append(executablePathList, generated)
		}
	}

	if len(executablePathList) == 0 {
		fmt.Println("")
		fmt.Println("Unable to find any executable or library to analyze.")
		exitGracefully(0, tmpDir)
	}
	fmt.Printf("executablePathList: %v\n", executablePathList)

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
	var cmd *exec.Cmd

	var executableInfoList []ExecutableInfo
	totalRaceCnt := 0
	time_start := time.Now()
	timeout, _ := time.ParseDuration(conflib.GetString("timeout", "8h"))

	for i := 0; i < len(executablePathList); i++ {
		var executableInfo ExecutableInfo

		fmt.Printf("\nAnalyzing %s ...\n", executablePathList[i])
		directBCFile := executablePathList[i]
		if len(directBCFile) == 0 {
			logger.Errorf("Invalid path to the executable. path=%s", directBCFile)
			panicGracefully("Invalid path to the executable", nil, tmpDir)
		}

		bcFile, _ := filepath.Abs(directBCFile)
		if bcFile == "" || !util.FileExists(bcFile) {
			continue // skip if BC file does not exist
		}
		logger.Infof("Analyzing BC file to detect races. bcFile=%s.bc", bcFile)

		// Step 3. Call racedetect
		executablePath := executablePathList[i]
		_, executableName := filepath.Split(executablePath)
		executableInfo.Name = executableName
		tmpJsonPath := filepath.Join(tmpDir, fmt.Sprintf("coderrect_%s_report.json", executableName))
		logger.Infof("Generating the json file. jsonPath=%s", tmpJsonPath)
		racedetect := filepath.Join(coderrectHome, "bin", "racedetect")
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
		if len(detectModeFlag) > 0 {
			cmdArgument = append(cmdArgument, detectModeFlag)
		}
		if len(detectPlanFlag) > 0 {
			cmdArgument = append(cmdArgument, detectPlanFlag)
		}
		if len(displayFlag) > 0 {
			cmdArgument = append(cmdArgument, displayFlag)
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

	// Step 4. call reporter to generate the report if there are races
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

	for i := 0; i < len(rawJsonPathList); i++ {
		os.Remove(rawJsonPathList[i])
	}
	os.RemoveAll(tmpDir)

	logger.Infof("End the session")
	//fmt.Println("Exit code: ", exitCode)
	os.Exit(exitCode)
}
