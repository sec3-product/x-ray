package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/buger/jsonparser"
	"github.com/gookit/color"
	toml "github.com/pelletier/go-toml"

	"github.com/sec3-product/x-ray/pkg/conflib"
	"github.com/sec3-product/x-ray/pkg/logger"
	"github.com/sec3-product/x-ray/pkg/reporter"
)

const (
	colorReset = "\033[0m"
)

var usage = `
usage: xray [option]... <your build command line>

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
	exitGracefully(1, tmpDir)
}

func exitGracefully(exitCode int, tmpDir string) {
	if len(tmpDir) > 0 {
		if _, err := os.Stat(tmpDir); err == nil {
			os.RemoveAll(tmpDir)
		}
	}

	logger.Infof("End of the session. exitCode=%d", exitCode)
	os.Exit(exitCode)
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
	if buf, err := os.ReadFile(versionFilepath); err != nil {
		return "Unknown"
	} else {
		versionStr := string(buf)
		return versionStr[len("Package "):]
	}
}

func findAllXargoTomlDirectory(rootPath string) []string {
	var pathList []string
	var ignorePaths []string
	walkFunc := func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// check for malicious path
		if strings.Contains(path, ";") || strings.Contains(path, "`") || strings.Contains(path, "<") {
			logger.Infof("skip malicious path: %s", path)
			return filepath.SkipDir
		}

		if strings.HasSuffix(path, "soteria.ignore") || strings.HasSuffix(path, "sec3.ignore") {
			// ignorePaths
			bytesRead, _ := os.ReadFile(path)
			file_content := string(bytesRead)
			lines := strings.Split(file_content, "\n")
			for _, line := range lines {
				if len(line) > 0 {
					ignorePaths = append(ignorePaths, line)
				}
			}
		}

		// check folder
		if d.IsDir() {
			if strings.HasPrefix(d.Name(), ".") {
				// Skip hidden directories.
				return filepath.SkipDir
			}
			parent, _ := filepath.Split(path)
			if strings.Contains(parent, "/proptest") ||
				strings.Contains(parent, "/logs") ||
				strings.Contains(parent, "/target") ||
				strings.Contains(parent, "/docs") ||
				strings.Contains(parent, "/migrations") ||
				strings.Contains(parent, "/services") ||
				strings.Contains(parent, "/js") ||
				strings.HasSuffix(parent, "/test") {
				logger.Infof("skip path: %s", path)

				return filepath.SkipDir
			}

			// if path contains ignore path
			for _, ignore_path := range ignorePaths {
				path1 := path + "/"
				ignore_path1 := ignore_path + "/"
				if strings.HasSuffix(path, ignore_path) || strings.HasSuffix(path1, ignore_path) || strings.HasSuffix(path, ignore_path1) {
					fmt.Println("ignored path: ", path)
					return filepath.SkipDir
				}
			}
		}

		if strings.HasSuffix(path, "argo.toml") {
			//check section [dependencies] solana-program
			config, err := toml.LoadFile(path)
			if err == nil {
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
	}
	if err := filepath.WalkDir(rootPath, walkFunc); err != nil {
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
	exePath := filepath.Join(coderrectHome, "bin", "sol-code-parser")
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
	cmd := exec.Command("bash", "-c", cmdline)
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
		"v": "version",
		"h": "help",
		"o": "report.outputDir",
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
		version, err := os.ReadFile(filepath.Join(coderrectHome, "VERSION"))
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
		fmt.Println(usage)
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
	fmt.Printf("X-Ray %s\n", version)

	cwd, err := os.Getwd()
	if err != nil {
		panicGracefully("Failed to obtain the current working directory", err, "")
	}
	// Create the xray working directory.
	fi, err := os.Stat(cwd)
	if err != nil {
		panicGracefully("Failed to stat the current working directory", err, "")
	}
	coderrectWorkingDir := filepath.Join(cwd, ".xray")

	// create the build directory
	coderrectBuildDir := filepath.Join(coderrectWorkingDir, "build")
	if _, err := os.Stat(coderrectBuildDir); os.IsNotExist(err) {
		logger.Infof("Create the build directory. dir=%s", coderrectBuildDir)
		if err := os.Mkdir(coderrectBuildDir, fi.Mode()); err != nil {
			panicGracefully("Failed to create build directory.", err, "")
		}
	}

	// set the following environment variables
	//   - PATH - make sure coderrect_home/bin in the front of other folders
	//   - CODERRECT_HOME
	//   - CODERRECT_BUILD_DIR
	//   - CODERRECT_CWD
	//   - CODERRECT_TMPDIR
	//   - CODERRECT_TOKEN - the token of this run. it's used by all components
	//                       as the key (or part of the key) to retrieve diskv db
	//   - CODERRECT_CMDLINE_OPTS - command-line opts passed to child components

	// Because of COD-1456, tmp directory for analyzer have to create
	token := fmt.Sprintf("%d", time.Now().Unix())
	tmpDir := filepath.Join(coderrectWorkingDir, "tmp", "coderrect-"+token)
	os.Setenv("CODERRECT_TMPDIR", tmpDir)
	if merr := os.MkdirAll(tmpDir, os.ModePerm); merr != nil {
		panicGracefully("Failed to create tmp directory", merr, "")
	}

	os.Setenv("CODERRECT_HOME", coderrectHome)
	os.Setenv("CODERRECT_TOKEN", token)
	os.Setenv("CODERRECT_BUILD_DIR", coderrectBuildDir)
	os.Setenv("CODERRECT_CWD", coderrectWorkingDir)

	// Per COD-665, we add (or replace) "-logger.logFolder" in the cmdline
	// opts.
	cmdlineOpts := addOrReplaceCommandline(
		conflib.GetCmdlineOpts(), "logger.logFolder", logger.GetLogFolder())
	cmdlineOpts = addOrReplaceCommandline(cmdlineOpts, "cwd", cwd)

	os.Setenv("CODERRECT_CMDLINE_OPTS", cmdlineOpts)
	logger.Infof("Passed cmdline args via env. cmdlineOpts=%v", os.Getenv("CODERRECT_CMDLINE_OPTS"))

	var executablePathList []string

	if conflib.GetBool("solana.on", false) && len(remainingArgs) > 0 {
		logger.Infof("Execute sol-code-parser to generate IR files. cmdline=%v", remainingArgs)
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
	// runs sol-code-analyzer against them one by one. For each binary, we
	// generate a json file containing race data. For example, we genarete
	// prog1.json for "prog1" under .xray/build.
	//
	// We also generate index.json under .xray/build. Below is an example
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

	var executableInfoList []ExecutableInfo
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
		// skip if BC file does not exist
		if bcFile == "" {
			continue
		}
		if _, err := os.Stat(bcFile); os.IsNotExist(err) {
			continue
		}
		logger.Infof("Analyzing BC file to detect races. bcFile=%s.bc", bcFile)

		// Step 3. Call sol-code-analyzer
		executablePath := executablePathList[i]
		_, executableName := filepath.Split(executablePath)
		executableInfo.Name = executableName
		tmpJsonPath := filepath.Join(tmpDir, fmt.Sprintf("coderrect_%s_report.json", executableName))
		logger.Infof("Generating the json file. jsonPath=%s", tmpJsonPath)
		analyzer := filepath.Join(coderrectHome, "bin", "sol-code-analyzer")

		noprogress := conflib.GetBool("noprogress", false)
		displayFlag := ""
		if noprogress {
			displayFlag += "--no-progress"
		}

		var cmdArgument []string
		if len(displayFlag) > 0 {
			cmdArgument = append(cmdArgument, displayFlag)
		}
		cmdArgument = append(cmdArgument, "-o", tmpJsonPath, bcFile)
		omplibPath := filepath.Join(coderrectHome, "bin", "libomp.so")
		cmdline := fmt.Sprintf("export LD_PRELOAD=%s; %s ", omplibPath, analyzer)
		for _, s := range cmdArgument {
			cmdline = cmdline + normalizeCmdlineArg(s) + " "
		}
		cmdline = "(" + cmdline + ")"
		cmd := exec.Command("bash", "-c", cmdline)
		cmd.Stderr = os.Stderr
		cmd.Stdout = os.Stdout
		if err := cmd.Run(); err != nil {
			panicGracefully(fmt.Sprintf("Failed to parse the BC file. cmdline=%s %s", analyzer, cmdArgument), err, tmpDir)
		}

		// For each executable, generate raw_$executable.json under
		// .xray/build.
		raceJsonBytes, err := os.ReadFile(tmpJsonPath)
		if err != nil {
			panicGracefully("Unable to read json file", err, tmpDir)
		}
		// copy the raw json file to build
		rawJsonPath := filepath.Join(coderrectBuildDir, fmt.Sprintf("raw_%s.json", executableName))
		os.WriteFile(rawJsonPath, raceJsonBytes, fi.Mode())
		executableInfo.RaceJSON = rawJsonPath

		//check number of race detected
		currentRaces := 0
		raceTypes := []string{"untrustfulAccounts", "unsafeOperations", "cosplayAccounts"}
		for _, raceType := range raceTypes {
			jsonparser.ArrayEach(raceJsonBytes, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
				currentRaces++
			}, raceType)
		}
		executableInfo.DataRaces = currentRaces
		executableInfo.NewBugs = false
		executableInfoList = append(executableInfoList, executableInfo)

		time_duration := time.Since(time_start)
		if time_duration > timeout {
			fmt.Printf("Timeout (exceeded %v), skipping the rest %v executables\n", timeout, (len(executablePathList) - i))
			break
		}

	}

	// reset the color of terminal
	color.Reset()

	// Generate results in index.json.
	indexInfo := IndexInfo{
		Executables:  executableInfoList,
		CoderrectVer: version,
	}
	reporter.WriteIndexJSON(indexInfo, coderrectBuildDir)

	// Dump used configurations.
	configRawString := []byte(conflib.Dump())
	configJSONPath := filepath.Join(coderrectBuildDir, "configuration.json")
	if err := os.WriteFile(configJSONPath, configRawString, fi.Mode()); err != nil {
		logger.Errorf("Fail to create configuration.json. err=%v", err)
		panicGracefully("Failed to create configuration.json", err, tmpDir)
	}

	// Step 4. Generate the report.
	if err := reporter.GenerateReport(coderrectBuildDir); err != nil {
		panicGracefully("Failed to generate report", err, tmpDir)
	}

	for i := 0; i < len(rawJsonPathList); i++ {
		os.Remove(rawJsonPathList[i])
	}
	os.RemoveAll(tmpDir)

	logger.Infof("End the session")
}
