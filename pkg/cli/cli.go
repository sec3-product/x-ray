package cli

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
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
	codeParserExe   = "sol-code-parser"
	codeAnalyzerExe = "sol-code-analyzer"
	colorReset      = "\033[0m"
)

func normalizeCmdlineArg(arg string) string {
	tokens := strings.SplitN(arg, "=", 2)
	if len(tokens) == 1 {
		return arg
	}
	if strings.Contains(tokens[1], "\"") {
		return fmt.Sprintf("%s='%s'", tokens[0], tokens[1])
	}
	if strings.Contains(tokens[1], "'") || strings.Contains(tokens[1], " ") {
		return fmt.Sprintf("%s=\"%s\"", tokens[0], tokens[1])
	}
	return arg
}

func addOrReplaceCommandline(jsonOptStr, key, value string) string {
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
	b, err := json.Marshal(&cmdlineOpts)
	if err != nil {
		logger.Warnf("Failed to marshal json. str=%s, err=%v", jsonOptStr, err)
		return jsonOptStr
	}
	return string(b)
}

func getPackageVersion(home string) string {
	path := path.Join(home, "VERSION")
	buf, err := os.ReadFile(path)
	if err != nil {
		logger.Warnf("Unable to read the version file: %v", err)
		return "Unknown"
	}
	versionStr := string(buf)
	return versionStr[len("Package "):]
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

		if strings.HasSuffix(path, "soteria.ignore") ||
			strings.HasSuffix(path, "sec3.ignore") {
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

	// if path contains ignore path
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
	return pathList
}

func generateSolanaIR(args []string, homeDir, srcFile, tmpDir string) (string, error) {
	exePath := filepath.Join(homeDir, "bin", codeParserExe)
	srcFolder, irFileName := filepath.Split(srcFile)
	if len(irFileName) == 0 {
		srcFolder, irFileName = filepath.Split(strings.TrimSuffix(srcFolder, "/"))
	}
	_, name := filepath.Split(strings.TrimSuffix(srcFolder, "/"))
	if len(name) > 0 {
		irFileName = name + "_" + irFileName
	}

	fileInfo, err := os.Stat(srcFile)
	if err != nil {
		return "", fmt.Errorf("unable to stat file %q: %w", srcFile, err)
	}

	if fileInfo.IsDir() {
		srcFolder = srcFile
	}
	irFilePath := filepath.Join(srcFolder, irFileName+".ll")

	cmdline := fmt.Sprintf("%s --dump-ir -o %s ", exePath, irFilePath)
	cmdline = cmdline + normalizeCmdlineArg(srcFile) + " "
	for _, s := range args {
		cmdline = cmdline + normalizeCmdlineArg(s) + " "
	}

	cmdline = "(" + cmdline + ")"
	cmd := exec.Command("bash", "-c", cmdline)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to generate Solana IR (cmdline: %s): %w", cmdline, err)
	}

	return irFilePath, nil
}

func Start(coderrectHome string, args []string) error {
	version := getPackageVersion(coderrectHome)
	logger.Infof("Start a new session. version=%s, args=%v", version, args)
	fmt.Printf("X-Ray %s\n", version)

	// Set up working directories.
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("unable to get the current working directory: %w", err)
	}
	fi, err := os.Stat(cwd)
	if err != nil {
		return fmt.Errorf("unable to stat the current working directory: %w", err)
	}
	coderrectWorkingDir := filepath.Join(cwd, ".xray")

	// create the build directory
	coderrectBuildDir := filepath.Join(coderrectWorkingDir, "build")
	if _, err := os.Stat(coderrectBuildDir); os.IsNotExist(err) {
		logger.Infof("Create the build directory. dir=%s", coderrectBuildDir)
		if err := os.Mkdir(coderrectBuildDir, fi.Mode()); err != nil {
			return fmt.Errorf("unable to create build directory: %w", err)
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
		return fmt.Errorf("unable to create tmp directory: %w", merr)
	}
	defer func() {
		_ = os.RemoveAll(tmpDir)
	}()

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

	var fileList []string
	if conflib.GetBool("solana.on", false) && len(args) > 0 {
		logger.Infof("Execute sol-code-parser to generate IR files. cmdline=%v", args)
		srcFilePath, err := filepath.Abs(args[0])
		if err != nil {
			return fmt.Errorf("unable to get the absolute path of source file %q: %w", args[0], err)
		}

		var restArgs []string
		if len(args) > 1 {
			restArgs = args[1:]
		}

		// Step 1. Find all directories with the Xargo.toml file.
		solanaProgramDirectories := findAllXargoTomlDirectory(srcFilePath)
		if len(solanaProgramDirectories) == 0 {
			fmt.Println("No Xargo.toml or Cargo.toml file found in:", srcFilePath)
			os.Exit(0)
		}

		// Step 2. Generate IR files for each program.
		for _, programPath := range solanaProgramDirectories {
			generated, err := generateSolanaIR(restArgs, coderrectHome, programPath, tmpDir)
			if err != nil {
				logger.Fatalf("Failed to generate IR file: %v", err)
			}
			fileList = append(fileList, generated)
		}
	}

	if len(fileList) == 0 {
		log.Fatalf("Unable to find any files to analyze.")
	}
	fmt.Printf("List of paths: %v\n", fileList)

	// Assume user picks up 3 programs "prog1", "prog2", and "prog3". The
	// code below runs `sol-code-analyzer` against each of them in
	// sequence. For each program, it generates a JSON file containing the
	// detected data. For example, it genaretes prog1.json for "prog1"
	// under `.xray/build`.
	//
	// It also generates an `index.json` under `.xray/build`. Below is an
	// example.
	//
	// {
	//   "Executables": [
	//     {
	//        "Name": "prog1",
	//        "RaceJson": "path/to/prog1_report.json",
	//        "DataRaces": 2  // number of data races
	//     },
	//     ... ...
	//   ]
	// }
	//
	var executableInfoList []reporter.ExecutableInfo
	time_start := time.Now()
	timeout, _ := time.ParseDuration(conflib.GetString("timeout", "8h"))

	for i := 0; i < len(fileList); i++ {
		var executableInfo reporter.ExecutableInfo

		fmt.Printf("\nAnalyzing %s ...\n", fileList[i])
		directBCFile := fileList[i]
		if len(directBCFile) == 0 {
			logger.Errorf("Invalid path to the executable. path=%s", directBCFile)
			return fmt.Errorf("invalid path to the files: %v", directBCFile)
		}

		bcFile, _ := filepath.Abs(directBCFile)
		// skip if BC file does not exist
		if bcFile == "" {
			continue
		}
		if _, err := os.Stat(bcFile); os.IsNotExist(err) {
			continue
		}
		logger.Infof("Analyzing file to detect issues. file=%s", bcFile)

		// Step 3. Call sol-code-analyzer
		executablePath := fileList[i]
		_, executableName := filepath.Split(executablePath)
		executableInfo.Name = executableName
		tmpJsonPath := filepath.Join(tmpDir, fmt.Sprintf("coderrect_%s_report.json", executableName))
		logger.Infof("Generating the json file. jsonPath=%s", tmpJsonPath)

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
		cmdline := filepath.Join(coderrectHome, "bin", codeAnalyzerExe)
		for _, s := range cmdArgument {
			cmdline += " " + normalizeCmdlineArg(s)
		}
		cmdline = "(" + cmdline + ")"
		cmd := exec.Command("bash", "-c", cmdline)
		cmd.Stderr = os.Stderr
		cmd.Stdout = os.Stdout
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("unable to analyze file (cmdline=%s): %w", cmdArgument, err)
		}

		// For each file , generate `raw_${file}.json` under `.xray/build`.
		raceJsonBytes, err := os.ReadFile(tmpJsonPath)
		if err != nil {
			return fmt.Errorf("unable to read JSON file %q: %w", tmpJsonPath, err)
		}
		// copy the raw json file to build
		rawJsonPath := filepath.Join(coderrectBuildDir, fmt.Sprintf("raw_%s.json", executableName))
		os.WriteFile(rawJsonPath, raceJsonBytes, fi.Mode())
		executableInfo.RaceJSON = rawJsonPath

		// Check number of detected issues.
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
			fmt.Printf("Timeout (exceeded %v), skipping the rest %v executables\n", timeout, (len(fileList) - i))
			break
		}

	}

	// reset the color of terminal
	color.Reset()

	// Generate results in index.json.
	indexInfo := reporter.IndexInfo{
		Executables:  executableInfoList,
		CoderrectVer: version,
	}
	reporter.WriteIndexJSON(indexInfo, coderrectBuildDir)

	// Dump used configurations.
	configRawString := []byte(conflib.Dump())
	configJSONPath := filepath.Join(coderrectBuildDir, "configuration.json")
	if err := os.WriteFile(configJSONPath, configRawString, fi.Mode()); err != nil {
		return fmt.Errorf("unable to write configuration.json: %w", err)
	}

	// Step 4. Generate the report.
	if err := reporter.GenerateReport(coderrectBuildDir); err != nil {
		return fmt.Errorf("unable to generate report: %w", err)
	}

	logger.Infof("End the session")
	return nil
}
