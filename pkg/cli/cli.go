package cli

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
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

func Start(coderrectHome, version string, args []string) error {
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
	//        "ReportJSON": "path/to/prog1_report.json",
	//        "IssueCount": 2  // number of issues
	//     },
	//     ... ...
	//   ]
	// }
	//
	var executableInfoList []reporter.ExecutableInfo

	timeout, _ := time.ParseDuration(conflib.GetString("timeout", "8h"))
	startTime := time.Now()

	for i, filePath := range fileList {
		fmt.Printf("\nAnalyzing %s ...\n", filePath)
		if len(filePath) == 0 {
			logger.Errorf("Invalid path to the executable. path=%s", filePath)
			return fmt.Errorf("invalid file path: %q", filePath)
		}

		irFile, err := filepath.Abs(filePath)
		if err != nil {
			return fmt.Errorf("unable to get the absolute path of file %q: %w", filePath, err)
		}
		// Skip if the file does not exist.
		if _, err := os.Stat(irFile); os.IsNotExist(err) {
			logger.Warnf("File %q does not exist; skipped", irFile)
			continue
		}
		logger.Infof("Analyzing IR file %q for issues...", irFile)

		// Step 3. Call sol-code-analyzer to analyze the file.
		filename := filepath.Base(filePath)
		tmpJSONPath := filepath.Join(tmpDir, fmt.Sprintf("xray_%s_report.json", filename))
		logger.Infof("Generating the JSON file. jsonPath=%s", tmpJSONPath)

		noprogress := conflib.GetBool("noprogress", false)
		displayFlag := ""
		if noprogress {
			displayFlag += "--no-progress"
		}

		var cmdArgument []string
		if len(displayFlag) > 0 {
			cmdArgument = append(cmdArgument, displayFlag)
		}
		cmdArgument = append(cmdArgument, "-o", tmpJSONPath, irFile)
		analyzer := filepath.Join(coderrectHome, "bin", codeAnalyzerExe)
		libompPath := filepath.Join(coderrectHome, "bin", "libomp.so")
		cmdline := fmt.Sprintf("export LD_PRELOAD=%s; %s", libompPath, analyzer)
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

		// For each file, generate `raw_${file}.json` under `.xray/build`.
		rawJSON, err := os.ReadFile(tmpJSONPath)
		if err != nil {
			return fmt.Errorf("unable to read JSON file %q: %w", tmpJSONPath, err)
		}
		// Copy the raw JSON file to build/ directory.
		rawJSONPath := filepath.Join(coderrectBuildDir, fmt.Sprintf("raw_%s.json", filename))
		if err := os.WriteFile(rawJSONPath, rawJSON, fi.Mode()); err != nil {
			return fmt.Errorf("unable to write JSON file %q: %w", rawJSONPath, err)
		}

		// Check number of detected issues.
		issues := 0
		for _, ty := range []string{"untrustfulAccounts", "unsafeOperations", "cosplayAccounts"} {
			jsonparser.ArrayEach(rawJSON, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
				issues++
			}, ty)
		}
		executableInfoList = append(executableInfoList,
			reporter.ExecutableInfo{
				Name:       filename,
				ReportJSON: rawJSONPath,
				IssueCount: issues,
			})

		// TODO: This should be implemented via context.
		if time.Since(startTime) > timeout {
			fmt.Printf("Exceeded overall timeout (%v), skipping the rest %v files\n", timeout, (len(fileList) - i))
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
