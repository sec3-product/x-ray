package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/transform"

	"github.com/coderrect-inc/coderrect/gllvm/shared"
	"github.com/coderrect-inc/coderrect/util/conflib"
	"github.com/coderrect-inc/coderrect/util/logger"
	bolt "go.etcd.io/bbolt"
)

const (
	bucketNameExecFile = "execfiles"
	bucketNameBcFile   = "bcfiles"
)

func panicGracefully(msg string, err error, tmpDir string) {
	logger.Errorf("%s err=%v", msg, err)
	//os.Stderr.WriteString(fmt.Sprintf("%v\n", err))
	exitGracefully(1, tmpDir)
}

func exitGracefully(exitCode int, tmpDir string) {
	if len(tmpDir) > 0 && fileExists(tmpDir) {
		os.RemoveAll(tmpDir)
	}

	logger.Infof("End of the session. exitCode=%d", exitCode)
	os.Exit(exitCode)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func args(s string) []string {
	return strings.Split(s, string(' '))
}

func initBoldDB(boldDbFilePath string) {
	os.Remove(boldDbFilePath)
	boltDb, err := bolt.Open(boldDbFilePath, 0666, nil)
	if err != nil {
		panicGracefully("Failed to to create db.", err, "")
	}
	err = boltDb.Update(func(tx *bolt.Tx) error {
		_, err = tx.CreateBucket([]byte(bucketNameBcFile))
		if err != nil {
			return err
		}
		_, err = tx.CreateBucket([]byte(bucketNameExecFile))
		return err
	})
	if err != nil {
		panicGracefully("Failed to create a bucket in the db.", err, "")
	}
	boltDb.Close()
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*
*                     New Stuff Here
*
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/

// Similar to ioutil.ReadFile() but decodes UTF-16.  Useful when
// reading data from MS-Windows systems that generate UTF-16BE files,
// but will do the right thing if other BOMs are found.
func ReadFileUTF16(filename string) ([]byte, error) {
	// Read the file into a []byte:
	raw, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// Make an tranformer that converts MS-Win default to UTF8:
	win16be := unicode.UTF16(unicode.BigEndian, unicode.IgnoreBOM)
	// Make a transformer that is like win16be, but abides by BOM:
	utf16bom := unicode.BOMOverride(win16be.NewDecoder())

	// Make a Reader that uses utf16bom:
	unicodeReader := transform.NewReader(bytes.NewReader(raw), utf16bom)

	// decode and print:
	decoded, err := ioutil.ReadAll(unicodeReader)
	return decoded, err
}

// Split the contents of an @something.rsp file into a list of flags
func parseRSP(data string) []string {
	// Split on space unless we are in pair of "" like: "C:\Program Files\..."

	args := []string{}
	arg := ""
	for i := 0; i < len(data); i++ {
		c := data[i]
		if c == ' ' || c == '\n' || c == '\r' {
			if len(arg) > 0 {
				args = append(args, arg)
			}
			arg = ""
		} else if c == '"' {
			i++
			for data[i] != '"' {
				arg += string(data[i])
				i++
			}
		} else {
			arg += string(c)
		}
	}

	if len(arg) > 0 {
		args = append(args, arg)
	}

	return args
}

// takes path to rsp file and returns the list of flags from the file
func readRSPFile(rspfile string) ([]string, error) {
	//bytedata, err := ioutil.ReadFile(rspfile)
	bytedata, err := ReadFileUTF16(rspfile)
	if err != nil {
		return nil, fmt.Errorf("reading rsp file: '%v': %w", rspfile, err)
	}

	flags := parseRSP(string(bytedata))

	return flags, nil
}

func preprocessClangcl(args []string) (exitCode int) {

	exitCode = 0
	// Get the true clang -cc1 flags by calling 'clang-cl -### ...' then reading commands from stderr
	args = append([]string{"-###"}, args...)
	compilerExecName := shared.GetCompilerExecName("clang-cl")
	cmd := exec.Command(compilerExecName, args...)
	logger.Debugf("run cmd: %v", cmd.String())

	stderr, err := cmd.StderrPipe()
	if err != nil {
		logger.Errorf("Failed cgetting stderr: %v\n", err)
		return 1
	}

	err = cmd.Start()
	if err != nil {
		logger.Errorf("Failed to run '%s': %v", cmd.String(), err)
		return 1
	}

	output, err := ioutil.ReadAll(stderr)
	if err != nil {
		logger.Errorf("Failed to read stderr: %v", err)
		return 1
	}

	commands := strings.Split(string(output), "\n")
	for _, command := range commands {
		logger.Debugf("clang-cl stderr line: %s", command)
		if len(command) == 0 {
			continue
		}
		command := strings.TrimSpace(command)

		if !strings.HasPrefix(command, "\"") {
			continue
		}

		command = command[1 : len(command)-1]
		clangArgs := strings.Split(command, "\" \"")
		logger.Debugf("subcmd: %v", command)

		processedArgs := []string{}
		isBuildingPCH := false
		for _, clangArg := range clangArgs {

			processedArg := strings.TrimSpace(clangArg)
			processedArg = strings.ReplaceAll(processedArg, "\\\"", "\"")
			processedArg = strings.ReplaceAll(processedArg, "\\\\", "\\")
			processedArg = strings.ReplaceAll(processedArg, "\\\\", "\\")

			if processedArg == "-emit-pch" {
				// run the command instead
				isBuildingPCH = true
			}
			if len(processedArg) > 0 {
				processedArgs = append(processedArgs, processedArg)
			}
		}

		if isBuildingPCH {
			//pchBuildCmd := exec.Command("clang-cl", processedArgs[1:]...)
			//err = pchBuildCmd.Start()
			//if err != nil {
			//	logger.Fatalf("Failed to run '%s': %v", cmd.String(), err)
			//  return
			//}
		} else {
			shared.Compile(processedArgs[1:], "clang-cl")
		}
	}
	return
}

func main() {
	argAlias := map[string]string{}
	_, err := conflib.InitializeWithEnvCmdline(argAlias)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	_, err = logger.Init("coderrect-dispatcher")
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to init logger. err=%v", err))
		os.Exit(1)
	}

	logger.Infof("Start a new session. args=%v", os.Args)

	if len(os.Args) <= 1 {
		logger.Warnf(fmt.Sprintf("Invalid call"))
		os.Exit(0)
	}

	exe := os.Args[1]
	cmdlineargs := os.Args[2:]

	// Only intercept calls to CL.exe (compiler) and link.exe (linker)
	baseExe := strings.ToLower(filepath.Base(exe))
	baseExe = strings.TrimSuffix(baseExe, filepath.Ext(baseExe))
	if baseExe != "cl" && baseExe != "link" {
		// os.Exit(0)
		logger.Debugf("skip build")
		return
	}

	// Windows sometimes uses .rsp files to pass args to commands like './cmd.exe @flags.rsp'
	// an @ followed by a file means the file contains the args to be passed
	// Print them for debugging purposes
	args := []string{}
	for _, arg := range cmdlineargs {
		if arg[0] == '@' {
			rspfile := arg[1:]
			rspargs, err := readRSPFile(rspfile)
			if err != nil {
				panic(err)
			}
			args = append(args, rspargs...)
		} else {
			args = append(args, arg)
		}
	}
	logger.Debugf("Expanded Args: ")
	for i, arg := range args {
		logger.Debugf("\t%v %v", i, arg)
	}

	exitCode := 0
	if baseExe == "cl" {
		exitCode = preprocessClangcl(cmdlineargs)
	} else if baseExe == "link" {
		exitCode = shared.Compile(args, "link")
	} else {
		logger.Debugf("Skip")
	}

	logger.Infof("compile done. exitCode=%d", exitCode)

	//important to pretend to look like the actual wrapped command
	os.Exit(exitCode)
}
