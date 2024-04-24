// post-process linux shell for file move
// for example: cp src dst

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/coderrect-inc/coderrect/gllvm/shared"
	"github.com/coderrect-inc/coderrect/util/conflib"
	"github.com/coderrect-inc/coderrect/util/logger"
)

func main() {
	argAlias := map[string]string{}

	_, err := conflib.InitializeWithEnvCmdline(argAlias)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	_, err = logger.Init("coderrect-cp")
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to init logger. err=%v", err))
		os.Exit(1)
	}

	logger.Infof("Start a new cp session. args=%v", os.Args)

	// Parse command line
	args := os.Args
	// cp/mv
	op := strings.ToLower(args[1])
	args = args[2:]

	var exitCode = 0
	if op == "cp" {
		exitCode = shared.OnCopyFile(args)
	} else {
		exitCode = shared.OnMoveFile(args)
	}
	logger.Debugf("cp done. exitCode=%d", exitCode)

	//important to pretend to look like the actual wrapped command
	os.Exit(exitCode)
}
