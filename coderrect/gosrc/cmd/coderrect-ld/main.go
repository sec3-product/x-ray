package main

import (
	"fmt"
	"os"

	"github.com/coderrect-inc/coderrect/util/conflib"
	"github.com/coderrect-inc/coderrect/util/logger"

	"github.com/coderrect-inc/coderrect/gllvm/shared"
)

func main() {
	argAlias := map[string]string{}

	_, err := conflib.InitializeWithEnvCmdline(argAlias)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	_, err = logger.Init("coderrect-ld")
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to init logger. err=%v", err))
		os.Exit(1)
	}

	logger.Infof("Start a new session. args=%v", os.Args)

	// Parse command line
	args := os.Args
	args = args[1:]
	//if !shared.FileExists(target + ".bc") {}
	exitCode := shared.Compile(args, "coderrect-ld")
	logger.Debugf("Compile done. exitCode=%d", exitCode)

	//important to pretend to look like the actual wrapped command
	os.Exit(exitCode)
}
