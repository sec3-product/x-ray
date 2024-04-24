package main

import (
	"fmt"

	"github.com/coderrect-inc/coderrect/util/conflib"
	"github.com/coderrect-inc/coderrect/util/logger"

	//"fmt"
	"os"
	//"strings"
	"github.com/coderrect-inc/coderrect/gllvm/shared"
)

func main() {
	argAlias := map[string]string{}

	_, err := conflib.InitializeWithEnvCmdline(argAlias)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	_, err = logger.Init("coderrect-ar")
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to init logger. err=%v", err))
		os.Exit(1)
	}

	logger.Infof("Start a new session. args=%v", os.Args)

	// Parse command line
	args := os.Args
	args = args[1:]

	if len(args) > 2 {
		//this can be a bit complicated
		//skip the ar operations qc?
		//c: create the archive
		//q: append to archive without checking for replacement
		//var opt = args[0]
		var target = args[1]
		//fmt.Printf("CODERRECT ar target: %v\n", target)
		logger.Debugf("archive the target. target=%s", target)
		//if strings.Contains(opt,"q")
		{
			// FIXME: Not sure if this is intended.
			// target and args[1:] both include args[1].
			// This will cause infinite loop in on-demand linking.
			// Not sure if this is necessary for other cases.
			// Current we have a check at on-demand linking to avoid infinite loop.
			args = append([]string{"-o", target}, args[1:]...)
		}
	}

	//if !shared.FileExists(target + ".bc") {}
	exitCode := shared.Compile(args, "coderrect-ar")
	logger.Debugf("Compile done. exitCode=%d", exitCode)

	//important to pretend to look like the actual wrapped command
	os.Exit(exitCode)
}
