package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/sec3-product/x-ray/pkg/cli"
	"github.com/sec3-product/x-ray/pkg/conflib"
	"github.com/sec3-product/x-ray/pkg/logger"
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

Examples:

1. Detect vulnerabilities in helloworld
$ cd path/to/examples/helloworld && xray .

2. Detect vulnerabilities in all solana-program-library projects
$ xray path/to/solana-program-library
`

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
		fmt.Printf("Failed to initialize configuration: %v\n", err)
		os.Exit(1)
	}
	if _, err := logger.InitWithLogRotation("xray"); err != nil {
		os.Exit(1)
	}

	coderrectHome, err := conflib.GetCoderrectHome()
	if err != nil {
		logger.Fatalf("Unable to find xray home - %v", err)
	}

	// show versions of all components
	if conflib.GetBool("version", false) || conflib.GetBool("-version", false) {
		version, err := os.ReadFile(filepath.Join(coderrectHome, "VERSION"))
		if err != nil {
			logger.Fatalf("Unable to read %s/VERSION - %v\n", coderrectHome, err)
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

	if err := cli.Start(coderrectHome, remainingArgs); err != nil {
		logger.Fatalf("Failed to run xray: %v", err)
	}
}
