// The OpenRace Reporter
// This is the main entry to generate report for OpenRace

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/coderrect-inc/coderrect/reporter"
)

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) > 1 {
		fmt.Errorf("Too many input files, only one input file is allowed")
		os.Exit(1)
	}
	// the first remaining argument is the source json
	src := args[0]

	reporter.PrintTerminalReport(src)
}
