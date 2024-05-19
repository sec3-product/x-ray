package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"clio/internal/executor"
)

const helpMsg = `usage: binrunner [--help] [--out <dir>] [--max-parallel <n>] [--timeout <s>] 
               <bin> [<bin2>, ...] -- <file.bc> [<file2.bc>, ...]

binrunner runs one or more binaries on one or more .bc files and records the results
E.x.
    binrunner --out ./myreports racedetect-dev racedetect-mine -- lulesh.bc


--out          directory to save results to (default ./reports)
--max-parallel maximum number of jobs to run at once (default 128)
--timeout      max time to let a job run in seconds (default 600s)

**Note: It is up to you to make sure your system is capable if running the specificed number
        of parallel jobs. If you run out of memory while running many jobs, the processes will
        be killed and recorded as crashes.
`

func main() {
	helpFlag := flag.Bool("help", false, "display help msg")
	outDir := flag.String("out", "./reports", "directory to store reports in")
	maxParallel := flag.Int("max-parallel", 128, "max number of jobs to run in parallel")
	timeout := flag.Int("timeout", 600, "max time to allow job to run")
	flag.Parse()

	if len(flag.Args()) < 1 || *helpFlag {
		fmt.Println(helpMsg)
		return
	}

	// Split args into binaries and bc files
	binaries := []string{}
	bcFiles := []string{}
	for i, val := range flag.Args() {
		if val == "--" {
			binaries = flag.Args()[:i]
			bcFiles = flag.Args()[i+1:]
			break
		}
	}
	if len(binaries) == 0 || len(bcFiles) == 0 {
		fmt.Fprintf(os.Stderr, "requires atleast one binary and bc file\n")
		fmt.Fprintf(os.Stderr, "binaries: %v\n", binaries)
		fmt.Fprintf(os.Stderr, "bc files: %v\n", bcFiles)
		os.Exit(1)
	}

	// Each bc needs a unique name
	bcMap := map[string]string{}
	for _, bcPath := range bcFiles {
		// for now use filename as identifier
		base := filepath.Base(bcPath)
		// If not already in map, directly add
		if otherPath, exists := bcMap[base]; !exists {
			bcMap[base] = bcPath
		} else {
			// Handle this case later
			fmt.Fprintf(os.Stderr, "Currently cannot handle two bcs with same name\n\t'%v' and '%v'\n", otherPath, bcPath)
			os.Exit(1)
		}
	}

	err := executor.GenerateReports(executor.GenerateReportsOpts{
		Binaries:    binaries,
		BcFiles:     bcMap,
		OutDir:      *outDir,
		MaxParallel: *maxParallel,
		Timeout:     *timeout,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v", err)
		os.Exit(1)
	}
}
