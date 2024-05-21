package executor

import (
	"clio/internal/bcs"
	"clio/internal/utils"
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type workload struct {
	binPath, bcPath, targetDir string
}

func getBcName(bcPath string) string {
	filename := filepath.Base(bcPath)
	extension := filepath.Ext(filename)
	return filename[:len(filename)-len(extension)]
}

// pctMemAvail returns the percentage of memory availabble on the system
func pctMemAvail() float64 {
	data, err := ioutil.ReadFile("/proc/meminfo")
	if err != nil {
		// TODO
		panic(err)
	}
	splits := strings.SplitAfterN(string(data), "\n", 3)
	totall := strings.Fields(splits[0])
	totals := totall[len(totall)-2]
	freel := strings.Fields(splits[1])
	frees := freel[len(freel)-2]

	total, err := strconv.Atoi(totals)
	free, err := strconv.Atoi(frees)
	if err != nil {
		panic(err)
	}

	return float64(free) / float64(total)
}

func binHasFlag(binPath, flag string) bool {
	output, err := exec.Command(binPath, "--help").Output()
	if err != nil {
		return false
	}

	return strings.Contains(string(output), flag)
}

func parallelWorker(ctx context.Context, work workload, done chan bool) {
	binPath := work.binPath
	bcPath := work.bcPath
	targetDir := work.targetDir
	os.MkdirAll(targetDir, 0777)

	tryFlags := []string{"--no-filter", "--no-limit", "--nolimit", "--no-av", "--no-ov", "--no-progress"}
	args := make([]string, 0, len(tryFlags))
	for _, flag := range tryFlags {
		if binHasFlag(binPath, flag) {
			args = append(args, flag)
		}
	}
	args = append(args, bcPath)

	cmdctx, cancel := context.WithCancel(context.Background())
	innerdone := make(chan bool)
	go func() {
		cmd := exec.CommandContext(cmdctx, binPath, args...)
		cmd.Dir = targetDir

		output, err := cmd.CombinedOutput()
		if err != nil {
			errFile := filepath.Join(targetDir, "error.txt")
			err = ioutil.WriteFile(errFile, []byte(err.Error()), 0666)
			if err != nil {
				return
			}
		}

		logFile := filepath.Join(targetDir, "output.txt")
		logData := []byte(cmd.String() + "\n")
		if output != nil {
			logData = append(logData, output...)
		}
		ioutil.WriteFile(logFile, logData, 0666)
		innerdone <- true
	}()

	working := true
	for working {
		select {
		case <-ctx.Done():
			cancel()
			// Delete any partially done work
			os.RemoveAll(work.targetDir)
			return
		case <-innerdone:
			working = false
			break
		}
	}

	cancel()
	done <- true
	return
}

// GenReports generates races.json reports for all the bin bc pairs
func GenReports(reportPath, bcDir string, binPaths, names []string) error {
	reportPath, err := filepath.Abs(reportPath)
	if err != nil {
		return fmt.Errorf("could not get abs path for '%s': %v", reportPath, err)
	}
	work := map[*workload]string{}
	for _, name := range names {
		bcsPaths, err := bcs.ListBcs(bcDir, name)
		if err != nil {
			return fmt.Errorf("could not get bcs for '%s': %v", name, err)
		}
		for _, bcPath := range bcsPaths {
			bcPath, err = filepath.Abs(filepath.Join(bcDir, name, bcPath))
			if err != nil {
				return fmt.Errorf("could not get abs path to bc '%s': %v", bcPath, err)
			}
			for _, binPath := range binPaths {
				binPath, err = filepath.Abs(binPath)
				if err != nil {
					return fmt.Errorf("could not get abs path to bin '%s': %v", binPath, err)
				}
				targetDir := filepath.Join(reportPath, name, getBcName(bcPath), filepath.Base(binPath))

				if exists, err := utils.FileDirExists(targetDir); err != nil || !exists {
					load := &workload{
						binPath:   binPath,
						bcPath:    bcPath,
						targetDir: targetDir,
					}
					work[load] = "stop"
				}
			}
		}
	}

	totalCounter := len(work)
	doneCounter := 0
	runningCounter := 0

	done := make(chan bool)
	cancels := map[*workload]context.CancelFunc{}
	running := make([]*workload, 0)
	threadBudget := 32
	memThresh := 0.20

	lastDelete := time.Now()

	for doneCounter != totalCounter {
		select {
		case <-done:
			doneCounter++
			runningCounter--
		default:
			// skip
		}

		freeMem := pctMemAvail()
		fmt.Printf("\r(%d/%d) Done - (%d/%d) parallel jobs - %f mem free", doneCounter, totalCounter, runningCounter, threadBudget, freeMem)

		if freeMem > memThresh || runningCounter == 0 {
			if runningCounter < threadBudget {
				// Spawn new work
				for load, status := range work {
					if status == "stop" {
						ctx, cancel := context.WithCancel(context.Background())
						cancels[load] = cancel
						go parallelWorker(ctx, *load, done)
						work[load] = "run"
						runningCounter++
						running = append(running, load)
						break
					}
				}
			}
		} else if runningCounter > 1 && time.Since(lastDelete) > time.Second {
			// Kill most recently start job
			killit := running[len(running)-1]
			cancels[killit]()
			work[killit] = "stop"
			running = running[:len(running)-1]
			runningCounter--
			//reduce thread budget
			threadBudget = runningCounter

			// Wait a second for memory pressure to go down before deleting more
			lastDelete = time.Now()
		}
	}

	return nil
}

// ListCommits lists all commits that have been run for the named project
func ListCommits(reportPath, name string) ([]string, error) {
	return nil, nil
}

type job struct {
	binary    string
	bcfile    string
	outputDir string
	timeout   int
}

func (j *job) run() {
	notify := make(chan *job)
	j.runWithContext(context.Background(), notify)
	_ = <-notify
}

func (j *job) runWithContext(parentctx context.Context, notify chan *job) {
	fmt.Println("Called")
	// Determine which flags are available
	tryFlags := []string{"--no-filter", "--no-limit", "--nolimit", "--no-av", "--no-ov", "--no-progress"}
	args := make([]string, 0, len(tryFlags))
	for _, flag := range tryFlags {
		if binHasFlag(j.binary, flag) {
			args = append(args, flag)
		}
	}
	args = append(args, j.bcfile)

	// Create timout / cancel context
	var ctx context.Context
	//var cancel context.CancelFunc
	//if j.timeout > 0 {
	//	ctx, cancel = context.WithTimeout(parentctx, time.Duration(j.timeout)*time.Second)
	//} else {
	ctx, _ = context.WithCancel(parentctx)
	//}

	// Setup and run command
	cmd := exec.CommandContext(ctx, j.binary, args...)
	cmd.Dir = j.outputDir
	fmt.Println(cmd.String())
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(output))
	//cancel()

	// If command failed log output to error file
	if err != nil {
		errFile := filepath.Join(j.outputDir, "error.txt")
		err = ioutil.WriteFile(errFile, []byte(err.Error()), 0666)
		if err != nil {
			return
		}
	}

	// Else log output to output file
	logFile := filepath.Join(j.outputDir, "output.txt")
	logData := []byte(cmd.String() + "\n")
	if output != nil {
		logData = append(logData, output...)
	}
	ioutil.WriteFile(logFile, logData, 0666)

	// If running in parallel, let them know we are done
	if notify != nil {
		notify <- j
	}
}

// GenerateReportsOpts wraps the arguments to the GenerateReports function
type GenerateReportsOpts struct {
	// Paths to all the binaries
	Binaries []string

	// map of names to bc file paths
	BcFiles map[string]string

	// Directory to save results to
	OutDir string

	// Maximum number of jobs to run in parallel
	MaxParallel int

	// max time to allow job to run (0 for unlimited)
	Timeout int
}

// GenerateReports is the CLI wrapper to run a number of binaries on a number of bc files
func GenerateReports(opts GenerateReportsOpts) error {

	// Build list of jobs to be done
	jobs := make([]*job, 0, len(opts.BcFiles)*len(opts.Binaries))
	for _, bin := range opts.Binaries {
		binPath, err := filepath.Abs(bin)
		if err != nil {
			return fmt.Errorf("could not get abs path to bin '%v': %v", bin, err)
		}
		for name, bc := range opts.BcFiles {
			bcPath, err := filepath.Abs(bc)
			if err != nil {
				return fmt.Errorf("could not get abs path to bc '%v': %v", bc, err)
			}

			outDir, err := filepath.Abs(filepath.Join(opts.OutDir, name, bin))
			if err != nil {
				return fmt.Errorf("could not get abs path to outdir '%v': %v", filepath.Join(opts.OutDir, name, bin), err)
			}

			if exists, err := utils.FileDirExists(outDir); err != nil || !exists {
				jobs = append(jobs, &job{
					binary:    binPath,
					bcfile:    bcPath,
					timeout:   opts.Timeout,
					outputDir: outDir,
				})
			} else {
				fmt.Printf("Skipping %v on %v. See '%v'\n", filepath.Base(bin), filepath.Base(bc), outDir)
			}

		}
	}

	total := len(jobs)
	if opts.MaxParallel == 1 {
		fmt.Printf("(0/%v)\n", total)
		for i, j := range jobs {
			fmt.Printf("(%v/%v) %v on %v\n", i+1, total, filepath.Base(j.binary), filepath.Base(j.bcfile))
			// Create output directory
			if err := os.MkdirAll(j.outputDir, 0777); err != nil {
				fmt.Fprintf(os.Stderr, "Could not create dir: '%v': %v\n", j.outputDir, err)
				continue
			}
			j.run()
		}
		fmt.Printf("(%v/%v) Done\n", total, total)
		return nil
	}

	type JobStatus int
	const (
		Stopped JobStatus = iota
		Running JobStatus = iota
		Done    JobStatus = iota
		Error   JobStatus = iota
	)

	parallelBudget := opts.MaxParallel
	jobStatus := map[*job]JobStatus{}
	for _, j := range jobs {
		jobStatus[j] = Stopped
	}
	jobCancels := map[*job]context.CancelFunc{}
	runningCounter := 0
	doneCounter := 0

	notify := make(chan *job)

	for doneCounter != total {
		select {
		case j := <-notify:
			doneCounter++
			runningCounter--
			fmt.Printf("(%v Running) (%v/%v Done) Completed %v on %v\n", runningCounter, doneCounter, total, filepath.Base(j.binary), filepath.Base(j.bcfile))
		default:
			//skip
		}

		if runningCounter < parallelBudget {
			// Find Stopped Job
			for j, status := range jobStatus {
				if status == Stopped {
					// Create output directory
					if err := os.MkdirAll(j.outputDir, 0777); err != nil {
						fmt.Fprintf(os.Stderr, "Could not create dir: '%v': %v\n", j.outputDir, err)
						jobStatus[j] = Error
						doneCounter++
						continue
					}
					ctx, cancel := context.WithCancel(context.Background())
					go j.runWithContext(ctx, notify)
					jobStatus[j] = Running
					jobCancels[j] = cancel
					runningCounter++
					fmt.Printf("(%v Running) (%v/%v Done) Started %v on %v\n", runningCounter, doneCounter, total, filepath.Base(j.binary), filepath.Base(j.bcfile))
					break
				}
			}
		}
	}

	return nil
}
