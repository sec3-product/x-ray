package binaries

import (
	"clio/internal/git"
	"clio/internal/utils"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

// buildaAth gets the path to the build directory given the gitPAth
func buildPath(gitPath string) string {
	return filepath.Join(gitPath, "build")
}

// gitDir Exists return true of the git directory for sure exists within the given data directory
func gitDirExists(gitPath string) bool {
	if exists, err := utils.FileDirExists(gitPath); err == nil && exists {
		return true
	}
	return false
}

// GenerateGitDir clones the git directory into the dataDir
func GenerateGitDir(gitURL, gitPath string) error {
	_, err := git.Clone(gitURL, gitPath)
	return err
}

// RunCmake creates a build directory and runs cmake
func RunCmake(gitPath string, cmakeArgs ...string) error {
	err := os.Mkdir(buildPath(gitPath), os.ModePerm)
	if err != nil {
		return fmt.Errorf("creating build dir: %v", err)
	}

	// -S and -B are not available in earlier version of cmake
	//args := []string{"-S", gitPath, "-B", buildPath(gitPath), "-G", "Ninja"}
	args := []string{"-G", "Ninja", ".."}
	args = append(args, cmakeArgs...)
	cmd := exec.Command("cmake", args...)
	cmd.Dir = buildPath(gitPath)
	out, err := cmd.CombinedOutput()
	log.Println(cmd.String() + " \n" + string(out))
	return err
}

// SetUpRepo clones and sets up build directroy if it has not already been done
func SetUpRepo(gitURL, gitPath string, cmakeArgs []string) error {
	if gitDirExists(gitPath) {
		return nil
	}
	err := GenerateGitDir(gitURL, gitPath)
	if err != nil {
		return fmt.Errorf("generating git dir: %v", err)
	}
	err = RunCmake(gitPath, cmakeArgs...)
	if err != nil {
		return fmt.Errorf("cmake: %v", err)
	}
	return nil
}

func buildAndCopy(gitPath, destPath string) error {
	newBin := destPath
	if exists, err := utils.FileDirExists(newBin); err == nil && exists {
		// Already exists
		return nil
	}

	cmd := exec.Command("ninja", "-C", buildPath(gitPath), "racedetect")
	log.Println(cmd.String())
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	cmd.Wait()
	if err != nil {
		path, commit := filepath.Split(newBin)
		os.Create(filepath.Join(path, "failed_build"))

		ioutil.WriteFile(newBin, []byte(`#!/bin/bash
echo "Failed to build"
exit 1
`), 777)
		fmt.Fprintf(os.Stderr, "failed to build %v\n", commit)
		// TODO check os create for failure
		//return fmt.Errorf("building failed: %v", err)
		return nil
	}

	oldBin := filepath.Join(buildPath(gitPath), "bin", "racedetect")

	err = os.Rename(oldBin, newBin)
	if err != nil {
		return fmt.Errorf("err copying: %v", err)
	}

	return nil

}

func generate(gitPath, binPath, commitHash string) error {
	err := os.MkdirAll(binPath, os.ModePerm)
	if err != nil {
		return fmt.Errorf("failed to create directory '%s': %v", binPath, err)
	}

	commitHash, err = git.FullCommit(gitPath, commitHash)
	if err != nil {
		return fmt.Errorf("could not find commit '%s': %v", commitHash, err)
	}

	_, err = git.Checkout(gitPath, commitHash)
	if err != nil {
		return fmt.Errorf("checkout: %v", err)
	}
	destPath := filepath.Join(binPath, commitHash)
	return buildAndCopy(gitPath, destPath)
}

// GenerateSingle generates a single binary for the given git commit hash
func GenerateSingle(gitPath, binPath, commitHash string) error {
	if !gitDirExists(gitPath) {
		return fmt.Errorf("No git directory found at '%v'", gitPath)
	}

	return generate(gitPath, binPath, commitHash)
}

// GenerateMany generates binaries for the list of commit hashes
func GenerateMany(gitPath, binPath string, commitHashes []string) error {
	if !gitDirExists(gitPath) {
		return fmt.Errorf("No git directory found at '%v'", gitPath)
	}

	for _, hash := range commitHashes {
		if err := generate(gitPath, binPath, hash); err != nil {
			return fmt.Errorf("error on commit '%s': %v", hash, err)
		}
	}
	return nil
}

// Get the path to the binary for the specified commit
func Get(binPath, commitHash string) string {
	return filepath.Join(binPath, commitHash)
}

// IsBuilt returns true if a binary for the specified commit exists
func IsBuilt(binPath, commitHash string) bool {
	path := Get(binPath, commitHash)
	exists, err := utils.FileDirExists(path)
	return err == nil && exists
}

// List lists info about each commit a binary has been built for
func List(gitPath, binPath string) ([]string, error) {
	dirs, err := ioutil.ReadDir(binPath)
	if err != nil {
		return nil, fmt.Errorf("Could not read bins from '%s': %v", binPath, err)
	}

	result := []string{}

	for _, d := range dirs {
		info, err := git.CommitString(gitPath, d.Name())
		if err != nil {
			return nil, fmt.Errorf("err gitting infor for commit '%s': %v", d.Name(), err)
		}
		result = append(result, info)
	}
	return result, nil
}

// ListBins gives path ot all generatesd binaries
func ListBins(binPath string) ([]string, error) {
	dirs, err := ioutil.ReadDir(binPath)
	if err != nil {
		return nil, fmt.Errorf("Could not read bins from '%s': %v", binPath, err)
	}

	result := []string{}

	for _, d := range dirs {
		result = append(result, filepath.Join(binPath, d.Name()))
	}
	return result, nil
}

// BuildBinsOptions wraps the arguments and options for the BuildBins func
type BuildBinsOptions struct {
	// git URL to clone
	GitURL string

	// path projet will be cloned to
	CloneDir string

	// path bins will be saved to
	BinsDir string

	// commits to be built
	Commits []string

	// args passed to cmake
	CmakeFlags []string

	// when set the cloned directory will be deleted when done
	CleanupFlag bool
}

// BuildBins is the main CLI wrapper for building binaries
func BuildBins(opts BuildBinsOptions) error {

	gitURL := opts.GitURL
	cloneDir := opts.CloneDir
	binsDir := opts.BinsDir
	commits := opts.Commits
	cmakeFlags := opts.CmakeFlags
	rmFlag := opts.CleanupFlag

	// Clone and run Cmake if git project does not already exist
	if exists, err := utils.FileDirExists(cloneDir); err == nil && exists {
		fmt.Printf("Found existing directory: %v\n", cloneDir)
		if resp, err := git.FetchAll(cloneDir); err != nil {
			fmt.Fprintf(os.Stderr, "error fetching latest commits: %v\n%s", err, string(resp))
		}
	} else {
		if err := GenerateGitDir(gitURL, cloneDir); err != nil {
			return fmt.Errorf("clone failed: %v", err)
		}

		// If we cloned and rm flag was set, delete when we are done
		if rmFlag {
			defer func() {
				fmt.Printf("Deleting %s\n", cloneDir)
				os.RemoveAll(cloneDir)
			}()
		}

		fmt.Println("cmake", cmakeFlags)

		if err := RunCmake(cloneDir, cmakeFlags...); err != nil {
			return fmt.Errorf("cmake failed: %v", err)
		}
	}

	// Transform all commits to full commit hashes
	for i, commit := range commits {
		fullCommit, err := git.FullCommit(cloneDir, commit)
		if err != nil {
			return fmt.Errorf("unable to parse commit '%v': %v", commit, err)
		}
		commits[i] = fullCommit
	}

	// Remove any already existing binaries from list of binaries to be built
	if exists, err := utils.FileDirExists(binsDir); err == nil && exists {
		if files, err := ioutil.ReadDir(binsDir); err == nil {
			for _, file := range files {
				if !file.IsDir() {
					found := -1
					name := file.Name()
					for i, commit := range commits {
						if name == commit {
							found = i
							fmt.Printf("%s already built\n", commit)
							break
						}
					}
					// Remove item from list
					if found != -1 {
						commits[len(commits)-1], commits[found] = commits[found], commits[len(commits)-1]
						commits = commits[:len(commits)-1]
					}
				}
			}
		} else {
			fmt.Fprintf(os.Stderr, "error reading bins di. Rebuilding all binaries: %v\n", err)
		}
	}

	if err := GenerateMany(cloneDir, binsDir, commits); err != nil {
		return fmt.Errorf("could not generate binaries: %v", err)
	}

	fmt.Printf("Wrote binaries to %s\n", binsDir)
	return nil

}
