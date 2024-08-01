package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	cp "github.com/otiai10/copy"
)

const (
	xrayExecutable   = "coderrect"
	defaultImageName = "x-ray:latest"

	// The container uses `/workspace` as the working directory, where it
	// writes the result files.
	workspaceInContainer = "/workspace"
)

var (
	// testAppDir indicates the path in relative to the test app (i.e.
	// starting from the `e2e/` dir).
	testAppDir = filepath.Join("..", "demo", "jet-v1")

	// testAppResultJSON contains the expected result JSON for the test app
	// (i.e. jet-v1). It contains a PREFIX placeholder that will be
	// replaced with the actual path.
	testAppResultJSON = `{
 "Executables": [
  {
   "Name": "libraries_cpi.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_libraries_cpi.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "programs_jet.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_programs_jet.ll.json",
   "DataRaces": 17,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "programs_test-writer.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_programs_test-writer.ll.json",
   "DataRaces": 1,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "tools_cli.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_tools_cli.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  }
 ],
 "CoderrectVer": "Unknown"
}`

	// testAppAltResultJSON is an alternative result JSON that works around
	// an issue where x-ray may give a slightly different result (in
	// `programs_jet.ll` entry).
	testAppAltResultJSON = `{
 "Executables": [
  {
   "Name": "libraries_cpi.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_libraries_cpi.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "programs_jet.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_programs_jet.ll.json",
   "DataRaces": 20,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "programs_test-writer.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_programs_test-writer.ll.json",
   "DataRaces": 1,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "tools_cli.ll",
   "RaceJSON": "PREFIX/.coderrect/build/raw_tools_cli.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  }
 ],
 "CoderrectVer": "Unknown"
}`
)

func TestContainerE2E(t *testing.T) {
	imageName := os.Getenv("X_RAY_IMAGE")
	if imageName == "" {
		imageName = defaultImageName
	}

	// Cannot use `t.TempDir()` as docker won't be able to see the temp dir
	// on host.
	workingDir := os.Getenv("WORKING_DIR")
	if workingDir == "" {
		t.Log("WORKING_DIR not set; will use the current directory")
		wd, err := os.Getwd()
		if err != nil {
			t.Fatalf("Failed to get the current working directory: %v", err)
		}
		workingDir = wd
	}

	e2eDir := prepareTestApp(t, workingDir, "e2e-app-container")
	cmd := exec.Command(
		"docker", "run",
		"--rm",
		"--user", fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()),
		"--volume", fmt.Sprintf("%s:%s", e2eDir, workspaceInContainer),
		imageName,
		workspaceInContainer)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to analyze the test program: %v", err)
	}

	verifyResultJSON(t, e2eDir, workspaceInContainer)
}

func TestNativeE2E(t *testing.T) {
	execPath, err := exec.LookPath(xrayExecutable)
	if err != nil {
		t.Fatalf("Failed to find %v executable: %v", xrayExecutable, err)
	}
	t.Logf("Found %v executable at %v", xrayExecutable, execPath)

	workingDir := os.Getenv("WORKING_DIR")
	if workingDir == "" {
		t.Log("WORKING_DIR not set; will use temp dir")
		workingDir = t.TempDir()
	}

	e2eDir := prepareTestApp(t, workingDir, "e2e-app-native")
	cmd := exec.Command(execPath, e2eDir)
	cmd.Dir = e2eDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to analyze the test program: %v", err)
	}

	verifyResultJSON(t, e2eDir, e2eDir)
}

func prepareTestApp(t *testing.T, workingDir, e2eDirSuffix string) string {
	e2eDir := filepath.Join(workingDir, e2eDirSuffix)
	t.Logf("E2E working directory: %v", e2eDir)

	if _, err := os.Stat(e2eDir); err == nil {
		// Clean up the E2E working directory, as x-ray would otherwise
		// compute delta against the previous run.
		if err := os.RemoveAll(e2eDir); err != nil {
			t.Fatalf("Failed to remove E2E working directory: %v", err)
		}
	}
	if os.Getenv("KEEP_E2E") != "" {
		t.Logf("KEEP_E2E set; will keep E2E working directory: %v", e2eDir)
	} else {
		t.Cleanup(func() {
			_ = os.RemoveAll(e2eDir)
		})
	}

	if err := cp.Copy(testAppDir, e2eDir); err != nil {
		t.Fatalf("Failed to copy test app to working directory: %v", err)
	}

	return e2eDir
}

func verifyResultJSON(t *testing.T, pathOnHost, pathInTest string) {
	path := filepath.Join(pathOnHost, ".coderrect", "build", "index.json")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("Failed to stat result JSON at %v", path)
	}
	resultJSON, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read result JSON file: %v", err)
	}

	want := strings.ReplaceAll(testAppResultJSON, "PREFIX", pathInTest)
	diff, err := compareJSONs(resultJSON, []byte(want))
	if err != nil {
		t.Fatalf("Failed to compare result JSON files: %v", err)
	}
	if diff != "" {
		// TODO: Work around an issue where x-ray may give a slightly
		// different result.
		want := strings.ReplaceAll(testAppAltResultJSON, "PREFIX", pathInTest)
		altDiff, err := compareJSONs(resultJSON, []byte(want))
		if err != nil {
			t.Fatalf("Failed to compare alternative result JSON files: %v", err)
		}
		if altDiff != "" {
			t.Errorf("Found unexpected results (-want,+got):\n%v", altDiff)
		}
	}
}

func compareJSONs(gotJSON, wantJSON []byte) (string, error) {
	var got any
	if err := json.Unmarshal(gotJSON, &got); err != nil {
		return "", fmt.Errorf("failed to unmarshal got JSON: %v", err)
	}
	var want any
	if err := json.Unmarshal(wantJSON, &want); err != nil {
		return "", fmt.Errorf("failed to unmarshal want JSON: %v", err)
	}
	return cmp.Diff(want, got), nil
}
