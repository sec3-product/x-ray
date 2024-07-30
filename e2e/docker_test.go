package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	cp "github.com/otiai10/copy"
)

const (
	defaultImageName = "x-ray:latest"
)

var (
	// testAppDir indicates the path in relative to the test app (i.e.
	// starting from the `e2e/` dir).
	testAppDir        = filepath.Join("..", "demo", "jet-v1")
	testAppResultJSON = []byte(`{
 "Executables": [
  {
   "Name": "libraries_cpi.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_libraries_cpi.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "programs_jet.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_programs_jet.ll.json",
   "DataRaces": 17,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "programs_test-writer.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_programs_test-writer.ll.json",
   "DataRaces": 1,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "tools_cli.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_tools_cli.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  }
 ],
 "CoderrectVer": "Unknown"
}`)

	// testAppAltResultJSON is an alternative result JSON that works around
	// an issue where x-ray may give a slightly different result (in
	// `programs_jet.ll` entry).
	testAppAltResultJSON = []byte(`{
 "Executables": [
  {
   "Name": "libraries_cpi.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_libraries_cpi.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "programs_jet.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_programs_jet.ll.json",
   "DataRaces": 20,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "programs_test-writer.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_programs_test-writer.ll.json",
   "DataRaces": 1,
   "RaceConditions": 0,
   "NewBugs": true
  },
  {
   "Name": "tools_cli.ll",
   "RaceJSON": "/workspace/.coderrect/build/raw_tools_cli.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  }
 ],
 "CoderrectVer": "Unknown"
}`)
)

func TestDockerE2E(t *testing.T) {
	imageName := os.Getenv("X_RAY_IMAGE")
	if imageName == "" {
		imageName = defaultImageName
	}

	// Cannot use `t.TempDir()` as docker won't be able to see it.
	workingDir := os.Getenv("WORKING_DIR")
	if workingDir == "" {
		t.Fatal("WORKING_DIR must be set")
	}
	t.Logf("Working directory: %v", workingDir)

	e2eDir := filepath.Join(workingDir, "e2e-app")
	if _, err := os.Stat(e2eDir); err == nil {
		// Clean up the E2E working directory, as x-ray would otherwise
		// compute delta against the previous run.
		if err := os.RemoveAll(e2eDir); err != nil {
			t.Fatalf("Failed to remove E2E working directory: %v", err)
		}
	}
	if os.Getenv("KEEP_E2E_DIR") == "" {
		t.Cleanup(func() {
			_ = os.RemoveAll(e2eDir)
		})
	} else {
		t.Logf("Will keep E2E working directory: %v", e2eDir)
	}

	if err := cp.Copy(testAppDir, e2eDir); err != nil {
		t.Fatalf("Failed to copy test app to working directory: %v", err)
	}

	// The container uses `/workspace` as the working directory, where it
	// writes the result files.
	cmd := exec.Command(
		"docker", "run",
		"--rm",
		"--user", fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()),
		"--volume", fmt.Sprintf("%s:/workspace", e2eDir),
		imageName,
		"/workspace")
	// Stream the outputs as it runs.
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to analyze the test program: %v", err)
	}

	// Assert that the result JSON is generated.
	resultJSONPath := filepath.Join(e2eDir, ".coderrect", "build", "index.json")
	if _, err := os.Stat(resultJSONPath); err != nil {
		t.Fatalf("Failed to stat result JSON at %v", resultJSONPath)
	}
	resultJSON, err := os.ReadFile(resultJSONPath)
	if err != nil {
		t.Fatalf("failed to read result JSON file: %v", err)
	}

	diff, err := compareJSONs(resultJSON, testAppResultJSON)
	if err != nil {
		t.Fatalf("Failed to compare result JSON files: %v", err)
	}
	if diff != "" {
		// TODO: Work around an issue where x-ray may give a slightly
		// different result.
		altDiff, err := compareJSONs(resultJSON, testAppAltResultJSON)
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
