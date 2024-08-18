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
	xrayExecutable   = "xray"
	defaultImageName = "x-ray:latest"

	// The container uses `/workspace` as the working directory, where it
	// writes the result files.
	workspaceInContainer = "/workspace"
	e2eAppDirName        = "e2e-app"
)

var (
	// testAppDirDefault indicates the relative path to the test app (i.e.
	// starting from the `e2e/` dir).
	testAppDirDefault = filepath.Join("..", "workspace", "dexterity")

	// testAppResultJSON contains the expected result JSON for the test app
	// (i.e. dexterity). It contains a PREFIX placeholder that will be
	// replaced with the actual path.
	testAppResultJSON = `{
 "Executables": [
  {
   "Name": "e2e-app_dex.ll",
   "RaceJSON": "PREFIX/.xray/build/raw_e2e-app_dex.ll.json",
   "DataRaces": 63,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "e2e-app_dummy-oracle.ll",
   "RaceJSON": "PREFIX/.xray/build/raw_e2e-app_dummy-oracle.ll.json",
   "DataRaces": 0,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "fees_constant-fees.ll",
   "RaceJSON": "PREFIX/.xray/build/raw_fees_constant-fees.ll.json",
   "DataRaces": 1,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "e2e-app_instruments.ll",
   "RaceJSON": "PREFIX/.xray/build/raw_e2e-app_instruments.ll.json",
   "DataRaces": 4,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "risk_alpha-risk-engine.ll",
   "RaceJSON": "PREFIX/.xray/build/raw_risk_alpha-risk-engine.ll.json",
   "DataRaces": 5,
   "RaceConditions": 0,
   "NewBugs": false
  },
  {
   "Name": "risk_noop-risk-engine.ll",
   "RaceJSON": "PREFIX/.xray/build/raw_risk_noop-risk-engine.ll.json",
   "DataRaces": 5,
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

	// Ignore the returned E2E app dir as we don't need it.
	_ = prepareTestApp(t, workingDir)
	cmd := exec.Command(
		"docker", "run",
		"--rm",
		"--user", fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()),
		"--volume", fmt.Sprintf("%s:%s", workingDir, workspaceInContainer),
		imageName,
		filepath.Join(workspaceInContainer, e2eAppDirName))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to analyze the test program: %v", err)
	}

	verifyResultJSON(t, workingDir, workspaceInContainer)
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
	} else {
		if !filepath.IsAbs(workingDir) {
			t.Fatalf("WORKING_DIR must be an absolute path: %v", workingDir)
		}
	}

	e2eDir := prepareTestApp(t, workingDir)
	cmd := exec.Command(execPath, e2eDir)
	cmd.Dir = e2eDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to analyze the test program: %v", err)
	}

	verifyResultJSON(t, e2eDir, e2eDir)
}

func prepareTestApp(t *testing.T, workingDir string) string {
	e2eDir := filepath.Join(workingDir, e2eAppDirName)
	t.Logf("E2E working directory: %v", e2eDir)

	if _, err := os.Stat(e2eDir); err == nil {
		// Clean up the E2E working directory, as x-ray would otherwise
		// compute delta against the previous run.
		if err := os.RemoveAll(e2eDir); err != nil {
			t.Fatalf("Failed to remove E2E working directory: %v", err)
		}
	}

	testAppDir := os.Getenv("E2E_TEST_APP")
	if testAppDir == "" {
		testAppDir = testAppDirDefault
	}
	if _, err := os.Stat(testAppDir); err != nil {
		t.Fatalf("Unable to access test app directory %q: %v", testAppDir, err)
	}

	if os.Getenv("E2E_KEEP") != "" {
		if os.Getenv("WORKING_DIR") == "" {
			t.Fatalf("Must define WORKING_DIR when E2E_KEEP is set")
		}
		t.Logf("E2E_KEEP set; will keep E2E working directory: %v", e2eDir)
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
	path := filepath.Join(pathOnHost, ".xray", "build", "index.json")
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
		t.Errorf("Found unexpected results (-want,+got):\n%v", diff)
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
