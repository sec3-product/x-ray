package git

import (
	"log"
	"os/exec"
	"strings"
)

// Clone calls git clone and returns the output and error (if any) from calling the cmd
func Clone(gitURL, destDir string) ([]byte, error) {
	args := []string{"clone", "-q", gitURL}
	if destDir != "" {
		args = append(args, destDir)
	}
	cmd := exec.Command("git", args...)
	out, err := cmd.CombinedOutput()
	log.Println(cmd.String()+"\n", string(out))
	return out, err
}

// FetchAll runs git fetch --all
func FetchAll(gitPath string) ([]byte, error) {
	args := []string{"-C", gitPath, "fetch", "--all"}
	cmd := exec.Command("git", args...)
	out, err := cmd.CombinedOutput()
	log.Println(cmd.String()+"\n", string(out))
	return out, err
}

// Checkout calls git checkout branch on the specificed git directory
func Checkout(gitPath, branch string) ([]byte, error) {
	args := []string{"-C", gitPath, "checkout", branch}
	cmd := exec.Command("git", args...)
	out, err := cmd.CombinedOutput()
	log.Println(cmd.String()+"\n", string(out))
	return out, err
}

// FullCommit gets the full length commit hash from a partial commit hash
func FullCommit(gitPath, commitHash string) (string, error) {
	args := []string{"-C", gitPath, "rev-parse", commitHash}
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimRight(string(out), "\n"), err
}

// CommitString gives the short form of the commit id and the commit message
func CommitString(gitPath, commitHash string) (string, error) {
	out, err := exec.Command("git", "-C", gitPath, "rev-list", commitHash, "--oneline", "--max-count=1").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimRight(string(out), "\n"), nil
}
