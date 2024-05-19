package platform

import (
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
)

// Command check os type, call cmd
func Command(name string, arg ...string) *exec.Cmd {
	if runtime.GOOS == "windows" {
		data := append([]string{"/C", name}, arg...)
		cmd := exec.Command("cmd", data...)
		return cmd
	}
	// Linux
	// NOTE: This is a hack to get this function work properly on both linux and windows
	// On Windows, to run some commands (tar.exe, specifically),
	// we cannot put the entire command in a string and pass it here. (weird crash happened)
	// However, if we pass them separately -- Command(cmd, arg1, arg2, arg3),
	// then it works fine.
	// Sadly, on Linux, the situation is exactly opposite.
	// Therefore, to make this API consistent on both OS,
	// we do a join before creating the linux command to merge `name` and `arg`
	// into a big fat string.
	// Let me know if anyone has a better solution in the future
	// -- Yanze
	name = strings.Join(append([]string{name}, arg...), " ")
	data := append([]string{"-c", name})
	return exec.Command("bash", data...)
}

// IsExecutable check os type
func IsExecutable(path string) bool {
	if runtime.GOOS == "windows" {
		if strings.HasSuffix(path, ".exe") {
			return true
		}
		return false
	}

	info, err := os.Stat(path)
	if err != nil {
		return false
	}

	return (info.Mode() & 0111) != 0
}

// IsLibrary check os type
func IsLibrary(path string) bool {
	if runtime.GOOS == "windows" {
		if strings.HasSuffix(path, ".lib") || strings.HasSuffix(path, ".dll") {
			return true
		}
		return false
	}

	if strings.HasSuffix(path, ".a") || strings.HasSuffix(path, ".so") || strings.HasSuffix(path, ".rlib") {
		return true
	}

	aEs := `.+\.a[\.\d+]*$`
	if r, _ := regexp.MatchString(aEs, path); r {
		return r
	}

	soEs := `.+\.so[\.\d+]*$`
	if r, _ := regexp.MatchString(soEs, path); r {
		return r
	}

	rlibEs := `.+\.rlib[\.\d+]*$`
	if r, _ := regexp.MatchString(rlibEs, path); r {
		return r
	}

	return false
}
