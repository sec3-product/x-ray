package util

import (
	"os"
	"strings"
)

func IsCppFile(file string) bool {
	return strings.HasSuffix(file, ".cpp")
}

// NOTE: this function may need to be extended (like ".hpp")
func IsSourceFile(file string) bool {
	return strings.HasSuffix(file, ".h") || strings.HasSuffix(file, ".cpp")
}

// FileExists returns true when the file specified by `path` exists in the OS
// returns false otherwise
func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// LinkFileExists returns true when the corresponding link file specified by `path` exists in the OS
// returns false otherwise
// e.g. if the path is /home/coderrect/memcached/memcached.bc
//  the corresponding link file is /home/coderrect/memcached/memcached.bc.link
func LinkFileExists(path string) bool {
	_, err := os.Stat(path + ".link")
	return err == nil
}

// use with caution
// use map if performance matters
func Contains(arr []string, elem string) bool {
	for _, a := range arr {
		if a == elem {
			return true
		}
	}
	return false
}
