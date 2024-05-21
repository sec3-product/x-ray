package bcs

import (
	"clio/internal/utils"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// Register the bc(s) under the following name
func Register(bcPath, name string, files ...string) error {
	destDir := filepath.Join(bcPath, name)
	if exists, err := utils.FileDirExists(destDir); err == nil && exists {
		return fmt.Errorf("There is already a bc registered under '%s'", name)
	}

	// Check all files exist
	for _, file := range files {
		if exists, err := utils.FileDirExists(file); err != nil || !exists {
			return fmt.Errorf("Could not read file: %s", file)
		}
		if !strings.HasSuffix(file, ".bc") && !strings.HasSuffix(file, ".ll") {
			return fmt.Errorf("file does not appear to be LLVM IR: %s", file)
		}
	}

	// Create Dest Dir
	err := os.MkdirAll(destDir, os.ModePerm)
	if err != nil {
		return fmt.Errorf("could not create dir '%s': %v", destDir, err)
	}

	// Copy files into dest
	err = nil
	for _, file := range files {
		err = utils.Copy(file, filepath.Join(destDir, filepath.Base(file)))
		if err != nil {
			err = fmt.Errorf("could not copy file '%s': %v", file, err)
			break
		}
	}

	// Delete any part way done work
	if err != nil {
		os.RemoveAll(destDir)
	}

	return err
}

// ListNames lists all names with bcs registered
func ListNames(bcPath string) ([]string, error) {
	dirs, err := ioutil.ReadDir(bcPath)
	if err != nil {
		return nil, fmt.Errorf("Could not read bcs: %v", err)
	}

	names := []string{}

	for _, d := range dirs {
		if d.IsDir() {
			names = append(names, d.Name())
		}
	}

	return names, nil
}

// ListBcs lists the bcs registred to a name
func ListBcs(bcPath, name string) ([]string, error) {
	destDir := filepath.Join(bcPath, name)
	if exists, err := utils.FileDirExists(destDir); err != nil || !exists {
		return nil, fmt.Errorf("'%s' has not been registered", name)
	}

	dirs, err := ioutil.ReadDir(destDir)
	if err != nil {
		return nil, fmt.Errorf("Could not read bcs: %v", err)
	}

	bcs := []string{}

	for _, d := range dirs {
		if !d.IsDir() {
			bcs = append(bcs, d.Name())
		}
	}

	return bcs, nil
}

// IsRegistered returns true if there are bcs reistered to the given name
func IsRegistered(bcPath, name string) bool {
	// TODO
	return true
}
