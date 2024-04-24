package logger

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"
)

func TestInit(t *testing.T) {
	log, err := Init("unittest")
	if err != nil {
		t.Errorf("Failed to init logger. err=%v", err)
	}
	log.Infof("hello")
	Errorf("hello %d", 1)
}

func TestGetLogList(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Errorf("Getwd() failed. err=%v", err)
	}
	logFolder := fmt.Sprintf("%s/.coderrect/logs", cwd)

	if dirExists(logFolder) {
		if err := os.RemoveAll(logFolder); err != nil {
			t.Errorf("Failed to remove log folder. logFolder=%s", logFolder)
		}
	}
	if !dirExists(logFolder) {
		if err := os.MkdirAll(logFolder, getFileMode()); err != nil {
			t.Errorf("Failed to create testing log folder. logFolder=%s", logFolder)
		}
	}

	// case 1 - an empty log folder
	fileList, err := getLogList(logFolder)
	if err != nil {
		t.Errorf("getLogList() failed. logFolder=%s, err=%v", logFolder, err)
	}
	if len(fileList) != 0 {
		t.Errorf("the length of fileList should be 0. fileList=%v", fileList)
	}

	// case 2 - an log folder with only log.current
	createFile(fmt.Sprintf("%s/log.current", logFolder))
	fileList, err = getLogList(logFolder)
	if err != nil {
		t.Errorf("getLogList() failed. logFolder=%s, err=%v", logFolder, err)
	}
	if len(fileList) != 0 {
		t.Errorf("the length of fileList should be 0. fileList=%v", fileList)
	}

	// case 3 - log.0 and log.current
	createFile(fmt.Sprintf("%s/log.0", logFolder))
	fileList, err = getLogList(logFolder)
	if err != nil {
		t.Errorf("getLogList() failed. logFolder=%s, err=%v", logFolder, err)
	}
	if len(fileList) != 1 {
		t.Errorf("the length of fileList should be 1. fileList=%v", fileList)
	}

	// case 4 - log.0, log.1, and log.current
	createFile(fmt.Sprintf("%s/log.1", logFolder))
	fileList, err = getLogList(logFolder)
	if err != nil {
		t.Errorf("getLogList() failed. logFolder=%s, err=%v", logFolder, err)
	}
	if len(fileList) != 2 {
		t.Errorf("the length of fileList should be 2. fileList=%v", fileList)
	}
}

func TestGetOrd(t *testing.T) {
	i := getOrd("abc/log.1")
	if i != 1 {
		t.Errorf("Failed to extract ord, str=%s", "abc/log.1")
	}
}

func TestRotateLogWhenRotateOne(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Errorf("Getwd() failed. err=%v", err)
	}
	logFolder := fmt.Sprintf("%s/.coderrect/logs", cwd)

	createBigFile(logFolder+"/log.current", 2*1024*1024)
	if err := rotateLogs(logFolder, 2, 1*1024*1024); err != nil {
		t.Errorf("Failed to rotate en empty folder. logFolder=%s", logFolder)
	}
	if fileExists(logFolder + "/log.current") {
		t.Errorf("log.current is still there")
	}
	if !fileExists(logFolder + "/log.0") {
		t.Errorf("log.0 wasn't created")
	}
}

func TestRotateLogWhenThereTwoFile(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Errorf("Getwd() failed. err=%v", err)
	}
	logFolder := fmt.Sprintf("%s/.coderrect/logs", cwd)

	if dirExists(logFolder) {
		if err := os.RemoveAll(logFolder); err != nil {
			t.Errorf("Failed to remove log folder. logFolder=%s", logFolder)
		}
	}
	if !dirExists(logFolder) {
		if err := os.MkdirAll(logFolder, getFileMode()); err != nil {
			t.Errorf("Failed to create testing log folder. logFolder=%s", logFolder)
		}
	}

	createFile(logFolder + "/log.0")
	createBigFile(logFolder+"/log.current", 2*1024*1024)
	InitWithLogRotation("test")
	if !fileExists(logFolder + "/log.0") {
		t.Errorf("log.0 wasn't created")
	}
	if !fileExists(logFolder + "/log.1") {
		t.Errorf("log.1 wasn't created")
	}
	if !fileExists(logFolder + "/log.current") {
		t.Errorf("log.current is still there")
	}
}

func TestRotateLogs(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Errorf("Getwd() failed. err=%v", err)
	}
	logFolder := fmt.Sprintf("%s/.coderrect/logs", cwd)

	if dirExists(logFolder) {
		if err := os.RemoveAll(logFolder); err != nil {
			t.Errorf("Failed to remove log folder. logFolder=%s", logFolder)
		}
	}
	if !dirExists(logFolder) {
		if err := os.MkdirAll(logFolder, getFileMode()); err != nil {
			t.Errorf("Failed to create testing log folder. logFolder=%s", logFolder)
		}
	}

	// case 1 - empty folder
	if err := rotateLogs(logFolder, 2, 1*1024*1024); err != nil {
		t.Errorf("Failed to rotate en empty folder. logFolder=%s", logFolder)
	}

	// case 2 - only log.current whose size is smaller than the limit
	createFile(logFolder + "/log.current")
	if err := rotateLogs(logFolder, 2, 1*1024*1024); err != nil {
		t.Errorf("Failed to rotate en empty folder. logFolder=%s", logFolder)
	}

	// case 3 - only log.current whose size is > the limit
	createBigFile(logFolder+"/log.current", 2*1024*1024)
	if err := rotateLogs(logFolder, 2, 1*1024*1024); err != nil {
		t.Errorf("Failed to rotate en empty folder. logFolder=%s", logFolder)
	}
	if fileExists(logFolder + "/log.current") {
		t.Errorf("log.current is still there")
	}
	if !fileExists(logFolder + "/log.0") {
		t.Errorf("log.0 wasn't created")
	}

	// case 4 - log.current whose size > limit, log.0
	createFile(logFolder + "/log.0")
	createBigFile(logFolder+"/log.current", 2*1024*1024)
	if err := rotateLogs(logFolder, 2, 1*1024*1024); err != nil {
		t.Errorf("Failed to rotate en empty folder. logFolder=%s", logFolder)
	}
	if fileExists(logFolder + "/log.current") {
		t.Errorf("log.current is still there")
	}
	if !fileExists(logFolder + "/log.0") {
		t.Errorf("log.0 wasn't created")
	}
	if !fileExists(logFolder + "/log.1") {
		t.Errorf("log.1 wasn't created")
	}

	// case 5 - log.current whose size > limit, log.0, log.1
	createFile(logFolder + "/log.0")
	createBigFile(logFolder+"/log.1", 512*1024)
	createBigFile(logFolder+"/log.current", 2*1024*1024)
	if err := rotateLogs(logFolder, 2, 1*1024*1024); err != nil {
		t.Errorf("Failed to rotate en empty folder. logFolder=%s", logFolder)
	}
	if fileExists(logFolder + "/log.current") {
		t.Errorf("log.current is still there")
	}
	if !fileExists(logFolder + "/log.0") {
		t.Errorf("log.0 wasn't created")
	}
	if fileSize(logFolder+"/log.0") < 1024 {
		t.Errorf("log.0 is too small")
	}
	if !fileExists(logFolder + "/log.1") {
		t.Errorf("log.1 wasn't created")
	}
	if fileSize(logFolder+"/log.1") < 1024 {
		t.Errorf("log.1 is too small")
	}
}

func fileSize(filepath string) int64 {
	fi, _ := os.Stat(filepath)
	return fi.Size()
}

func fileExists(filepath string) bool {
	_, err := os.Stat(filepath)
	return err == nil
}

func createBigFile(filepath string, sz int) (err error) {
	os.Remove(filepath)
	f, err := os.OpenFile(filepath, os.O_APPEND|os.O_WRONLY|os.O_CREATE, getFileMode())
	if err != nil {
		return err
	}
	defer f.Close()

	data := make([]byte, sz)
	_, err = f.Write(data)
	return err
}

func createFile(filePath string) error {
	os.Remove(filePath)
	return ioutil.WriteFile(filePath, []byte("hello"), getFileMode())
}

func getFileMode() os.FileMode {
	cwd, _ := os.Getwd()
	fileInfo, _ := os.Stat(cwd)

	return fileInfo.Mode()
}
