package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"testing"
)

func TestCommand(t *testing.T) {

	var outBuf bytes.Buffer
	cmd := exec.Command("cmd", "/C", "notepad")
	cmd.Stderr = BufferedStderr{
		buffer: &outBuf,
	}
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		fmt.Print("ok")
	} else {
		fmt.Print("failed")
	}
}

func TestAddOrReplaceLogFolder(t *testing.T) {
	// case 1 - replace
	jsonOptStr := `
{
  "Opts": [
    "-report.color=red",
    "-logger.logFolder=logs"
  ]
}
`
	jsonOptStr = addOrReplaceLogFolder(jsonOptStr, "/tmp")
	fmt.Printf("new=%s\n", jsonOptStr)

	// case 2 - add
	jsonOptStr = `
{
  "Opts": [
    "-report.color=red"
  ]
}
`
	jsonOptStr = addOrReplaceLogFolder(jsonOptStr, "/tmp")
	fmt.Printf("new=%s\n", jsonOptStr)

	// case 3 - empty
	jsonOptStr = `
{
  "Opts": [
  ]
}
`
	jsonOptStr = addOrReplaceLogFolder(jsonOptStr, "/tmp")
	fmt.Printf("new=%s\n", jsonOptStr)
}

func Test_sortFoundExecutables(t *testing.T) {
	var strs []string
	strs = append(strs, "world")
	strs = append(strs, "hello")
	i := 0
	for i < 20 {
		strs = append(strs, fmt.Sprintf("hahaha%d", i))
		i++
	}
	strs = append(strs, "test_hello")
	strs = append(strs, "world_test")
	strs = append(strs, ".123434.tmp")

	strs = sortFoundExecutables(strs)
	fmt.Printf("%v\n", strs)
}

func Test_getRelativePath(t *testing.T) {
	cwd := "/home/jsong/source/memcached/build"
	filepath := "/home/jsong/source/memcached/build/memcached"
	relativePath := getRelativePath(cwd, filepath)
	if relativePath != "memcached" {
		t.Errorf("expected=%s, actual=%s", "memcached", relativePath)
	}

	cwd = "/home/jsong/memcached/build"
	filepath = "/home/jsong/source/memcached/build/memcached"
	relativePath = getRelativePath(cwd, filepath)
	if relativePath != "../../source/memcached/build/memcached" {
		t.Errorf("expected=%s, actual=%s", "../../source/memcached/build/memcached", relativePath)
	}
}

func Test_normalizeCmdlineArg(t *testing.T) {
	s := "cxxflags=foo bar"
	s1 := normalizeCmdlineArg(s)
	if s1 != "cxxflags=\"foo bar\"" {
		t.Errorf("actual=%s", s1)
	}

	s = "foo"
	s1 = normalizeCmdlineArg(s)
	if s1 != "foo" {
		t.Errorf("actual=%s", s1)
	}

	s = "cxx=clang"
	s1 = normalizeCmdlineArg(s)
	if s1 != "cxx=clang" {
		t.Errorf("actual=%s", s1)
	}

	s = "cxx='clang'"
	s1 = normalizeCmdlineArg(s)
	if s1 != "cxx=\"'clang'\"" {
		t.Errorf("actual=%s", s1)
	}
}
