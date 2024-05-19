//
// OCCAM
//
// Copyright (c) 2017, SRI International
//
//  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of SRI International nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

package shared

import (
	"bytes"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/coderrect-inc/coderrect/util"
	"github.com/coderrect-inc/coderrect/util/logger"
	"github.com/schollz/progressbar/v3"
)

// Executes a command then returns true for success, false if there was an error, err is either nil or the error.
func execCmd(cmdExecName string, args []string, workingDir string) (success bool, err error) {
	cmd := exec.Command(cmdExecName, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		logger.Warnf("Failed to execute a command. out=%s", string(out))
	}
	return err == nil, err
}

// Executes a command then returns the output as a string, err is either nil or the error.
func runCmd(cmdExecName string, args []string) (output string, err error) {
	var outb bytes.Buffer
	var errb bytes.Buffer
	cmd := exec.Command(cmdExecName, args...)
	cmd.Stdout = &outb
	cmd.Stderr = &errb
	cmd.Stdin = os.Stdin
	err = cmd.Run()
	LogDebug("runCmd: %v %v\n", cmdExecName, args)
	if err != nil {
		LogDebug("runCmd: error was %v\n", err)
	}
	output = outb.String()
	return
}

func isAssembly(file string) bool {
	return strings.HasSuffix(file, ".S") || strings.HasSuffix(file, ".s")
}

func isLinkFile(file string) bool {
	return strings.HasSuffix(file, ".link")
}

func isBcFile(file string) bool {
	return strings.HasSuffix(file, ".bc")
}

// check if a file is so file
// there are in general 2 cases:
//   1. xxx.so -- this is simple, just check suffix ".so"
//   2. xxx.so.1.2.3 -- need regex
func isSharedObject(file string) bool {
	// this pattern requires the ending of the string either:
	//  - end with ".so"
	//  - or end with ".so" followed by an optional version number, e.g. ".so.1.2.3"
	pattern := regexp.MustCompile(".+\\.so(\\.\\d+)*$")
	return pattern.MatchString(file)
}

func writeLinkFile(linker string, outputFile string, bcFiles []string) {
	linkFile := outputFile + ".link"
	if !util.FileExists(linkFile) {
		// create link file if it doesn't exist
		content := linker + ";" + outputFile + ";" + strings.Join(bcFiles, "&")
		err := ioutil.WriteFile(linkFile, []byte(content), 0644)
		if err != nil {
			logger.Errorf("Error writing to link file. linkFile=%v, content=%v", linkFile, content)
			panic(err)
		}
		logger.Infof("Created link file. linkFile=%v, content=%v", linkFile, content)
	} else {
		// NOTE: This is to fix a bug in the linking process for GraphBLAS
		// where libgraphblas.a is linked three times to get its final version
		// each time link against its previous version plus some additional object files
		file, err := os.OpenFile(linkFile, os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			logger.Errorf("Error opening the existing link file. linkFile=%v", linkFile)
			panic(err)
		}
		defer file.Close()

		_, _, deps := parseLinkFile(linkFile)
		content := ""
		for _, bc := range bcFiles {
			if !util.Contains(deps, bc) {
				content += " " + bc
			}
		}

		if _, err := file.WriteString(content); err != nil {
			logger.Errorf("Error appending to the link file. linkFile=%v, content=%v", linkFile, content)
		}
		logger.Infof("Appended to an exisiting link file. linkFile=%v, content=%v", linkFile, content)
	}
}

// the returned strings are:
// output file path, dependent bc files
func parseLinkFile(linkfile string) (string, string, []string) {
	bytes, err := ioutil.ReadFile(linkfile)
	if err != nil {
		logger.Errorf("Error parsing link file. linkFile=%v, err=%v", linkfile, err)
		panic(err)
	}
	content := strings.Split(string(bytes), ";")
	return content[0], content[1], strings.Split(content[2], "&")
}

func createLinkProgressbar(target string, items []string) *progressbar.ProgressBar {
	var cnt int64
	cnt = 0
	for _, item := range items {
		if isBcFile(item) {
			cnt++
		}
	}
	return progressbar.Default(cnt+2, "Linking "+target)
}
