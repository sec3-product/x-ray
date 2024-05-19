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
	"debug/elf"
	"debug/macho"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/coderrect-inc/coderrect/util/logger"
)

//for distilling the desired commandline options
type ExtractionArgs struct {
	Failure             bool // indicates failure in parsing the cmd line args
	Verbose             bool // inform the user of what is going on
	WriteManifest       bool // write a manifest of bitcode files used
	SortBitcodeFiles    bool // sort the arguments to linking and archiving (debugging too)
	BuildBitcodeModule  bool // buld an archive rather than a module
	KeepTemp            bool // keep temporary linking folder
	LinkArgSize         int  // maximum size of a llvm-link command line
	InputType           int
	ObjectTypeInArchive int // Type of file that can be put into an archive
	InputFile           string
	OutputFile          string
	LlvmLinkerName      string
	LlvmArchiverName    string
	ArchiverName        string
	ArArgs              []string
	Extractor           func(string) []string
}

//for printing out the parsed arguments, some have been skipped.
func (ea ExtractionArgs) String() string {
	format :=
		`
ea.Verbose:            %v
ea.WriteManifest:      %v
ea.SortBitcodeFiles:   %v
ea.BuildBitcodeModule: %v
ea.KeepTemp:           %v
ea.LinkArgSize:        %v
ea.InputFile:          %v
ea.OutputFile:         %v
ea.LlvmArchiverName:   %v
ea.LlvmLinkerName:     %v
ea.ArchiverName:       %v
`
	return fmt.Sprintf(format, ea.Verbose, ea.WriteManifest, ea.SortBitcodeFiles, ea.BuildBitcodeModule,
		ea.KeepTemp, ea.LinkArgSize, ea.InputFile, ea.OutputFile, ea.LlvmArchiverName,
		ea.LlvmLinkerName, ea.ArchiverName)
}

func ParseSwitches(args []string) (ea ExtractionArgs) {

	var flagSet *flag.FlagSet = flag.NewFlagSet(args[0], flag.ContinueOnError)

	flagSet.BoolVar(&ea.Verbose, "v", false, "verbose mode")
	flagSet.BoolVar(&ea.WriteManifest, "m", false, "write the manifest")
	flagSet.BoolVar(&ea.SortBitcodeFiles, "s", false, "sort the bitcode files")
	flagSet.BoolVar(&ea.BuildBitcodeModule, "b", false, "build a bitcode module")
	flagSet.StringVar(&ea.OutputFile, "o", "", "the output file")
	flagSet.StringVar(&ea.LlvmArchiverName, "a", "llvm-ar", "the llvm archiver (i.e. llvm-ar)")
	flagSet.StringVar(&ea.ArchiverName, "r", "ar", "the system archiver (i.e. ar)")
	flagSet.StringVar(&ea.LlvmLinkerName, "l", "llvm-link", "the llvm linker (i.e. llvm-link)")
	flagSet.IntVar(&ea.LinkArgSize, "n", 0, "maximum llvm-link command line size (in bytes)")
	flagSet.BoolVar(&ea.KeepTemp, "t", false, "keep temporary linking folder")

	err := flagSet.Parse(args[1:])

	if err != nil {
		ea.Failure = true
		return
	}

	ea.LlvmArchiverName = resolveTool("llvm-ar", LLVMARName, ea.LlvmArchiverName)
	ea.LlvmLinkerName = resolveTool("llvm-link", LLVMLINKName, ea.LlvmLinkerName)
	inputFiles := flagSet.Args()
	if len(inputFiles) != 1 {
		LogError("Can currently only deal with exactly one input file, sorry. You gave me %v input files.\n", len(inputFiles))
		ea.Failure = true
		return
	}
	ea.InputFile = inputFiles[0]
	if _, err := os.Stat(ea.InputFile); os.IsNotExist(err) {
		LogError("The input file %s  does not exist.", ea.InputFile)
		ea.Failure = true
		return
	}
	realPath, err := filepath.EvalSymlinks(ea.InputFile)
	if err != nil {
		LogError("There was an error getting the real path of %s.", ea.InputFile)
		ea.Failure = true
		return
	}
	ea.InputFile = realPath
	ea.InputType = getFileType(realPath)

	LogInfo("%v", ea)

	return
}

//Extract extracts the LLVM bitcode according to the arguments it is passed.
func Extract(args []string) (exitCode int) {
	exitCode = 1

	ea := ParseSwitches(args)

	if ea.Failure {
		return
	}

	// Set arguments according to runtime OS
	success := setPlatform(&ea)
	if !success {
		return
	}

	// Create output filename if not given
	setOutputFile(&ea)

	switch ea.InputType {
	case fileTypeELFEXECUTABLE,
		fileTypeELFSHARED,
		fileTypeELFOBJECT,
		fileTypeMACHEXECUTABLE,
		fileTypeMACHSHARED,
		fileTypeMACHOBJECT:
		success = handleExecutable(ea)
	case fileTypeARCHIVE:
		success = handleArchive(ea)
	case fileTypeTHINARCHIVE:
		success = handleThinArchive(ea)
	case fileTypeERROR:
	default:
		logger.Errorf("Incorrect input file type. type=%v", ea.InputType)
		return
	}

	if success {
		exitCode = 0
	}

	return
}

// Set arguments according to runtime OS
func setPlatform(ea *ExtractionArgs) (success bool) {
	switch platform := runtime.GOOS; platform {
	case osFREEBSD, osLINUX:
		ea.Extractor = extractSectionUnix
		if ea.Verbose {
			ea.ArArgs = append(ea.ArArgs, "xv")
		} else {
			ea.ArArgs = append(ea.ArArgs, "x")
		}
		ea.ObjectTypeInArchive = fileTypeELFOBJECT
		success = true
	case osDARWIN:
		ea.Extractor = extractSectionDarwin
		ea.ArArgs = append(ea.ArArgs, "-x")
		if ea.Verbose {
			ea.ArArgs = append(ea.ArArgs, "-v")
		}
		ea.ObjectTypeInArchive = fileTypeMACHOBJECT
		success = true
	default:
		logger.Errorf("Unsupported platform. platform=%s", platform)
	}
	return
}

// Create output filename if not given
func setOutputFile(ea *ExtractionArgs) {
	if ea.OutputFile == "" {
		if ea.InputType == fileTypeARCHIVE || ea.InputType == fileTypeTHINARCHIVE {
			var ext string
			if ea.BuildBitcodeModule {
				ext = ".a.bc"
			} else {
				ext = ".bca"
			}
			ea.OutputFile = strings.TrimSuffix(ea.InputFile, ".a") + ext
		} else {
			ea.OutputFile = ea.InputFile + ".bc"
		}
	}
}

func resolveTool(defaultPath string, envPath string, usrPath string) (path string) {
	if usrPath != defaultPath {
		path = usrPath
	} else {
		if LLVMToolChainBinDir != "" {
			if envPath != "" {
				path = filepath.Join(LLVMToolChainBinDir, envPath)
			} else {
				path = filepath.Join(LLVMToolChainBinDir, defaultPath)
			}
		} else {
			if envPath != "" {
				path = envPath
			} else {
				path = defaultPath
			}
		}
	}
	logger.Debugf("More information. defaultPath=%s, envPath=%s, usrPath=%s, path=%s",
		defaultPath, envPath, usrPath, path)
	return
}

func handleExecutable(ea ExtractionArgs) (success bool) {
	// get the list of bitcode paths
	artifactPaths := ea.Extractor(ea.InputFile)

	if len(artifactPaths) < 20 {
		// naert: to avoid saturating the log when dealing with big file lists
		logger.Infof("handle executable. artifactPaths=%v", artifactPaths)
	}

	if len(artifactPaths) == 0 {
		return
	}
	filesToLink := make([]string, len(artifactPaths))
	for i, artPath := range artifactPaths {
		filesToLink[i] = resolveBitcodePath(artPath)
	}

	// Sort the bitcode files
	if ea.SortBitcodeFiles {
		logger.Debugf("Sorting bitcode files. filesToLink=%v, artifactPaths=%v", filesToLink, artifactPaths)
		sort.Strings(filesToLink)
		sort.Strings(artifactPaths)
	}

	// Write manifest
	if ea.WriteManifest {
		if !writeManifest(ea, filesToLink, artifactPaths) {
			return
		}
	}

	success = linkBitcodeFiles(ea, filesToLink)
	return
}

func handleThinArchive(ea ExtractionArgs) (success bool) {
	// List bitcode files to link
	var artifactFiles []string

	var objectFiles []string
	var bcFiles []string

	objectFiles = listArchiveFiles(ea, ea.InputFile)

	logger.Infof("get list of object files. ExtractionArgs=%v, objectFiles=%v", ea, objectFiles)

	for index, obj := range objectFiles {
		logger.Debugf("Processing an object file. obj=%v", obj)
		if len(obj) > 0 {
			artifacts := ea.Extractor(obj)
			logger.Debugf("Extract artifacts. artifacts=%v", artifacts)
			artifactFiles = append(artifactFiles, artifacts...)
			for _, bc := range artifacts {
				bcPath := resolveBitcodePath(bc)
				if bcPath != "" {
					bcFiles = append(bcFiles, bcPath)
				}
			}
		} else {
			logger.Debugf("\tskipping empty entry. index=%v", index)
		}
	}

	logger.Infof("BC file list info. bcFiles=%v, len=%v", bcFiles, len(bcFiles))

	if len(bcFiles) > 0 {
		// Sort the bitcode files
		if ea.SortBitcodeFiles {
			logger.Debugf("Sorting bitcode files. bcFiles=%v, artifactFiles=%v", bcFiles, artifactFiles)
			sort.Strings(bcFiles)
			sort.Strings(artifactFiles)
		}

		// Build archive
		if ea.BuildBitcodeModule {
			success = linkBitcodeFiles(ea, bcFiles)
		} else {
			success = archiveBcFiles(ea, bcFiles)
		}

		if !success {
			return
		}

		// Write manifest
		if ea.WriteManifest {
			success = writeManifest(ea, bcFiles, artifactFiles)
		}
	} else {
		logger.Errorf("No bitcode files found")
		success = false
	}
	return
}

func listArchiveFiles(ea ExtractionArgs, inputFile string) (contents []string) {
	var arArgs []string
	arArgs = append(arArgs, "-t")
	arArgs = append(arArgs, inputFile)
	output, err := runCmd(ea.ArchiverName, arArgs)
	if err != nil {
		logger.Errorf("Failed to extract contents from archive. arCmd=%v, args=%v, inputFile=%v, err=%v",
			ea.ArchiverName, arArgs, inputFile, err)
		return
	}
	contents = strings.Split(output, "\n")
	return
}

func extractFile(ea ExtractionArgs, archive string, filename string, instance int) (success bool) {
	var arArgs []string
	if runtime.GOOS != osDARWIN {
		arArgs = append(arArgs, "xN")
		arArgs = append(arArgs, strconv.Itoa(instance))
	} else {
		if instance > 1 {
			logger.Warnf("Cannot extract instance from archive for instance > 1. instance=%v, filename=%s, archive=%v",
				instance, filename, archive)
			return
		}
		arArgs = append(arArgs, "x")
	}
	arArgs = append(arArgs, archive)
	arArgs = append(arArgs, filename)
	_, err := runCmd(ea.ArchiverName, arArgs)
	if err != nil {
		logger.Warnf("The archiver failed to extract instance from archive. archiverName=%v, instance=%v, filename=%s, archive=%v, err=%v",
			ea.ArchiverName, instance, filename, archive, err)
		return
	}
	success = true
	return
}

func fetchTOC(ea ExtractionArgs, inputFile string) map[string]int {
	toc := make(map[string]int)

	contents := listArchiveFiles(ea, inputFile)

	for _, item := range contents {
		//iam: this is a hack to make get-bc work on libcurl.a
		if item != "" && !strings.HasPrefix(item, "__.SYMDEF") {
			toc[item]++
		}
	}
	return toc
}

//handleArchive processes an archive, and creates either a bitcode archive, or a module, depending on the flags used.
//
//    Archives are strange beasts. handleArchive processes the archive by:
//
//      1. first creating a table of contents of the archive, which maps file names (in the archive) to the number of
//    times a file with that name is stored in the archive.
//
//      2. for each OCCURRENCE of a file (name and count) it extracts the section from the object file, and adds the
//    bitcode paths to the bitcode list.
//
//      3. it then either links all these bitcode files together using llvm-link,  or else is creates a bitcode
//    archive using llvm-ar
//
//iam: 5/1/2018
func handleArchive(ea ExtractionArgs) (success bool) {
	// List bitcode files to link
	var bcFiles []string
	var artifactFiles []string

	inputFile, _ := filepath.Abs(ea.InputFile)

	logger.Infof("Start. ExtractionArgs=%v", ea)

	// Create tmp dir
	tmpDirName, err := ioutil.TempDir("", "gllvm")
	if err != nil {
		logger.Errorf("The temporary directory in which to extract object files could not be created.")
		return
	}

	defer CheckDefer(func() error { return os.RemoveAll(tmpDirName) })

	homeDir, err := os.Getwd()
	if err != nil {
		logger.Errorf("Could not ascertain our whereabouts. err=%v", err)
		return
	}

	err = os.Chdir(tmpDirName)
	if err != nil {
		logger.Errorf("Could not cd. tmpDirName=%s, err=%v", tmpDirName, err)
		return
	}

	//1. fetch the Table of Contents
	toc := fetchTOC(ea, inputFile)

	logger.Debugf("Table of Contents. inputFile=%v, toc=%v", inputFile, toc)

	for obj, instance := range toc {
		for i := 1; i <= instance; i++ {

			if obj != "" && extractFile(ea, inputFile, obj, i) {

				artifacts := ea.Extractor(obj)
				logger.Infof("\tartifacts=%v", artifacts)
				artifactFiles = append(artifactFiles, artifacts...)
				for _, bc := range artifacts {
					bcPath := resolveBitcodePath(bc)
					if bcPath != "" {
						bcFiles = append(bcFiles, bcPath)
					}
				}
			}
		}
	}

	err = os.Chdir(homeDir)
	if err != nil {
		logger.Errorf("Could not cd. homeDir=%s, err=%v", homeDir, err)
		return
	}

	logger.Debugf("walked nartifactFiles. tmpDirName=%s, artifactFiles=%v, bcFiles=%v",
		tmpDirName, artifactFiles, bcFiles)

	if len(bcFiles) > 0 {

		// Sort the bitcode files
		if ea.SortBitcodeFiles {
			logger.Warnf("Sorting bitcode files.")
			sort.Strings(bcFiles)
			sort.Strings(artifactFiles)
		}

		// Build archive
		if ea.BuildBitcodeModule {
			success = linkBitcodeFiles(ea, bcFiles)
		} else {
			success = archiveBcFiles(ea, bcFiles)
		}

		if !success {
			//hopefully the failure has alreadu been reported...
			return
		}

		// Write manifest
		if ea.WriteManifest {
			success = writeManifest(ea, bcFiles, artifactFiles)
		}
	} else {
		logger.Errorf("No bitcode files found")
		return
	}
	return
}

func archiveBcFiles(ea ExtractionArgs, bcFiles []string) (success bool) {
	// We do not want full paths in the archive, so we need to chdir into each
	// bitcode's folder. Handle this by calling llvm-ar once for all bitcode
	// files in the same directory
	dirToBcMap := make(map[string][]string)
	for _, bcFile := range bcFiles {
		dirName, baseName := path.Split(bcFile)
		dirToBcMap[dirName] = append(dirToBcMap[dirName], baseName)
	}

	// Call llvm-ar from each directory
	absOutputFile, _ := filepath.Abs(ea.OutputFile)
	for dir, bcFilesInDir := range dirToBcMap {
		var args []string
		var err error
		args = append(args, "rs", absOutputFile)
		args = append(args, bcFilesInDir...)
		success, err = execCmd(ea.LlvmArchiverName, args, dir)
		logger.Infof("ea.LlvmArchiverName=%s, args=%v, dir=%s", ea.LlvmArchiverName, args, dir)
		if !success {
			logger.Errorf("There was an error creating the bitcode archive. err=%v", err)
			return
		}
	}
	informUser("Built bitcode archive: %s.\n", ea.OutputFile)
	success = true
	return
}

func getsize(stringslice []string) (totalLength int) {
	totalLength = 0
	for _, s := range stringslice {
		totalLength += len(s)
	}
	return totalLength
}

func formatStdOut(stdout bytes.Buffer, usefulIndex int) string {
	infoArr := strings.Split(stdout.String(), "\n")[usefulIndex]
	ret := strings.Fields(infoArr)
	return ret[0]
}

func fetchArgMax(ea ExtractionArgs) (argMax int) {
	if ea.LinkArgSize == 0 {
		getArgMax := exec.Command("getconf", "ARG_MAX")
		var argMaxStr bytes.Buffer
		getArgMax.Stdout = &argMaxStr
		err := getArgMax.Run()
		if err != nil {
			logger.Errorf("getconf ARG_MAX failed. err=%s", err)
		}
		argMax, err = strconv.Atoi(formatStdOut(argMaxStr, 0))
		if err != nil {
			logger.Errorf("string conversion for argMax failed. err=%v", err)
		}
		argMax = int(0.9 * float32(argMax)) // keeping a comfort margin
	} else {
		argMax = ea.LinkArgSize
	}
	logger.Infof("argMax=%v", argMax)
	return
}

func linkBitcodeFilesIncrementally(ea ExtractionArgs, filesToLink []string, argMax int, linkArgs []string) (success bool) {
	var tmpFileList []string
	// Create tmp dir
	tmpDirName, err := ioutil.TempDir(".", "glinking")
	if err != nil {
		logger.Errorf("The temporary directory in which to put temporary linking files could not be created. err=%v", err)
		return
	}
	if !ea.KeepTemp { // delete temporary folder after used unless told otherwise
		logger.Infof("Temporary folder will be deleted")
		defer CheckDefer(func() error { return os.RemoveAll(tmpDirName) })
	} else {
		logger.Infof("Keeping the temporary folder")
	}

	tmpFile, err := ioutil.TempFile(tmpDirName, "tmp")
	if err != nil {
		logger.Errorf("The temporary linking file could not be created. err=%v", err)
		return
	}
	tmpFileList = append(tmpFileList, tmpFile.Name())
	linkArgs = append(linkArgs, "-o", tmpFile.Name())

	logger.Infof("llvm-link-argument-size=%d", getsize(filesToLink))
	for _, file := range filesToLink {
		linkArgs = append(linkArgs, file)
		if getsize(linkArgs) > argMax {
			logger.Infof("Linking command size exceeding system capacity : splitting the command")
			success, err = execCmd(ea.LlvmLinkerName, linkArgs, "")
			if !success || err != nil {
				logger.Errorf("There was an error linking input files. outputFile=%s, file=%v, err=%v",
					ea.OutputFile, file, err)
				success = false
				return
			}
			linkArgs = nil

			if ea.Verbose {
				linkArgs = append(linkArgs, "-v")
			}
			tmpFile, err = ioutil.TempFile(tmpDirName, "tmp")
			if err != nil {
				logger.Errorf("Could not generate a temp file. tmpDirName=%s, err=%v", tmpDirName, err)
				success = false
				return
			}
			tmpFileList = append(tmpFileList, tmpFile.Name())
			linkArgs = append(linkArgs, "-o", tmpFile.Name())
		}

	}
	success, err = execCmd(ea.LlvmLinkerName, linkArgs, "")
	if !success || err != nil {
		logger.Errorf("There was an error linking input files. tmpFile=%s, err=%v", tmpFile.Name(), err)
		success = false
		return
	}
	linkArgs = nil
	if ea.Verbose {
		linkArgs = append(linkArgs, "-v")
	}
	linkArgs = append(linkArgs, tmpFileList...)

	linkArgs = append(linkArgs, "-o", ea.OutputFile)

	success, err = execCmd(ea.LlvmLinkerName, linkArgs, "")
	if !success {
		logger.Errorf("There was an error linking input file. outputFile=%s, err=%v", ea.OutputFile, err)
		return
	}
	logger.Infof("Bitcode file is extracted. to=%s, from=%v", ea.OutputFile, tmpFileList)
	success = true
	return
}

func linkBitcodeFiles(ea ExtractionArgs, filesToLink []string) (success bool) {
	var linkArgs []string
	// Extracting the command line max size from the environment if it is not specified
	argMax := fetchArgMax(ea)
	if ea.Verbose {
		linkArgs = append(linkArgs, "-v")
	}
	if getsize(filesToLink) > argMax { //command line size too large for the OS (necessitated by chromium)
		return linkBitcodeFilesIncrementally(ea, filesToLink, argMax, linkArgs)
	} else {
		var err error
		linkArgs = append(linkArgs, "-o", ea.OutputFile)
		linkArgs = append(linkArgs, filesToLink...)
		success, err = execCmd(ea.LlvmLinkerName, linkArgs, "")
		if !success {
			logger.Errorf("There was an error linking input files. outputFile=%s, err=%v", ea.OutputFile, err)
			return
		}
		informUser("Bitcode file extracted to: %s.\n", ea.OutputFile)
	}
	success = true
	return
}

func extractSectionDarwin(inputFile string) (contents []string) {
	machoFile, err := macho.Open(inputFile)
	if err != nil {
		logger.Errorf("Mach-O file could not be read. inputFile=%s", inputFile)
		return
	}
	section := machoFile.Section(DarwinSectionName)
	if section == nil {
		logger.Warnf("The section is missing! section=%s, inputFile=%s", DarwinSectionName, inputFile)
		return
	}
	sectionContents, errContents := section.Data()
	if errContents != nil {
		logger.Warnf("Failed to read the section of Mach-O file. section=%s, inputFile=%s", DarwinSectionName, inputFile)
		return
	}
	contents = strings.Split(strings.TrimSuffix(string(sectionContents), "\n"), "\n")
	return
}

func extractSectionUnix(inputFile string) (contents []string) {
	elfFile, err := elf.Open(inputFile)
	if err != nil {
		logger.Errorf("ELF file could not be read. inputFile=%s", inputFile)
		return
	}
	section := elfFile.Section(ELFSectionName)
	if section == nil {
		logger.Warnf("Error reading the section of ELF file. section=%s, inputFile=%s", ELFSectionName, inputFile)
		return
	}
	sectionContents, errContents := section.Data()
	if errContents != nil {
		logger.Warnf("Error reading the section of ELF file. section=%s, inputFile=%s", ELFSectionName, inputFile)
		return
	}
	contents = strings.Split(strings.TrimSuffix(string(sectionContents), "\n"), "\n")
	return
}

// Return the actual path to the bitcode file, or an empty string if it does not exist
func resolveBitcodePath(bcPath string) string {
	if _, err := os.Stat(bcPath); os.IsNotExist(err) {
		// If the bitcode file does not exist, try to find it in the store
		if LLVMBitcodeStorePath != "" {
			// Compute absolute path hash
			absBcPath, _ := filepath.Abs(bcPath)
			storeBcPath := path.Join(LLVMBitcodeStorePath, getHashedPath(absBcPath))
			if _, err := os.Stat(storeBcPath); os.IsNotExist(err) {
				return ""
			}
			return storeBcPath
		}
		logger.Warnf("Failed to find the file. bcPath=%v", bcPath)
		return ""
	}
	return bcPath
}

func writeManifest(ea ExtractionArgs, bcFiles []string, artifactFiles []string) (success bool) {
	manifestFilename := ea.OutputFile + ".llvm.manifest"
	//only go into the gory details if we have a store around.
	if LLVMBitcodeStorePath != "" {
		section1 := "Physical location of extracted files:\n" + strings.Join(bcFiles, "\n") + "\n\n"
		section2 := "Build-time location of extracted files:\n" + strings.Join(artifactFiles, "\n")
		contents := []byte(section1 + section2)
		if err := ioutil.WriteFile(manifestFilename, contents, 0644); err != nil {
			logger.Errorf("There was an error while writing the manifest file. err=%v", err)
			return
		}
	} else {
		contents := []byte("\n" + strings.Join(bcFiles, "\n") + "\n")
		if err := ioutil.WriteFile(manifestFilename, contents, 0644); err != nil {
			logger.Errorf("There was an error while writing the manifest file. err=%v", err)
			return
		}
	}
	informUser("Manifest file written to %s.\n", manifestFilename)
	success = true
	return
}
