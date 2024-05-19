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
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/coderrect-inc/coderrect/util"
	"github.com/coderrect-inc/coderrect/util/conflib"
	"github.com/coderrect-inc/coderrect/util/logger"

	bolt "go.etcd.io/bbolt"
)

const (
	BucketNameExecFile = "execfiles"
	// BucketNameBcFile stores the mapping between source/executable file paths to bc file paths
	BucketNameBcFile   = "bcfiles"
	BucketNameMoveFile = "movefiles"
	BucketNameLinkFile = "linkfiles"

	xOptionsNameGenerateFortranModule = "XenableGenerateFortranModule"
	fortranModuleFileName             = "fortran-mod"
)

func insertIntoBoltDb(bucketName string, key string, value string) {
	//logger.Debugf("Insert a record into bold db. bucketName=%s key=%s, value=%s", bucketName, key, value)

	boltDbPath := os.Getenv("CODERRECT_BOLD_DB_PATH")
	boltDb, err := bolt.Open(boltDbPath, 0666, nil)
	if err != nil {
		logger.Warnf("Failed to open bolt db. boltDbPath=%s, err=%v", boltDbPath, err)
		return
	}

	err = boltDb.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			// On windows, when you scan the same project many times, the bolt DB will loss bucket data.
			// We must double check the bucket is already built.
			logger.Warnf("Failed to open bucket, recreated. boltDbPath=%s, bucketName=%s", boltDbPath, bucketName)
			b, err = tx.CreateBucket([]byte(bucketName))
			if err != nil {
				return err
			}
		}
		err := b.Put([]byte(key), []byte(value))
		return err
	})
	if err != nil {
		logger.Warnf("Failed to insert a record into the bold db. err=%v", err)
	}
	boltDb.Close()
}

func queryBoltDb(bucketName string, key string) string {
	boltDbPath := os.Getenv("CODERRECT_BOLD_DB_PATH")
	boltDb, err := bolt.Open(boltDbPath, 0666, nil)
	if err != nil {
		logger.Warnf("Failed to open bolt db. boltDbPath=%s, err=%v", boltDbPath, err)
		return ""
	}

	rs := ""
	boltDb.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			return nil
		}

		v := b.Get([]byte(key))
		if v != nil {
			rs = string(v)
		}
		return nil
	})
	boltDb.Close()

	//logger.Debugf("result from bolt db. key=%s, value=%s", key, rs)
	return rs
}

func deleteFromBoltDb(bucketName string, key string) error {
	boltDbPath := os.Getenv("CODERRECT_BOLD_DB_PATH")
	boltDb, err := bolt.Open(boltDbPath, 0666, nil)
	if err != nil {
		logger.Warnf("Failed to open bolt db. boltDbPath=%s, err=%v", boltDbPath, err)
		return err
	}

	err = boltDb.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			return nil
		}
		return b.Delete([]byte(key))
	})
	boltDb.Close()

	logger.Debugf("delete key from bolt db. key=%s", key)
	if err != nil {
		logger.Errorf("Encountered error when deleting key from bolt db. key=%s", key)
	}
	return err
}

//GetFileName returns the file name part of a path
func GetFileName(path string) string {
	_, filename := filepath.Split(path)
	return filename
}

func substr(s string, startIdx int) string {
	a := []rune(s)
	return string(a[startIdx:])
}

func substr2(s string, startIdx int, len int) string {
	a := []rune(s)
	return string(a[startIdx:len])
}

func getBinaryFilename(path string) string {
	filename := GetFileName(path)
	if strings.HasPrefix(filename, ".") {
		filename = substr(filename, 1)
	}

	if strings.HasSuffix(filename, ".bc") {
		filename = substr2(filename, 0, len(filename)-3)
	}

	return filename
}

func getFileNameWithoutExt(path string) string {
	return strings.TrimSuffix(path, filepath.Ext(path))
}

func removeOption(args []string, option string, arity int) []string {
	var clonedArgList []string
	for i := 0; i < len(args); {
		arg := args[i]
		if arg == option {
			i = i + arity
			continue
		} else {
			if strings.HasPrefix(arg, option) {
				i++
				continue
			}
		}
		clonedArgList = append(clonedArgList, arg)
		i++
	}
	return clonedArgList
}

var codes = []string{
	"SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.1", "AVX", "CORE-AVX2",
	"CORE-AVX-I", "ATOM_SSE4.2", "ATOM_SSSE3",
	"ATOM_SSSE3", "KNM", "CORE-AVX512", "COMMON-AVX512",
	"BROADWELL", "CANNONLAKE", "HASWELL", "ICELAKE-CLIENT",
	"ICELAKE", "ICELAKE_SERVER", "IVYBRIDGE", "KNL", "KNM",
	"SANDYBRIDGE", "SILVERMONT", "GOLDMONT", "GOLDMONT-PLUS",
	"TREMONT", "SKYLAKE", "SKYLAKE-AVX512", "CASCADELAKE",
	"KABYLAKE", "AMBERLAKE", "COFFEELAKE", "AMBERLAKE", "WHISKEYLAKE",
	"TIGERLAKE", "SAPPHIRERAPIDS",
}

func removeIntelCodeGenOptions(args []string) []string {

	for _, code := range codes {
		args = removeOption(args, "-x"+code, 1)
	}

	return args
}

func getFortranModuleDirPath() string {
	coderrectBuildDir := os.Getenv("CODERRECT_BUILD_DIR")
	return filepath.Join(coderrectBuildDir, fortranModuleFileName)
}

// COD-1721, .mod problem only for fortran
func generateFortranModule(compilerExecName string, pr parserResult) bool {
	_, onFortran := os.LookupEnv("CODERRECT_FORTRAN_IN_PROGRESS")
	if !onFortran {
		return true
	}
	if !pr.IsCompileOnly {
		return true
	}

	args := pr.CompileArgs[:]
	moduleDirPath := getFortranModuleDirPath()
	if _, serr := os.Stat(moduleDirPath); serr != nil {
		merr := os.MkdirAll(moduleDirPath, os.ModePerm)
		if merr != nil {
			logger.Errorf("Failed to create fortran module directory. dir=%v, err=%v",
				moduleDirPath, merr)
			return false
		}
	}

	if pr.IsCompileOnly {
		args = removeIntelCodeGenOptions(args)
		args = append(removeOption(args, "-J", 2),
			"-fsyntax-only",
			"-J", moduleDirPath)
		for _, srcFilePath := range pr.InputFiles {
			args = append(args, srcFilePath)
		}
	} else {
		// we do not need to link
	}

	logger.Debugf("Compiling module file. compiler=%s, args=%v", compilerExecName, args)
	success, err := execCmd(compilerExecName, args, "")
	if !success {
		logger.Errorf("Failed to compile module file. compilerExecname=%v, args=%v, err=%v",
			compilerExecName, args, err)
		return false
	}

	return true
}

func LinkSolanaBCFiles(linkArgs []string) (success bool) {
	linker := os.Getenv("LLVM_LINK_NAME")
	if len(linker) == 0 {
		linker = "llvm-link"
	}
	logger.Debugf("Find llvm-link. linker=%s", linker)
	success, _ = execCmd(linker, linkArgs, "")
	if !success {
		logger.Errorf("CODERRECT link Solana bcFile error: %v %v\n", linker, linkArgs)
		return false
	}
	return true
}

//Compile wraps a call to the compiler with the given args.
func Compile(args []string, compiler string) (exitCode int) {
	//logger.Debugf("Gllvm start. compiler=%s, args=%v", compiler, args)

	exitCode = 0
	//in the configureOnly case we have to know the exit code of the compile
	//because that is how configure figures out what it can and cannot do.
	var pr = parserResult{}
	if compiler == "link" {
		pr = parseLinkExe(args)
	} else {
		pr = parse(args)
	}
	if len(pr.InputFiles) == 1 && pr.InputFiles[0] == "conftest.c" {
		//logger.Debugf("Skip conftest.c")
		return 0
	}
	outputFileName := pr.OutputFilename
	if outputFileName == "" {
		if pr.IsCompileOnly && len(pr.InputFiles) == 1 {
			//JEFF: single source file "clang x.c -c" generates "x.o"
			//clang -std=c99 -pedantic -c -O3 -fPIC -Wall -W -Wstrict-prototypes -Wwrite-strings -Wno-missing-field-initializers -g -ggdb read.c
			inputFileName := pr.InputFiles[0]
			i := strings.LastIndex(inputFileName, ".")
			outputFileName = inputFileName[0:i] + ".o"
			//logger.Debugf("Single input file compile -c. outputFileName =%v", outputFileName)
		} else {
			outputFileName = "a.out"
		}
	}

	value, _ := filepath.Abs(outputFileName)

	compilerExecName := GetCompilerExecName(compiler)
	if conflib.GetBool(xOptionsNameGenerateFortranModule, true) {
		generateFortranModule(compilerExecName, pr)
	}

	if pr.IsDependencyOnly && !pr.IsCompileOnly {
		//logger.Infof("Don't need to generate BC file. pr=%v", pr)
		return 0
	}

	if !strings.HasSuffix(value, ".o") && !strings.HasSuffix(value, ".lo") &&
		!strings.HasSuffix(value, ".obj") && !strings.HasSuffix(value, ".dll") {
		insertIntoBoltDb(BucketNameExecFile, value, "")
		logger.Infof("Add a binary into db. path=%s", value)
	}

	optionAnalyzeBinary := strings.TrimSpace(conflib.GetString("analyzeBinaries", ""))
	if !pr.IsCompileOnly &&
		len(optionAnalyzeBinary) > 0 &&
		!isSharedObject(outputFileName) &&
		!strings.HasSuffix(compiler, "-ar") {
		binaryOutputFileName := strings.ToLower(getBinaryFilename(outputFileName))
		isFound := false
		optionAnalyzeBinaryArray := strings.Split(optionAnalyzeBinary, ",")
		for _, item := range optionAnalyzeBinaryArray {
			if binaryOutputFileName == strings.ToLower(item) {
				isFound = true
				break
			}
		}

		if !isFound {
			logger.Infof("Don't need to generate BC file for analyzeBinary. pr=%v", pr)
			return 0
		}
	}

	//skip bitcode generation
	if strings.HasSuffix(value, ".cu.o.NVCC-depend") {
		return
	}
	//coderrect: to append -O1 as compiler flag here?
	// pr.InputList = append(pr.InputList, "-O1")

	//fmt.Printf("CODERRECT Compiling %v\n", value)
	//logger.Debugf("Compiling for bitcode. file=%s", value)

	if !buildBitcode(compilerExecName, pr, value) {
		return 1
	}
	//if skipBitcodeGeneration(pr) {}
	return 0
}

func getBitcodeFile(v string) string {
	//if exists "," -Wl,-soname,xxx.so.3 only take the last part
	//-Wp,-MT,/git/linux/tools/objtool/fixdep.o
	id := strings.LastIndex(v, ",")
	if id >= 0 {
		v = v[id+1:]
	}

	i := strings.LastIndex(v, string(filepath.Separator))
	if i >= 0 {
		dir := v[0 : i+1]
		name := v[i+1:]
		return dir + "." + name + ".bc"
	} else {
		return "." + v + ".bc"
	}

}

func unique(strs []string) []string {

	keys := make(map[string]bool)
	list := []string{}
	for _, entry := range strs {
		if _, value := keys[entry]; !value {
			keys[entry] = true
			list = append(list, entry)
		}
	}
	return list
}

func dirExists(dir string) bool {
	info, err := os.Stat(dir)
	return err == nil && info.IsDir()
}

// FindBCFilePathMapping append the buildpath or currentpath to generate the BC file path.
func FindBCFilePathMapping(bcFile string,
	coderrectBuildDir string,
	createIfAbsent bool) (string, error) {
	coderrectCwd := coderrectBuildDir[0 : len(coderrectBuildDir)-len(filepath.FromSlash(".coderrect/build"))]
	/*
		logger.Debugf("entering findBCFilePathMapping. cwd=%s, coderrectBuildDir=%s, bcFile=%s, createIfAbsent=%v",
			cwd, coderrectBuildDir, bcFile, createIfAbsent)
	*/
	bcFile, _ = filepath.Abs(bcFile)

	if strings.HasPrefix(bcFile, coderrectCwd) {
		relativePath := bcFile[len(coderrectCwd):]
		bcFile = filepath.Join(coderrectCwd, ".coderrect", "build", relativePath)
	} else {
		// what if the file doesn't reside under coderrect CWD - prepend coderrectBuildDir
		if bcFile[1] == ':' {
			// for windows, we cut the drive letter
			bcFile = filepath.Join(coderrectBuildDir, bcFile[2:])
		} else {
			bcFile = filepath.Join(coderrectBuildDir, bcFile)
		}
	}

	dirName, _ := filepath.Split(bcFile)
	if len(dirName) > 0 && !dirExists(dirName) && createIfAbsent {
		fi, _ := os.Stat(coderrectBuildDir) // Assumes coderrectBuildDir has already been created
		if err := os.MkdirAll(dirName, fi.Mode()); err != nil {
			return "", err
		}
		//logger.Debugf("Create the BC mapping dir. dir=%s", dirName)
	}

	return bcFile, nil
}

var commonMissingLibraries = []string{"libm.", "libgomp.", "libz.", "libgcc_s.", "libSM.", "libstdc++.", "libgcov.", "libpthread.", "libdl.", "librt.", "libc."}
var crtMissingFiles = []string{"crti.o", "crtn.o", "crtbeginS.o", "crtendS.o"}

func isCommonMissingLibrary(name string) bool {
	for _, v := range commonMissingLibraries {
		if strings.HasPrefix(name, v) {
			return true
		}
	}
	for _, v := range crtMissingFiles {
		if strings.HasSuffix(name, v) {
			return true
		}
	}
	return false
}

func buildBitcode(compilerExecName string, pr parserResult, target string) bool {
	logger.Infof("Generating bc file for binary. complilerExecName=%s, target=%s", compilerExecName, target)

	// a list of paths that may have .coderrect/build directory. If we can't find
	// a BC file in the current coderrect build fold, we'll look into those
	// folders
	coderrectBCPaths := strings.Split(os.Getenv("CODERRECT_BC_PATHS"), ",")

	// [findBCFilePathMapping] Check if we need to create the BC mapping dir.
	// dir=/home/jsong/source/RedisGraph/graph_race/.coderrect/build/home/jsong/source/RedisGraph/deps/GraphBLAS/build/,
	// createIfAbsent=false
	// [buildBitcode] BC file doesn't exist.
	// file=/home/jsong/source/RedisGraph/graph_race/.coderrect/build/home/jsong/source/RedisGraph/deps/GraphBLAS/build/.libgraphblas.a.bc

	coderrectBuildDir := os.Getenv("CODERRECT_BUILD_DIR")

	// var outputFile = target + ".bc"
	outputFile, _ := FindBCFilePathMapping(target+".bc", coderrectBuildDir, true)
	//coderrectCwd := coderrectBuildDir[0 : len(coderrectBuildDir)-len(filepath.FromSlash("/.coderrect/build"))]
	// logger.Infof("Derive BC file path from the binary. coderrectCwd=%s, BCFilePath=%s",
	// 	coderrectCwd, outputFile)

	//TODO: to distinguish if the target is a dynamic lib or an executable
	//for library files or archive command
	var validLibFiles = regexp.MustCompile(`^(.*\.(dll|lib|so|a))[.0-9]*$`)
	if validLibFiles.MatchString(target) || strings.HasSuffix(compilerExecName, "-ar") {
		outputFile = getBitcodeFile(target)
		outputFile, _ = FindBCFilePathMapping(outputFile, coderrectBuildDir, true)
		//logger.Infof("Get the BC file path for a library file. target=%s, outputFile=%s", target, outputFile)
	}

	inputFiles := unique(pr.InputFiles)
	objectFiles := unique(pr.ObjectFiles)

	libFiles := []string{}
	for _, v := range pr.LinkArgs {
		if strings.HasPrefix(v, "-l") {
			libFiles = append(libFiles, v)
		}
		// windows does not have -l in link.exe
		if strings.HasSuffix(v, ".dll") || strings.HasSuffix(v, ".lib") {
			libFiles = append(libFiles, v)
		}
	}
	// logger.Debugf("More information. inputFiles=%v, objectFiles=%v, linkingLibFiles=%v",
	// 	inputFiles, objectFiles, libFiles)

	var objectAndLibFiles = unique(append(objectFiles, libFiles...))

	var bcFiles []string
	var hidden = !pr.IsCompileOnly

	for i, srcFile := range inputFiles {
		objectFile, bcFile := getArtifactNames(pr, i, hidden)

		idx1 := strings.LastIndexByte(objectFile, os.PathSeparator)
		if objectFile[idx1+1] != '.' {
			bcFile = getBitcodeFile(objectFile)
		}

		//prepend inputfile path
		idx := strings.LastIndex(objectFile, string(filepath.Separator))
		if idx >= 0 {
			dir := objectFile[0 : idx+1]
			if !strings.HasPrefix(objectFile, dir) {
				bcFile = dir + bcFile
			}
		}

		//for CUDA only
		//remove .cu from bcFile
		//.lulesh.cu.o.bc => .lulesh.o.bc
		bcFile = strings.Replace(bcFile, ".cu.o.bc", ".o.bc", 1)

		bcFile, err := FindBCFilePathMapping(bcFile, coderrectBuildDir, true)
		if err != nil {
			os.Stderr.WriteString(fmt.Sprintf("Failed to create directory for a BC file. err=%v", err))
			logger.Errorf("Failed to create the directory for a BC file. err=%v, bcFile=%s",
				err, bcFile)
			return false
		}
		//logger.Debugf("Try to build BC file. srcFile=%s, bcFile=%s", srcFile, bcFile)
		success := buildBitcodeFile(compilerExecName, pr, srcFile, bcFile)
		if success {
			objectFile, _ = filepath.Abs(objectFile)
			insertIntoBoltDb(BucketNameBcFile, objectFile, bcFile)
			insertIntoBoltDb(BucketNameBcFile, getBinaryFilename(bcFile), bcFile)
			bcFiles = append(bcFiles, bcFile)
		}
	}
	if len(objectAndLibFiles) > 0 {
		for _, objAndLibFileName := range objectAndLibFiles {
			//logger.Debugf("Trying to find bc file path for an object/lib file. file=%s", objAndLibFileName)
			if strings.HasPrefix(objAndLibFileName, "-l") {
				objAndLibFileName = objAndLibFileName[2:]

				if !strings.HasSuffix(objAndLibFileName, ".a") {
					objAndLibFileName = objAndLibFileName + ".a"
				}
				idx := strings.LastIndex(objAndLibFileName, "/")
				if idx >= 0 {
					path := objAndLibFileName[0 : idx+1]
					name := objAndLibFileName[idx+1:]
					if !strings.HasPrefix(name, "lib") {
						name = "lib" + name
					}
					objAndLibFileName = path + name

				} else {

					if !strings.HasPrefix(objAndLibFileName, "lib") {
						objAndLibFileName = "lib" + objAndLibFileName
					}
				}
			}

			if validLibFiles.MatchString(objAndLibFileName) {
				//TODO: statically linked library .a or shared object .so
				//fmt.Printf("CODERRECT Linking library: %v\n", v)

				//TODO: handle symbolic link
				//check if v is a symbolic link, it yes,

				for {
					fi, err := os.Lstat(objAndLibFileName)
					if err == nil {
						if fi.Mode()&os.ModeSymlink != 0 {
							s, _ := os.Readlink(objAndLibFileName)
							logger.Debugf("Handle a symbol link. from=%v, to=%v", objAndLibFileName, s)
							//assign absolute path
							objAndLibFileName = filepath.Join(filepath.Dir(objAndLibFileName), s)
						} else {
							break
						}
					} else {
						break
					}

				}
			}

			//for CUDA
			//match temporal obj file
			///tmp/tmpxft_00003213_00000000-10_block_error.o => block_error.o
			if strings.HasPrefix(objAndLibFileName, "/tmp/tmpxft_") {
				idx1 := strings.Index(objAndLibFileName, "-") // 1st '-'
				objAndLibFileName = objAndLibFileName[idx1:]
				idx2 := strings.Index(objAndLibFileName, "_") // 1st '_' in the remaining
				objAndLibFileName = objAndLibFileName[idx2+1:]
			}

			//JEFF: skip common lib file eg -lm .libm.a.bc
			if isCommonMissingLibrary(objAndLibFileName) {
				logger.Debugf("Skipped common missing lib. objAndLibFileName=%v", objAndLibFileName)
				continue
			}

			// now try to find where the BC is
			//logger.Debugf("normalized objAndLibFileName=%s", objAndLibFileName)
			oldBCFilePath := getBitcodeFile(objAndLibFileName)
			bcFile, _ := FindBCFilePathMapping(oldBCFilePath, coderrectBuildDir, false)
			//JEFF: fixing a tricky recursion bug?
			// if bcFile is the same as outputFile, skip
			if bcFile == outputFile {
				logger.Debugf("Skipped possible recursive library linking. bcFile=%v, oldBCFilePath=%v, outputFile=%v, objAndLibFileName=%v", bcFile, oldBCFilePath, outputFile, objAndLibFileName)
				continue
			}

			//logger.Debugf("Got the bc file path. oldBCFilePath=%s, bcFile=%s", oldBCFilePath, bcFile)
			_, err := os.Stat(bcFile)
			if err != nil {
				//on-demand link: it is possible that only .link file exists
				_, err = os.Stat(bcFile + ".link")
			}
			if err == nil {
				if skipBCFile(bcFile) {
					fmt.Printf("CODERRECT skipped BC file: %v\n", bcFile)
					logger.Debugf("CODERRECT skipped BC file: %v\n", bcFile)
				} else {
					var OnDemandLinking = true //go compiler will do the optimize
					if !OnDemandLinking && len(bcFiles) > 0 {
						bcFiles = append(bcFiles, "--override", bcFile)
					} else {
						bcFiles = append(bcFiles, bcFile)
					}
				}
			} else {
				logger.Debugf("The bc file doesn't exist in the default location and search different places")
				found := false
				for _, searchPath := range coderrectBCPaths {
					if searchPath == "" {
						continue
					}

					fullSearchPath := path.Join(searchPath, ".coderrect", "build")
					logger.Debugf("Search alternative path. path=%s", fullSearchPath)
					var bcPathMapping map[string]string = make(map[string]string)
					err := filepath.Walk(fullSearchPath,
						func(path string, info os.FileInfo, err error) error {
							if err != nil {
								return err
							}

							if !info.IsDir() && filepath.Ext(path) == ".bc" {
								bcPathMapping[info.Name()] = path
							}
							return nil
						})
					if err != nil {
						logger.Warnf("Walk directory error, err=%v", err)
					}

					bcFile, found = bcPathMapping[oldBCFilePath]
					if found {
						break
					}
				}
				if !found {
					absObjFilePath, _ := filepath.Abs(objAndLibFileName)
					logger.Debugf("searching BC File Mapping ABS: absFile=%s", absObjFilePath)
					bcFile = queryBoltDb(BucketNameBcFile, absObjFilePath)
					found = bcFile != ""
				}
				if !found {
					objFileBaseName := GetFileName(objAndLibFileName)
					logger.Debugf("searching BC File Mapping BASE: baseName=%s", objFileBaseName)
					// now we query bolt db as the last resort
					bcFile = queryBoltDb(BucketNameBcFile, objFileBaseName)
					found = bcFile != ""
				}

				if found {
					logger.Debugf("Found. bcFile=%s", bcFile)
					if skipBCFile(bcFile) {
						fmt.Printf("CODERRECT skipped BC file: %v\n", bcFile)
						logger.Debugf("CODERRECT skipped BC file: %v\n", bcFile)
					} else {
						var OnDemandLinking = true //go compiler will do the optimize
						if !OnDemandLinking && len(bcFiles) > 0 {
							bcFiles = append(bcFiles, "--override", bcFile)
						} else {
							bcFiles = append(bcFiles, bcFile)
						}
					}
				} else {
					logger.Errorf("BC file doesn't exist. file=%s, oldBCFilePath=%s", objAndLibFileName, oldBCFilePath)
				}
			}
		}
	}

	if hidden && (!strings.HasPrefix(target, ".") || len(objectAndLibFiles) > 0) && len(bcFiles) > 0 {
		var linkArgs []string
		linkArgs = append(linkArgs, "-o", outputFile)
		linkArgs = append(linkArgs, bcFiles...)

		linker := LLVMLINKName

		/** per COD-1719 we always use clang/bin/llvm-link
		if _, ok := os.LookupEnv("CODERRECT_FORTRAN_IN_PROGRESS"); ok {
			linker = FortranLLVMLinkName
		}
		*/
		if len(linker) == 0 {
			linker = "llvm-link"
		}
		logger.Debugf("Find llvm-link. linker=%s", linker)

		// FIXME: This is a ugly fix
		// NOTE: Related to code in "/coderrect-ar/main.go"
		// Seems our compile args parsing has some problem.
		// `bcFiles` will also include the `outputFile`
		// Later this will cause infite loop when on-demand linking
		var saneBcFiles []string
		for _, bc := range bcFiles {
			// remove outputFile from bcFiles
			// so there's no recursive dependency
			if bc == outputFile {
				continue
			}
			saneBcFiles = append(saneBcFiles, bc)
		}

		writeLinkFile(linker, outputFile, saneBcFiles)

		insertIntoBoltDb(BucketNameBcFile, target, outputFile)
		insertIntoBoltDb(BucketNameBcFile, GetFileName(target), outputFile)
		// NOTE: store an extra info in the link file bucket
		// if "x.bc" has yet linked, there will be a "x.bc" -> "x.bc.link" mapping exist in the link file bucket
		insertIntoBoltDb(BucketNameLinkFile, outputFile, outputFile+".link")
		//logger.Infof("Created the BC file. file=%s", outputFile)
	}

	return true
}

// LinkBitcodeInParallel link BC file in parallel
//'linkfile' is a BC link file path
func LinkBitcodeInParallel(bc, linkfile string, wg *sync.WaitGroup) error {
	if wg != nil {
		defer wg.Done()
	}

	// parse link file, construct command for linking
	linker, outputFile, bcFiles := parseLinkFile(linkfile)
	var linkArgs, realBcList []string
	linkArgs = append(linkArgs, "-o", outputFile)
	bar := createLinkProgressbar(GetFileName(bc), bcFiles)
	// link files recursively
	var wg2 sync.WaitGroup
	for _, f := range bcFiles {
		linkfile2 := queryBoltDb(BucketNameLinkFile, f)
		if linkfile2 != "" {
			//create a new thread to link
			wg2.Add(1)
			//bar.Add(1)
			go LinkBitcodeInParallel(f, linkfile2, &wg2)
		}
	}
	//wait for all children threads to finish
	wg2.Wait()

	for _, f := range bcFiles {
		//make sure f exist
		//libdbms.a.bc: error: Could not open input file
		if util.FileExists(f) {
			if len(realBcList) > 0 {
				realBcList = append(realBcList, "--override", f)
			} else {
				realBcList = append(realBcList, f)
			}
		}
		bar.Add(1)
	}
	linkArgs = append(linkArgs, realBcList...)
	bar.Add(1)
	// start linking
	success, err := execCmd(linker, linkArgs, "")
	if !success {
		command := append([]string{linker}, linkArgs...)
		logger.Errorf("On-demand linking failed. full link command=%v", command)
	}
	//there might be a race condition between delete and query, but that's fine
	deleteFromBoltDb(BucketNameLinkFile, bc)
	bar.Add(1)
	return err
}

// LinkBitcode on-demand links bc modules to get the target bc file.
// `target` can be either a executable path, or a bc file path
// returns the path of the linked bc file
// FIXME: buildDir seems no longer needed
func LinkBitcode(target, buildDir string) (string, error) {
	// check if the target is already linked
	var bc string
	if isBcFile(target) {
		bc = target
	} else {
		bc = queryBoltDb(BucketNameBcFile, target)
		if bc == "" {
			logger.Errorf("Cannot find the corresponding bc path for the target. target=%v", bc)
		}
	}

	linkfile := queryBoltDb(BucketNameLinkFile, bc)
	// found the link file, the actual bc is yet to be linked
	if linkfile != "" {
		//JEFF: let's link bc in parallel by default
		if conflib.GetBool("link.parallel", true) {
			err := LinkBitcodeInParallel(bc, linkfile, nil)
			return bc, err
		}

		// parse link file, construct command for linking
		linker, outputFile, bcFiles := parseLinkFile(linkfile)
		var linkArgs, realBcList []string
		linkArgs = append(linkArgs, "-o", outputFile)
		// generate real bc list
		// if there's a ".link" file in the dependency
		// we need to generate the actual bc for the ".link" file
		bar := createLinkProgressbar(GetFileName(target), bcFiles)
		for _, f := range bcFiles {
			//for on-demand linking, do --override here
			//in order to avoid --override --override
			if isBcFile(f) {
				f, err := LinkBitcode(f, buildDir)
				//make sure f exist
				//libdbms.a.bc: error: Could not open input file
				if err == nil && f != "" && util.FileExists(f) {
					if len(realBcList) > 0 {
						realBcList = append(realBcList, "--override", f)
					} else {
						realBcList = append(realBcList, f)
					}
				}
				bar.Add(1)
			} else {
				//for on-demand linking, should not get here?
				realBcList = append(realBcList, f)
			}
		}
		linkArgs = append(linkArgs, realBcList...)
		// start linking
		//logger.Infof("On-demand linking. linkArgs=%v", linkArgs)
		success, err := execCmd(linker, linkArgs, "")
		bar.Add(1)
		if !success {
			command := append([]string{linker}, linkArgs...)
			logger.Errorf("On-demand linking failed. full link command=%v", command)
			//panic(err)
		}
		// insertIntoBoltDb(BucketNameBcFile, target, outputFile)
		deleteFromBoltDb(BucketNameLinkFile, bc)
		bar.Add(1)
		return outputFile, err
	}
	return bc, nil
}

var includeFileNameList = []string{}
var excludeFileNameList = []string{}
var notInitializedInExcludeFiles = true

func skipSrcFile(srcFile string) (success bool) {
	//logger.Debugf("CODERRECT building bitcode for file: %v\n", srcFile)
	if notInitializedInExcludeFiles {
		includeFileNameList = conflib.GetStrings("includeFiles")
		//fmt.Printf("CODERRECT includeFileNames: %v\n", includeFileNameList)
		//logger.Debugf("CODERRECT includeFileNames: %v\n", includeFileNameList)
		excludeFileNameList = conflib.GetStrings("excludeFiles")
		//fmt.Printf("CODERRECT excludeFileNames: %v\n", excludeFileNameList)
		//logger.Debugf("CODERRECT excludeFileNames: %v\n", excludeFileNameList)
		notInitializedInExcludeFiles = false
	}
	for i := 0; i < len(includeFileNameList); i++ {
		if strings.Contains(srcFile, includeFileNameList[i]) {
			return false
		}
	}
	for i := 0; i < len(excludeFileNameList); i++ {
		if strings.Contains(srcFile, excludeFileNameList[i]) {
			return true
		}
	}
	return false
}
func skipBCFile(bcFile string) (success bool) {
	//fmt.Printf("CODERRECT linking bitcode file: %v\n", bcFile)
	//logger.Debugf("CODERRECT linking bitcode file: %v\n", bcFile)
	if notInitializedInExcludeFiles {
		includeFileNameList = conflib.GetStrings("includeFiles")
		//fmt.Printf("CODERRECT includeFileNames: %v\n", includeFileNameList)
		//logger.Debugf("CODERRECT includeFileNames: %v\n", includeFileNameList)
		excludeFileNameList = conflib.GetStrings("excludeFiles")
		//fmt.Printf("CODERRECT excludeFileNames: %v\n", excludeFileNameList)
		//logger.Debugf("CODERRECT excludeFileNames: %v\n", excludeFileNameList)
		notInitializedInExcludeFiles = false
	}
	//strings.Replace(bcFile,".bc","",1)
	for i := 0; i < len(includeFileNameList); i++ {
		//if strings.ToLower(srcFile) == strings.ToLower(includeFileNameList[i]) {
		if strings.Contains(bcFile, includeFileNameList[i]) {
			return false
		}
	}
	for i := 0; i < len(excludeFileNameList); i++ {
		//if strings.ToLower(srcFile) == strings.ToLower(excludeFileNameList[i]) {
		if strings.Contains(bcFile, excludeFileNameList[i]) {
			return true
		}
	}
	return false
}

// Tries to build the specified source file to bitcode
func buildBitcodeFile(compilerExecName string, pr parserResult, srcFile string, bcFile string) (success bool) {
	if skipSrcFile(srcFile) {
		fmt.Printf("CODERRECT skipped file: %v\n", srcFile)
		logger.Debugf("CODERRECT skipped file: %v\n", srcFile)
		return false
	}
	if isAssembly(srcFile) {
		logger.Debugf("skip building bitcode for assembly: %v\n", srcFile)
		return false
	}

	_, onFortran := os.LookupEnv("CODERRECT_FORTRAN_IN_PROGRESS")

	//skip temp files /tmp/tmpxft_00001d5b_00000000-1.cpp
	if strings.HasPrefix(srcFile, "/tmp/tmpxft_") && strings.HasSuffix(srcFile, ".cpp") {
		logger.Debugf("CODERRECT cuda skipped temp file: %v\n", srcFile)
		return false
	}

	//TODO: clang throws errors on some options
	//error: unknown warning option '-Wsuggest-override'; did you mean '-Wshift-overflow'? [-Werror,-Wunknown-warning-option]
	args := pr.CompileArgs[:]

	if onFortran {

		if conflib.GetBool(xOptionsNameGenerateFortranModule, true) {
			moduleDirPath := getFortranModuleDirPath()
			args = removeOption(args, "-J", 2)
			args = append(args,
				"-J", moduleDirPath, // search directory for .mod file
			)
		}

		args = append(args, "-mp", "-c", "-O1", "-fno-discard-value-names", "-Wno-everything",
			"-emit-llvm",
			"-disable-llvm-optzns",
			"-D_Float128=double",
			srcFile)

		// check if there is "-g". Remove it if so
		args = removeOption(args, "-g", 1)
		args = removeIntelCodeGenOptions(args)
	} else if pr.IsCC1 {
		args = append(args, "-D", "_SILENCE_CLANG_COROUTINE_MESSAGE", "-O1", "-Wno-everything", "-S", "-emit-llvm",
			"-disable-llvm-optzns", "-debug-info-kind=limited", "-dwarf-version=4", srcFile)
	} else {
		args = append(args, "-O1", "-Wno-everything", "-g", "-Xclang", "-S", "-Xclang", "-emit-llvm", "-fno-discard-value-names",
			"-D_Float128=double",
			"-Xclang", "-disable-llvm-optzns", "-c", srcFile)
	}
	//for linux crypto/jitterentropy.c:54:3: error: "The CPU Jitter random number generator must not be compiled with optimizations. See documentation. Use the compiler switch -O0 for compiling jitterentropy.c."
	// #error "The CPU Jitter random number generator must not be compiled with optimizations. See documentation. Use the compiler switch -O0 for compiling jitterentropy.c."
	if strings.HasSuffix(srcFile, "jitterentropy.c") {
		args = append(args, "-O0")
	}

	isCuda := strings.HasSuffix(srcFile, ".cu")
	if isCuda || pr.IsCuda {
		//clang -c -emit-llvm --cuda-gpu-arch=sm_75 /CUDA/thundersvm/src/thundersvm/kernel/smo_kernel.cu -I/CUDA/thundersvm/include -std=c++11 -I/CUDA/thundersvm/build
		logger.Debugf("set arg for cuda file. srcFile=%v", srcFile)
		args = append(args, "--cuda-gpu-arch=sm_75")
	} else {
		args = append(args, "-o", bcFile)
	}

	logger.Debugf("Build the BC file. compiler=%s, args=%v", compilerExecName, args)
	success, err := execCmd(compilerExecName, args, "")
	if !success {
		logger.Errorf("Failed to generate bitcode. compilerExecname=%v, args=%v, err=%v",
			compilerExecName, args, err)
		return false
	}

	if isCuda || pr.IsCuda {
		//special handling for cuda bitcode
		//llvm-link smo_kernel.bc --override smo_kernel-cuda-nvptx64-nvidia-cuda-sm_75.bc -o .thundersvm_generated_smo_kernel.cu.o.bc
		//mv .thundersvm_generated_smo_kernel.cu.o.bc /CUDA/thundersvm/build/src/thundersvm/CMakeFiles/thundersvm.dir/kernel/./
		//rm smo_kernel.bc smo_kernel-cuda-nvptx64-nvidia-cuda-sm_75.bc
		basename := filepath.Base(srcFile) //smo_kernel.cu
		idx := strings.Index(basename, ".")
		if idx > 0 {
			basename = basename[0:idx] //smo_kernel
		}
		//remove .cu from bcFile  -- this moved to the caller
		//.lulesh.cu.o.bc => .lulesh.o.bc
		//bcFile = strings.Replace(bcFile, ".cu.o.bc", ".o.bc", 1)

		var linkArgs []string
		linkArgs = append(linkArgs, "--suppress-warnings", basename+".bc", "--override", basename+"-cuda-nvptx64-nvidia-cuda-sm_75.bc", "-o", bcFile)

		//fmt.Printf("CODERRECT llvm-link cuda: %v\n", linkArgs)
		logger.Debugf("CODERRECT llvm-link cuda: %v\n", linkArgs)
		linker := LLVMLINKName
		if len(linker) == 0 {
			linker = "llvm-link"
		}
		success, _ = execCmd(linker, linkArgs, "")
		if !success {
			logger.Errorf("CODERRECT link cuda bcFile error: %v %v\n", linker, linkArgs)
			return false
		}
	}
	return true
}

// GetCompilerExecName returns the full path of the executable
func GetCompilerExecName(compiler string) string {
	switch compiler {
	case "clang":
		return LLVMCCName
	case "clang++":
		return LLVMCXXName
	case "coderrect-ar":
		return LLVMARName
	case "coderrect-ld":
		return LLVMLDName
	case "fortran":
		return LLVMFORTRANName
	case "clang-cl":
		return LLVMCLName
	case "link":
		return "link"
	default:
		logger.Errorf("The compiler is not supported by this tool. compiler=%s", compiler)
		return ""
	}
}

//CheckDefer is used to check the return values of defers
func CheckDefer(f func() error) {
	if err := f(); err != nil {
		logger.Warnf("CheckDefer received error. err=%v", err)
	}
}
