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
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/coderrect-inc/coderrect/util/logger"
)

type parserResult struct {
	InputList        []string
	InputFiles       []string
	ObjectFiles      []string
	OutputFilename   string
	CompileArgs      []string
	LinkArgs         []string
	ForbiddenFlags   []string
	IsVerbose        bool
	IsDependencyOnly bool
	IsPreprocessOnly bool
	IsAssembleOnly   bool
	IsAssembly       bool
	IsCompileOnly    bool
	IsBitcodeOnly    bool
	IsEmitLLVM       bool
	IsPrintOnly      bool
	IsCuda           bool
	IsCC1            bool
}

type flagInfo struct {
	arity   int
	handler func(string, []string)
}

type argPattern struct {
	pattern string
	finfo   flagInfo
}

func skipBitcodeGeneration(pr parserResult) bool {
	if LLVMConfigureOnly != "" {
		return true
	}
	if len(pr.InputFiles) == 0 ||
		pr.IsEmitLLVM ||
		pr.IsAssembly ||
		pr.IsAssembleOnly ||
		(pr.IsDependencyOnly && !pr.IsCompileOnly) ||
		pr.IsPreprocessOnly ||
		pr.IsPrintOnly {
		return true
	}
	return false

}

func parseLinkExe(argList []string) parserResult {
	var pr = parserResult{}
	pr.InputList = argList

	// iam: this is a list because matching needs to be done in order.
	// if you add a NEW pattern; make sure it is before any existing pattern that also
	// matches and has a conflicting flagInfo value.
	var argPatterns = [...]argPattern{
		{`(?i)^\/OUT:.+$`, flagInfo{0, pr.outputFileCallback}},
		{`(?i)^\-OUT:.+$`, flagInfo{0, pr.outputFileCallback}},
		{`^\/IMPLIB:.+$`, flagInfo{0, pr.objectFileCallback}},
		{`^\/.+$`, flagInfo{0, pr.skipCallback}},
		{`^.+\.(lib|dll|obj)$`, flagInfo{0, pr.objectFileCallback}},
	}

	for len(argList) > 0 {
		var elem = argList[0]

		var listShift = 0
		for _, argPat := range argPatterns {
			pattern := argPat.pattern
			fi := argPat.finfo
			var regExp = regexp.MustCompile(pattern)
			if regExp.MatchString(elem) {
				fi.handler(elem, argList[1:1+fi.arity])
				listShift = fi.arity
				break
			}
		}
		// skip unmatched
		// we do not care when linking, just need to collect all the dependencies
		argList = argList[1+listShift:]
	}

	return pr
}

func parse(argList []string) parserResult {
	var pr = parserResult{}
	pr.InputList = argList

	if argList[0] == "-cc1" {
		pr.IsCC1 = true
		pr.IsCompileOnly = true
	} else {
		pr.IsCC1 = false
	}

	// special case for -cc1 do not need to translate anything
	// this is already the driver mode clang all the flag should be compatible already
	var argsExactMatches = map[string]flagInfo{

		"/dev/null": {0, pr.inputFileCallback}, //iam: linux kernel

		"-":               {0, pr.printOnlyCallback},
		"-o":              {1, pr.outputFileCallback},
		"-c":              {0, pr.compileOnlyCallback},
		"-cc1":            {0, pr.compileUnaryCallback},
		"-triple":         {1, pr.compileBinaryCallback},
		"-E":              {0, pr.preprocessOnlyCallback},
		"-S":              {0, pr.assembleOnlyCallback},
		"-main-file-name": {1, pr.compileBinaryCallback},
		"-target-cpu":     {1, pr.compileBinaryCallback},

		// clang-cl -ccl
		"-fdebug-compilation-dir": {1, pr.compileBinaryCallback},
		"-ferror-limit":           {1, pr.compileBinaryCallback},
		"-fmessage-length":        {1, pr.compileBinaryCallback},
		"-fmessage-limit":         {1, pr.compileBinaryCallback},
		"-internal-isystem":       {1, pr.compileBinaryCallback},
		"-pic-level":              {1, pr.compileBinaryCallback},
		"-mthread-model":          {1, pr.compileBinaryCallback},
		"-x":                      {1, pr.compileBinaryCallback},
		"-mrelocation-model":      {1, pr.compileBinaryCallback},
		"-stack-protector":        {1, pr.compileBinaryCallback},
		"-fdiagnostics-format":    {1, pr.compileBinaryCallback},
		"-mllvm":                  {1, pr.compileBinaryCallback},
		"-include-pch":            {0, pr.skipCallback},
		"-resource-dir":           {1, pr.compileBinaryCallback},
		"-gcodeview":              {0, pr.skipCallback},
		"-gcodeview-ghash":        {0, pr.skipCallback},

		"-emit-obj": {0, pr.skipCallback},
		"--verbose": {0, pr.verboseFlagCallback},
		"--param":   {1, pr.defaultBinaryCallback},
		"-aux-info": {1, pr.defaultBinaryCallback},

		"--version": {0, pr.compileOnlyCallback},
		"-v":        {0, pr.compileOnlyCallback},
		"-bc":       {0, pr.bitcodeOnlyCallback},

		"-w":                           {0, pr.skipCallback}, //jeff: skip option -w(suppress all warnning)
		"-W":                           {0, pr.skipCallback}, //jeff: skip option -W
		"-Werror":                      {0, pr.skipCallback}, //jeff: chromium
		"-Wno-builtin-macro-redefined": {0, pr.skipCallback}, //jeff: chromium
		"-Wno-unused-parameter":        {0, pr.skipCallback}, //jeff: chromium
		"-fexceptions":                 {0, pr.skipCallback}, //jeff: chromium
		"-g2":                          {0, pr.skipCallback}, //jeff: chromium
		"-gsplit-dwarf":                {0, pr.skipCallback}, //jeff: chromium
		"-ggnu-pubnames":               {0, pr.skipCallback}, //jeff: chromium
		//cuda related
		//"-arch":{1, pr.skipCallback},
		"-ccbin":                      {1, pr.skipCallback}, //jeff: skip -ccbin /usr/bin/cc for cuda
		"-Xcompiler":                  {1, pr.skipCallback}, //jeff: skip "-Xcompiler ,"-D_MWAITXINTRIN_H_INCLUDED","-D_FORCE_INLINES","-fPIC","-O3","-DNDEBUG"
		"-Wno-deprecated-gpu-targets": {0, pr.skipCallback}, //jeff: skip Wno-deprecated-gpu-targets
		"--expt-extended-lambda":      {0, pr.skipCallback},
		"--expt-relaxed-constexpr":    {0, pr.skipCallback},
		"--display_error_number":      {0, pr.skipCallback},
		"-restrict":                   {0, pr.skipCallback},
		"-Xcudafe":                    {0, pr.skipCallback},
		"-gencode":                    {0, pr.skipCallback},
		"-mrecord-mcount":             {0, pr.skipCallback}, //jeff: linux

		"-emit-llvm": {0, pr.emitLLVMCallback},

		"-pipe":                  {0, pr.compileUnaryCallback},
		"-undef":                 {0, pr.compileUnaryCallback},
		"-nostdinc":              {0, pr.compileUnaryCallback},
		"-nostdinc++":            {0, pr.compileUnaryCallback},
		"-Qunused-arguments":     {0, pr.compileUnaryCallback},
		"-no-integrated-as":      {0, pr.compileUnaryCallback},
		"-integrated-as":         {0, pr.compileUnaryCallback},
		"-no-canonical-prefixes": {0, pr.compileLinkUnaryCallback},

		"--sysroot": {1, pr.compileLinkBinaryCallback}, //iam: musl stuff

		//<archaic flags>
		"-no-cpp-precomp": {0, pr.compileUnaryCallback},
		//</archaic flags>

		"-pthread":     {0, pr.linkUnaryCallback},
		"-nostdlibinc": {0, pr.compileUnaryCallback},

		"-mno-omit-leaf-frame-pointer":     {0, pr.compileUnaryCallback},
		"-maes":                            {0, pr.compileUnaryCallback},
		"-mno-aes":                         {0, pr.compileUnaryCallback},
		"-mavx":                            {0, pr.compileUnaryCallback},
		"-mno-avx":                         {0, pr.compileUnaryCallback},
		"-mavx2":                           {0, pr.compileUnaryCallback},
		"-mno-avx2":                        {0, pr.compileUnaryCallback},
		"-mno-red-zone":                    {0, pr.compileUnaryCallback},
		"-mmmx":                            {0, pr.compileUnaryCallback},
		"-mno-mmx":                         {0, pr.compileUnaryCallback},
		"-mno-global-merge":                {0, pr.compileUnaryCallback}, //iam: linux kernel stuff
		"-mno-80387":                       {0, pr.compileUnaryCallback}, //iam: linux kernel stuff
		"-msse":                            {0, pr.compileUnaryCallback},
		"-mno-sse":                         {0, pr.compileUnaryCallback},
		"-msse2":                           {0, pr.compileUnaryCallback},
		"-mno-sse2":                        {0, pr.compileUnaryCallback},
		"-msse3":                           {0, pr.compileUnaryCallback},
		"-mno-sse3":                        {0, pr.compileUnaryCallback},
		"-mssse3":                          {0, pr.compileUnaryCallback},
		"-mno-ssse3":                       {0, pr.compileUnaryCallback},
		"-msse4":                           {0, pr.compileUnaryCallback},
		"-mno-sse4":                        {0, pr.compileUnaryCallback},
		"-msse4.1":                         {0, pr.compileUnaryCallback},
		"-mno-sse4.1":                      {0, pr.compileUnaryCallback},
		"-msse4.2":                         {0, pr.compileUnaryCallback},
		"-mno-sse4.2":                      {0, pr.compileUnaryCallback},
		"-msoft-float":                     {0, pr.compileUnaryCallback},
		"-mpopcnt":                         {0, pr.compileUnaryCallback},
		"-m3dnow":                          {0, pr.compileUnaryCallback},
		"-mno-3dnow":                       {0, pr.compileUnaryCallback},
		"-m16":                             {0, pr.compileLinkUnaryCallback}, //iam: linux kernel stuff
		"-m32":                             {0, pr.compileLinkUnaryCallback},
		"-m64":                             {0, pr.compileLinkUnaryCallback},
		"-mstackrealign":                   {0, pr.compileUnaryCallback},
		"-mretpoline-external-thunk":       {0, pr.compileUnaryCallback}, //iam: linux kernel stuff
		"--param=allow-store-data-races=0": {0, pr.skipCallback},         //jeff: linux kernel stuff
		"-A":                               {1, pr.compileBinaryCallback},
		"-D":                               {1, pr.compileBinaryCallback},
		"-U":                               {1, pr.compileBinaryCallback},

		"-arch": {1, pr.compileBinaryCallback}, //iam: openssl

		"-P": {1, pr.compileUnaryCallback}, //iam: linux kernel stuff (linker script stuff)
		"-C": {1, pr.compileUnaryCallback}, //iam: linux kernel stuff (linker script stuff)

		"-M":   {0, pr.dependencyOnlyCallback},
		"-MM":  {0, pr.dependencyOnlyCallback},
		"-MF":  {1, pr.dependencyBinaryCallback},
		"-MG":  {0, pr.dependencyOnlyCallback},
		"-MP":  {0, pr.dependencyOnlyCallback},
		"-MT":  {1, pr.dependencyBinaryCallback},
		"-MQ":  {1, pr.dependencyBinaryCallback},
		"-MD":  {0, pr.dependencyOnlyCallback},
		"-MMD": {0, pr.dependencyOnlyCallback},

		"-I":                 {1, pr.compileBinaryCallback},
		"-J":                 {1, pr.compileBinaryCallback},
		"-idirafter":         {1, pr.compileBinaryCallback},
		"-include":           {1, pr.compileBinaryCallback},
		"-imacros":           {1, pr.compileBinaryCallback},
		"-iprefix":           {1, pr.compileBinaryCallback},
		"-iwithprefix":       {1, pr.compileBinaryCallback},
		"-iwithprefixbefore": {1, pr.compileBinaryCallback},
		"-isystem":           {1, pr.compileBinaryCallback},
		"-isysroot":          {1, pr.compileBinaryCallback},
		"-iquote":            {1, pr.compileBinaryCallback},
		"-imultilib":         {1, pr.compileBinaryCallback},
		"-dwarf-column-info": {0, pr.compileUnaryCallback},

		"-ansi":     {0, pr.compileUnaryCallback},
		"-pedantic": {0, pr.compileUnaryCallback},

		"-g":                    {0, pr.compileUnaryCallback},
		"-g0":                   {0, pr.compileUnaryCallback},
		"-g1":                   {0, pr.compileUnaryCallback},
		"-g3":                   {0, pr.compileUnaryCallback},
		"-ggdb":                 {0, pr.compileUnaryCallback},
		"-ggdb3":                {0, pr.compileUnaryCallback},
		"-gdwarf":               {0, pr.compileUnaryCallback},
		"-gdwarf-2":             {0, pr.compileUnaryCallback},
		"-gdwarf-3":             {0, pr.compileUnaryCallback},
		"-gdwarf-4":             {0, pr.compileUnaryCallback},
		"-gdwarf-aranges":       {0, pr.compileUnaryCallback},
		"-gline-tables-only":    {0, pr.compileUnaryCallback},
		"-grecord-gcc-switches": {0, pr.compileUnaryCallback},

		"-p":  {0, pr.compileUnaryCallback},
		"-pg": {0, pr.compileUnaryCallback},

		"-O":     {0, pr.compileUnaryCallback},
		"-O0":    {0, pr.compileUnaryCallback},
		"-O1":    {0, pr.compileUnaryCallback},
		"-O2":    {0, pr.compileUnaryCallback},
		"-O3":    {0, pr.compileUnaryCallback},
		"-Os":    {0, pr.compileUnaryCallback},
		"-Ofast": {0, pr.compileUnaryCallback},
		"-Og":    {0, pr.compileUnaryCallback},
		"-Oz":    {0, pr.compileUnaryCallback}, //iam: linux kernel

		"-Xclang":        {1, pr.compileXclangBinaryCallback},
		"-Xpreprocessor": {1, pr.defaultBinaryCallback},
		"-Xassembler":    {1, pr.defaultBinaryCallback},
		"-Xlinker":       {1, pr.defaultBinaryCallback},

		"-l":            {1, pr.linkBinaryCallback},
		"-L":            {1, pr.linkBinaryCallback},
		"-T":            {1, pr.linkBinaryCallback},
		"-u":            {1, pr.linkBinaryCallback},
		"-install_name": {1, pr.linkBinaryCallback},

		"-e":     {1, pr.linkBinaryCallback},
		"-rpath": {1, pr.linkBinaryCallback},

		"-shared":        {0, pr.linkUnaryCallback},
		"-static":        {0, pr.linkUnaryCallback},
		"-static-libgcc": {0, pr.linkUnaryCallback}, //iam: musl stuff
		"-pie":           {0, pr.linkUnaryCallback},
		"-nostdlib":      {0, pr.linkUnaryCallback},
		"-nodefaultlibs": {0, pr.linkUnaryCallback},
		"-rdynamic":      {0, pr.linkUnaryCallback},

		"-dynamiclib":            {0, pr.linkUnaryCallback},
		"-current_version":       {1, pr.linkBinaryCallback},
		"-compatibility_version": {1, pr.linkBinaryCallback},

		"-print-multi-directory":  {0, pr.compileUnaryCallback},
		"-print-multi-lib":        {0, pr.compileUnaryCallback},
		"-print-libgcc-file-name": {0, pr.compileUnaryCallback},
		"-print-search-dirs":      {0, pr.compileUnaryCallback},

		"-fprofile-arcs": {0, pr.compileLinkUnaryCallback},
		"-coverage":      {0, pr.compileLinkUnaryCallback},
		"--coverage":     {0, pr.compileLinkUnaryCallback},
		"-fopenmp":       {0, pr.compileLinkUnaryCallback},
		"-qopenmp":       {0, pr.compileIntelOpenMPLinkUnaryCallback},

		"-Wl,-dead_strip": {0, pr.warningLinkUnaryCallback},
		"-dead_strip":     {0, pr.warningLinkUnaryCallback}, //iam: tor does this. We lose the bitcode :-(

		"-plugin":            {1, pr.linkerPluginBinaryCallback}, //jeff: -plugin /usr/lib/gcc/x86_64-linux-gnu/7/liblto_plugin.so
		"-Wsuggest-override": {0, pr.skipCallback},               //jeff: skip option Wsuggest-override
	}

	// iam: this is a list because matching needs to be done in order.
	// if you add a NEW pattern; make sure it is before any existing pattern that also
	// matches and has a conflicting flagInfo value.
	var argPatterns = [...]argPattern{
		{`^-Wno-.*$`, flagInfo{0, pr.skipCallback}},             //jeff: chromium
		{`^-Werror.*$`, flagInfo{0, pr.skipCallback}},           //jeff: chromium
		{`^-Warray.*$`, flagInfo{0, pr.skipCallback}},           //linux kernel
		{`^-Wstrict-*$`, flagInfo{0, pr.skipCallback}},          //linux kernel
		{`^-mpreferred-.*$`, flagInfo{0, pr.skipCallback}},      //linux kernel
		{`^-mindirect-branch.*$`, flagInfo{0, pr.skipCallback}}, //linux kernel
		{`^-mno-.*$`, flagInfo{0, pr.skipCallback}},             //linux kernel
		{`^-mskip-.*$`, flagInfo{0, pr.skipCallback}},           //linux kernel
		{`^-fconserve-.*$`, flagInfo{0, pr.skipCallback}},       //linux kernel
		{`^-falign-.*$`, flagInfo{0, pr.skipCallback}},          //linux kernel
		{`^-fmerge-.*$`, flagInfo{0, pr.skipCallback}},          //linux kernel
		{`^-fno-.*$`, flagInfo{0, pr.skipCallback}},             //linux kernel
		{`^.+\.(c|cc|cpp|C|cxx|i|s|S|bc|cu)$`, flagInfo{0, pr.inputFileCallback}},
		{`^.+\.([fF](|[0-9][0-9]|or|OR|pp|PP))$`, flagInfo{0, pr.inputFileCallback}},
		{`^.+\.(o|lo|So|so|po|a|dylib|al|obj)$`, flagInfo{0, pr.objectFileCallback}},
		{`^.+\.dylib(\.\d)+$`, flagInfo{0, pr.objectFileCallback}},
		//jeff: palo
		{`^.+\.pic$`, flagInfo{0, pr.objectFileCallback}},
		//jeff: dropbear
		{`^-mfunction-return=.*$`, flagInfo{0, pr.skipCallback}},
		{`-pch-through-header=.*$`, flagInfo{0, pr.skipCallback}},
		//jeff:  cuda
		{`^-ccbin=.*$`, flagInfo{0, pr.skipCallback}},
		{`^-Xcompiler=.*$`, flagInfo{0, pr.skipCallback}},
		{`^--diag_suppress=.*$`, flagInfo{0, pr.skipCallback}},
		//jeff: file
		{`^-plugin-opt=.*$`, flagInfo{0, pr.linkerPluginOptCallback}},
		//-Wl,-h,libfakeopenh264.so
		{`^-Wl,-h,.*$`, flagInfo{0, pr.sharedLibrarySonameCallback}},
		//jeff: -Wl,-soname,xxx.so.3
		{`^-Wl,-soname,.*$`, flagInfo{0, pr.sharedLibrarySonameCallback}},
		//jeff: rsp file
		{`^@.*\.rsp$`, flagInfo{0, pr.objectRSPFileCallback}},
		{`^.+\.(So|so|dll)(\.\d)+$`, flagInfo{0, pr.objectFileCallback}},
		{`^-(l|L).+$`, flagInfo{0, pr.linkUnaryCallback}},
		{`^-I.+$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^-D.+$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^-B.+$`, flagInfo{0, pr.compileLinkUnaryCallback}},
		{`^-isystem.+$`, flagInfo{0, pr.compileLinkUnaryCallback}},
		{`^-U.+$`, flagInfo{0, pr.compileUnaryCallback}},
		//iam, need to be careful here, not mix up linker and warning flags.
		{`^-Wl,.+$`, flagInfo{0, pr.linkUnaryCallback}},
		{`^-W[^l].*$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^-W[l][^,].*$`, flagInfo{0, pr.compileUnaryCallback}}, //iam: tor has a few -Wl...
		{`^-fsanitize=.+$`, flagInfo{0, pr.compileLinkUnaryCallback}},
		{`^-fuse-ld=.+$`, flagInfo{0, pr.linkUnaryCallback}}, //iam:  musl stuff
		{`^-f.+$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^-rtlib=.+$`, flagInfo{0, pr.linkUnaryCallback}},
		{`^-std=.+$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^-stdlib=.+$`, flagInfo{0, pr.compileLinkUnaryCallback}},
		{`^-mtune=.+$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^--sysroot=.+$`, flagInfo{0, pr.compileLinkUnaryCallback}}, //both compile and link time
		{`^-print-prog-name=.*$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^-print-file-name=.*$`, flagInfo{0, pr.compileUnaryCallback}},
		{`^-mmacosx-version-min=.+$`, flagInfo{0, pr.compileLinkUnaryCallback}},
		{`^-mstack-alignment=.+$`, flagInfo{0, pr.compileUnaryCallback}},          //iam, linux kernel stuff
		{`^-march=.+$`, flagInfo{0, pr.compileUnaryCallback}},                     //iam: linux kernel stuff
		{`^-mregparm=.+$`, flagInfo{0, pr.compileUnaryCallback}},                  //iam: linux kernel stuff
		{`^-mcmodel=.+$`, flagInfo{0, pr.compileUnaryCallback}},                   //iam: linux kernel stuff
		{`^-mpreferred-stack-boundary=.+$`, flagInfo{0, pr.compileUnaryCallback}}, //iam: linux kernel stuff
		{`^--param=.+$`, flagInfo{0, pr.compileUnaryCallback}},                    //iam: linux kernel stuff
	}

	for len(argList) > 0 {
		var elem = argList[0]

		// Try to match the flag exactly
		if fi, ok := argsExactMatches[elem]; ok {
			fi.handler(elem, argList[1:1+fi.arity])
			argList = argList[1+fi.arity:]
			// Else try to match a pattern
		} else {
			var listShift = 0
			var matched = false

			for _, argPat := range argPatterns {
				pattern := argPat.pattern
				fi := argPat.finfo
				var regExp = regexp.MustCompile(pattern)
				if regExp.MatchString(elem) {
					fi.handler(elem, argList[1:1+fi.arity])
					listShift = fi.arity
					matched = true
					break
				}
			}
			if !matched {
				//jeff: skip ld -plugin -plugin-opt
				logger.Warnf("Did not recognize the compiler flag: %v\n", elem)
				if strings.HasPrefix(elem, "-") {
					pr.compileUnaryCallback(elem, argList[1:1])
				} else {
					pr.specialUnaryCallback(elem, argList[1:1])
				}
			}
			argList = argList[1+listShift:]
		}
	}
	return pr
}

// Return the object and bc filenames that correspond to the i-th source file
func getArtifactNames(pr parserResult, srcFileIndex int, hidden bool) (objBase string, bcBase string) {
	if len(pr.InputFiles) == 1 && pr.IsCompileOnly && len(pr.OutputFilename) > 0 {
		objBase = pr.OutputFilename
		dir, baseName := path.Split(objBase)
		bcBaseName := fmt.Sprintf(".%s.bc", baseName)
		bcBase = path.Join(dir, bcBaseName)

		// logger.Debugf("Generate the BC file path for the object file. OutputFilename=%s, BCFilePath=%s",
		// 	pr.OutputFilename, bcBase)
	} else {
		srcFile := pr.InputFiles[srcFileIndex]
		var _, baseNameWithExt = path.Split(srcFile)
		// issue #30:  main.cpp and main.c cause conflicts.
		var baseName = strings.TrimSuffix(baseNameWithExt, filepath.Ext(baseNameWithExt))
		bcBase = fmt.Sprintf(".%s.o.bc", baseNameWithExt)
		if hidden {
			objBase = fmt.Sprintf(".%s.o", baseNameWithExt)
		} else {
			objBase = fmt.Sprintf("%s.o", baseName)
		}

		// logger.Debugf("Generate the BC file path for the object file. srcFile=%s, objFilePath=%s, BCFilePath=%s",
		// 	srcFile, objBase, bcBase)
	}

	return
}

// Return a hash for the absolute object path
func getHashedPath(path string) string {
	inputBytes := []byte(path)
	hasher := sha256.New()
	//Hash interface claims this never returns an error
	hasher.Write(inputBytes)
	hash := hex.EncodeToString(hasher.Sum(nil))
	return hash
}

func (pr *parserResult) inputFileCallback(flag string, _ []string) {
	var regExp = regexp.MustCompile(`\.(s|S)$`)
	pr.InputFiles = append(pr.InputFiles, flag)
	if regExp.MatchString(flag) {
		pr.IsAssembly = true
	}
}

func (pr *parserResult) outputFileCallback(flag string, args []string) {
	//logger.Debugf("output file: %v", flag)
	if strings.HasPrefix(flag, "/out:") {
		pr.OutputFilename = strings.TrimPrefix(flag, "/out:")
	} else if strings.HasPrefix(flag, "/OUT:") {
		pr.OutputFilename = strings.TrimPrefix(flag, "/OUT:")
	} else if strings.HasPrefix(flag, "-out:") {
		pr.OutputFilename = strings.TrimPrefix(flag, "-out:")
	} else if strings.HasPrefix(flag, "-OUT:") {
		pr.OutputFilename = strings.TrimPrefix(flag, "-OUT:")
	} else {
		pr.OutputFilename = args[0]
	}
}

func (pr *parserResult) sharedLibrarySonameCallback(flag string, _ []string) {
	logger.Infof("shared library soname: %v\n", flag)
}

func (pr *parserResult) objectRSPFileCallback(flag string, _ []string) {

	logger.Infof("rsp file: %v\n", flag)
	filename := flag[1:]
	content, err := ioutil.ReadFile(filename)
	if err == nil {
		names := strings.Split(string(content), " ")
		for _, name := range names {
			pr.ObjectFiles = append(pr.ObjectFiles, name)
		}
	} else {

		logger.Warnf("rsp file does not exist: %v\n", filename)
	}

}
func (pr *parserResult) objectFileCallback(flag string, _ []string) {
	if strings.HasPrefix(flag, "/IMPLIB:") {
		flag = strings.TrimPrefix(flag, "/IMPLIB:")
		// is used nowhere else in the code
		pr.ObjectFiles = append(pr.ObjectFiles, flag)
		// We append the object files to link args to handle the
		// -Wl,--start-group obj_1.o ... obj_n.o -Wl,--end-group case
		pr.LinkArgs = append(pr.LinkArgs, flag)
	} else {
		// FIXME: the object file is appended to ObjectFiles that
		// is used nowhere else in the code
		pr.ObjectFiles = append(pr.ObjectFiles, flag)
		// We append the object files to link args to handle the
		// -Wl,--start-group obj_1.o ... obj_n.o -Wl,--end-group case
		pr.LinkArgs = append(pr.LinkArgs, flag)
	}
}

func (pr *parserResult) preprocessOnlyCallback(_ string, _ []string) {
	pr.IsPreprocessOnly = true
}

func (pr *parserResult) dependencyOnlyCallback(flag string, _ []string) {
	pr.IsDependencyOnly = true
	pr.CompileArgs = append(pr.CompileArgs, flag)
}

func (pr *parserResult) printOnlyCallback(flag string, _ []string) {
	pr.IsPrintOnly = true
}

func (pr *parserResult) assembleOnlyCallback(_ string, _ []string) {
	pr.IsAssembleOnly = true
}

func (pr *parserResult) verboseFlagCallback(_ string, _ []string) {
	pr.IsVerbose = true
}

func (pr *parserResult) compileOnlyCallback(_ string, _ []string) {
	pr.IsCompileOnly = true
}

func (pr *parserResult) bitcodeOnlyCallback(_ string, _ []string) {
	pr.IsBitcodeOnly = true
}
func (pr *parserResult) emitLLVMCallback(_ string, _ []string) {
	pr.IsCompileOnly = true
	pr.IsEmitLLVM = true
}

func (pr *parserResult) linkUnaryCallback(flag string, _ []string) {
	pr.LinkArgs = append(pr.LinkArgs, flag)
}

func (pr *parserResult) skipCallback(flag string, _ []string) {

	//if flag=="-Wsuggest-override" {
	//fmt.Printf("compile link unary callback: %v\n", flag)
	//}
}
func (pr *parserResult) specialUnaryCallback(flag string, _ []string) {

	//must this be linker script file?
	logger.Warnf("linker script file: %v\n", flag)
	f, err := os.Open(flag)
	if err == nil {
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			//LogDebug("linker script line: %v\n", line)
			idx1 := strings.Index(line, "\"")
			idx2 := strings.LastIndex(line, "\"")
			if idx2 > idx1 {
				name := line[idx1+1 : idx2]
				LogDebug("linker script object name: %v\n", name)
				if strings.HasSuffix(name, ".o") {
					pr.ObjectFiles = append(pr.ObjectFiles, name)
				}
			}
		}

	} else {

		logger.Warnf("linker script file does not exist: %v\n", flag)
	}
}
func (pr *parserResult) compileUnaryCallback(flag string, _ []string) {
	pr.CompileArgs = append(pr.CompileArgs, flag)
}

func (pr *parserResult) warningLinkUnaryCallback(flag string, _ []string) {
	logger.Warnf("The flag cannot be used with this tool, we ignore it, else we lose the bitcode section. flag=%v",
		flag)
	pr.ForbiddenFlags = append(pr.ForbiddenFlags, flag)
}

func (pr *parserResult) defaultBinaryCallback(_ string, _ []string) {
	// Do nothing
}

func (pr *parserResult) dependencyBinaryCallback(flag string, args []string) {
	pr.CompileArgs = append(pr.CompileArgs, flag, args[0])
	pr.IsDependencyOnly = true
}

func (pr *parserResult) compileBinaryCallback(flag string, args []string) {
	if flag == "-arch" && strings.HasPrefix(args[0], "sm_") {
		return
	}
	if flag == "-x" && strings.HasPrefix(args[0], "cu") {
		pr.CompileArgs = append(pr.CompileArgs, flag, "cuda")
		pr.IsCuda = true
		return
	}
	if flag == "-x" && strings.HasPrefix(args[0], "c++") {
		if !pr.IsCC1 {
			return
		}
	}

	pr.CompileArgs = append(pr.CompileArgs, flag, args[0])
}
func (pr *parserResult) compileXclangBinaryCallback(flag string, args []string) {
	if flag == "-Xclang" {
		return
		/*
			if strings.HasPrefix(args[0],"-debug-info-kind=") {
				return
			} else if strings.HasPrefix(args[0],"find-bad-constructs"){
				return
			} else if strings.HasPrefix(args[0],"check-ipc"){
				return
			} else if strings.HasPrefix(args[0],"-plugin-arg-find-bad-constructs"){
				return
			}*/
	}
	pr.CompileArgs = append(pr.CompileArgs, flag, args[0])
}

func (pr *parserResult) linkBinaryCallback(flag string, args []string) {
	pr.LinkArgs = append(pr.LinkArgs, flag, args[0])
}

func (pr *parserResult) linkerPluginBinaryCallback(flag string, args []string) {
	//fmt.Printf("ld plugin: %v\n", args[0])
}
func (pr *parserResult) linkerPluginOptCallback(flag string, args []string) {
	//fmt.Printf("ld plugin-opt: %v\n", flag)
}

func (pr *parserResult) compileIntelOpenMPLinkUnaryCallback(flag string, _ []string) {
	if strings.HasPrefix(flag, "-qopenmp") {
		flag = "-fopenmp"
	}
	pr.LinkArgs = append(pr.LinkArgs, flag)
	pr.CompileArgs = append(pr.CompileArgs, flag)
}

func (pr *parserResult) compileLinkUnaryCallback(flag string, _ []string) {
	pr.LinkArgs = append(pr.LinkArgs, flag)
	//for cuda? gtest
	if strings.HasPrefix(flag, "-isystem=") {
		flag = "-I" + flag[9:]
	}
	pr.CompileArgs = append(pr.CompileArgs, flag)
}

func (pr *parserResult) compileLinkBinaryCallback(flag string, args []string) {
	pr.LinkArgs = append(pr.LinkArgs, flag, args[0])
	pr.CompileArgs = append(pr.CompileArgs, flag, args[0])
}
