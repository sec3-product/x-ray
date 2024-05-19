package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"
)

const (
	irDir        = "./ir"
	irFortran    = "./ir-fortran"
	expectedFile = "truth.json"
	numTests     = 116

	colorReset = "\033[0m"
	colorRed   = "\033[31m"
)

type access struct {
	Line, Col uint
}

func (a access) Equals(other access) bool {
	return a.Line == other.Line && a.Col == other.Col
}

func (a access) EqualsLine(other access) bool {
	return a.Line == other.Line
}

func (a access) String() string {
	return fmt.Sprintf("%d:%d", a.Line, a.Col)
}

type race struct {
	Access1, Access2 access
}

func (r race) Equals(other race) bool {
	return (r.Access1.Equals(other.Access1) && r.Access2.Equals(other.Access2)) || (r.Access1.Equals(other.Access2) && r.Access2.Equals(other.Access1))
}

func (r race) EqualsLine(other race) bool {
	return (r.Access1.EqualsLine(other.Access1) && r.Access2.EqualsLine(other.Access2)) || (r.Access1.EqualsLine(other.Access2) && r.Access2.EqualsLine(other.Access1))
}

func (r race) String() string {
	return fmt.Sprintf("(%v %v)", r.Access1, r.Access2)
}

type racereport struct {
	DataRaces []race
}

type expectedreport struct {
	Races []race
}

func loadExpected(file string) map[string]expectedreport {
	jsondata, err := ioutil.ReadFile(file)
	if err != nil {
		panic(err)
	}

	var expected map[string]expectedreport
	if err := json.Unmarshal(jsondata, &expected); err != nil {
		panic(err)
	}

	return expected
}

// Given path to file return test name
func getTestName(path string) string {
	filename := filepath.Base(path)
	extension := filepath.Ext(filename)
	name := filename[0 : len(filename)-len(extension)]
	return name
}

// return tre if the both slices contain the same races
func equalRaces(r1, r2 []race) bool {
	if len(r1) != len(r2) {
		return false
	}

	for _, race1 := range r1 {
		found := false
		for _, race2 := range r2 {
			if race1.Equals(race2) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	for _, race1 := range r2 {
		found := false
		for _, race2 := range r1 {
			if race1.Equals(race2) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// return true if all races in expected are also in actual (by line only)
func equalLineRaces(actual, expected []race) bool {
	nolines := map[race]bool{}
	for _, r := range expected {
		r.Access1.Col = 0
		r.Access2.Col = 0
		nolines[r] = false
	}

	for _, actualRace := range actual {
		found := false
		for expectedRace := range nolines {
			if actualRace.EqualsLine(expectedRace) {
				found = true
				nolines[expectedRace] = true
				break
			}
		}
		if !found {
			return false
		}
	}

	for _, found := range nolines {
		if !found {
			return false
		}
	}

	return true
}

func generateIR() {
	// Make directory if not exist
	if _, err := os.Stat(irDir); os.IsNotExist(err) {
		os.Mkdir(irDir, 0777)
	}

	crHome := os.Getenv("CODERRECT_HOME")
	if crHome == "" {
		panic("$CODERRECT_HOME must be set to generate ir")
	}

	clangPath := filepath.Join(crHome, "clang/bin/clang")
	clangppPath := filepath.Join(crHome, "clang/bin/clang++")
	// clangPath := "coderrect-clang"
	// clangppPath := "coderrect-clang++"

	cFlags := strings.Fields("-g -O1 -Wall -std=c99 -fopenmp -fopenmp-version=45 -fno-discard-value-names -O1 -g -S -emit-llvm -mllvm -disable-llvm-optzns -mllvm -instcombine-lower-dbg-declare=0")
	ctests, err := filepath.Glob("./micro-benchmarks/*.c")
	if err != nil {
		panic(err)
	}

	cppFlags := strings.Fields("-g -O1 -Wall -fopenmp -fopenmp-version=45 -fno-discard-value-names -O1 -g -S -emit-llvm -mllvm -disable-llvm-optzns -mllvm -instcombine-lower-dbg-declare=0")
	cpptests, err := filepath.Glob("./micro-benchmarks/*.cpp")
	if err != nil {
		panic(err)
	}

	for _, test := range ctests {
		name := getTestName(test)
		outPath := filepath.Join(irDir, name) + ".ll"
		args := make([]string, len(cFlags))
		copy(args, cFlags)
		args = append(args, []string{test, "-o", outPath}...)
		cmd := exec.Command(clangPath, args...)
		fmt.Println("Building", name)
		out, err := cmd.CombinedOutput()
		//if err = cmd.Run(); err != nil {
		if err != nil {
			fmt.Print("out", string(out))
			fmt.Println(err)
		}
	}

	for _, test := range cpptests {
		name := getTestName(test)
		outPath := filepath.Join(irDir, name) + ".ll"
		args := make([]string, len(cppFlags))
		copy(args, cppFlags)
		args = append(args, []string{test, "-o", outPath}...)
		cmd := exec.Command(clangppPath, args...)
		fmt.Println("Building", name)
		out, err := cmd.CombinedOutput()
		//if err = cmd.Run(); err != nil {
		if err != nil {
			fmt.Print("out", string(out))
			fmt.Println(err)
		}
	}
}

func generateFortranIR() {
	// Make directory if not exist
	if _, err := os.Stat(irFortran); os.IsNotExist(err) {
		os.Mkdir(irFortran, 0777)
	}

	flangPath := "/workspaces/flang/build/install/bin/flang"
	flags := []string{"-fopenmp", "-S", "-emit-llvm", "-g", "-fno-discard-value-names"}
	tests, err := filepath.Glob("./micro-benchmarks-fortran/*.f95")
	if err != nil {
		panic(err)
	}

	for _, test := range tests {
		name := getTestName(test)
		outPath := filepath.Join(irFortran, name) + ".ll"
		args := make([]string, len(flags))
		copy(args, flags)
		args = append(args, []string{test, "-o", outPath}...)
		cmd := exec.Command(flangPath, args...)
		fmt.Println("Building", name)
		out, err := cmd.CombinedOutput()
		//if err = cmd.Run(); err != nil {
		if err != nil {
			fmt.Print("out", string(out))
			fmt.Println(err)
		}
	}
}

func getTestFiles() []string {
	files, err := filepath.Glob(irDir + "/*.ll")
	if err != nil {
		panic(err)
	}
	if len(files) == numTests {
		return files
	}

	// We are missing some tests, try rebuilding
	//generateIR()

	files, err = filepath.Glob(irDir + "/*.ll")
	if err != nil {
		panic(err)
	}
	return files

	// if len(files) == numTests {
	// 	return files
	// }

	// panic("There was a problem building the tests")
}

// TestResult represents the result of a group of tests
type TestResult struct {
	yesPass uint
	yesFail uint
	noPass  uint
	noFail  uint
	Crash   uint
}

func (l TestResult) combine(r TestResult) TestResult {
	return TestResult{
		yesPass: l.yesPass + r.yesPass,
		yesFail: l.yesFail + r.yesFail,
		noPass:  l.noPass + r.noPass,
		noFail:  l.noFail + r.noFail,
		Crash:   l.Crash + r.Crash,
	}
}

func main() {
	// read in expected results
	expected := loadExpected(expectedFile)

	// Get list of test files
	files := getTestFiles()
	//generateFortranIR()

	var cppResult TestResult

	// Run cpp tests
	fmt.Println("=== C/C++ ===")
	fmt.Println("Result Time    \tTest")
	for _, file := range files {
		// Run racedetect
		start := time.Now()
		cmd := exec.Command("racedetect", "--no-progress", "--no-filter", file)
		_, err := cmd.CombinedOutput()
		if err != nil {
			panic(err)
		}
		elapsed := time.Since(start)

		// Load results from races.json
		jsondata, err := ioutil.ReadFile("races.json")
		if err != nil {
			panic(err)
		}

		// Read results into struct
		var report racereport
		if err := json.Unmarshal(jsondata, &report); err != nil {
			panic(err)
		}

		raceCount := len(report.DataRaces)
		testName := getTestName(file)

		// Check if results are as expected
		errMsg := ""
		result := "FAIL"
		if strings.HasSuffix(file, "no.ll") && raceCount == 0 {
			result = "PASS"
		} else if strings.HasSuffix(file, "yes.ll") {
			// Check the races
			expectedRaces := expected[testName].Races
			actualRaces := report.DataRaces
			if equalRaces(actualRaces, expectedRaces) {
				result = "PASS"
			} else {
				errMsg = fmt.Sprintf("%s       Expected: %+v\n       Actual: %+v%s\n", colorRed, expectedRaces, actualRaces, colorReset)
			}
		}

		// Count result
		if strings.HasSuffix(file, "no.ll") {
			if result == "PASS" {
				cppResult.noPass++
			} else {
				cppResult.noFail++
			}
		} else {
			if result == "PASS" {
				cppResult.yesPass++
			} else {
				cppResult.yesFail++
			}
		}

		fmt.Printf("%s   %s\t%s\n", result, elapsed, file)
		if errMsg != "" {
			fmt.Printf(errMsg)
		}
	}

	// Run fortran tests
	var fortranResult TestResult
	testingFortran := false
	if _, err := os.Stat(irFortran); err == nil {
		expected = loadExpected("truth-fortran.json")

		files, err := filepath.Glob(irFortran + "/*.ll")
		if err != nil {
			panic(err)
		}

		testingFortran = true
		fmt.Println("=== Fortran ===")
		fmt.Println("Result Time    \tTest")
		for _, file := range files {
			// Run racedetect
			start := time.Now()
			cmd := exec.Command("racedetect", "--no-progress", "--no-filter", "-Xuse-fi-model", file)
			_, err := cmd.CombinedOutput()
			if err != nil {
				panic(err)
			}
			elapsed := time.Since(start)

			// Load results from races.json
			jsondata, err := ioutil.ReadFile("races.json")
			if err != nil {
				panic(err)
			}

			// Read results into struct
			var report racereport
			if err := json.Unmarshal(jsondata, &report); err != nil {
				panic(err)
			}

			raceCount := len(report.DataRaces)
			testName := getTestName(file)

			// Check if results are as expected
			errMsg := ""
			result := "FAIL"
			if strings.HasSuffix(file, "no.ll") && raceCount == 0 {
				result = "PASS"
			} else if strings.HasSuffix(file, "yes.ll") {
				// Check the races
				expectedRaces := expected[testName].Races
				actualRaces := report.DataRaces
				if equalLineRaces(actualRaces, expectedRaces) {
					result = "PASS"
				} else {
					errMsg = fmt.Sprintf("%s       Expected: %+v\n       Actual: %+v%s\n", colorRed, expectedRaces, actualRaces, colorReset)
				}
				//fmt.Printf("       Expected: %+v\n       Actual: %+v\n", expectedRaces, actualRaces)
			}

			// Count result
			if strings.HasSuffix(file, "no.ll") {
				if result == "PASS" {
					fortranResult.noPass++
				} else {
					fortranResult.noFail++
				}
			} else {
				if result == "PASS" {
					fortranResult.yesPass++
				} else {
					fortranResult.yesFail++
				}
			}

			fmt.Printf("%s   %s\t%s\n", result, elapsed, file)
			if errMsg != "" {
				fmt.Printf(errMsg)
			}
		}
	}

	// Print final results
	fmt.Println("")
	w := tabwriter.NewWriter(os.Stdout, 1, 1, 1, ' ', 0)
	fmt.Println("=== C/C++ ===")
	fmt.Fprintln(w, "\tPass\tFail")
	fmt.Fprintf(w, "Yes\t%d\t%d\n", cppResult.yesPass, cppResult.yesFail)
	fmt.Fprintf(w, "No\t%d\t%d\n", cppResult.noPass, cppResult.noFail)
	fmt.Fprintf(w, "Total\t%d\t%d\n", cppResult.yesPass+cppResult.noPass, cppResult.yesFail+cppResult.noFail)
	w.Flush()
	if testingFortran {
		fmt.Println("=== Fortran ===")
		fmt.Fprintln(w, "\tPass*\tFail")
		fmt.Fprintf(w, "Yes\t%d\t%d\n", fortranResult.yesPass, fortranResult.yesFail)
		fmt.Fprintf(w, "No\t%d\t%d\n", fortranResult.noPass, fortranResult.noFail)
		fmt.Fprintf(w, "Total\t%d\t%d\n", fortranResult.yesPass+fortranResult.noPass, fortranResult.yesFail+fortranResult.noFail)
		w.Flush()
		fmt.Println("*Only checked by line number.\n Not for the correct race.")

		totalResult := cppResult.combine(fortranResult)
		fmt.Println("=== Combined Results ===")
		fmt.Fprintln(w, "\tPass\tFail")
		fmt.Fprintf(w, "Yes\t%d\t%d\n", totalResult.yesPass, totalResult.yesFail)
		fmt.Fprintf(w, "No\t%d\t%d\n", totalResult.noPass, totalResult.noFail)
		fmt.Fprintf(w, "Total\t%d\t%d\n", totalResult.yesPass+totalResult.noPass, totalResult.yesFail+totalResult.noFail)
		w.Flush()
	}

}
