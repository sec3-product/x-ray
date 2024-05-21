package main

import (
	"clio/internal/regress"
	"fmt"
	"os"
)

const helpMsg = `usage: check [--help] <path-to-races.json> <truth> [name]

check ensures the races specified in the "truth" file are present in a races.json file

The truth file should contain a list of races (one per line) in the format:
file1 line1 file2 line2

Optionally, a truth file may contain races for multiple projects.
Each project must be given a name in the format "[project-name]"
If the truth file contains multiple projects, name must specify the project

Example truth file:
[project1]
main.cpp 12 main.cpp 120
main.cpp 124 main.cpp 126

[project2]
sample.cpp 8 runner.cpp 42

Example usage:
./check races.json truth.txt project1

`

func main() {
	if len(os.Args) < 3 || os.Args[1] == "-h" || os.Args[1] == "--help" {
		fmt.Println(helpMsg)
		return
	}

	racesFile := os.Args[1]
	truthFile := os.Args[2]
	projectName := ""
	if len(os.Args) > 2 {
		projectName = os.Args[3]
	}

	expected, err := regress.ReadTruthRaces(truthFile, projectName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not read truth file: %v\n", err)
		os.Exit(1)
	}

	actual, err := regress.ReadJSONRaces(racesFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not read json file: %v\n", err)
		os.Exit(1)
	}

	missing := regress.CheckMissing(actual, expected)
	if len(missing) == 0 {
		fmt.Println("PASS")
		return
	}

	fmt.Println("FAILED")
	for _, r := range missing {
		fmt.Printf("Missed: %v %v %v %v\n", r.Access1.Filename, r.Access1.Line, r.Access2.Filename, r.Access2.Line)
	}
	os.Exit(1)
}
