package main

import (
	"fmt"
	"os"
	"path/filepath"
)

const helpMsg = `usage: rdiff [--help] <path-to-races.json> <races.json2> [<races3.json>, ...]

rdiff displays the difference between two or more races.json files
The first races.json passed is assumed to be the most recent and the last argument is assumed to be the oldest.

For each file the number of races detected is shown along with:
	new:   Races in this report that were not in any previous report
	gone:  Races that were in previous report but not this one
	renew: Races in this report that have been seen before, but were not in the previous report
`

func main() {
	// Read all ll and bc files from ./bcs
	bcs, err := filepath.Glob("./bcs/*.bc")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading './bcs': %v", err)
	}

	lls, err := filepath.Glob("./bcs/*.ll")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading './bcs': %v\n", err)
	}

	bcs = append(bcs, lls...)

	for i, bc := range bcs {
		fmt.Printf("%d\t%s", i, bc)
	}

	// Run racedetect on all bc/ll
	exec.Command("racedetect")
	// check for expected TP
	// Compare to previous version (if any)
}
