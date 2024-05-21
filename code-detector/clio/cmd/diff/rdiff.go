package main

import (
	"clio/internal/diff"
	"fmt"
	"os"
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
	if len(os.Args) < 2 || os.Args[1] == "-h" || os.Args[1] == "--help" {
		fmt.Println(helpMsg)
		return
	}

	report, err := diff.GetReport(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating diffs: %v", err)
	}

	fmt.Println(report)
}
