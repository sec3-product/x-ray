package reporter

import (
	"encoding/json"
	"fmt"
	"os"
)

// data structures for deserializing report config json

// overall config
// for now there's only one subfield report
// maybe we'll add more in the future
type Config struct {
	ReportConf ReportConfig `json:"report"`
}

// repoter related config
// now there's only Filters
// Filters consists of a list of Nodes
type ReportConfig struct {
	Filters []Node `json:"filters"`
}

// Node is the where user specify filter instructions
// it can be either a Line/Variable/Function, based on the value of NType
type Node struct {
	NType    string `json:"node"`
	Name     string `json:"name"`
	SrcFile  string `json:"sourceFile"`
	SrcLine  int    `json:"line"`
	CsFile   string `json:"callsiteSourceFile"`
	CsLine   int    `json:"callsiteLine"`
	Function string `json:"function"`
	Ty       string `json:"type"`
	Code     string `json:"code"`
}

// below are the internal representation of different types of Nodes
// FIXME: the `file` field for Line node is actually dir + filename
// because sometimes user want to filter some library code based on the path
// Later we should separate this into two fields: `dir` and `file`
type Line struct {
	file     string
	line     int
	function string
	code     string
}

// NOTE: `line` is not used for now
type Variable struct {
	file string
	line int
	name string
	ty   string
}

type Function struct {
	file   string
	csFile string
	csLine int
	name   string
}

// Rewrite takes the path of rawJSON race report and the configurations for
// filter nodes. It returns the number of issues.
func Rewrite(raceJSONPath string) int {
	raceJSONBytes, err := os.ReadFile(raceJSONPath)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to read the data file - %v\n", err))
		panic(err)
	}

	var report FullReport
	if err = json.Unmarshal(raceJSONBytes, &report); err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to parse the data file - %v\n", err))
		panic(err)
	}

	b, err := json.Marshal(report)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to marshall the filtered race report into JSON"))
		panic(err)
	}
	file, err := os.OpenFile(raceJSONPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	defer file.Close()
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to open JSON race report for rewriting"))
		panic(err)
	}
	// overwrite the old raw JSON
	err = file.Truncate(0)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to truncate original JSON report"))
		panic(err)
	}
	// re-write the filtered JSON
	_, err = file.Write(b)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to rewrite JSON race report"))
		panic(err)
	}
	return report.bugNum()
}
