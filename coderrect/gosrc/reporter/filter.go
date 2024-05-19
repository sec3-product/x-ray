package reporter

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/coderrect-inc/coderrect/util"
	"github.com/gobwas/glob"
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

type Filters struct {
	Lines     []Line
	Vars      []Variable
	Functions []Function
}

func (filter *Filters) size() int {
	return len(filter.Lines) + len(filter.Vars) + len(filter.Functions)
}

//NOTE: This is not used, because we will use conflib to read configuration file
func readConf(path string) Config {
	confData, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println("Fail to read reporter config file")
		panic(err)
	}
	var conf Config
	err = json.Unmarshal(confData, &conf)
	if err != nil {
		fmt.Println("Fail to parse the config file as JSON")
		panic(err)
	}
	return conf
}

func parseConf(rawConfig []string) Filters {
	var n Node
	var filters Filters
	for _, raw := range rawConfig {
		err := json.Unmarshal([]byte(raw), &n)
		if err != nil {
			fmt.Println("Fail to parse the filtes defined in coderrect.json")
			panic(err)
		}
		if n.NType == "function" {
			f := Function{n.SrcFile, n.CsFile, n.CsLine, n.Name}
			filters.Functions = append(filters.Functions, f)
		}

		if n.NType == "line" {
			l := Line{n.SrcFile, n.SrcLine, n.Function, n.Code}
			filters.Lines = append(filters.Lines, l)
		}

		if n.NType == "variable" {
			v := Variable{n.SrcFile, n.SrcLine, n.Name, n.Ty}
			filters.Vars = append(filters.Vars, v)
		}
	}
	return filters
}

// NOTE: Deprecated
func parse(conf Config) Filters {
	var filters Filters
	for _, n := range conf.ReportConf.Filters {
		if n.NType == "function" {
			f := Function{n.SrcFile, n.CsFile, n.CsLine, n.Name}
			filters.Functions = append(filters.Functions, f)
		}

		if n.NType == "line" {
			l := Line{n.SrcFile, n.SrcLine, n.Function, n.Code}
			filters.Lines = append(filters.Lines, l)
		}

		if n.NType == "variable" {
			v := Variable{n.SrcFile, n.SrcLine, n.NType, n.Ty}
			filters.Vars = append(filters.Vars, v)
		}
	}
	return filters
}

type trace struct {
	function string
	csFile   string
	csLine   int
	file     string
}

// parse the stack trace string to structs
func parseStackTrace(stacktrace []string) []*trace {
	var st []*trace
	var prev *trace
	var fun string
	var csFile string
	var csLine int
	var err error
	// pattern to match a function call in the stacktrace
	p := regexp.MustCompile(`(?P<fname>.+)\[(?P<callsite>.+)\:(?P<line>\d+)\]$`)
	for _, l := range stacktrace {
		// function "main" is special, it doesn't have callsite
		// thus not comply with the pattern
		if p.MatchString(l) {
			grps := p.FindStringSubmatch(l)
			// group "fname"
			fun = grps[1]
			// group "callsite"
			csFile = grps[2]
			// group "line"
			csLine, err = strconv.Atoi(grps[3])
			if err != nil {
				fmt.Println("Fail to convert stack trace line number to int, trace=%v\n", l)
				panic(err)
			}
		} else {
			fun = l
			csFile = ""
			csLine = 0
		}
		// NOTE:
		// The callsite file of the next level function call
		// is the file where the previous level function is defined
		// (the next level function must be called within the body of previous function)
		if prev != nil {
			prev.file = csFile
		}
		fun = strings.TrimSpace(fun)
		prev = &trace{fun, csFile, csLine, ""}
		st = append(st, prev)
	}
	return st
}

// NOTE: the frequent re-compilation may cause a performance issue
// if the number of race to filter is much larger than the filter,
// consider compile all glob at first
func filterStackTrace(traces []*trace, functions []Function) bool {
	var gName, gCsFile glob.Glob
	for _, f := range functions {
		for _, t := range traces {
			gName = glob.MustCompile(f.name)
			gCsFile = glob.MustCompile(f.csFile)
			if gName.Match(t.function) && gCsFile.Match(t.csFile) {
				return true
			}

			// match the file for function implementation
			// g = glob.MustCompile(f.file)
			// if g.Match(t.file) {
			// 	return true
			// }
		}
	}
	return false
}

// TODO: could there be more?
// keywords that could go before a type
// for example:
//  static item *heads[LARGEST_ID];
var keywords = []string{"static"}

// Parse the type information and name of the shared obj
// NOTE: this should be a bug in core engine
func parseObjTypeAndName(sourceLine string) (string, string) {
	// strip off the line number first
	code := parseCode(sourceLine)
	code = strings.TrimSpace(code)
	lst := strings.Split(code, " ")
	var ty, name string
	for i, tok := range lst {
		// skip keywords at the front
		if !util.Contains(keywords, tok) {
			ty = tok
			// check if type is a pointer
			// and get variable name
			if i+1 < len(lst) {
				next := lst[i+1]
				if next == "=" {
					next = ty
					ty = ""
				}
				for j, c := range next {
					if c == '*' {
						ty = ty + string(c)
					} else {
						name = next[j:]
						break
					}
				}
			}
			break
		}
	}
	return ty, name
}

func filterSharedObj(sharedObj *SharedObj, vars []Variable) bool {
	var gName, gFile glob.Glob
	for _, v := range vars {
		if v.name != "" {
			gName = glob.MustCompile(v.name)
		} else {
			gName = glob.MustCompile("*")
		}
		if v.file != "" {
			gFile = glob.MustCompile(v.file)
		} else {
			gFile = glob.MustCompile("*")
		}

		var name string
		if len(sharedObj.Name) == 0 {
			_, name = parseObjTypeAndName(sharedObj.SourceLine)
		} else {
			name = sharedObj.Name
		}

		if gName.Match(name) && gFile.Match(sharedObj.Filename) {
			return true
		}
	}
	return false
}

// NOTE: this may not needed
// Try to strip the line number in the source code line
// for example:
// change: " 451|    if (*head == it) {\n"
// to: " 451|    if (*head == it) {\n"
// This could prevent the case where the glob user want to filter has a constant number
func parseCode(code string) string {
	lst := strings.Split(code, "|")
	return strings.TrimSpace(lst[1])
}

func filterCode(threadAcc *ThreadAccess, lines []Line) bool {
	code := parseCode(threadAcc.SourceLine)
	var gCode, gFile glob.Glob
	for _, l := range lines {
		if l.code != "" {
			gCode = glob.MustCompile(l.code)
		} else {
			gCode = glob.MustCompile("*")
		}
		if l.file != "" {
			gFile = glob.MustCompile("*" + l.file)
		} else {
			gFile = glob.MustCompile("*")
		}
		if gCode.Match(code) && gFile.Match(threadAcc.Dir+"/"+threadAcc.Filename) {
			return true
		}
	}
	return false
}

// FIXME: for now we only filter by stacktrace
func filterThreadAccess(threadAcc *ThreadAccess, lines []Line, functions []Function) bool {
	st := parseStackTrace(threadAcc.Stacktrace)
	return filterCode(threadAcc, lines) || filterStackTrace(st, functions)
}

func apply(report FullReport, filters Filters) FullReport {
	newDataRaces := make([]DataRace, 0)
	for _, race := range report.DataRaces {
		if filterSharedObj(&race.SharedObj, filters.Vars) ||
			filterThreadAccess(&race.Access1, filters.Lines, filters.Functions) ||
			filterThreadAccess(&race.Access2, filters.Lines, filters.Functions) {
			continue
		} else {
			newDataRaces = append(newDataRaces, race)
		}
	}

	report.DataRaces = newDataRaces
	return report
}

// Rewrite takes the path of rawJSON race report and the configurations for filter nodes
// The return value is the number of data races after the filtering
// -1 means the filter is not performed
// FIXME: the return value should be the number of all types of bugs!
func Rewrite(raceJSONPath string, rawConfig []string) int {
	filters := parseConf(rawConfig)
	if filters.size() == 0 {
		return -1
	}

	raceJSONBytes, err := ioutil.ReadFile(raceJSONPath)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to read the data file - %v\n", err))
		panic(err)
	}

	var raceReport FullReport
	if err = json.Unmarshal(raceJSONBytes, &raceReport); err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to parse the data file - %v\n", err))
		panic(err)
	}

	filteredReport := apply(raceReport, filters)
	b, err := json.Marshal(filteredReport)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to marshall the filtered race report into JSON"))
		panic(err)
	}
	// fmt.Println("Final JSON:")
	// fmt.Println(string(b))
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
	return filteredReport.bugNum()
}
