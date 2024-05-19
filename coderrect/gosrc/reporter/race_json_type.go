package reporter

import "strings"

// ExecutableInfo describes the analysis summary for a executable
type ExecutableInfo struct {
	Name           string
	RaceJSON       string
	DataRaces      int
	RaceConditions int
	NewBugs        bool
}

// IndexInfo describes the format of index.json
type IndexInfo struct {
	Executables  []ExecutableInfo
	CoderrectVer string //current version of coderrect
}

type ThreadAccess struct {
	Col        int      `json:"col"`
	Dir        string   `json:"dir"`
	Filename   string   `json:"filename"`
	Line       int      `json:"line"`
	Snippet    string   `json:"snippet"`
	SourceLine string   `json:"sourceLine"`
	Stacktrace []string `json:"stacktrace"`
}

type SharedObj struct {
	Dir        string `json:"dir"`
	Filename   string `json:"filename"`
	Line       int    `json:"line"`
	Name       string `json:"name"`
	SourceLine string `json:"sourceLine"`
}

type DataRace struct {
	SharedObj SharedObj    `json:"sharedObj"`
	Access1   ThreadAccess `json:"access1"`
	Access2   ThreadAccess `json:"access2"`
	IsOmpRace bool         `json:"isOmpRace"`
	Priority  int          `json:"priority"`
}

// type DeadLock struct {
// }

type MismatchedAPI struct {
	ErrorMsg string       `json:"errorMsg"`
	Inst     ThreadAccess `json:"inst"`
	Priority int          `json:"priority"`
}
type UntrustfulAccount struct {
	ErrorMsg string       `json:"errorMsg"`
	Inst     ThreadAccess `json:"inst"`
	Priority int          `json:"priority"`
}

type UnsafeOperation struct {
	ErrorMsg string       `json:"errorMsg"`
	Inst     ThreadAccess `json:"inst"`
	Priority int          `json:"priority"`
}
type CosplayAccount struct {
	ErrorMsg string       `json:"errorMsg"`
	Inst     ThreadAccess `json:"inst"`
	Priority int          `json:"priority"`
}
type RaceCondition struct {
	SharedObj SharedObj    `json:"sharedObj"`
	Access1   ThreadAccess `json:"access1"`
	Access2   ThreadAccess `json:"access2"`
	Access3   ThreadAccess `json:"access3"`
	IsAV      bool         `json:"isAV"`
	Priority  int          `json:"priority"`
}

type RaceReport struct {
	DataRaces      []DataRace      `json:"dataRaces"`
	TocTous        []DataRace      `json:"toctou"`
	RaceConditions []RaceCondition `json:"raceConditions"`
}

type FullReport struct {
	Version            int                 `json:"version"`
	Bcfile             string              `json:"bcfile"`
	GeneratedAt        string              `json:"generatedAt"`
	DataRaces          []DataRace          `json:"dataRaces"`
	TocTous            []DataRace          `json:"toctou"`
	MismatchedAPIs     []MismatchedAPI     `json:"mismatchedAPIs"`
	UntrustfulAccounts []UntrustfulAccount `json:"untrustfulAccounts"`
	UnsafeOperations   []UnsafeOperation   `json:"unsafeOperations"`
	CosplayAccounts    []CosplayAccount    `json:"cosplayAccounts"`
	RaceConditions     []RaceCondition     `json:"raceConditions"`
	// Deadlocks      []DeadLock
}

func (r *FullReport) bugNum() int {
	return len(r.DataRaces) + len(r.MismatchedAPIs) + len(r.UntrustfulAccounts) + len(r.UnsafeOperations) + len(r.CosplayAccounts) + len(r.TocTous)
}

func removeFolderName(line string) string {
	idx := strings.LastIndex(line, "/")
	if idx != -1 {
		return line[idx+1:]
	}
	return line
}
func removeLineNumber(line string) string {
	idx := strings.Index(line, "|")
	if idx != -1 {
		return line[idx+1:]
	}
	return line
}

func normalizeCode(codeSnippet string) string {
	var builder strings.Builder

	lines := strings.Split(codeSnippet, "\n")
	for _, line := range lines {
		l := strings.TrimSpace(removeLineNumber(line))
		if len(l) <= 0 {
			continue
		}
		builder.WriteString(l)
		builder.WriteString("\n")
	}

	return builder.String()
}

func normalizeStackTrace(stackTrace []string) string {
	var builder strings.Builder

	for _, v := range stackTrace {
		s := strings.TrimSpace(v)
		// Rust stacktrace sometimes contains empty string
		if s == "" {
			continue
		}
		if s[len(s)-1] == ']' {
			for k := len(s) - 1; k > 0; k-- {
				if s[k] == '[' {
					s = s[:k]
					break
				}
			}
		}
		builder.WriteString(s)
		builder.WriteString("\n")
	}

	return builder.String()
}

func (r *DataRace) ComputeSignature() string {
	var builder strings.Builder

	builder.WriteString(strings.TrimSpace(removeFolderName(r.SharedObj.Filename)))
	builder.WriteString(strings.TrimSpace(removeLineNumber(r.SharedObj.SourceLine)))
	builder.WriteString("\n")

	builder.WriteString(strings.TrimSpace(removeFolderName(r.Access1.Filename)))
	builder.WriteString(normalizeCode(r.Access1.Snippet))
	builder.WriteString(normalizeStackTrace(r.Access1.Stacktrace))

	builder.WriteString(strings.TrimSpace(removeFolderName(r.Access2.Filename)))
	builder.WriteString(normalizeCode(r.Access2.Snippet))
	builder.WriteString(normalizeStackTrace(r.Access2.Stacktrace))
	return builder.String()
}

func (r *RaceCondition) ComputeSignature() string {
	var builder strings.Builder
	builder.WriteString(strings.TrimSpace(removeFolderName(r.SharedObj.Filename)))
	builder.WriteString(strings.TrimSpace(removeLineNumber(r.SharedObj.SourceLine)))
	builder.WriteString("\n")

	builder.WriteString(strings.TrimSpace(removeFolderName(r.Access1.Filename)))
	builder.WriteString(normalizeCode(r.Access1.Snippet))
	builder.WriteString(normalizeStackTrace(r.Access1.Stacktrace))

	builder.WriteString(strings.TrimSpace(removeFolderName(r.Access2.Filename)))
	builder.WriteString(normalizeCode(r.Access2.Snippet))
	builder.WriteString(normalizeStackTrace(r.Access2.Stacktrace))

	builder.WriteString(strings.TrimSpace(removeFolderName(r.Access3.Filename)))
	builder.WriteString(normalizeCode(r.Access3.Snippet))
	builder.WriteString(normalizeStackTrace(r.Access3.Stacktrace))
	return builder.String()
}
func (m *MismatchedAPI) ComputeSignature() string {
	var builder strings.Builder

	builder.WriteString(strings.TrimSpace(removeFolderName(m.Inst.Filename)))
	builder.WriteString(normalizeCode(m.Inst.Snippet))
	builder.WriteString(normalizeStackTrace(m.Inst.Stacktrace))

	return builder.String()
}
func (m *UntrustfulAccount) ComputeSignature() string {
	var builder strings.Builder

	builder.WriteString(m.Inst.Filename)
	builder.WriteString(normalizeCode(m.Inst.Snippet))
	builder.WriteString(normalizeStackTrace(m.Inst.Stacktrace))

	return builder.String()
}
func (m *UnsafeOperation) ComputeSignature() string {
	var builder strings.Builder

	builder.WriteString(m.Inst.Filename)
	builder.WriteString(normalizeCode(m.Inst.Snippet))
	builder.WriteString(normalizeStackTrace(m.Inst.Stacktrace))

	return builder.String()
}
func (m *CosplayAccount) ComputeSignature() string {
	var builder strings.Builder

	builder.WriteString(m.Inst.Filename)
	builder.WriteString(normalizeCode(m.Inst.Snippet))
	builder.WriteString(normalizeStackTrace(m.Inst.Stacktrace))

	return builder.String()
}
