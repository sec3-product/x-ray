package reporter

import "strings"

// ExecutableInfo describes the analysis summary for an analyzed file.
type ExecutableInfo struct {
	Name       string
	ReportJSON string
	IssueCount int
}

// IndexInfo describes the format of index.json.
type IndexInfo struct {
	Executables  []ExecutableInfo
	CoderrectVer string
}

type Instruction struct {
	Col        int      `json:"col"`
	Dir        string   `json:"dir"`
	Filename   string   `json:"filename"`
	Line       int      `json:"line"`
	Snippet    string   `json:"snippet"`
	SourceLine string   `json:"sourceLine"`
	Stacktrace []string `json:"stacktrace"`
}

type UntrustfulAccount struct {
	ErrorMsg string      `json:"errorMsg"`
	Inst     Instruction `json:"inst"`
	Priority int         `json:"priority"`
}

type UnsafeOperation struct {
	ErrorMsg string      `json:"errorMsg"`
	Inst     Instruction `json:"inst"`
	Priority int         `json:"priority"`
}

type CosplayAccount struct {
	ErrorMsg string      `json:"errorMsg"`
	Inst     Instruction `json:"inst"`
	Priority int         `json:"priority"`
}

type FullReport struct {
	Version            int                 `json:"version"`
	IRFile             string              `json:"irFile"`
	GeneratedAt        string              `json:"generatedAt"`
	UntrustfulAccounts []UntrustfulAccount `json:"untrustfulAccounts"`
	UnsafeOperations   []UnsafeOperation   `json:"unsafeOperations"`
	CosplayAccounts    []CosplayAccount    `json:"cosplayAccounts"`
}

func (r *FullReport) bugNum() int {
	return len(r.UntrustfulAccounts) + len(r.UnsafeOperations) + len(r.CosplayAccounts)
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
