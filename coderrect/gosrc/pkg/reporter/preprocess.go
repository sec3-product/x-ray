// The preprocessor for Race Detection Report
// Mainly used for collecting source code information
// and formatting debugging information in the raw json files.

package reporter

import (
	"os"
	"fmt"
	"bufio"
	"path/filepath"
)

// GetLine retrieves the source code line based on line number and path provided
// @lineNum - line number of the line we want to retrieve
// @path 	- once the string list is joined by file separator, 
//  	   	  they should be the path to the source file
// Usage:
//  GetLine(1, path)
//  GetLine(1, dir, filename)
func GetLine(lineNum int, path...string) string {
	lines:= GetSnippetAtRange(lineNum, 0, 0, path...)
	if lines == nil {
		return ""
	} else if len(lines) == 1 {
		return lines[0]
	} else {
		panic("[REPORTER][INTERNAL ERROR]: GetSnippetAtRange should only return a single-element list when called within GetLine!")
	}
}

// GetSnippet retrieves a 5-line source code snippet, with 2 extra lines before and after the target line
func GetSnippet(lineNum int, path...string) []string {
	return GetSnippetAtRange(lineNum, 2, 2, path...)
}

// RenderSnippet renders a 5-line source code snippet for terminal report
func RenderSnippet(lineNum int, path...string) string {
	return RenderSnippetAtRange(lineNum, 2, 2, path...)
}

// GetSnippetAtRange retrieves an arbitrary length code snippet
// @lineNum - the target line number we want to get a code snippet for
// @before  - number of lines before the target line
// @after   - number of lines after the target line
// @path    - once the string list is joined by file separator, 
//  	      they should be the path to the source file
// Return Value:
//  The return value include the code snippet (a list of string)
// Usage:
//  GetSnippetAtRange(8, 2, 2, path)
//  GetSnippetAtRange(8, 2, 2, dir, filename)
// NOTE: Reading line-by-line is implemented using scanner.
// Scanner does not deal well with lines longer than 65536 chars.
func GetSnippetAtRange(lineNum, before, after int, path...string) []string {
	// lineNum is set to 0 when we fail to find debugging info
	if lineNum == 0 {
		return nil
	}

	// get the path of the source file
	p := filepath.Join(path...)

	file, err := os.Open(p)
	if err != nil {
		fmt.Errorf("Cannot find the source code file. error=%v", err)
		return nil
	}
	defer file.Close()

	// total length of the code snippet
	length := before + after + 1
	// the last line number of the code snippet
	end := lineNum + after

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() && end > 0 {
		if (end <= length) {
			lines = append(lines, scanner.Text())
		}
		end--
	}

	// compute the starting line number of the code snippet (for rendering)
	start := lineNum - before
	if (start <= 0) {
		start = 1
	}

	return lines
}

// RenderSnippetAtRange renders an arbitrary length code snippet for terminal report
// @lineNum - the target line number we want to get a code snippet for
// @before  - number of lines before the target line
// @after   - number of lines after the target line
// @path    - once the string list is joined by file separator, 
//  	      they should be the path to the source file
// Return Value:
//  The return value is the rendered string to print to terminal
// Usage:
//  RenderSnippetAtRange(8, 2, 2, path)
//  RenderSnippetAtRange(8, 2, 2, dir, filename)
func RenderSnippetAtRange(lineNum, before, after int, path...string) string {
	snippet := GetSnippetAtRange(lineNum, before, after, path...)
	start := lineNum - before
	if (start <= 0) {
		start = 1
	}

	// compute string padding
	end := start + len(snippet)
	padding := digit(end)

	result := ""
	for i, line := range snippet {
		current := start+i
		if current == lineNum {
			result += fmt.Sprintf(">%*d| %s", padding, current, line)
		} else {
			result += fmt.Sprintf(" %*d| %s", padding, current, line)
		}

		if len(snippet)-1 != i {
			result += "\n"
		}
	}
	return result
}

func digit(num int) int {
	d := 0
	for num != 0 {
		num /= 10
		d++
	}
	return d
}
