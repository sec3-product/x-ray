package preprocessor

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/coderrect-inc/coderrect/util"
	"github.com/coderrect-inc/coderrect/util/logger"
	"golang.org/x/net/html/charset"
)

// PreprocessMFC is to tranlsate MFC code to be clang-compatible
// The code pattern we are dealing with is:
// ```
//   BEGIN_MESSAGE_MAP(CMapLamDataDrivePropDlg, CDialog)
// 	     ON_BN_CLICKED(IDC_TEST_BUTTON, OnTestButton)
// 	     ON_BN_CLICKED(IDC_OK, OnClickOk)
//   END_MESSAGE_MAP()
// ```
//
// The binded message handler like `OnClickOk` will cause compilation error in clang
// (error: call to non-static member function without an object argument)
//
// A clang-compatible fix would be translating them to:
// ```
//   BEGIN_MESSAGE_MAP(CMapLamDataDrivePropDlg, CDialog)
// 	     ON_BN_CLICKED(IDC_TEST_BUTTON, &ThisClass::OnTestButton)
// 	     ON_BN_CLICKED(IDC_OK, &ThisClass::OnClickOk)
//   END_MESSAGE_MAP()
// ```
//
// `ThisClass` is a typedef in macro `BEGIN_MESSAGE_MAP`
func PreprocessMFC(path string) (string, bool) {
	file, err := os.Open(path)
	if err != nil {
		logger.Errorf("Cannot find the source code file. error=%v", err)
		panic(err)
	}
	defer file.Close()

	found := false
	changed := false
	result := ""
	scanner := bufio.NewScanner(file)
	// Example:
	// 	"     ON_BN_CLICKED(IDC_TEST_BUTTON, &ThisClass::OnTestButton)"
	//
	// split the string in to two capture groups:
	// 1. (.*\(.*,\s*) -- match anything before the last comma (including the comma)
	// 2. (\w+\)) -- match things after the last comma and the closing parenthesis
	pattern := regexp.MustCompile(`(.*\(.*,\s*)(\w+\))`)
	for scanner.Scan() {
		line := scanner.Text()
		code := strings.TrimSpace(line)
		if strings.HasPrefix(code, "BEGIN_MESSAGE_MAP") {
			found = true
			result += line + "\n"
			continue
		}

		if strings.HasPrefix(code, "END_MESSAGE_MAP") {
			found = false
			result += line + "\n"
			continue
		}

		if found {
			// insert "&ThisClass::" before $2 (the second capture group, i.e., the last argument)
			result += pattern.ReplaceAllString(line, `$1&ThisClass::$2`) + "\n"
			changed = true
		} else {
			result += line + "\n"
		}
	}
	return result, changed
}

// Rewrite a single MFC file
// NOTE: we assume the MFC code pattern only resides in cpp files
func RewriteMFC(path string) {
	if !util.IsCppFile(path) {
		return
	}

	result, changed := PreprocessMFC(path)
	if changed {
		logger.Infof("Rewritting MFC code. file=%v", path)
		fmt.Printf("Rewritting MFC code. file=%v\n", path)
		ioutil.WriteFile(path, []byte(result), 0644)
	}
}

// DEPRECATED: Rewrite all MFC files under the directory
func RewriteMFCProject(dir string) {
	items, err := ioutil.ReadDir(dir)
	if err != nil {
		logger.Errorf("Cannot open the project directory. dir=%v", dir)
		return
	}

	for _, item := range items {
		path := filepath.Join(dir, item.Name())
		if item.IsDir() {
			RewriteMFCProject(path)
		} else {
			RewriteMFC(path)
		}
	}
}

func PreProcessWindowsProject(dir string) {
	items, err := ioutil.ReadDir(dir)
	if err != nil {
		logger.Errorf("Cannot open the project directory. dir=%v", dir)
		return
	}

	for _, item := range items {
		path := filepath.Join(dir, item.Name())
		if item.IsDir() {
			PreProcessWindowsProject(path)
		} else {
			RewriteMFC(path)
			RewriteUTF16(path)
		}
	}
}

func RewriteUTF16(path string) {
	if !util.IsSourceFile(path) {
		return
	}
	encoding := DetermineEncoding(path)
	if encoding == "utf-8" {
		return
	} else if strings.HasPrefix(encoding, "utf-16") {
		logger.Infof("Rewritting UTF-16 code. file=%v", path)
		fmt.Printf("Rewritting UTF-16 code. file=%v\n", path)
		err := ConvertUTF16ToUTF8(path)
		if err != nil {
			logger.Errorf("Failed to convert UTF-16 file to UTF-8. file=%v", path)
		}
	} else if encoding == "windows-1252" {
		// NOTE: we don't explicitly convert windows-1252 for now, because:
		// 1. For ASCII characters windows-1252 is the same as UTF-8
		// 2. `charset.DetermineEncoding` returns windows-1252 if all other encodings don't fit
		logger.Warnf("We do know convert windows-1252 for now. file=%v", path)
	} else if encoding == "UNKNOWN" {
		logger.Warnf("Cannot determine the file encoding. file=%v", path)
	} else {
		logger.Warnf("Encountered a file encoding we don't explicitly handle. file=%v", encoding)
	}
}

func DetermineEncoding(path string) string {
	f, _ := os.Open(path)
	defer f.Close()
	r := bufio.NewReader(f)
	if data, err := r.Peek(1024); err == nil {
		_, name, certain := charset.DetermineEncoding(data, "")
		if !certain {
			logger.Warnf("Uncertain about the file encoding. file=%v", path)
		}
		return name
	}
	return "UNKNOWN"
}

// Taken from https://gist.github.com/bradleypeabody/185b1d7ed6c0c2ab6cec
// Could also consider using "transform" package
func ConvertUTF16ToUTF8(path string) error {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		panic(err)
	}

	if len(b)%2 != 0 {
		panic("UTF-16 string must have even length byte slice!")
	}
	u16s := make([]uint16, 1)
	ret := &bytes.Buffer{}
	b8buf := make([]byte, 4)

	lb := len(b)
	for i := 0; i < lb; i += 2 {
		u16s[0] = uint16(b[i]) + (uint16(b[i+1]) << 8)
		r := utf16.Decode(u16s)
		n := utf8.EncodeRune(b8buf, r[0])
		ret.Write(b8buf[:n])
	}

	return ioutil.WriteFile(path, ret.Bytes(), 0644)
}
