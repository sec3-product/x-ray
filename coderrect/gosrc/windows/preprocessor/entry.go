package preprocessor

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/coderrect-inc/coderrect/util"
	"github.com/coderrect-inc/coderrect/util/logger"
)

type OpenLibConfig struct {
	OpenLib EntryPoints `json:"openlib"`
}

type EntryPoints struct {
	Entries []string `json:"entryPoints"`
}

func FindAllEntries(path string) ([]string, error) {
	items, err := ioutil.ReadDir(path)
	if err != nil {
		logger.Errorf("Cannot open the project directory. dir=%v", path)
		return []string{}, fmt.Errorf("Cannot open the project directory. dir=%v", path)
	}

	var entries []string
	for _, item := range items {
		path := filepath.Join(path, item.Name())
		if item.IsDir() {
			subs, _ := FindAllEntries(path)
			entries = append(entries, subs...)
		} else {
			entries = append(entries, findEntries(path)...)
		}
	}

	var entreisNoDup []string
	keys := make(map[string]bool)
	for _, e := range entries {
		if _, value := keys[e]; !value {
			keys[e] = true
			entreisNoDup = append(entreisNoDup, e)
		}
	}
	return entreisNoDup, nil
}

func EntriesToConfig(entries []string) string {
	openlibConf := OpenLibConfig{EntryPoints{entries}}
	j, err := json.MarshalIndent(openlibConf, "", "    ")
	if err != nil {
		panic("Error generating JSON config for Windows entry points")
	}
	return string(j)
}

func findEntries(path string) []string {
	if !util.IsCppFile(path) {
		return []string{}
	}

	file, err := os.Open(path)
	if err != nil {
		logger.Errorf("Cannot find the source code file. error=%v", err)
		panic(err)
	}
	defer file.Close()

	var result []string
	found := false
	scanner := bufio.NewScanner(file)
	// Example:
	// 	"     ON_BN_CLICKED(IDC_TEST_BUTTON, &ThisClass::OnTestButton)"
	//
	// split the string in to two capture groups:
	// 1. (.*\(.*,\s*) -- match anything before the last comma (including the comma)
	// 2. (\w+\)) -- match things after the last comma and the closing parenthesis
	pattern := regexp.MustCompile(`(.*\(.*,\s*)([&:\w]+\))`)
	for scanner.Scan() {
		line := scanner.Text()
		code := strings.TrimSpace(line)
		if strings.HasPrefix(code, "BEGIN_MESSAGE_MAP") {
			found = true
			continue
		}

		if strings.HasPrefix(code, "END_MESSAGE_MAP") {
			found = false
			continue
		}

		if found {
			// insert "&ThisClass::" before $2 (the second capture group, i.e., the last argument)
			submatches := pattern.FindStringSubmatch(line)
			// the length of `submatches` should be 3
			// the whole matching string, the first capture group, and the second
			if len(submatches) >= 3 {
				entry := submatches[2]
				entry = strings.TrimSpace(entry)
				entry = strings.TrimPrefix(entry, "&")
				entry = strings.TrimPrefix(entry, "ThisClass::")
				entry = strings.TrimSuffix(entry, ")")
				result = append(result, entry)
			}
		}
	}
	return result
}
