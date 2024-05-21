package regress

import (
	"clio/internal/diff/race"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
)

// ReadTruthRaces reads races from a truth file
func ReadTruthRaces(truthFile, projectName string) ([]*race.Race, error) {
	data, err := ioutil.ReadFile(truthFile)
	if err != nil {
		return nil, fmt.Errorf("could not open truth file: %v", err)
	}
	lines := strings.Split(string(data), "\n")

	skipped := -1
	if projectName != "" {
		for i, line := range lines {
			if line == "["+projectName+"]" {
				lines = lines[i+1:]
				skipped = i + 1
				break
			}
		}
		if skipped == -1 {
			return nil, fmt.Errorf("could not find project in truth file")
		}
	} else {
		skipped = 0
	}

	races := []*race.Race{}

	for i, line := range lines {
		if strings.HasPrefix(line, "[") {
			break
		}

		if strings.TrimSpace(line) == "" {
			continue
		}

		splits := strings.Split(line, " ")
		if len(splits) != 4 {
			return nil, fmt.Errorf("line %d %d is not recognized as a race", skipped+i, len(splits))
		}

		line1, err := strconv.ParseUint(splits[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("line %d: '%s' is not a line number", skipped+i, splits[1])
		}

		line2, err := strconv.ParseUint(splits[3], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("line %d: '%s' is not a line number", skipped+i, splits[3])
		}

		race := &race.Race{
			Access1: race.Access{
				Filename: splits[0],
				Line:     uint(line1),
			},
			Access2: race.Access{
				Filename: splits[2],
				Line:     uint(line2),
			},
		}

		races = append(races, race)
	}

	return races, nil
}

// ReadJSONRaces reads races from a race.json file
func ReadJSONRaces(raceJSON string) ([]race.Race, error) {
	data, err := ioutil.ReadFile(raceJSON)
	if err != nil {
		return nil, fmt.Errorf("could not read json file '%s': %v", raceJSON, err)
	}

	var report race.Report
	if err = json.Unmarshal(data, &report); err != nil {
		return nil, fmt.Errorf("unable to parse json report '%s': %v", raceJSON, err)
	}

	return report.DataRaces, nil
}

// only checks that file and line match
func accessMatch(lhs, rhs race.Access) bool {
	return lhs.Filename == rhs.Filename && lhs.Line == rhs.Line
}

func match(lhs, rhs *race.Race) bool {
	return (accessMatch(lhs.Access1, rhs.Access1) && accessMatch(lhs.Access2, rhs.Access2)) || (accessMatch(lhs.Access1, rhs.Access2) && accessMatch(lhs.Access2, rhs.Access1))
}

// CheckMissing checks that all expected races are in actual and return any that are not
func CheckMissing(actual []race.Race, expected []*race.Race) []*race.Race {
	missing := []*race.Race{}

	for _, e := range expected {
		found := false
		for _, a := range actual {
			if match(e, &a) {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, e)
		}
	}

	return missing
}
