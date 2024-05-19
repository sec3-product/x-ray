package diff

import (
	"clio/internal/diff/race"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
)

// RaceMap maps race Hashes to race objects
type RaceMap map[race.Hash]race.Race

// Diff represents the changes observed for this commit
type Diff struct {
	Commit string
	Crash  bool

	// never before reported
	New []race.Hash

	// previosuly ssen, but not on last commit
	Renew []race.Hash

	// gone from last commit
	Gone []race.Hash

	// also on prev commit
	Same []race.Hash
}

func (d *Diff) String() string {
	var str strings.Builder

	str.WriteString(d.Commit)
	str.WriteString(" (")
	str.WriteString(strconv.Itoa(len(d.Races())))
	str.WriteString(" races)")

	return str.String()
}

// DetailString gives a string with detailed information about the diff
func (d *Diff) DetailString(hashPerLine int) string {
	var str strings.Builder

	str.WriteString(d.String())

	printHashes := func(hashes []race.Hash, name string) {
		if len(hashes) > 0 {
			str.WriteString("\n\t ")
			str.WriteString(strconv.Itoa(len(hashes)))
			str.WriteString(" " + name)

			for i, hash := range hashes {
				if i%hashPerLine == 0 {
					str.WriteString("\n\t\t")
				}
				str.WriteString(hash.String() + " ")
			}
		}
	}

	printHashes(d.New, "New")
	printHashes(d.Gone, "Gone")
	printHashes(d.Renew, "Renew")

	if len(d.New) == 0 && len(d.Gone) == 0 && len(d.Renew) == 0 {
		str.WriteString("\n\t(no change)")
	}

	str.WriteString("\n")

	return str.String()
}

func (d Diff) Races() []race.Hash {
	races := make([]race.Hash, 0, len(d.New)+len(d.Renew)+len(d.Same))
	races = append(races, d.New...)
	races = append(races, d.Renew...)
	races = append(races, d.Same...)
	return races
}

func (d *Diff) contains(h race.Hash) bool {
	for _, hash := range d.Races() {
		if h == hash {
			return true
		}
	}
	return false
}

// Report includes all diffs and races
type Report struct {
	Races RaceMap
	Diffs []*Diff
}

func (r *Report) String() string {
	// Config how many race hashes are printed per line
	hashPerLine := 5

	var str strings.Builder

	str.WriteString("====All Races====\n")
	for hash, r := range r.Races {
		str.WriteString(hash.String())
		str.WriteString(" ")
		str.WriteString(r.Access1.String())
		str.WriteString("|")
		str.WriteString(r.Access2.String())
		str.WriteString("\n")
	}

	str.WriteString("====Diffs====\n")

	intialDiff := r.Diffs[0]
	str.WriteString("#0 ")
	str.WriteString(intialDiff.String())
	str.WriteString("\n\t(initial)\n")

	for i, diff := range r.Diffs[1:] {
		str.WriteString("#")
		str.WriteString(strconv.Itoa(i + 1))
		str.WriteString(" ")
		str.WriteString(diff.DetailString(hashPerLine))
	}

	return str.String()
}

// getRaceData takes a a bc report directory (e.g. reports/drb/DRB001) and returns races associated with each commit
// nil slice indicates the commit crashed
func getRaceData(dir string, commits []string, raceMap *map[race.Hash]race.Race) (map[string][]race.Hash, error) {
	raceData := map[string][]race.Hash{}

	for _, commit := range commits {
		data, err := ioutil.ReadFile(filepath.Join(dir, commit, "races.json"))
		if err != nil {
			if _, err2 := ioutil.ReadFile(filepath.Join(dir, commit, "error.txt")); err2 != nil {
				return nil, fmt.Errorf("could not read json for commit '%s': %v", commit[:8], err)
			}
			raceData[commit] = nil
			continue
		}

		var report race.Report
		if err = json.Unmarshal(data, &report); err != nil {
			return nil, fmt.Errorf("unable to parse json report for commit '%s': %v", commit[:8], err)
		}

		races := make([]race.Hash, len(report.DataRaces))

		for i, r := range report.DataRaces {
			hash := r.GetHash()
			races[i] = hash
			(*raceMap)[hash] = r
		}

		raceData[commit] = races
	}

	return raceData, nil
}

// getBcDirs takes a root level report project directory and returns all teh bc directories contained within
func getBcDirs(projectDir string) ([]string, error) {
	files, err := ioutil.ReadDir(projectDir)
	if err != nil {
		return nil, fmt.Errorf("Could not read project reports: %v", err)
	}

	bcDirs := make([]string, 0, len(files))
	for _, file := range files {
		if file.IsDir() {
			path := filepath.Join(projectDir, file.Name())
			bcDirs = append(bcDirs, path)
		}
	}

	return bcDirs, nil
}

func buildDiffs(raceData map[string][]race.Hash, order []string) []*Diff {
	diffs := make([]*Diff, len(order))

	var prevDiff *Diff
	prevDiff = nil

	allRaces := map[race.Hash]bool{}

	for i, commit := range order {
		diff := &Diff{
			Commit: commit,
			Crash:  false,
			New:    []race.Hash{},
			Renew:  []race.Hash{},
			Gone:   []race.Hash{},
			Same:   []race.Hash{},
		}

		races, ok := raceData[commit]
		if !ok || races == nil {
			diff.Crash = true
			diffs[i] = diff
			continue
		}

		if prevDiff == nil {
			diff.New = races
		} else {
			for _, r := range races {
				if prevDiff.contains(r) {
					diff.Same = append(diff.Same, r)
				} else if _, ok := allRaces[r]; ok {
					diff.Renew = append(diff.Renew, r)
				} else {
					diff.New = append(diff.New, r)
				}
			}
			for _, r := range prevDiff.Races() {
				if !diff.contains(r) {
					diff.Gone = append(diff.Gone, r)
				}
			}
			for _, r := range diff.New {
				allRaces[r] = true
			}
		}

		diffs[i] = diff
		prevDiff = diff
	}

	return diffs
}

// Get produces the difference over a sorted list of commits for the specified project
func Get(reportPath, name string, commitHashes []string) (*Report, error) {
	projectReportDir := filepath.Join(reportPath, name)
	bcDirs, err := getBcDirs(projectReportDir)
	if err != nil {
		return nil, err
	}

	// raceData is a map of commit hashes to races
	raceData := map[string][]race.Hash{}
	raceMap := map[race.Hash]race.Race{}
	for _, bcDir := range bcDirs {
		// Get the races associated with this bc
		bcRaceData, err := getRaceData(bcDir, commitHashes, &raceMap)
		if err != nil {
			return nil, fmt.Errorf("err reading race data: %v", err)
		}

		// Add the races to the race map
		for commit, races := range bcRaceData {
			existing, ok := raceData[commit]
			if !ok {
				raceData[commit] = races
			} else {
				raceData[commit] = append(existing, races...)
			}
		}
	}

	diffs := buildDiffs(raceData, commitHashes)

	return &Report{
		Diffs: diffs,
		Races: raceMap,
	}, nil
}

// GetReport takes an order slice of paths to races.json files and returns a diff report
func GetReport(reportPaths []string) (*Report, error) {
	// raceMap tracks all races seen
	raceMap := map[race.Hash]race.Race{}

	raceData := map[string][]race.Hash{}
	for _, reportPath := range reportPaths {
		data, err := ioutil.ReadFile(reportPath)
		if err != nil {
			return nil, fmt.Errorf("could not read json file '%s': %v", reportPath, err)
		}

		var report race.Report
		if err = json.Unmarshal(data, &report); err != nil {
			return nil, fmt.Errorf("unable to parse json report '%s': %v", reportPath, err)
		}

		races := make([]race.Hash, len(report.DataRaces))

		for i, r := range report.DataRaces {
			hash := r.GetHash()
			races[i] = hash
			raceMap[hash] = r
		}

		raceData[reportPath] = races
	}

	diffs := buildDiffs(raceData, reportPaths)
	return &Report{
		Diffs: diffs,
		Races: raceMap,
	}, nil
}
