package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func Test_showTerminalReport(t *testing.T) {
	cwd, _ := os.Getwd()
	rawRaceJson := filepath.Join(cwd, "../../test/memcached.json")
	showTerminalReport(rawRaceJson)

}

type ref struct {
	a string
}

func passByValue(r ref) {
	r.a = "xyz"
}

func passByPointer(r *ref) {
	r.a = "xyz"
}

func TestByReference(t *testing.T) {
	r := ref{"abc"}
	passByValue(r)
	fmt.Println(r.a)

	passByPointer(&r)
	fmt.Println(r.a)
}

func TestSort(t *testing.T) {
	r1 := ReportInfo{"project 1", 10}
	r2 := ReportInfo{"project 2", 0}
	r3 := ReportInfo{"project 3", 30}
	r4 := ReportInfo{"project 4", 11}
	r5 := ReportInfo{"project 5", 50}
	idx := indexMenuBuildParam{[]ReportInfo{r1, r2, r3, r4, r5}, []string{}, "1"}
	fmt.Println(idx.ReportList)
	sort.Sort(&idx)
	fmt.Println(idx.ReportList)
}
