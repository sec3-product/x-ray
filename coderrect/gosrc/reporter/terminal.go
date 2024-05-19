package reporter

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"

	"github.com/coderrect-inc/coderrect/reporter/datatypes"
)

func readRacesJSON(path ...string) []datatypes.DataRace {
	p := filepath.Join(path...)
	data, err := ioutil.ReadFile(p)
	if err != nil {
		panic("Fail to read races.json")
	}

	var report []datatypes.DataRace
	err = json.Unmarshal(data, &report)
	if err != nil {
		panic(err)
	}
	return report
}

func PrintTerminalReport(path ...string) {
	races := readRacesJSON(path...)
	for i, r := range races {
		if i != 0 {
			fmt.Println("")
		}
		fmt.Printf("=====Race No.%v=====\n", i+1)
		fmt.Printf("Race detected between %v, line %v and %v, line %v\n", r.Access1.Filename, r.Access1.Line, r.Access2.Filename, r.Access2.Line)
		fmt.Printf("Thread 1 has a %v access below:\n", r.Access1.Type)
		fmt.Println(RenderSnippet(r.Access1.Line, r.Access1.Dir, r.Access1.Filename))
		fmt.Println("")
		fmt.Printf("Which conflicts with the %v access in Thread 2 below:\n", r.Access2.Type)
		fmt.Println(RenderSnippet(r.Access2.Line, r.Access2.Dir, r.Access2.Filename))
	}
}
