package reporter

import (
	"encoding/json"
	//"fmt"
	"io/ioutil"
)

type RaceJSON struct {
	DataRaces          []DataRace          `json:"dataRaces"`
	RaceConditions     []RaceCondition     `json:"raceConditions"`
	TocTous            []DataRace          `json:"toctou"`
	MismatchedAPIs     []MismatchedAPI     `json:"mismatchedAPIs"`
	UntrustfulAccounts []UntrustfulAccount `json:"untrustfulAccounts"`
	UnsafeOperations   []UnsafeOperation   `json:"unsafeOperations"`
	CosplayAccounts    []CosplayAccount    `json:"cosplayAccounts"`
}

type RaceJSONParser struct {
	raceJSON *RaceJSON

	DataRaceSignatureMap  map[string]int
	DataRaceSignatureList []string

	RaceConditionSignatureMap  map[string]int
	RaceConditionSignatureList []string

	TocTousSignatureMap  map[string]int
	TocTousSignatureList []string

	MismatchedAPISignatureMap  map[string]int
	MismatchedAPISignatureList []string

	UntrustfulAccountSignatureMap  map[string]int
	UntrustfulAccountSignatureList []string

	UnsafeOperationSignatureMap  map[string]int
	UnsafeOperationSignatureList []string

	CosplayAccountSignatureMap  map[string]int
	CosplayAccountSignatureList []string
}

// NewRaceJSONParser initialize variable
// return instance of parser
func NewRaceJSONParser() *RaceJSONParser {
	p := new(RaceJSONParser)
	p.DataRaceSignatureMap = make(map[string]int)
	p.TocTousSignatureMap = make(map[string]int)
	p.MismatchedAPISignatureMap = make(map[string]int)
	p.UntrustfulAccountSignatureMap = make(map[string]int)
	p.UnsafeOperationSignatureMap = make(map[string]int)
	p.CosplayAccountSignatureMap = make(map[string]int)
	p.RaceConditionSignatureMap = make(map[string]int)

	return p
}

// LoadRaceJSON open race.json parse it as json file
func (p *RaceJSONParser) LoadRaceJSON(path string) error {
	bytesRead, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	p.raceJSON = new(RaceJSON)
	if err = json.Unmarshal(bytesRead, p.raceJSON); err != nil {
		return err
	}

	return nil
}

// ComputeSignature as a wrapper to compute all kinds of codesnip's signature
// For identify race item
func (p *RaceJSONParser) ComputeSignature() {
	p.computeDataRaceSignature()
	p.computeRaceConditionSignature()
	p.computeTocTousSignature()
	p.computeMismatchedAPISignature()
	p.computeUntrustfulAccountSignature()
	p.computeUnsafeOperationSignature()
	p.computeCosplayAccountSignature()

}

func (p *RaceJSONParser) computeDataRaceSignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.DataRaces {
		sig := v.ComputeSignature()
		//fmt.Println("newDataRace sig: ", sig)
		p.DataRaceSignatureList = append(p.DataRaceSignatureList, sig)
		p.DataRaceSignatureMap[sig] = i
	}
}

func (p *RaceJSONParser) computeRaceConditionSignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.RaceConditions {
		sig := v.ComputeSignature()
		p.DataRaceSignatureList = append(p.DataRaceSignatureList, sig)
		p.DataRaceSignatureMap[sig] = i
	}
}

func (p *RaceJSONParser) computeTocTousSignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.TocTous {
		sig := v.ComputeSignature()
		p.TocTousSignatureList = append(p.TocTousSignatureList, sig)
		p.TocTousSignatureMap[sig] = i
	}
}

func (p *RaceJSONParser) computeMismatchedAPISignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.MismatchedAPIs {
		sig := v.ComputeSignature()
		p.MismatchedAPISignatureList = append(p.MismatchedAPISignatureList, sig)
		p.MismatchedAPISignatureMap[sig] = i
	}
}
func (p *RaceJSONParser) computeUntrustfulAccountSignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.UntrustfulAccounts {
		sig := v.ComputeSignature()
		p.UntrustfulAccountSignatureList = append(p.UntrustfulAccountSignatureList, sig)
		p.UntrustfulAccountSignatureMap[sig] = i
	}
}
func (p *RaceJSONParser) computeUnsafeOperationSignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.UnsafeOperations {
		sig := v.ComputeSignature()
		p.UnsafeOperationSignatureList = append(p.UnsafeOperationSignatureList, sig)
		p.UnsafeOperationSignatureMap[sig] = i
	}
}
func (p *RaceJSONParser) computeCosplayAccountSignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.CosplayAccounts {
		sig := v.ComputeSignature()
		p.CosplayAccountSignatureList = append(p.CosplayAccountSignatureList, sig)
		p.CosplayAccountSignatureMap[sig] = i
	}
}
