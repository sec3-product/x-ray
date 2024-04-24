package reporter

import (
	"encoding/json"
	"io/ioutil"
)

type RaceJSON struct {
	DataRaces      []DataRace      `json:"dataRaces"`
	RaceConditions []RaceCondition `json:"raceConditions"`
	TocTous        []DataRace      `json:"toctou"`
	MismatchedAPIs []MismatchedAPI `json:"mismatchedAPIs"`
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
}

// NewRaceJSONParser initialize variable
// return instance of parser
func NewRaceJSONParser() *RaceJSONParser {
	p := new(RaceJSONParser)
	p.DataRaceSignatureMap = make(map[string]int)
	p.TocTousSignatureMap = make(map[string]int)
	p.MismatchedAPISignatureMap = make(map[string]int)
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
}

func (p *RaceJSONParser) computeDataRaceSignature() {
	if p.raceJSON == nil {
		return
	}

	for i, v := range p.raceJSON.DataRaces {
		sig := v.ComputeSignature()
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
