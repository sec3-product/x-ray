package datatypes

// TODO: OpenRace's report currently does not contain SharedObj yet
type SharedObj struct{}

type MemAccess struct {
	Col      int    `json:col`
	Line     int    `json:line`
	Dir      string `json:dir`
	Filename string `json:filename`
	Type     string `json:type`
}

type DataRace struct {
	Access1 MemAccess `json:"access1"`
	Access2 MemAccess `json:"access2"`
}

type Report struct {
	Races []DataRace
}
