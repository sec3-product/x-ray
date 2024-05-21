package race

import (
	"crypto/md5"
	"fmt"
)

// Access represents a source-code level access
type Access struct {
	Line, Col     uint
	Dir, Filename string
}

func (a *Access) String() string {
	return fmt.Sprintf("%s:%d", a.Filename, a.Line)
}

// Hash is used as a unique id for a race
type Hash string

func (h Hash) String() string {
	return string(h[:8])
}

// Race represents a data race
type Race struct {
	Access1, Access2 Access
}

// GetHash returns the Hash for this race
func (r Race) GetHash() Hash {
	data := fmt.Sprintf("%v%v-%v%v", r.Access1.Line, r.Access1.Filename, r.Access2.Line, r.Access2.Filename)
	return Hash(fmt.Sprintf("%x", md5.Sum([]byte(data))))
}

// Report mirrors races.json
type Report struct {
	DataRaces []Race
}
