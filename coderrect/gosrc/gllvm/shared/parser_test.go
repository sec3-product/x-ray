package shared

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRegex(t *testing.T) {
	elem := "/out:hello.exe"
	pattern := `(?i)^\/OUT:.+$`
	var regExp = regexp.MustCompile(pattern)
	assert.True(t, regExp.MatchString(elem))

	elem = "-out:hello.exe"
	pattern = `(?i)^\-OUT:.+$`
	regExp = regexp.MustCompile(pattern)
	assert.True(t, regExp.MatchString(elem))
}
