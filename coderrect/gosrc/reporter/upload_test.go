package reporter

import (
	"fmt"
	"testing"
)

func TestUpload_zipReport(t *testing.T) {
	zipReport("test", "/home/yanze/code/memcached/.coderrect/report")
}

func TestUpload_collectRepoURL(t *testing.T) {
	out := collectRepoURL()
	fmt.Println(out)
	t.Fail()
}
