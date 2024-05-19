package reporter

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/gobwas/glob"
)

func TestFilter_Parse(t *testing.T) {
	str := `
	{ "report": { 
		"filters": [ 
			{ "node": "function", "name": "taosInitLog" }, 
			{ "node": "line", "sourceFile": "tlog.c" }, 
			{ "node": "variable", "name": "tLogBuff" } ] 
		} 
	}`
	b := []byte(str)
	var conf Config
	err := json.Unmarshal(b, &conf)
	filters := parse(conf)
	fmt.Println(filters.Lines[0].file)
	fmt.Println("code:")
	fmt.Println(filters.Lines[0].code == "")
	// filters.Lines[0].code = "*"
	g, err := glob.Compile(filters.Lines[0].code)
	if err != nil {
		fmt.Println(err)
	}
	g.Match("test")
	t.Fail()
	if err != nil {
		t.Fail()
	}
}

func TestFilter_parseCode(t *testing.T) {
	raw := " 55|static item *heads[LARGEST_ID];\n"
	res := parseCode(raw)
	fmt.Println(res)
	t.Fail()
	if res != "static item *heads[LARGEST_ID];\n" {
		t.Fail()
	}
}

func TestFilter_parseSharedObjType(t *testing.T) {
	raw := " 55|static item *heads[LARGEST_ID];\n"
	res := parseObjType(raw)

	if res != "item*" {
		fmt.Println(res)
		t.Fail()
	}
}

func TestFilter_glob(t *testing.T) {
	g := glob.MustCompile("*/memcached/*")
	r := g.Match("/home/yanze/code/memcached/memcached.c")
	fmt.Println(r)
	t.Fail()
}
