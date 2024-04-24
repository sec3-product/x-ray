package reporter

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestMap(t *testing.T) {

	m := make(map[string]int)
	m["a"] = 1
	m["b"] = 2

	if val, ok := m["key"]; ok {
		fmt.Printf("map %v", val)
	} else {
		fmt.Println("key not exist")
	}
}

func TestPrintf(t *testing.T) {
	fmt.Printf("%t\n", true)

	testArray := []int{1, 2, 3, 4, 5}
	fmt.Println(testArray)
}

func TestJson(t *testing.T) {
	testArray := []int{1, 2, 3, 4, 5}
	versionBytes, _ := json.Marshal(testArray)
	fmt.Println(versionBytes)
}
