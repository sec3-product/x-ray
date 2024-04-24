package reporter

import (
	"encoding/json"
	"io/ioutil"
	"path/filepath"

	"github.com/coderrect-inc/coderrect/util"
	"github.com/coderrect-inc/coderrect/util/logger"
)

// WriteIndexJSON write the `iinfo` back into index.json
// ps: it will overwrite the original index.json
func WriteIndexJSON(indexInfo IndexInfo, buildDir string) {
	indexJSON, _ := json.MarshalIndent(indexInfo, "", " ")
	indexJSONPath := filepath.Join(buildDir, "index.json")
	if err := ioutil.WriteFile(indexJSONPath, indexJSON, 0644); err != nil {
		logger.Errorf("Fail to create index.json. err=%v", err)
		panic(err)
	}
}

// ReadIndexJSON returns 0 if index.json is not found
// Otherwise, returns 1
// The parsed JSON will be stored into `indexInfo`
func ReadIndexJSON(indexInfo *IndexInfo, rawJSONDir string) int {
	var indexJSONPath string
	if rawJSONDir == "" {
		indexJSONPath = filepath.Join(".coderrect", "build", "index.json")
	} else {
		indexJSONPath = filepath.Join(rawJSONDir, "index.json")
	}

	if !util.FileExists(indexJSONPath) {
		return 0
	}

	indexJSONBytes, err := ioutil.ReadFile(indexJSONPath)
	if err != nil {
		logger.Errorf("Fail to read index.json. indexjson=%s, err=%v", indexJSONPath, err)
		panic(err)
	}

	// var indexInfo IndexInfo
	if err = json.Unmarshal(indexJSONBytes, indexInfo); err != nil {
		logger.Errorf("Fail to parse index.json. err=%v", err)
		panic(err)
	}
	return 1
}
