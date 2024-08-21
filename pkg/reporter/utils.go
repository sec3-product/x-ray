package reporter

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// WriteIndexJSON write the `iinfo` back into index.json. It will overwrite
// the original index.json.
func WriteIndexJSON(indexInfo IndexInfo, buildDir string) error {
	indexJSON, err := json.MarshalIndent(indexInfo, "", " ")
	if err != nil {
		return fmt.Errorf("unable to marshal indexInfo: %w", err)
	}
	indexJSONPath := filepath.Join(buildDir, "index.json")
	if err := os.WriteFile(indexJSONPath, indexJSON, 0644); err != nil {
		return fmt.Errorf("unable to write index.json: %w", err)
	}
	return nil
}

// ReadIndexJSON returns parsed JSON.
func ReadIndexJSON(indexInfo *IndexInfo, rawJSONDir string) error {
	indexJSONPath := filepath.Join(".xray", "build", "index.json")
	if rawJSONDir != "" {
		indexJSONPath = filepath.Join(rawJSONDir, "index.json")
	}

	if _, err := os.Stat(indexJSONPath); os.IsNotExist(err) {
		return fmt.Errorf("index.json not found: %s", indexJSONPath)
	}

	indexJSONBytes, err := os.ReadFile(indexJSONPath)
	if err != nil {
		return fmt.Errorf("unable to read index.json at %s: %w", indexJSONPath, err)
	}
	if err := json.Unmarshal(indexJSONBytes, indexInfo); err != nil {
		return fmt.Errorf("unable to unmarshal index.json: %w", err)
	}
	return nil
}
