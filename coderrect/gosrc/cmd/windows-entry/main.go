package main

import (
	"fmt"
	"os"

	"github.com/coderrect-inc/coderrect/util/logger"
	"github.com/coderrect-inc/coderrect/windows/preprocessor"
)

func main() {
	_, err := logger.Init("windows-entry")
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to init logger. err=%v", err))
		os.Exit(1)
	}

	if len(os.Args) != 2 {
		os.Stderr.WriteString("You must pass exactly one argument!\n")
		fmt.Println("Usage:")
		fmt.Println("./preprocessor [Directory]")
		os.Exit(1)
	}

	path := os.Args[1]
	entries, err := preprocessor.FindAllEntries(path)
	if err != nil {
		panic(err)
	}
	jsonStr := preprocessor.EntriesToConfig(entries)
	fmt.Println(jsonStr)
}
