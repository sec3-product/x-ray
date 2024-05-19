package test

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/coderrect-inc/coderrect/gllvm/shared"
	"github.com/coderrect-inc/coderrect/util/conflib"
	"github.com/coderrect-inc/coderrect/util/logger"
	bolt "go.etcd.io/bbolt"
)

func TestExecutableGenerator(t *testing.T) {
	argAlias := map[string]string{}

	_, err := conflib.InitializeWithEnvCmdline(argAlias)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	_, err = logger.InitWithLogRotation("coderrect-clang")
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("Failed to init logger. err=%v", err))
		os.Exit(1)
	}

	coderrectHome, _ := conflib.GetCoderrectHome()
	os.Setenv("CODERRECT_BUILD_DIR", coderrectHome)

	boldDbParentPath := filepath.Join(coderrectHome, ".coderrect/")
	if _, err := os.Stat(boldDbParentPath); os.IsNotExist(err) {
		os.Mkdir(boldDbParentPath, os.ModePerm)
	}

	boldDbFilePath := filepath.Join(coderrectHome, ".coderrect/", "bold.db")
	os.Setenv("CODERRECT_BOLD_DB_PATH", boldDbFilePath)

	os.Remove(boldDbFilePath)
	boltDb, err := bolt.Open(boldDbFilePath, 0666, nil)
	if err != nil {
		t.Error("Failed to create db.", err)
		return
	}
	err = boltDb.Update(func(tx *bolt.Tx) error {
		_, err = tx.CreateBucket([]byte("execfiles"))
		return err
	})
	if err != nil {
		return
	}
	boltDb.Close()

	args := []string{"../data/helloworld.c", "-o", "../data/hello"}
	shared.Compile(args, "clang")

}

func TestWalkDir(t *testing.T) {
	cwd, err := os.Getwd()
	err = filepath.Walk(filepath.Dir(cwd),
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			fmt.Println(path, info.Size())
			return nil
		})
	if err != nil {
		log.Println(err)
	}
}
