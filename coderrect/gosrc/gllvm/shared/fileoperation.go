// Called by interceptor
// monitor all the file operation
// when copy/move file, rebuild mapping

package shared

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/coderrect-inc/coderrect/util"
	"github.com/coderrect-inc/coderrect/util/logger"
)

type fileOperationParseResult struct {
	IsRecursive bool
}

// simple method to parse mv/cp command line.
func parseFileOperation(argList []string, fileOperationMap map[string]string) {

	var inputFileList []string
	for len(argList) > 0 {
		var listShift = 0
		var elem = argList[0]
		// ignore all the options
		if strings.HasPrefix(elem, "-") || strings.HasPrefix(elem, "--") {
			argList = argList[1:]
			continue
		}

		absPath, _ := filepath.Abs(elem)
		inputFileList = append(inputFileList, absPath)
		argList = argList[1+listShift:]
	}

	dstPath := inputFileList[len(inputFileList)-1]
	srcFileList := inputFileList[:len(inputFileList)-1]
	for _, srcPath := range srcFileList {
		if util.FileExists(srcPath) {
			if dirExists(dstPath) {
				fileName := filepath.Base(srcPath)
				dstPath = filepath.Join(dstPath, fileName)
			}
			// single file operation
			fileOperationMap[srcPath] = dstPath
		} else {
			relpath := filepath.Base(srcPath)
			err := filepath.Walk(srcPath,
				func(path string, info os.FileInfo, err error) error {
					if err != nil {
						return err
					}

					if !info.IsDir() {
						from := filepath.Join(srcPath, info.Name())
						// relative path must be insert into dst path
						to := filepath.Join(dstPath, relpath, info.Name())
						fileOperationMap[from] = to
					}
					return nil
				})
			if err != nil {
				logger.Errorf("file operation walk directory error, err=%v", err)
			}
		}
	}
}

// OnCopyFile parser args and modify bc file mapping
func OnCopyFile(args []string) (exitCode int) {
	exitCode = 0

	fileOperationMap := make(map[string]string)
	parseFileOperation(args, fileOperationMap)
	logger.Debugf("Insert bolt db copy file. mapping=%v", fileOperationMap)
	for srcPath, dstPath := range fileOperationMap {
		bcFile := queryBoltDb(BucketNameBcFile, srcPath)
		if len(bcFile) > 0 {
			insertIntoBoltDb(BucketNameBcFile, dstPath, bcFile)
		}
	}
	return
}

// OnMoveFile parser args and modify bc file mapping
func OnMoveFile(args []string) (exitCode int) {
	exitCode = 0

	fileOperationMap := make(map[string]string)
	parseFileOperation(args, fileOperationMap)
	logger.Debugf("Insert bolt db move file. mapping=%v", fileOperationMap)
	for srcPath, dstPath := range fileOperationMap {
		bcFile := queryBoltDb(BucketNameBcFile, srcPath)
		if len(bcFile) > 0 {
			// add new path mapping, update name mapping, remove old path mapping
			insertIntoBoltDb(BucketNameBcFile, dstPath, bcFile)
			binFileName := getBinaryFilename(srcPath)
			insertIntoBoltDb(BucketNameBcFile, binFileName, bcFile)
			insertIntoBoltDb(BucketNameBcFile, srcPath, "")
		}
	}
	return
}
