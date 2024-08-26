package logger

/**
 * This is a singleton. The developer should follow the sequence below to initialize
 * and use logger
 *
 * conflib.init()
 * logger.init("componennt")
 * logger.Info("blah blah")
 *
 */
import (
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"

	log "github.com/sirupsen/logrus"

	"github.com/sec3-product/x-ray/pkg/conflib"
)

func stringToLogLevel(levelStr string) log.Level {
	if levelStr == "debug" {
		return log.DebugLevel
	} else if levelStr == "info" {
		return log.InfoLevel
	} else if levelStr == "warn" {
		return log.WarnLevel
	} else if levelStr == "error" {
		return log.ErrorLevel
	} else if levelStr == "trace" {
		return log.TraceLevel
	} else {
		return log.DebugLevel
	}
}

var _logger *log.Entry

/**
 * Keeps the value of log folder
 */
var _logFolder string

type CoderrectFormatter struct{}

func (f *CoderrectFormatter) Format(entry *log.Entry) ([]byte, error) {
	t := entry.Time
	time := fmt.Sprintf("%02d:%02d:%02d", t.Hour(), t.Minute(), t.Second())

	var fields string
	for k, v := range entry.Data {
		if k == "component" || k == "source" {
			// do nothing
		} else {
			fields += fmt.Sprintf("%s=%s, ", k, v)
		}
	}

	var source string
	if pc, file, line, ok := runtime.Caller(7); ok {
		funcName := runtime.FuncForPC(pc).Name()
		source = fmt.Sprintf("%s:%v:%s", path.Base(file), line, path.Base(funcName))
	}

	formatted := fmt.Sprintf("%s [%s] [%s] [%s] %s", time, entry.Level,
		entry.Data["component"], source, entry.Message)
	if fields != "" {
		formatted += " " + fields
	}

	return []byte(formatted + "\n"), nil
}

type CoderrectLogWritter struct {
	fileWriter io.Writer
	toStderr   bool
}

func (iw *CoderrectLogWritter) Write(data []byte) (n int, err error) {
	if iw.toStderr {
		if _, err := os.Stderr.Write(data); err != nil {
			return 0, err
		}
	}

	return iw.fileWriter.Write(data)
}

func dirExists(dir string) bool {
	_, err := os.Stat(dir)
	return err == nil
}

/**
 * a logPath has the format of ".../log.N". The ord is "N"
 */
func getOrd(logPath string) int {
	m := regexp.MustCompile(`.(\d+)$`)
	s := m.FindString(logPath)
	i, _ := strconv.Atoi(s[1:])
	return i
}

/**
 * returns full paths of files like "log.N". The list is sorted
 * by file name.
 */
func getLogList(logFolder string) ([]string, error) {
	var fileList []string

	err := filepath.Walk(logFolder, func(path string, info os.FileInfo, err error) error {
		matched, err := regexp.MatchString(`log.\d+$`, path)
		if matched {
			fileList = append(fileList, path)
		}
		return nil
	})
	if err != nil {
		return fileList, err
	}

	sort.Slice(fileList, func(i, j int) bool {
		ord1 := getOrd(fileList[i])
		ord2 := getOrd(fileList[j])
		return ord1 < ord2
	})

	return fileList, nil
}

func renameFile(oldPath string, newPath string) error {
	return os.Rename(oldPath, newPath)
}

func rotateLogs(logFolder string, maxLogFileCount int, maxLogFileSize int64) error {
	logCurr := filepath.Join(logFolder, "log.current")

	fileInfo, err := os.Stat(logCurr)
	if err == nil && fileInfo.Size() > maxLogFileSize {
		// let's rotate
		fileList, err := getLogList(logFolder)
		if err != nil {
			return err
		}
		if len(fileList) < int(maxLogFileCount) {
			if err = renameFile(logCurr, filepath.Join(logFolder, fmt.Sprintf("log.%d", len(fileList)))); err != nil {
				return err
			}
		} else {
			for i, file := range fileList {
				if i == 0 {
					if err = os.Remove(file); err != nil {
						return err
					}
				} else {
					if err = renameFile(file, filepath.Join(logFolder, fmt.Sprintf("log.%d", i-1))); err != nil {
						return err
					}
				}
			}
			return renameFile(logCurr, filepath.Join(logFolder, fmt.Sprintf("log.%d", maxLogFileCount-1)))
		}
	}

	return nil
}

func internalInit(component string, logRotation bool) (*log.Entry, error) {
	_logger = log.WithField("component", component)
	log.SetFormatter(new(CoderrectFormatter))
	log.SetLevel(stringToLogLevel(conflib.GetString("logger.level", "debug")))

	// get the log folder as an absolute path
	var mode os.FileMode
	logFolder := conflib.GetString("logger.logFolder", "logs")
	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	fileInfo, err := os.Stat(cwd)
	if err != nil {
		return nil, err
	}
	mode = fileInfo.Mode()

	if runtime.GOOS == "windows" {
		if logFolder[1] != ':' {
			// for local disk driver letter, c:\
			logFolder = filepath.Join(cwd, ".xray", logFolder)
		}
	} else {
		if !strings.HasPrefix(logFolder, "/") {
			logFolder = filepath.Join(cwd, ".xray", logFolder)
		}
	}
	_logFolder = logFolder

	// if the path doesn't exist, create it
	if _, err := os.Stat(logFolder); os.IsNotExist(err) {
		if err = os.MkdirAll(logFolder, mode); err != nil {
			return nil, err
		}
	}

	if logRotation {
		// rotate log files
		maxCount := int(conflib.GetInt("logger.maximumLogFileCount", 8))
		maxSize := conflib.GetInt("logger.maximumLogFileSize", 128) * 1024 * 1024
		if err := rotateLogs(logFolder, maxCount, maxSize); err != nil {
			return nil, err
		}
	}

	// open the current log file for logs
	activeLogFilePath := filepath.Join(logFolder, "log.current")
	fileWriter, err := os.OpenFile(activeLogFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, mode^0111)
	if err != nil {
		return nil, err
	}

	log.SetOutput(&CoderrectLogWritter{
		toStderr:   conflib.GetBool("logger.toStderr", false),
		fileWriter: fileWriter,
	})

	return _logger, nil
}

func Init(component string) (*log.Entry, error) {
	return internalInit(component, false)
}

func InitWithLogRotation(component string) (*log.Entry, error) {
	return internalInit(component, true)
}

func GetLogger() *log.Entry {
	return _logger
}

func GetLogFolder() string {
	return _logFolder
}

func Tracef(format string, args ...interface{}) {
	_logger.Tracef(format, args...)
}

func Debugf(format string, args ...interface{}) {
	_logger.Debugf(format, args...)
}

func Infof(format string, args ...interface{}) {
	_logger.Infof(format, args...)
}

func Warnf(format string, args ...interface{}) {
	_logger.Warnf(format, args...)
}

func Errorf(format string, args ...interface{}) {
	_logger.Errorf(format, args...)
}

func Fatalf(format string, args ...interface{}) {
	_logger.Fatalf(format, args...)
}
