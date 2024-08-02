package conflib

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/buger/jsonparser"
)

const _ENV_INSTALLATION_DIRECTORY = "CODERRECT_HOME"

var defaultConf []byte
var homeConf []byte
var projectConf []byte
var customConf []byte
var cmdlineConf []byte

var consumedOpts []string

/**
 * search -conf in the command line and return its
 * value if found
 */
func findCustomConfPath() (string, error) {
	for i := 1; i < len(os.Args); i++ {
		tmp := os.Args[i]
		if len(tmp) < 6 || tmp[0:6] != "-conf=" {
			continue
		}

		stages := strings.Split(tmp, "=")
		return stages[1], nil
	}

	return "", nil
}

func isNumeric(a string) bool {
	if _, err := strconv.Atoi(a); err == nil {
		return true
	}

	if _, err := strconv.ParseFloat(a, 64); err == nil {
		return true
	}

	return false
}

func getArgsFromEnvironment() []string {
	var args []string
	jsonstr := os.Getenv("CODERRECT_CMDLINE_OPTS")
	if len(jsonstr) > 0 {
		type Tmp struct {
			Opts []string
		}
		tmp := Tmp{}
		json.Unmarshal([]byte(jsonstr), &tmp)
		args = tmp.Opts
	}

	return args
}

/**
 * "a": "allFiles" - this flag should be followed by a value
 * "a:" "allFiles" - this flag is a switch (true or false)
 */
func getLongArg(shortArgsMap map[string]string, shortArg string) (string, bool, bool) {
	longArgs, ok := shortArgsMap[shortArg]
	if !ok {
		shortArg = shortArg + ":"
		longArgs, ok = shortArgsMap[shortArg]
		if !ok {
			return "", false, false
		}
	}

	if strings.HasSuffix(shortArg, ":") {
		return longArgs, false, true
	}
	return longArgs, true, true
}

/**
 * depends on "useEnvCmdlineOpts", parseCmdlineArgs() initialize cmdlingConf
 * by parse os.Args or the env variable CODERRECT_CMDLINE_OPTS.
 *
 * CODERRECT_CMDLINE_OPTS is a json string in the format of
 *
 * {
 *   Opts: [arg1, arg2, arg3, ...]
 * }
 */
func parseCmdlineArgs(shortArgsMap map[string]string, useEnvCmdlineOpts bool) ([]string, error) {
	var err error

	consumedOpts = nil

	var args []string
	if useEnvCmdlineOpts {
		args = getArgsFromEnvironment()
	} else {
		args = os.Args[1:]
	}

	cmdlineConf = []byte("{}")
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg[0] != '-' {
			// we can't consume arguments not starting with "-".
			// this and subsequent arguments will be returned to the caller
			return args[i:], nil
		}
		if len(arg) > 6 && arg[0:6] == "-conf=" {
			consumedOpts = append(consumedOpts, arg)
			// NOTE: `useEnvCmdlineOpts` will be set to true when coderrect internally calls other executables (like reporter)
			// if the config file is specified using `-conf=xxx.json`
			// then we need to reload the customized config file, so that the internal executables have access to those configs
			if useEnvCmdlineOpts {
				confPath := strings.Split(arg, "=")[1]
				if len(confPath) > 0 {
					if customConf, err = ioutil.ReadFile(confPath); err != nil {
						panic(err)
					}
				}
			}
		}

		// the arg is single-char such as "-o". we need
		// to check if it's a short alias of a formal
		// argument such as "-report.outputDir" by looking
		// up "shortArgsMap"
		if len(arg) == 2 {
			tmp := arg[1:]
			longArgs, hasValue, ok := getLongArg(shortArgsMap, tmp)
			if ok {
				// yes, it's a short alias
				if !hasValue || i == len(args)-1 || args[i+1][0] == '-' {
					arg = "-" + longArgs + "=true"
				} else {
					arg = "-" + longArgs + "=" + args[i+1]
					i++
				}
			}
		}
		consumedOpts = append(consumedOpts, arg)

		var key, value string
		arg = arg[1:]
		var valueBytes []byte
		if strings.Contains(arg, "=") {
			tmp := strings.Split(arg, "=")
			key, value = tmp[0], tmp[1]
			if value != "true" && value != "false" {
				valueBytes, _ = json.Marshal(value)
			} else {
				valueBytes = []byte(value)
			}
		} else {
			key = arg
			valueBytes = []byte("true")
		}

		stages := strings.Split(key, ".")
		cmdlineConf, err = jsonparser.Set(cmdlineConf, valueBytes, stages...)
		if err != nil {
			return nil, err
		}
	}

	return nil, nil
}

/**
 * We assume all executables calling conflib reside under CODERRECT_HOME/bin
 */
func findCoderrectHome() (string, error) {
	homeDir := os.Getenv("CODERRECT_HOME")
	if homeDir != "" {
		return homeDir, nil
	}

	dir, err := os.Executable()
	if err != nil {
		return "", err
	}

	dir = filepath.Dir(dir)
	if !strings.HasSuffix(dir, string(filepath.Separator)+"bin") {
		//        return "", errors.New("The executable calling conflib must reside in coderrect package")
		return "", nil
	}

	return filepath.Dir(dir), nil
}

func internalInit(shortArgsMap map[string]string, useEnvCmdlineOpts bool) ([]string, error) {
	var confPath string
	var err error

	// load the default configuration file
	homeDir, err := findCoderrectHome()
	if err != nil {
		return nil, err
	}
	if homeDir != "" {
		confPath = filepath.Join(homeDir, "conf", "coderrect.json")
		if defaultConf, err = ioutil.ReadFile(confPath); err != nil {
			return nil, err
		}
	} else {
		defaultConf = []byte(`{}`)
	}

	// load the config file under user's home directory
	confPath = filepath.Join(os.Getenv("HOME"), ".coderrect.json")
	if homeConf, err = ioutil.ReadFile(confPath); err != nil {
		// no home config
		homeConf = []byte("{}")
	}

	// load the config file under current directory (project conf)
	if projectConf, err = ioutil.ReadFile("coderrect.json"); err != nil {
		projectConf = []byte("{}")
	}

	// load the custom config file
	if confPath, err = findCustomConfPath(); err != nil {
		return nil, err
	}
	if len(confPath) > 0 {
		if customConf, err = ioutil.ReadFile(confPath); err != nil {
			return nil, err
		}
	} else {
		customConf = []byte("{}")
	}

	return parseCmdlineArgs(shortArgsMap, useEnvCmdlineOpts)
}

/**
 * Initializes different levels of configuration file, and returns the
 * trailing arguments that can't be understood by conflib.
 *
 * coderrect -report.color=blue -o /tmp abc.c def.c
 *
 * where, "-report.color=blue -o /tmp" will be consumed by conflibe, and
 * {"abc.c", "def.c"} will be returned.
 */
func Initialize(shortArgsMap map[string]string) ([]string, error) {
	return internalInit(shortArgsMap, false)
}

/**
 * this is similar to Initialize(). But it doesn't consume the cmdline opts. Instead,
 * it looks for an environment variable CODERRECT_CMDLINE_OPTS to get parameters.
 */
func InitializeWithEnvCmdline(shortArgsMap map[string]string) ([]string, error) {
	return internalInit(shortArgsMap, true)
}

func GetInt(key string, defaultValue int64) int64 {
	stages := strings.Split(key, ".")
	if rv, err := jsonparser.GetString(cmdlineConf, stages...); err == nil {
		if rrv, err := strconv.Atoi(rv); err == nil {
			return int64(rrv)
		} else {
			return defaultValue
		}
	}
	if rv, err := jsonparser.GetString(customConf, stages...); err == nil {
		if rrv, err := strconv.Atoi(rv); err == nil {
			return int64(rrv)
		} else {
			return defaultValue
		}
	}
	if rv, err := jsonparser.GetString(projectConf, stages...); err == nil {
		if rrv, err := strconv.Atoi(rv); err == nil {
			return int64(rrv)
		} else {
			return defaultValue
		}
	}
	if rv, err := jsonparser.GetString(homeConf, stages...); err == nil {
		if rrv, err := strconv.Atoi(rv); err == nil {
			return int64(rrv)
		} else {
			return defaultValue
		}
	}
	if rv, err := jsonparser.GetString(defaultConf, stages...); err == nil {
		if rrv, err := strconv.Atoi(rv); err == nil {
			return int64(rrv)
		} else {
			return defaultValue
		}
	}
	if rv, err := jsonparser.GetInt(cmdlineConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetInt(customConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetInt(projectConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetInt(homeConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetInt(defaultConf, stages...); err == nil {
		return rv
	}

	return defaultValue
}

func GetDouble(key string, defaultValue float64) float64 {
	stages := strings.Split(key, ".")
	if rv, err := jsonparser.GetFloat(cmdlineConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetFloat(customConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetFloat(projectConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetFloat(homeConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetFloat(defaultConf, stages...); err == nil {
		return rv
	}

	return defaultValue
}

func GetString(key string, defaultValue string) string {
	stages := strings.Split(key, ".")
	if rv, err := jsonparser.GetString(cmdlineConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetString(customConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetString(projectConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetString(homeConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetString(defaultConf, stages...); err == nil {
		return rv
	}

	return defaultValue
}

func GetBool(key string, defaultValue bool) bool {
	stages := strings.Split(key, ".")
	if rv, err := jsonparser.GetBoolean(cmdlineConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetBoolean(customConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetBoolean(projectConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetBoolean(homeConf, stages...); err == nil {
		return rv
	}
	if rv, err := jsonparser.GetBoolean(defaultConf, stages...); err == nil {
		return rv
	}

	return defaultValue
}

func GetStrings(key string) []string {
	if rv, err := getStrings(key, cmdlineConf); err == nil {
		return rv
	}
	if rv, err := getStrings(key, customConf); err == nil {
		return rv
	}
	if rv, err := getStrings(key, projectConf); err == nil {
		return rv
	}
	if rv, err := getStrings(key, homeConf); err == nil {
		return rv
	}
	if rv, err := getStrings(key, defaultConf); err == nil {
		return rv
	}

	return []string{}
}

func getStrings(key string, conf []byte) ([]string, error) {
	rs := []string{}

	stages := strings.Split(key, ".")
	j, _, _, err := jsonparser.Get(conf, stages...)
	if err != nil {
		return rs, err
	}

	jsonparser.ArrayEach(j, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		rs = append(rs, string(value))
	})

	return rs, nil
}

/**
 * Apply all keys in anotherConf to oneConf and then return onfConf
 */
func combineConf(prefix []string, oneConf, anotherConf []byte) []byte {
	var handler func([]byte, []byte, jsonparser.ValueType, int) error
	handler = func(key []byte, value []byte, dataType jsonparser.ValueType, offset int) error {
		prefix2 := append(prefix, string(key))
		if dataType != jsonparser.Object {
			if dataType == jsonparser.String {
				a := "\"" + string(value) + "\""
				oneConf, _ = jsonparser.Set(oneConf, []byte(a), prefix2...)
			} else {
				oneConf, _ = jsonparser.Set(oneConf, value, prefix2...)
			}
		} else {
			oneConf = combineConf(prefix2, oneConf, value)
		}

		return nil
	}

	jsonparser.ObjectEach(anotherConf, handler)

	return oneConf
}

func Dump() string {
	var tmpConf = make([]byte, len(defaultConf))
	copy(tmpConf, defaultConf)

	var prefix []string

	tmpConf = combineConf(prefix, tmpConf, homeConf)
	prefix = nil
	tmpConf = combineConf(prefix, tmpConf, projectConf)
	tmpConf = combineConf(prefix, tmpConf, customConf)
	tmpConf = combineConf(prefix, tmpConf, cmdlineConf)

	var out bytes.Buffer
	json.Indent(&out, tmpConf, "", "    ")
	return string(out.Bytes())
}

func GetCoderrectHome() (string, error) {
	dir := os.Getenv("CODERRECT_HOME")
	if dir != "" {
		return dir, nil
	}

	dir, err := os.Executable()
	if err != nil {
		return "", err
	}

	dir = filepath.Dir(filepath.Dir(dir))
	return dir, nil
}

/**
 * return the command-line options consumed by initialize()
 */
func GetCmdlineOpts() string {
	jsonstr := `
{
    "Opts": [%s]
}
`
	var tmp string
	for _, arg := range consumedOpts {
		if len(tmp) > 0 {
			tmp += ", "
		}
		tmp += fmt.Sprintf("\"%s\"", arg)
	}

	return fmt.Sprintf(jsonstr, tmp)
}
