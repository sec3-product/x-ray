package conflib

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetArgsFromEnvironment(t *testing.T) {
	// case 1 - no such env variable
	args := getArgsFromEnvironment()
	if len(args) != 0 {
		t.Errorf("No-env-variable. args=%v", args)
	}

	// case 2 - a norla env variable
	jsonstr := `
{
  "opts": ["k1", "k2", "k3"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	args = getArgsFromEnvironment()
	if len(args) != 3 {
		t.Errorf("normal-env_variable, args=%v", args)
	} else {
		fmt.Printf("result=%v\n", args)
	}
}

func TestGetLongArgs(t *testing.T) {
	alias := map[string]string{
		"o":  "report.outputDir",
		"b:": "bcOnly",
	}

	longarg, hasValue, ok := getLongArg(alias, "b")
	if hasValue {
		t.Errorf("hasValue=%v", hasValue)
	}
	if !ok {
		t.Errorf("ok=%v", ok)
	}
	if longarg != "bcOnly" {
		t.Errorf("longarg=%v", longarg)
	}
}

func TestGetStrings(t *testing.T) {
	jsonstr := `
{
  "opts": ["k1", "k2", "k3"]
}
`
	if rs, err := getStrings("opts", []byte(jsonstr)); err == nil {
		if len(rs) != 3 {
			t.Errorf("%v", rs)
		}
	} else {
		t.Errorf("%v", err)
	}
}

func TestParseCmdlineArgs(t *testing.T) {
	type CmdlineOpts struct {
		Opts []string
	}
	cmdlinOpts := CmdlineOpts{}

	alias := map[string]string{
		"o": "report.outputDir",
		"e": "report.expand",
	}

	// case 1 - no command-line opts
	remainingArgs, err := parseCmdlineArgs(alias, true)
	if len(remainingArgs) != 0 {
		t.Errorf("remainingArgs should be empty. remainingArgs=%v", remainingArgs)
	}
	if err != nil {
		t.Errorf("failed. err=%v", err)
	}
	json.Unmarshal([]byte(GetCmdlineOpts()), &cmdlinOpts)
	if len(cmdlinOpts.Opts) != 0 {
		t.Errorf("shouldn't have cmdline opts. cmdlinOpts=%v", cmdlinOpts)
	}

	// case 2 - all cmdline opts will be consumed
	jsonstr := `
{
  "opts": ["-e", "123", "-o", "debug"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	remainingArgs, err = parseCmdlineArgs(alias, true)
	if err != nil {
		t.Errorf("failed. err=%v", err)
	}
	expand := GetInt("report.expand", 0)
	if expand != 123 {
		t.Errorf("Expand should be equal to 123. actual=%d", expand)
	}
	debug := GetInt("report.outputDir", 0)
	if debug != 0 {
		t.Errorf("Debug should be equal to 0. actual=%d", debug)
	}

	// case 2 - all cmdline opts will be consumed
	jsonstr = `
{
  "opts": ["-logger.level=info", "-logger.debugToStderr=true", "-logger.maximumLogFileCount=9"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	remainingArgs, err = parseCmdlineArgs(alias, true)
	if len(remainingArgs) != 0 {
		t.Errorf("remainingArgs should be empty. remainingArgs=%v", remainingArgs)
	}
	if err != nil {
		t.Errorf("failed. err=%v", err)
	}
	json.Unmarshal([]byte(GetCmdlineOpts()), &cmdlinOpts)
	if len(cmdlinOpts.Opts) != 3 {
		t.Errorf("should have cmdline opts. cmdlinOpts=%v", cmdlinOpts)
	}

	// case 3 - short alias
	jsonstr = `
{
  "opts": ["-logger.level=info", "-logger.debugToStderr=true", "-e"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	remainingArgs, err = parseCmdlineArgs(alias, true)
	if len(remainingArgs) != 0 {
		t.Errorf("remainingArgs should be empty. remainingArgs=%v", remainingArgs)
	}
	if err != nil {
		t.Errorf("failed. err=%v", err)
	}
	json.Unmarshal([]byte(GetCmdlineOpts()), &cmdlinOpts)
	if len(cmdlinOpts.Opts) != 3 {
		t.Errorf("should have cmdline opts. cmdlinOpts=%v", cmdlinOpts)
	}

	// case 4 - remaining args
	jsonstr = `
{
  "opts": ["-logger.level=info", "-logger.debugToStderr=true", "rkey1", "rkey2"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	remainingArgs, err = parseCmdlineArgs(alias, true)
	if len(remainingArgs) != 2 {
		t.Errorf("remainingArgs shouldn't be empty. remainingArgs=%v", remainingArgs)
	} else {
		fmt.Printf("remainingArgs=%v\n", remainingArgs)
	}
	if err != nil {
		t.Errorf("failed. err=%v", err)
	}
	json.Unmarshal([]byte(GetCmdlineOpts()), &cmdlinOpts)
	if len(cmdlinOpts.Opts) != 2 {
		t.Errorf("should have cmdline opts. cmdlinOpts=%v", cmdlinOpts)
	}
	fmt.Printf("cmdlinOpts=%v\n", cmdlinOpts)
	fmt.Printf("jsonstr=%v\n", GetCmdlineOpts())
}

func TestParseCmdlineArgsWhenContainPath(t *testing.T) {
	type CmdlineOpts struct {
		Opts []string
	}
	alias := map[string]string{}

	jsonstr := `
{
  "opts": ["-logger.logFolder=d:\\path"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	parseCmdlineArgs(alias, true)
	v := GetString("logger.logFolder", "")
	fmt.Printf("%s\n", v)
	assert.Equal(t, "d:\\path", v)
}

func TestParseCmdlineArgsWhenContainBool(t *testing.T) {
	type CmdlineOpts struct {
		Opts []string
	}
	alias := map[string]string{}

	jsonstr := `
{
  "opts": ["-logger.debugToStderr=true", "-e"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	parseCmdlineArgs(alias, true)
	v := GetBool("logger.debugToStderr", false)
	fmt.Printf("%v\n", v)
	assert.True(t, v)
}

func TestParseCmdlineArgsWhenContainDigital(t *testing.T) {
	type CmdlineOpts struct {
		Opts []string
	}
	alias := map[string]string{}

	jsonstr := `
{
  "opts": ["-logger.maximumLogFileCount=9"]
}
`
	os.Setenv("CODERRECT_CMDLINE_OPTS", jsonstr)
	parseCmdlineArgs(alias, true)
	v := GetInt("logger.maximumLogFileCount", 0)
	fmt.Printf("%v\n", v)
	assert.EqualValues(t, 9, v)
}
