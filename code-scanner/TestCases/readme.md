# Test Runner

To use, make sure `aser-race` is in your PATH and run `./test.py` and you should eventually see results that looks something like:

```
DRB001-antidep1-orig-yes       PASS    2 races    0.05s
...
DRB116-target-teams-orig-yes   FAIL    0 races    0.07s
PASS/FAIL/CRASH
Total:   62 / 41 / 9
yes:     30 / 26 / 1
no:      32 / 15 / 8
```

Currently the test runner only runs on dataracebench.

## Results

Each output line is in the format 

```
File  Result  N  Time
```

`File` - file being tested

`Result` - 
  - `PASS` - The tool reported races for the correct lines
  - `FAIL` - The tool either missed races or reported false races
  - `CRASH` - The tool terminated with a non-zero exit code

`N` - Number of races reported

`Time` - number of seconds it took the tool to run

Currently to check correctness we only check that the line numbers of reported races match.

## Test Single File

The tool can also be used to test a single file using the `-s` or `--single` flags.

```
./test.py --single dataracebench/micro-benchmarks/DRB001-antidep1-orig-yes.c
```

If the file ends in `.ll` or `.bc` the test runner will run the race detection tool directly. 

If the file ends in `.c` or `.cpp` the test runner will attempt to compile to bitcode and then run the race detection tool.

The single file must be entirely self contained.

## Extending

The expected results are stored in a json file. You can see `drbexpect.json` for an example.

The top level of the json file contains the name of each test:

```
{
    "DRB001-antidep1-orig-yes": {
        ...
    },
    "DRB002-antidep1-var-yes": {
        ...
    },
    ...
}
```

Each test object contains a list of races. Each race contains a list of locations (line, col) that race with eachother.

For example
```
"TEST001": {
        "races": [
            [ [64,10], [64,5] ],
            [ [42, 10], [42, 12], [43, 9] ],
        ]
    },
```

The above example shows the following races:
- 64:10 races with 64:5
- 42:10 races with 42:12
- 42:10 races with 43:9
- 42:12 races with 43:9

All unique pairs within each race must be considered.




