# clio

To use run `./test.sh`

To add more files to be tested add .ll or .bc files to the `bcs` directory.

To add more True Positives to be tested during regression testing, add an entry to ./bcs/truth.txt with a label matching the bc name.
For Example, if the regression check should ensure that the races.json produced for "example.bc" contains a race between main.c:24 and main.c29, the following should be appended to the truth.txt file:

```
[example.bc]
main.c 24 main.c 29
```

## Internals

First `test.sh` script runs racedetect on all .ll and .bc files in the bc directory and saves the reports to the `reports` directory.

Next, the script looks for a directory called `reports-old` and shows a diff for each projects that has a races.json in both `reports` and `reports-old`.

Then, for each .ll or .bc file the script does regression testing using the races in `./bcs/truth.txt` as the oracle.

Lastly, the script prints some summary information and renames the `report` directory to `reports-old` to be compared against for the next run.
