#!/bin/bash

make all > /dev/null

BLUE='\033[1;36m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

BIN=$(which racedetect)
echo -ne "${PURPLE}Testing with ${BIN}${NC}\n"

TESTS=""
if [ -z "$1" ]; then
    # Gather all the files
    BCS=$(ls ./bcs/*.bc 2>/dev/null)
    LLS=$(ls ./bcs/*.ll 2>/dev/null)
    TESTS="$BCS $LLS"
    echo -ne "${PURPLE}Testing bc/ll in bcs dir ${BIN}${NC}\n"
else
    TESTS=$1
    echo -ne "${PURPLE}Testing ${TESTS} ${BIN}${NC}\n"
fi



# Run racedetect on all the bcs
echo -ne "${PURPLE}Running racedetect on all bc/ll files${NC}\n"
#./binrunner $BIN -- $BCS $LLS # whole redis broke binrunner
for file in $TESTS
do
    name=$(basename ${file})
    
    DIR=./reports/${name}${BIN}/
    mkdir -p $DIR
    
    if [ "${name}" == "stubbed_module.bc" ]; then
        continue
    fi
    
    echo -ne "${BLUE}------${name}------${NC}\n"
    $BIN --no-limit --no-ov --no-av $file  2>$DIR/error.txt
    if [ $? -eq 0 ]; then
        rm -rf $DIR/error.txt 2>/dev/null
        mv races.json $DIR
    fi
    
done

# Call rdiff
NCRASH=0
if [ -d ./reports-old ]; then
    for file in $TESTS
    do
        name=$(basename ${file})
        if [ "${name}" == "stubbed_module.bc" ]; then
            continue
        fi

        echo -ne "${BLUE}------${name}------${NC}\n"

        if [ -f ./reports/${name}${BIN}/error.txt ]; then 
            echo -ne "${RED}CRASH${NC}\n"
            NCRASH=$((NCRASH+1))
            continue
        fi

        if [ ! -f ./reports-old/${name}${BIN}/races.json ]; then
            echo -ne "${YELLOW}No previous report to compare with${NC}\n"
            continue
        fi

        ./rdiff ./reports-old/${name}${BIN}/races.json ./reports/${name}${BIN}/races.json 

    done
else
    echo -ne "${YELLOW}No old reports to compare against${NC}\n"
fi

# run check for TP
FAILS=0
echo -ne "${PURPLE}Checking regressions${NC}\n"
for file in $TESTS
do
    name=$(basename ${file})
    if grep ${name} ./bcs/truth.txt > /dev/null 2>&1; then
        echo -ne "${BLUE}${name} ${NC} "
        if [ -f ./reports/${name}${BIN}/error.txt ]; then 
            echo -ne "${RED}FAIL (CRASH)${NC}\n"
            continue
        fi

        ./check ./reports/${name}${BIN}/races.json ./bcs/truth.txt ${name}
        if [ $? -ne 0 ]; then
            FAILS=$((FAILS+1))
        fi
    fi
done


# Print summary
echo -ne "${PURPLE}Summary${NC}\n"
if [ "$NCRASH" -gt 0 ]; then 
    echo -ne "${RED}${NCRASH} CRASHES${NC}\n"
else
    echo -ne "${GREEN}${NCRASH} CRASHES${NC}\n"
fi
if [ "$FAILS" -gt 0 ]; then 
    echo -ne "${RED}${FAILS} Regressions${NC}\n"
else
    echo -ne "${GREEN}${FAILS} Regressions${NC}\n"
fi

# Move current results to old for next check
rm -rf reports-old
mv reports reports-old
