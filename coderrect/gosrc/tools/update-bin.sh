#!/bin/bash

# Copy all the binaries under `gosrc/bin` to the PATH where `coderrect` is
# located. This script must be used under `gosrc/`.

INSTALL_PATH="/usr/local/sec3"

if ! which coderrect > /dev/null; then
    echo "\`coderrect\` binary not found, creating directories and file."

    export PATH=$INSTALL_PATH/bin:$PATH
    echo "export PATH=$INSTALL_PATH/bin:\$PATH" >> ~/.bashrc

    for dir in bin conf data/reporter/artifacts/images; do
        mkdir -p "$INSTALL_PATH/$dir"
    done
fi

GOSRC=$(pwd | grep -Eo '.+\/gosrc')

echo "Copying all binaries under $GOSRC/bin to $INSTALL_PATH/bin"
cp $GOSRC/bin/* $INSTALL_PATH/bin
cp $GOSRC/../package/conf/coderrect.json $INSTALL_PATH/conf/
cp $GOSRC/../package/data/reporter/*.html $INSTALL_PATH/data/reporter/
cp $GOSRC/../package/data/reporter/artifacts/coderrect* $INSTALL_PATH/data/reporter/artifacts/
cp $GOSRC/../package/data/reporter/artifacts/images/* $INSTALL_PATH/data/reporter/artifacts/images/

if ! which racedetect > /dev/null; then
    cp $GOSRC/../../code-detector/build/bin/racedetect $INSTALL_PATH/bin/racedetect
fi

if ! which sol-racedetect > /dev/null; then
    cp $GOSRC/../../code-parser/build/bin/sol-racedetect $INSTALL_PATH/bin/sol-racedetect
fi
