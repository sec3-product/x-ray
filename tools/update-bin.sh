#!/bin/bash

# Copy all the binaries under `bin` to the PATH where `coderrect` is located.
# This script must be used under the root of the repo.

INSTALL_PATH="/usr/local/sec3"

if ! which coderrect > /dev/null; then
    echo "\`coderrect\` binary not found, creating directories and file."

    export PATH=$INSTALL_PATH/bin:$PATH
    echo "export PATH=$INSTALL_PATH/bin:\$PATH" >> ~/.bashrc

    for dir in bin conf data/reporter/artifacts/images; do
        mkdir -p "$INSTALL_PATH/$dir"
    done
fi

echo "Copying all binaries under bin to $INSTALL_PATH/bin"
cp bin/* $INSTALL_PATH/bin
cp package/conf/coderrect.json $INSTALL_PATH/conf/
cp package/data/reporter/*.html $INSTALL_PATH/data/reporter/
cp package/data/reporter/artifacts/coderrect* $INSTALL_PATH/data/reporter/artifacts/
cp package/data/reporter/artifacts/images/* $INSTALL_PATH/data/reporter/artifacts/images/

if ! which racedetect > /dev/null; then
    cp code-detector/build/bin/racedetect $INSTALL_PATH/bin/racedetect
fi

if ! which sol-racedetect > /dev/null; then
    cp code-parser/build/bin/sol-racedetect $INSTALL_PATH/bin/sol-racedetect
fi
