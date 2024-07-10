# Help copy all binareis under gosrc/bin to the PATH where coderrect is located
# This script must be used under gosrc

BIN_PATH="/usr/local/bin/sec3"   # x-ray tools path, can be changed as needed

if ! which coderrect > /dev/null; then
    echo "coderrect binary not found, creating directories and file."

    export PATH=$BIN_PATH:$PATH
    echo 'export PATH=$BIN_PATH:$PATH' >> ~/.bashrc

    mkdir -p $BIN_PATH
    touch $BIN_PATH/coderrect
    chmod 755 $BIN_PATH/coderrect
fi

CR=$(dirname $(which coderrect))

GOSRC=$(pwd | grep -Eo '.+\/gosrc')
GOSRC_BIN=$GOSRC/bin

echo "Copying all binaries under $GOSRC_BIN to $CR"
cp $GOSRC_BIN/* $CR
cp $GOSRC/../package/conf/coderrect.json $CR/../conf/
cp $GOSRC/../package/data/reporter/*.html $CR/../data/reporter/
cp $GOSRC/../package/data/reporter/artifacts/coderrect* $CR/../data/reporter/artifacts/
cp $GOSRC/../package/data/reporter/artifacts/images/* $CR/../data/reporter/artifacts/images/

if ! which racedetect > /dev/null; then
ln -s $GOSRC/../../code-detector/build/bin/racedetect $CR/racedetect
fi

if ! which sol-racedetect > /dev/null; then
ln -s $GOSRC/../../code-parser/build/bin/sol-racedetect $CR/sol-racedetect
fi