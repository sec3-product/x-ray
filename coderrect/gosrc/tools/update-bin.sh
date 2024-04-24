# Help copy all binareis under gosrc/bin to the PATH where coderrect is located
# This script must be used under gosrc
CR=$(dirname $(which coderrect))

GOSRC=$(pwd | grep -Eo '.+\/gosrc')
GOSRC_BIN=$GOSRC/bin

echo "Copying all binaries under $GOSRC_BIN to $CR"
cp $GOSRC_BIN/* $CR