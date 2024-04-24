MY_PATH="`dirname \"$0\"`"
if [ -z "$CODERRECT_HOME" ]
then
    CLANG=clang-9
else
    CLANG=$CODERRECT_HOME/clang/bin/clang
fi
echo "using clang: $CLANG"

cd $MY_PATH
FILES=*.stub.cpp
mkdir -p ./bc

for F in $FILES
do 
    echo "compiling stub: $F"
    cd bc
    ${CLANG} -c -O1 -mllvm -disable-llvm-optzns -emit-llvm -fno-cxx-exceptions ../$F
    cd ..
done