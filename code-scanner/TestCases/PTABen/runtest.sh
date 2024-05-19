#/bin/bash
CLANG='clang'
CLANGXX='clang++'

CLANGFLAG='-O0 -g -S -emit-llvm -I.'

TESTWITHOPT=$1
COMPILELOG="compile.log"

### Add the fold of c files to be tested
### TODO: add basic_cpp_tests as well
TestFolders="basic_c_tests" ##basic_cpp_tests"

### remove previous compile log
rm -rf $COMPILELOG

### start testing
for folder in $TestFolders 
  do
    echo Entering Folder $folder for testing ...
    echo "#################COMPILATION LOG##############" > $COMPILELOG
    ### test plain c program files
    for i in `find $folder -name '*.c'` 
    do
	FileName=$(dirname $i)/`basename $i .c`
	$CLANG -I$PTATEST $CLANGFLAG $i -o $FileName.ll >>$COMPILELOG 2>&1
	echo -e "\n######\e[32manalyzing $FileName.c\e[0m#######"
	wpa $FileName.ll
	if [ $? -ne 0 ]
	then
	    echo -e "\e[31mcrash on $FileName.c\e[0m"
	fi
    done

    ### test plain cpp program files
    for i in `find $folder -name '*.cpp'` 
    do
	FileName=$(dirname $i)/`basename $i .cpp`
	$CLANGXX -I$PTATEST $CLANGFLAG $i -o $FileName.ll >>$COMPILELOG 2>&1
	echo -e "\n######\e[32manalyzing $FileName.cpp\e[0m#######"
	wpa $FileName.ll
	if [ $? -ne 0 ]
	then
	    echo -e "\e[31mcrash on $FileName.cpp\e[0m"
	fi
    done

  done
echo analysis finished
