#!/bin/bash

cd build/coderrect-jenkins-plugin
mvn -DskipTests package > build.log 2>&1 
if [ $? -ne 0 ]; then
    exit 1
fi

cp target/coderrect.hpi /build/package/bin/
