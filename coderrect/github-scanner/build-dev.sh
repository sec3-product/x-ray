#!/bin/sh

#./mvnw package -DskipTests=true
docker build -t coderrect/github-app:dev .
