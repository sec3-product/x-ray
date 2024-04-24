# Github scanner build

from https://coderrect.atlassian.net/wiki/spaces/COD/pages/1066237955/Github+scanner+build

This is a java project, you must install dependencies

JDK 11

## How to build 

```
cd /the/path/to/coderrect
cd github-scanner
uncomment the third line
./build-dev.sh
```
## how to run

At here you have already created docker image tagged ‘github-app:dev'
```
./dev.sh /path/to/www
```
this will run the docker, and mapping full local path as a directory to store uploaded report.
